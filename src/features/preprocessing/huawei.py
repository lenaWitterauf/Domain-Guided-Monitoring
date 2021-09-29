from src.features.preprocessing.huawei_traces import HuaweiTracePreprocessor
import dataclass_cli
import dataclasses
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Set
import http
import re
from .base import Preprocessor
from collections import Counter
from .drain import Drain, DrainParameters
import numpy as np


@dataclass_cli.add
@dataclasses.dataclass
class HuaweiPreprocessorConfig:
    aggregated_log_file: Path = Path("data/logs_aggregated_concurrent.csv")
    traces_root_directory: Path = Path("data/concurrent_data/traces/")
    final_log_file: Path = Path("data/huawei.pkl")
    relevant_aggregated_log_columns: List[str] = dataclasses.field(
        default_factory=lambda: [
            "Hostname",
            "log_level",
            "programname",
            "python_module",
            "http_status",
            "http_method",
        ],
    )
    relevant_trace_columns: List[str] = dataclasses.field(
        default_factory=lambda: [
            "Hostname",
            "trace_name",
            "trace_service",
            "python_module",
            "trace_project",
            "payload",
            "etype",
            "http_method",
            "function",
        ],
    )
    use_trace_data: bool = False
    aggregate_per_trace: bool = False
    aggregate_per_max_number: int = -1
    aggregate_per_time_frequency: str = ""
    log_datetime_column_name: str = "@timestamp"
    log_payload_column_name: str = "Payload"
    use_log_hierarchy: bool = True
    fine_drain_log_depth: int = 10
    fine_drain_log_st: float = 0.75
    coarse_drain_log_depth: int = 4
    coarse_drain_log_st: float = 0.2
    drain_log_depths: List[int] = dataclasses.field(default_factory=lambda: [],)
    drain_log_sts: List[float] = dataclasses.field(default_factory=lambda: [],)
    url_column_name: str = "http_url"
    drain_url_depth: int = 10
    drain_url_st: float = 0.5
    add_log_clusters: bool = True
    min_logs_per_trace: int = 2
    min_causality: float = 0.0
    log_only_causality: bool = False
    relevant_log_column: str = "fine_log_cluster_template"


class ConcurrentAggregatedLogsPreprocessor(Preprocessor):
    sequence_column_name: str = "all_events"
    request_drain_regex: str = "[^a-zA-Z0-9\-\.]"

    def __init__(self, config: HuaweiPreprocessorConfig):
        self.config = config
        self.relevant_columns = set(
            [x for x in self.config.relevant_aggregated_log_columns]
        )
        self.relevant_columns.add("fine_log_cluster_template")
        self.relevant_columns.add("coarse_log_cluster_template")
        self.relevant_columns.add("url_cluster_template")
        for i in range(len(self.config.drain_log_depths)):
            self.relevant_columns.add(str(i) + "_log_cluster_template")
        if self.config.use_trace_data:
            self.relevant_columns.update(self.config.relevant_trace_columns)

    def load_data(self) -> pd.DataFrame:
        if self.config.aggregate_per_trace:
            return self._load_data_per_trace()
        elif self.config.aggregate_per_max_number > 0:
            log_only_data = (
                self._load_log_only_data()
                .sort_values(by="timestamp")
                .reset_index(drop=True)
                .reset_index(drop=False)
            )
            log_only_data["grouper"] = log_only_data["index"].apply(
                lambda x: int(x / self.config.aggregate_per_max_number)
            )
            return self._aggregate_per(log_only_data, aggregation_column="grouper")
        elif len(self.config.aggregate_per_time_frequency) > 0:
            log_only_data = (
                self._load_log_only_data()
                .sort_values(by="timestamp")
                .reset_index(drop=True)
                .reset_index(drop=False)
            )
            aggregated_log_data = self._aggregate_per(
                log_only_data,
                aggregation_column=pd.Grouper(
                    key="timestamp", freq=self.config.aggregate_per_time_frequency
                ),
            )
            return aggregated_log_data[aggregated_log_data["num_events"] > 1]
        else:
            log_only_data = self._load_log_only_data()
            log_only_data["grouper"] = 1
            return self._aggregate_per(log_only_data, aggregation_column="grouper")

    def _load_data_per_trace(self) -> pd.DataFrame:
        full_df = self.load_full_data()
        aggregated_df = self._aggregate_per(full_df)
        aggregated_df = aggregated_df[
            aggregated_df["num_events"] >= self.config.min_logs_per_trace
        ]
        logging.info(
            "Summary of num_events:\n %s",
            aggregated_df["num_events"].describe().to_string(),
        )
        return aggregated_df

    def _load_log_only_data(self) -> pd.DataFrame:
        log_df = self._read_log_df()
        log_df = self._add_url_drain_clusters(log_df)
        for prefix in ["fine_", "coarse_"] + [
            str(i) + "_" for i in range(len(self.config.drain_log_depths))
        ]:
            log_df[prefix + "log_cluster_template"] = (
                log_df[prefix + "log_cluster_template"]
                .fillna("")
                .astype(str)
                .replace(np.nan, "", regex=True)
                .apply(lambda x: x if len(x) > 0 else "___empty___")
            )
        return log_df

    def load_full_data(self) -> pd.DataFrame:
        logging.info(
            "Trying to read full huawei_df from %s", self.config.final_log_file
        )
        if not self.config.final_log_file.is_file():
            full_df = self._load_full_data()
            full_df.to_pickle(self.config.final_log_file)

        return pd.read_pickle(self.config.final_log_file)

    def _load_full_data(self) -> pd.DataFrame:
        log_df = self._read_log_df()
        trace_df = self._read_trace_df()
        merged_df = self._merge_logs_traces(log_df, trace_df)
        return self._add_url_drain_clusters(merged_df)

    def _aggregate_per(
        self, merged_df: pd.DataFrame, aggregation_column: str = "parent_trace_id"
    ) -> pd.DataFrame:
        logging.debug("Aggregating huawei data per %s", aggregation_column)
        for column in self.relevant_columns:
            merged_df[column] = merged_df[column].apply(
                lambda x: column + "#" + x.lower() if len(x) > 0 else ""
            )

        merged_df["all_events"] = merged_df[self.relevant_columns].values.tolist()
        merged_df["attributes"] = merged_df[
            [x for x in self.relevant_columns if not "log_cluster_template" in x]
        ].values.tolist()
        for log_template_column in [
            x for x in self.relevant_columns if "log_cluster_template" in x
        ]:
            merged_df[log_template_column] = merged_df[log_template_column].apply(
                lambda x: [x]
            )
        events_per_trace = (
            merged_df.sort_values(by="timestamp")
            .groupby(aggregation_column)
            .agg(
                {
                    column_name: lambda x: list(x)
                    for column_name in ["all_events", "attributes",]
                    + [x for x in self.relevant_columns if "log_cluster_template" in x]
                }
            )
            .reset_index()
        )
        events_per_trace["num_logs"] = events_per_trace[
            self.config.relevant_log_column
        ].apply(lambda x: len([loglist for loglist in x if len(loglist[0]) > 0]))
        events_per_trace["num_events"] = events_per_trace[
            self.config.relevant_log_column
        ].apply(lambda x: len(x))
        return events_per_trace[
            ["num_logs", "num_events", "all_events", "attributes",]
            + [x for x in self.relevant_columns if "log_cluster_template" in x]
        ]

    def _merge_logs_traces(self, log_df: pd.DataFrame, trace_df: pd.DataFrame):
        log_df_with_trace_id = self._match_logs_to_traces(log_df, trace_df)
        if self.config.use_trace_data:
            return pd.concat(
                [log_df_with_trace_id, trace_df], ignore_index=True
            ).reset_index(drop=True)
        else:
            return log_df_with_trace_id.reset_index(drop=True)

    def _match_logs_to_traces(self, log_df: pd.DataFrame, trace_df: pd.DataFrame):
        max_timestamp_by_trace = trace_df.groupby(by="parent_trace_id").agg(
            {"timestamp": max,}
        )
        min_timestamp_by_trace = trace_df.groupby(by="parent_trace_id").agg(
            {"timestamp": min,}
        )
        timestamps_merged = pd.merge(
            max_timestamp_by_trace,
            min_timestamp_by_trace,
            left_index=True,
            right_index=True,
            suffixes=("_max", "_min"),
        )
        merged_dfs = []
        for idx, row in tqdm(
            timestamps_merged.iterrows(),
            total=len(timestamps_merged),
            desc="Matching logs to traces...",
        ):
            rel_df = log_df.loc[
                (log_df["timestamp"] >= row["timestamp_min"])
                & (log_df["timestamp"] <= row["timestamp_max"])
            ].copy()
            rel_df["parent_trace_id"] = idx
            merged_dfs.append(rel_df)
        return pd.concat(merged_dfs).drop_duplicates().reset_index(drop=True)

    def _read_trace_df(self) -> pd.DataFrame:
        preprocessor = HuaweiTracePreprocessor(
            trace_base_directory=self.config.traces_root_directory
        )
        trace_df = preprocessor.load_data()
        return trace_df

    def _read_log_df(self) -> pd.DataFrame:
        df = (
            pd.read_csv(self.config.aggregated_log_file)
            .fillna("")
            .astype(str)
            .replace(np.nan, "", regex=True)
        )
        rel_df = df[
            self.config.relevant_aggregated_log_columns
            + [self.config.log_datetime_column_name]
            + [self.config.log_payload_column_name]
            + [self.config.url_column_name]
        ]
        rel_df = self._add_log_drain_clusters(rel_df)
        rel_df["timestamp"] = pd.to_datetime(
            rel_df[self.config.log_datetime_column_name]
        )
        return rel_df

    def _add_url_drain_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        url_df = pd.DataFrame(
            df[self.config.url_column_name].dropna().drop_duplicates()
        )
        drain = Drain(
            DrainParameters(
                depth=self.config.drain_url_depth,
                st=self.config.drain_url_st,
                rex=[(self.request_drain_regex, " "),],
            ),
            data_df=url_df,
            data_df_column_name=self.config.url_column_name,
        )
        drain_result_df = (
            drain.load_data().drop_duplicates(ignore_index=False).set_index("log_idx")
        )
        url_result_df = (
            pd.merge(
                df,
                pd.merge(
                    url_df,
                    drain_result_df,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
                .drop_duplicates()
                .reset_index(drop=True),
                on=self.config.url_column_name,
                how="left",
            )
            .rename(
                columns={
                    "cluster_template": "url_cluster_template",
                    "cluster_path": "url_cluster_path",
                }
            )
            .drop(columns=["cluster_id"])
        )
        url_result_df["url_cluster_template"] = (
            url_result_df["url_cluster_template"]
            .fillna("")
            .astype(str)
            .replace(np.nan, "", regex=True)
        )
        return url_result_df

    def _add_log_drain_clusters_prefix(
        self, log_df: pd.DataFrame, depth: int, st: float, prefix: str
    ) -> pd.DataFrame:
        all_logs_df = pd.DataFrame(
            log_df[self.config.log_payload_column_name].dropna().drop_duplicates()
        )
        drain = Drain(
            DrainParameters(
                depth=depth,
                st=st,
                rex=[
                    ("(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)", ""),
                    (self.request_drain_regex, " "),
                    ("[^a-zA-Z\d\s:]", ""),
                ],
            ),
            data_df=all_logs_df,
            data_df_column_name=self.config.log_payload_column_name,
        )
        drain_result_df = drain.load_data().drop_duplicates().set_index("log_idx")
        log_result_df = (
            pd.merge(
                log_df,
                pd.merge(
                    all_logs_df,
                    drain_result_df,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
                .drop_duplicates()
                .reset_index(drop=True),
                on=self.config.log_payload_column_name,
                how="left",
            )
            .rename(
                columns={
                    "cluster_template": prefix + "log_cluster_template",
                    "cluster_path": prefix + "log_cluster_path",
                }
            )
            .drop(columns=["cluster_id"])
        )
        log_result_df[prefix + "log_cluster_template"] = (
            log_result_df[prefix + "log_cluster_template"]
            .fillna("")
            .astype(str)
            .replace(np.nan, "", regex=True)
        )
        return log_result_df

    def _add_log_drain_clusters(self, log_df: pd.DataFrame) -> pd.DataFrame:
        log_result_df = self._add_log_drain_clusters_prefix(
            log_df=log_df,
            depth=self.config.fine_drain_log_depth,
            st=self.config.fine_drain_log_st,
            prefix="fine_",
        )
        log_result_df = self._add_log_drain_clusters_prefix(
            log_df=log_result_df,
            depth=self.config.coarse_drain_log_depth,
            st=self.config.coarse_drain_log_st,
            prefix="coarse_",
        )
        for i in range(len(self.config.drain_log_depths)):
            log_result_df = self._add_log_drain_clusters_prefix(
                log_df=log_result_df,
                depth=self.config.drain_log_depths[i],
                st=self.config.drain_log_sts[i],
                prefix=str(i) + "_",
            )
        return log_result_df


class ConcurrentAggregatedLogsDescriptionPreprocessor(Preprocessor):
    def __init__(
        self, config: HuaweiPreprocessorConfig,
    ):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        preprocessor = ConcurrentAggregatedLogsPreprocessor(self.config)
        huawei_df = preprocessor._load_log_only_data()
        return self._load_column_descriptions(huawei_df, preprocessor.relevant_columns)

    def _load_column_descriptions(
        self, huawei_df: pd.DataFrame, relevant_columns: Set[str]
    ) -> pd.DataFrame:
        http_descriptions = self._load_http_descriptions()
        column_descriptions = self._get_column_descriptions()
        description_records = []
        for column in relevant_columns:
            values = set(
                huawei_df[column].dropna().astype(str).replace(np.nan, "", regex=True)
            )
            values = set([str(x).lower() for x in values if len(str(x)) > 0])
            for value in tqdm(values, desc="Loading descriptions for column " + column):
                description = ""
                if column == "Hostname":
                    name = value.rstrip("0123456789")
                    number = value[len(name) :]
                    description = name + " " + number
                elif column == "http_status":
                    description = http_descriptions[value]
                else:
                    description = " ".join(re.split("[,._\-\*]+", value))

                if column in column_descriptions:
                    description = column_descriptions[column] + " " + description

                description_records.append(
                    {"label": column + "#" + value, "description": description,},
                )

        return (
            pd.DataFrame.from_records(description_records)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _get_column_descriptions(self) -> Dict[str, str]:
        return {
            "Hostname": "Host name",
            "log_level": "Log level",
            "programname": "Program name",
            "python_module": "Python module",
            "http_status": "HTTP status",
            "http_method": "HTTP method",
            "trace_name": "Trace name",
            "trace_service": "Trace service",
            "trace_project": "Trace project",
            "etype": "Error type",
            "function": "Function",
            "url_cluster_template": "Url Cluster",
        }

    def _load_http_descriptions(self) -> Dict[str, str]:
        logging.debug("Initializing HTTP Status descriptions")
        http_descriptions = {}
        for status in list(http.HTTPStatus):
            http_descriptions[str(status.value) + ".0"] = status.name.lower().replace(
                "_", " "
            )

        return http_descriptions


class ConcurrentAggregatedLogsHierarchyPreprocessor(Preprocessor):
    def __init__(
        self, config: HuaweiPreprocessorConfig,
    ):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        if self.config.use_log_hierarchy:
            return self._load_log_only_hierarchy()
        else:
            return self._load_attribute_only_hierarchy()

    def _load_log_only_hierarchy(self) -> pd.DataFrame:
        preprocessor = ConcurrentAggregatedLogsPreprocessor(self.config)
        huawei_df = preprocessor._load_log_only_data()
        relevant_log_columns = set(
            [x for x in preprocessor.relevant_columns if "log_cluster_template" in x]
            + ["coarse_log_cluster_path"]
        )
        attribute_hierarchy = self._load_attribute_hierarchy(
            huawei_df, set(["coarse_log_cluster_path"])
        )
        return (
            attribute_hierarchy.append(
                self._load_log_hierarchy(huawei_df, relevant_log_columns),
                ignore_index=True,
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _load_attribute_only_hierarchy(self) -> pd.DataFrame:
        preprocessor = ConcurrentAggregatedLogsPreprocessor(self.config)
        huawei_df = preprocessor._load_log_only_data()
        relevant_columns = set(
            [
                x
                for x in preprocessor.relevant_columns
                if "log_cluster_template" not in x
            ]
        )
        attribute_hierarchy = self._load_attribute_hierarchy(
            huawei_df, relevant_columns
        )
        return (
            attribute_hierarchy.append(
                self._load_log_hierarchy(huawei_df, relevant_columns),
                ignore_index=True,
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _load_log_hierarchy(
        self, huawei_df: pd.DataFrame, relevant_columns: Set[str]
    ) -> pd.DataFrame:
        hierarchy_records = []
        for _, row in tqdm(
            huawei_df.iterrows(),
            desc="Adding huawei log hierarchy",
            total=len(huawei_df),
        ):
            log_template = str(row[self.config.relevant_log_column]).lower()
            for column in relevant_columns:
                row_value = (
                    column + "#" + str(row[column]).lower()
                    if len(str(row[column])) > 0
                    else ""
                )
                if len(row_value) == 0:
                    continue

                hierarchy_records.append(
                    {
                        "parent_id": row_value,
                        "parent_name": row_value.split("#")[1],
                        "child_id": self.config.relevant_log_column + "#" + log_template,
                        "child_name": log_template,
                    },
                )
        return (
            pd.DataFrame.from_records(hierarchy_records)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _load_attribute_hierarchy(
        self, huawei_df: pd.DataFrame, relevant_columns: Set[str]
    ) -> pd.DataFrame:
        hierarchy_df = pd.DataFrame(
            columns=["parent_id", "child_id", "parent_name", "child_name"]
        )
        for column in relevant_columns:
            hierarchy_df = hierarchy_df.append(
                {
                    "parent_id": "root",
                    "parent_name": "root",
                    "child_id": column,
                    "child_name": column,
                },
                ignore_index=True,
            )
            values = set(
                [
                    str(x).lower()
                    for x in huawei_df[column]
                    .dropna()
                    .astype(str)
                    .replace(np.nan, "", regex=True)
                    if len(str(x)) > 0 and str(x).lower() != "nan"
                ]
            )
            for value in tqdm(values, desc="Loading hierarchy for column " + column):
                hierarchy_elements = [column]
                if column == "Hostname":
                    hierarchy_elements.append(value.rstrip("0123456789"))
                elif column == "http_status":
                    hierarchy_elements.append(value[0] + "00")
                elif "cluster" in column:
                    hierarchy_elements = hierarchy_elements + value.split()
                else:
                    hierarchy_elements = hierarchy_elements + re.split(
                        "[,._\-\*]+", value
                    )
                    hierarchy_elements = [
                        x.strip() for x in hierarchy_elements if len(x.strip()) > 0
                    ]
                if hierarchy_elements[len(hierarchy_elements) - 1] == value:
                    hierarchy_elements = hierarchy_elements[
                        : len(hierarchy_elements) - 1
                    ]

                hierarchy = []
                for i in range(1, len(hierarchy_elements) + 1):
                    hierarchy.append("->".join(hierarchy_elements[0:i]))
                hierarchy.append(column + "#" + value)

                parent_id = column
                parent_name = column
                for i in range(len(hierarchy)):
                    child_id = hierarchy[i]
                    child_name = child_id.split("->")[-1]
                    if not parent_id == child_id:
                        hierarchy_df = hierarchy_df.append(
                            {
                                "parent_id": parent_id,
                                "parent_name": parent_name,
                                "child_id": child_id,
                                "child_name": child_name,
                            },
                            ignore_index=True,
                        )
                    parent_id = child_id
                    parent_name = child_name

        return hierarchy_df[["parent_id", "child_id", "parent_name", "child_name"]]

    def _generate_hostname_hierarchy(self, hostname: str) -> List[str]:
        name = hostname.rstrip("0123456789")
        return [name]

    def _generate_http_hierarchy(self, http_code: str) -> List[str]:
        return [http_code[0] + "XX"]


class ConcurrentAggregatedLogsCausalityPreprocessor(Preprocessor):
    def __init__(
        self, config: HuaweiPreprocessorConfig,
    ):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        preprocessor = ConcurrentAggregatedLogsPreprocessor(self.config)
        huawei_df = preprocessor._load_log_only_data().fillna("")
        relevant_columns = set(
            [
                x
                for x in preprocessor.relevant_columns
                if not self.config.log_only_causality or "log" in x
            ]
        )
        counted_causality = self._generate_counted_causality(
            huawei_df, relevant_columns
        )

        causality_records = []
        for from_value, to_values in tqdm(
            counted_causality.items(),
            desc="Generating causality df from counted causality",
        ):
            total_to_counts = len(to_values)
            to_values_counter: Dict[str, int] = Counter(to_values)
            for to_value, to_count in to_values_counter.items():
                if to_count / total_to_counts > self.config.min_causality:
                    causality_records.append(
                        {
                            "parent_id": from_value,
                            "parent_name": from_value.split("#")[1],
                            "child_id": to_value,
                            "child_name": to_value.split("#")[1],
                        },
                    )

        return (
            pd.DataFrame.from_records(causality_records)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _generate_counted_causality(
        self, df: pd.DataFrame, relevant_columns: Set[str]
    ) -> Dict[str, List[str]]:
        causality: Dict[str, List[str]] = {}
        previous_row = None
        for _, row in tqdm(
            df.iterrows(),
            desc="Generating counted causality for Huawei log data",
            total=len(df),
        ):
            if previous_row is None:
                previous_row = row
                continue
            for previous_column in relevant_columns:
                previous_column_value = (
                    previous_column + "#" + str(previous_row[previous_column]).lower()
                    if len(str(previous_row[previous_column])) > 0
                    else ""
                )
                if len(previous_column_value) < 1:
                    continue
                if previous_column_value not in causality:
                    causality[previous_column_value] = []
                for current_column in relevant_columns:
                    current_column_value = (
                        current_column + "#" + str(row[current_column]).lower()
                        if len(str(row[current_column])) > 0
                        else ""
                    )
                    if len(current_column_value) < 1:
                        continue
                    if current_column_value not in causality[previous_column_value]:
                        causality[previous_column_value].append(current_column_value)
                    else:
                        causality[previous_column_value].append(current_column_value)
            previous_row = row
        return causality

