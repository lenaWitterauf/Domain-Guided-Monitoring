import unittest
import pandas as pd
from pathlib import Path

from src.features.preprocessing.huawei import (
    ConcurrentAggregatedLogsPreprocessor,
    ConcurrentAggregatedLogsDescriptionPreprocessor,
    ConcurrentAggregatedLogsHierarchyPreprocessor,
)
from typing import List
import dataclasses
from ...test_utils import transform_to_string


class TestHuawei(unittest.TestCase):
    def test_huawei_preprocessing(self):
        config = HuaweiPreprocessorConfig()
        config.aggregated_log_file = Path(
            "../tests/resources/test_huawei_aggregated_logs.csv"
        )
        config.log_datetime_column_name = "time"
        config.relevant_aggregated_log_columns = ["a", "b"]
        config.log_payload_column_name = "text"
        config.url_column_name = "url"
        fixture = ConcurrentAggregatedLogsPreprocessor(config=config,)

        expected_df = pd.DataFrame(
            data={
                fixture.sequence_column_name: [
                    [
                        [
                            "a#a.a1",
                            "b#b.b1",
                            "log_cluster_template#this is a log line",
                            "url_cluster_template#this is an url",
                        ],
                        [
                            "a#a.a2",
                            "b#b.b2",
                            "log_cluster_template#this is a log line",
                            "url_cluster_template#this is an url",
                        ],
                        [
                            "",
                            "b#b_b2",
                            "log_cluster_template#this is a log line",
                            "url_cluster_template#this is an url",
                        ],
                        [
                            "a#a.a3",
                            "b#b",
                            "log_cluster_template#this is a log line",
                            "url_cluster_template#this is an url",
                        ],
                    ],
                ],
            },
        )
        expected_df["str_df"] = expected_df[fixture.sequence_column_name].apply(
            lambda x: transform_to_string(x)
        )
        aggregated_df = fixture.load_data()
        aggregated_df["str_df"] = aggregated_df[fixture.sequence_column_name].apply(
            lambda x: transform_to_string(x)
        )
        pd.testing.assert_frame_equal(
            expected_df[["str_df"]], aggregated_df[["str_df"]], check_like=True,
        )

    def test_huawei_descriptions(self):
        config = HuaweiPreprocessorConfig()
        config.aggregated_log_file = Path(
            "../tests/resources/test_huawei_aggregated_logs.csv"
        )
        config.log_datetime_column_name = "time"
        config.relevant_aggregated_log_columns = ["a", "b"]
        config.log_payload_column_name = "text"
        config.url_column_name = "url"
        fixture = ConcurrentAggregatedLogsDescriptionPreprocessor(config=config,)

        expected_df = pd.DataFrame(
            data={
                "label": [
                    "a#a.a1",
                    "a#a.a2",
                    "a#a.a3",
                    "b#b.b1",
                    "b#b.b2",
                    "b#b_b2",
                    "b#b",
                ],
                "description": ["a a1", "a a2", "a a3", "b b1", "b b2", "b b2", "b"],
            },
        )
        aggregated_df = fixture.load_data()
        aggregated_df = aggregated_df[~aggregated_df["label"].str.contains("cluster")]

        pd.testing.assert_frame_equal(
            expected_df.sort_values(by="label", ignore_index=True),
            aggregated_df.sort_values(by="label", ignore_index=True),
            check_like=True,
        )

    def test_huawei_hierarchy(self):
        config = HuaweiPreprocessorConfig()
        config.aggregated_log_file = Path(
            "../tests/resources/test_huawei_aggregated_logs.csv"
        )
        config.log_datetime_column_name = "time"
        config.relevant_aggregated_log_columns = ["a", "b"]
        config.log_payload_column_name = "text"
        config.url_column_name = "url"
        fixture = ConcurrentAggregatedLogsHierarchyPreprocessor(config=config,)

        expected_df = pd.DataFrame(
            data={
                "parent_id": [
                    "root",
                    "root",
                    "a",
                    "a->a",
                    "a->a->a1",
                    "a->a",
                    "a->a->a2",
                    "a->a",
                    "a->a->a3",
                    "b",
                    "b",
                    "b->b",
                    "b->b->b1",
                    "b->b",
                    "b->b->b2",
                    "b->b->b2",
                ],
                "child_id": [
                    "a",
                    "b",
                    "a->a",
                    "a->a->a1",
                    "a#a.a1",
                    "a->a->a2",
                    "a#a.a2",
                    "a->a->a3",
                    "a#a.a3",
                    "b->b",
                    "b#b",
                    "b->b->b1",
                    "b#b.b1",
                    "b->b->b2",
                    "b#b.b2",
                    "b#b_b2",
                ],
            },
        )
        aggregated_df = fixture.load_data().drop_duplicates()[["parent_id", "child_id"]]
        aggregated_df = aggregated_df[
            ~aggregated_df["parent_id"].str.contains("cluster")
        ]
        aggregated_df = aggregated_df[
            ~aggregated_df["child_id"].str.contains("cluster")
        ]
        pd.testing.assert_frame_equal(
            expected_df.sort_values(by=["parent_id", "child_id"], ignore_index=True),
            aggregated_df.sort_values(by=["parent_id", "child_id"], ignore_index=True),
            check_like=True,
        )


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
    log_datetime_column_name: str = "@timestamp"
    log_payload_column_name: str = "Payload"
    drain_log_depth: int = 10
    drain_log_st: float = 0.75
    url_column_name: str = "http_url"
    drain_url_depth: int = 10
    drain_url_st: float = 0.5
    add_log_clusters: bool = True
    min_logs_per_trace: int = 2
