from unicodedata import digit
import dataclass_cli
import dataclasses
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
from .base import Preprocessor
from .icd9data import ICD9DataPreprocessor, ICD9KnowlifeMatcher


def _convert_to_icd9(dxStr: str):
    if dxStr.startswith("E"):
        if len(dxStr) > 4:
            return dxStr[:4] + "." + dxStr[4:]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3] + "." + dxStr[3:]
        else:
            return dxStr


def _convert_to_3digit_icd9(dxStr: str):
    if dxStr.startswith("E"):
        if len(dxStr) > 4:
            return dxStr[:4]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3]
        else:
            return dxStr


@dataclass_cli.add
@dataclasses.dataclass
class MimicPreprocessorConfig:
    admission_file: Path = Path("data/ADMISSIONS.csv")
    diagnosis_file: Path = Path("data/DIAGNOSES_ICD.csv")
    hierarchy_file: Path = Path("data/ccs_multi_dx_tool_2015.csv")
    icd9_file: Path = Path("data/icd9.csv")
    use_icd9_data: bool = True
    min_admissions_per_user: int = 2
    sequence_column_name: str = "icd9_code_converted_3digits"
    add_icd9_info_to_sequences: bool = True
    cluster_file: Path = Path("data/icd9_clusters.csv")
    knowlife_file: Path = Path("data/knowlife_dump.tsv")
    umls_file: Path = Path("data/umls.csv")
    umls_api_key: str = ""
    replace_keys: List[str] = dataclasses.field(default_factory=lambda: [],)
    replace_with_keys: List[str] = dataclasses.field(default_factory=lambda: [],)
    replacement_percentages: List[float] = dataclasses.field(
        default_factory=lambda: [],
    )
    replace_columns: List[str] = dataclasses.field(default_factory=lambda: [],)
    prediction_column: str = ""

class ICD9HierarchyPreprocessor(Preprocessor):
    def __init__(self, config: MimicPreprocessorConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        logging.info("Starting to preprocess ICD9 hierarchy")
        hierarchy_df = self._read_hierarchy_df()
        hierarchy_df = self._transform_hierarchy_df(hierarchy_df)
        if len(self.config.replace_keys) > 0:
            hierarchy_df = self._add_noise_connections(hierarchy_df)
        return hierarchy_df

    def _read_hierarchy_df(self) -> pd.DataFrame:
        return ICD9DataPreprocessor(self.config.icd9_file).load_data()

    def _transform_hierarchy_df(self, hierarchy_df: pd.DataFrame):
        hierarchy_df["parent_id"] = hierarchy_df["parent_code"]
        hierarchy_df["child_id"] = hierarchy_df["child_code"]

        if len(self.config.prediction_column) > 0:
            hierarchy_df["child_id"] = hierarchy_df["child_id"].apply(
                lambda x: self.config.prediction_column + "#" + str(x)
            )
            hierarchy_df["parent_id"] = hierarchy_df["parent_id"].apply(
                lambda x: self.config.prediction_column + "#" + str(x)
            )

        return hierarchy_df[
            ["parent_id", "child_id", "parent_name", "child_name"]
        ]

    def _add_noise_connections(self, hierarchy_df: pd.DataFrame):
        to_replace_keys = [str(x) for x in self.config.replace_keys]
        replacement_keys = [str(x) for x in self.config.replace_with_keys]
        if not len(to_replace_keys) == len(replacement_keys):
            logging.error(
                "Unable to add MIMIC noise connections, different list sizes: %d, %d",
                len(to_replace_keys),
                len(replacement_keys),
            )
            return hierarchy_df

        for idx in tqdm(
            range(len(to_replace_keys)),
            desc="Adding noise connections for MIMIC Hierarchy",
        ):
            to_replace_name = (
                hierarchy_df[hierarchy_df["child_id"] == to_replace_keys[idx]]
                .reset_index(drop=True)["child_name"]
                .to_list()[0]
            )
            hierarchy_df = hierarchy_df.append(
                {
                    "child_id": to_replace_keys[idx],
                    "child_name": to_replace_name,
                    "parent_id": "NOISENODE" + str(idx),
                    "parent_name": "NOISENODE" + str(idx),
                },
                ignore_index=True,
            ).append(
                {
                    "child_id": replacement_keys[idx],
                    "child_name": replacement_keys[idx],
                    "parent_id": "NOISENODE" + str(idx),
                    "parent_name": "NOISENODE" + str(idx),
                },
                ignore_index=True,
            )

        return hierarchy_df


class CCSHierarchyPreprocessor(Preprocessor):
    def __init__(self, config: MimicPreprocessorConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        logging.info("Starting to preprocess CCS hierarchy")
        hierarchy_df = self._read_hierarchy_df()
        return self._transform_hierarchy_df(hierarchy_df)

    def _read_hierarchy_df(self) -> pd.DataFrame:
        logging.info("Reading hierarchy_df from %s", self.config.hierarchy_file)
        return pd.read_csv(self.config.hierarchy_file, quotechar="'", dtype=str)

    def _transform_hierarchy_df(self, hierarchy_df: pd.DataFrame):
        transformed_hierarchy_df = pd.DataFrame(
            columns=["parent_id", "child_id", "parent_name", "child_name"]
        )
        for _, row in tqdm(
            hierarchy_df.iterrows(),
            desc="Building flat hierarchy df",
            total=len(transformed_hierarchy_df),
        ):
            all_parents: List[Tuple[str, str]] = list(
                zip(
                    [
                        row["CCS LVL 1"],
                        row["CCS LVL 2"],
                        row["CCS LVL 3"],
                        row["CCS LVL 4"],
                    ],
                    [
                        row["CCS LVL 1 LABEL"],
                        row["CCS LVL 2 LABEL"],
                        row["CCS LVL 3 LABEL"],
                        row["CCS LVL 4 LABEL"],
                    ],
                )
            )
            all_parents = [(str(id).strip(), name) for (id, name) in all_parents]
            all_parents = [(id, name) for (id, name) in all_parents if len(id) > 0]

            # Labels are sorted from general -> specific

            transformed_hierarchy_df = transformed_hierarchy_df.append(
                pd.DataFrame(
                    data={
                        "parent_id": [id for (id, _) in all_parents],
                        "parent_name": [name for (_, name) in all_parents],
                        "child_id": [id for (id, _) in all_parents[1:]]
                        + [_convert_to_3digit_icd9(row["ICD-9-CM CODE"])],
                        "child_name": [name for (_, name) in all_parents[1:]]
                        + [_convert_to_3digit_icd9(row["ICD-9-CM CODE"])],
                    }
                )
            )

        return transformed_hierarchy_df


class ICD9DescriptionPreprocessor(Preprocessor):
    def __init__(self, config: MimicPreprocessorConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        logging.info("Starting to preprocess ICD9 descriptions")
        description_df = self._read_description_df()
        description_df["label"] = description_df["child_code"]
        if len(self.config.prediction_column) > 0:
            description_df["label"] = description_df["label"].apply(lambda x: self.config.prediction_column + "#" + x)

        description_df["description"] = description_df["child_name"].apply(
            lambda x: x.replace('"', "")
        )
        if len(self.config.replace_keys) > 0:
            description_df = self._add_noise_connections(description_df)
        return description_df[["label", "description"]]

    def _read_description_df(self) -> pd.DataFrame:
        return ICD9DataPreprocessor(self.config.icd9_file).load_data()

    def _add_noise_connections(self, description_df: pd.DataFrame):
        to_replace_keys = [str(x) for x in self.config.replace_keys]
        replacement_keys = [str(x) for x in self.config.replace_with_keys]
        if not len(to_replace_keys) == len(replacement_keys):
            logging.error(
                "Unable to add MIMIC noise connections, different list sizes: %d, %d",
                len(to_replace_keys),
                len(replacement_keys),
            )
            return description_df

        for idx in tqdm(
            range(len(to_replace_keys)),
            desc="Adding noise connections for MIMIC Descriptions",
        ):
            description_df.loc[
                description_df["label"] == to_replace_keys[idx], "description"
            ] = (
                description_df.loc[
                    description_df["label"] == to_replace_keys[idx], "description"
                ]
                + " NOISENODE"
                + str(idx)
            )
            description_df = description_df.append(
                {
                    "label": replacement_keys[idx],
                    "description": "NOISENODE" + str(idx),
                },
                ignore_index=True,
            )

        return description_df


class KnowlifePreprocessor(Preprocessor):
    def __init__(
        self, config: MimicPreprocessorConfig,
    ):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        logging.info("Starting to preprocess Knowlife causality")
        knowlife_df = self._read_knowlife_df()
        knowlife_icd9_matching = (
            self._read_knowlife_icd_mapping(knowlife_df)
            .drop_duplicates()
            .groupby(by="cui")
            .agg({"icd9_code": lambda x: list(x),})
        )
        left_knowlife_df = pd.merge(
            knowlife_df,
            knowlife_icd9_matching,
            left_on="leftfactentity",
            right_on="cui",
            how="left",
        )
        right_knowlife_df = pd.merge(
            knowlife_df,
            knowlife_icd9_matching,
            left_on="rightfactentity",
            right_on="cui",
            how="left",
        )
        knowlife_df["parent_id"] = left_knowlife_df["icd9_code"]
        knowlife_df["child_id"] = right_knowlife_df["icd9_code"]
        knowlife_df = (
            knowlife_df.explode(column="parent_id")
            .explode(column="child_id")
            .dropna(subset=["parent_id", "child_id"],)
            .drop_duplicates(subset=["parent_id", "child_id"],)
            .reset_index(drop=True)
        )

        knowlife_df["parent_name"] = knowlife_df["parent_id"]
        knowlife_df["child_name"] = knowlife_df["child_id"]
        if len(self.config.prediction_column) > 0:
            knowlife_df["child_id"] = knowlife_df["child_id"].apply(
                lambda x: self.config.prediction_column + "#" + str(x)
            )
            knowlife_df["parent_id"] = knowlife_df["parent_id"].apply(
                lambda x: self.config.prediction_column + "#" + str(x)
            )

        if len(self.config.replace_keys) > 0:
            knowlife_df = self._add_noise_connections(knowlife_df)
        return knowlife_df[["parent_id", "child_id", "parent_name", "child_name"]]

    def _read_knowlife_icd_mapping(self, knowlife_df: pd.DataFrame) -> pd.DataFrame:
        return ICD9KnowlifeMatcher(
            self.config.umls_file, self.config.umls_api_key
        ).load_data(knowlife_df)

    def _read_knowlife_df(self) -> pd.DataFrame:
        knowlife_df = pd.read_csv(self.config.knowlife_file, sep="\t")
        knowlife_df = knowlife_df[knowlife_df["relation"] == "causes"].reset_index(
            drop=True
        )
        return knowlife_df

    def _add_noise_connections(self, knowlife_df: pd.DataFrame):
        to_replace_keys = [str(x) for x in self.config.replace_keys]
        replacement_keys = [str(x) for x in self.config.replace_with_keys]
        if not len(to_replace_keys) == len(replacement_keys):
            logging.error(
                "Unable to add MIMIC noise connections, different list sizes: %d, %d",
                len(to_replace_keys),
                len(replacement_keys),
            )
            return knowlife_df

        for idx in tqdm(
            range(len(to_replace_keys)),
            desc="Adding noise connections for MIMIC Knowlife Causality",
        ):
            knowlife_df = knowlife_df.append(
                {
                    "child_id": to_replace_keys[idx],
                    "child_name": to_replace_keys[idx],
                    "parent_id": "NOISENODE" + str(idx),
                    "parent_name": "NOISENODE" + str(idx),
                },
                ignore_index=True,
            ).append(
                {
                    "child_id": replacement_keys[idx],
                    "child_name": replacement_keys[idx],
                    "parent_id": "NOISENODE" + str(idx),
                    "parent_name": "NOISENODE" + str(idx),
                },
                ignore_index=True,
            )

        return knowlife_df


class MimicPreprocessor(Preprocessor):
    def __init__(
        self, config: MimicPreprocessorConfig,
    ):
        self.config = config
        self.aggregation_column_names = set(
            ["icd9_code", "icd9_code_converted", "icd9_code_converted_3digits",]
        )

    def load_data(self) -> pd.DataFrame:
        logging.info("Starting to preprocess MIMIC dataset")
        admission_df = self._read_admission_df()
        diagnosis_df = self._read_diagnosis_df()
        aggregated_df = self._aggregate_codes_per_admission(
            diagnosis_df=diagnosis_df, admission_df=admission_df
        )
        return aggregated_df[
            aggregated_df["num_admissions"] >= self.config.min_admissions_per_user
        ]

    def _read_admission_df(self) -> pd.DataFrame:
        logging.info("Reading admission_df from %s", self.config.admission_file)
        admission_df = pd.read_csv(self.config.admission_file)
        admission_df.columns = [x.lower() for x in admission_df.columns]
        admission_df["admittime"] = pd.to_datetime(admission_df["admittime"])
        admission_df["dischtime"] = pd.to_datetime(admission_df["dischtime"])
        admission_df["deathtime"] = pd.to_datetime(admission_df["deathtime"])
        admission_df["edregtime"] = pd.to_datetime(admission_df["edregtime"])
        admission_df["edouttime"] = pd.to_datetime(admission_df["edouttime"])
        return admission_df

    def _read_diagnosis_df(self) -> pd.DataFrame:
        logging.info("Reading diagnosis_df from %s", self.config.diagnosis_file)
        diagnosis_df = pd.read_csv(self.config.diagnosis_file)
        diagnosis_df.columns = [x.lower() for x in diagnosis_df.columns]

        diagnosis_df["icd9_code"] = diagnosis_df["icd9_code"].fillna("").apply(str)
        diagnosis_df["icd9_code_converted"] = diagnosis_df["icd9_code"].apply(
            _convert_to_icd9
        )
        diagnosis_df["icd9_code_converted_3digits"] = diagnosis_df["icd9_code"].apply(
            _convert_to_3digit_icd9
        )

        if self.config.add_icd9_info_to_sequences:
            diagnosis_df = self._add_icd9_information(diagnosis_df)
        if self.config.cluster_file.exists():
            diagnosis_df = self._add_cluster_information(diagnosis_df)
        if len(self.config.replace_keys) > 0:
            diagnosis_df = self._add_noise(diagnosis_df)
        if len(self.config.prediction_column) > 0:
            for column in self.aggregation_column_names:
                diagnosis_df[column] = diagnosis_df[column].apply(lambda x: str(column) + "#" + str(x))

        diagnosis_df["level_all"] = diagnosis_df[self.aggregation_column_names].apply(lambda x: list(x), axis=1)
        self.aggregation_column_names.add("level_all")
        return diagnosis_df

    def _add_cluster_information(self, diagnosis_df: pd.DataFrame) -> pd.DataFrame:
        cluster_df = pd.read_csv(self.config.cluster_file)
        self.aggregation_column_names.update(cluster_df.columns)
        return pd.merge(
            diagnosis_df,
            cluster_df,
            how="inner",
            left_on="icd9_code_converted",
            right_on="original_level_cluster",
        )

    def _add_noise(self, diagnosis_df: pd.DataFrame) -> pd.DataFrame:
        to_replace_keys = [str(x) for x in self.config.replace_keys]
        replacement_keys = [str(x) for x in self.config.replace_with_keys]
        replacement_percentages = [
            float(x) for x in self.config.replacement_percentages
        ]
        replacement_columns = self.config.replace_columns
        if (
            not len(to_replace_keys) == len(replacement_keys)
            or not len(to_replace_keys) == len(replacement_percentages)
            or not len(to_replace_keys) == len(replacement_columns)
        ):
            logging.error(
                "Unable to add MIMIC noise, different list sizes: %d, %d, %d, %d",
                len(to_replace_keys),
                len(replacement_keys),
                len(replacement_percentages),
                len(replacement_columns),
            )
            return diagnosis_df

        for idx in tqdm(
            range(len(to_replace_keys)), desc="Adding noise to MIMIC dataset"
        ):
            replace_samples = diagnosis_df[
                diagnosis_df[replacement_columns[idx]] == to_replace_keys[idx]
            ].sample(frac=replacement_percentages[idx])

            diagnosis_df.loc[
                replace_samples.index, replacement_columns[idx]
            ] = replacement_keys[idx]

        return diagnosis_df

    def _add_icd9_information(self, diagnosis_df: pd.DataFrame) -> pd.DataFrame:
        icd9_preprocessor = ICD9DataPreprocessor(self.config.icd9_file)

        icd9_df = icd9_preprocessor.load_data()[
            ["child_code", "child_name"]
        ].drop_duplicates()
        diagnosis_df["icd9_code_name"] = pd.merge(
            diagnosis_df,
            icd9_df,
            how="left",
            left_on="icd9_code_converted",
            right_on="child_code",
        )["child_name"].fillna(diagnosis_df["icd9_code_converted"])
        diagnosis_df["icd9_code_name_3digits"] = pd.merge(
            diagnosis_df,
            icd9_df,
            how="left",
            left_on="icd9_code_converted_3digits",
            right_on="child_code",
        )["child_name"].fillna(diagnosis_df["icd9_code_converted_3digits"])
        self.aggregation_column_names.update(
            ["icd9_code_name", "icd9_code_name_3digits",]
        )

        icd9_hierarchy_df = icd9_preprocessor.load_data_as_hierarchy()
        self.aggregation_column_names.update(icd9_hierarchy_df.columns)
        return pd.merge(
            diagnosis_df,
            icd9_hierarchy_df,
            how="inner",
            left_on="icd9_code_converted",
            right_on="level_0",
        )

    def _aggregate_codes_per_admission(
        self, diagnosis_df: pd.DataFrame, admission_df: pd.DataFrame
    ) -> pd.DataFrame:
        codes_per_admission = diagnosis_df.groupby("hadm_id").agg(
            {
                column_name: lambda x: list(x)
                for column_name in self.aggregation_column_names
            }
        )
        if "level_all" in codes_per_admission.columns:
            codes_per_admission["level_all"] = codes_per_admission["level_all"].apply(
                lambda x: [c for sublist in x for c in sublist]
            )

        combined_df = pd.merge(admission_df, codes_per_admission, on=["hadm_id"])

        subject_aggregation_column_names = list(self.aggregation_column_names) + [
            "hadm_id",
            "admittime",
            "diagnosis",
        ]
        admissions_per_subject = (
            combined_df.groupby("subject_id")
            .agg(
                {
                    column_name: lambda x: list(x)
                    for column_name in set(subject_aggregation_column_names)
                }
            )
            .reset_index()
        )
        admissions_per_subject["num_admissions"] = admissions_per_subject[
            "hadm_id"
        ].apply(len)
        return admissions_per_subject
