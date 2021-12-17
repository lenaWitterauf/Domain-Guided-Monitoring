import pandas as pd
import ast
from pathlib import Path
import json
from typing import Dict, List, Optional


def load_input_frequency_dict(
    run_id: str, local_mlflow_dir: str
) -> Optional[Dict[str, Dict[str, float]]]:
    run_frequency_path = Path(
        local_mlflow_dir + run_id + "/artifacts/train_frequency.csv"
    )
    if not run_frequency_path.exists():
        print("No frequency file for run {} in local MlFlow dir".format(run_id))
        return None

    input_frequency_df = pd.read_csv(run_frequency_path).set_index("feature")
    if "absolue_frequency" in input_frequency_df.columns:
        input_frequency_df["absolute_frequency"] = input_frequency_df[
            "absolue_frequency"
        ]
    input_frequency_df["relative_frequency"] = input_frequency_df[
        "absolute_frequency"
    ] / sum(input_frequency_df["absolute_frequency"])
    return input_frequency_df.to_dict("index")


def load_attention_weights(
    run_id: str, local_mlflow_dir: str
) -> Optional[Dict[str, Dict[str, float]]]:
    attention_path = Path(local_mlflow_dir + run_id + "/artifacts/attention.json")
    if not attention_path.exists():
        print("No attention file for run {} in local MlFlow dir".format(run_id))
        return None

    with open(attention_path) as attention_file:
        return json.load(attention_file)["attention_weights"]


def load_output_percentile_mapping_dict(
    run_id: str, local_mlflow_dir: str
) -> Optional[Dict[str, int]]:
    run_percentile_mapping_path = Path(
        local_mlflow_dir + run_id + "/artifacts/percentile_mapping.json"
    )
    if not run_percentile_mapping_path.exists():
        print(
            "No percentile mapping file for run {} in local MlFlow dir".format(run_id)
        )
        return None

    with open(run_percentile_mapping_path) as percentile_file:
        percentile_mapping = json.load(percentile_file)
        return {
            str(label): int(percentile)
            for percentile, percentile_info in percentile_mapping.items()
            for label in percentile_info["percentile_classes"]
        }


def get_best_rank_of(output: str, predictions_str: str) -> int:
    predictions: Dict[str, float] = ast.literal_eval(predictions_str)
    return len([x for x in predictions if predictions[x] > predictions[output]])


def get_frequency_list(
    input: str,
    input_frequency_dict: Dict[str, Dict[str, float]],
    frequency_type: str = "relative_frequency",
) -> List[float]:
    frequencies = [
        input_frequency_dict[input_feature][frequency_type]
        for _, input_features in sorted(
            ast.literal_eval(input).items(), key=lambda x: x[0]
        )
        for input_feature in sorted(input_features)
    ]
    return frequencies

def load_icd9_text() -> Dict[str, Dict[str, str]]:
    icd9_df = pd.read_csv("../data/icd9.csv")
    return (
        icd9_df[["child_name", "child_code"]]
        .drop_duplicates()
        .rename(columns={"child_name": "description", "child_code": "code",})
        .set_index("code")
        .to_dict("index")
    )

