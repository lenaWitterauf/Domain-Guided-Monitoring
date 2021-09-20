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


def convert_prediction_df(
    prediction_df: pd.DataFrame,
    input_frequency_dict: Dict[str, Dict[str, float]],
    output_percentile_dict: Dict[str, int],
    num_percentiles: int = 10,
) -> pd.DataFrame:
    prediction_df["input_converted"] = prediction_df["input"].apply(
        lambda x: " -> ".join(
            [
                ", ".join([str(val) for val in sorted(v)])
                for (_, v) in sorted(ast.literal_eval(x).items(), key=lambda y: y[0])
            ]
        )
    )
    prediction_df["inputs"] = prediction_df["input"].apply(
        lambda x: ", ".join(
            [
                ", ".join([str(val) for val in sorted(v)])
                for (_, v) in sorted(ast.literal_eval(x).items(), key=lambda y: y[0])
            ]
        )
        + ","
    )
    prediction_df["inputs_frequencies"] = prediction_df["inputs"].apply(
        lambda x: [
            input_frequency_dict[i.strip()]["relative_frequency"]
            for i in x.split(",")
            if len(i.strip()) > 0
        ]
    )
    prediction_df["num_inputs"] = prediction_df["inputs_frequencies"].apply(
        lambda x: len(x)
    )
    prediction_df["input_frequencies"] = prediction_df["input"].apply(
        lambda x: get_frequency_list(x, input_frequency_dict)
    )
    prediction_df["avg_input_frequencies"] = prediction_df["input_frequencies"].apply(
        lambda x: sum(x) / len(x)
    )
    prediction_df["median_input_frequencies"] = prediction_df[
        "input_frequencies"
    ].apply(lambda x: sorted(x)[int(len(x) * 0.1)])
    prediction_df["min_input_frequencies"] = prediction_df["input_frequencies"].apply(
        lambda x: min(x)
    )
    prediction_df["p10_input_frequencies"] = prediction_df["input_frequencies"].apply(
        lambda x: sorted(x)[int(len(x) * 0.1)]
    )
    prediction_df["unknown_inputs"] = prediction_df["input_frequencies"].apply(
        lambda x: len([y for y in x if float(y) == 0.0]) / len(x)
    )
    prediction_df = add_input_frequency_percentile(
        prediction_df,
        input_frequency_column="avg_input_frequencies",
        num_percentiles=num_percentiles,
    )
    prediction_df = add_input_frequency_percentile(
        prediction_df,
        input_frequency_column="median_input_frequencies",
        num_percentiles=num_percentiles,
    )
    prediction_df = add_input_frequency_percentile(
        prediction_df,
        input_frequency_column="min_input_frequencies",
        num_percentiles=num_percentiles,
    )
    prediction_df = add_input_frequency_percentile(
        prediction_df,
        input_frequency_column="p10_input_frequencies",
        num_percentiles=num_percentiles,
    )
    prediction_df = add_input_frequency_percentile(
        prediction_df,
        input_frequency_column="unknown_inputs",
        num_percentiles=num_percentiles,
    )
    prediction_df = add_input_frequency_range(
        prediction_df,
        input_frequency_column="avg_input_frequencies",
        num_percentiles=num_percentiles,
    )
    prediction_df = add_input_frequency_range(
        prediction_df,
        input_frequency_column="median_input_frequencies",
        num_percentiles=num_percentiles,
    )
    prediction_df = add_input_frequency_range(
        prediction_df,
        input_frequency_column="min_input_frequencies",
        num_percentiles=num_percentiles,
    )
    prediction_df = add_input_frequency_range(
        prediction_df,
        input_frequency_column="p10_input_frequencies",
        num_percentiles=num_percentiles,
    )
    prediction_df = add_input_frequency_range(
        prediction_df,
        input_frequency_column="unknown_inputs",
        num_percentiles=num_percentiles,
    )
    prediction_df["output"] = prediction_df["output"].apply(
        lambda x: ast.literal_eval(x)
    )
    prediction_df = prediction_df.explode("output")
    prediction_df["output_frequency_percentile"] = prediction_df["output"].apply(
        lambda x: output_percentile_dict[x]
    )
    prediction_df["output_rank_noties"] = prediction_df[
        ["output", "predictions"]
    ].apply(lambda x: get_best_rank_of(x[0], x[1]), axis=1)
    return prediction_df


def add_input_frequency_range(
    prediction_df: pd.DataFrame,
    num_percentiles: int = 10,
    input_frequency_column: str = "avg_input_frequencies",
) -> pd.DataFrame:
    max_value = max(prediction_df[input_frequency_column])
    min_value = min(prediction_df[input_frequency_column])
    bucket_size = (max_value - min_value) / num_percentiles
    input_percentile_values = [-1] * num_percentiles
    for i in range(num_percentiles):
        input_percentile_values[i] = min_value + i * bucket_size
    input_percentile_values.append(max_value)
    prediction_df[input_frequency_column + "_range"] = prediction_df[
        input_frequency_column
    ].apply(
        lambda x: max(
            [i for i in range(num_percentiles) if input_percentile_values[i] <= x]
        )
    )
    return prediction_df


def add_input_frequency_percentile(
    prediction_df: pd.DataFrame,
    num_percentiles: int = 10,
    input_frequency_column: str = "avg_input_frequencies",
) -> pd.DataFrame:
    prediction_df = prediction_df.reset_index(drop=True)
    bucket_size = len(prediction_df) / num_percentiles
    sorted_indices = prediction_df[input_frequency_column].sort_values().index
    for i in range(num_percentiles):
        percentile_indices = sorted_indices[
            int(i * bucket_size) : int((i + 1) * bucket_size)
        ]
        prediction_df.loc[
            percentile_indices, input_frequency_column + "_percentile"
        ] = int(i)
    return prediction_df


def load_prediction_df(
    run_id: str,
    local_mlflow_dir: str,
    num_percentiles: int = 10,
    convert_df: bool = True,
) -> Optional[pd.DataFrame]:
    run_mlflow_dir = Path(local_mlflow_dir + run_id)
    if not run_mlflow_dir.is_dir():
        print("Run {} is not in local MlFlow dir".format(run_id))
        return None

    input_frequency_dict = load_input_frequency_dict(run_id, local_mlflow_dir)
    if input_frequency_dict is None:
        print("No frequency file for run {} in local MlFlow dir".format(run_id))
        return None

    output_percentile_dict = load_output_percentile_mapping_dict(
        run_id, local_mlflow_dir
    )
    if output_percentile_dict is None:
        print("No output percentile file for run {} in local MlFlow dir".format(run_id))
        return None

    run_prediction_output_path = Path(
        local_mlflow_dir + run_id + "/artifacts/prediction_output.csv"
    )
    if not run_prediction_output_path.exists():
        print("No prediction output file for run {} in local MlFlow dir".format(run_id))
        return None
    prediction_output_df = pd.read_csv(run_prediction_output_path)

    if convert_df:
        prediction_output_df = convert_prediction_df(
            prediction_df=prediction_output_df,
            input_frequency_dict=input_frequency_dict,
            output_percentile_dict=output_percentile_dict,
            num_percentiles=num_percentiles,
        )

    return prediction_output_df
        
def load_icd9_text() -> Dict[str, Dict[str, str]]:
    icd9_df = pd.read_csv('../data/icd9.csv')
    return icd9_df[['child_name', 'child_code']].drop_duplicates().rename(columns={
        'child_name': 'description',
        'child_code': 'code',
    }).set_index('code').to_dict('index')