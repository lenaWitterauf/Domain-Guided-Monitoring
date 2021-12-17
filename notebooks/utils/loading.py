import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Callable
from base import (
    get_best_rank_of,
    get_frequency_list,
    load_input_frequency_dict,
    load_output_percentile_mapping_dict,
    load_attention_weights,
)
from attention_graph import calculate_attention_importances


def add_input_percentiles(
    prediction_df: pd.DataFrame,
    input_frequency_dict: Optional[Dict[str, Dict[str, float]]],
    num_percentiles: int,
) -> pd.DataFrame:
    if input_frequency_dict is None:
        return prediction_df

    prediction_df["inputs_frequencies"] = prediction_df["original_inputs"].apply(
        lambda x: [
            input_frequency_dict.get(i.strip(), {}).get("relative_frequency", 0.0)
            for i in x.split(",")
            if len(i.strip()) > 0
        ]
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

    for column in [
        "avg_input_frequencies",
        "median_input_frequencies",
        "min_input_frequencies",
        "p10_input_frequencies",
        "unknown_inputs",
    ]:
        if column in prediction_df.columns:
            prediction_df = add_frequency_percentile(
                prediction_df,
                frequency_column=column,
                num_percentiles=num_percentiles,
            )
            prediction_df = add_frequency_range(
                prediction_df,
                frequency_column=column,
                num_percentiles=num_percentiles,
                min_max_range=(0,1),
            )
    return prediction_df


def _unclustered_inputs(
    inputs: List[str],
    attention_weights: Dict[str, Dict[str, float]],
    threshold: float = 0.9,
) -> int:
    unclustered_inputs = [
        input
        for input in inputs
        if len(
            [
                x
                for x in attention_weights.get(input, {input: 1.0}).values()
                if float(x) > threshold
            ]
        )
        == 0
    ]
    return len(unclustered_inputs) 

def _single_clustered_inputs(
    inputs: List[str],
    attention_weights: Dict[str, Dict[str, float]],
    attention_importances: Dict[str, List[Tuple[str, float]]],
    threshold: float = 0.9,
) -> int:
    input_clusters = [
        [
            x
            for x,weight in attention_weights.get(input, {input: 1.0}).items()
            if float(weight) > threshold
        ]
        for input in inputs
    ]
    input_cluster_nodes = [x[0] for x in input_clusters if len(x) > 0]
    single_clusters = [
        x for x in input_cluster_nodes 
        if len([y for y in attention_importances.get(x, []) if float(y[1]) > threshold]) <= 1
    ]
    return len(single_clusters)

def _input_cluster_frequency(
    inputs: List[str],
    attention_weights: Dict[str, Dict[str, float]],
    attention_importances: Dict[str, List[Tuple[str, float]]],
    input_frequency_dict: Dict[str, Dict[str, float]],
    threshold: float = 0.9,
    aggregation: Callable[[List[float]], float] = np.mean,
) -> float:
    input_clusters = {
        input:[
            x
            for x,weight in attention_weights.get(input, {input: 1.0}).items()
            if float(weight) > threshold
        ]
        for input in inputs
    }
    input_cluster_nodes = [
        (x[0] if len(x) > 0 else input)
        for input,x in input_clusters.items()
    ]
    input_cluster_frequencies = [
        sum([
            input_frequency_dict.get(x.strip(), {}).get("relative_frequency", 0.0)
            for x,importance in attention_importances.get(node, set([(node, 1.0)]))
            if importance > threshold
        ])
        for node in input_cluster_nodes
    ]
    return aggregation(input_cluster_frequencies)

def add_attention_percentiles(
    prediction_df: pd.DataFrame,
    attention_weights: Dict[str, Dict[str, float]],
    input_frequency_dict: Optional[Dict[str, Dict[str, float]]],
    num_percentiles: int,
    cluster_threshold: float, 
) -> pd.DataFrame:
    attention_importances = calculate_attention_importances(attention_weights)
    prediction_df["inputs_list"] = prediction_df["original_inputs"].apply(
        lambda x: [y.strip() for y in x.split(",") if len(y.strip()) > 0]
    )
    prediction_df["unclustered_inputs"] = prediction_df["inputs_list"].apply(
        lambda x: _unclustered_inputs(x, attention_weights=attention_weights, threshold=cluster_threshold)
    )
    prediction_df["unclustered_inputs_perc"] = prediction_df[["unclustered_inputs", "inputs_list"]].apply(
        lambda x: x[0] / len(x[1]), axis=1
    )
    prediction_df["clustered_inputs"] = prediction_df[["unclustered_inputs", "inputs_list"]].apply(
        lambda x: len(x[1]) - x[0], axis=1
    )
    prediction_df["clustered_inputs_perc"] = prediction_df[["clustered_inputs", "inputs_list"]].apply(
        lambda x: x[0] / len(x[1]), axis=1
    )
    prediction_df["single_clustered_inputs"] = prediction_df["inputs_list"].apply(
        lambda x: _single_clustered_inputs(x, attention_weights=attention_weights, attention_importances=attention_importances, threshold=cluster_threshold)
    )
    prediction_df["single_clustered_inputs_perc"] = prediction_df[["single_clustered_inputs", "inputs_list"]].apply(
        lambda x: x[0] / len(x[1]), axis=1
    )
    prediction_df["single_clustered_inputs_clusterperc"] = prediction_df[["single_clustered_inputs", "clustered_inputs"]].apply(
        lambda x: (x[0] / x[1]) if x[1] > 0 else 0, axis=1
    )
    prediction_df["multi_clustered_inputs"] = prediction_df[["single_clustered_inputs", "clustered_inputs"]].apply(
        lambda x: x[1] - x[0], axis=1
    )
    prediction_df["multi_clustered_inputs_perc"] = prediction_df[["multi_clustered_inputs", "inputs_list"]].apply(
        lambda x: x[0] / len(x[1]), axis=1
    )
    prediction_df["multi_clustered_inputs_clusterperc"] = prediction_df[["multi_clustered_inputs", "clustered_inputs"]].apply(
        lambda x: (x[0] / x[1]) if x[1] > 0 else 0, axis=1
    )
    if input_frequency_dict is not None:
        prediction_df["avg_cluster_input_frequency"] = prediction_df["inputs_list"].apply(
            lambda x: _input_cluster_frequency(x, attention_weights=attention_weights, attention_importances=attention_importances, input_frequency_dict=input_frequency_dict, aggregation=np.mean, threshold=cluster_threshold)
        )
        prediction_df["median_cluster_input_frequency"] = prediction_df["inputs_list"].apply(
            lambda x: _input_cluster_frequency(x, attention_weights=attention_weights, attention_importances=attention_importances, input_frequency_dict=input_frequency_dict, aggregation=np.median, threshold=cluster_threshold)
        )
    for column in [
        "unclustered_inputs",
        "unclustered_inputs_perc",
        "clustered_inputs",
        "clustered_inputs_perc",
        "single_clustered_inputs",
        "single_clustered_inputs_perc",
        "single_clustered_inputs_clusterperc",
        "multi_clustered_inputs",
        "multi_clustered_inputs_perc",
        "multi_clustered_inputs_clusterperc",
        "avg_cluster_input_frequency",
        "median_cluster_input_frequency",
    ]:
        if column not in prediction_df.columns:
            continue
        prediction_df = add_frequency_percentile(
            prediction_df,
            frequency_column=column,
            num_percentiles=num_percentiles,
        )
        prediction_df = add_frequency_range(
            prediction_df,
            frequency_column=column,
            num_percentiles=num_percentiles,
            min_max_range=((0.,1.) if "perc" in column else None),
        )

    return prediction_df


def convert_prediction_df(
    prediction_df: pd.DataFrame,
    input_frequency_dict: Optional[Dict[str, Dict[str, float]]],
    output_percentile_dict: Optional[Dict[str, int]],
    attention_weights: Dict[str, Dict[str, float]] = {},
    num_percentiles: int = 10,
    feature_replacements: Dict[str, str] = {},
    cluster_threshold: float = 0.9,
) -> pd.DataFrame:
    prediction_df["input_converted"] = prediction_df["input"].apply(
        lambda x: " -> ".join(
            [
                ", ".join(
                    sorted(
                        set([feature_replacements.get(str(val), str(val)) for val in v])
                    )
                )
                for (_, v) in sorted(ast.literal_eval(x).items(), key=lambda y: y[0])
            ]
        )
    )
    prediction_df["original_inputs"] = prediction_df["input"].apply(
        lambda x: ", ".join(
            [
                ", ".join(sorted(set([str(val) for val in v])))
                for (_, v) in sorted(ast.literal_eval(x).items(), key=lambda y: y[0])
            ]
        )
        + ","
    )
    prediction_df["inputs"] = prediction_df["input"].apply(
        lambda x: ", ".join(
            [
                ", ".join(
                    sorted(
                        set([feature_replacements.get(str(val), str(val)) for val in v])
                    )
                )
                for (_, v) in sorted(ast.literal_eval(x).items(), key=lambda y: y[0])
            ]
        )
        + ","
    )
    prediction_df["num_inputs"] = prediction_df["inputs"].apply(
        lambda x: len(x.split(","))
    )
    prediction_df = add_input_percentiles(
        prediction_df=prediction_df,
        input_frequency_dict=input_frequency_dict,
        num_percentiles=num_percentiles,
    )
    prediction_df = add_attention_percentiles(
        prediction_df=prediction_df,
        attention_weights=attention_weights,
        input_frequency_dict=input_frequency_dict,
        num_percentiles=num_percentiles,
        cluster_threshold=cluster_threshold,
    )

    prediction_df["output"] = prediction_df["output"].apply(
        lambda x: ast.literal_eval(x)
    )
    prediction_df = prediction_df.explode("output")
    prediction_df["output_rank_noties"] = prediction_df[
        ["output", "predictions"]
    ].apply(lambda x: get_best_rank_of(x[0], x[1]), axis=1)

    if output_percentile_dict is not None:
        prediction_df["output_frequency_percentile"] = prediction_df["output"].apply(
            lambda x: output_percentile_dict[x]
        )
    return prediction_df


def add_frequency_range(
    prediction_df: pd.DataFrame,
    num_percentiles: int = 10,
    frequency_column: str = "avg_input_frequencies",
    min_max_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    min_value = min(prediction_df[frequency_column]) if min_max_range is None else min_max_range[0]
    max_value = max(prediction_df[frequency_column]) if min_max_range is None else min_max_range[1]

    bucket_size = (max_value - min_value) / num_percentiles
    input_range_values = [-1] * num_percentiles
    for i in range(num_percentiles):
        input_range_values[i] = min_value + i * bucket_size
    input_range_values[num_percentiles - 1] = max_value
    prediction_df[frequency_column + "_range"] = prediction_df[
        frequency_column
    ].apply(
        lambda x: max(
            [i for i in range(num_percentiles) if input_range_values[i] <= x]
        )
    )
    return prediction_df


def add_frequency_percentile(
    prediction_df: pd.DataFrame,
    num_percentiles: int = 10,
    frequency_column: str = "avg_input_frequencies",
) -> pd.DataFrame:
    prediction_df = prediction_df.reset_index(drop=True)
    bucket_size = len(prediction_df) / num_percentiles
    sorted_indices = prediction_df[frequency_column].sort_values().index
    for i in range(num_percentiles):
        percentile_indices = sorted_indices[
            int(i * bucket_size) : int((i + 1) * bucket_size)
        ]
        prediction_df.loc[
            percentile_indices, frequency_column + "_percentile"
        ] = int(i)
    return prediction_df


def load_prediction_df(
    run_id: str,
    local_mlflow_dir: str,
    num_percentiles: int = 10,
    convert_df: bool = True,
    feature_replacements: Dict[str, str] = {},
    cluster_threshold: float = 0.9,
) -> Optional[pd.DataFrame]:
    run_mlflow_dir = Path(local_mlflow_dir + run_id)
    if not run_mlflow_dir.is_dir():
        print("Run {} is not in local MlFlow dir".format(run_id))

    input_frequency_dict = load_input_frequency_dict(run_id, local_mlflow_dir)
    if input_frequency_dict is None:
        print("No frequency file for run {} in local MlFlow dir".format(run_id))
    elif len(feature_replacements) > 0:
        for child, parent in feature_replacements.items():
            input_frequency_dict[parent] = input_frequency_dict.get(parent, {})
            input_frequency_dict[parent]["absolute_frequency"] = input_frequency_dict[
                parent
            ].get("absolute_frequency", 0) + input_frequency_dict.get(child, {}).get(
                "absolute_frequency", 0
            )
            input_frequency_dict[parent]["absolue_frequency"] = input_frequency_dict[
                parent
            ].get("absolue_frequency", 0) + input_frequency_dict.get(child, {}).get(
                "absolue_frequency", 0
            )
            input_frequency_dict[parent]["relative_frequency"] = input_frequency_dict[
                parent
            ].get("relative_frequency", 0) + input_frequency_dict.get(child, {}).get(
                "relative_frequency", 0
            )

    output_percentile_dict = load_output_percentile_mapping_dict(
        run_id, local_mlflow_dir
    )
    if output_percentile_dict is None:
        print("No output percentile file for run {} in local MlFlow dir".format(run_id))

    attention_weights = load_attention_weights(run_id, local_mlflow_dir)
    if attention_weights is None:
        print("No attention file for run {} in local MlFlow dir".format(run_id))
        attention_weights = {}

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
            feature_replacements=feature_replacements,
            attention_weights=attention_weights,
            cluster_threshold=cluster_threshold,
        )

    return prediction_output_df


def load_icd9_text() -> Dict[str, Dict[str, str]]:
    icd9_df = pd.read_csv("../data/icd9.csv")
    return (
        icd9_df[["child_name", "child_code"]]
        .drop_duplicates()
        .rename(columns={"child_name": "description", "child_code": "code",})
        .set_index("code")
        .to_dict("index")
    )

