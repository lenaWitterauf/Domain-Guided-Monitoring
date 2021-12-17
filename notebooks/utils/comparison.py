import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import plotly.express as px

from loading import (
    load_attention_weights,
    load_input_frequency_dict,
    load_prediction_df,
)


def load_comparison_df_from_pkl(
    run_id_1: str, run_id_2: str, pickle_dir: str, 
) -> Optional[pd.DataFrame]:
    if pickle_dir is None or len(pickle_dir) == 0:
        return None
    comparison_pkl_path = Path(
        "{pickle_dir}{run_id_1}_{run_id_2}.pkl".format(
            pickle_dir=pickle_dir, run_id_1=run_id_1, run_id_2=run_id_2
        )
    )
    if not comparison_pkl_path.exists():
        return None

    return pd.read_pickle(comparison_pkl_path)


def load_comparison_df_from_predictions(
    run_id_1: str,
    suffix_1: str,
    run_id_2: str,
    suffix_2: str,
    local_mlflow_dir: str,
    num_percentiles: int,
    feature_replacements: Dict[str, str] = {},
) -> Optional[pd.DataFrame]:
    prediction_df_1 = load_prediction_df(
        run_id_1, num_percentiles=num_percentiles, local_mlflow_dir=local_mlflow_dir, feature_replacements=feature_replacements,
    )
    prediction_df_2 = load_prediction_df(
        run_id_2, num_percentiles=num_percentiles, local_mlflow_dir=local_mlflow_dir, feature_replacements=feature_replacements,
    )
    if prediction_df_1 is None or prediction_df_2 is None:
        print("Invalid run ids!")
        return None

    if len(prediction_df_1) == len(prediction_df_2):
        comparison_df = pd.merge(
            prediction_df_1.sort_values(by=["input_converted", "output"])
            .reset_index(drop=True)
            .reset_index(drop=False),
            prediction_df_2.sort_values(by=["input_converted", "output"])
            .reset_index(drop=True)
            .reset_index(drop=False),
            on=["index", "input_converted", "inputs", "output"],
            suffixes=(suffix_1, suffix_2),
        )
    else:
        comparison_df = pd.merge(
            prediction_df_1.drop_duplicates(subset=["input_converted", "output"]),
            prediction_df_2.drop_duplicates(subset=["input_converted", "output"]),
            on=["input_converted", "output"],
            suffixes=(suffix_1, suffix_2),
        )

    print(len(prediction_df_1), len(prediction_df_2), len(comparison_df))
    if len(comparison_df) == 0:
        print("Comparison df for {} {} is empty!!!".format(run_id_1, run_id_2))
        return comparison_df
    print(len(comparison_df[comparison_df["output_rank_noties" + suffix_1] < 20]) / len(comparison_df))
    print(len(comparison_df[comparison_df["output_rank_noties" + suffix_2] < 20]) / len(comparison_df))

    for column_name in [
        "output_frequency_percentile",
        "avg_input_frequencies_percentile",
        "median_input_frequencies_percentile",
        "min_input_frequencies_percentile",
        "p10_input_frequencies_percentile",
        "avg_input_frequencies",
        "median_input_frequencies",
        "min_input_frequencies",
        "p10_input_frequencies",
        "unknown_inputs",
        "unknown_inputs_percentile",
    ]:
        if (column_name + suffix_1) in set(comparison_df.columns):
            comparison_df[column_name] = comparison_df[
                [column_name + suffix_1, column_name + suffix_2]
            ].apply(lambda x: int((x[0] + x[1]) / 2), axis=1)
    comparison_df["input_length"] = comparison_df["input_converted"].apply(
        lambda x: x.count("->") + 1
    )
    return comparison_df


def load_comparison_df(
    run_id_1: str,
    suffix_1: str,
    run_id_2: str,
    suffix_2: str,
    local_mlflow_dir: str,
    num_percentiles: int = 10,
    pickle_dir: Optional[str] = None,
    feature_replacements: Dict[str, str] = {},
) -> Optional[pd.DataFrame]:
    comparison_df: Optional[pd.DataFrame] = None
    if pickle_dir is not None:
        comparison_df = load_comparison_df_from_pkl(run_id_1, run_id_2, pickle_dir)

    if comparison_df is None:
        comparison_df = load_comparison_df_from_predictions(
            run_id_1=run_id_1,
            suffix_1=suffix_1,
            run_id_2=run_id_2,
            suffix_2=suffix_2,
            local_mlflow_dir=local_mlflow_dir,
            num_percentiles=num_percentiles,
            feature_replacements=feature_replacements
        )

    if comparison_df is None:
        return None

    if pickle_dir is not None:
        comparison_pkl_path = Path(
            "{pickle_dir}{run_id_1}_{run_id_2}.pkl".format(
                pickle_dir=pickle_dir, run_id_1=run_id_1, run_id_2=run_id_2
            )
        )
        pd.to_pickle(comparison_df, comparison_pkl_path)

    comparison_df["outlier_distance"] = comparison_df[
        ["output_rank_noties" + suffix_1, "output_rank_noties" + suffix_2]
    ].apply(lambda x: (int(x[0]) - int(x[1])) / np.sqrt(2), axis=1)
    comparison_df["outlier_distance_abs"] = comparison_df[
        ["output_rank_noties" + suffix_1, "output_rank_noties" + suffix_2]
    ].apply(lambda x: abs(int(x[0]) - int(x[1])) / np.sqrt(2), axis=1)
    if "index" in comparison_df:
        comparison_df = comparison_df.drop(columns=["index"])
    if "level_0" in comparison_df:
        comparison_df = comparison_df.drop(columns=["level_0"])
    return (
        comparison_df.sort_values(by="outlier_distance", ascending=False)
        .reset_index(drop=True)
        .reset_index(drop=False)
    )


class Comparison:
    def __init__(
        self,
        run_id_1: str,
        suffix_1: Optional[str],
        run_id_2: str,
        suffix_2: Optional[str],
        local_mlflow_dir: str,
        num_percentiles: int,
        comparison_pkl_dir: Optional[str] = None,
        feature_replacements: Dict[str, str] = {},
    ):
        self.run_id_1 = run_id_1
        self.suffix_1 = run_id_1 if suffix_1 is None else suffix_1
        self.run_id_2 = run_id_2
        self.suffix_2 = run_id_2 if suffix_2 is None else suffix_2

        self.comparison_df = load_comparison_df(
            run_id_1=self.run_id_1,
            suffix_1=self.suffix_1,
            run_id_2=self.run_id_2,
            suffix_2=self.suffix_2,
            local_mlflow_dir=local_mlflow_dir,
            num_percentiles=num_percentiles,
            pickle_dir=comparison_pkl_dir,
            feature_replacements=feature_replacements,
        )
        self.input_frequencies: Dict[str, Optional[Dict[str, Dict[str, float]]]] = {
            self.suffix_1: load_input_frequency_dict(
                run_id=self.run_id_1, local_mlflow_dir=local_mlflow_dir
            ),
            self.suffix_2: load_input_frequency_dict(
                run_id=self.run_id_2, local_mlflow_dir=local_mlflow_dir
            ),
        }
        self.attention_weights: Dict[str, Optional[Dict[str, Dict[str, float]]]] = {
            self.suffix_1: load_attention_weights(
                run_id=self.run_id_1, local_mlflow_dir=local_mlflow_dir
            ),
            self.suffix_2: load_attention_weights(
                run_id=self.run_id_2, local_mlflow_dir=local_mlflow_dir
            ),
        }

    def attention_weights_for(self, suffix) -> Dict[str, Dict[str, float]]:
        attention_weights = self.attention_weights.get(suffix, {})
        if attention_weights is None:
            attention_weights = {}

        return attention_weights

    def input_frequencies_for(self, suffix) -> Dict[str, Dict[str, float]]:
        input_frequencies = self.input_frequencies.get(suffix, {})
        if input_frequencies is None:
            input_frequencies = {}

        return input_frequencies

    def min_comparison_index(self) -> int:
        if self.comparison_df is None:
            return -1

        return min(self.comparison_df.index)

    def max_comparison_index(self) -> int:
        if self.comparison_df is None:
            return -1

        return max(self.comparison_df.index)



def plot_comparison(comparison: Comparison, plot_column="avg_input_frequencies_percentile", color="outlier_distance", hover_data=[]):
    if comparison.comparison_df is None or len(comparison.comparison_df) == 0:
        print("Comparison has invalid comparison_df!")
        return None

    fig = px.scatter(
        comparison.comparison_df,
        x=plot_column + comparison.suffix_1,
        y=plot_column + comparison.suffix_2,
        color=color,
        hover_data=[
            "output",
            "outlier_distance",
            "index",
            "input_length",
            "unknown_inputs",
            "min_input_frequencies",
            "p10_input_frequencies",
        ] + hover_data,
        width=750,
        height=500,
    )
    fig.show()

def plot_rank_comparison(comparison: Comparison, color="unknown_inputs", hover_data=[]):
    if comparison.comparison_df is None or len(comparison.comparison_df) == 0:
        print("Comparison has invalid comparison_df!")
        return None

    fig = px.scatter(
        comparison.comparison_df,
        x="output_rank_noties" + comparison.suffix_1,
        y="output_rank_noties" + comparison.suffix_2,
        color=color,
        hover_data=[
            "output",
            "outlier_distance",
            "index",
            "input_length",
            "unknown_inputs",
            "min_input_frequencies",
            "p10_input_frequencies",
        ] + hover_data,
        width=750,
        height=500,
    )
    fig.show()


def plot_outlier_distances(comparison: Comparison):
    if comparison.comparison_df is None or len(comparison.comparison_df) == 0:
        print("Comparison has invalid comparison_df!")
        return None

    fig_outputp = px.scatter(
        comparison.comparison_df,
        x="outlier_distance",
        y="output_frequency_percentile",
        color="avg_input_frequencies_percentile",
        hover_data=[
            "index",
            "avg_input_frequencies_percentile",
            "output_frequency_percentile",
            "output_rank_noties" + comparison.suffix_1,
            "output_rank_noties" + comparison.suffix_2,
        ],
        marginal_x="box",
        width=1000,
        height=500,
    )
    fig_outputp.show()

    fig_inputp = px.scatter(
        comparison.comparison_df,
        x="outlier_distance",
        y="avg_input_frequencies_percentile",
        color="output_frequency_percentile",
        hover_data=[
            "index",
            "avg_input_frequencies_percentile",
            "output_frequency_percentile",
            "output_rank_noties" + comparison.suffix_1,
            "output_rank_noties" + comparison.suffix_2,
        ],
        marginal_x="box",
        width=1000,
        height=500,
    )
    fig_inputp.show()

    fig_input = px.scatter(
        comparison.comparison_df,
        x="outlier_distance",
        y="unknown_inputs",
        color="output_frequency_percentile",
        hover_data=[
            "index",
            "avg_input_frequencies_percentile",
            "output_frequency_percentile",
            "output_rank_noties" + comparison.suffix_1,
            "output_rank_noties" + comparison.suffix_2,
        ],
        marginal_x="box",
        width=1000,
        height=500,
    )
    fig_input.show()


def analyse_sequence_at_index(
    comparison: Comparison, index: int, descriptions: Dict[str, Dict[str, str]] = {}
):
    if comparison.comparison_df is None:
        print("Comparison has invalid comparison_df!")
        return

    sequence_data = comparison.comparison_df.loc[index][
        [
            "index",
            "input_converted",
            "inputs",
            "output",
            "outlier_distance",
            "avg_input_frequencies_percentile",
            "output_frequency_percentile",
            "output_rank_noties" + comparison.suffix_1,
            "output_rank_noties" + comparison.suffix_2,
            "original_inputs" + comparison.suffix_1,
            "original_inputs" + comparison.suffix_2,
        ]
    ].to_dict()
    print(sequence_data)

    unknown_inputs = [
        x.strip()
        for x in sequence_data["original_inputs" + comparison.suffix_2].split(",")
        if comparison.input_frequencies_for(comparison.suffix_2)
        .get(x.strip(), {})
        .get("absolute_frequency", 0.0)
        == 0.0 and len(x.strip()) > 0
    ]
    for unknown_input in unknown_inputs:
        print("__________")
        print(
            "Analysing attention weights for unknown input feature {} ({})".format(
                unknown_input, descriptions.get(unknown_input, {}),
            )
        )
        attention_weights = comparison.attention_weights_for(comparison.suffix_2).get(
            unknown_input, {}
        )
        print(attention_weights)

        most_important_attention_nodes = [
            k for k, v in attention_weights.items() if float(v) > 0.1
        ]
        for attention_node in most_important_attention_nodes:
            print("____")
            attention_description = descriptions.get(attention_node, {}).get(
                "description", attention_node
            )
            print(
                "Important attention node for unknown input {}: {} with attention weight {} ({})".format(
                    unknown_input,
                    attention_node,
                    attention_weights.get(attention_node),
                    attention_description,
                )
            )
            other_attention_features = [
                (
                    k,
                    v[attention_node],
                    comparison.input_frequencies_for(comparison.suffix_2)
                    .get(k, {})
                    .get("absolute_frequency", 0),
                )
                for k, v in comparison.attention_weights_for(
                    comparison.suffix_2
                ).items()
                if attention_node in v
            ]
            print(
                "Node {} has {} input feature connections ({}  of which are unknown features) with avg weights {}".format(
                    attention_node,
                    len(other_attention_features),
                    len([x for x in other_attention_features if x[2] == 0]),
                    sum([float(x[1]) for x in other_attention_features])
                    / len(other_attention_features),
                )
            )
            print("____")


def analyse_best_worst_sequences(
    comparison: Comparison, num_best_sequences: int = 1, num_worst_sequences: int = 1, descriptions: Dict[str, Dict[str, str]] = {},
):
    print(
        "Analysing sequences with",
        comparison.suffix_2,
        "better than",
        comparison.suffix_1,
    )
    for i in range(num_best_sequences):
        print("==========")
        analyse_sequence_at_index(
            comparison, index=comparison.min_comparison_index() + i, descriptions=descriptions
        )

    print("========================================")

    print(
        "Analysing sequences with",
        comparison.suffix_1,
        "better than",
        comparison.suffix_2,
    )
    for i in range(num_worst_sequences):
        print("==========")
        analyse_sequence_at_index(
            comparison, index=comparison.max_comparison_index() - i, descriptions=descriptions
        )

