import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Any

from loading import (
    load_input_frequency_dict,
    load_prediction_df,
)


def _nullsafe_accuracy_for(top: pd.DataFrame, bottom: pd.DataFrame) -> float:
    if len(bottom) == 0:
        return -1

    return len(top) / len(bottom)


def _nullsafe_weighted_accuracy_for(top: pd.DataFrame, bottom: pd.DataFrame) -> float:
    if len(bottom) == 0:
        return -1

    return sum(top["percentile_weight"]) / sum(bottom["percentile_weight"])


def calculate_accuracy_at_k(
    relevant_run_ids: Set[str],
    local_mlflow_dir: str,
    k: int,
    num_percentiles: int = 10,
    percentile_names: List[str] = ["output_frequency_percentile"],
    num_input_percentiles: int = 10,
) -> pd.DataFrame:
    data_records = []
    for run_id in tqdm(relevant_run_ids, desc="Calculating accuracy@k per percentile"):
        prediction_df = load_prediction_df(
            run_id=run_id,
            local_mlflow_dir=local_mlflow_dir,
            num_percentiles=num_percentiles,
            convert_df=True,
        )
        if prediction_df is None:
            print("Unable to load prediction_df for run_id {}".format(run_id))
            continue

        prediction_df_topk = prediction_df[prediction_df["output_rank_noties"] < k]
        for name in percentile_names:
            data_records.append(
                {
                    "run_id": run_id,
                    "type": name,
                    "percentile": -1,
                    "accuracy": _nullsafe_accuracy_for(
                        prediction_df_topk, prediction_df
                    ),
                }
            )
            for percentile in range(num_percentiles):
                prediction_df_percentile = prediction_df[
                    prediction_df[name] == percentile
                ]
                if len(prediction_df_percentile) == 0:
                    continue
                prediction_df_percentile_topk = prediction_df_percentile[
                    prediction_df_percentile["output_rank_noties"] < k
                ]
                data_records.append(
                    {
                        "run_id": run_id,
                        "type": name,
                        "percentile": percentile,
                        "accuracy": _nullsafe_accuracy_for(
                            prediction_df_percentile_topk, prediction_df_percentile
                        ),
                    }
                )

        data_records.extend(
            calculate_accuracy_at_k_for_input_percentiles(
                run_id,
                prediction_df,
                k,
                num_percentiles=num_input_percentiles,
                local_mlflow_dir=local_mlflow_dir,
            )
        )

    return pd.DataFrame.from_records(data_records)


def load_input_percentiles(
    frequencies: Dict[str, Dict[str, float]], num_percentiles: int
) -> Dict[int, Set[str]]:
    sorted_features = [
        str(x[0])
        for x in sorted(frequencies.items(), key=lambda x: x[1]["relative_frequency"])
    ]
    bucket_size = len(sorted_features) / num_percentiles
    return {
        i: set(sorted_features[int(i * bucket_size) : int((i + 1) * bucket_size)])
        for i in range(num_percentiles)
    }


def calculate_accuracy_at_k_for_input_percentiles(
    run_id: str,
    prediction_df: pd.DataFrame,
    k: int,
    local_mlflow_dir: str,
    num_percentiles: int = 10,
) -> List[Dict[str, Any]]:
    data_records = []
    frequencies = load_input_frequency_dict(
        run_id=run_id, local_mlflow_dir=local_mlflow_dir
    )
    if prediction_df is None or frequencies is None:
        print(
            "Unable to load prediction_df or frequencies for run_id {}".format(run_id)
        )
        return []

    input_percentiles = load_input_percentiles(frequencies, num_percentiles)
    prediction_df_topk = prediction_df[prediction_df["output_rank_noties"] < k]
    data_records.append(
        {
            "run_id": run_id,
            "type": "input_percentile",
            "percentile": -1,
            "accuracy": _nullsafe_accuracy_for(prediction_df_topk, prediction_df),
        }
    )
    data_records.append(
        {
            "run_id": run_id,
            "type": "input_weighted_percentile",
            "percentile": -1,
            "accuracy": _nullsafe_accuracy_for(prediction_df_topk, prediction_df),
        }
    )
    for percentile in range(num_percentiles):
        prediction_df["percentile_weight"] = prediction_df["inputs"].apply(
            lambda x: len(
                [
                    i
                    for i in str(x).split(",")
                    if i.strip() in input_percentiles[percentile]
                ]
            )
        )
        prediction_df_percentile = prediction_df[prediction_df["percentile_weight"] > 0]
        prediction_df_percentile_topk = prediction_df_percentile[
            prediction_df_percentile["output_rank_noties"] < k
        ]
        data_records.append(
            {
                "run_id": run_id,
                "type": "input_percentile",
                "percentile": percentile,
                "accuracy": _nullsafe_accuracy_for(
                    prediction_df_percentile_topk, prediction_df_percentile
                ),
            }
        )
        data_records.append(
            {
                "run_id": run_id,
                "type": "input_weighted_percentile",
                "percentile": percentile,
                "accuracy": _nullsafe_weighted_accuracy_for(
                    prediction_df_percentile_topk, prediction_df_percentile
                ),
            }
        )

    return data_records


def _highlight_max(data: pd.DataFrame, color="yellow") -> pd.DataFrame:
    attr = "background-color: {}".format(color)
    data = data.astype(float)
    max_data = data.max(axis=1, level=0)
    is_max = data.eq(max_data, axis=1)
    return pd.DataFrame(
        np.where(is_max, attr, ""), index=data.index, columns=data.columns
    )


def plot_accuracies_per_percentiles(
    relevant_run_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    comparison_column: str = "data_params_ModelConfigbase_hidden_embeddings_trainable",
    comparison_column_order: List[str] = ["False", "True"],
    share_y: bool = False,
    show_plot: bool = False,
    show_improvements: bool = True,
):
    grouped_df = (
        pd.merge(relevant_run_df, accuracy_df, left_on="info_run_id", right_on="run_id")
        .groupby(
            [
                "data_tags_model_type",
                comparison_column,
                "info_run_id",
                "type",
                "percentile",
            ],
            as_index=False,
        )
        .agg({"accuracy": max,})
    )

    # print("----- MEAN -----")
    mean_grouped_df = (
        grouped_df.groupby(
            [
                comparison_column,
                "type",
                "percentile",
                "data_tags_model_type",
            ],
            as_index=True,
        )
        .agg({"accuracy": np.mean,})
        .reset_index(drop=False)
        .rename(
            columns={
                "data_tags_model_type": "model_type",
            }
        )
        .pivot(
            index=["type", "percentile"],
            columns=[comparison_column, "model_type"],
            values="accuracy",
        )
    )
    # display(mean_grouped_df.style.apply(_highlight_max, axis=None))

    # print("----- MEDIAN -----")
    median_grouped_df = (
        grouped_df.groupby(
            [
                comparison_column,
                "type",
                "percentile",
                "data_tags_model_type",
            ],
            as_index=True,
        )
        .agg({"accuracy": np.median,})
        .reset_index(drop=False)
        .rename(
            columns={
                "data_tags_model_type": "model_type",
            }
        )
        .pivot(
            index=["type", "percentile"],
            columns=[comparison_column, "model_type"],
            values="accuracy",
        )
    )
    # display(median_grouped_df.style.apply(_highlight_max, axis=None))

    if show_plot:
        g = sns.relplot(
            data=grouped_df,
            x="percentile",
            y="accuracy",
            row="type",
            hue="data_tags_model_type",
            col=comparison_column,
            kind="line",
            palette=None,
        )
        g.set_titles("Type: {row_name}, Comparison: {col_name}").set_axis_labels(
            "", "accuracy"
        )
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        plt.show()

    if show_improvements:
        comparison_columns = set([x[0] for x in mean_grouped_df.columns])
        for value in comparison_columns:
            mean_grouped_df[value] = mean_grouped_df.loc[:, (value)].sub(
                mean_grouped_df[(value, "simple")], axis=0
            )
            mean_grouped_df[value] = mean_grouped_df.loc[:, (value)].sub(
                mean_grouped_df[(value, "simple")], axis=0
            )
        comparison_dfs = []
        for value in comparison_columns:
            mean_grouped_df_value = mean_grouped_df[value].copy()
            mean_grouped_df_value[comparison_column] = value
            comparison_dfs.append(mean_grouped_df_value)
        
        mean_grouped_df = pd.concat(comparison_dfs).reset_index()
        mean_grouped_df = mean_grouped_df.melt(
            id_vars=["type", "percentile", comparison_column],
            value_vars=["causal", "gram", "text"],
            value_name="mean_accuracy_diff",
        )

        g = sns.relplot(
            data=mean_grouped_df[mean_grouped_df["percentile"] > -1],
            x="percentile",
            y="mean_accuracy_diff",
            row="type",
            hue="model_type",
            col=comparison_column,
            kind="line",
            palette=None,
            facet_kws={"sharey": share_y, "sharex": True},
            col_order=comparison_column_order
        )
        g.set_titles("Type: {row_name}, Comparison: {col_name}").set_axis_labels(
            "", "mean_accuracy_diff"
        )
        g.map(plt.axhline, y=0, color=".7", dashes=(2, 1), zorder=0)
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        plt.show()

        comparison_columns = set([x[0] for x in median_grouped_df.columns])
        for value in comparison_columns:
            median_grouped_df[value] = median_grouped_df.loc[:, (value)].sub(
                median_grouped_df[(value, "simple")], axis=0
            )
            median_grouped_df[value] = median_grouped_df.loc[:, (value)].sub(
                median_grouped_df[(value, "simple")], axis=0
            )
        comparison_dfs = []
        for value in comparison_columns:
            median_grouped_df_value = median_grouped_df[value].copy()
            median_grouped_df_value[comparison_column] = value
            comparison_dfs.append(median_grouped_df_value)
        
        median_grouped_df = pd.concat(comparison_dfs).reset_index()
        median_grouped_df = median_grouped_df.melt(
            id_vars=["type", "percentile", comparison_column],
            value_vars=["causal", "gram", "text"],
            value_name="median_accuracy_diff",
        )

        g = sns.relplot(
            data=median_grouped_df[median_grouped_df["percentile"] > -1],
            x="percentile",
            y="median_accuracy_diff",
            row="type",
            hue="model_type",
            col=comparison_column,
            kind="line",
            palette=None,
            facet_kws={"sharey": share_y, "sharex": True},
        )
        g.set_titles("Type: {row_name}, Comparison: {col_name}").set_axis_labels(
            "", "median_accuracy_diff"
        )
        g.map(plt.axhline, y=0, color=".7", dashes=(2, 1), zorder=0)
        for ax in g.axes.flatten():
            ax.tick_params(labelbottom=True)
        plt.show()


def calculate_accuracies_per_percentiles(
    relevant_run_df: pd.DataFrame,
    k: int,
    percentile_names: List[str],
    num_percentiles: int,
    num_input_percentiles: int,
    local_mlflow_dir: str,
):
    accuracy_df = calculate_accuracy_at_k(
        relevant_run_ids=set(relevant_run_df["info_run_id"]),
        k=k,
        percentile_names=percentile_names,
        num_percentiles=num_percentiles,
        num_input_percentiles=num_input_percentiles,
        local_mlflow_dir=local_mlflow_dir,
    )

    return accuracy_df


def calculate_accuracy_at_k_per_input(run_id: str, k: int, local_mlflow_dir: str) -> List[Dict[str, Any]]:
    prediction_df = load_prediction_df(run_id=run_id, local_mlflow_dir=local_mlflow_dir)
    frequencies = load_input_frequency_dict(
        run_id=run_id, local_mlflow_dir=local_mlflow_dir
    )
    if prediction_df is None or frequencies is None:
        print(
            "Unable to load prediction_df or frequencies for run_id {}".format(run_id)
        )
        return []

    prediction_df["input_list"] = prediction_df["inputs"].apply(
        lambda x: [i.strip() for i in x.split(",") if len(i.strip()) > 0]
    )
    prediction_df = prediction_df.explode("input_list")
    return [
        {
            "run_id": run_id,
            "input": input,
            "rel_frequency": frequencies[input]["relative_frequency"],
            "accuracy_k": _nullsafe_accuracy_for(
                prediction_df[
                    (prediction_df["input_list"] == input)
                    & (prediction_df["output_rank_noties"] < k)
                ],
                prediction_df[prediction_df["input_list"] == input],
            ),
        }
        for input in tqdm(frequencies, desc="Calculating accuracy@k per input feature")
    ]


def calculate_accuracy_at_k_per_inputs(
    run_ids: Set[str], k: int, local_mlflow_dir: str
) -> pd.DataFrame:
    records = []
    for run_id in run_ids:
        records.extend(
            calculate_accuracy_at_k_per_input(
                run_id=run_id, k=k, local_mlflow_dir=local_mlflow_dir
            )
        )

    return pd.DataFrame.from_records(records)


def calculate_accuracy_per_inputs_comparison(rel_df: pd.DataFrame, k: int, local_mlflow_dir: str):
    accuracy_df = calculate_accuracy_at_k_per_inputs(
        run_ids=set(rel_df["info_run_id"]), k=k, local_mlflow_dir=local_mlflow_dir
    )
    accuracy_df = accuracy_df[accuracy_df["accuracy_k"] > -1]
    return pd.pivot_table(
        pd.merge(
            accuracy_df, rel_df, how="inner", left_on="run_id", right_on="info_run_id"
        ),
        values=["accuracy_k", "rel_frequency"],
        columns=[
            "data_tags_model_type",
            "data_params_ModelConfigbase_hidden_embeddings_trainable",
        ],
        index=["input"],
        aggfunc=[np.mean, np.median, np.min, np.max],
    )
