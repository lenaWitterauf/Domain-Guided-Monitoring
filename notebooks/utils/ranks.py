import pandas as pd
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

from comparison import load_comparison_df


def calculate_comparison_run_ids(
    relevant_dfs: List[pd.DataFrame],
) -> List[Dict[str, str]]:
    grouped_df = (
        pd.concat(relevant_dfs)
        .groupby(
            by=[
                "data_tags_sequence_type",
                "data_params_ModelConfigbase_hidden_embeddings_trainable",
                "data_params_ModelConfigfeature_embedding_initializer_seed",
                "data_tags_model_type",
            ]
        )
        .agg({"info_run_id": list,})
    )
    comparison_runs: List[Dict[str, str]] = []
    for sequence_type in set([x[0] for x in grouped_df.index]):
        sequence_df = grouped_df.loc[sequence_type]
        for trainable_type in set([x[0] for x in sequence_df.index]):
            trainable_df = sequence_df.loc[trainable_type]
            for seed_value in set([x[0] for x in trainable_df.index]):
                seed_df = trainable_df.loc[seed_value]
                model_types = set(seed_df.index)
                for model_type_1 in model_types - set(["simple"]):
                    model_1_run_ids = set(seed_df.at[model_type_1, "info_run_id"])
                    for model_type_2 in ["simple"]:  # model_types:
                        model_2_run_ids = set(seed_df.at[model_type_2, "info_run_id"])
                        for model_1_run_id in model_1_run_ids:
                            for model_2_run_id in model_2_run_ids:
                                if (
                                    len(
                                        [
                                            x
                                            for x in comparison_runs
                                            if x["run_id_1"] == model_1_run_id
                                            and x["run_id_2"] == model_2_run_id
                                        ]
                                    )
                                    > 0
                                ):
                                    continue
                                comparison_runs.append(
                                    {
                                        "run_id_1": model_1_run_id,
                                        "run_id_2": model_2_run_id,
                                    }
                                )
    return comparison_runs


def calculate_rank_comparisons(
    relevant_dfs: List[pd.DataFrame],
    local_mlflow_dir: str,
    num_percentiles: int = 10,
) -> pd.DataFrame:
    comparison_run_ids = calculate_comparison_run_ids(relevant_dfs)
    comparisons: List[Dict[str, Any]] = []
    aggregations = [
        "mean",
        "sum",
        "median",
        "amin",
        "amax",
        "std",
    ]

    for comparison in tqdm(comparison_run_ids):
        if (
            len(
                [
                    x
                    for x in comparisons
                    if x["run_id_1"] == comparison["run_id_1"]
                    and x["run_id_2"] == comparison["run_id_2"]
                ]
            )
            > 0
        ):
            continue
        comparison_df = load_comparison_df(
            run_id_1=comparison["run_id_1"],
            suffix_1=comparison["run_id_1"],
            run_id_2=comparison["run_id_2"],
            suffix_2=comparison["run_id_2"],
            local_mlflow_dir=local_mlflow_dir,
            num_percentiles=num_percentiles,
            pickle_dir="comparison/",
        )
        if comparison_df is None or len(comparison_df) == 0:
            continue

        for metric in [
            "avg_input_frequencies_percentile",
            "median_input_frequencies_percentile",
            "min_input_frequencies_percentile",
            "p10_input_frequencies_percentile",
            "unknown_inputs_percentile",
            "output_frequency_percentile",
        ]:
            grouped_comparison_df = comparison_df.groupby(by=[metric]).agg(
                {
                    "outlier_distance": [
                        np.mean,
                        np.sum,
                        np.median,
                        np.min,
                        np.max,
                        np.std,
                    ],
                }
            )
            for percentile_value in set(grouped_comparison_df.index):
                for aggregation in aggregations:
                    comparisons.append(
                        {
                            "run_id_1": comparison["run_id_1"],
                            "run_id_2": comparison["run_id_2"],
                            "metric": metric,
                            "percentile": percentile_value,
                            "aggregation": aggregation,
                            "value": grouped_comparison_df.at[
                                percentile_value, ("outlier_distance", aggregation)
                            ],
                        }
                    )

    full_comparison_df = pd.DataFrame.from_records(comparisons)
    full_comparison_df = pd.merge(full_comparison_df, pd.concat(relevant_dfs), left_on="run_id_1", right_on="info_run_id", suffixes=("", "_1"))
    full_comparison_df = pd.merge(full_comparison_df, pd.concat(relevant_dfs), left_on="run_id_2", right_on="info_run_id", suffixes=("", "_2"))
    return full_comparison_df