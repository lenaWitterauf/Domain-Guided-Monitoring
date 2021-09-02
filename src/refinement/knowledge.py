from .config import RefinementConfig
import json
from typing import Dict, List, Tuple, Set
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm


class KnowledgeProcessor:
    def __init__(self, config: RefinementConfig):
        self.config = config

    def load_original_knowledge(self) -> Dict[str, Set[str]]:
        with open(self.config.original_file_knowledge) as knowledge_file:
            return json.load(knowledge_file)

    def _calculate_added_edges(
        self,
        attention_base: Dict[str, Dict[str, float]],
        attention_comp: Dict[str, Dict[str, float]],
    ) -> Set[Tuple[str, str]]:
        edges_base = set([(c, p) for c, ps in attention_base.items() for p in ps])
        edges_comp = set([(c, p) for c, ps in attention_comp.items() for p in ps])
        logging.debug(
            "Found %d edges in base, %d in comp", len(edges_base), len(edges_comp)
        )
        return edges_comp - edges_base

    def _calculate_refinement_metric(
        self, input_feature: str, comparison_df: pd.DataFrame
    ) -> float:
        # refinement_metric > 0 -> comparison is better than base
        relevant_df = comparison_df[
            comparison_df["inputs"].apply(lambda x: input_feature + "," in x)
        ].copy()
        if len(relevant_df) == 0:
            return -1
        if self.config.refinement_metric_maxrank > 0:
            relevant_df["output_rank_base"] = relevant_df["output_rank_base"].apply(
                lambda x: min(x, self.config.refinement_metric_maxrank)
            )
            relevant_df["output_rank_comp"] = relevant_df["output_rank_comp"].apply(
                lambda x: min(x, self.config.refinement_metric_maxrank)
            )

        if "outlier_score" in self.config.refinement_metric:
            outlier_scores = (
                relevant_df[["output_rank_base", "output_rank_comp"]]
                .apply(lambda x: (int(x[0]) - int(x[1])) / np.sqrt(2), axis=1)
                .to_list()
            )
            if "median" in self.config.refinement_metric:
                return np.median(outlier_scores)
            elif "mean" in self.config.refinement_metric:
                return np.mean(outlier_scores)
        elif "accuracy" in self.config.refinement_metric:
            accuracy_ats = [
                int(s) for s in self.config.refinement_metric.split("_") if s.isdigit()
            ]
            accuracy_at = accuracy_ats[0] if len(accuracy_ats) > 0 else 1
            accuracy_base = len(
                relevant_df[relevant_df["output_rank_base"] < accuracy_at]
            ) / len(relevant_df)
            accuracy_comp = len(
                relevant_df[relevant_df["output_rank_comp"] < accuracy_at]
            ) / len(relevant_df)
            return accuracy_comp - accuracy_base

        logging.error("Unknown refinement metric: %s", self.config.refinement_metric)
        return -1

    def _calculate_edge_comparison(
        self,
        attention_base: Dict[str, Dict[str, float]],
        attention_comp: Dict[str, Dict[str, float]],
        train_frequency: Dict[str, Dict[str, float]],
        comparison_df: pd.DataFrame,
    ) -> pd.DataFrame:
        added_edges = self._calculate_added_edges(
            attention_base=attention_base, attention_comp=attention_comp
        )

        records = []
        for c, p in tqdm(added_edges):
            if c == p:
                continue

            relevant_df = comparison_df[
                comparison_df["inputs"].apply(lambda x: c + "," in x)
            ]
            if len(relevant_df) == 0:
                continue

            edge_weight = attention_comp.get(c, {}).get(p, -1)
            if float(edge_weight) < self.config.min_edge_weight:
                continue

            frequency = train_frequency.get(c, {}).get("absolue_frequency", 0.0)
            if frequency > self.config.max_train_examples:
                continue

            records.append(
                {
                    "child": c,
                    "parent": p,
                    "refinement_metric": self._calculate_refinement_metric(
                        c, relevant_df
                    ),
                }
            )
        return pd.DataFrame.from_records(
            records, columns=["child", "parent", "refinement_metric"]
        )

    def load_refined_knowledge(
        self, refinement_run_id: str, reference_run_id: str
    ) -> Dict[str, List[str]]:
        attention_base = self._load_attention_weights(reference_run_id)
        attention_comp = self._load_attention_weights(refinement_run_id)
        train_frequency = self._load_input_frequency_dict(refinement_run_id)
        comparison_df = self._load_comparison_df(
            run_id_base=reference_run_id, run_id_comp=refinement_run_id
        )

        edge_comparison_df = (
            self._calculate_edge_comparison(
                attention_base=attention_base,
                attention_comp=attention_comp,
                train_frequency=train_frequency,
                comparison_df=comparison_df,
            )
            .sort_values(by="refinement_metric", ascending=True)
            .head(n=self.config.max_edges_to_remove)
        )
        refined_knowledge = {c: [c] for c in attention_comp}
        for child, parents in attention_comp.items():
            for parent in parents:
                if (
                    len(
                        edge_comparison_df[
                            (edge_comparison_df["child"] == child)
                            & (edge_comparison_df["parent"] == parent)
                        ]
                    )
                    > 0
                ):
                    continue

                refined_knowledge[child].append(parent)

        return refined_knowledge

    def _load_attention_weights(self, run_id):
        attention_path = Path(
            self.config.mlflow_dir
            + "{run_id}/artifacts/attention.json".format(run_id=run_id)
        )
        if not attention_path.exists():
            logging.debug(
                "No attention file for run {} in local MlFlow dir".format(run_id)
            )
            return {}

        with open(attention_path) as attention_file:
            return json.load(attention_file)["attention_weights"]

    def _get_best_rank_of(self, output: str, predictions_str: str) -> int:
        predictions = ast.literal_eval(predictions_str)
        return len([x for x in predictions if predictions[x] > predictions[output]])

    def _convert_prediction_df(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        prediction_df["input_converted"] = prediction_df["input"].apply(
            lambda x: " -> ".join(
                [
                    ", ".join([str(val) for val in sorted(v)])
                    for (_, v) in sorted(
                        ast.literal_eval(x).items(), key=lambda y: y[0]
                    )
                ]
            )
        )
        prediction_df["inputs"] = prediction_df["input"].apply(
            lambda x: ",".join(
                sorted(
                    set(
                        [
                            x
                            for xs in [
                                [str(val) for val in sorted(v)]
                                for (_, v) in sorted(
                                    ast.literal_eval(x).items(), key=lambda y: y[0]
                                )
                            ]
                            for x in xs
                        ]
                    )
                )
            )
            + ","
        )
        prediction_df["output"] = prediction_df["output"].apply(
            lambda x: ast.literal_eval(x)
        )
        prediction_df = prediction_df.explode("output")
        prediction_df["output_rank"] = prediction_df[["output", "predictions"]].apply(
            lambda x: self._get_best_rank_of(x[0], x[1]), axis=1
        )
        return prediction_df

    def _load_prediction_df(self, run_id) -> pd.DataFrame:
        run_prediction_output_path = Path(
            self.config.mlflow_dir
            + "{run_id}/artifacts/prediction_output.csv".format(run_id=run_id)
        )
        if not run_prediction_output_path.exists():
            logging.debug(
                "No prediction output file for run {} in local MlFlow dir".format(
                    run_id
                )
            )
            return pd.DataFrame()

        prediction_output_df = pd.read_csv(run_prediction_output_path)
        return self._convert_prediction_df(prediction_output_df)

    def _load_input_frequency_dict(self, run_id) -> Dict[str, Dict[str, float]]:
        run_frequency_path = Path(
            self.config.mlflow_dir
            + "{run_id}/artifacts/train_frequency.csv".format(run_id=run_id)
        )
        if not run_frequency_path.exists():
            logging.debug("No frequency file for run {} in MlFlow dir".format(run_id))
            return {}

        input_frequency_df = pd.read_csv(run_frequency_path).set_index("feature")
        input_frequency_df["relative_frequency"] = input_frequency_df[
            "absolue_frequency"
        ] / sum(input_frequency_df["absolue_frequency"])
        return input_frequency_df.to_dict("index")

    def _load_comparison_df(
        self, run_id_base, run_id_comp, suffix_base="_base", suffix_comp="_comp"
    ) -> pd.DataFrame:
        prediction_df_base = self._load_prediction_df(run_id_base).drop_duplicates(
            subset=["input_converted", "output"]
        )
        prediction_df_comp = self._load_prediction_df(run_id_comp).drop_duplicates(
            subset=["input_converted", "output"]
        )
        if prediction_df_base is None or prediction_df_comp is None:
            logging.error(
                "Unable to load prediction_dfs for runs {} and {}".format(
                    run_id_base, run_id_comp
                )
            )
            return pd.DataFrame()

        comparison_df = pd.merge(
            prediction_df_base,
            prediction_df_comp,
            on=["input_converted", "inputs", "output"],
            suffixes=(suffix_base, suffix_comp),
        )
        return comparison_df
