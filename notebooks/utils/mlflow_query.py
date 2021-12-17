import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Set, Dict, Any, List, Optional


class MlflowHelper:
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        local_mlflow_dir_prefix: str = "../gsim01/mlruns/",
        experiment_name: str = "Domain Guided Monitoring",
        experiment_id: Optional[str] = "1",
        pkl_file: Optional[Path] = None,
    ):
        self.mlflow_client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment_id = experiment_id if experiment_id is not None else self.mlflow_client.get_experiment_by_name(experiment_name).experiment_id
        self.local_mlflow_dir = local_mlflow_dir_prefix + str(self.experiment_id) + "/"
        if pkl_file is not None and pkl_file.exists():
            self.run_df = pd.read_pickle("mlflow_run_df.pkl") 
            print("Initialized with", len(self.run_df), "MLFlow runs from pkl")
        else:
            self.run_df = pd.DataFrame(columns=["info_run_id"])
        self.metric_history_names: Set[str] = set()

    def query_valid_runs(self, 
            pkl_file: Optional[Path] = None, 
            valid_sequence_types: List[str] = ['mimic', 'huawei_logs'], 
            filter_string_suffix: Optional[str] = " and params.ModelConfigrnn_type = 'gru'"):
        for sequence_type in valid_sequence_types:
            filter_string = "tags.sequence_type = '" + sequence_type + "'"
            if filter_string_suffix is not None:
                filter_string = filter_string + filter_string_suffix
            
            self.query_runs(filter_string=filter_string)
            print("Queried", len(self.run_df), "runs from MLFlow for", sequence_type)
            
        if pkl_file is not None:
            self.run_df.to_pickle(pkl_file)

    def query_runs(self, filter_string: Optional[str] = None, pkl_file: Optional[Path] = None,):
        runs = self.mlflow_client.search_runs(
            experiment_ids=[self.experiment_id], max_results=10000, filter_string=filter_string,
        )
        for run in tqdm(runs, desc="Querying data per run..."):
            self._handle_run(run)

        if pkl_file is not None:
            self.run_df.to_pickle(pkl_file)

    def _handle_run(self, run):
        if (
            len(self.run_df) > 0
            and run.info.run_id in set(self.run_df["info_run_id"])
            and run.info.status == "FINISHED"
            and len(
                self.run_df[
                    (self.run_df["info_run_id"] == run.info.run_id)
                    & (self.run_df["info_status"] == run.info.status)
                ]
            )
            == 1
        ):
            return

        if not run.info.status == "FINISHED" and not run.info.run_id in set(
            self.run_df["info_run_id"]
        ):
            return
        
        run_dict = {
            (k + "_" + sk): v
            for k, sd in run.to_dictionary().items()
            for sk, v in sd.items()
        }
        final_run_dict = {
            (k + "_" + sk): v
            for k, sd in run_dict.items()
            if type(sd) == type(dict())
            for sk, v in sd.items()
        }
        final_run_dict.update(
            {k: v for k, v in run_dict.items() if not (type(v) == type(dict()))}
        )
        if (
            final_run_dict.get("data_tags_model_type", "") == "causal"
            and final_run_dict.get(
                "data_params_KnowledgeConfigadd_causality_prefix", "False"
            )
            == "True"
        ):
            final_run_dict["data_tags_model_type"] = "causal2"
        if (
            (final_run_dict.get("data_tags_model_type", "") == "causal"
            or final_run_dict.get("data_tags_model_type", "") == "causal2")
            and final_run_dict.get("data_tags_sequence_type", "") == "huawei_logs"
            and final_run_dict.get("data_params_HuaweiPreprocessorConfiglog_only_causality", "") == "True"
        ):
            final_run_dict["data_tags_model_type"] = final_run_dict["data_tags_model_type"] + "_logonly"
        if (
            final_run_dict.get("data_tags_model_type", "") == "text"
            and final_run_dict.get(
                "data_params_KnowledgeConfigbuild_text_hierarchy", "False"
            )
            == "True"
        ):
            final_run_dict["data_tags_model_type"] = "text_hierarchy"
        if (
            final_run_dict.get("data_tags_model_type", "") == "gram"
            and final_run_dict.get("data_tags_sequence_type", "") == "huawei_logs"
            and final_run_dict.get("data_params_KnowledgeConfigadd_causality_prefix")
            and final_run_dict.get(
                "data_params_HuaweiPreprocessorConfiguse_log_hierarchy", "False"
            )
            == "True"
        ):
            final_run_dict["data_tags_model_type"] = "gram_logs"

        self.run_df = self.run_df.append(
            final_run_dict, ignore_index=True
        ).drop_duplicates(subset=["info_run_id"], keep="last", ignore_index=True)
    
    
    def mimic_run_df(
        self, include_noise: bool = False, include_refinements: bool = False, 
        risk_prediction: bool = False,
        valid_x_columns: List[str]=["level_0"],
        valid_y_columns: List[str]=["level_3"],
    ) -> pd.DataFrame:
        mimic_run_df = self.run_df[
            (self.run_df["data_tags_sequence_type"] == "mimic")
            & (self.run_df["data_params_ModelConfigrnn_type"] == "gru")
            & (self.run_df["data_params_SequenceConfigtest_percentage"].fillna("").astype(str) == "0.2")
            & (self.run_df["data_params_ModelConfigbest_model_metric"] == "val_loss")
            & (self.run_df["info_status"] == "FINISHED")
            & (self.run_df["data_params_ModelConfigrnn_dim"] == "200")
            & (self.run_df["data_params_ModelConfigoptimizer"].fillna("adam") == "adam")
            & (self.run_df["data_params_ModelConfigdropout_rate"].fillna("0.0").astype(str) == "0.5")
            & (self.run_df["data_params_ModelConfigrnn_dropout"].fillna("0.0").astype(str) == "0.0")
            & (self.run_df["data_params_ModelConfigkernel_regularizer_scope"].fillna("[]") == "[]")
            & (self.run_df["data_params_SequenceConfigpredict_full_y_sequence_wide"].astype(str).fillna("") == "True")
            & (
                (
                    (self.run_df["data_params_SequenceConfigy_sequence_column_name"].astype(str) == "level_3")
                    & (self.run_df["data_params_ExperimentConfigbatch_size"].astype(str).fillna("") == "128")
                ) |
                (
                    (self.run_df["data_params_SequenceConfigy_sequence_column_name"].astype(str) == "level_2")
                    & (self.run_df["data_params_ExperimentConfigbatch_size"].astype(str).fillna("") == "16")
                )
            )
            & (self.run_df["data_params_MimicPreprocessorConfigreplace_keys"].fillna("[]") == "[]")
        ]

        if risk_prediction:
            mimic_run_df = mimic_run_df[
                (mimic_run_df["data_tags_task_type"] == "risk_prediction") &
                (mimic_run_df["data_params_ModelConfigfinal_activation_function"] == "sigmoid")
            ]
        else:
            mimic_run_df = mimic_run_df[
                (mimic_run_df["data_params_ModelConfigfinal_activation_function"] == "softmax")
                & (mimic_run_df["data_params_SequenceConfigflatten_y"] == "True")
            ]

        if len(valid_x_columns) > 0:
            mimic_run_df = mimic_run_df[
                mimic_run_df["data_params_SequenceConfigx_sequence_column_name"].apply(lambda x: x in valid_x_columns)
            ]
        if len(valid_y_columns) > 0:
            mimic_run_df = mimic_run_df[
                mimic_run_df["data_params_SequenceConfigy_sequence_column_name"].apply(lambda x: x in valid_y_columns)
            ]

        if not include_noise:
            mimic_run_df = mimic_run_df[
                (mimic_run_df["data_tags_noise_type"].fillna("").apply(len) == 0)
            ]
        if not include_refinements:
            mimic_run_df = mimic_run_df[
                (mimic_run_df["data_tags_refinement_type"].fillna("") == "")
            ]

        return mimic_run_df

    def huawei_run_df(
        self, include_noise: bool = False, include_refinements: bool = False,
        risk_prediction: bool = False,
        valid_x_columns: List[str]=["log_cluster_template", "fine_log_cluster_template"],
        valid_y_columns: List[str]=["attributes"],
        include_drain_hierarchy: bool=False,
    ) -> pd.DataFrame:
        huawei_run_df = self.run_df[
            (self.run_df["data_tags_sequence_type"] == "huawei_logs")
            & (self.run_df["data_params_ModelConfigrnn_type"] == "gru")
            & (self.run_df["data_params_SequenceConfigtest_percentage"].fillna("").astype(str) == "0.1")
            & (self.run_df["data_params_ModelConfigbest_model_metric"] == "val_loss")
            & (self.run_df["info_status"] == "FINISHED")
            & (self.run_df["data_params_ModelConfigrnn_dim"] == "200")
            & (self.run_df["data_params_ModelConfigoptimizer"].fillna("adam") == "adam")
            & (self.run_df["data_params_ModelConfigdropout_rate"].fillna("0.0").astype(str) == "0.5")
            & (self.run_df["data_params_ModelConfigrnn_dropout"].fillna("0.0").astype(str) == "0.0")
            & (self.run_df["data_params_ModelConfigkernel_regularizer_scope"].fillna("[]") == "[]")
            & (self.run_df["data_params_ExperimentConfigbatch_size"].astype(str).fillna("") == "128")
            & (
                (self.run_df["data_params_HuaweiPreprocessorConfigfine_drain_log_st"].astype(str).fillna("") == "0.75")
                | (self.run_df["data_params_HuaweiPreprocessorConfigdrain_log_st"].astype(str).fillna("") == "0.75")
            )
            & (
                (self.run_df["data_params_HuaweiPreprocessorConfigfine_drain_log_depth"].astype(str).fillna("") == "10")
                | (self.run_df["data_params_HuaweiPreprocessorConfigdrain_log_depth"].astype(str).fillna("") == "10")
            )
            & (
                (~ (
                    (self.run_df["data_params_SequenceConfigx_sequence_column_name"].astype(str).fillna("") == "coarse_log_cluster_template")
                    | (self.run_df["data_params_SequenceConfigy_sequence_column_name"].astype(str).fillna("") == "coarse_log_cluster_template")
                    | (self.run_df["data_params_HuaweiPreprocessorConfigdrain_log_sts"].fillna("[]").astype(str).apply(len) > 2)
                )) | (
                    (self.run_df["data_params_HuaweiPreprocessorConfigcoarse_drain_log_st"].astype(str).fillna("") == "0.2")
                    & (self.run_df["data_params_HuaweiPreprocessorConfigcoarse_drain_log_depth"].astype(str).fillna("") == "4")
                )
            )
        ]

        if risk_prediction:
            huawei_run_df = huawei_run_df[
                (huawei_run_df["data_tags_task_type"] == "risk_prediction") &
                (huawei_run_df["data_params_ModelConfigfinal_activation_function"] == "sigmoid")
            ]
        else:
            huawei_run_df = huawei_run_df[
                (huawei_run_df["data_params_ModelConfigfinal_activation_function"] == "softmax")
                & (huawei_run_df["data_params_SequenceConfigflatten_y"] == "True")
            ]

        if len(valid_x_columns) > 0:
            huawei_run_df = huawei_run_df[
                huawei_run_df["data_params_SequenceConfigx_sequence_column_name"].apply(lambda x: x in valid_x_columns)
            ]
        if len(valid_y_columns) > 0:
            huawei_run_df = huawei_run_df[
                huawei_run_df["data_params_SequenceConfigy_sequence_column_name"].apply(lambda x: x in valid_y_columns)
            ]

        if not include_noise:
            huawei_run_df = huawei_run_df[
                (huawei_run_df["data_tags_noise_type"].fillna("").apply(len) == 0)
            ]
        if not include_refinements:
            huawei_run_df = huawei_run_df[
                (huawei_run_df["data_tags_refinement_type"].fillna("") == "")
                & (huawei_run_df["data_params_HuaweiPreprocessorConfigmin_causality"].fillna(0.0).astype(str) == "0.01")
            ]
        if not include_drain_hierarchy:
            huawei_run_df = huawei_run_df[
                huawei_run_df["data_params_HuaweiPreprocessorConfigdrain_log_sts"].fillna("[]").astype(str).apply(len) <= 2
            ]

        return huawei_run_df

    def _load_metrics_from_local(self, run_id: str) -> Optional[Dict[str, List[float]]]:
        local_run_dir = Path(self.local_mlflow_dir + "/" + run_id + "/metrics/")
        if not local_run_dir.exists() or not local_run_dir.is_dir():
            return None
        
        metric_dict: Dict[str, List[float]] = {}
        for metric_file in local_run_dir.iterdir():
            metric = metric_file.name
            metric_history = pd.read_csv(metric_file, sep=" ", names=["time", "value", "step"]).to_dict(orient='index')
            metric_dict[metric+"_history"] = [x["value"] for x in sorted(metric_history.values(), key=lambda x: x["step"])]
            metric_dict[metric+"_times"] = [x["time"] for x in sorted(metric_history.values(), key=lambda x: x["step"])]


        return metric_dict

    def _load_metrics_from_remote(self, run_id: str) -> Dict[str, List[float]]:
        run = self.mlflow_client.get_run(run_id)
        metric_dict: Dict[str, Any] = {}
        for metric in run.data.metrics.keys():
            metric_history = self.mlflow_client.get_metric_history(
                run.info.run_id, metric
            )
            metric_dict[metric + "_history"] = [
                metric.value
                for metric in sorted(metric_history, key=lambda x: x.step)
            ]
            metric_dict[metric + "_times"] = [
                metric.time
                for metric in sorted(metric_history, key=lambda x: x.step)
            ]
        return metric_dict

    def load_metric_history_for_ids(
        self, run_ids: Set[str],
    ):
        metric_records = []
        for run_id in tqdm(run_ids, desc="Querying metrics for runs"):
            metric_dict = self._load_metrics_from_local(run_id=run_id)
            if metric_dict is None:
                metric_dict = self._load_metrics_from_remote(run_id=run_id)
            
            for metric, metric_history in metric_dict.items():
                for epoch in range(len(metric_history)):
                    metric_records.append({
                        "run_id": run_id,
                        metric: metric_history[epoch],
                        "epoch": epoch,
                    })

        return pd.merge(
            pd.DataFrame.from_records(metric_records), self.run_df, left_on="run_id", right_on="info_run_id", how="left"
        )

    def load_training_times_for_ids(
        self, run_ids: Set[str], reference_metric_name: str = "val_loss_times"
    ):
        metric_records = []
        for run_id in tqdm(run_ids, desc="Querying metrics for runs"):
            metric_dict = self._load_metrics_from_local(run_id=run_id)
            if metric_dict is None or reference_metric_name not in metric_dict:
                metric_dict = self._load_metrics_from_remote(run_id=run_id)
            if reference_metric_name not in metric_dict:
                print("Error! Reference Metric not in metric_dict", reference_metric_name, run_id)
                continue

            times = [int(x) for x in metric_dict[reference_metric_name]]
            metric_records.append({
                "run_id": run_id,
                "num_epochs": len(times),
                "total_duration": max(times) - min(times),
                "avg_per_epoch": (max(times) - min(times)) / len(times),
            })

        return pd.merge(
            pd.DataFrame.from_records(metric_records), self.run_df, left_on="run_id", right_on="info_run_id", how="inner"
        )
    
    def load_best_metrics_for_ids(
        self, run_ids: Set[str], best_metric_name: str = "val_loss_history"
    ):
        metric_records = []
        for run_id in tqdm(run_ids, desc="Querying metrics for runs"):
            metric_dict = self._load_metrics_from_local(run_id=run_id)
            if metric_dict is None or best_metric_name not in metric_dict:
                metric_dict = self._load_metrics_from_remote(run_id=run_id)
            if best_metric_name not in metric_dict:
                print("Error! Best Metric not in metric_dict", best_metric_name, run_id)
                continue

            best_epoch = [
                idx
                for idx, _ in sorted(
                    enumerate(metric_dict[best_metric_name]),
                    key=lambda x: x[1],
                    reverse=False,
                )
            ][0]
            best_metric_dict = {
                metric_name + "_best": metric_dict[metric_name][best_epoch]
                for metric_name in metric_dict
                if len(metric_dict[metric_name]) > best_epoch
            }
            best_metric_dict["run_id"] = run_id
            best_metric_dict["epoch"] = best_epoch
            metric_records.append(best_metric_dict)

        return pd.merge(
            pd.DataFrame.from_records(metric_records), self.run_df, left_on="run_id", right_on="info_run_id", how="inner"
        )

