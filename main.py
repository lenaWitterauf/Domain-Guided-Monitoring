from src import ExperimentRunner
import logging
import mlflow
from src import ExperimentConfig
from src.features import preprocessing, sequences
from src.training import models

logging.basicConfig(level=logging.DEBUG)

def _log_all_configs_to_mlflow():
    for config in [
        ExperimentConfig(),
        preprocessing.huawei.HuaweiPreprocessorConfig(),
        preprocessing.mimic.MimicPreprocessorConfig(),
        sequences.SequenceConfig(),
        models.ModelConfig(),
    ]:
        for config_name, config_value in vars(config).items():
            full_config_name = config.__class__.__name__ + config_name
            mlflow.log_param(full_config_name, str(config_value))

if __name__ == "__main__":
    mlflow.set_experiment("Domain Guided Monitoring")
    with mlflow.start_run() as run:
        _log_all_configs_to_mlflow()
        runner = ExperimentRunner(run.info.run_id)
        runner.run()




