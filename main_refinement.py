import logging
from src import _main
from src import features, refinement
import json
import time
from typing import Dict, Any
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib.font_manager").disabled = True


def _write_file_knowledge(knowledge: Dict[Any, Any]):
    knowledge_config = features.knowledge.KnowledgeConfig()
    with open(knowledge_config.file_knowledge, "w") as knowledge_file:
        json.dump(knowledge, knowledge_file)


def _write_reference_knowledge(refinement_config: refinement.RefinementConfig):
    logging.info("Writing reference knowledge...")
    original_knowledge = refinement.KnowledgeProcessor(
        refinement_config
    ).load_original_knowledge()
    _write_file_knowledge(original_knowledge)


def _write_original_knowledge(refinement_config: refinement.RefinementConfig):
    logging.info("Writing original knowledge...")
    original_knowledge = refinement.KnowledgeProcessor(
        refinement_config
    ).load_original_knowledge()
    _write_file_knowledge(original_knowledge)


def _write_refined_knowledge(
    refinement_config: refinement.RefinementConfig,
    refinement_run_id: str,
    reference_run_id: str,
):
    logging.info("Writing refined knowledge...")
    refined_knowledge = refinement.KnowledgeProcessor(
        refinement_config
    ).load_refined_knowledge(
        refinement_run_id=refinement_run_id, reference_run_id=reference_run_id
    )
    _write_file_knowledge(refined_knowledge)


def _add_mlflow_tag(run_id: str, refinement_timestamp: int, suffix: str):
    mlflow_client = MlflowClient()
    mlflow_client.set_tag(
        run_id=run_id,
        key="refinement_type",
        value="{identifier}_{suffix}".format(
            identifier=str(refinement_timestamp), suffix=suffix
        ),
    )


def main_refinement():
    refinement_timestamp = time.time()
    refinement_config = refinement.RefinementConfig()
    _write_reference_knowledge(refinement_config)
    reference_run_id = _main()
    _add_mlflow_tag(reference_run_id, refinement_timestamp, suffix="reference")

    _write_original_knowledge(refinement_config)
    refinement_run_id = _main()
    _add_mlflow_tag(refinement_run_id, refinement_timestamp, suffix="original")

    for i in range(refinement_config.num_refinements):
        _write_refined_knowledge(
            refinement_config,
            reference_run_id=reference_run_id,
            refinement_run_id=refinement_run_id,
        )
        refinement_run_id = _main()
        _add_mlflow_tag(
            refinement_run_id, refinement_timestamp, suffix="refinement_" + str(i)
        )


if __name__ == "__main__":
    main_refinement()
