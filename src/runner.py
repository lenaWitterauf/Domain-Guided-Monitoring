from src.features.sequences.transformer import SequenceMetadata
from src.training.analysis import confusion
from src.training import analysis, models
from src.features import preprocessing, sequences, knowledge
import pandas as pd
import logging
import tensorflow as tf
from typing import Any, Tuple
from .config import ExperimentConfig
import mlflow
from pathlib import Path


class ExperimentRunner:
    sequence_df_pkl_file: str = "data/sequences_df.pkl"

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.config = ExperimentConfig()
        self.multilabel_classification = self.config.multilabel_classification

    def run(self):
        logging.info("Starting run %s", self.run_id)
        tf.random.set_seed(self.config.tensorflow_seed)
        sequence_df = self._load_sequences()
        if self.config.max_data_size > 0 and self.config.max_data_size < len(
            sequence_df
        ):
            logging.info(
                "Only using first %d rows of sequence_df", self.config.max_data_size
            )
            sequence_df = sequence_df[0 : self.config.max_data_size]

        metadata = self._collect_sequence_metadata(sequence_df)
        (train_dataset, test_dataset) = self._create_dataset(sequence_df)
        (knowledge, model) = self._load_model(metadata)

        model.train_dataset(
            train_dataset,
            test_dataset,
            self.multilabel_classification,
            self.config.n_epochs,
        )

        self._log_dataset_info(train_dataset, test_dataset, metadata)
        self._generate_artifacts(
            metadata, train_dataset, test_dataset, knowledge, model
        )
        self._set_mlflow_tags(metadata)

    def _log_dataset_info(
        self,
        train_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        metadata: SequenceMetadata,
    ):
        mlflow.log_metric("train_size", len([x for x in train_dataset]))
        mlflow.log_metric("test_size", len([x for x in test_dataset]))
        mlflow.log_metric("x_vocab_size", len(metadata.x_vocab))
        mlflow.log_metric("y_vocab_size", len(metadata.y_vocab))

    def _set_mlflow_tags(self, metadata: sequences.SequenceMetadata):
        mlflow.set_tag("sequence_type", self.config.sequence_type)
        mlflow.set_tag("model_type", self.config.model_type)
        if len(metadata.y_vocab) == 1:
            mlflow.set_tag("task_type", "risk_prediction")
        else:
            mlflow.set_tag("task_type", "sequence_prediction")

    def _generate_artifacts(
        self,
        metadata: sequences.SequenceMetadata,
        train_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        knowledge: Any,
        model: models.BaseModel,
    ):
        artifact_dir = "artifacts/run_{}/".format(self.run_id)
        artifact_path = Path(artifact_dir)
        if not artifact_path.exists():
            artifact_path.mkdir()

        self._generate_metric_artifacts(artifact_dir, model)
        self._generate_embedding_artifacts(artifact_dir, metadata, knowledge, model)
        self._generate_confusion_artifacts(
            artifact_dir, metadata, model, train_dataset, test_dataset
        )

        mlflow.log_artifacts(artifact_dir)

    def _generate_metric_artifacts(
        self, artifact_dir: str, model: models.BaseModel,
    ):
        metric_plotter = analysis.MetricPlotter(model, plot_path=artifact_dir)
        metric_plotter.plot_all_metrics()

    def _generate_confusion_artifacts(
        self,
        artifact_dir: str,
        metadata: sequences.SequenceMetadata,
        model: models.BaseModel,
        train_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
    ):
        if self.multilabel_classification:
            logging.error(
                "Can't create confusion matrix for multilabel classification!"
            )
            return

        confusion_calculator = confusion.ConfusionCalculator(
            metadata, model.prediction_model
        )
        confusion_calculator.write_confusion_for_dataset(
            train_dataset, out_file_name=artifact_dir + "confusion_train.csv"
        )
        confusion_calculator.write_confusion_for_dataset(
            test_dataset, out_file_name=artifact_dir + "confusion_test.csv"
        )

    def _generate_embedding_artifacts(
        self,
        artifact_dir: str,
        metadata: sequences.SequenceMetadata,
        knowledge: Any,
        model: models.BaseModel,
    ):
        embedding_helper = analysis.EmbeddingHelper(
            metadata.x_vocab, knowledge, model.embedding_layer
        )
        if self.config.model_type in ["simple", "text_paper"]:
            embedding_helper.write_embeddings(
                vec_file_name=artifact_dir + "vecs.tsv",
                meta_file_name=artifact_dir + "meta.tsv",
                include_base_embeddings=False,
            )
        else:
            embedding_helper.write_embeddings(
                vec_file_name=artifact_dir + "vecs.tsv",
                meta_file_name=artifact_dir + "meta.tsv",
                include_base_embeddings=True,
            )
            embedding_helper.write_attention_weights(
                file_name=artifact_dir + "attention.json",
            )

    def _create_dataset(
        self, sequence_df: pd.DataFrame
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        if self.config.use_dataset_generator:
            sequence_df.to_pickle(self.sequence_df_pkl_file)
            train_dataset = (
                tf.data.Dataset.from_generator(
                    sequences.generate_train,
                    args=(self.sequence_df_pkl_file, self.sequence_column_name),
                    output_types=(tf.float32, tf.float32),
                )
                .cache(self._get_cache_file_name(is_test=False))
                .shuffle(
                    self.config.dataset_shuffle_buffer,
                    seed=self.config.dataset_shuffle_seed,
                    reshuffle_each_iteration=True,
                )
                .batch(self.config.batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )
            test_dataset = (
                tf.data.Dataset.from_generator(
                    sequences.generate_test,
                    args=(self.sequence_df_pkl_file, self.sequence_column_name),
                    output_types=(tf.float32, tf.float32),
                )
                .cache(self._get_cache_file_name(is_test=True))
                .batch(self.config.batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

            return (train_dataset, test_dataset)
        else:
            transformer = sequences.load_sequence_transformer()
            split = transformer.transform_train_test_split(
                sequence_df, self.sequence_column_name
            )
            train_dataset = (
                tf.data.Dataset.from_tensor_slices((split.train_x, split.train_y),)
                .batch(self.config.batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
                .cache()
                .shuffle(
                    self.config.dataset_shuffle_buffer,
                    seed=self.config.dataset_shuffle_seed,
                    reshuffle_each_iteration=True,
                )
            )
            test_dataset = (
                tf.data.Dataset.from_tensor_slices((split.test_x, split.test_y),)
                .batch(self.config.batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
                .cache()
            )

            return (train_dataset, test_dataset)

    def _get_cache_file_name(self, is_test: bool) -> str:
        if len(self.config.dataset_generator_cache_file) < 1:
            return ""
        else:
            return self.config.dataset_generator_cache_file + (
                "_test" if is_test else "_train"
            )

    def _load_model(
        self, metadata: sequences.SequenceMetadata
    ) -> Tuple[Any, models.BaseModel]:
        model: models.BaseModel
        if self.config.model_type == "simple":
            model = models.SimpleModel()
            model.build(metadata, metadata.x_vocab)
            return (None, model)

        elif self.config.model_type == "gram" or self.config.model_type == "hierarchy":
            hierarchy = self._load_hierarchy_knowledge(metadata)
            model = models.GramModel()
            model.build(metadata, hierarchy)
            return (hierarchy, model)

        elif self.config.model_type == "text":
            description_knowledge = self._load_description_knowledge(metadata)
            model = models.DescriptionModel()
            model.build(metadata, description_knowledge)
            return (description_knowledge, model)

        elif self.config.model_type == "text_paper":
            description_knowledge = self._load_description_knowledge(metadata)
            model = models.DescriptionPaperModel()
            model.build(metadata, description_knowledge)
            return (description_knowledge, model)

        elif self.config.model_type == "causal":
            causality_knowledge = self._load_causal_knowledge(metadata)
            model = models.CausalityModel()
            model.build(metadata, causality_knowledge)
            return (causality_knowledge, model)

        else:
            logging.fatal("Unknown model type %s", self.config.model_type)
            raise InputError(
                message="Unknown model type: " + str(self.config.model_type)
            )

    def _load_description_knowledge(
        self, metadata: sequences.SequenceMetadata
    ) -> knowledge.DescriptionKnowledge:
        description_preprocessor: preprocessing.Preprocessor
        if self.config.sequence_type == "mimic":
            description_preprocessor = preprocessing.ICD9DescriptionPreprocessor()
            description_df = description_preprocessor.load_data()
            description_knowledge = knowledge.DescriptionKnowledge()
            description_knowledge.build_knowledge_from_df(
                description_df, metadata.x_vocab
            )
            return description_knowledge
        elif self.config.sequence_type == "huawei_logs":
            description_preprocessor = (
                preprocessing.ConcurrentAggregatedLogsDescriptionPreprocessor()
            )
            description_df = description_preprocessor.load_data()
            description_knowledge = knowledge.DescriptionKnowledge()
            description_knowledge.build_knowledge_from_df(
                description_df, metadata.x_vocab
            )
            return description_knowledge
        else:
            logging.fatal(
                "Description knowledge not available for data type %s",
                self.config.sequence_type,
            )
            raise InputError(
                message="Description knowledge not available for data type: "
                + str(self.config.sequence_type)
            )

    def _load_causal_knowledge(
        self, metadata: sequences.SequenceMetadata
    ) -> knowledge.CausalityKnowledge:
        causality_preprocessor: preprocessing.Preprocessor
        if self.config.sequence_type == "huawei_logs":
            causality_preprocessor = (
                preprocessing.ConcurrentAggregatedLogsCausalityPreprocessor()
            )
            causality_df = causality_preprocessor.load_data()
            causality = knowledge.CausalityKnowledge()
            causality.build_causality_from_df(causality_df, metadata.x_vocab)
            return causality
        else:
            logging.fatal(
                "Causal knowledge not available for data type %s",
                self.config.sequence_type,
            )
            raise InputError(
                message="Causal knowledge not available for data type: "
                + str(self.config.sequence_type)
            )

    def _load_hierarchy_knowledge(
        self, metadata: sequences.SequenceMetadata
    ) -> knowledge.HierarchyKnowledge:
        hierarchy_preprocessor: preprocessing.Preprocessor
        if self.config.sequence_type == "mimic":
            hierarchy_preprocessor = preprocessing.ICD9HierarchyPreprocessor()
            hierarchy_df = hierarchy_preprocessor.load_data()
            hierarchy = knowledge.HierarchyKnowledge()
            hierarchy.build_hierarchy_from_df(hierarchy_df, metadata.x_vocab)
            return hierarchy
        elif self.config.sequence_type == "huawei_logs":
            hierarchy_preprocessor = (
                preprocessing.ConcurrentAggregatedLogsHierarchyPreprocessor()
            )
            hierarchy_df = hierarchy_preprocessor.load_data()
            hierarchy = knowledge.HierarchyKnowledge()
            hierarchy.build_hierarchy_from_df(hierarchy_df, metadata.x_vocab)
            return hierarchy
        else:
            logging.fatal(
                "Hierarchy knowledge not available for data type %s",
                self.config.sequence_type,
            )
            raise InputError(
                message="Hierarchy knowledge not available for data type: "
                + str(self.config.sequence_type)
            )

    def _load_sequences(self) -> pd.DataFrame:
        sequence_preprocessor: preprocessing.Preprocessor

        if self.config.sequence_type == "mimic":
            mimic_config = preprocessing.MimicPreprocessorConfig()
            sequence_preprocessor = preprocessing.MimicPreprocessor(
                admission_file=mimic_config.admission_file,
                diagnosis_file=mimic_config.diagnosis_file,
                min_admissions_per_user=mimic_config.min_admissions_per_user,
            )
            self.sequence_column_name = mimic_config.sequence_column_name
            return sequence_preprocessor.load_data()

        elif self.config.sequence_type == "huawei_logs":
            huawei_config = preprocessing.HuaweiPreprocessorConfig()
            sequence_preprocessor = preprocessing.ConcurrentAggregatedLogsPreprocessor(
                log_file=huawei_config.aggregated_log_file,
                relevant_columns=huawei_config.relevant_aggregated_log_columns,
                datetime_column_name=huawei_config.datetime_column_name,
                max_sequence_length=huawei_config.max_sequence_length,
            )
            self.sequence_column_name = sequence_preprocessor.sequence_column_name
            return sequence_preprocessor.load_data()
        else:
            logging.fatal("Unknown data type %s", self.config.sequence_type)
            raise InputError(
                message="Unknown data type: " + str(self.config.sequence_type)
            )

    def _collect_sequence_metadata(
        self, sequence_df: pd.DataFrame
    ) -> sequences.SequenceMetadata:
        if self.config.max_data_size > 0:
            logging.debug(
                "Using subset of length %d instead total df of length %d",
                self.config.max_data_size,
                len(sequence_df),
            )
            sequence_df = sequence_df[0 : self.config.max_data_size]

        transformer = sequences.load_sequence_transformer()
        if not transformer.flatten_y:
            self.multilabel_classification = False
        return transformer.collect_metadata(sequence_df, self.sequence_column_name)


class InputError(Exception):
    """Exception raised for errors in the input."""

    def __init__(self, message):
        self.message = message
