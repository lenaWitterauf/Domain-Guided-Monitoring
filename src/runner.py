from src import models
from src import features
from src.features import preprocessing, sequences, knowledge
import pandas as pd
import logging
import dataclass_cli
import dataclasses
import tensorflow as tf
from typing import Any, Tuple
from pathlib import Path

@dataclass_cli.add
@dataclasses.dataclass
class ExperimentRunner:
    sequence_type: str = 'mimic'
    model_type: str = 'simple'
    max_data_size: int = -1
    use_dataset_generator: bool = False
    sequence_df_pkl_file: str = 'data/sequences_df.pkl'
    batch_size: int = 32

    def run(self):
        sequence_df = self._load_sequences()
        if self.max_data_size > 0 and self.max_data_size < len(sequence_df):
            logging.info('Only using first %d rows of sequence_df', self.max_data_size)
            sequence_df = sequence_df[0:self.max_data_size]

        metadata = self._collect_sequence_metadata(sequence_df)
        (train_dataset, test_dataset) = self._create_dataset(sequence_df)
        (knowledge, model) = self._load_model(metadata)

        model.train_dataset(train_dataset, test_dataset)

        embedding_helper = models.analysis.EmbeddingHelper(metadata.x_vocab, knowledge, model.embedding_layer)
        embedding_helper.print_embeddings()
        logging.info('Learned attention weights:', embedding_helper.load_attention_weights())

    def _create_dataset(self, sequence_df: pd.DataFrame) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        if self.use_dataset_generator:
            sequence_df.to_pickle(self.sequence_df_pkl_file)
            train_dataset = tf.data.Dataset.from_generator(
                sequences.generate_train,
                args=(self.sequence_df_pkl_file, self.sequence_column_name),
                output_types=(tf.float32, tf.float32),
            ).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
            test_dataset = tf.data.Dataset.from_generator(
                sequences.generate_test,
                args=(self.sequence_df_pkl_file, self.sequence_column_name),
                output_types=(tf.float32, tf.float32),
            ).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()

            return (train_dataset, test_dataset)
        else:
            transformer = sequences.load_sequence_transformer()
            split = transformer.transform_train_test_split(sequence_df, self.sequence_column_name)
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (split.train_x, split.train_y),
            ).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (split.test_x, split.test_y),
            ).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()

            return (train_dataset, test_dataset)

    def _load_model(self, metadata: sequences.SequenceMetadata) -> Tuple[Any, models.BaseModel]:
        model: models.BaseModel
        if self.model_type == 'simple':
            model = models.SimpleModel()
            model.build(metadata, metadata.x_vocab)
            return (None, model)

        elif self.model_type == 'gram' or self.model_type == 'hierarchy':
            hierarchy = self._load_hierarchy_knowledge(metadata)
            model = models.GramModel()
            model.build(metadata, hierarchy)
            return (hierarchy, model)
                
        elif self.model_type == 'text':
            description_knowledge = self._load_description_knowledge(metadata)
            model = models.DescriptionModel()
            model.build(metadata, description_knowledge)
            return (description_knowledge, model)

        elif self.model_type == 'text_paper':
            description_knowledge = self._load_description_knowledge(metadata)
            model = models.DescriptionPaperModel()
            model.build(metadata, description_knowledge)
            return (description_knowledge, model)

        elif self.model_type == 'causal':
            causality_knowledge = self._load_causal_knowledge(metadata)
            model = models.CausalityModel()
            model.build(metadata, causality_knowledge)
            return (causality_knowledge, model)

        else: 
            logging.fatal('Unknown model type %s', self.model_type)
            raise InputError(message='Unknown model type: ' + str(self.model_type))

    def _load_description_knowledge(self, metadata: sequences.SequenceMetadata) -> knowledge.DescriptionKnowledge:
        description_preprocessor: preprocessing.Preprocessor
        if self.sequence_type == 'mimic':
            description_preprocessor = preprocessing.ICDDescriptionPreprocessor()
            description_df = description_preprocessor.load_data()
            description_knowledge = knowledge.DescriptionKnowledge()
            description_knowledge.build_knowledge_from_df(description_df, metadata.x_vocab)
            return description_knowledge
        elif self.sequence_type == 'huawei_logs':
            description_preprocessor = preprocessing.ConcurrentAggregatedLogsDescriptionPreprocessor()
            description_df = description_preprocessor.load_data()
            description_knowledge = knowledge.DescriptionKnowledge()
            description_knowledge.build_knowledge_from_df(description_df, metadata.x_vocab)
            return description_knowledge
        else:
            logging.fatal('Description knowledge not available for data type %s', self.sequence_type)
            raise InputError(message='Description knowledge not available for data type: ' + str(self.sequence_type))

    def _load_causal_knowledge(self, metadata: sequences.SequenceMetadata) -> knowledge.CausalityKnowledge:
        logging.fatal('Causal knowledge not available for data type %s', self.sequence_type)
        raise InputError(message='Causal knowledge not available for data type: ' + str(self.sequence_type))

    def _load_hierarchy_knowledge(self, metadata: sequences.SequenceMetadata) -> knowledge.HierarchyKnowledge:
        hierarchy_preprocessor: preprocessing.Preprocessor
        if self.sequence_type == 'mimic':
            hierarchy_preprocessor = preprocessing.HierarchyPreprocessor()
            hierarchy_df = hierarchy_preprocessor.load_data()
            hierarchy = knowledge.HierarchyKnowledge()
            hierarchy.build_hierarchy_from_df(hierarchy_df, metadata.x_vocab)
            return hierarchy
        elif self.sequence_type == 'huawei_logs':
            hierarchy_preprocessor = preprocessing.ConcurrentAggregatedLogsHierarchyPreprocessor()
            hierarchy_df = hierarchy_preprocessor.load_data()
            hierarchy = knowledge.HierarchyKnowledge()
            hierarchy.build_hierarchy_from_df(hierarchy_df, metadata.x_vocab)
            return hierarchy
        else:
            logging.fatal('Hierarchy knowledge not available for data type %s', self.sequence_type)
            raise InputError(message='Hierarchy knowledge not available for data type: ' + str(self.sequence_type))

    def _load_sequences(self) -> pd.DataFrame:
        sequence_preprocessor: preprocessing.Preprocessor

        if self.sequence_type == 'mimic':
            mimic_config = preprocessing.MimicPreprocessorConfig()
            sequence_preprocessor = preprocessing.MimicPreprocessor(
                admission_file=mimic_config.admission_file,
                diagnosis_file=mimic_config.diagnosis_file,
                min_admissions_per_user=mimic_config.min_admissions_per_user,
            )
            self.sequence_column_name = mimic_config.sequence_column_name
            return sequence_preprocessor.load_data()

        elif self.sequence_type == 'huawei_logs':
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
            logging.fatal('Unknown data type %s', self.sequence_type)
            raise InputError(message='Unknown data type: ' + str(self.sequence_type))

    def _collect_sequence_metadata(self, sequence_df: pd.DataFrame) -> sequences.SequenceMetadata:
        if self.max_data_size > 0:
            logging.debug('Using subset of length %d instead total df of length %d', self.max_data_size, len(sequence_df))
            sequence_df = sequence_df[0:self.max_data_size]
        
        transformer = sequences.load_sequence_transformer()
        return transformer.collect_metadata(sequence_df, self.sequence_column_name)

class InputError(Exception):
    """Exception raised for errors in the input."""

    def __init__(self, message):
        self.message = message