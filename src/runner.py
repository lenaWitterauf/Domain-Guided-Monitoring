from src.features.preprocessing.mimic import MimicPreprocessor, MimicPreprocessorConfig
from src.features.preprocessing.huawei import HuaweiPreprocessorConfig
from src import models
from src import features
from src.features import preprocessing, sequences, knowledge
import pandas as pd
import logging
import dataclass_cli
import dataclasses
from typing import Any, Tuple

@dataclass_cli.add
@dataclasses.dataclass
class ExperimentRunner:
    sequence_type: str = 'mimic'
    model_type: str = 'simple'
    sequence_column_name: str = 'icd9_code_converted' #'icd9_code_converted_3digits'
    max_data_size: int = -1

    def run(self):
        sequence_df = self._load_sequences()
        split = self._split_sequences(sequence_df)

        (knowledge, model) = self._load_model(split)
        model.train(split)

        embedding_helper = models.analysis.EmbeddingHelper(split.x_vocab, knowledge, model.embedding_layer)
        embedding_helper.print_embeddings()
        print('Learned attention weights:', embedding_helper.load_attention_weights())

    def _load_model(self, split: sequences.TrainTestSplit) -> Tuple[Any, models.BaseModel]:
        model: models.BaseModel
        if self.model_type == 'simple':
            model = models.SimpleModel()
            model.build(split, split.x_vocab)
            return (None, model)

        elif self.model_type == 'gram' or self.model_type == 'hierarchy':
            hierarchy = self._load_hierarchy_knowledge(split)
            model = models.GramModel()
            model.build(split, hierarchy)
            return (hierarchy, model)
                
        elif self.model_type == 'text':
            description_knowledge = self._load_description_knowledge(split)
            model = models.DescriptionModel()
            model.build(split, description_knowledge)
            return (description_knowledge, model)

        elif self.model_type == 'text_paper':
            description_knowledge = self._load_description_knowledge(split)
            model = models.DescriptionPaperModel()
            model.build(split, description_knowledge)
            return (description_knowledge, model)

        elif self.model_type == 'causal':
            causality_knowledge = self._load_causal_knowledge(split)
            model = models.CausalityModel()
            model.build(split, causality_knowledge)
            return (causality_knowledge, model)

        else: 
            logging.fatal('Unknown model type %s', self.model_type)
            raise InputError(message='Unknown model type: ' + str(self.model_type))

    def _load_description_knowledge(self, split: sequences.TrainTestSplit) -> knowledge.DescriptionKnowledge:
        description_preprocessor: preprocessing.Preprocessor
        if self.sequence_type == 'mimic':
            description_preprocessor = preprocessing.ICDDescriptionPreprocessor()
            description_df = description_preprocessor.load_data()
            description_knowledge = knowledge.DescriptionKnowledge()
            description_knowledge.build_knowledge_from_df(description_df, split.x_vocab)
            return description_knowledge
        elif self.sequence_type == 'huawei_logs':
            description_preprocessor = preprocessing.ConcurrentAggregatedLogsDescriptionPreprocessor()
            description_df = description_preprocessor.load_data()
            description_knowledge = knowledge.DescriptionKnowledge()
            description_knowledge.build_knowledge_from_df(description_df, split.x_vocab)
            return description_knowledge
        else:
            logging.fatal('Description knowledge not available for data type %s', self.sequence_type)
            raise InputError(message='Description knowledge not available for data type: ' + str(self.sequence_type))

    def _load_causal_knowledge(self, split: sequences.TrainTestSplit) -> knowledge.CausalityKnowledge:
        logging.fatal('Causal knowledge not available for data type %s', self.sequence_type)
        raise InputError(message='Causal knowledge not available for data type: ' + str(self.sequence_type))

    def _load_hierarchy_knowledge(self, split: sequences.TrainTestSplit) -> knowledge.HierarchyKnowledge:
        hierarchy_preprocessor: preprocessing.Preprocessor
        if self.sequence_type == 'mimic':
            hierarchy_preprocessor = preprocessing.HierarchyPreprocessor()
            hierarchy_df = hierarchy_preprocessor.load_data()
            hierarchy = knowledge.HierarchyKnowledge()
            hierarchy.build_hierarchy_from_df(hierarchy_df, split.x_vocab)
            return hierarchy
        elif self.sequence_type == 'huawei_logs':
            hierarchy_preprocessor = preprocessing.ConcurrentAggregatedLogsHierarchyPreprocessor()
            hierarchy_df = hierarchy_preprocessor.load_data()
            hierarchy = knowledge.HierarchyKnowledge()
            hierarchy.build_hierarchy_from_df(hierarchy_df, split.x_vocab)
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
            return sequence_preprocessor.load_data()

        elif self.sequence_type == 'huawei_logs':
            huawei_config = preprocessing.HuaweiPreprocessorConfig()
            sequence_preprocessor = preprocessing.ConcurrentAggregatedLogsPreprocessor(
                log_file=huawei_config.aggregated_log_file,
                relevant_columns=huawei_config.relevant_aggregated_log_columns,
                datetime_column_name=huawei_config.datetime_column_name,
                max_sequence_length=huawei_config.max_sequence_length,
                pkl_file=huawei_config.aggregated_log_pkl_file,
            )
            self.sequence_column_name = sequence_preprocessor.sequence_column_name
            return sequence_preprocessor.load_data()
        else:
            logging.fatal('Unknown data type %s', self.sequence_type)
            raise InputError(message='Unknown data type: ' + str(self.sequence_type))

    def _split_sequences(self, sequence_df: pd.DataFrame) -> features.sequences.TrainTestSplit:
        if self.max_data_size > 0:
            logging.debug('Using subset of length %d instead total df of length %d', self.max_data_size, len(sequence_df))
            sequence_df = sequence_df[0:self.max_data_size]
        
        transformer: sequences.NextSequenceTransformer
        config = sequences.SequenceConfig()
        if len(config.valid_y_features) > 0:
            logging.debug('Using only features %s as prediction goals', ','.join(config.valid_y_features))

            transformer = sequences.NextPartialSequenceTransformer(
                test_percentage=config.test_percentage,
                random_test_split=config.random_test_split,
                random_state=config.random_state,
                flatten_x=config.flatten_x,
                flatten_y=config.flatten_y,
                max_window_size=config.max_window_size,
                min_window_size=config.min_window_size,
                window_overlap=config.window_overlap,
            )
            transformer.valid_y_features = config.valid_y_features
            transformer.remove_empty_v_vecs = config.remove_empty_v_vecs
        else:
            transformer = sequences.NextSequenceTransformer(
                test_percentage=config.test_percentage,
                random_test_split=config.random_test_split,
                random_state=config.random_state,
                flatten_x=config.flatten_x,
                flatten_y=config.flatten_y,
                max_window_size=config.max_window_size,
                min_window_size=config.min_window_size,
                window_overlap=config.window_overlap,
            )

        return transformer.transform_train_test_split(sequence_df, self.sequence_column_name)

class InputError(Exception):
    """Exception raised for errors in the input."""

    def __init__(self, message):
        self.message = message