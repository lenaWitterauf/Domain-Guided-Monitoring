from src import features, models
from src.features import preprocessing, sequences, knowledge
import pandas as pd
import logging
import dataclass_cli
import dataclasses

@dataclass_cli.add
@dataclasses.dataclass
class ExperimentRunner:
    sequence_type: str = 'mimic'
    model_type: str = 'simple'
    sequence_column_name: str = 'icd9_code_converted' #'icd9_code_converted_3digits'
    need_sequence_preprocessing: bool = True
    max_data_size: int = -1

    def run(self):
        sequence_df = self.load_sequences()
        if self.max_data_size > 0:
            logging.debug('Using subset of length %d instead total df of length %d', self.max_data_size, len(sequence_df))
            sequence_df = sequence_df[0:self.max_data_size]
        handler_config = sequences.SequenceHandlerConfig()
        handler = sequences.SequenceHandler(
            test_percentage=handler_config.test_percentage,
            random_state=handler_config.random_state,
            flatten=handler_config.flatten,
        )
        split = handler.transform_train_test_split(sequence_df, self.sequence_column_name)

        model = self.load_model(split)
        model.train(split)

    def load_model(self, split: sequences.TrainTestSplit):
        if self.model_type == 'simple':
            model = models.SimpleModel()
            model.build(split, split.vocab)
            return model

        elif self.model_type == 'gram':
            hierarchy = self.load_hierarchy_knowledge()
            model = models.GramModel()
            model.build(split, hierarchy)
            return model
                
        elif self.model_type == 'text':
            description_knowledge = self.load_description_knowledge()
            model = models.DescriptionModel()
            model.build(split, description_knowledge)
            return model

        elif self.model_type == 'text_paper':
            description_knowledge = self.load_description_knowledge()
            model = models.DescriptionPaperModel()
            model.build(split, description_knowledge)
            return model

        elif self.model_type == 'causal':
            causality_knowledge = self.load_causal_knowledge()
            model = models.CausalityModel()
            model.build(split, causality_knowledge)
            return model

        else: 
            logging.fatal('Unknown model type %s', self.model_type)
            return

    def load_description_knowledge(self) -> knowledge.DescriptionKnowledge:
        if self.sequence_type == 'mimic':
            description_preprocessor = preprocessing.ICDDescriptionPreprocessor()
            description_df = description_preprocessor.load_descriptions()
            description_knowledge = knowledge.DescriptionKnowledge()
            description_knowledge.build_knowledge_from_df(description_df, split.vocab)
            return description_knowledge
        else:
            logging.fatal('Description knowledge not available for data type %s', self.sequence_type)
            return 

    def load_causal_knowledge(self) -> knowledge.CausalityKnowledge:
        logging.fatal('Causal knowledge not available for data type %s', self.sequence_type)
        return 

    def load_hierarchy_knowledge(self) -> knowledge.HierarchyKnowledge:
        if self.sequence_type == 'mimic':
            hierarchy_preprocessor = preprocessing.HierarchyPreprocessor()
            hierarchy_df = hierarchy_preprocessor.preprocess_hierarchy()
            hierarchy = knowledge.HierarchyKnowledge()
            hierarchy.build_hierarchy_from_df(hierarchy_df, split.vocab)
        else:
            logging.fatal('Hierarchy knowledge not available for data type %s', self.sequence_type)
            return 

    def load_sequences(self) -> pd.DataFrame:
        if self.sequence_type == 'mimic':
            preprocessor_config = preprocessing.MimicPreprocessorConfig()
            preprocessor = preprocessing.MimicPreprocessor(
                admission_file=preprocessor_config.admission_file,
                diagnosis_file=preprocessor_config.diagnosis_file,
                min_admissions_per_user=preprocessor_config.min_admissions_per_user,
            )
            if self.need_sequence_preprocessing:
                sequence_df = preprocessor.preprocess_mimic()
                preprocessor.write_mimic_to_pkl(sequence_df)
                return sequence_df
            else:
                return preprocessor.load_mimic_from_pkl()
        elif self.sequence_type == 'huawei_logs':
            preprocessor_config = preprocessing.HuaweiPreprocessorConfig()
            preprocessor = preprocessing.ConcurrentAggregatedLogsPreprocessor(
                log_file=preprocessor_config.aggregated_log_file,
                relevant_columns=preprocessor_config.relevant_aggregated_log_columns,
                datetime_column_name=preprocessor_config.datetime_column_name,
                max_sequence_length=preprocessor_config.max_sequence_length,
                pkl_file=preprocessor_config.aggregated_log_pkl_file,
            )
            self.sequence_column_name = preprocessor.sequence_column_name
            if self.need_sequence_preprocessing:
                sequence_df = preprocessor.preprocess_data()
                preprocessor.write_logs_to_pkl(sequence_df)
                return sequence_df
            else:
                return preprocessor.load_logs_from_pkl()
        else:
            logging.fatal('Unknown data type %s', self.sequence_type)
            return
