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
    model_type: str = 'text'
    sequence_column_name: str = 'icd9_code_converted' #'icd9_code_converted_3digits'
    need_sequence_preprocessing: bool = True

    def run(self):
        sequence_df = self.load_sequences()
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
            model = models.SimpleLSTMModel()
            model.build(split.max_length, len(split.vocab))
            return model

        elif self.model_type == 'gram':
            hierarchy_preprocessor = preprocessing.HierarchyPreprocessor()
            hierarchy_df = hierarchy_preprocessor.preprocess_hierarchy()
            hierarchy = knowledge.HierarchyKnowledge()
            hierarchy.build_hierarchy_from_df(hierarchy_df, split.vocab)

            model = models.GramModel()
            model.build(hierarchy, split.max_length, len(split.vocab))
            return model
        
        elif self.model_type == 'text':
            description_preprocessor = preprocessing.ICDDescriptionPreprocessor()
            description_df = description_preprocessor.load_descriptions()
            description_knowledge = knowledge.DescriptionKnowledge()
            description_knowledge.build_knowledge_from_df(description_df, split.vocab)

            model = models.TextualModel()
            model.build(description_knowledge, split.max_length, len(split.vocab))
            return model

        else: 
            logging.fatal('Unknown model type %s', self.model_type)
            return

    def load_sequences(self) -> pd.DataFrame:
        if self.sequence_type != 'mimic':
            logging.fatal('Unknown sequence type %s, please use MIMIC only for now!', self.sequence_type)
            return
        
        preprocessor_config = preprocessing.PreprocessorConfig()
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


