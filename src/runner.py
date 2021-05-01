from src import features, models
from src.features import preprocessing, sequences
import pandas as pd
import logging
import dataclass_cli
import dataclasses

@dataclass_cli.add
@dataclasses.dataclass
class ExperimentRunner:
    sequence_type: str = 'mimic'
    model_type: str = 'simple'
    sequence_column_name: str = 'icd9_code_converted_3digits'
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

        model = models.SimpleLSTMModel()
        model.build(split.max_length, len(split.vocab))
        model.train(split)

    def load_model(self):
        if self.model_type == 'simple':
            return models.SimpleLSTMModel()

        elif self.model_type == 'gram':
            # TODO: initialize GRAM hierarchy
            logging.fatal('GRAM cannot be used yet - hierarchy parsing not implemented!')
            return models.GramModel()
        
        else: 
            logging.fatal('Unknown model type %s', self.model_type)
            return

    def load_sequences(self) -> pd.DataFrame:
        if self.sequence_type != 'mimic':
            logging.fatal('Unknown sequence type %s, please use MIMIC only for now!', self.sequence_type)
            return
        
        preprocessor_config = preprocessor.mimic.PreprocessorConfig()
        preprocessor = preprocessing.mimic.Preprocessor(
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



