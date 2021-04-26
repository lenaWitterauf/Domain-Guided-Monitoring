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
    sequence_column_name: str = 'icd9_code_converted_3digits'
    need_sequence_preprocessing: bool = True

    def run(self):
        sequence_df = self.load_sequences()
        handler = sequences.SequenceHandler()
        split = handler.transform_train_test_split(sequence_df, self.sequence_column_name)

        model = models.SimpleLSTMModel()
        model.build(split.max_length, len(split.vocab))
        model.train(split)

    def load_sequences(self) -> pd.DataFrame:
        if self.sequence_type != 'mimic':
            logging.fatal('Unknown sequence type %s, please use MIMIC only for now!', self.sequence_type)
            return
        
        preprocessor = preprocessing.mimic.Preprocessor()
        if self.need_sequence_preprocessing:
            return preprocessor.preprocess_mimic()
        else:
            return preprocessor.load_mimic_from_pkl()



