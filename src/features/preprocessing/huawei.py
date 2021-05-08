import dataclass_cli
import dataclasses
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List

@dataclass_cli.add
@dataclasses.dataclass
class HuaweiPreprocessorConfig:
    aggregated_log_file: Path = Path('data/logs_aggregated_concurrent.csv')
    relevant_aggregated_log_columns: List[str] = dataclasses.field(
        default_factory=lambda: ['Hostname', 'log_level', 'programname', 'python_module', 'http_status', 'http_method'],
    )
    datetime_column_name: str = '@timestamp'
    max_sequence_length: int = 100
    aggregated_log_pkl_file: Path = Path('data/huwawei_logs.pkl')

class ConcurrentAggregatedLogsPreprocessor:
    log_file: Path
    pkl_file: Path
    relevant_columns: List[str]
    datetime_column_name: str
    max_sequence_length: int
    sequence_column_name: str = 'sequence'

    def __init__(self,
            log_file: Path = Path('data/logs_aggregated_concurrent.csv'),
            relevant_columns: List[str] = ['Hostname', 'log_level', 'programname', 'python_module', 'http_status', 'http_method'],
            datetime_column_name: str = '@timestamp',
            max_sequence_length: int = 100,
            pkl_file: Path = Path('data/huwawei_logs.pkl')):
        self.log_file = log_file
        self.relevant_columns = relevant_columns
        self.datetime_column_name = datetime_column_name
        self.max_sequence_length = max_sequence_length
        self.pkl_file = pkl_file

    def preprocess_data(self) -> pd.DataFrame:
        df = self._read_raw_df()
        labels = df.values.tolist()
        labels = [[str(l).lower() for l in label_list if len(str(l)) > 0] for label_list in labels]
        subsequence_list = self._transform_to_subsequences(labels)
        return pd.DataFrame(data={
            self.sequence_column_name: subsequence_list,
        })

    def _transform_to_subsequences(self, labels: List[List[str]]) -> List[List[List[str]]]:
        num_datapoints = len(labels)
        next_start_idx = 0
        next_end_idx = min(self.max_sequence_length, num_datapoints)
        subsequences = []
        with tqdm(total=num_datapoints, desc='Generating subsequences from Huawei data') as pbar:
            while next_end_idx < num_datapoints:
                subsequences.append(labels[next_start_idx:next_end_idx])
                next_start_idx = next_end_idx
                next_end_idx = min(next_end_idx + self.max_sequence_length, num_datapoints)
                pbar.update(self.max_sequence_length)

        return subsequences

    def _read_raw_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.log_file)
        rel_df = df[self.relevant_columns]
        rel_df['DateTime'] = pd.to_datetime(df[self.datetime_column_name])
        rel_df = rel_df.fillna('')
        rel_df = rel_df.sort_values(by='DateTime')
        return rel_df.drop(columns=['DateTime'])

    def load_logs_from_pkl(self) -> pd.DataFrame:
        return pd.read_pickle(self.pkl_file)

    def write_logs_to_pkl(self, aggregated_df: pd.DataFrame):
        aggregated_df.to_pickle(self.pkl_file)