import dataclass_cli
import dataclasses
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import http
import re
from .base import Preprocessor

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

class ConcurrentAggregatedLogsPreprocessor(Preprocessor):
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

    def load_data(self) -> pd.DataFrame:
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
            while next_end_idx > next_start_idx+1:
                subsequences.append(labels[next_start_idx:next_end_idx])
                next_start_idx = next_end_idx
                next_end_idx = min(next_end_idx + self.max_sequence_length, num_datapoints)
                pbar.update(self.max_sequence_length)

        return subsequences

    def _read_raw_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.log_file)
        rel_df = df[self.relevant_columns + [self.datetime_column_name]]
        rel_df[self.datetime_column_name] = pd.to_datetime(rel_df[self.datetime_column_name])
        rel_df = rel_df.fillna('')
        rel_df = rel_df.sort_values(by=self.datetime_column_name)
        return rel_df.drop(columns=[self.datetime_column_name])

    def load_from_pkl(self) -> pd.DataFrame:
        return pd.read_pickle(self.pkl_file)

    def write_to_pkl(self, aggregated_df: pd.DataFrame):
        aggregated_df.to_pickle(self.pkl_file)

class ConcurrentAggregatedLogsDescriptionPreprocessor(Preprocessor):
    log_file: Path
    relevant_columns: List[str]

    def __init__(self,
            log_file: Path = Path('data/logs_aggregated_concurrent.csv'),
            relevant_columns: List[str] = ['Hostname', 'log_level', 'programname', 'python_module', 'http_status', 'http_method']):
        self.log_file = log_file
        self.relevant_columns = relevant_columns

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.log_file)
        rel_df = df[self.relevant_columns]

        http_descriptions = self._load_http_descriptions()
        description_df = pd.DataFrame(columns=['label', 'description'])
        for column in self.relevant_columns:
            values = set(rel_df[column].dropna())
            values = set([str(x).lower() for x in values if len(str(x)) > 0])
            for value in tqdm(values, desc='Loading descriptions for column ' + column):
                description = ''
                if column == 'Hostname':
                    name = value.rstrip('0123456789')
                    number = value[len(name):]
                    description = name + ' ' + number
                elif column == 'http_status':
                    description = http_descriptions[value]
                else:
                    description = ' '.join(re.split('[,._-]+', value))
                
                description_df = description_df.append({
                    'label': value,
                    'description': description,
                }, ignore_index=True)
    
        return description_df[['label', 'description']]

    def _load_http_descriptions(self) -> Dict[str, str]:
        logging.debug('Initializing HTTP Status descriptions')
        http_descriptions = {}
        for status in list(http.HTTPStatus):
            http_descriptions[str(status.value) + '.0'] = status.name.lower().replace('_', ' ')
        
        return http_descriptions

class ConcurrentAggregatedLogsHierarchyPreprocessor(Preprocessor):
    log_file: Path
    relevant_columns: List[str]

    def __init__(self,
            log_file: Path = Path('data/logs_aggregated_concurrent.csv'),
            relevant_columns: List[str] = ['Hostname', 'log_level', 'programname', 'python_module', 'http_status', 'http_method']):
        self.log_file = log_file
        self.relevant_columns = relevant_columns

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.log_file)
        rel_df = df[self.relevant_columns]

        hierarchy_df = pd.DataFrame(columns=['parent', 'child'])
        for column in self.relevant_columns:
            hierarchy_df = hierarchy_df.append({
                'parent': 'root',
                'child': column,
            }, ignore_index=True)

            values = set(rel_df[column].dropna())
            values = set([str(x).lower() for x in values if len(str(x)) > 0])
            for value in tqdm(values, desc='Loading hierarchy for column ' + column):
                hierarchy_elements = [column]
                if column == 'Hostname':
                    hierarchy_elements.append(value.rstrip('0123456789'))
                elif column == 'http_status':
                    hierarchy_elements.append(value[0] + '00')
                else:
                    hierarchy_elements = hierarchy_elements + re.split('[,._-]+', value)
                if hierarchy_elements[len(hierarchy_elements)-1] == value:
                    hierarchy_elements = hierarchy_elements[:len(hierarchy_elements)-1]
                
                hierarchy = []
                for i in range(1, len(hierarchy_elements) + 1):
                    hierarchy.append('->'.join(hierarchy_elements[0:i]))
                hierarchy.append(value)
                
                parent = column
                for i in range(len(hierarchy)):
                    child = hierarchy[i]
                    if not parent == child: 
                        hierarchy_df = hierarchy_df.append({
                            'parent': parent,
                            'child': child,
                        }, ignore_index=True)
                    parent = child
    
        return hierarchy_df[['parent', 'child']]

    def _generate_splitted_hierarchy(self, value: str) -> List[str]:
        hierarchy = []
        hierarchy_elements = re.split('[,._-]+', value)
        for i in range(1, len(hierarchy_elements)):
            hierarchy.append('->'.join(hierarchy_elements[:i]))

        return hierarchy

    def _generate_hostname_hierarchy(self, hostname: str) -> List[str]:
        name = hostname.rstrip('0123456789')
        return [name]

    def _generate_http_hierarchy(self, http_code: str) -> List[str]:
        return [http_code[0] + 'XX']

