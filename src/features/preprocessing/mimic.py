import dataclass_cli
import dataclasses
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from .base import Preprocessor

def _convert_to_icd9(dxStr: str):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr

def _convert_to_3digit_icd9(dxStr: str):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3]
        else: return dxStr

@dataclass_cli.add
@dataclasses.dataclass
class MimicPreprocessorConfig:
    admission_file: Path = Path('data/ADMISSIONS.csv')
    diagnosis_file: Path = Path('data/DIAGNOSES_ICD.csv')
    pkl_file: Path = Path('data/mimic_processed.pkl')
    hierarchy_file: Path = Path('data/ccs_multi_dx_tool_2015.csv')
    min_admissions_per_user: int = 2

class HierarchyPreprocessor(Preprocessor):
    hierarchy_file: Path

    def __init__(self,
            hierarchy_file=Path('data/ccs_multi_dx_tool_2015.csv')):
        self.hierarchy_file = hierarchy_file

    def load_data(self) -> pd.DataFrame:
        logging.info('Starting to preprocess ICD9 hierarchy')
        hierarchy_df = self._read_hierarchy_df()
        return self._transform_hierarchy_df(hierarchy_df)

    def _read_hierarchy_df(self) -> pd.DataFrame:
        logging.info('Reading hierarchy_df from %s', self.hierarchy_file)
        return pd.read_csv(self.hierarchy_file, quotechar='\'', dtype=str)

    def _transform_hierarchy_df(self, hierarchy_df: pd.DataFrame):
        transformed_hierarchy_df = pd.DataFrame(columns=['parent', 'child'])
        for _, row in tqdm(hierarchy_df.iterrows(), desc='Building flat hierarchy df', total=len(transformed_hierarchy_df)):
            all_parents = [
                    row['CCS LVL 1'],
                    row['CCS LVL 2'],
                    row['CCS LVL 3'],
                    row['CCS LVL 4'],
            ]
            all_parents = [str(label).strip() for label in all_parents]
            all_parents = [label for label in all_parents if len(label) > 0]
            # Labels are sorted from general -> specific

            transformed_hierarchy_df = transformed_hierarchy_df.append(pd.DataFrame(data={
                'parent': all_parents,
                'child': all_parents[1:] + [_convert_to_3digit_icd9(row['ICD-9-CM CODE'])]
            }))
        
        return transformed_hierarchy_df

class ICDDescriptionPreprocessor(Preprocessor):
    description_file: Path

    def __init__(self, description_file=Path('data/D_ICD_DIAGNOSES.csv')):
        self.description_file = description_file

    def load_data(self):
        description_df = pd.read_csv(self.description_file)
        description_df['label'] = description_df['icd9_code'].apply(_convert_to_icd9)
        description_df['description'] = description_df['long_title'].apply(lambda x: x.replace('\"', ''))
        return description_df[['label', 'description']]


class MimicPreprocessor(Preprocessor):
    admission_file: Path
    diagnosis_file: Path 
    pkl_file: Path
    min_admissions_per_user: int

    def __init__(self, 
            admission_file=Path('data/ADMISSIONS.csv'), 
            diagnosis_file=Path('data/DIAGNOSES_ICD.csv'), 
            pkl_file=Path('data/mimic_processed.pkl'),
            min_admissions_per_user=2):
        self.admission_file = admission_file
        self.diagnosis_file = diagnosis_file
        self.pkl_file = pkl_file
        self.min_admissions_per_user = min_admissions_per_user

    def load_data(self) -> pd.DataFrame:
        logging.info('Starting to preprocess MIMIC dataset')
        admission_df = self._read_admission_df()
        diagnosis_df = self._read_diagnosis_df()
        aggregated_df = self._aggregate_codes_per_admission(diagnosis_df=diagnosis_df, admission_df=admission_df)
        return aggregated_df[aggregated_df['num_admissions'] >= self.min_admissions_per_user]

    def load_from_pkl(self) -> pd.DataFrame:
        return pd.read_pickle(self.pkl_file)

    def write_to_pkl(self, aggregated_df: pd.DataFrame):
        self._write_to_pkl(aggregated_df)

    def _read_admission_df(self) -> pd.DataFrame:
        logging.info('Reading admission_df from %s', self.admission_file)
        admission_df = pd.read_csv(self.admission_file)
        admission_df['admittime']= pd.to_datetime(admission_df['admittime'])
        admission_df['dischtime']= pd.to_datetime(admission_df['dischtime'])
        admission_df['deathtime']= pd.to_datetime(admission_df['deathtime'])
        admission_df['edregtime']= pd.to_datetime(admission_df['edregtime'])
        admission_df['edouttime']= pd.to_datetime(admission_df['edouttime'])
        return admission_df

    def _read_diagnosis_df(self) -> pd.DataFrame:
        logging.info('Reading diagnosis_df from %s', self.diagnosis_file)
        diagnosis_df = pd.read_csv(self.diagnosis_file)
        diagnosis_df['icd9_code'] = diagnosis_df['icd9_code'].apply(str)
        diagnosis_df['icd9_code_converted'] = diagnosis_df['icd9_code'].apply(_convert_to_icd9)
        diagnosis_df['icd9_code_converted_3digits'] = diagnosis_df['icd9_code'].apply(_convert_to_3digit_icd9)
        return diagnosis_df

    def _aggregate_codes_per_admission(self, diagnosis_df: pd.DataFrame, admission_df: pd.DataFrame) -> pd.DataFrame:
        codes_per_admission = diagnosis_df.groupby('hadm_id').agg({
            'icd9_code': lambda x: list(x),
            'icd9_code_converted': lambda x: list(x),
            'icd9_code_converted_3digits': lambda x: list(x),
        })
        combined_df = pd.merge(admission_df, codes_per_admission, on=['hadm_id'])
        admissions_per_subject = combined_df.groupby('subject_id').agg({
            'hadm_id': lambda x: list(x),
            'admittime': lambda x: list(x),
            'diagnosis': lambda x: list(x),
            'icd9_code': lambda x: list(x),
            'icd9_code_converted': lambda x: list(x),
            'icd9_code_converted_3digits': lambda x: list(x),
        }).reset_index()
        admissions_per_subject['num_admissions'] = admissions_per_subject['hadm_id'].apply(len)
        return admissions_per_subject

    def _write_to_pkl(self, aggregated_df: pd.DataFrame):
        relevant_df = aggregated_df[aggregated_df['num_admissions'] >= self.min_admissions_per_user]
        relevant_df.to_pickle(self.pkl_file)