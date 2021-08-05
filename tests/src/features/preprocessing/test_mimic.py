import unittest
import pandas as pd
from pathlib import Path
from typing import List

from src.features.preprocessing.mimic import MimicPreprocessor
from ...test_utils import transform_to_string

class TestMimic(unittest.TestCase):
    def test_mimic_preprocessing(self):
        fixture = MimicPreprocessor(
            config=TestMimicPreprocessorConfig(),
        )

        expected_aggregated_visits = pd.DataFrame(
            data={
                'subject_id': [1,2],
                'icd9_code': [
                    [ # subject_id=1
                        [111, 112, 113], # admission_id=1
                        [121, 122, 123], # admission_id=2
                        [131, 132, 133], # admission_id=3
                    ],
                    [ # subject_id=2
                        [210, 111, 112, 113], # admission_id=4
                        [121, 122, 123], # admission_id=5
                    ],
                ],
            },
        )
        expected_aggregated_visits['str_visits'] = expected_aggregated_visits['icd9_code'].apply(lambda x: transform_to_string(x))
        aggregated_df = fixture.load_data()
        aggregated_df['str_visits'] = aggregated_df['icd9_code'].apply(lambda x: transform_to_string(x))

        print(expected_aggregated_visits[['subject_id', 'str_visits']])
        print(aggregated_df[['subject_id', 'str_visits']])
        pd.testing.assert_frame_equal(
            expected_aggregated_visits[['subject_id', 'str_visits']],
            aggregated_df[['subject_id', 'str_visits']],
            check_like=True,
        )

class TestMimicPreprocessorConfig:
    admission_file: Path = Path('../tests/resources/test_mimic_admissions.csv')
    diagnosis_file: Path = Path('../tests/resources/test_mimic_diagnoses.csv')
    hierarchy_file: Path = Path("data/ccs_multi_dx_tool_2015.csv")
    icd9_file: Path = Path("data/icd9.csv")
    use_icd9_data: bool = True
    min_admissions_per_user: int = 2
    sequence_column_name: str = "icd9_code_converted_3digits"
    add_icd9_info_to_sequences: bool = False
    knowlife_file: Path = Path("data/knowlife_dump.tsv")
    umls_file: Path = Path("data/umls.csv")
    umls_api_key: str = ""
    replace_keys: List[str] = []
    replace_with_keys: List[str] = []
    replacement_percentages: List[float] = []
    replace_columns: List[str] = []