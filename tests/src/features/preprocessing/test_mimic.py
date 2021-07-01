import unittest
import pandas as pd
from pathlib import Path

from src.features.preprocessing.mimic import MimicPreprocessor
from ...test_utils import transform_to_string

class TestMimic(unittest.TestCase):
    def test_mimic_preprocessing(self):
        fixture = MimicPreprocessor(
            admission_file=Path('tests/resources/test_mimic_admissions.csv'),
            diagnosis_file=Path('tests/resources/test_mimic_diagnoses.csv'),
            add_icd9_info_to_sequences=False,
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