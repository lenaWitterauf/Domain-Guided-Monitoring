import unittest
import pandas as pd
from pathlib import Path
from typing import List
import numpy as np

from src.features.preprocessing.huawei import ConcurrentAggregatedLogsPreprocessor, ConcurrentAggregatedLogsDescriptionPreprocessor, ConcurrentAggregatedLogsHierarchyPreprocessor
from ...test_utils import transform_to_string

class TestHuawei(unittest.TestCase):
    def test_huawei_preprocessing(self):
        fixture = ConcurrentAggregatedLogsPreprocessor(
            log_file=Path('tests/resources/test_huawei_aggregated_logs.csv'),
            datetime_column_name='time',
            relevant_columns=['a', 'b'],
            max_sequence_length=2
        )

        expected_df = pd.DataFrame(
            data={
                fixture.sequence_column_name: [
                    [
                        ['a.a1', 'b.b1'],
                        ['a.a2', 'b.b2'],
                    ],
                    [
                        ['b_b2'],
                        ['a.a3', 'b'],
                    ],
                ],
            },
        )
        expected_df['str_df'] = expected_df[fixture.sequence_column_name].apply(lambda x: transform_to_string(x))
        aggregated_df = fixture.preprocess_data()
        aggregated_df['str_df'] = aggregated_df[fixture.sequence_column_name].apply(lambda x: transform_to_string(x))

        pd.testing.assert_frame_equal(
            expected_df[['str_df']],
            aggregated_df[['str_df']],
            check_like=True,
        )

    def test_huawei_descriptions(self):
        fixture = ConcurrentAggregatedLogsDescriptionPreprocessor(
            log_file=Path('tests/resources/test_huawei_aggregated_logs.csv'),
            relevant_columns=['a', 'b'],
        )

        expected_df = pd.DataFrame(
            data={
                'label': ['a.a1', 'a.a2', 'a.a3', 'b.b1', 'b.b2', 'b_b2', 'b'],
                'description': ['a a1', 'a a2', 'a a3', 'b b1', 'b b2', 'b b2', 'b'],
            },
        )
        aggregated_df = fixture.load_descriptions()
        pd.testing.assert_frame_equal(
            expected_df.sort_values(by='label', ignore_index=True),
            aggregated_df.sort_values(by='label', ignore_index=True),
            check_like=True,
        )

    def test_huawei_hierarchy(self):
        fixture = ConcurrentAggregatedLogsHierarchyPreprocessor(
            log_file=Path('tests/resources/test_huawei_aggregated_logs.csv'),
            relevant_columns=['a', 'b'],
        )

        expected_df = pd.DataFrame(
            data={
                'parent': ['root', 'root', 'a', 'a->a', 'a->a->a1', 'a->a', 'a->a->a2', 'a->a', 'a->a->a3', 'b', 'b->b', 'b->b->b1', 'b->b', 'b->b->b2', 'b->b->b2'],
                'child': ['a', 'b', 'a->a', 'a->a->a1', 'a.a1', 'a->a->a2', 'a.a2', 'a->a->a3', 'a.a3', 'b->b', 'b->b->b1', 'b.b1', 'b->b->b2', 'b.b2', 'b_b2'],
            },
        )
        aggregated_df = fixture.preprocess_hierarchy().drop_duplicates()
        print(aggregated_df)
        pd.testing.assert_frame_equal(
            expected_df.sort_values(by='child', ignore_index=True),
            aggregated_df.sort_values(by='child', ignore_index=True),
            check_like=True,
        )