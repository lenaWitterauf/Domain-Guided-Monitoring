import unittest
import pandas as pd
import tensorflow as tf
from typing import Dict

from src.features.sequences.transformer import NextSequenceTransformer, SplittedSequence
from ...test_utils import transform_to_string

class TestHandler(unittest.TestCase):
    def test_sequence_handler(self):
        sequence_df = self._load_sequence_df()
        fixture = NextSequenceTransformer(flatten_x=True)
        split = fixture.transform_train_test_split(sequence_df, 'sequence')

        self._check_vocab(split.x_vocab)
        self._check_vocab(split.y_vocab)
        self._check_split_sizes(split)
        self._check_tensors(split)
        self._check_tensor_contents(split)

    def _check_tensors(self, split: SplittedSequence):
        combined_x = tf.concat([split.train_x, split.test_x], axis=0)
        combined_y = tf.concat([split.train_y, split.test_y], axis=0)

        self.assertEquals(combined_x.shape[0], combined_y.shape[0], 3) # number of datapoints
        self.assertEquals(combined_x.shape[2], combined_y.shape[2], 4) # number of features
        self.assertEquals(combined_x.shape[1], 2) # max sequence length - 1
        self.assertEquals(combined_y.shape[1], 1) # y dim predicts only first next visit

    def _check_tensor_contents(self, split: SplittedSequence):
        combined_x = tf.concat([split.train_x, split.test_x], axis=0)
        combined_y = tf.concat([split.train_y, split.test_y], axis=0)
        expected_data = self._load_expected_data()
        actual_data = pd.DataFrame(data={
            'x': self._load_data_from_tensor(combined_x, split),
            'y': self._load_data_from_tensor(combined_y, split),
        })

        expected_data['str_x'] = expected_data['x'].apply(lambda x: transform_to_string(x))
        expected_data['str_y'] = expected_data['y'].apply(lambda x: transform_to_string(x))
        actual_data['str_x'] = actual_data['x'].apply(lambda x: transform_to_string(x))
        actual_data['str_y'] = actual_data['y'].apply(lambda x: transform_to_string(x))

        print(expected_data)
        print(actual_data)
        pd.testing.assert_frame_equal(
            expected_data[['str_x', 'str_y']], 
            actual_data[['str_x', 'str_y']],
            check_like=True,
        )
        
        
    def _load_data_from_tensor(self, tensor: tf.Tensor, split: SplittedSequence):
        data = []
        for data_idx in range(tensor.shape[0]):
            visits = []
            for visit_idx in range(tensor.shape[1]):
                visit_data = []
                for multi_idx in range(tensor.shape[2]):
                    if tensor[data_idx, visit_idx, multi_idx] == 1:
                        visit_data.append([name for name in split.x_vocab.keys() if split.x_vocab[name] == multi_idx][0])
                visits.append(visit_data)
            data.append(visits)

        return data


    def _check_split_sizes(self, split: SplittedSequence):
        self.assertEquals(len(split.train_x), len(split.train_y))
        self.assertEquals(len(split.test_x), len(split.test_y))
        self.assertEquals(
            len(split.train_x) + len(split.test_x), 
            len(split.train_y) + len(split.test_y),
            3,
        )


    def _check_vocab(self, vocab: Dict[str, int]):
        self.assertEquals(len(vocab), 4)
        self.assertSetEqual(set(['a', 'b', 'c', 'd']), set(vocab.keys()))
        self.assertSetEqual(set([0, 1, 2, 3]), set(vocab.values()))

    def _load_sequence_df(self):
        return pd.DataFrame(data={
            'sequence': [
                [ # sequence1
                    ['a', 'b'], # visit1
                    ['a', 'c'], # visit2
                ], 
                [ # sequence2
                    ['a', 'b', 'c'],
                    ['a'],
                    ['d'],
                ],
            ]
        })

    def _load_expected_data(self):
        return pd.DataFrame(data={
            'x': [
                [[], ['a', 'b']],
                [[], ['a', 'b', 'c']],
                [['a', 'b', 'c'], ['a']],
                [[], ['a']],
            ],
            'y': [
                [['a', 'c']],
                [['a']],
                [['d']],
                [['d']],
            ],
        })