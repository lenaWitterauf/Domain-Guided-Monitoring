import unittest
import pandas as pd
import tensorflow as tf
from typing import Dict, List
import dataclasses

from src.features.sequences.transformer import NextSequenceTransformer, TrainTestSplit
from ...test_utils import transform_to_string


class TestHandler(unittest.TestCase):
    def test_sequence_handler(self):
        sequence_df = self._load_sequence_df()
        config = TestSequenceConfig()
        config.remove_empty_x_vecs = False
        config.min_window_size = 1
        config.flatten_y = True
        config.allow_subwindows = True
        fixture = NextSequenceTransformer(config)
        split = fixture.transform_train_test_split(sequence_df, "sequence")

        self._check_vocab(split.metadata.x_vocab)
        self._check_vocab(split.metadata.y_vocab)
        self._check_split_sizes(split)
        self._check_tensors(split)
        self._check_tensor_contents(split)

    def _check_tensors(self, split: TrainTestSplit):
        combined_x = tf.concat([split.train_x, split.test_x], axis=0)
        combined_y = tf.concat([split.train_y, split.test_y], axis=0)
        combined_y = tf.expand_dims(combined_y, axis=1)

        self.assertEquals(
            combined_x.shape[0], combined_y.shape[0], 3
        )  # number of datapoints
        self.assertEquals(
            combined_x.shape[2], combined_y.shape[2], 4
        )  # number of features
        self.assertEquals(combined_x.shape[1], 2)  # max sequence length - 1

    def _check_tensor_contents(self, split: TrainTestSplit):
        combined_x = tf.concat([split.train_x, split.test_x], axis=0)
        combined_y = tf.concat([split.train_y, split.test_y], axis=0)
        combined_y = tf.expand_dims(combined_y, axis=1)

        expected_data = self._load_expected_data()
        actual_data = pd.DataFrame(
            data={
                "x": self._load_data_from_tensor(combined_x, split),
                "y": self._load_data_from_tensor(combined_y, split),
            }
        )

        expected_data["str_x"] = expected_data["x"].apply(
            lambda x: transform_to_string(x)
        )
        expected_data["str_y"] = expected_data["y"].apply(
            lambda x: transform_to_string(x)
        )
        actual_data["str_x"] = actual_data["x"].apply(lambda x: transform_to_string(x))
        actual_data["str_y"] = actual_data["y"].apply(lambda x: transform_to_string(x))

        pd.testing.assert_frame_equal(
            expected_data[["str_x", "str_y"]],
            actual_data[["str_x", "str_y"]],
            check_like=True,
        )

    def _load_data_from_tensor(self, tensor: tf.Tensor, split: TrainTestSplit):
        data = []
        for data_idx in range(tensor.shape[0]):
            visits = []
            for visit_idx in range(tensor.shape[1]):
                visit_data = []
                for multi_idx in range(tensor.shape[2]):
                    if tensor[data_idx, visit_idx, multi_idx] == 1:
                        visit_data.append(
                            [
                                name
                                for name in split.metadata.x_vocab.keys()
                                if split.metadata.x_vocab[name] == multi_idx
                            ][0]
                        )
                visits.append(visit_data)
            data.append(visits)

        return data

    def _check_split_sizes(self, split: TrainTestSplit):
        self.assertEquals(len(split.train_x), len(split.train_y))
        self.assertEquals(len(split.test_x), len(split.test_y))
        self.assertEquals(
            len(split.train_x) + len(split.test_x),
            len(split.train_y) + len(split.test_y),
            3,
        )

    def _check_vocab(self, vocab: Dict[str, int]):
        self.assertEquals(len(vocab), 4)
        self.assertSetEqual(set(["a", "b", "c", "d"]), set(vocab.keys()))
        self.assertSetEqual(set([0, 1, 2, 3]), set(vocab.values()))

    def _load_sequence_df(self):
        return pd.DataFrame(
            data={
                "sequence": [
                    [["a", "b"], ["a", "c"],],  # sequence1  # visit1  # visit2
                    [["a", "b", "c"], ["a"], ["d"],],  # sequence2
                ]
            }
        )

    def _load_expected_data(self):
        return pd.DataFrame(
            data={
                "x": [
                    [[], ["a", "b"]],
                    [[], ["a", "b", "c"]],
                    [["a", "b", "c"], ["a"]],
                    [[], ["a"]],
                ],
                "y": [[["a", "c"]], [["a"]], [["d"]], [["d"]],],
            }
        )


@dataclasses.dataclass
class TestSequenceConfig:
    test_percentage: float = 0.1  # how much of the data should be used for testing
    random_test_split: bool = True  # if true, split randomly; if false, split after 1-test_percentage datapoints
    random_state: int = 12345  # seed used for random test split
    flatten_x: bool = True  # if true, produces one mulit-hot encoded vector per timestamp;
    flatten_y: bool = False  #       if false, produces multiple (number of features in timestamp) one-hot encoded vectors per timestamp
    max_window_size: int = 10  # max number of timestamps per prediction input
    min_window_size: int = 2  # min number of timestamps per prediction input
    window_overlap: bool = True  # if true, timestamps for different prediction inputs may overlap
    allow_subwindows: bool = False  # if true, all subsequences of a given sequence are used; if false, resembles sliding window approach
    valid_y_features: List[str] = dataclasses.field(
        default_factory=lambda: [],
    )  # if not empty, only these features are used as prediction goals
    remove_empty_y_vecs: bool = True  # if true, removes (x,y) pairs where y is a zero vector
    remove_empty_x_vecs: bool = True  # if true, removes (x) inputs where x is a zero vector
    x_sequence_column_name: str = ""
    y_sequence_column_name: str = ""
