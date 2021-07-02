from src.features.sequences.transformer import SequenceMetadata
import unittest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import tensorflow as tf

from src.training.analysis import predictions


class TestPredictionOutputCalculator(unittest.TestCase):
    def test_prediction_output(self):
        x = [[[0, 1, 1], [0, 1, 0]], [[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
        y = [[0, 1], [0, 1], [1, 0]]
        y_pred = [[0.1, 0.2], [0.2, 0.1], [0.1, 0.2]]
        x_vocab = {
            "x0": 0,
            "x1": 1,
            "x2": 2,
        }
        y_vocab = {
            "y0": 0,
            "y1": 1,
        }

        model = NoopModel(expected_output=y_pred)
        dataset = tf.data.Dataset.from_tensors((x, y))
        expected_df = pd.DataFrame(
            {
                "input": [{0: ["x1", "x2"], 1: ["x1"]}, {0: ["x0"]}, {},],
                "output": [["y1"], ["y1"], ["y0"],],
                "output_rank": [[1], [0], [0],],
            }
        )

        fixture = predictions.PredictionOutputCalculator(
            metadata=SequenceMetadata(
                max_x_length=-1,
                max_sequence_length=-1,
                max_features_per_time=-1,
                max_features_per_sequence=-1,
                x_vocab=x_vocab,
                y_vocab=y_vocab,
            ),
            model=model,
        )
        df = fixture._calculate_prediction_output_for_dataset(dataset)
        assert_frame_equal(expected_df, df)


class NoopModel(tf.keras.Model):
    def __init__(self, expected_output: np.array):
        super(NoopModel, self).__init__()
        self.expected_output = tf.convert_to_tensor(expected_output)

    def call(self, _):
        return self.expected_output
