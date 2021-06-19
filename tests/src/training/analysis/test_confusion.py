from src.features.sequences.transformer import SequenceMetadata
import unittest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import tensorflow as tf

from src.training.analysis import confusion


class TestConfusionCalculator(unittest.TestCase):
    def test_confusion_matrix_simple(self):
        self._test_confusion_matrix(
            true_labels=np.array([[0, 0, 1], [0, 1, 0],]),
            predictions=np.array([[0.3, 0.1, 0.7], [0.9, 0.8, 0.1],]),
            expected_confusion_matrix=np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1],]),
        )

    def test_confusion_matrix_advanced(self):
        self._test_confusion_matrix(
            true_labels=np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0],]),
            predictions=np.array(
                [[0.1, 0.1, 0.9], [0.1, 0.9, 0.1], [0.9, 0.1, 0.1], [0.9, 0.1, 0.1],]
            ),
            expected_confusion_matrix=np.array([[0, 0, 0], [2, 0, 0], [0, 1, 1],]),
        )

    def test_confusion_df(self):
        self._test_confusion_df(
            true_labels=np.array([[0, 0, 1], [0, 1, 0],]),
            predictions=np.array([[0.3, 0.1, 0.7], [0.9, 0.8, 0.1],]),
            expected_confusion_df=pd.DataFrame(
                {"label0": [0, 1, 0], "label1": [0, 0, 0], "label2": [0, 0, 1],},
                index=["label0", "label1", "label2"],
            ),
        )

    def _test_confusion_matrix(
        self,
        true_labels: np.array,
        predictions: np.array,
        expected_confusion_matrix: np.array,
    ):
        dataset = tf.data.Dataset.from_tensors((true_labels, true_labels))
        model = NoopModel(predictions)
        metadata = SequenceMetadata(
            max_x_length=-1,
            max_sequence_length=-1,
            max_features_per_time=-1,
            max_features_per_sequence=-1,
            x_vocab={},
            y_vocab={("label" + str(i)): i for i in range(true_labels.shape[1])},
        )

        confusion_calculator = confusion.ConfusionCalculator(metadata, model)
        confusion_matrix = confusion_calculator._calculate_confusion_matrix_for_dataset(
            dataset
        )
        np.testing.assert_array_equal(confusion_matrix, expected_confusion_matrix)

    def _test_confusion_df(
        self,
        true_labels: np.array,
        predictions: np.array,
        expected_confusion_df: pd.DataFrame,
    ):
        dataset = tf.data.Dataset.from_tensors((true_labels, true_labels))
        model = NoopModel(predictions)
        metadata = SequenceMetadata(
            max_x_length=-1,
            max_sequence_length=-1,
            max_features_per_time=-1,
            max_features_per_sequence=-1,
            x_vocab={},
            y_vocab={("label" + str(i)): i for i in range(true_labels.shape[1])},
        )

        confusion_calculator = confusion.ConfusionCalculator(metadata, model)
        confusion_df = confusion_calculator._calculate_confusion_df_for_dataset(dataset)
        assert_frame_equal(
            confusion_df,
            expected_confusion_df,
            check_like=True,
            check_dtype=False,
        )


class NoopModel(tf.keras.Model):
    def __init__(self, expected_output: np.array):
        super(NoopModel, self).__init__()
        self.expected_output = tf.convert_to_tensor(expected_output)

    def call(self, _):
        return self.expected_output
