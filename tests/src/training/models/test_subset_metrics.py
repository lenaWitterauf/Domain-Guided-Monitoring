import unittest
import numpy as np
from numpy.lib.ufunclike import fix
import tensorflow as tf
from src.training.models import metrics


class TestSubsetMetrics(unittest.TestCase):
    def test_subset_metric(self):
        y_true = self._get_y_true()
        y_pred = self._get_y_pred()

        fixture = metrics.SubsetMetric(
            dataset_mask=np.array([True, False, True]),
            nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=2),
        )
        fixture.update_state(y_true, y_pred)
        self.assertEquals(0.5, fixture.result())

    def test_percentiles(self):
        y_true = self._get_y_true()
        y_vocab = self._get_y_vocab()

        fixture = metrics.PercentileSubsetMetricHelper(
            dataset=tf.data.Dataset.from_tensors([y_true, y_true]),
            num_percentiles=2,
            y_vocab=y_vocab,
        )

        percentile_mapping = fixture._create_percentile_mapping()
        self.assertDictEqual(
            percentile_mapping,
            {
                0: {
                    "percentile_steps": [0.0, 50.0],
                    "percentile_values": [-1.0, 1.0],
                    "percentile_classes": ["y1", "y2",],
                },
                1: {
                    "percentile_steps": [50.0, 100.0],
                    "percentile_values": [1.0, 2.0],
                    "percentile_classes": ["y0",],
                },
            },
        )

    def _get_y_true(self):
        return tf.constant([[1, 0, 0], [0, 1, 0], [1, 0, 0],])

    def _get_y_pred(self):
        return tf.constant([[0.7, 0.3, 0.1], [0.24, 0.8, 0.3], [0.1, 0.6, 0.2],])

    def _get_y_vocab(self):
        return {
            "y0": 0,
            "y1": 1,
            "y2": 2,
        }

