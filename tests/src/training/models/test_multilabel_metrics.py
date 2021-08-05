import unittest
import numpy as np
from numpy.lib.ufunclike import fix
import tensorflow as tf
from src.training.models import metrics


class TestMultilabelMetrics(unittest.TestCase):
    def test_multilabel(self):
        y_true = self._get_y_true_multilabel()
        y_pred = self._get_y_pred_multilabel()

        fixture = metrics.MultilabelNestedMetric(
                nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=2),
            )
        fixture.update_state(y_true, y_pred)
        self.assertEquals(4.0/6.0, fixture.result())

    def _get_y_true_multilabel(self):
        return tf.constant([[1, 1, 0], [1, 1, 0], [1, 0, 1],])

    def _get_y_pred_multilabel(self):
        return tf.constant([[0.7, 0.3, 0.1], [0.24, 0.8, 0.3], [0.1, 0.6, 0.2],])

    def _get_y_vocab(self):
        return {
            "y0": 0,
            "y1": 1,
            "y2": 2,
        }

