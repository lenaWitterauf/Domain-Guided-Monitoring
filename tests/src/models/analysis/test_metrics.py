import unittest
import numpy as np

from src.training.models import metrics 

class TestMulticlassMetrics(unittest.TestCase):
    def test_base_metric(self):
        y_true = self._get_y_true()
        y_pred = self._get_y_pred()

        fixture = metrics.MulticlassMetric()
        self.assertEquals(0.0, fixture.true_positive_predictions)
        self.assertEquals(0.0, fixture.true_negative_predictions)
        self.assertEquals(0.0, fixture.false_positive_predictions)
        self.assertEquals(0.0, fixture.false_negative_predictions)

        fixture.update_state(y_true, y_pred)
        self.assertEquals(3.0, fixture.true_positive_predictions)
        self.assertEquals(2.0, fixture.true_negative_predictions)
        self.assertEquals(0.0, fixture.false_positive_predictions)
        self.assertEquals(1.0, fixture.false_negative_predictions)

        fixture.update_state(y_true, y_pred)
        self.assertEquals(6.0, fixture.true_positive_predictions)
        self.assertEquals(4.0, fixture.true_negative_predictions)
        self.assertEquals(0.0, fixture.false_positive_predictions)
        self.assertEquals(2.0, fixture.false_negative_predictions)

        fixture.reset_states()
        fixture.update_state(y_true, y_pred)
        self.assertEquals(3.0, fixture.true_positive_predictions)
        self.assertEquals(2.0, fixture.true_negative_predictions)
        self.assertEquals(0.0, fixture.false_positive_predictions)
        self.assertEquals(1.0, fixture.false_negative_predictions)


    def test_accuracy(self):
        fixture = metrics.MulticlassAccuracy()
        y_true = self._get_y_true()
        y_pred = self._get_y_pred()
        fixture.update_state(y_true, y_pred)

        self.assertEquals(5/6, fixture.result())

    def test_true_negative_rate(self):
        fixture = metrics.MulticlassTrueNegativeRate()
        y_true = self._get_y_true()
        y_pred = self._get_y_pred()
        fixture.update_state(y_true, y_pred)

        self.assertEquals(1.0, fixture.result())

    def test_true_positive_rate(self):
        fixture = metrics.MulticlassTruePositiveRate()
        y_true = self._get_y_true()
        y_pred = self._get_y_pred()
        fixture.update_state(y_true, y_pred)

        self.assertEquals(3/4, fixture.result())

    def _get_y_true(self):
        return np.array([
            [1, 0],
            [0, 1],
            [1, 1],
        ])

    def _get_y_pred(self):
        return np.array([
            [0.7, 0.3],
            [0.24, 0.8],
            [0.1, 0.6],
        ])
