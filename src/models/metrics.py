import tensorflow as tf
import numpy as np

class MulticlassMetric(tf.keras.metrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_positive_predictions = self.add_weight(name='true_positive_predictions', initializer='zeros')
        self.false_positive_predictions = self.add_weight(name='false_positive_predictions', initializer='zeros')
        self.true_negative_predictions = self.add_weight(name='true_negative_predictions', initializer='zeros')
        self.false_negative_predictions = self.add_weight(name='false_negative_predictions', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)

        correct_positive_predictions = tf.where((y_pred == 1) & (y_true == 1), x=1, y=0)
        wrong_positive_predictions = tf.where((y_pred == 1) & (y_true == 0), x=1, y=0)
        correct_negative_predictions = tf.where((y_pred == 0) & (y_true == 0), x=1, y=0)
        wrong_negative_predictions = tf.where((y_pred == 0) & (y_true == 1), x=1, y=0)

        self.true_positive_predictions.assign_add(tf.cast(tf.reduce_sum(correct_positive_predictions), dtype='float32'))
        self.false_positive_predictions.assign_add(tf.cast(tf.reduce_sum(wrong_positive_predictions), dtype='float32'))
        self.true_negative_predictions.assign_add(tf.cast(tf.reduce_sum(correct_negative_predictions), dtype='float32'))
        self.false_negative_predictions.assign_add(tf.cast(tf.reduce_sum(wrong_negative_predictions), dtype='float32'))

    def result(self):
        raise NotImplementedError("This should be implemented by subclass!!!!!")

    def reset_states(self):
        self.true_positive_predictions.assign(0.0)
        self.false_positive_predictions.assign(0.0)
        self.true_negative_predictions.assign(0.0)
        self.false_negative_predictions.assign(0.0)

class MulticlassAccuracy(MulticlassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name='MulticlassAccuracy', *args, **kwargs)

    def result(self):
        return (self.true_positive_predictions + self.true_negative_predictions) / (self.true_positive_predictions + self.false_positive_predictions + self.true_negative_predictions + self.false_negative_predictions)

class MulticlassTruePositiveRate(MulticlassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name='MulticlassTruePositiveRate', *args, **kwargs)

    def result(self):
        return self.true_positive_predictions / (self.true_positive_predictions + self.false_positive_predictions)

class MulticlassFalsePositiveRate(MulticlassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name='MulticlassFalsePositiveRate', *args, **kwargs)

    def result(self):
        return self.false_positive_predictions / (self.true_positive_predictions + self.false_positive_predictions)

class MulticlassTrueNegativeRate(MulticlassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name='MulticlassTrueNegativeRate', *args, **kwargs)

    def result(self):
        return self.true_negative_predictions / (self.true_negative_predictions + self.false_negative_predictions)

class MulticlassFalseNegativeRate(MulticlassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name='MulticlassFalseNegativeRate', *args, **kwargs)

    def result(self):
        return self.false_negative_predictions / (self.true_negative_predictions + self.false_negative_predictions)