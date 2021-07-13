import tensorflow as tf
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm
import logging
import mlflow


class MulticlassMetric(tf.keras.metrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_positive_predictions = self.add_weight(
            name="true_positive_predictions", initializer="zeros"
        )
        self.false_positive_predictions = self.add_weight(
            name="false_positive_predictions", initializer="zeros"
        )
        self.true_negative_predictions = self.add_weight(
            name="true_negative_predictions", initializer="zeros"
        )
        self.false_negative_predictions = self.add_weight(
            name="false_negative_predictions", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)

        correct_positive_predictions = tf.where((y_pred == 1) & (y_true == 1), x=1, y=0)
        wrong_positive_predictions = tf.where((y_pred == 1) & (y_true == 0), x=1, y=0)
        correct_negative_predictions = tf.where((y_pred == 0) & (y_true == 0), x=1, y=0)
        wrong_negative_predictions = tf.where((y_pred == 0) & (y_true == 1), x=1, y=0)

        self.true_positive_predictions.assign_add(
            tf.cast(tf.reduce_sum(correct_positive_predictions), dtype="float32")
        )
        self.false_positive_predictions.assign_add(
            tf.cast(tf.reduce_sum(wrong_positive_predictions), dtype="float32")
        )
        self.true_negative_predictions.assign_add(
            tf.cast(tf.reduce_sum(correct_negative_predictions), dtype="float32")
        )
        self.false_negative_predictions.assign_add(
            tf.cast(tf.reduce_sum(wrong_negative_predictions), dtype="float32")
        )

    def result(self):
        raise NotImplementedError("This should be implemented by subclass!!!!!")

    def reset_states(self):
        self.true_positive_predictions.assign(0.0)
        self.false_positive_predictions.assign(0.0)
        self.true_negative_predictions.assign(0.0)
        self.false_negative_predictions.assign(0.0)


class MulticlassAccuracy(MulticlassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name="multiclass_accuracy", *args, **kwargs)

    def result(self):
        return (self.true_positive_predictions + self.true_negative_predictions) / (
            self.true_positive_predictions
            + self.false_positive_predictions
            + self.true_negative_predictions
            + self.false_negative_predictions
        )


class MulticlassTruePositiveRate(MulticlassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name="multiclass_true_positive_rate", *args, **kwargs)

    def result(self):
        return self.true_positive_predictions / (
            self.true_positive_predictions + self.false_negative_predictions
        )


class MulticlassTrueNegativeRate(MulticlassMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(name="multiclass_true_negative_rate", *args, **kwargs)

    def result(self):
        return self.true_negative_predictions / (
            self.true_negative_predictions + self.false_positive_predictions
        )


class SubsetMetric(tf.keras.metrics.Metric):
    def __init__(
        self,
        dataset_mask: np.array,
        nested_metric: tf.keras.metrics.Metric,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dataset_mask = dataset_mask
        self.nested_metric = nested_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.nested_metric.update_state(
            y_true,
            y_pred,
            sample_weight=tf.reduce_sum(
                tf.where(self.dataset_mask, x=y_true, y=0), axis=1
            ),
        )

    def result(self):
        return self.nested_metric.result()

    def reset_states(self):
        self.nested_metric.reset_states()


class PercentileSubsetMetricHelper:
    def __init__(
        self, dataset: tf.data.Dataset, num_percentiles: int, y_vocab: Dict[str, int]
    ):
        self.dataset = dataset
        self.num_percentiles = num_percentiles
        self.y_vocab = y_vocab
        self._init_percentiles()
        self._log_percentile_mapping_to_mlflow()

    def get_accuracy_at_k_for_percentiles(self, k) -> List[tf.keras.metrics.Metric]:
        metrics = []
        for i in range(self.num_percentiles):
            name = "top_" + str(k) + "_categorical_accuracy_p" + str(i)
            mask = np.where(
                (self.frequency_ranks > self.percentiles[i])
                & (self.frequency_ranks <= self.percentiles[i + 1]),
                True,
                False,
            )
            if not np.any(mask):
                logging.warn("No class labels in percentile %d", i)
                continue

            metrics.append(
                SubsetMetric(
                    dataset_mask=mask,
                    nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(
                        k=k, name=name
                    ),
                    name=name,
                )
            )

        return metrics

    def _init_percentiles(self):
        num_classes = len(self.y_vocab)
        absolute_class_frequencies = np.zeros(shape=(num_classes,), dtype=np.int32)
        for (_, y_true) in tqdm(
            self.dataset.as_numpy_iterator(),
            desc="Calculating percentile frequencies...",
        ):
            next_sum = np.sum(y_true, axis=0,)
            absolute_class_frequencies = absolute_class_frequencies + next_sum

        self.frequencies = absolute_class_frequencies / np.sum(
            absolute_class_frequencies
        )
        print(self.frequencies)
        sorted_frequencies = self.frequencies.argsort()
        self.frequency_ranks = np.empty_like(sorted_frequencies)
        self.frequency_ranks[sorted_frequencies] = np.arange(len(self.frequencies))
        self.percentile_steps = [
            100 * i / self.num_percentiles for i in range(self.num_percentiles + 1)
        ]
        self.percentiles = np.percentile(self.frequency_ranks, self.percentile_steps)
        self.percentiles[0] = -1

    def _log_percentile_mapping_to_mlflow(self):
        percentile_mapping = self._create_percentile_mapping()
        mlflow.log_dict(percentile_mapping, "percentile_mapping.json")

    def _create_percentile_mapping(self) -> Dict[int, Any]:
        percentile_mapping = {}
        for i in range(self.num_percentiles):
            percentile_mapping[i] = {
                "percentile_steps": [
                    self.percentile_steps[i],
                    self.percentile_steps[i + 1],
                ],
                "percentile_values": [self.percentiles[i], self.percentiles[i + 1]],
                "percentile_classes": [
                    name
                    for (name, idx) in self.y_vocab.items()
                    if self.frequency_ranks[idx] > self.percentiles[i]
                    and self.frequency_ranks[idx] <= self.percentiles[i + 1]
                ],
            }
        return percentile_mapping
