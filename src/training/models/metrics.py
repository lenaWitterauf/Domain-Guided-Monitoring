from numpy.core.numeric import full
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


class MultilabelNestedMetric(tf.keras.metrics.Metric):
    def __init__(self, nested_metric: tf.keras.metrics.Metric, full_prediction: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nested_metric = nested_metric
        self.full_prediction = full_prediction

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.full_prediction:
            y_true = tf.reshape(y_true, (tf.shape(y_true)[0] * tf.shape(y_true)[1], tf.shape(y_true)[2]))
            y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0] * tf.shape(y_pred)[1], tf.shape(y_pred)[2]))
        id_tensor = tf.eye(tf.shape(y_true)[1], dtype="int32")
        id_tensor_expanded = tf.reshape(
            tf.broadcast_to(
                tf.expand_dims(id_tensor, axis=0),
                (tf.shape(y_true)[0], tf.shape(y_true)[1], tf.shape(y_true)[1]),
            ),
            (tf.shape(y_true)[0] * tf.shape(y_true)[1], tf.shape(y_true)[1]),
        )
        cleaned_id_tensor = tf.where(
            (id_tensor_expanded == 1)
            & (tf.repeat(y_true, repeats=tf.shape(y_true)[1], axis=0) == 1),
            x=1,
            y=0,
        )

        weights = tf.reduce_sum(cleaned_id_tensor, axis=1,)
        if sample_weight is not None:
            weights = weights * tf.repeat(sample_weight, tf.shape(y_true)[1], axis=0)

        self.nested_metric.update_state(
            y_true=cleaned_id_tensor,
            y_pred=tf.repeat(y_pred, tf.shape(y_true)[1], axis=0),
            sample_weight=weights,
        )

    def result(self):
        return self.nested_metric.result()

    def reset_states(self):
        self.nested_metric.reset_states()


class SubsetMetric(tf.keras.metrics.Metric):
    def __init__(
        self,
        dataset_mask: np.array,
        nested_metric: tf.keras.metrics.Metric,
        full_prediction: bool,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dataset_mask = dataset_mask
        self.nested_metric = nested_metric
        self.full_prediction = full_prediction

    def update_state(self, y_true, y_pred, sample_weight=None):        
        if self.full_prediction:
            y_true = tf.reshape(y_true, (tf.shape(y_true)[0] * tf.shape(y_true)[1], tf.shape(y_true)[2]))
            y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0] * tf.shape(y_pred)[1], tf.shape(y_pred)[2]))
        weights = tf.reduce_sum(tf.where(self.dataset_mask, x=y_true, y=0), axis=1)
        if sample_weight is not None:
            weights = weights * sample_weight
        self.nested_metric.update_state(
            y_true, y_pred, sample_weight=weights,
        )

    def result(self):
        return self.nested_metric.result()

    def reset_states(self):
        self.nested_metric.reset_states()


class PercentileSubsetMetricHelper:
    def __init__(
        self, dataset: tf.data.Dataset, num_percentiles: int, y_vocab: Dict[str, int], full_prediction: bool,
    ):
        self.dataset = dataset
        self.num_percentiles = num_percentiles
        self.y_vocab = y_vocab
        self.full_prediction = full_prediction
        self._init_percentiles()
        self._log_percentile_mapping_to_mlflow()

    def get_accuracy_at_k_for(
        self, k, is_multilabel: bool, use_cumulative: bool
    ) -> List[tf.keras.metrics.Metric]:
        metrics = []
        for i in range(self.num_percentiles):
            name = (
                "top_"
                + str(k)
                + "_categorical_accuracy_"
                + ("cp" if use_cumulative else "p")
                + str(i)
            )
            mask = self._get_mask_for_percentile(i, use_cumulative=use_cumulative)

            metrics.append(
                self._get_accuracy_at_k_with_mask(k, is_multilabel=is_multilabel, mask=mask, name=name)
            )

        return metrics

    def _get_accuracy_at_k_with_mask(self, k, is_multilabel: bool, mask, name: str) -> tf.keras.metrics.Metric:
        if is_multilabel:
            return MultilabelNestedMetric(
                nested_metric=SubsetMetric(
                    dataset_mask=mask,
                    nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(
                        k=k, name=name
                    ),
                    full_prediction=False,
                ),
                full_prediction=self.full_prediction,
                name=name,
            )
        else:
            return SubsetMetric(
                dataset_mask=mask,
                nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(
                    k=k, name=name
                ),
                full_prediction=self.full_prediction,
                name=name,
            )


    def _get_mask_for_percentile(self, p, use_cumulative: bool):
        if use_cumulative:
            mask = np.where(
                (self.cpercentile_ranks > self.percentile_steps[p])
                & (self.cpercentile_ranks <= self.percentile_steps[p + 1]),
                True,
                False,
            )
        else:
            mask = np.where(
                (self.frequency_ranks > self.percentiles[p])
                & (self.frequency_ranks <= self.percentiles[p + 1]),
                True,
                False,
            )
        if not np.any(mask):
            logging.warn("No class labels in percentile %d", p)

        return mask

    def _init_percentiles(self):
        num_classes = len(self.y_vocab)
        absolute_class_frequencies = np.zeros(shape=(num_classes,), dtype=np.int32)
        for (_, y_true) in tqdm(
            self.dataset.as_numpy_iterator(),
            desc="Calculating percentile frequencies...",
        ):        
            if self.full_prediction:
                y_true = tf.reshape(y_true, (tf.shape(y_true)[0] * tf.shape(y_true)[1], tf.shape(y_true)[2]))
            next_sum = np.sum(y_true, axis=0,)
            absolute_class_frequencies = absolute_class_frequencies + next_sum

        self.frequencies = absolute_class_frequencies / np.sum(
            absolute_class_frequencies
        )
        self.frequency_ranks = np.empty_like(self.frequencies.argsort())
        self.frequency_ranks[self.frequencies.argsort()] = np.arange(
            len(self.frequencies)
        )
        self._init_percentile_values()
        self._init_cpercentiles()

    def _init_percentile_values(self):
        self.percentile_steps = [
            100 * i / self.num_percentiles for i in range(self.num_percentiles + 1)
        ]
        self.percentiles = np.percentile(self.frequency_ranks, self.percentile_steps)
        self.percentiles[0] = -1
        self.percentile_steps[0] = -1

    def _init_cpercentiles(self):
        sorted_frequencies = self.frequencies[self.frequencies.argsort()]
        self.cfrequencies = np.cumsum(sorted_frequencies)[self.frequency_ranks]
        self.cpercentile_ranks = (self.cfrequencies - 0.5 * self.frequencies) * 100

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
                "cpercentile_classes": [
                    name
                    for (name, idx) in self.y_vocab.items()
                    if self.cpercentile_ranks[idx] > self.percentile_steps[i]
                    and self.cpercentile_ranks[idx] <= self.percentile_steps[i + 1]
                ],
            }
        return percentile_mapping
