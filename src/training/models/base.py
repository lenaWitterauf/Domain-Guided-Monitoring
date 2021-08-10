from tensorflow.python.keras.layers.core import Dropout, Masking
from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
from typing import Any, List, Dict
from .metrics import (
    MulticlassAccuracy,
    MulticlassTrueNegativeRate,
    MulticlassTruePositiveRate,
    PercentileSubsetMetricHelper,
    MultilabelNestedMetric,
)
from .config import ModelConfig
from .callbacks import MLFlowCallback, BestModelRestoreCallback
from .initializers import FastTextInitializer
import logging
import mlflow
import datetime

def full_prediction_binary_accuracy_loss(y_true, y_pred):
    sum = tf.reduce_sum(y_true, axis=-1)
    weights = tf.where(sum>1, x=1., y=sum)
    weights = tf.cast(weights, dtype='float32')
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    loss = tf.reduce_mean(weights * loss, axis=-1)
    return tf.reduce_sum(loss)

class BaseEmbedding:
    config: ModelConfig
    num_features: int = 0
    num_hidden_features: int = 0
    num_connections: int = 0

    basic_feature_embeddings: tf.Variable  # shape: (num_features, embedding_size)
    basic_hidden_embeddings: tf.Variable  # shape: (num_hidden_features, embedding_size)

    def _final_embedding_matrix(self):
        """Overwrite this in case embedding uses attention mechanism etc"""
        return self.basic_feature_embeddings

    def _get_kernel_regularizer(self, scope: str):
        if scope not in self.config.kernel_regularizer_scope:
            logging.debug("Regularization not enabled for %s", scope)
            return None
        elif self.config.kernel_regularizer_value <= 0.0:
            return None
        elif self.config.kernel_regularizer_type == "l2":
            return tf.keras.regularizers.l2(self.config.kernel_regularizer_value)
        elif self.config.kernel_regularizer_type == "l2":
            return tf.keras.regularizers.l1(self.config.kernel_regularizer_value)
        else:
            return None

    def _get_initializer(
        self,
        initializer_name: str,
        initializer_seed: int,
        description_vocab: Dict[int, str],
    ) -> tf.keras.initializers.Initializer:
        if initializer_name == "random_uniform":
            return tf.keras.initializers.GlorotUniform(seed=initializer_seed)
        elif initializer_name == "random_normal":
            return tf.keras.initializers.GlorotNormal(seed=initializer_seed)
        elif initializer_name == "fasttext":
            initializer = FastTextInitializer(self.config.embedding_dim)
            return initializer.get_initializer(description_vocab)
        else:
            logging.error("Unknown initializer %s", initializer_name)

    def _get_feature_initializer(
        self, description_vocab: Dict[int, str]
    ) -> tf.keras.initializers.Initializer:
        return self._get_initializer(
            self.config.feature_embedding_initializer,
            self.config.feature_embedding_initializer_seed,
            description_vocab,
        )

    def _get_hidden_initializer(
        self, description_vocab: Dict[int, str]
    ) -> tf.keras.initializers.Initializer:
        return self._get_initializer(
            self.config.hidden_embedding_initializer,
            self.config.hidden_embedding_initializer_seed,
            description_vocab,
        )


class BaseModel:
    def __init__(self):
        self.prediction_model: tf.keras.Model = None
        self.embedding_layer: BaseEmbedding = None
        self.metrics: List[tf.keras.metrics.Metric] = []
        self.config = ModelConfig()

    def _get_embedding_layer(
        self, metadata: SequenceMetadata, knowledge: Any
    ) -> BaseEmbedding:
        raise NotImplementedError("This should be implemented by the subclass!!!")

    def _select_distribute_strategy(self) -> tf.distribute.Strategy:
        if self.config.distribute_strategy == "mirrored":
            return tf.distribute.MirroredStrategy()
        elif self.config.distribute_strategy.startswith("/gpu"):
            return tf.distribute.OneDeviceStrategy(
                device=self.config.distribute_strategy
            )
        else:
            return tf.distribute.get_strategy()

    def build(self, metadata: SequenceMetadata, knowledge: Any):
        self.metadata = metadata
        self.strategy = self._select_distribute_strategy()
        logging.info(
            "Using strategy with %d workers", self.strategy.num_replicas_in_sync
        )

        with self.strategy.scope():
            self.embedding_layer = self._get_embedding_layer(metadata, knowledge)
            self._log_embedding_stats()
            self.prediction_model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(
                        shape=(metadata.max_x_length, len(metadata.x_vocab))
                    ),
                    self.embedding_layer,
                    tf.keras.layers.Masking(mask_value=0),
                    self._get_rnn_layer(),
                    tf.keras.layers.Dropout(
                        rate=self.config.dropout_rate, seed=self.config.dropout_seed
                    ),
                    tf.keras.layers.Dense(
                        len(metadata.y_vocab),
                        activation=self.config.final_activation_function,
                        kernel_regularizer=self.embedding_layer._get_kernel_regularizer(
                            scope="prediction_dense"
                        ),
                    ),
                ]
            )

    def _log_embedding_stats(self):
        mlflow.log_metric("num_features", self.embedding_layer.num_features)
        mlflow.log_metric(
            "num_hidden_features", self.embedding_layer.num_hidden_features
        )
        mlflow.log_metric("num_connections", self.embedding_layer.num_connections)

    def _get_rnn_layer(self):
        if self.config.rnn_type == "rnn":
            return tf.keras.layers.SimpleRNN(
                units=self.config.rnn_dim,
                kernel_regularizer=self.embedding_layer._get_kernel_regularizer(
                    scope="prediction_rnn"
                ),
                return_sequences=self.metadata.full_y_prediction,
            )
        elif self.config.rnn_type == "lstm":
            return tf.keras.layers.LSTM(
                units=self.config.rnn_dim,
                kernel_regularizer=self.embedding_layer._get_kernel_regularizer(
                    scope="prediction_rnn"
                ),
                return_sequences=self.metadata.full_y_prediction,
            )
        elif self.config.rnn_type == "gru":
            return tf.keras.layers.GRU(
                units=self.config.rnn_dim,
                kernel_regularizer=self.embedding_layer._get_kernel_regularizer(
                    scope="prediction_rnn"
                ),
                return_sequences=self.metadata.full_y_prediction,
            )
        else:
            logging.error("Unknown rnn layer type: %s", self.config.rnn_type)

    def train_dataset(
        self,
        train_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        multilabel_classification: bool,
        n_epochs: int,
    ):
        with self.strategy.scope():
            if self.metadata.full_y_prediction:
                self._compile_full_prediction(train_dataset)
            elif len(self.metadata.y_vocab) == 1:
                self._compile_singleclass()
            elif multilabel_classification:
                self._compile_multilabel(train_dataset)
            else:
                self._compile_multiclass(train_dataset)

            model_summary = []
            self.prediction_model.summary(print_fn=lambda x: model_summary.append(x))
            mlflow.log_text("\n".join(model_summary), artifact_file="model_summary.txt")

            self.history = self.prediction_model.fit(
                train_dataset,
                validation_data=test_dataset,
                epochs=n_epochs,
                callbacks=[
                    MLFlowCallback(),
                    BestModelRestoreCallback(
                        metric=self.config.best_model_metric,
                        minimize=self.config.best_model_metric_minimize,
                        early_stopping_epochs=self.config.early_stopping_epochs,
                    ),
                ],
            )

    def _compile_singleclass(self):
        self.metrics = [
            tf.keras.metrics.Accuracy(),
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
        ]
        self.prediction_model.compile(
            loss=self.config.loss, optimizer=tf.optimizers.Adam(), metrics=self.metrics,
        )

    def _compile_full_prediction(self, train_dataset: tf.data.Dataset):
        self.metrics = [
            MultilabelNestedMetric(
                nested_metric=tf.keras.metrics.CategoricalAccuracy(),
                name="categorical_accuracy",
                full_prediction=self.metadata.full_y_prediction,
            ),
            MultilabelNestedMetric(
                nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=5),
                name="top_5_categorical_accuracy",
                full_prediction=self.metadata.full_y_prediction,
            ),
            MultilabelNestedMetric(
                nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=10),
                name="top_10_categorical_accuracy",
                full_prediction=self.metadata.full_y_prediction,
            ),
            MultilabelNestedMetric(
                nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=20),
                name="top_20_categorical_accuracy",
                full_prediction=self.metadata.full_y_prediction,
            ),
        ]
        metric_helper = PercentileSubsetMetricHelper(
            train_dataset,
            num_percentiles=self.config.metrics_num_percentiles,
            y_vocab=self.metadata.y_vocab,
            full_prediction=self.metadata.full_y_prediction,
        )
        for k in [5, 10, 20]:
            self.metrics = (
                self.metrics
                + metric_helper.get_accuracy_at_k_for(
                    k=k, is_multilabel=True, use_cumulative=True
                )
                + metric_helper.get_accuracy_at_k_for(
                    k=k, is_multilabel=True, use_cumulative=False
                )
            )

        self.prediction_model.compile(
            loss=full_prediction_binary_accuracy_loss, optimizer=tf.optimizers.Adam(), metrics=self.metrics,
        )

    def _compile_multilabel(self, train_dataset: tf.data.Dataset):
        self.metrics = [
            MultilabelNestedMetric(
                nested_metric=tf.keras.metrics.CategoricalAccuracy(),
                name="categorical_accuracy",
                full_prediction=self.metadata.full_y_prediction,
            ),
            MultilabelNestedMetric(
                nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=5),
                name="top_5_categorical_accuracy",
                full_prediction=self.metadata.full_y_prediction,
            ),
            MultilabelNestedMetric(
                nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=10),
                name="top_10_categorical_accuracy",
                full_prediction=self.metadata.full_y_prediction,
            ),
            MultilabelNestedMetric(
                nested_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=20),
                name="top_20_categorical_accuracy",
                full_prediction=self.metadata.full_y_prediction,
            ),
        ]
        metric_helper = PercentileSubsetMetricHelper(
            train_dataset,
            num_percentiles=self.config.metrics_num_percentiles,
            y_vocab=self.metadata.y_vocab,
            full_prediction=self.metadata.full_y_prediction,
        )
        for k in [5, 10, 20]:
            self.metrics = (
                self.metrics
                + metric_helper.get_accuracy_at_k_for(
                    k=k, is_multilabel=True, use_cumulative=True
                )
                + metric_helper.get_accuracy_at_k_for(
                    k=k, is_multilabel=True, use_cumulative=False
                )
            )

        self.prediction_model.compile(
            loss=self.config.loss, optimizer=tf.optimizers.Adam(), metrics=self.metrics,
        )

    def _compile_multiclass(self, train_dataset: tf.data.Dataset):
        metric_helper = PercentileSubsetMetricHelper(
            train_dataset,
            num_percentiles=self.config.metrics_num_percentiles,
            y_vocab=self.metadata.y_vocab,
            full_prediction=self.metadata.full_y_prediction,
        )
        self.metrics = [
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=5, name="top_5_categorical_accuracy"
            ),
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=10, name="top_10_categorical_accuracy"
            ),
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=20, name="top_20_categorical_accuracy"
            ),
        ]
        for k in [5, 10, 20]:
            self.metrics = (
                self.metrics
                + metric_helper.get_accuracy_at_k_for(
                    k=k, is_multilabel=False, use_cumulative=True
                )
                + metric_helper.get_accuracy_at_k_for(
                    k=k, is_multilabel=False, use_cumulative=False
                )
            )

        self.prediction_model.compile(
            loss=self.config.loss, optimizer=tf.optimizers.Adam(), metrics=self.metrics,
        )

