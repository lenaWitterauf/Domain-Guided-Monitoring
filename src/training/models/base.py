from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
from typing import Any, List, Dict
from .metrics import (
    MulticlassAccuracy,
    MulticlassTrueNegativeRate,
    MulticlassTruePositiveRate,
)
from .config import ModelConfig
from .callbacks import MLFlowCallback, BestModelRestoreCallback
from .initializers import FastTextInitializer
import logging
import mlflow


class BaseEmbedding:
    config: ModelConfig
    num_features: int
    num_hidden_features: int
    num_connections: int

    basic_feature_embeddings: tf.Variable  # shape: (num_features, embedding_size)
    basic_hidden_embeddings: tf.Variable  # shape: (num_hidden_features, embedding_size)

    embedding_mask: tf.Variable  # shape: (num_features, num_all_features, 1)

    def _final_embedding_matrix(self):
        """Overwrite this in case embedding uses attention mechanism etc"""
        return self.basic_feature_embeddings

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
        self.embedding_layer: tf.keras.Model = None
        self.metrics: List[tf.keras.metrics.Metric] = []
        self.config = ModelConfig()

    def _get_embedding_layer(
        self, metadata: SequenceMetadata, knowledge: Any
    ) -> tf.keras.Model:
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
                    self._get_rnn_layer(),
                    tf.keras.layers.Dense(len(metadata.y_vocab), activation="relu"),
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
            return tf.keras.layers.SimpleRNN(units=self.config.rnn_dim)
        elif self.config.rnn_type == "lstm":
            return tf.keras.layers.LSTM(units=self.config.rnn_dim)
        elif self.config.rnn_type == "gru":
            return tf.keras.layers.GRU(units=self.config.rnn_dim)
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
            if multilabel_classification:
                self._compile_multilabel()
            else:
                self._compile_multiclass()

            self.history = self.prediction_model.fit(
                train_dataset,
                validation_data=test_dataset,
                epochs=n_epochs,
                callbacks=[
                    MLFlowCallback(),
                    BestModelRestoreCallback(
                        metric=self.config.best_model_metric,
                        minimize=self.config.best_model_metric_minimize,
                    ),
                ],
            )

    def _compile_multilabel(self):
        self.metrics = [
            MulticlassAccuracy(),
            MulticlassTrueNegativeRate(),
            MulticlassTruePositiveRate(),
        ]
        self.prediction_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.optimizers.Adam(),
            metrics=self.metrics,
        )

    def _compile_multiclass(self):
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
        self.prediction_model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.optimizers.Adam(),
            metrics=self.metrics,
        )

