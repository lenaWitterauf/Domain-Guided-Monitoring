from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
from typing import Any
from .analysis.metrics import MulticlassAccuracy, MulticlassTrueNegativeRate, MulticlassTruePositiveRate

class BaseEmbedding:
    embedding_size: int
    num_features: int
    num_hidden_features: int

    basic_feature_embeddings: tf.Variable # shape: (num_features, embedding_size)
    basic_hidden_embeddings: tf.Variable # shape: (num_hidden_features, embedding_size)
    
    embedding_mask: tf.Variable # shape: (num_features, num_all_features, 1)

    def _final_embedding_matrix(self):
        """Overwrite this in case embedding uses attention mechanism etc"""
        return self.basic_feature_embeddings

class BaseModel:
    lstm_dim: int = 32
    n_epochs: int = 100

    def __init__(self):
        self.prediction_model: tf.keras.Model = None
        self.embedding_layer: tf.keras.Model = None
        self.metrics: List[tf.keras.metrics.Metric] = []

    def _get_embedding_layer(self, metadata: SequenceMetadata, knowledge: Any) -> tf.keras.Model:
        raise NotImplementedError("This should be implemented by the subclass!!!")

    def build(self, metadata: SequenceMetadata, knowledge: Any):
        input_layer = tf.keras.layers.Input(shape=(metadata.max_x_length, len(metadata.x_vocab)))
        self.embedding_layer = self._get_embedding_layer(metadata, knowledge)
        self.prediction_model = tf.keras.models.Sequential([
            input_layer,
            self.embedding_layer,
            tf.keras.layers.LSTM(self.lstm_dim),
            tf.keras.layers.Dense(len(metadata.y_vocab), activation='relu'),
        ])
        
    def train_dataset(self, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, multilabel_classification: bool):
        if multilabel_classification:
            self._compile_multilabel()
        else: 
            self._compile_multiclass()
        
        self.history = self.prediction_model.fit(
            train_dataset, 
            validation_data=test_dataset,
            epochs=self.n_epochs)

    def _compile_multilabel(self):
        self.metrics = [
            MulticlassAccuracy(),
            MulticlassTrueNegativeRate(),
            MulticlassTruePositiveRate(),
        ]
        self.prediction_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(), 
            optimizer=tf.optimizers.Adam(), 
            metrics=self.metrics)

    def _compile_multiclass(self):
        self.metrics = [
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_categorical_accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy'),
        ]
        self.prediction_model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(), 
            optimizer=tf.optimizers.Adam(), 
            metrics=self.metrics)