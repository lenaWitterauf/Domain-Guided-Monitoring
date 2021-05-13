import tensorflow as tf
from typing import Any
from ..features.sequences import TrainTestSplit
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
    prediction_model: tf.keras.Model = None
    embedding_model: tf.keras.Model = None
    embedding_layer: tf.keras.Model = None

    def _get_embedding_layer(self, split: TrainTestSplit, knowledge: Any) -> tf.keras.Model:
        raise NotImplementedError("This should be implemented by the subclass!!!")

    def build(self, split: TrainTestSplit, knowledge: Any):
        input_layer = tf.keras.layers.Input(shape=(split.max_x_length, len(split.x_vocab)))
        self.embedding_layer = self._get_embedding_layer(split, knowledge)
        self.prediction_model = tf.keras.models.Sequential([
            input_layer,
            self.embedding_layer,
            tf.keras.layers.LSTM(self.lstm_dim),
            tf.keras.layers.Dense(len(split.y_vocab), activation='relu'),
        ])
        self.embedding_model = tf.keras.models.Sequential([
            input_layer,
            self.embedding_layer,
        ])
        
    def train(self, data: TrainTestSplit):
        self.prediction_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(), 
            optimizer=tf.optimizers.Adam(), 
            metrics=[
                MulticlassAccuracy(),
                MulticlassTrueNegativeRate(),
                MulticlassTruePositiveRate(),
            ])

        self.prediction_model.fit(
            x=data.train_x, 
            y=data.train_y, 
            validation_data=(data.test_x, data.test_y),
            epochs=self.n_epochs)
