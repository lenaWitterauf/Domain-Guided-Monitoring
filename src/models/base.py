import tensorflow as tf
from typing import Any
from ..features.sequences import TrainTestSplit
from .metrics import MulticlassAccuracy, MulticlassFalseNegativeRate, MulticlassFalsePositiveRate, MulticlassTrueNegativeRate, MulticlassTruePositiveRate

class BaseModel():
    lstm_dim: int = 32
    n_epochs: int = 100
    prediction_model: tf.keras.Model = None
    embedding_model: tf.keras.Model = None

    def _get_embedding_layer(self, split: TrainTestSplit, knowledge: Any) -> tf.keras.Model:
        raise NotImplementedError("This should be implemented by the subclass!!!")

    def build(self, split: TrainTestSplit, knowledge: Any):
        input_layer = tf.keras.layers.Input(shape=(split.max_length, len(split.vocab)))
        embedding_layer = self._get_embedding_layer(split, knowledge)
        self.prediction_model = tf.keras.models.Sequential([
            input_layer,
            embedding_layer,
            tf.keras.layers.LSTM(self.lstm_dim),
            tf.keras.layers.Dense(len(split.vocab), activation='relu'),
        ])
        self.embedding_model = tf.keras.models.Sequential([
            input_layer,
            embedding_layer,
        ])
        
    def train(self, data: TrainTestSplit):
        self.prediction_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(), 
            optimizer=tf.optimizers.Adam(), 
            metrics=[
                MulticlassAccuracy(),
                MulticlassTrueNegativeRate(),
                MulticlassFalseNegativeRate(),
                MulticlassTruePositiveRate(),
                MulticlassFalsePositiveRate(),
            ])

        self.prediction_model.fit(
            x=data.train_x, 
            y=data.train_y, 
            validation_data=(data.test_x, data.test_y),
            epochs=self.n_epochs)
