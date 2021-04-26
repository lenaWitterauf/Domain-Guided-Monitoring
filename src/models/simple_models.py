import tensorflow as tf
from ..features.sequences import TrainTestSplit
from typing import Dict, Tuple, List, Any
import dataclass_cli
import dataclasses
import logging

@dataclass_cli.add
@dataclasses.dataclass
class SimpleLSTMModel:
    embedding_dim: int = 64
    lstm_dim: int = 32
    n_epochs: int = 100

    prediction_model: tf.keras.Model = None
    embedding_model: tf.keras.Model = None

    def build(self, max_length: int, vocab_size: int):
        input_layer = tf.keras.layers.Input(shape=(max_length, vocab_size))
        embedding_layer = tf.keras.layers.Dense(self.embedding_dim)
        self.lstm_model = tf.keras.models.Sequential([
            input_layer,
            embedding_layer,
            tf.keras.layers.LSTM(self.lstm_dim),
            tf.keras.layers.Dense(vocab_size, activation='relu'),
        ])
        self.embedding_model = tf.keras.models.Sequential([
            input_layer,
            embedding_layer,
        ])

    def train(self, data: TrainTestSplit):
        self.lstm_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(), 
            optimizer=tf.optimizers.Adam(), 
            metrics=['CategoricalAccuracy'])

        
        self.lstm_model.fit(
            x=data.train_x, 
            y=data.train_y, 
            validation_data=(data.test_x, data.test_y),
            epochs=self.n_epochs)
        

