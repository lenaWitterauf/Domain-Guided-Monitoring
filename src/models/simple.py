import tensorflow as tf
import pandas as pd
from typing import Dict
import logging
from tqdm import tqdm
from ..features.sequences import TrainTestSplit
from .base import BaseModel, BaseEmbedding

class SimpleEmbedding(tf.keras.Model, BaseEmbedding):

    def __init__(self, 
            vocab: Dict[str, int],
            embedding_size: int = 16):
        super(SimpleEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.num_features = len(vocab)
        self.num_hidden_features = 0

        self._init_basic_embedding_variables(vocab)

    def _init_basic_embedding_variables(self, vocab: Dict[str, int]):
        logging.info('Initializing SIMPLE basic embedding variables')
        self.basic_feature_embeddings = self.add_weight(
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name='simple_embedding/basic_feature_embeddings',
            shape=(self.num_features,self.embedding_size),
        )

    def call(self, values): # values shape: (dataset_size, max_sequence_length, num_variables)
        embedding_matrix = self._final_embedding_matrix()
        return tf.linalg.matmul(values, embedding_matrix) # shape: (dataset_size, max_sequence_length, embedding_size)


class SimpleModel(BaseModel):
    def _get_embedding_layer(self, split: TrainTestSplit, vocab: Dict[str, int]) -> tf.keras.Model:
        return SimpleEmbedding(vocab)

