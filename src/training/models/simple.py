from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
import pandas as pd
from typing import Dict
import logging
from .base import BaseModel, BaseEmbedding
from .config import ModelConfig

class SimpleEmbedding(tf.keras.Model, BaseEmbedding):

    def __init__(self, 
            vocab: Dict[str, int], config: ModelConfig):
        super(SimpleEmbedding, self).__init__()
        self.config = config

        self.num_features = len(vocab)
        self.num_hidden_features = 0

        self._init_basic_embedding_variables(vocab)

    def _init_basic_embedding_variables(self, vocab: Dict[str, int]):
        logging.info('Initializing SIMPLE basic embedding variables')
        self.basic_feature_embeddings = self.add_weight(
            initializer=self._get_feature_initializer(
                {idx:name for name,idx in vocab.items()}
            ),
            trainable=self.config.base_feature_embeddings_trainable,
            name='simple_embedding/basic_feature_embeddings',
            shape=(self.num_features,self.config.embedding_dim),
        )
        self.basic_hidden_embeddings = None

    def call(self, values): # values shape: (dataset_size, max_sequence_length, num_variables)
        embedding_matrix = self._final_embedding_matrix()
        return tf.linalg.matmul(values, embedding_matrix) # shape: (dataset_size, max_sequence_length, embedding_size)


class SimpleModel(BaseModel):
    def _get_embedding_layer(self, metadata: SequenceMetadata, vocab: Dict[str, int]) -> tf.keras.Model:
        return SimpleEmbedding(vocab, self.config)

