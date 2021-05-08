import tensorflow as tf
import pandas as pd
from typing import Dict
import logging
from tqdm import tqdm
from ..features.sequences import TrainTestSplit
from .base import BaseModel

class SimpleEmbedding(tf.keras.Model):
    embedding_size: int
    embeddings: Dict[int, tf.Variable]
    concatenated_embeddings: tf.Variable # shape: (num_variables, embedding_size)

    def __init__(self, 
            vocab: Dict[str, int],
            embedding_size: int = 16):
        super(SimpleEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self._init_embedding_variables(vocab)

    def _init_embedding_variables(self, vocab: Dict[str, int]):
        logging.info('Initializing Simple embedding variables')
        self.embeddings = {}
        for name, idx in tqdm(vocab.items(), desc='Initializing Simple embedding variables'):
            self.embeddings[idx] = tf.Variable(
                initial_value=tf.random.normal(shape=(1,self.embedding_size)),
                trainable=True,
                name=name,
            )
        
        self.concatenated_embeddings = tf.Variable(
            tf.concat(
                [self.embeddings[idx] for idx in range(len(vocab))], 
                axis=0),
            trainable=True,
            name='concatenated_embeddings',
        )

    def call(self, values): # values shape: (dataset_size, max_sequence_length, num_variables)
        return tf.linalg.matmul(values, self.concatenated_embeddings) # shape: (dataset_size, max_sequence_length, embedding_size)


class SimpleModel(BaseModel):

    def _get_embedding_layer(self, split: TrainTestSplit, vocab: Dict[str, int]) -> tf.keras.Model:
        return SimpleEmbedding(vocab)

