import tensorflow as tf
import pandas as pd
from typing import Dict
import logging
from tqdm import tqdm
from ..features.knowledge import CausalityKnowledge
from ..features.sequences import TrainTestSplit
from .base import BaseModel

class CausalityEmbedding(tf.keras.Model):
    embedding_size: int
    embeddings: Dict[int, tf.Variable]
    neighbour_embeddings: Dict[int, tf.Tensor]
    concatenated_embeddings: tf.Variable # shape: (num_used_nodes, embedding_size)
    concatenated_neighbour_embeddings: tf.Variable # shape: (num_used_nodes, num_nodes, embedding_size)

    def __init__(self, 
            causality: CausalityKnowledge, 
            embedding_size: int = 16, 
            hidden_size: int = 16):
        super(CausalityEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.w1 = tf.keras.layers.Dense(hidden_size)
        self.w2 = tf.keras.layers.Dense(hidden_size)
        self.u = tf.keras.layers.Dense(1)
        self._init_embedding_variables(causality)
        self._init_neighbour_variables(causality)

    def _init_embedding_variables(self, causality: CausalityKnowledge):
        logging.info('Initializing Causality embedding variables')
        self.embeddings = {}
        for name, idx in tqdm(causality.extended_vocab.items(), desc='Initializing Causality embedding variables'):
            self.embeddings[idx] = tf.Variable(
                initial_value=tf.random.normal(shape=(1,self.embedding_size)),
                trainable=True,
                name=name,
            )
        
        self.concatenated_embeddings = tf.Variable(
            tf.expand_dims(
                tf.concat(
                    [self.embeddings[idx] for idx in range(len(causality.vocab))], 
                    axis=0),
                1),
            trainable=True,
            name='concatenated_embeddings',
        )

    def _init_neighbour_variables(self, causality: CausalityKnowledge): 
        logging.info('Initializing Causality neighbour embedding variables')
        self.neighbour_embeddings = {}
        for idx, node in tqdm(causality.nodes.items(), desc='Initializing Causality neighbour embedding variables'):
            if node.label_str not in causality.vocab: continue # we only need embeddings for nodes that are in the dataset
            neighbour_idxs = set(node.get_neighbour_label_idxs() + [idx])
            id_neighbour_embeddings = [
                self.embeddings[x]  if (x in neighbour_idxs) 
                else tf.constant(0, shape=(self.embeddings[0].shape), dtype='float32')
                for x in range(len(causality.extended_vocab))
            ]
            self.neighbour_embeddings[idx] = tf.concat(id_neighbour_embeddings, axis=0)

        all_neighbour_embeddings = [
            self.neighbour_embeddings[idx] for idx in range(len(causality.vocab))
        ]
        self.concatenated_neighbour_embeddings = tf.Variable(
            tf.concat([all_neighbour_embeddings], axis=1),
            trainable=True,
            name='concatenated_neighbour_embeddings',
        )

    def _calculate_attention_embeddings(self):
        score = self.u(tf.nn.tanh(
            self.w1(self.concatenated_embeddings) + self.w2(self.concatenated_neighbour_embeddings)
        )) # shape: (num_used_nodes, num_nodes, 1)

        attention_weights = tf.nn.softmax(score, axis=0) # shape: (num_used_nodes, num_nodes, 1)
        context_vector = attention_weights * self.concatenated_neighbour_embeddings  # shape: (num_used_nodes, num_nodes, embedding_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape: (num_used_nodes, embedding_size)

        return (context_vector, attention_weights)

    def call(self, values): # values shape: (dataset_size, max_sequence_length, num_used_nodes)
        context_vector, _ = self._calculate_attention_embeddings()
        return tf.linalg.matmul(values, context_vector) # shape: (dataset_size, max_sequence_length, embedding_size)


class CausalityModel(BaseModel):

    def _get_embedding_layer(self, split: TrainTestSplit, knowledge: CausalityKnowledge) -> tf.keras.Model:
        return CausalityEmbedding(knowledge)
