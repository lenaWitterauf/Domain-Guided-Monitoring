import tensorflow as tf
import pandas as pd
from typing import Dict
import logging
from tqdm import tqdm
from ..features.knowledge import HierarchyKnowledge
from ..features.sequences import TrainTestSplit
from .base import BaseModel

class GramEmbedding(tf.keras.Model):
    embedding_size: int
    embeddings: Dict[int, tf.Variable]
    ancestor_embeddings: Dict[int, tf.Tensor]
    concatenated_embeddings: tf.Variable # shape: (num_leaf_nodes, embedding_size)
    concatenated_ancestor_embeddings: tf.Variable # shape: (num_leaf_nodes, num_nodes, embedding_size)

    def __init__(self, 
            hierarchy: HierarchyKnowledge, 
            embedding_size: int = 16, 
            hidden_size: int = 16):
        super(GramEmbedding, self).__init__()
        self.w1 = tf.keras.layers.Dense(hidden_size)
        self.w2 = tf.keras.layers.Dense(hidden_size)
        self.u = tf.keras.layers.Dense(1)
        self._init_embedding_variables(hierarchy, embedding_size)
        self._init_ancestor_variables(hierarchy)

    def _init_embedding_variables(self, hierarchy: HierarchyKnowledge, embedding_size: int):
        logging.info('Initializing GRAM embedding variables')
        self.embeddings = {}
        for name, idx in tqdm(hierarchy.extended_vocab.items(), desc='Initializing GRAM embedding variables'):
            self.embeddings[idx] = tf.Variable(
                initial_value=tf.random.normal(shape=(1,embedding_size)),
                trainable=True,
                name=name,
            )
        
        self.concatenated_embeddings = tf.Variable(
            tf.expand_dims(
                tf.concat(
                    [self.embeddings[idx] for idx in range(len(hierarchy.vocab))], 
                    axis=0),
                1),
            trainable=True,
            name='concatenated_embeddings',
        )

    def _init_ancestor_variables(self, hierarchy: HierarchyKnowledge): 
        logging.info('Initializing GRAM ancestor embedding variables')
        self.ancestor_embeddings = {}
        for idx, node in tqdm(hierarchy.nodes.items(), desc='Initializing GRAM ancestor embedding variables'):
            if not node.is_leaf(): continue
            ancestor_idxs = set(node.get_ancestor_label_idxs() + [idx])
            id_ancestor_embeddings = [
                self.embeddings[x]  if (x in ancestor_idxs) 
                else tf.constant(0, shape=(self.embeddings[0].shape), dtype='float32')
                for x in range(len(hierarchy.extended_vocab))
            ]
            self.ancestor_embeddings[idx] = tf.concat(id_ancestor_embeddings, axis=0)

        all_ancestor_embeddings = [
            self.ancestor_embeddings[idx] for idx in range(len(hierarchy.vocab))
        ]
        self.concatenated_ancestor_embeddings = tf.Variable(
            tf.concat([all_ancestor_embeddings], axis=1),
            trainable=True,
            name='concatenated_ancestor_embeddings',
        )

    def _calculate_attention_embeddings(self):
        score = self.u(tf.nn.tanh(
            self.w1(self.concatenated_embeddings) + self.w2(self.concatenated_ancestor_embeddings)
        )) # shape: (num_leaf_nodes, num_nodes, 1)

        attention_weights = tf.nn.softmax(score, axis=0) # shape: (num_leaf_nodes, num_nodes, 1)
        context_vector = attention_weights * self.concatenated_ancestor_embeddings  # shape: (num_leaf_nodes, num_nodes, embedding_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape: (num_leaf_nodes, embedding_size)

        return (context_vector, attention_weights)

    def call(self, values): # values shape: (dataset_size, max_sequence_length, num_leaf_nodes)
        context_vector, _ = self._calculate_attention_embeddings()
        return tf.linalg.matmul(values, context_vector) # shape: (dataset_size, max_sequence_length, embedding_size)


class GramModel(BaseModel):
    def _get_embedding_layer(self, split: TrainTestSplit, knowledge: HierarchyKnowledge()) -> tf.keras.Model:
        return CausalityEmbedding(knowledge)
