import tensorflow as tf
import pandas as pd
from typing import Dict, List
import logging
from tqdm import tqdm
from ..features.knowledge import HierarchyKnowledge
from ..features.sequences import TrainTestSplit
from .base import BaseModel

class GramEmbedding(tf.keras.Model):
    embedding_size: int
    num_features: int
    num_hidden_features: int

    basic_feature_embeddings: tf.Variable # shape: (num_features, embedding_size)
    basic_hidden_embeddings: tf.Variable # shape: (num_hidden_features, embedding_size)
    
    embedding_mask: tf.Variable # shape: (num_features, num_all_features, 1)

    def __init__(self, 
            hierarchy: HierarchyKnowledge, 
            embedding_size: int = 16, 
            hidden_size: int = 16):
        super(GramEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.num_features = len(hierarchy.vocab)
        self.num_hidden_features = len(hierarchy.extended_vocab) - len(hierarchy.vocab)

        self.w = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self.u = tf.keras.layers.Dense(1, use_bias=False)

        self._init_basic_embedding_variables(hierarchy)
        self._init_embedding_mask(hierarchy)

    def _init_basic_embedding_variables(self, hierarchy: HierarchyKnowledge):
        logging.info('Initializing GRAM basic embedding variables')
        self.basic_feature_embeddings = self.add_weight(
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name='gram_embedding/basic_feature_embeddings',
            shape=(self.num_features,self.embedding_size),
        )
        self.basic_hidden_embeddings = self.add_weight(
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name='gram_embedding/basic_hidden_embeddings',
            shape=(self.num_hidden_features,self.embedding_size),
        )

    def _init_embedding_mask(self, hierarchy: HierarchyKnowledge): 
        logging.info('Initializing GRAM ancestor information')
        embedding_masks = {}
        for idx, node in tqdm(hierarchy.nodes.items(), desc='Initializing GRAM ancestor information'):
            if node.label_idx >= self.num_features: continue

            ancestor_idxs = set(node.get_ancestor_label_idxs() + [idx])
            embedding_masks[idx] = [
                (x in ancestor_idxs)
                for x in range(self.num_features + self.num_hidden_features)
            ]

        all_embedding_masks = [
            tf.concat(embedding_masks[idx], axis=0)
            for idx in range(self.num_features)
        ]
        self.embedding_mask = tf.Variable(tf.expand_dims(
            tf.concat([all_embedding_masks], axis=1),
            2
        ), trainable=False)

    def _load_full_embedding_matrix(self):
        return tf.repeat(
            tf.expand_dims(
                tf.concat(
                    [self.basic_feature_embeddings, self.basic_hidden_embeddings],
                    axis=0
                ), # shape: (num_all_features, embedding_size)
                axis=0
            ), # shape: (1, num_all_features, embedding_size)
            repeats=self.num_features,
            axis=0
        ) # shape: (num_features, num_all_features, embedding_size)

    def _load_attention_embedding_matrix(self):
        feature_embeddings = tf.repeat(
            tf.expand_dims(self.basic_feature_embeddings, axis=1), # shape: (num_features, 1, embedding_size)
            repeats=self.num_features+self.num_hidden_features,
            axis=1,
        ) # shape: (num_features, num_all_features, embedding_size)
        full_embeddings = self._load_full_embedding_matrix()

        return tf.concat([feature_embeddings, full_embeddings], axis=2) # shape: (num_features, num_all_features, 2*embedding_size)

    def _calculate_attention_embeddings(self):
        full_embedding_matrix = self._load_full_embedding_matrix()
        attention_embedding_matrix = self._load_attention_embedding_matrix()
        
        score = self.u(tf.nn.tanh(
            self.w(attention_embedding_matrix)
        )) # shape: (num_features, num_all_features, 1)
        score = tf.where(self.embedding_mask, tf.math.exp(score), 0)
        score_sum = tf.reduce_sum(score, axis=1, keepdims=True) # shape: (num_features, 1, 1)

        attention_weights = score / score_sum # shape: (num_features, num_all_features, 1)
        context_vector = attention_weights * full_embedding_matrix  # shape: (num_features, num_all_features, embedding_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape: (num_features, embedding_size)

        return (context_vector, attention_weights)

    def call(self, values): # values shape: (dataset_size, max_sequence_length, num_features)
        context_vector, _ = self._calculate_attention_embeddings()
        return tf.linalg.matmul(values, context_vector) # shape: (dataset_size, max_sequence_length, embedding_size)


class GramModel(BaseModel):
    def _get_embedding_layer(self, split: TrainTestSplit, knowledge: HierarchyKnowledge()) -> tf.keras.Model:
        return GramEmbedding(knowledge)
