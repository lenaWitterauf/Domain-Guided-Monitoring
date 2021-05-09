import tensorflow as tf
import pandas as pd
from typing import Dict, List
import logging
from tqdm import tqdm
import fasttext.util
from ..features.knowledge import DescriptionKnowledge
from ..features.sequences import TrainTestSplit
from .base import BaseModel

class DescriptionEmbedding(tf.keras.Model):
    embedding_size: int
    num_features: int
    num_hidden_features: int

    basic_feature_embeddings: tf.Variable # shape: (num_features, embedding_size)
    basic_hidden_embeddings: tf.Variable # shape: (num_hidden_features, embedding_size)
    
    embedding_mask: tf.Variable # shape: (num_features, num_hidden_features, 1)

    def __init__(self, 
            descriptions: DescriptionKnowledge, 
            embedding_size: int = 16, 
            hidden_size: int = 16):
        super(DescriptionEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.num_features = len(descriptions.vocab)
        self.num_hidden_features = len(descriptions.words)

        self.w = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self.u = tf.keras.layers.Dense(1, use_bias=False)

        self._init_basic_embedding_variables(descriptions)
        self._init_embedding_mask(descriptions)

    def _init_basic_embedding_variables(self, descriptions: DescriptionKnowledge):
        logging.info('Initializing DESCRIPTION basic embedding variables')
        self.basic_feature_embeddings = self.add_weight(
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name='description_embeddings/basic_feature_embeddings',
            shape=(self.num_features,self.embedding_size),
        )

        word_model = self._load_fasttext_model()
        word_embeddings = {}
        for name, idx in tqdm(descriptions.words_vocab.items(), desc='Initializing word embeddings from model'):
            word_tensor = tf.expand_dims(
                tf.convert_to_tensor(word_model.get_word_vector(name)),
                axis=0,
            )
            word_embeddings[idx] = tf.Variable(
                initial_value=word_tensor,
                trainable=False,
                shape=(1,self.embedding_size),
            )

        concatenated_word_embeddings = tf.concat(
            [word_embeddings[x] for x in range(self.num_features, self.num_features+self.num_hidden_features)],
            axis=1,
        )
        self.basic_hidden_embeddings = self.add_weight(
            initializer=tf.keras.initializers.Constant(value=concatenated_word_embeddings),
            trainable=False,
            name='description_embeddings/basic_hidden_embeddings',
            shape=(self.num_hidden_features,self.embedding_size),
        )

    def _load_fasttext_model(self):
        logging.info('(Down)loading fasttext English language model')
        fasttext.util.download_model('en', if_exists='ignore')
        model = fasttext.load_model('cc.en.300.bin')
        if model.get_dimension() > self.embedding_size:
            logging.info('Reducing dimension of FastText word model from %d to %d', model.get_dimension(), self.embedding_size)
            fasttext.util.reduce_model(model, self.embedding_size)

        return model

    def _init_embedding_mask(self, descriptions: DescriptionKnowledge): 
        logging.info('Initializing DESCRIPTION words information')
        embedding_masks = {}
        for idx, words in tqdm(descriptions.descriptions_set.items(), desc='Initializing Description word embedding variables'):
            id_word_idx = set([descriptions.words_vocab[x] for x in words])
            embedding_masks[idx] = [
                (x in id_word_idx)
                for x in range(self.num_features, self.num_features+self.num_hidden_features)
            ]
        all_embedding_masks = [
            tf.concat(embedding_masks[idx], axis=0)
            for idx in range(self.num_features)
        ]
        self.embedding_mask = tf.Variable(tf.expand_dims(
            tf.concat([all_embedding_masks], axis=1),
            axis=2
        ), trainable=False)


    def _load_full_embedding_matrix(self):
        return tf.repeat(
            tf.expand_dims(
                self.basic_hidden_embeddings,
                axis=0
            ), # shape: (1, num_hidden_features, embedding_size)
            repeats=self.num_features,
            axis=0
        ) # shape: (num_features, num_hidden_features, embedding_size)

    def _load_attention_embedding_matrix(self):
        feature_embeddings = tf.repeat(
            tf.expand_dims(self.basic_feature_embeddings, axis=1), # shape: (num_features, 1, embedding_size)
            repeats=self.num_hidden_features,
            axis=1,
        ) # shape: (num_features, num_hidden_features, embedding_size)
        full_embeddings = self._load_full_embedding_matrix()

        return tf.concat([feature_embeddings, full_embeddings], axis=2) # shape: (num_features, num_hidden_features, 2*embedding_size)

    def _calculate_attention_embeddings(self):
        full_embedding_matrix = self._load_full_embedding_matrix()
        attention_embedding_matrix = self._load_attention_embedding_matrix()
        
        score = self.u(tf.nn.tanh(
            self.w(attention_embedding_matrix)
        )) # shape: (num_features, num_hidden_features, 1)
        score = tf.where(self.embedding_mask, tf.math.exp(score), 0)
        score_sum = tf.reduce_sum(score, axis=1, keepdims=True) # shape: (num_features, 1, 1)

        attention_weights = score / score_sum # shape: (num_features, num_hidden_features, 1)
        context_vector = attention_weights * full_embedding_matrix  # shape: (num_features, num_hidden_features, embedding_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape: (num_features, embedding_size)

        return (context_vector, attention_weights)

    def call(self, values): # values shape: (dataset_size, max_sequence_length, num_features)
        context_vector, _ = self._calculate_attention_embeddings()
        return tf.linalg.matmul(values, context_vector) # shape: (dataset_size, max_sequence_length, embedding_size)


class DescriptionModel(BaseModel):
    def _get_embedding_layer(self, split: TrainTestSplit, knowledge: DescriptionKnowledge) -> tf.keras.Model:
        return DescriptionEmbedding(knowledge)
