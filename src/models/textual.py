import tensorflow as tf
import pandas as pd
from typing import Dict
import logging
from tqdm import tqdm
import fasttext.util
from ..features.knowledge import DescriptionKnowledge
from ..features.sequences import TrainTestSplit
from .base import BaseModel

class DescriptionEmbedding(tf.keras.Model):
    embedding_size: int
    embeddings: Dict[int, tf.Variable]
    word_embeddings: Dict[int, tf.Tensor]
    concatenated_embeddings: tf.Variable # shape: (num_features, 1, embedding_size)
    concatenated_text_embeddings: tf.Variable # shape: (num_features, num_words, embedding_size)

    def __init__(self, 
            descriptions: DescriptionKnowledge, 
            embedding_size: int = 16, 
            hidden_size: int = 16):
        super(DescriptionEmbedding, self).__init__()
        self.w1 = tf.keras.layers.Dense(hidden_size)
        self.w2 = tf.keras.layers.Dense(hidden_size)
        self.u = tf.keras.layers.Dense(1)
        self._init_embedding_variables(descriptions, embedding_size)
        self._init_text_embedding_variables(descriptions)

    def _init_embedding_variables(self, descriptions: DescriptionKnowledge, embedding_size: int):
        logging.info('Initializing basic description embedding variables')
        self.embeddings = {}
        for name, idx in tqdm(descriptions.vocab.items(), desc='Initializing basic description embedding variables'):
            self.embeddings[idx] = tf.Variable(
                initial_value=tf.random.normal(shape=(1,embedding_size)),
                trainable=True,
                name=name,
            )
            
        word_model = self._load_fasttext_model(embedding_size)
        for name, idx in tqdm(descriptions.words_vocab.items(), desc='Initializing word embeddings from model'):
            self.embeddings[idx] = tf.Variable(
                initial_value=tf.constant(
                    word_model.get_word_vector(name), 
                    shape=(1,word_model.get_dimension())),
                trainable=False,
                name=name,
            )
        
        self.concatenated_embeddings = tf.Variable(
            tf.expand_dims(
                tf.concat(
                    [self.embeddings[idx] for idx in range(len(descriptions.vocab))], 
                    axis=0),
                1),
            trainable=True,
            name='concatenated_embeddings',
        )

    def _load_fasttext_model(self, embedding_size: int):
        logging.info('(Down)loading fasttext English language model')
        fasttext.util.download_model('en', if_exists='ignore')
        model = fasttext.load_model('cc.en.300.bin')
        if model.get_dimension() > embedding_size:
            logging.info('Reducing dimension of FastText word model from %d to %d', model.get_dimension(), embedding_size)
            fasttext.util.reduce_model(model, embedding_size)

        return model

    def _init_text_embedding_variables(self, descriptions: DescriptionKnowledge): 
        logging.info('Initializing text embedding variables')
        self.text_embeddings = {}
        for idx, words in tqdm(descriptions.descriptions_set.items(), desc='Initializing Description word embedding variables'):
            id_word_idx = set([descriptions.words_vocab[x] for x in words])
            id_word_embeddings = [
                self.embeddings[x]  if (x in id_word_idx) 
                else tf.constant(0, shape=(self.embeddings[len(descriptions.vocab)].shape), dtype='float32')
                for x in range(
                    len(descriptions.vocab), 
                    len(descriptions.vocab) + len(descriptions.words_vocab))
            ]
            self.text_embeddings[idx] = tf.concat(id_word_embeddings, axis=0)

        all_text_embeddings = [
            self.text_embeddings[idx] for idx in range(len(descriptions.vocab))
        ]
        self.concatenated_text_embeddings = tf.Variable(
            tf.concat([all_text_embeddings], axis=1),
            trainable=True,
            name='concatenated_description_embeddings',
        )

    def _calculate_attention_embeddings(self):
        score = self.u(tf.nn.tanh(
            self.w1(self.concatenated_embeddings) + self.w2(self.concatenated_text_embeddings)
        )) # shape: (num_features, num_words, 1)

        attention_weights = tf.nn.softmax(score, axis=0) # shape: (num_features, num_words, 1)
        context_vector = attention_weights * self.concatenated_text_embeddings  # shape: (num_features, num_words, embedding_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape: (num_features, embedding_size)

        return (context_vector, attention_weights)

    def call(self, values): # values shape: (dataset_size, max_sequence_length, num_features)
        context_vector, _ = self._calculate_attention_embeddings()
        return tf.linalg.matmul(values, context_vector) # shape: (dataset_size, max_sequence_length, embedding_size)


class DescriptionModel(BaseModel):

    def _get_embedding_layer(self, split: TrainTestSplit, knowledge: DescriptionKnowledge) -> tf.keras.Model:
        return DescriptionEmbedding(knowledge)
