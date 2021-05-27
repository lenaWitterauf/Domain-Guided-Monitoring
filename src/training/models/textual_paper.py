from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
from typing import Dict
import logging
import fasttext.util
from tqdm import tqdm
from src.features.knowledge import DescriptionKnowledge
from .base import BaseModel

class DescriptionPaperEmbedding(tf.keras.Model):
    embedding_size: int
    filter_dim: int = 16
    kernel_dim: int = 3
    pool_size: int = 5
    pool_strides: int = 2

    embeddings: Dict[int, tf.Tensor]
    concatenated_embeddings: tf.Tensor # shape: (num_variables, max_words_per_description, word_embedding_size)

    def __init__(self, 
            descriptions: DescriptionKnowledge, 
            embedding_size: int = 16):
        super(DescriptionPaperEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.out_layer = tf.keras.layers.Dense(embedding_size)
        self._init_embedding_variables(descriptions)
        self._init_embedding_layers(descriptions)

    def _load_fasttext_model(self):
        logging.info('(Down)loading fasttext English language model')
        fasttext.util.download_model('en', if_exists='ignore')
        return fasttext.load_model('cc.en.300.bin')
    
    def _init_embedding_variables(self, descriptions: DescriptionKnowledge):
        logging.info('Initializing Description embedding variables')
        self.embeddings = {}
        self.concatenated_embeddings = {}
        word_model = self._load_fasttext_model()
        pad_vector = tf.constant(0.0, shape=(word_model.get_dimension(),))

        for idx, description_words in tqdm(descriptions.descriptions.items(), desc='Initializing Description embedding variables'):
            self.embeddings[idx] = tf.stack(
                [
                    tf.constant(word_model.get_word_vector(word)) 
                    for word in description_words
                ] + [
                    pad_vector 
                    for i in range(descriptions.max_description_length) 
                    if i >= len(description_words)
                ], axis=0
            )
                    
        self.concatenated_embeddings = tf.stack(
            [
                self.embeddings[i] 
                for i in range(len(descriptions.descriptions))
            ], axis=0)

    def _init_embedding_layers(self, descriptions: DescriptionKnowledge): 
        logging.info('Initializing Description embedding layers')
        self.conv_layer = tf.keras.layers.Conv1D(
            filters=self.filter_dim,
            kernel_size=self.kernel_dim,
            activation='relu',
            input_shape=(descriptions.max_description_length, 300))
        self.pool_layer = tf.keras.layers.MaxPooling1D(
            pool_size=self.pool_size,
            strides=self.pool_strides)
        self.embedding_matrix = tf.keras.layers.Flatten()(
            self.pool_layer(
                self.conv_layer(
                    self.concatenated_embeddings
                )
            )
        ) # shape: (num_variables, pool_layers * filter_dim)


    def call(self, values): # values shape: (dataset_size, max_sequence_length, num_leaf_nodes)
        embedding_representation = tf.linalg.matmul(values, self.embedding_matrix)
        return self.out_layer(embedding_representation) # shape: (dataset_size, max_sequence_length, embedding_size)


class DescriptionPaperModel(BaseModel):
    def _get_embedding_layer(self, metadata: SequenceMetadata, knowledge: DescriptionKnowledge) -> tf.keras.Model:
        return DescriptionPaperEmbedding(knowledge, embedding_size=self.config.embedding_dim)