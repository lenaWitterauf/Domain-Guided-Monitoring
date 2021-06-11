import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import Dict
import logging
import fasttext
import fasttext.util
import re

class FastTextInitializer:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.fasttext_model = self._load_fasttext_model()

    def _load_fasttext_model(self):
        logging.info('(Down)loading fasttext English language model')
        fasttext.util.download_model('en', if_exists='ignore')
        model = fasttext.load_model('cc.en.300.bin')
        if model.get_dimension() > self.embedding_dim:
            logging.info('Reducing dimension of FastText word model from %d to %d', model.get_dimension(), self.embedding_dim)
            fasttext.util.reduce_model(model, self.embedding_dim)

        return model

    def _load_word_embedding(self, description: str) -> tf.Tensor:
        description_words = ' '.join(re.split('[,._-]+', description)).split(' ')
        description_vectors = [
            self.fasttext_model.get_word_vector(word)
            for word in description_words
        ]     
        return tf.convert_to_tensor(np.mean(description_vectors, axis=0))

    def _load_word_embeddings(self, description_vocab: Dict[int, str]) -> Dict[int, tf.Variable]:
        word_embeddings = {}
        for idx, description in tqdm(description_vocab.items(), desc='Initializing word embeddings from model'):
            word_embeddings[idx] = tf.constant(
                tf.expand_dims(
                    self._load_word_embedding(description),
                    axis=0,
                ),
                shape=(1,self.embedding_dim),
            )
        return word_embeddings

    def get_initializer(self, description_vocab: Dict[int, str]) -> tf.keras.initializers.Initializer:
        word_embeddings = self._load_word_embeddings(description_vocab)
        concatenated_word_embeddings = tf.concat(
            [word_embeddings[x] for x in sorted(word_embeddings.keys())],
            axis=1,
        )
        return tf.keras.initializers.Constant(value=concatenated_word_embeddings)