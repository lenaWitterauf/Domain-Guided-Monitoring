from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
import logging
from tqdm import tqdm
from src.features.knowledge import DescriptionKnowledge
from .base import BaseModel, BaseEmbedding
from .config import ModelConfig

class DescriptionEmbedding(tf.keras.Model, BaseEmbedding):

    def __init__(self, 
            descriptions: DescriptionKnowledge, 
            config: ModelConfig):
        super(DescriptionEmbedding, self).__init__()
        self.config = config

        self.num_features = len(descriptions.vocab)
        self.num_hidden_features = len(descriptions.words)

        self.w = tf.keras.layers.Dense(self.config.attention_dim, use_bias=True, activation='tanh')
        self.u = tf.keras.layers.Dense(1, use_bias=False)

        self._init_basic_embedding_variables(descriptions)
        self._init_embedding_mask(descriptions)

    def _init_basic_embedding_variables(self, descriptions: DescriptionKnowledge):
        logging.info('Initializing DESCRIPTION basic embedding variables')
        self.basic_feature_embeddings = self.add_weight(
            initializer=self._get_feature_initializer(
                {idx:' '.join(words) for idx,words in descriptions.descriptions.items() if idx in set(descriptions.vocab.values())}
            ),
            trainable=self.config.base_feature_embeddings_trainable,
            name='description_embeddings/basic_feature_embeddings',
            shape=(self.num_features,self.config.embedding_dim),
        )
        self.basic_hidden_embeddings = self.add_weight(
            initializer=self._get_hidden_initializer(
                {idx:word for word,idx in descriptions.words_vocab.items()}
            ),
            trainable=self.config.base_hidden_embeddings_trainable,
            name='description_embeddings/basic_hidden_embeddings',
            shape=(self.num_hidden_features,self.config.embedding_dim),
        )

    def _init_embedding_mask(self, descriptions: DescriptionKnowledge): 
        logging.info('Initializing DESCRIPTION words information')
        embedding_masks = {}
        for idx, words in tqdm(descriptions.descriptions_set.items(), desc='Initializing Description word embedding variables'):
            id_word_idx = set([descriptions.extended_vocab[x] for x in words])
            embedding_masks[idx] = [
                (x in id_word_idx)
                for x in range(self.num_features+self.num_hidden_features)
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
        
        score = self.u(
            self.w(attention_embedding_matrix)
        ) # shape: (num_features, num_all_features, 1)
        score = tf.where(self.embedding_mask, tf.math.exp(score), 0)
        score_sum = tf.reduce_sum(score, axis=1, keepdims=True) # shape: (num_features, 1, 1)
        score_sum = tf.where(score_sum == 0, 1., score_sum)

        attention_weights = score / score_sum # shape: (num_features, num_all_features, 1)
        context_vector = attention_weights * full_embedding_matrix  # shape: (num_features, num_all_features, embedding_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape: (num_features, embedding_size)

        return (context_vector, attention_weights)

    def _final_embedding_matrix(self):
        context_vector, _ = self._calculate_attention_embeddings()
        return context_vector

    def call(self, values): # values shape: (dataset_size, max_sequence_length, num_features)
        embedding_matrix = self._final_embedding_matrix()
        return tf.linalg.matmul(values, embedding_matrix) # shape: (dataset_size, max_sequence_length, embedding_size)


class DescriptionModel(BaseModel):
    def _get_embedding_layer(self, metadata: SequenceMetadata, knowledge: DescriptionKnowledge) -> tf.keras.Model:
        return DescriptionEmbedding(knowledge, self.config)
