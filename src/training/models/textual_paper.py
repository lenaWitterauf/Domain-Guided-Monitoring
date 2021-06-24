from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
from typing import List
import logging
import fasttext.util
from tqdm import tqdm
from src.features.knowledge import DescriptionKnowledge
from .base import BaseEmbedding, BaseModel
from .config import ModelConfig, TextualPaperModelConfig


class DescriptionPaperEmbedding(tf.keras.Model, BaseEmbedding):
    def __init__(
        self,
        descriptions: DescriptionKnowledge,
        config: ModelConfig,
        textual_config: TextualPaperModelConfig,
    ):
        super(DescriptionPaperEmbedding, self).__init__()
        self.config = config
        self.textual_config = textual_config

        self.num_features = len(descriptions.vocab)
        self.num_hidden_features = 0
        self.num_connections = 0

        self._init_basic_embedding_variables(descriptions)
        self._init_convolution_layers(descriptions)

    def _load_fasttext_model(self):
        logging.info("(Down)loading fasttext English language model")
        fasttext.util.download_model("en", if_exists="ignore")
        return fasttext.load_model("cc.en.300.bin")

    def _init_basic_embedding_variables(self, descriptions: DescriptionKnowledge):
        logging.info("Initializing Description embedding variables")
        embeddings = {}
        word_model = self._load_fasttext_model()
        pad_vector = tf.constant(0.0, shape=(word_model.get_dimension(),))

        for idx, description_words in tqdm(
            descriptions.descriptions.items(),
            desc="Initializing Description embedding variables",
        ):
            embeddings[idx] = tf.stack(
                [
                    tf.constant(word_model.get_word_vector(word))
                    for word in description_words
                ]
                + [
                    pad_vector
                    for i in range(descriptions.max_description_length)
                    if i >= len(description_words)
                ],
                axis=0,
            )

        concatenated_embeddings = tf.stack(
            [embeddings[i] for i in range(len(descriptions.descriptions))], axis=0
        )  # shape: (num_variables, max_words_per_description, word_embedding_size)
        self.basic_feature_embeddings = self.add_weight(
            initializer=tf.keras.initializers.constant(
                value=concatenated_embeddings.numpy(),
            ),
            trainable=self.config.base_feature_embeddings_trainable,
            name="description_paper_embeddings/basic_hidden_embeddings",
            shape=(
                self.num_features,
                descriptions.max_description_length,
                word_model.get_dimension(),
            ),
        )

    def _init_convolution_layers(self, descriptions: DescriptionKnowledge):
        logging.info("Initializing Description convolution layers")
        conv_layers = {
            kernel_size:tf.keras.layers.Conv1D(
                filters=self.textual_config.num_filters,
                kernel_size=kernel_size,
                activation="relu",
                input_shape=(
                    descriptions.max_description_length,
                    self.basic_feature_embeddings.shape[2],
                ),
            )
            for kernel_size in self.textual_config.kernel_sizes
        }
        pool_layers = {
            kernel_size:tf.keras.layers.MaxPooling1D(
                pool_size=descriptions.max_description_length-kernel_size+1, 
                strides=None,
            )
            for kernel_size in self.textual_config.kernel_sizes
        }

        input_layer = tf.keras.layers.Input(
            shape=self.basic_feature_embeddings.shape[1:],
        )
        output = tf.keras.layers.Concatenate(axis=1)(
            [
                tf.keras.layers.Flatten()(
                    pool_layers[kernel_size](
                        conv_layers[kernel_size](input_layer)
                    )
                )
                for kernel_size in self.textual_config.kernel_sizes
            ]
        )  # shape: (num_variables, num_pool_layers * filter_dim))
        self.embedding_model = tf.keras.models.Model(inputs=input_layer, outputs=output)

    def _final_embedding_matrix(self):
        return self.embedding_model(
            self.basic_feature_embeddings
        )  # shape: (num_variables, pool_layers * filter_dim)

    def call(
        self, values
    ):  # values shape: (dataset_size, max_sequence_length, num_leaf_nodes)
        embedding_matrix = self._final_embedding_matrix()
        return tf.linalg.matmul(
            values, embedding_matrix
        )  # shape: (dataset_size, max_sequence_length, embedding_size)


class DescriptionPaperModel(BaseModel):
    def _get_embedding_layer(
        self, metadata: SequenceMetadata, knowledge: DescriptionKnowledge
    ) -> tf.keras.Model:
        return DescriptionPaperEmbedding(
            knowledge, self.config, textual_config=TextualPaperModelConfig()
        )

