import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import io
import json
from ...features.knowledge import (
    HierarchyKnowledge,
    CausalityKnowledge,
    DescriptionKnowledge,
)


class EmbeddingHelper:
    def __init__(
        self, vocab: Dict[str, int], knowledge: Any, embedding: tf.keras.Model
    ):
        self.vocab = vocab
        self.knowledge = knowledge
        self.embedding = embedding

    def load_base_embeddings(self):
        base_embeddings = {}
        base_embedding_matrix = self.embedding.basic_feature_embeddings
        for word, idx in self.vocab.items():
            base_embeddings[word + "_base"] = (
                base_embedding_matrix[idx].numpy().flatten()
            )
        base_hidden_embedding_matrix = self.embedding.basic_hidden_embeddings
        hidden_vocab = self._load_hidden_vocab()
        for word, idx in hidden_vocab.items():
            base_embeddings[word + "_hidden"] = (
                base_hidden_embedding_matrix[idx - len(self.vocab)].numpy().flatten()
            )

        return base_embeddings

    def _load_hidden_vocab(self) -> Dict[str, int]:
        if isinstance(self.knowledge, DescriptionKnowledge):
            return self.knowledge.words_vocab
        elif isinstance(self.knowledge, CausalityKnowledge) or isinstance(
            self.knowledge, HierarchyKnowledge
        ):
            return dict(
                [
                    (key, value)
                    for (key, value) in self.knowledge.extended_vocab.items()
                    if key not in self.knowledge.vocab
                ]
            )
        else:
            logging.error(
                "Knowledge type %s does not have hidden vocab", str(self.knowledge)
            )
            return dict()

    def load_final_embeddings(self):
        final_embeddings = {}
        final_embedding_matrix = self.embedding._final_embedding_matrix()
        for word, idx in self.vocab.items():
            final_embeddings[word] = final_embedding_matrix[idx].numpy().flatten()

        return final_embeddings

    def write_embeddings(
        self,
        vec_file_name: str = "data/vecs.tsv",
        meta_file_name: str = "data/meta.tsv",
        include_base_embeddings: bool = True,
    ):
        out_vecs = io.open(vec_file_name, "w", encoding="utf-8")
        out_meta = io.open(meta_file_name, "w", encoding="utf-8")
        embeddings = self.load_final_embeddings()
        if include_base_embeddings:
            embeddings = dict(
                list(embeddings.items()) + list(self.load_base_embeddings().items())
            )

        for word, vec in embeddings.items():
            out_vecs.write("\t".join([str(x) for x in vec]) + "\n")
            out_meta.write(word + "\n")

        out_vecs.close()
        out_meta.close()

    def load_attention_weights(self) -> Dict[str, Dict[str, str]]:
        return self._load_attention_weights(
            self._reverse_vocab(self.knowledge.extended_vocab)
        )

    def _reverse_vocab(self, vocab: Dict[str, int]) -> Dict[int, str]:
        return {v:k for k,v in vocab.items()}

    def write_attention_weights(self, file_name: str = "data/attention.json"):
        attention_weights = self.load_attention_weights()
        json_file = io.open(file_name, "w", encoding="utf-8")
        json_file.write(json.dumps({"attention_weights": attention_weights,}))
        json_file.close()

    def _load_attention_weights(
        self, vocab: Dict[int, str]
    ) -> Dict[str, Dict[str, str]]:
        attention_weights: Dict[str, Dict[str, str]] = {}
        _, attention_matrix = self.embedding._calculate_attention_embeddings()
        flattened_attention_matrix = [aw[0] for sublist in attention_matrix.numpy() for aw in sublist]
        connection_indices = self.embedding.flattened_connection_indices
        connection_partition = self.embedding.connection_partition

        for connection_idx in range(len(connection_indices)):
            from_idx = connection_partition[connection_idx]
            to_idx = connection_indices[connection_idx]

            from_word = vocab[from_idx]
            to_word = vocab[to_idx]

            if from_word not in attention_weights:
                attention_weights[from_word] = {}
            
            attention_weights[from_word][to_word] = str(flattened_attention_matrix[connection_idx])

        return attention_weights

    def _create_one_hot_vector_for(self, idx: int, total_length: int) -> tf.Tensor:
        vec = np.zeros(total_length)
        vec[idx] = 1
        return tf.expand_dims(tf.convert_to_tensor(vec, dtype="float32"), 0)

