import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import io
from ...features.knowledge import HierarchyKnowledge, CausalityKnowledge, DescriptionKnowledge

class EmbeddingHelper:
    vocab: Dict[str, int]
    embedding: tf.keras.Model

    def __init__(self, 
            vocab: Dict[str, int], 
            embedding: tf.keras.Model):
        self.vocab = vocab
        self.embedding = embedding

    def load_final_embeddings(self):
        final_embeddings = {}
        for word, idx in self.vocab.items():
            input_vec = self._create_one_hot_vector_for(idx, len(self.vocab))
            output_vec = self.embedding.call(input_vec)
            final_embeddings[word] = output_vec.numpy().flatten()

        return final_embeddings

    def print_final_embeddings(self, 
            vec_file_name: str = 'data/vecs.tsv',
            meta_file_name: str = 'data/meta.tsv'):
        out_vecs = io.open(vec_file_name, 'w', encoding='utf-8')
        out_meta = io.open(meta_file_name, 'w', encoding='utf-8')
        embeddings = self.load_final_embeddings()

        for word, vec in embeddings.items():
            out_vecs.write('\t'.join([str(x) for x in vec]) + "\n")
            out_meta.write(word + "\n")

        out_vecs.close()
        out_meta.close()

    def load_attention_weights(self, knowledge):
        if isinstance(knowledge, CausalityKnowledge):
            return self._load_attention_weights(self._load_relevant_words_for_causal(knowledge))
        elif isinstance(knowledge, HierarchyKnowledge):
            return self._load_attention_weights(self._load_relevant_words_for_hierarchy(knowledge))
        elif isinstance(knowledge, DescriptionKnowledge):
            return self._load_attention_weights(self._load_relevant_words_for_text(knowledge))
        else:
            logging.error('Unknown knowledge type %s', str(knowledge))
            return None

    def _load_attention_weights(self, relevant_words: Dict[str, List[Tuple[str, int]]]):
        attention_weights = {}
        _, attention_matrix = self.embedding._calculate_attention_embeddings()
        
        for word, idx in self.vocab.items():
            attention_weights[word] = {}
            attention_vector = attention_matrix[idx].numpy().flatten()

            for extra_word, extra_idx in relevant_words[word]:
                attention_weights[word][extra_word] = attention_vector[extra_idx]

        return attention_weights

    def _load_relevant_words_for_hierarchy(self, hierarchy: HierarchyKnowledge) -> Dict[str, List[Tuple[str, int]]]:
        relevant_words = {}
        for idx, node in hierarchy.nodes.items():
            relevant_words[node.label_str] = [(n.label_str, n.label_idx) for n in node.get_ancestors()]

        return relevant_words

    def _load_relevant_words_for_causal(self, causal_knowledge: CausalityKnowledge) -> Dict[str, List[Tuple[str, int]]]:
        relevant_words = {}
        for idx, node in causal_knowledge.nodes.items():
            relevant_words[node.label_str] = [(n.label_str, n.label_idx) for n in node.get_neighbours()]

        return relevant_words

    def _load_relevant_words_for_text(self, description_knowledge: DescriptionKnowledge) -> Dict[str, List[Tuple[str, int]]]:
        relevant_words = {}
        for word, idx in description_knowledge.vocab.items():
            relevant_words[word] = [
                (word, description_knowledge.words_vocab[word] - len(description_knowledge.vocab)) 
                for word in description_knowledge.descriptions_set[idx]
            ]

        return relevant_words

    def _create_one_hot_vector_for(self, idx: int, total_length: int) -> tf.Tensor:
        vec = np.zeros(total_length)
        vec[idx] = 1
        return tf.expand_dims(
            tf.convert_to_tensor(vec, dtype='float32'),
            0)