import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import io
from ...features.knowledge import HierarchyKnowledge, CausalityKnowledge, DescriptionKnowledge

class EmbeddingHelper:
    vocab: Dict[str, int]
    knowledge: Any
    embedding: tf.keras.Model #BaseEmbedding

    def __init__(self, 
            vocab: Dict[str, int], 
            knowledge: Any, 
            embedding: tf.keras.Model):
        self.vocab = vocab
        self.knowledge = knowledge
        self.embedding = embedding

    def load_base_embeddings(self):
        base_embeddings = {}
        base_embedding_matrix = self.embedding.basic_feature_embeddings
        for word, idx in self.vocab.items():
            base_embeddings[word + '_base'] = base_embedding_matrix[idx].numpy().flatten()
        base_hidden_embedding_matrix = self.embedding.basic_hidden_embeddings
        hidden_vocab = self._load_hidden_vocab()
        for word, idx in hidden_vocab.items():
            base_embeddings[word + '_hidden'] = base_hidden_embedding_matrix[idx-len(self.vocab)].numpy().flatten()

        return base_embeddings

    def _load_hidden_vocab(self) -> Dict[str, int]:
        if isinstance(self.knowledge, DescriptionKnowledge):
            return self.knowledge.words_vocab
        elif isinstance(self.knowledge, CausalityKnowledge) or isinstance(self.knowledge, HierarchyKnowledge):
            return dict([
                    (key, value) for (key, value) 
                    in self.knowledge.extended_vocab.items() 
                    if key not in self.knowledge.vocab
                ])
        else:
            logging.error('Knowledge type %s does not have hidden vocab', str(self.knowledge))
            return dict()


    def load_final_embeddings(self):
        final_embeddings = {}
        final_embedding_matrix = self.embedding._final_embedding_matrix()
        for word, idx in self.vocab.items():
            final_embeddings[word] = final_embedding_matrix[idx].numpy().flatten()

        return final_embeddings

    def print_embeddings(self, 
            vec_file_name: str = 'data/vecs.tsv',
            meta_file_name: str = 'data/meta.tsv',
            include_base_embeddings: bool = True):
        out_vecs = io.open(vec_file_name, 'w', encoding='utf-8')
        out_meta = io.open(meta_file_name, 'w', encoding='utf-8')
        embeddings = self.load_final_embeddings()
        if include_base_embeddings:
            embeddings = dict(list(embeddings.items()) + list(self.load_base_embeddings().items()))

        for word, vec in embeddings.items():
            out_vecs.write('\t'.join([str(x) for x in vec]) + "\n")
            out_meta.write(word + "\n")

        out_vecs.close()
        out_meta.close()

    def load_attention_weights(self):
        if isinstance(self.knowledge, CausalityKnowledge):
            return self._load_attention_weights(self._load_relevant_words_for_causal(self.knowledge))
        elif isinstance(self.knowledge, HierarchyKnowledge):
            return self._load_attention_weights(self._load_relevant_words_for_hierarchy(self.knowledge))
        elif isinstance(self.knowledge, DescriptionKnowledge):
            return self._load_attention_weights(self._load_relevant_words_for_text(self.knowledge))
        else:
            logging.error('Unknown knowledge type %s', str(self.knowledge))
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