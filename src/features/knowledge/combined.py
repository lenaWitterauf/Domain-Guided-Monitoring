from src.features.knowledge.base import BaseKnowledge
from .config import KnowledgeConfig
from typing import Dict, Set


class CombinedKnowledge(BaseKnowledge):
    def __init__(self, config: KnowledgeConfig):
        self.config = config
        self.vocab: Dict[str, int] = {}
        self.extended_vocab: Dict[str, int] = {}
        self.connections_per_idx: Dict[int, Set[int]] = {}

    def add_knowledge(self, knowledge: BaseKnowledge):
        self._add_vocab(knowledge.get_vocab())
        self._add_extended_vocab(knowledge.get_extended_vocab())
        self._update_connections(knowledge)

    def _reverse_vocab(self, vocab: Dict[str, int]) -> Dict[int, str]:
        return {idx:label for label,idx in vocab.items()}

    def _update_connections(self, knowledge: BaseKnowledge):
        reverse_vocab = self._reverse_vocab(knowledge.get_extended_vocab())
        for label, idx in knowledge.get_vocab().items():
            original_connections = knowledge.get_connections_for_idx(idx)
            connections = [self.extended_vocab[reverse_vocab[x]] for x in original_connections]
            self.connections_per_idx[self.vocab[label]].update(connections)


    def _add_vocab(self, vocab: Dict[str, int]):
        for label in vocab:
            if label not in self.vocab:
                new_label_idx = 0 if len(self.vocab) == 0 else (max(self.vocab.values()) + 1)
                self.vocab[label] = new_label_idx
            label_idx = self.vocab[label]
            if label_idx not in self.connections_per_idx:
                self.connections_per_idx[label_idx] = set([label_idx])

    def _add_extended_vocab(self, extended_vocab: Dict[str, int]):
        for label in extended_vocab:
            if label not in self.extended_vocab:
                if label in self.vocab:
                    self.extended_vocab[label] = self.vocab[label]
                else:
                    self.extended_vocab[label] = 0 if len(self.extended_vocab) == 0 else (max(self.extended_vocab.values()) + 1)


    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def get_extended_vocab(self) -> Dict[str, int]:
        return self.extended_vocab

    def get_connections_for_idx(self, idx: int) -> Set[int]:
        return self.connections_per_idx[idx]

    def get_description_vocab(self, ids: Set[int]) -> Dict[int, str]:
        return {}
