from src.features.knowledge.base import BaseKnowledge
from typing import Dict, Set, Tuple, List
from tqdm import tqdm
import random
from .base import BaseKnowledge


class NoiseKnowledge(BaseKnowledge):
    def __init__(self, knowledge: BaseKnowledge):
        self.knowledge = knowledge
        self.vocab: Dict[str, int] = knowledge.vocab
        self.extended_vocab: Dict[str, int] = knowledge.extended_vocab

        self._initialize_connections_from_knowledge(knowledge)
        self.original_num_connections = self.num_connections
        self.original_connections = {k: set(v) for k, v in self.connections.items()}
        self.original_reverse_connections = {
            k: set(v) for k, v in self.reverse_connections.items()
        }

    def get_text_connections(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        reverse_text_vocab: Dict[int, str] = {
            v:k for k,v in self.extended_vocab.items()
        }
        original_connections_text = {
            reverse_text_vocab[k]:[reverse_text_vocab[v] for v in vs] for k,vs in self.original_connections.items()
        }
        noise_connections_text = {
            reverse_text_vocab[k]:[reverse_text_vocab[v] for v in vs] for k,vs in self.connections.items()
        }
        return (original_connections_text, noise_connections_text)

    def _initialize_connections_from_knowledge(self, knowledge: BaseKnowledge):
        self.num_connections = 0
        self.reverse_connections: Dict[int, Set[int]] = {}
        self.connections: Dict[int, Set[int]] = {}
        for _, idx in knowledge.get_vocab().items():
            connections = knowledge.get_connections_for_idx(idx)
            self.connections[idx] = connections
            for connected_idx in connections:
                self.num_connections += 1
                if idx == connected_idx:
                    continue

                if connected_idx not in self.reverse_connections:
                    self.reverse_connections[connected_idx] = set()
                self.reverse_connections[connected_idx].add(idx)

    def add_random_connections(self, percentage: float = 0.1):
        num_connections_to_add = int(percentage * self.original_num_connections)
        added_connections = 0
        with tqdm(
            total=num_connections_to_add,
            desc="Adding {} random connections to knowledge".format(
                num_connections_to_add
            ),
        ) as pbar:
            while added_connections < num_connections_to_add:
                from_idx = random.choice(list(self.connections.keys()))
                to_idx = random.choice(list(self.reverse_connections.keys()))
                if (from_idx == to_idx) or (to_idx in self.connections[from_idx]):
                    continue

                self.connections[from_idx].add(to_idx)
                self.reverse_connections[to_idx].add(from_idx)
                added_connections += 1
                self.num_connections += 1
                pbar.update(n=1)

    def remove_random_connections(self, percentage: float = 0.1):
        num_connections_to_remove = int(percentage * self.original_num_connections)
        removed_connections = 0
        with tqdm(
            total=num_connections_to_remove,
            desc="Removing {} random connections to knowledge".format(
                num_connections_to_remove
            ),
        ) as pbar:
            while removed_connections < num_connections_to_remove:
                from_idx = random.choice(list(self.connections.keys()))
                to_idx = random.choice(list(self.reverse_connections.keys()))
                if (from_idx == to_idx) or (to_idx not in self.connections[from_idx]):
                    continue

                self.connections[from_idx].remove(to_idx)
                self.reverse_connections[to_idx].remove(from_idx)
                removed_connections += 1
                self.num_connections -= 1
                pbar.update(n=1)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def get_extended_vocab(self) -> Dict[str, int]:
        return self.extended_vocab

    def get_connections_for_idx(self, idx: int) -> Set[int]:
        return self.connections[idx]

    def get_description_vocab(self, ids: Set[int]) -> Dict[int, str]:
        return self.knowledge.get_description_vocab(ids)
