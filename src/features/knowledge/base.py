from .config import KnowledgeConfig
from typing import Dict, Set


class BaseKnowledge:
    def __init__(self, config: KnowledgeConfig):
        self.config = config
        self.vocab: Dict[str, int] = {}
        self.extended_vocab: Dict[str, int] = {}

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def get_extended_vocab(self) -> Dict[str, int]:
        return self.extended_vocab

    def get_connections_for_idx(self, idx: int) -> Set[int]:
        return set([idx])

    def get_description_vocab(self, ids: Set[int]) -> Dict[int, str]:
        return {}
