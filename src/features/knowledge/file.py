import pandas as pd
from typing import Dict, Set, List
from tqdm import tqdm
import logging
from .base import BaseKnowledge


class FileKnowledge(BaseKnowledge):
    def get_connections_for_idx(self, idx: int) -> Set[int]:
        return set(self.connections.get(idx, []) + [idx])

    def get_description_vocab(self, ids: Set[int]) -> Dict[int, str]:
        return {idx:name for name,idx in self.extended_vocab.items() if idx in ids}

    def build_knowledge_from_dict(
        self, knowledge: Dict[str, Set[str]], vocab: Dict[str, int]
    ):
        self.vocab: Dict[str, int] = vocab
        self.extended_vocab: Dict[str, int] = {
            k:v for k,v in self.vocab.items()
        }
        self.connections: Dict[int, List[int]] = {
            id:[id] for id in self.vocab.values()
        }

        for child, parents in tqdm(knowledge.items(), desc="Building knowledge from dict"):
            if child not in self.vocab:
                logging.debug("Ignoring node %s as not in dataset", child)
                continue

            for parent in parents:
                if parent not in self.extended_vocab:
                    self.extended_vocab[parent] = max(self.extended_vocab.values()) + 1

                self.connections[self.extended_vocab[child]].append(
                    self.extended_vocab[parent]
                )

        if len(self.extended_vocab) == len(self.vocab):
            logging.debug("Adding VOID node to ensure extended vocab > vocab")
            self.extended_vocab["_VOID_"] = max(self.extended_vocab.values()) + 1

