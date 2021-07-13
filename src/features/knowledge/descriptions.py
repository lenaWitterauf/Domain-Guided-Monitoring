import pandas as pd
import logging
import string
from tqdm import tqdm
from typing import Dict, Set, List
from .base import BaseKnowledge


class DescriptionKnowledge(BaseKnowledge):
    def get_connections_for_idx(self, idx: int) -> Set[int]:
        return set([self.extended_vocab[x] for x in self.descriptions_set[idx]] + [idx])

    def get_description_vocab(self, ids: Set[int]) -> Dict[int, str]:
        description_vocab = {
            idx: " ".join(words)
            for idx, words in self.descriptions.items()
            if idx in ids
        }
        description_vocab.update(
            {idx: word for word, idx in self.words_vocab.items() if idx in ids}
        )
        return description_vocab

    def build_knowledge_from_df(
        self, description_df: pd.DataFrame, vocab: Dict[str, int]
    ):
        self.vocab: Dict[str, int] = vocab
        self.extended_vocab: Dict[str, int] = {}
        self.words: Set[str] = set()
        self.words_vocab: Dict[str, int] = {}
        self.descriptions: Dict[int, List[str]] = {}
        self.descriptions_set: Dict[int, Set[str]] = {}
        self.max_description_length = 0

        for _, row in tqdm(
            description_df.iterrows(),
            desc="Preprocessing description words",
            total=len(description_df),
        ):
            label = row["label"]
            if label not in vocab:
                logging.debug(
                    "Ignoring text description of %s as it is not in sequence vocab",
                    label,
                )
                continue

            text_description = row["description"]
            text_description = text_description.translate(
                str.maketrans(string.punctuation, " " * len(string.punctuation))
            )

            description_words = text_description.split(" ")
            description_words = [str(x).lower().strip() for x in description_words]
            description_words = [x for x in description_words if len(x) > 0]

            self.descriptions[vocab[label]] = description_words
            self.max_description_length = max(
                self.max_description_length, len(description_words)
            )
            self.words.update(description_words)
            self.descriptions_set[vocab[label]] = set(description_words)

        for label, idx in vocab.items():
            self.extended_vocab[label] = idx
            if idx not in self.descriptions:
                logging.error("Failed to load description for label %s!", label)
                self.descriptions[idx] = []
                self.descriptions_set[idx] = set()
            self.descriptions_set[idx].add(label)

        for word in self.words:
            if word in self.vocab:
                continue
            self.words_vocab[word] = len(self.words_vocab) + len(self.vocab)
            self.extended_vocab[word] = self.words_vocab[word]
