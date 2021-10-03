import pandas as pd
import logging
import string
from tqdm import tqdm
from typing import Dict, Set, List
from .base import BaseKnowledge


class DescriptionKnowledge(BaseKnowledge):
    def get_connections_for_idx(self, idx: int) -> Set[int]:
        return self.connections_per_index.get(idx, set([idx]))

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
        self.connections_per_index: Dict[int, Set[int]] = {}

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

        if self.config.build_text_hierarchy:
            self._build_hierarchical_text_knowledge()
        else:
            self._build_word_text_knowledge()

    def _build_hierarchical_text_knowledge(self):
        logging.info("Building hierarchical text knowledge")
        word_overlaps: Dict[str, Set[int]] = {}
        reverse_word_overlaps: Dict[int, Set[str]] = {
            idx: set() for idx in self.descriptions_set
        }
        for feature_idx, feature_words in tqdm(
            self.descriptions_set.items(), desc="Calculating text hierarchy"
        ):
            for other_feature_idx, other_feature_words in self.descriptions_set.items():
                if feature_idx == other_feature_idx:
                    continue

                word_overlap = feature_words.intersection(other_feature_words)
                if len(word_overlap) == 0:
                    continue

                overlap_string = " ".join([x for x in sorted(word_overlap)])
                if overlap_string not in word_overlaps:
                    word_overlaps[overlap_string] = set()

                word_overlaps[overlap_string].update([feature_idx, other_feature_idx])
                reverse_word_overlaps[feature_idx].add(overlap_string)
                reverse_word_overlaps[other_feature_idx].add(overlap_string)

        for word_overlap in word_overlaps:
            self._add_to_word_vocab(word_overlap)

        for feature_idx in self.vocab.values():
            self.connections_per_index[feature_idx] = set(
                [self.extended_vocab[x] for x in reverse_word_overlaps[feature_idx]]
                + [feature_idx]
            )

    def _add_to_word_vocab(self, word: str):
        if word in self.vocab:
            return

        self.words_vocab[word] = len(self.words_vocab) + len(self.vocab)
        self.extended_vocab[word] = self.words_vocab[word]

    def _build_word_text_knowledge(self):
        logging.info("Building word text knowledge")
        for word in self.words:
            self._add_to_word_vocab(word)

        for feature_idx in self.vocab.values():
            self.connections_per_index[feature_idx] = set(
                [self.extended_vocab[x] for x in self.descriptions_set[feature_idx]]
                + [feature_idx]
            )
