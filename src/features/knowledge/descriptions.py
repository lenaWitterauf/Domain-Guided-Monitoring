import pandas as pd
import logging
import string
from tqdm import tqdm
from typing import Dict, Set, List

class DescriptionKnowledge:
    vocab: Dict[str, int]
    descriptions: Dict[int, List[str]]
    descriptions_set: Dict[int, Set[str]]
    words: Set[str]
    words_vocab: Dict[str, int]
    max_description_length: int

    def build_knowledge_from_df(self, description_df: pd.DataFrame, vocab: Dict[str, int]):
        self.vocab = vocab
        self.words = set()
        self.words_vocab = {}
        self.descriptions = {}
        self.descriptions_set = {}
        self.max_description_length = 0

        for _, row in tqdm(description_df.iterrows(), desc='Preprocessing description words', total=len(description_df)):
            label = row['label']
            if label not in vocab:
                logging.debug('Ignoring text description of %s as it is not in sequence vocab', label)
                continue

            text_description = row['description']            
            text_description = text_description.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
            
            description_words = text_description.split(' ')
            description_words = [str(x).lower().strip() for x in description_words]
            description_words = [x for x in description_words if len(x) > 0]
            
            self.descriptions[vocab[label]] = description_words
            self.max_description_length = max(self.max_description_length, len(description_words))
            self.words.update(description_words)
            self.descriptions_set[vocab[label]] = set(description_words)

        for label, idx in vocab.items():
            if idx not in self.descriptions:
                # TODO: find alternative knowledge bases in addition to MIMIC file - eg icd9data.com
                logging.error('Failed to load description for label %s!', label)  
                self.descriptions[idx] = []    

        for word in self.words:
            self.words_vocab[word] = len(self.words_vocab)  + len(self.vocab)