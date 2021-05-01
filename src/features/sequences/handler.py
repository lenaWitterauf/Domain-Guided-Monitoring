import pandas as pd
import numpy as np
import dataclass_cli
import dataclasses
import logging
from tqdm import tqdm
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split
import tensorflow as tf

class SplittedSequence:
    x: List[List[str]] = []
    y: List[str] = []
    x_vecs: List[tf.Tensor] = []
    x_vecs_stacked: tf.Tensor
    y_vec: tf.Tensor

class TrainTestSplit:
    train_x: tf.Tensor
    test_x: tf.Tensor
    train_y: tf.Tensor
    test_y: tf.Tensor

    max_length: int
    vocab: Dict[str, int]

    def __init__(self, train_x, test_x, train_y, test_y, max_length, vocab):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.max_length = max_length
        self.vocab = vocab

@dataclass_cli.add
@dataclasses.dataclass
class SequenceHandlerConfig:
    test_percentage: float = 0.1
    random_state: int = 12345
    flatten: bool = True

class SequenceHandler:
    test_percentage: float
    random_state: int
    flatten: bool

    def __init__(self, test_percentage=0.1, random_state=12345, flatten=True):
        self.test_percentage = test_percentage
        self.random_state = random_state
        self.flatten = flatten

    def transform_train_test_split(self, sequence_df: pd.DataFrame, sequence_column_name: str):
        vocab = self._generate_vocab(sequence_df, sequence_column_name)
        max_sequence_length = sequence_df[sequence_column_name].apply(len).max() - 1
        max_symptoms_per_sequence = sequence_df[sequence_column_name].apply(lambda x: sum([len(y) for y in x])).max()

        train_sequences, test_sequences = train_test_split(
            sequence_df[sequence_column_name], 
            test_size=self.test_percentage, 
            random_state=self.random_state)
        
        transformed_train_sequences = self._transform_sequences(
            sequences=train_sequences, 
            vocab=vocab, 
            max_sequence_length=max_sequence_length, 
            max_symptoms_per_sequence=max_symptoms_per_sequence)
        transformed_test_sequences = self._transform_sequences(
            sequences=test_sequences, 
            vocab=vocab, 
            max_sequence_length=max_sequence_length, 
            max_symptoms_per_sequence=max_symptoms_per_sequence)

        return TrainTestSplit(
            train_x=tf.stack([transformed.x_vecs_stacked for transformed in transformed_train_sequences]), 
            test_x=tf.stack([transformed.x_vecs_stacked for transformed in transformed_test_sequences]), 
            train_y=tf.stack([[transformed.y_vec] for transformed in transformed_train_sequences]), 
            test_y=tf.stack([[transformed.y_vec] for transformed in transformed_test_sequences]),
            max_length=(max_sequence_length if self.flatten else max_symptoms_per_sequences),
            vocab=vocab)

    def _transform_sequences(self, 
            sequences: List[List[List[str]]], 
            vocab: Dict[str, int], 
            max_sequence_length: int, 
            max_symptoms_per_sequence: int) -> List[SplittedSequence]:
        splitted_sequences = self._split_sequences(sequences)
        for splitted in tqdm(splitted_sequences, desc='Transforming splitted sequences to tensors'):
            self._translate_and_pad(splitted, vocab, max_sequence_length, max_symptoms_per_sequence)

        return splitted_sequences
        
    
    def _split_sequences(self, sequences: List[List[List[str]]]) -> List[SplittedSequence]:
        splitted_sequences = []
        for sequence in sequences:
            splitted_sequences.extend(self._split_sequence(sequence))

        return splitted_sequences
    
    def _split_sequence(self, sequence: List[List[str]]) -> List[SplittedSequence]:
        splitted = []
        for split_index in range(1, len(sequence)):
            splitted_sequence = SplittedSequence()
            splitted_sequence.x = sequence[0:split_index]
            splitted_sequence.y = sequence[split_index]
            splitted.append(splitted_sequence)

        return splitted

    def _transform_symptoms(self, symptoms: List[str], vocab: Dict[str, int]) -> tf.Tensor:
        symptom_vec = np.zeros(len(vocab))
        for symptom in symptoms:
            symptom_vec[vocab[symptom]] = 1
        return tf.convert_to_tensor(symptom_vec, dtype='float32')

    def _translate_and_pad_x_flat(self, 
            splitted: SplittedSequence, 
            vocab: Dict[str, int], 
            max_sequence_length: int):
        splitted.x_vecs = []
        for i in range(max_sequence_length - len(splitted.x)):
            splitted.x_vecs.append(self._transform_symptoms([], vocab))
        for x in splitted.x:
            splitted.x_vecs.append(self._transform_symptoms(x, vocab))
        splitted.x_vecs_stacked = tf.stack(splitted.x_vecs)

    def _translate_and_pad_x_wide(self, 
            splitted: SplittedSequence, 
            vocab: Dict[str, int], 
            max_symptoms_per_sequence: int):
        all_symptoms = [symptom for x in splitted.x for symptom in x]
        splitted.x_vecs = []
        for i in range(max_symptoms_per_sequence - len(all_symptoms)):
            splitted.x_vecs.append(self._transform_symptoms([], vocab))
        for symptom in all_symptoms:
            splitted.x_vecs.append(self._transform_symptoms([symptom], vocab))
        splitted.x_vecs_stacked = tf.stack(splitted.x_vecs)

    def _translate_and_pad(self, 
            splitted: SplittedSequence, 
            vocab: Dict[str, int], 
            max_sequence_length: int, 
            max_symptoms_per_sequence: int):
        splitted.y_vec = self._transform_symptoms(splitted.y, vocab)
        if self.flatten:
            self._translate_and_pad_x_flat(splitted, vocab, max_sequence_length)
        else:
            self._translate_and_pad_x_wide(splitted, vocab, max_symptoms_per_sequence)
        
    def _generate_vocab(self, sequence_df: pd.DataFrame, sequence_column_name: str) -> Dict[str, int]:
        flattened_sequences = sequence_df[sequence_column_name].agg(
            lambda x: [item for sublist in x for item in sublist] # flatten labels per timestamp to one list
        ).tolist()
        flattened_sequences = list(set([item for sublist in flattened_sequences for item in sublist]))

        vocab = {}
        index = 0
        for item in flattened_sequences:
            vocab[item] = index
            index = index + 1

        return vocab



