import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List
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

    max_x_length: int
    x_vocab: Dict[str, int]
    y_vocab: Dict[str, int]

    def __init__(self, train_x, test_x, train_y, test_y, max_x_length, x_vocab, y_vocab):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.max_x_length = max_x_length
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab

class NextSequenceTransformer:
    """Split Sequences for next sequence prediction."""
    test_percentage: float
    random_test_split: bool
    random_state: int
    flatten_x: bool
    flatten_y: bool
    max_window_size: int
    min_window_size: int
    window_overlap: bool

    def __init__(self, 
            test_percentage: float=0.1,
            random_test_split: bool=True,
            random_state: int=12345,
            flatten_x: bool=True,
            flatten_y: bool=True,
            max_window_size: int=10,
            min_window_size: int=1,
            window_overlap: bool=True):
        self.test_percentage = test_percentage
        self.random_test_split = random_test_split
        self.random_state = random_state
        self.flatten_x = flatten_x
        self.flatten_y  = flatten_y
        self.max_window_size = max_window_size
        self.min_window_size  = min_window_size
        self.window_overlap = window_overlap

    def transform_train_test_split(self, sequence_df: pd.DataFrame, sequence_column_name: str) -> TrainTestSplit:
        vocab = self._generate_vocab(sequence_df, sequence_column_name)
        max_sequence_length = sequence_df[sequence_column_name].apply(len).max() - 1
        max_features_per_sequence = sequence_df[sequence_column_name].apply(lambda x: sum([len(y) for y in x])).max()

        train_sequences, test_sequences = self._split_train_test(sequence_df, sequence_column_name)
        
        transformed_train_sequences = self._transform_sequences(
            sequences=train_sequences, 
            x_vocab=vocab, 
            y_vocab=vocab,
            max_sequence_length=max_sequence_length, 
            max_features_per_sequence=max_features_per_sequence)
        transformed_test_sequences = self._transform_sequences(
            sequences=test_sequences, 
            x_vocab=vocab, 
            y_vocab=vocab,
            max_sequence_length=max_sequence_length, 
            max_features_per_sequence=max_features_per_sequence)

        return TrainTestSplit(
            train_x=tf.stack([transformed.x_vecs_stacked for transformed in transformed_train_sequences]), 
            test_x=tf.stack([transformed.x_vecs_stacked for transformed in transformed_test_sequences]), 
            train_y=tf.stack([[transformed.y_vec] for transformed in transformed_train_sequences]), 
            test_y=tf.stack([[transformed.y_vec] for transformed in transformed_test_sequences]),
            max_x_length=(max_sequence_length if self.flatten_x else max_features_per_sequence),
            x_vocab=vocab,
            y_vocab=vocab)

    def _split_train_test(self, sequence_df: pd.DataFrame, sequence_column_name: str) -> Tuple[List[List[List[str]]], List[List[List[str]]]]:
        if self.random_test_split:
            return train_test_split(
                sequence_df[sequence_column_name], 
                test_size=self.test_percentage, 
                random_state=self.random_state)
        else:
            test_size = int(self.test_percentage * len(sequence_df))
            split_index = len(sequence_df) - test_size
            train_sequence_df = sequence_df[:split_index]
            test_sequence_df = sequence_df[split_index:len(sequence_df)]
            return (train_sequence_df[sequence_column_name].tolist(), test_sequence_df[sequence_column_name].tolist())

    def _transform_sequences(self, 
            sequences: List[List[List[str]]], 
            x_vocab: Dict[str, int], 
            y_vocab: Dict[str, int], 
            max_sequence_length: int, 
            max_features_per_sequence: int) -> List[SplittedSequence]:
        splitted_sequences = self._split_sequences(sequences)
        for splitted in tqdm(splitted_sequences, desc='Transforming splitted sequences to tensors'):
            self._translate_and_pad(splitted, 
                x_vocab=x_vocab, 
                y_vocab=y_vocab, 
                max_sequence_length=max_sequence_length, 
                max_features_per_sequence=max_features_per_sequence)

        return splitted_sequences
        
    def _split_sequences(self, sequences: List[List[List[str]]]) -> List[SplittedSequence]:
        splitted_sequences = []
        for sequence in tqdm(sequences, desc='Splitting sequences into x/y windows'):
            splitted_sequences.extend(self._split_sequence(sequence))

        return splitted_sequences
    
    def _split_sequence(self, sequence: List[List[str]]) -> List[SplittedSequence]:
        if self.window_overlap:
            return self._split_sequence_overlap(sequence)
        else:
            return self._split_sequence_no_overlap(sequence)

    def _split_sequence_overlap(self, sequence: List[List[str]]) -> List[SplittedSequence]:
        splitted_sequences: List[SplittedSequence] = []
        for start_index in range(0, len(sequence)):
            min_end_index = start_index + self.min_window_size
            max_end_index = min(start_index + self.max_window_size + 1, len(sequence))
            for end_index in range(min_end_index, max_end_index):
                if self.flatten_y:
                    splitted_sequences = splitted_sequences + self._split_sequence_y_flat(sequence, start_index, end_index)
                else:
                    splitted_sequences = splitted_sequences + self._split_sequence_y_wide(sequence, start_index, end_index)

        return splitted_sequences

    def _split_sequence_no_overlap(self, sequence: List[List[str]]) -> List[SplittedSequence]:
        splitted_sequences: List[SplittedSequence] = []
        start_index = 0
        max_start_index = len(sequence) - 1 - self.min_window_size
        while start_index <= max_start_index:
            end_index = start_index + self.min_window_size
            if self.flatten_y:
                splitted_sequences = splitted_sequences + self._split_sequence_y_flat(sequence, start_index, end_index)
            else:
                splitted_sequences = splitted_sequences + self._split_sequence_y_wide(sequence, start_index, end_index)
            start_index = end_index + 1

        return splitted_sequences

    def _split_sequence_y_flat(self, sequence: List[List[str]], start_index: int, end_index: int) -> List[SplittedSequence]:
        splitted_sequence = SplittedSequence()
        splitted_sequence.x = sequence[start_index:end_index]
        splitted_sequence.y = sequence[end_index]
        return [splitted_sequence]

    def _split_sequence_y_wide(self, sequence: List[List[str]], start_index: int, end_index: int) -> List[SplittedSequence]:
        splitted_sequences = []
        y_features = sequence[end_index]
        for feature in y_features:
            splitted_sequence = SplittedSequence()
            splitted_sequence.x = sequence[start_index:end_index]
            splitted_sequence.y = [feature]
            splitted_sequences.append(splitted_sequence)

        return splitted_sequences

    def _transform_to_tensor(self, active_features: List[str], vocab: Dict[str, int]) -> tf.Tensor:
        feature_vec = np.zeros(len(vocab))
        for active_feature in active_features:
            if active_feature in vocab:
                feature_vec[vocab[active_feature]] = 1
        return tf.convert_to_tensor(feature_vec, dtype='float32')

    def _translate_and_pad_x_flat(self, 
            splitted: SplittedSequence, 
            x_vocab: Dict[str, int], 
            max_sequence_length: int):
        splitted.x_vecs = []
        for _ in range(max_sequence_length - len(splitted.x)):
            splitted.x_vecs.append(self._transform_to_tensor([], x_vocab))
        for x in splitted.x:
            splitted.x_vecs.append(self._transform_to_tensor(x, x_vocab))
        splitted.x_vecs_stacked = tf.stack(splitted.x_vecs)

    def _translate_and_pad_x_wide(self, 
            splitted: SplittedSequence, 
            x_vocab: Dict[str, int], 
            max_features_per_sequence: int):
        all_features = [feature for x in splitted.x for feature in x]
        splitted.x_vecs = []
        for _ in range(max_features_per_sequence - len(all_features)):
            splitted.x_vecs.append(self._transform_to_tensor([], x_vocab))
        for feature in all_features:
            splitted.x_vecs.append(self._transform_to_tensor([feature], x_vocab))
        splitted.x_vecs_stacked = tf.stack(splitted.x_vecs)

    def _translate_and_pad(self, 
            splitted: SplittedSequence, 
            x_vocab: Dict[str, int], 
            y_vocab: Dict[str, int], 
            max_sequence_length: int, 
            max_features_per_sequence: int):
        splitted.y_vec = self._transform_to_tensor(splitted.y, y_vocab)
        if self.flatten_x:
            self._translate_and_pad_x_flat(splitted, x_vocab, max_sequence_length)
        else:
            self._translate_and_pad_x_wide(splitted, x_vocab, max_features_per_sequence)
        
    def _generate_vocab(self, sequence_df: pd.DataFrame, sequence_column_name: str) -> Dict[str, int]:
        flattened_sequences = sequence_df[sequence_column_name].agg(
            lambda x: [item for sublist in x for item in sublist] # flatten labels per timestamp to one list
        ).tolist()
        flattened_sequences = list(set([item for sublist in flattened_sequences for item in sublist]))
        return self._generate_vocab_from_list(flattened_sequences)

    def _generate_vocab_from_list(self, features: List[str]) -> Dict[str, int]:
        vocab = {}
        index = 0
        for feature in features:
            vocab[feature] = index
            index = index + 1

        return vocab

class NextPartialSequenceTransformer(NextSequenceTransformer):
    """Split Sequences for next sequence prediction, but only keep some of the features as prediciton goals."""

    valid_y_features: List[str]
    remove_empty_v_vecs: bool

    def transform_train_test_split(self, sequence_df: pd.DataFrame, sequence_column_name: str) -> TrainTestSplit:
        x_vocab = self._generate_vocab(sequence_df, sequence_column_name)
        y_vocab = self._generate_vocab_from_list(self.valid_y_features)

        max_sequence_length = sequence_df[sequence_column_name].apply(len).max() - 1
        max_features_per_sequence = sequence_df[sequence_column_name].apply(lambda x: sum([len(y) for y in x])).max()

        train_sequences, test_sequences = self._split_train_test(sequence_df, sequence_column_name)
        
        transformed_train_sequences = self._transform_sequences(
            sequences=train_sequences, 
            x_vocab=x_vocab, 
            y_vocab=y_vocab,
            max_sequence_length=max_sequence_length, 
            max_features_per_sequence=max_features_per_sequence)
        transformed_test_sequences = self._transform_sequences(
            sequences=test_sequences,
            x_vocab=x_vocab, 
            y_vocab=y_vocab,
            max_sequence_length=max_sequence_length, 
            max_features_per_sequence=max_features_per_sequence)

        return TrainTestSplit(
            train_x=tf.stack([transformed.x_vecs_stacked for transformed in transformed_train_sequences]), 
            test_x=tf.stack([transformed.x_vecs_stacked for transformed in transformed_test_sequences]), 
            train_y=tf.stack([[transformed.y_vec] for transformed in transformed_train_sequences]), 
            test_y=tf.stack([[transformed.y_vec] for transformed in transformed_test_sequences]),
            max_x_length=(max_sequence_length if self.flatten_x else max_features_per_sequence),
            x_vocab=x_vocab,
            y_vocab=y_vocab)

    def _transform_sequences(self, sequences: List[List[List[str]]], x_vocab: Dict[str, int], y_vocab: Dict[str, int], max_sequence_length: int, max_features_per_sequence: int) -> List[SplittedSequence]:
        transformed_sequences = super()._transform_sequences(sequences, x_vocab, y_vocab, max_sequence_length, max_features_per_sequence)
        if self.remove_empty_v_vecs:
            return [
                sequence for sequence in transformed_sequences
                if not set(sequence.y).isdisjoint(self.valid_y_features)
            ]
        else:
            return transformed_sequences