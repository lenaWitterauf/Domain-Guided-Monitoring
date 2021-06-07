import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
import tensorflow as tf
from .config import SequenceConfig
import logging

class SequenceMetadata:
    def __init__(self, max_x_length, max_sequence_length, max_features_per_time, max_features_per_sequence, x_vocab, y_vocab):
        self.max_x_length: int = max_x_length
        self.max_sequence_length: int = max_sequence_length
        self.max_features_per_time: int = max_features_per_time
        self.max_features_per_sequence: int = max_features_per_sequence
        self.x_vocab: Dict[str, int] = x_vocab
        self.y_vocab: Dict[str, int] = y_vocab

class TrainTestSplit:
    def __init__(self, train_x, test_x, train_y, test_y, metadata):
        self.train_x: tf.Tensor = train_x
        self.test_x: tf.Tensor = test_x
        self.train_y: tf.Tensor = train_y
        self.test_y: tf.Tensor = test_y
        self.metadata: SequenceMetadata = metadata

class _SplittedSequence:
    def __init__(self):
        self.x: List[List[str]] = []
        self.y: List[str] = []
        self.x_vecs_stacked: tf.Tensor = None
        self.y_vec: tf.Tensor = None

class NextSequenceTransformer:
    """Split Sequences for next sequence prediction."""
    def __init__(self, 
            test_percentage: float=0.1,
            random_test_split: bool=True,
            random_state: int=12345,
            flatten_x: bool=True,
            flatten_y: bool=True,
            max_window_size: int=10,
            min_window_size: int=1,
            window_overlap: bool=True,
            allow_subwindows: bool=True):
        self.test_percentage = test_percentage
        self.random_test_split = random_test_split
        self.random_state = random_state
        self.flatten_x = flatten_x
        self.flatten_y  = flatten_y
        self.max_window_size = max_window_size
        self.min_window_size  = min_window_size
        self.window_overlap = window_overlap
        self.allow_subwindows = allow_subwindows

    def collect_metadata(self, sequence_df: pd.DataFrame, sequence_column_name: str) -> SequenceMetadata:
        (x_vocab, y_vocab) = self._generate_vocabs(sequence_df, sequence_column_name)
        max_sequence_length = min(
            self.max_window_size,
            sequence_df[sequence_column_name].apply(len).max() - 1)
        max_features_per_time = sequence_df[sequence_column_name].apply(
            lambda list: max([len(sublist) for sublist in list])
        ).max()
        max_features_per_sequence = max_sequence_length * max_features_per_time

        return SequenceMetadata(
            max_x_length=(max_sequence_length if self.flatten_x else max_features_per_sequence),
            max_sequence_length=max_sequence_length,
            max_features_per_time=max_features_per_time,
            max_features_per_sequence=max_features_per_sequence,
            x_vocab=x_vocab,
            y_vocab=y_vocab
        )

    def transform_train_test_split(self, sequence_df: pd.DataFrame, sequence_column_name: str) -> TrainTestSplit:
        metadata = self.collect_metadata(sequence_df, sequence_column_name)
        train_sequences, test_sequences = self._split_train_test(sequence_df, sequence_column_name)
        
        transformed_train_sequences = self._transform_sequences(
            sequences=train_sequences, 
            metadata=metadata)
        transformed_test_sequences = self._transform_sequences(
            sequences=test_sequences, 
            metadata=metadata)

        return TrainTestSplit(
            train_x=tf.stack([transformed.x_vecs_stacked for transformed in transformed_train_sequences]), 
            test_x=tf.stack([transformed.x_vecs_stacked for transformed in transformed_test_sequences]), 
            train_y=tf.stack([transformed.y_vec for transformed in transformed_train_sequences]), 
            test_y=tf.stack([transformed.y_vec for transformed in transformed_test_sequences]),
            metadata=metadata)

    def _split_train_test(self, sequence_df: pd.DataFrame, sequence_column_name: str) -> Tuple[List[List[List[str]]], List[List[List[str]]]]:
        if len(sequence_df) == 1:
            sequence_list = sequence_df[sequence_column_name].tolist()[0]
            logging.debug('Splitting values of df with only one row and %d items as list', len(sequence_list))
            
            test_size = int(self.test_percentage * len(sequence_list))
            split_index = len(sequence_list) - test_size
            train_sequence_list = sequence_list[:split_index]
            test_sequence_list = sequence_list[split_index:len(sequence_list)]
            return ([train_sequence_list], [test_sequence_list])
        elif self.random_test_split:
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
            metadata: SequenceMetadata) -> List[_SplittedSequence]:
        splitted_sequences = self._split_sequences(sequences)
        resulting_splits = []
        for splitted in tqdm(splitted_sequences, desc='Transforming splitted sequences to tensors'):
            self._translate_and_pad(splitted, metadata)
            resulting_splits.append(splitted)

        return resulting_splits
        
    def _split_sequences(self, sequences: List[List[List[str]]]):
        for sequence in tqdm(sequences, desc='Splitting sequences into x/y windows'):
            splitted_sequences = self._split_sequence(sequence)
            for splitted_sequence in splitted_sequences:
                yield splitted_sequence
    
    def _split_sequence(self, sequence: List[List[str]]):
        if self.window_overlap:
            return self._split_sequence_overlap(sequence)
        else:
            return self._split_sequence_no_overlap(sequence)

    def _split_sequence_overlap(self, sequence: List[List[str]]):
        for start_index in range(0, len(sequence)):
            max_end_index = min(start_index + self.max_window_size + 1, len(sequence))
            min_end_index = start_index + self.min_window_size \
                            if self.allow_subwindows or start_index == 0 \
                            else max_end_index
            for end_index in range(min_end_index, max_end_index):
                if self.flatten_y:
                    splitted_sequences = self._split_sequence_y_flat(sequence, start_index, end_index)
                else:
                    splitted_sequences = self._split_sequence_y_wide(sequence, start_index, end_index)
                for splitted_sequence in splitted_sequences:
                    yield splitted_sequence

    def _split_sequence_no_overlap(self, sequence: List[List[str]]):
        start_index = 0
        max_start_index = len(sequence) - 1 - self.min_window_size
        while start_index <= max_start_index:
            end_index = start_index + self.min_window_size
            if self.flatten_y:
                splitted_sequences = self._split_sequence_y_flat(sequence, start_index, end_index)
            else:
                splitted_sequences = self._split_sequence_y_wide(sequence, start_index, end_index)
            for splitted_sequence in splitted_sequences:
                yield splitted_sequence
            start_index = end_index + 1

    def _split_sequence_y_flat(self, sequence: List[List[str]], start_index: int, end_index: int) -> List[_SplittedSequence]:
        splitted_sequence = _SplittedSequence()
        splitted_sequence.x = sequence[start_index:end_index]
        splitted_sequence.y = sequence[end_index]
        return [splitted_sequence]

    def _split_sequence_y_wide(self, sequence: List[List[str]], start_index: int, end_index: int) -> List[_SplittedSequence]:
        splitted_sequences = []
        y_features = sequence[end_index]
        for feature in y_features:
            splitted_sequence = _SplittedSequence()
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
            x_features: List[List[str]], 
            x_vocab: Dict[str, int], 
            max_sequence_length: int) -> tf.Tensor:
        x_vecs = []
        for _ in range(max_sequence_length - len(x_features)):
            x_vecs.append(self._transform_to_tensor([], x_vocab))
        for x in x_features:
            x_vecs.append(self._transform_to_tensor(x, x_vocab))
        return tf.stack(x_vecs)

    def _translate_and_pad_x_wide(self, 
            x_features: List[List[str]], 
            x_vocab: Dict[str, int], 
            max_features_per_sequence: int) -> tf.Tensor:
        all_features = [feature for x in x_features for feature in x]
        x_vecs = []
        for _ in range(max_features_per_sequence - len(all_features)):
            x_vecs.append(self._transform_to_tensor([], x_vocab))
        for feature in all_features:
            x_vecs.append(self._transform_to_tensor([feature], x_vocab))
        return tf.stack(x_vecs)

    def _translate_and_pad_generator(self, 
            x: List[List[str]],  
            y: List[str], 
            metadata: SequenceMetadata):
        
        y_vec = self._transform_to_tensor(y, metadata.y_vocab)
        if self.flatten_x:
            x_vecs_stacked = self._translate_and_pad_x_flat(x, metadata.x_vocab, metadata.max_sequence_length)
        else:
            x_vecs_stacked = self._translate_and_pad_x_wide(x, metadata.x_vocab, metadata.max_features_per_sequence)
        return (x_vecs_stacked, y_vec)

    def _translate_and_pad(self, 
            splitted: _SplittedSequence, 
            metadata: SequenceMetadata):
        splitted.y_vec = self._transform_to_tensor(splitted.y, metadata.y_vocab)
        if self.flatten_x:
            splitted.x_vecs_stacked = self._translate_and_pad_x_flat(splitted.x, metadata.x_vocab, metadata.max_sequence_length)
        else:
            splitted.x_vecs_stacked = self._translate_and_pad_x_wide(splitted.x, metadata.x_vocab, metadata.max_features_per_sequence)

    def _generate_vocabs(self, sequence_df: pd.DataFrame, sequence_column_name: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        vocab = self._generate_vocab(sequence_df, sequence_column_name)
        return (vocab, vocab)
        
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

    def _generate_vocabs(self, sequence_df: pd.DataFrame, sequence_column_name: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        x_vocab = self._generate_vocab(sequence_df, sequence_column_name)
        y_vocab = self._generate_vocab_from_list(self.valid_y_features)
        return (x_vocab, y_vocab)

    def _transform_sequences(self, sequences: List[List[List[str]]], metadata: SequenceMetadata) -> List[_SplittedSequence]:
        transformed_sequences = super()._transform_sequences(sequences, metadata)
        if self.remove_empty_v_vecs:
            return [
                sequence for sequence in transformed_sequences
                if not set(sequence.y).isdisjoint(self.valid_y_features)
            ]
        else:
            return transformed_sequences

def load_sequence_transformer() -> NextSequenceTransformer:
    config = SequenceConfig()
    if len(config.valid_y_features) > 0:
        logging.debug('Using only features %s as prediction goals', ','.join(config.valid_y_features))
        transformer = NextPartialSequenceTransformer(
            test_percentage=config.test_percentage,
            random_test_split=config.random_test_split,
            random_state=config.random_state,
            flatten_x=config.flatten_x,
            flatten_y=config.flatten_y,
            max_window_size=config.max_window_size,
            min_window_size=config.min_window_size,
            window_overlap=config.window_overlap,
        )
        transformer.valid_y_features = config.valid_y_features
        transformer.remove_empty_v_vecs = config.remove_empty_v_vecs
        return transformer
    else:
        return NextSequenceTransformer(
            test_percentage=config.test_percentage,
            random_test_split=config.random_test_split,
            random_state=config.random_state,
            flatten_x=config.flatten_x,
            flatten_y=config.flatten_y,
            max_window_size=config.max_window_size,
            min_window_size=config.min_window_size,
            window_overlap=config.window_overlap,
        )