import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List, Generator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from .config import SequenceConfig
import logging


class SequenceMetadata:
    def __init__(
        self,
        max_x_length,
        max_sequence_length,
        max_features_per_time,
        max_features_per_sequence,
        x_vocab,
        y_vocab,
        full_y_prediction,
    ):
        self.max_x_length: int = max_x_length
        self.max_sequence_length: int = max_sequence_length
        self.max_features_per_time: int = max_features_per_time
        self.max_features_per_sequence: int = max_features_per_sequence
        self.x_vocab: Dict[str, int] = x_vocab
        self.y_vocab: Dict[str, int] = y_vocab
        self.full_y_prediction: bool = full_y_prediction


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
        self.y: List[List[str]] = []
        self.x_vecs_stacked: tf.Tensor = None
        self.y_vec: tf.Tensor = None


class NextSequenceTransformer:
    """Split Sequences for next sequence prediction."""

    def __init__(
        self, config: SequenceConfig,
    ):
        self.config = config

    def collect_metadata(
        self, sequence_df: pd.DataFrame, sequence_column_name: str
    ) -> SequenceMetadata:
        (x_vocab, y_vocab) = self._generate_vocabs(sequence_df, sequence_column_name)
        max_sequence_length = sequence_df[sequence_column_name].apply(len).max() - 1
        if not self.config.predict_full_y_sequence:
            max_sequence_length = min(self.config.max_window_size, max_sequence_length)
        max_features_per_time = (
            sequence_df[sequence_column_name]
            .apply(lambda list: max([len(sublist) for sublist in list]))
            .max()
        )
        max_features_per_sequence = max_sequence_length * max_features_per_time

        return SequenceMetadata(
            max_x_length=(
                max_sequence_length
                if self.config.flatten_x
                else max_features_per_sequence
            ),
            max_sequence_length=max_sequence_length,
            max_features_per_time=max_features_per_time,
            max_features_per_sequence=max_features_per_sequence,
            x_vocab=x_vocab,
            y_vocab=y_vocab,
            full_y_prediction = self.config.predict_full_y_sequence
        )

    def transform_train_test_split(
        self, sequence_df: pd.DataFrame, sequence_column_name: str
    ) -> TrainTestSplit:
        metadata = self.collect_metadata(sequence_df, sequence_column_name)
        train_sequences, test_sequences = self._split_train_test(
            sequence_df, sequence_column_name
        )

        transformed_train_sequences = self._transform_sequences(
            sequences=train_sequences, metadata=metadata
        )
        transformed_test_sequences = self._transform_sequences(
            sequences=test_sequences, metadata=metadata
        )

        return TrainTestSplit(
            train_x=tf.stack(
                [
                    transformed.x_vecs_stacked
                    for transformed in transformed_train_sequences
                ]
            ),
            test_x=tf.stack(
                [
                    transformed.x_vecs_stacked
                    for transformed in transformed_test_sequences
                ]
            ),
            train_y=tf.stack(
                [transformed.y_vec for transformed in transformed_train_sequences]
            ),
            test_y=tf.stack(
                [transformed.y_vec for transformed in transformed_test_sequences]
            ),
            metadata=metadata,
        )

    def _split_train_test(
        self, sequence_df: pd.DataFrame, sequence_column_name: str
    ) -> Tuple[List[List[List[str]]], List[List[List[str]]]]:
        if len(sequence_df) == 1:
            sequence_list = sequence_df[sequence_column_name].tolist()[0]
            logging.debug(
                "Splitting values of df with only one row and %d items as list",
                len(sequence_list),
            )

            test_size = int(self.config.test_percentage * len(sequence_list))
            split_index = len(sequence_list) - test_size
            train_sequence_list = sequence_list[:split_index]
            test_sequence_list = sequence_list[split_index : len(sequence_list)]
            return ([train_sequence_list], [test_sequence_list])
        elif self.config.random_test_split:
            return train_test_split(
                sequence_df[sequence_column_name],
                test_size=self.config.test_percentage,
                random_state=self.config.random_state,
            )
        else:
            test_size = int(self.config.test_percentage * len(sequence_df))
            split_index = len(sequence_df) - test_size
            train_sequence_df = sequence_df[:split_index]
            test_sequence_df = sequence_df[split_index : len(sequence_df)]
            return (
                train_sequence_df[sequence_column_name].tolist(),
                test_sequence_df[sequence_column_name].tolist(),
            )

    def _transform_sequences(
        self, sequences: List[List[List[str]]], metadata: SequenceMetadata
    ) -> List[_SplittedSequence]:
        splitted_sequences = self._split_sequences(sequences)
        resulting_splits = []
        for splitted in tqdm(
            splitted_sequences, desc="Transforming splitted sequences to tensors"
        ):
            self._translate_and_pad(splitted, metadata)
            resulting_splits.append(splitted)

        return resulting_splits

    def _split_sequences(self, sequences: List[List[List[str]]]):
        for sequence in tqdm(sequences, desc="Splitting sequences into x/y windows"):
            splitted_sequences = self._split_sequence(sequence)
            for splitted_sequence in splitted_sequences:
                yield splitted_sequence

    def _split_sequence(
        self, sequence: List[List[str]]
    ) -> Generator[_SplittedSequence, None, None]:
        if self.config.predict_full_y_sequence:
            return self._split_sequence_full_window(sequence)
        if self.config.window_overlap:
            return self._split_sequence_overlap(sequence)
        else:
            return self._split_sequence_no_overlap(sequence)

    def _split_sequence_full_window(
        self, sequence: List[List[str]]
    ) -> Generator[_SplittedSequence, None, None]:
        splitted_sequence = _SplittedSequence()
        splitted_sequence.x = sequence[: len(sequence) - 1]
        splitted_sequence.y = sequence[1 : len(sequence)]
        yield splitted_sequence

    def _split_sequence_overlap(
        self, sequence: List[List[str]]
    ) -> Generator[_SplittedSequence, None, None]:
        for start_index in range(0, len(sequence)):
            max_end_index = min(
                start_index + self.config.max_window_size + 1, len(sequence)
            )
            min_end_index = (
                start_index + self.config.min_window_size
                if self.config.allow_subwindows or start_index == 0
                else max_end_index
            )
            for end_index in range(min_end_index, max_end_index):
                if self.config.flatten_y:
                    splitted_sequences = self._split_sequence_y_flat(
                        sequence, start_index, end_index
                    )
                else:
                    splitted_sequences = self._split_sequence_y_wide(
                        sequence, start_index, end_index
                    )
                for splitted_sequence in splitted_sequences:
                    yield splitted_sequence

    def _split_sequence_no_overlap(
        self, sequence: List[List[str]]
    ) -> Generator[_SplittedSequence, None, None]:
        start_index = 0
        max_start_index = len(sequence) - 1 - self.config.min_window_size
        while start_index <= max_start_index:
            end_index = start_index + self.config.min_window_size
            if self.config.flatten_y:
                splitted_sequences = self._split_sequence_y_flat(
                    sequence, start_index, end_index
                )
            else:
                splitted_sequences = self._split_sequence_y_wide(
                    sequence, start_index, end_index
                )
            for splitted_sequence in splitted_sequences:
                yield splitted_sequence
            start_index = end_index + 1

    def _split_sequence_y_flat(
        self, sequence: List[List[str]], start_index: int, end_index: int
    ) -> List[_SplittedSequence]:
        splitted_sequence = _SplittedSequence()
        splitted_sequence.x = sequence[start_index:end_index]
        splitted_sequence.y = [sequence[end_index]]
        return [splitted_sequence]

    def _split_sequence_y_wide(
        self, sequence: List[List[str]], start_index: int, end_index: int
    ) -> List[_SplittedSequence]:
        splitted_sequences = []
        y_features = sequence[end_index]
        for feature in set(y_features):
            splitted_sequence = _SplittedSequence()
            splitted_sequence.x = sequence[start_index:end_index]
            splitted_sequence.y = [[feature]]
            splitted_sequences.append(splitted_sequence)

        return splitted_sequences

    def _transform_to_tensor(
        self, active_features: List[str], vocab: Dict[str, int]
    ) -> tf.Tensor:
        feature_vec = np.zeros(len(vocab))
        for active_feature in active_features:
            if active_feature in vocab:
                feature_vec[vocab[active_feature]] = 1
        return tf.convert_to_tensor(feature_vec, dtype="float32")

    def _translate_and_pad_x_flat(
        self,
        x_features: List[List[str]],
        x_vocab: Dict[str, int],
        max_sequence_length: int,
    ) -> tf.Tensor:
        x_vecs = []
        for x in x_features:
            x_vecs.append(self._transform_to_tensor(x, x_vocab))
        for _ in range(max_sequence_length - len(x_features)):
            x_vecs.append(self._transform_to_tensor([], x_vocab))
        return tf.stack(x_vecs)

    def _translate_and_pad_x_wide(
        self,
        x_features: List[List[str]],
        x_vocab: Dict[str, int],
        max_features_per_sequence: int,
    ) -> tf.Tensor:
        all_features = [feature for x in x_features for feature in x]
        x_vecs = []
        for feature in all_features:
            x_vecs.append(self._transform_to_tensor([feature], x_vocab))
        for _ in range(max_features_per_sequence - len(all_features)):
            x_vecs.append(self._transform_to_tensor([], x_vocab))
        return tf.stack(x_vecs)

    def _translate_and_pad_generator(
        self, x: List[List[str]], y: List[List[str]], metadata: SequenceMetadata
    ):
        y_vec = (
            self._translate_and_pad_x_flat(
                y, metadata.y_vocab, metadata.max_sequence_length
            )
            if self.config.predict_full_y_sequence
            else self._transform_to_tensor(y[0], metadata.y_vocab)
        )
        if self.config.flatten_x:
            x_vecs_stacked = self._translate_and_pad_x_flat(
                x, metadata.x_vocab, metadata.max_sequence_length
            )
        else:
            x_vecs_stacked = self._translate_and_pad_x_wide(
                x, metadata.x_vocab, metadata.max_features_per_sequence
            )
        return (x_vecs_stacked, y_vec)

    def _translate_and_pad(
        self, splitted: _SplittedSequence, metadata: SequenceMetadata
    ):
        x_vecs_stacked, y_vec = self._translate_and_pad_generator(
            splitted.x, splitted.y, metadata
        )
        splitted.x_vecs_stacked = x_vecs_stacked
        splitted.y_vec = y_vec

    def _generate_vocabs(
        self, sequence_df: pd.DataFrame, sequence_column_name: str
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        vocab = self._generate_vocab(sequence_df, sequence_column_name)
        return (vocab, vocab)

    def _generate_vocab(
        self, sequence_df: pd.DataFrame, sequence_column_name: str
    ) -> Dict[str, int]:
        flattened_sequences = (
            sequence_df[sequence_column_name]
            .agg(
                lambda x: [
                    item for sublist in x for item in sublist
                ]  # flatten labels per timestamp to one list
            )
            .tolist()
        )
        flattened_sequences = list(
            set([item for sublist in flattened_sequences for item in sublist])
        )
        return self._generate_vocab_from_list(flattened_sequences)

    def _generate_vocab_from_list(self, features: List[str]) -> Dict[str, int]:
        vocab = {}
        index = 0
        for feature in features:
            if len(feature) == 0 or feature.lower() == "nan":
                continue
            vocab[feature] = index
            index = index + 1

        return vocab


class NextPartialSequenceTransformer(NextSequenceTransformer):
    """Split Sequences for next sequence prediction, but only keep some of the features as prediciton goals."""

    def __init__(self, config: SequenceConfig):
        super().__init__(config=config)
        self.valid_x_features: List[str] = []
        self.valid_y_features: List[str] = config.valid_y_features

    def set_valid_x_features(self, valid_x_features: List[str]):
        self.valid_x_features = valid_x_features

    def set_valid_y_features(self, valid_y_features: List[str]):
        self.valid_y_features = valid_y_features

    def _generate_vocabs(
        self, sequence_df: pd.DataFrame, sequence_column_name: str
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        x_vocab = (
            self._generate_vocab_from_list(self.valid_x_features)
            if len(self.valid_x_features) > 0
            else self._generate_vocab(sequence_df, sequence_column_name)
        )
        y_vocab = (
            self._generate_vocab_from_list(self.valid_y_features)
            if len(self.valid_y_features) > 0
            else self._generate_vocab(sequence_df, sequence_column_name)
        )

        return (x_vocab, y_vocab)

    def _split_sequence(
        self, sequence: List[List[str]]
    ) -> Generator[_SplittedSequence, None, None]:
        splitted_sequence_generator = super()._split_sequence(sequence)
        should_remove_empty_y_vecs = (
            self.config.remove_empty_y_vecs
            and len(self.valid_y_features) > 0
            and not self.config.predict_full_y_sequence
        )
        should_remove_empty_x_vecs = (
            self.config.remove_empty_x_vecs
            and len(self.valid_x_features) > 0
            and not self.config.predict_full_y_sequence
        )
        for splitted_sequence in splitted_sequence_generator:
            if should_remove_empty_y_vecs and set(splitted_sequence.y[0]).isdisjoint(
                self.valid_y_features
            ):
                continue
            if should_remove_empty_x_vecs:
                splitted_sequence.x = [
                    x
                    for x in splitted_sequence.x
                    if not set(x).isdisjoint(self.valid_x_features)
                ]
            if len(splitted_sequence.x) > 0:
                yield splitted_sequence


class NextPartialSequenceTransformerFromDataframe(NextPartialSequenceTransformer):
    """Split Sequences for next sequence prediction, but only keep some of the features as prediciton goals."""

    def _generate_vocabs(
        self, sequence_df: pd.DataFrame, sequence_column_name: str
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        x_vocab = self._generate_vocab(
            sequence_df,
            self.config.x_sequence_column_name
            if (
                self.config.x_sequence_column_name is not None
                and len(self.config.x_sequence_column_name) > 0
            )
            else sequence_column_name,
        )
        y_vocab = self._generate_vocab(
            sequence_df,
            self.config.y_sequence_column_name
            if (
                self.config.y_sequence_column_name is not None
                and len(self.config.y_sequence_column_name) > 0
            )
            else sequence_column_name,
        )

        super().set_valid_x_features([x for x in x_vocab.keys()])
        super().set_valid_y_features([y for y in y_vocab.keys()])

        return (x_vocab, y_vocab)


def load_sequence_transformer() -> NextSequenceTransformer:
    config = SequenceConfig()
    if len(config.valid_y_features) > 0:
        logging.debug(
            "Using only features %s as prediction goals",
            ",".join(config.valid_y_features),
        )
        return NextPartialSequenceTransformer(config=config)
    elif (
        len(config.x_sequence_column_name) > 0 or len(config.y_sequence_column_name) > 0
    ):
        logging.debug(
            "Using only features in column %s as inputs, and features from column %s as prediction goals",
            config.x_sequence_column_name,
            config.y_sequence_column_name,
        )
        return NextPartialSequenceTransformerFromDataframe(config=config)
    else:
        return NextSequenceTransformer(config=config)
