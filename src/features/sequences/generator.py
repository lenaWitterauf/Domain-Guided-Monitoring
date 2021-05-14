from src import models
from src import features
from src.features import preprocessing, sequences, knowledge
import pandas as pd
import tensorflow as tf
from typing import Any, Tuple
from .transformer import load_sequence_transformer
from pathlib import Path

def generate(sequence_df_pickle_path: str, sequence_column_name: str, for_train: bool):
    sequence_df = pd.read_pickle(Path(sequence_df_pickle_path))
    sequence_transformer = load_sequence_transformer()
    sequence_metadata = sequence_transformer.collect_metadata(sequence_df, sequence_column_name)
    
    train_sequences, test_sequences = sequence_transformer._split_train_test(sequence_df, sequence_column_name)
    relevant_sequences = train_sequences if for_train else test_sequences
    for sequence in relevant_sequences:
        splitted_sequences = sequence_transformer._split_sequences([sequence], tqdm_log=False)
        for splitted_sequence in splitted_sequences:
            sequence_transformer._translate_and_pad(splitted_sequence, sequence_metadata)
            yield splitted_sequence.x_vecs_stacked, splitted_sequence.y_vec


def generate_train(sequence_df_pickle_path: bytes, sequence_column_name: bytes):
    return generate(sequence_df_pickle_path.decode(), sequence_column_name.decode(), for_train=True)

def generate_test(sequence_df_pickle_path: bytes, sequence_column_name: bytes):
    return generate(sequence_df_pickle_path.decode(), sequence_column_name.decode(), for_train=False)
