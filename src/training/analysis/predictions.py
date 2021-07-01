import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from ...features.sequences import SequenceMetadata


class PredictionOutputCalculator:
    def __init__(self, metadata: SequenceMetadata, model: tf.keras.Model):
        self.x_vocab = metadata.x_vocab
        self.y_vocab = metadata.y_vocab
        self.model = model

    def write_prediction_output_for_dataset(
        self, dataset: tf.data.Dataset, out_file_name: str = "data/prediction_output.csv"
    ):
        prediction_output_df = self._calculate_prediction_output_for_dataset(dataset)
        prediction_output_df.to_csv(out_file_name, index=False)

    def _calculate_prediction_output_for_dataset(self, dataset: tf.data.Dataset) -> np.array:
        prediction_df = pd.DataFrame(columns=['input', 'output', 'output_rank'])
        for (x, y) in tqdm(
            dataset.as_numpy_iterator(), desc="Calculating x frequencies..."
        ):
            x_words = self._transform_to_words_x(x)
            y_words = self._transform_to_words_y(y)
            y_pred = self.model(x).numpy()
            label_ranks = self._calculate_label_ranks(y, y_pred)
            prediction_df = prediction_df.append(
                pd.DataFrame({
                    'input': x_words, 
                    'output': y_words,
                    'output_rank': label_ranks,
                }), ignore_index=True,
            )

        return prediction_df

    def _calculate_label_ranks(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        all_ranks = np.argsort(y_pred, axis=-1).argsort(axis=-1)
        label_ranks = []
        for batch_idx in range(y_true.shape[0]):
            y_batch = y_true[batch_idx]
            batch_ranks = []
            for idx in range(y_batch.shape[0]):
                if y_batch[idx] == 1:
                    batch_ranks.append(all_ranks[batch_idx][idx])
            label_ranks.append(batch_ranks)
        return label_ranks
    
    def _transform_to_words_x(self, x: tf.Tensor):
        all_batch_words = []
        for batch_idx in range(x.shape[0]):
            x_batch = x[batch_idx]
            batch_words = []
            for sequence_idx in range(x_batch.shape[0]):
                x_sequence = x_batch[sequence_idx]
                sequence_words = []
                for idx in range(x_sequence.shape[0]):
                    if x_sequence[idx] == 1:
                        sequence_words.append([word for (word, word_idx) in self.x_vocab.items() if word_idx == idx])
                if len(sequence_words) > 0: batch_words.append(sequence_words)
            all_batch_words.append(batch_words)

        return all_batch_words # shape: (batch_size, sequence_length, words_per_step)

    def _transform_to_words_y(self, y: tf.Tensor):
        all_batch_words = []
        for batch_idx in range(y.shape[0]):
            y_batch = y[batch_idx]
            batch_words = []
            for idx in range(y_batch.shape[0]):
                if y_batch[idx] == 1:
                    batch_words.append([word for (word, word_idx) in self.y_vocab.items() if word_idx == idx])
            all_batch_words.append(batch_words)

        return all_batch_words # shape: (batch_size, labels_per_prediction)

