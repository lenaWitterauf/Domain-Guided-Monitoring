import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List

from ...features.sequences import SequenceMetadata


class PredictionOutputCalculator:
    def __init__(self, metadata: SequenceMetadata, model: tf.keras.Model):
        self.x_vocab = metadata.x_vocab
        self.x_vocab_reverse = {v: k for k, v in metadata.x_vocab.items()}
        self.y_vocab = metadata.y_vocab
        self.y_vocab_reverse = {v: k for k, v in metadata.y_vocab.items()}
        self.model = model

    def write_prediction_output_for_dataset(
        self,
        dataset: tf.data.Dataset,
        out_file_name: str = "data/prediction_output.csv",
    ):
        prediction_output_df = self._calculate_prediction_output_for_dataset(dataset)
        prediction_output_df.to_csv(out_file_name, index=False)

    def _calculate_prediction_output_for_dataset(
        self, dataset: tf.data.Dataset
    ) -> np.array:
        all_prediction_dfs = []
        for (x, y) in tqdm(
            dataset.as_numpy_iterator(), desc="Calculating prediction outputs..."
        ):
            x_words = self._transform_to_words_x(x)
            y_words = self._transform_to_words_y(y)
            y_pred = self.model(x).numpy()
            label_ranks = self._calculate_label_ranks(y, y_pred)
            predictions = self._transform_to_words_per_prediction(y_pred)
            all_prediction_dfs.append(
                pd.DataFrame(
                    {
                        "input": x_words,
                        "output": y_words,
                        "output_rank": label_ranks,
                        "predictions": predictions,
                    }
                )
            )

        return pd.concat(all_prediction_dfs, ignore_index=True)

    def _calculate_label_ranks(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        ranks_per_idx: Dict[int, List[int]] = {
            idx: [] for idx in range(y_true.shape[0])
        }
        all_ranks = np.argsort(y_pred, axis=-1).argsort(axis=-1)
        all_indices = np.argwhere(y_true == 1)
        for idx in range(all_indices.shape[0]):
            indices = all_indices[idx]
            batch_idx = indices[0]
            feature_idx = indices[1]

            if batch_idx not in ranks_per_idx:
                ranks_per_idx[batch_idx] = []

            ranks_per_idx[batch_idx].append(all_ranks[batch_idx][feature_idx])

        return [ranks for _, ranks in sorted(ranks_per_idx.items(), key=lambda x: x[0])]

    def _transform_to_words_x(self, x: tf.Tensor) -> List[Dict[int, List[str]]]:
        words_per_idx: Dict[int, Dict[int, List[str]]] = {
            idx: {} for idx in range(x.shape[0])
        }
        all_indices = np.argwhere(x == 1)
        for idx in range(all_indices.shape[0]):
            indices = all_indices[idx]
            batch_idx = indices[0]
            sequence_idx = indices[1]
            feature_idx = indices[2]

            if batch_idx not in words_per_idx:
                words_per_idx[batch_idx] = {}
            if sequence_idx not in words_per_idx[batch_idx]:
                words_per_idx[batch_idx][sequence_idx] = []

            words_per_idx[batch_idx][sequence_idx].append(
                self.x_vocab_reverse[feature_idx]
            )

        return [words for _, words in sorted(words_per_idx.items(), key=lambda x: x[0])]

    def _transform_to_words_per_prediction(self, y_pred: tf.Tensor):
        predictions_per_idx: Dict[int, Dict[str, float]] = {
            idx: {} for idx in range(y_pred.shape[0])
        }
        for batch_idx in range(y_pred.shape[0]):
            for feature_idx in range(y_pred.shape[1]):
                predictions_per_idx[batch_idx][
                    self.y_vocab_reverse[feature_idx]
                ] = y_pred[batch_idx][feature_idx]

        return [
            predictions
            for _, predictions in sorted(
                predictions_per_idx.items(), key=lambda x: x[0]
            )
        ]

    def _transform_to_words_y(self, y: tf.Tensor):
        words_per_idx: Dict[int, List[str]] = {idx: [] for idx in range(y.shape[0])}
        all_indices = np.argwhere(y == 1)
        for idx in range(all_indices.shape[0]):
            indices = all_indices[idx]
            batch_idx = indices[0]
            feature_idx = indices[1]

            if batch_idx not in words_per_idx:
                words_per_idx[batch_idx] = []

            words_per_idx[batch_idx].append(self.y_vocab_reverse[feature_idx])

        return [words for _, words in sorted(words_per_idx.items(), key=lambda x: x[0])]

