import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix 

from tensorflow._api.v2 import data
from ...features.sequences import SequenceMetadata


class ConfusionCalculator:
    def __init__(self, metadata: SequenceMetadata, model: tf.keras.Model):
        self.y_vocab = metadata.y_vocab
        self.model = model

    def write_confusion_for_dataset(
        self, dataset: tf.data.Dataset, out_file_name: str = "data/confusion.csv"
    ):
        confusion_df = self._calculate_confusion_df_for_dataset(dataset)
        confusion_df.to_csv(out_file_name, index_label="true_label")

    def _calculate_confusion_df_for_dataset(
        self, dataset: tf.data.Dataset
    ) -> pd.DataFrame:
        confusion_matrix = self._calculate_confusion_matrix_for_dataset(dataset)
        sorted_features = [
            feature_name
            for (feature_name, _) in sorted(self.y_vocab.items(), key=lambda x: x[1])
        ]
        return pd.DataFrame(
            confusion_matrix, index=sorted_features, columns=sorted_features,
        )

    def _calculate_confusion_matrix_for_dataset(
        self, dataset: tf.data.Dataset
    ) -> np.array:
        num_labels = len(self.y_vocab)
        confusion_matrix = np.zeros(shape=(num_labels, num_labels), dtype=np.int32)
        for (x, y_true) in tqdm(
            dataset.as_numpy_iterator(), desc="Calculating confusion matrix..."
        ):
            y_pred = self.model(x).numpy()  # shape: (batch_size, num_labels)
            y_pred = self._convert_to_int_vector(y_pred)
            confusion_matrix = confusion_matrix + self._calculate_confusion_matrix(
                y_true, y_pred
            )

        return confusion_matrix

    def _convert_to_int_vector(self, y_pred: np.array) -> np.array:
        predicted_labels = np.argmax(y_pred, axis=1)  # size: batch_size
        y_vec = np.zeros(shape=y_pred.shape, dtype=np.int8)
        y_vec[np.arange(predicted_labels.size), predicted_labels] = 1
        return y_vec

    def _calculate_confusion_matrix(
        self, y_true: np.array, y_pred: np.array
    ) -> np.array:
        return confusion_matrix(
            y_true.argmax(axis=1), 
            y_pred.argmax(axis=1),
            labels=range(y_true.shape[1])
        )

