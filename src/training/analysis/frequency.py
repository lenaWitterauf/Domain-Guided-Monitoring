import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from ...features.sequences import SequenceMetadata


class FrequencyCalculator:
    def __init__(self, metadata: SequenceMetadata):
        self.x_vocab = metadata.x_vocab

    def write_frequency_for_dataset(
        self, dataset: tf.data.Dataset, out_file_name: str = "data/frequency.csv"
    ):
        frequency_df = self._calculate_frequency_df_for_dataset(dataset)
        frequency_df.to_csv(out_file_name, index_label="feature")

    def _calculate_frequency_df_for_dataset(
        self, dataset: tf.data.Dataset
    ) -> pd.DataFrame:
        frequencies = self._calculate_frequencies_for_dataset(dataset)
        sorted_features = [
            feature_name
            for (feature_name, _) in sorted(self.x_vocab.items(), key=lambda x: x[1])
        ]
        return pd.DataFrame(
            frequencies, index=sorted_features, columns=["absolue_frequency"],
        )

    def _calculate_frequencies_for_dataset(self, dataset: tf.data.Dataset) -> np.array:
        num_labels = len(self.x_vocab)
        frequencies = np.zeros(shape=(num_labels,), dtype=np.int32)
        for (x, _) in tqdm(
            dataset.as_numpy_iterator(), desc="Calculating x frequencies..."
        ):
            frequencies = frequencies + self._calculate_frequencies(x)

        return frequencies

    def _calculate_frequencies(
        self, x: np.array  # shape: (batch_size, num_steps, num_features)
    ) -> np.array:
        summed_batch = np.sum(x, axis=1, dtype=np.int32)  # shape: (batch_size, num_features)
        return np.sum(summed_batch, axis=0, dtype=np.int32)  # shape: (num_features,)

