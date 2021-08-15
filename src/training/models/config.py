import dataclass_cli
import dataclasses
from typing import List


@dataclass_cli.add
@dataclasses.dataclass
class ModelConfig:
    rnn_type: str = "lstm"
    rnn_dim: int = 32
    rnn_dropout: float = 0.0
    embedding_dim: int = 16
    attention_dim: int = 16
    base_feature_embeddings_trainable: bool = True
    base_hidden_embeddings_trainable: bool = True
    feature_embedding_initializer: str = "random_uniform"
    feature_embedding_initializer_seed: int = 12345
    hidden_embedding_initializer: str = "random_uniform"
    hidden_embedding_initializer_seed: int = 67890
    distribute_strategy: str = ""
    best_model_metric: str = "val_loss"
    best_model_metric_minimize: bool = True
    early_stopping_epochs: int = 5
    metrics_num_percentiles: int = 5
    final_activation_function: str = "softmax"
    loss: str = "binary_crossentropy"
    optimizer: str = "adam"
    dropout_rate: float = 0.5
    dropout_seed: int = 12345
    kernel_regularizer_type: str = "l2"
    kernel_regularizer_value: float = 0.001
    kernel_regularizer_scope: List[str] = dataclasses.field(
        default_factory=lambda: []
    )


@dataclass_cli.add
@dataclasses.dataclass
class TextualPaperModelConfig:
    num_filters: int = 16
    kernel_sizes: List[int] = dataclasses.field(default_factory=lambda: [2, 3, 4],)

