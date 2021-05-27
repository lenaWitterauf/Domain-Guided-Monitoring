import dataclass_cli
import dataclasses

@dataclass_cli.add
@dataclasses.dataclass
class ModelConfig:
    rnn_type: str = 'lstm'
    rnn_dim: int = 32
    embedding_dim: int = 16
    attention_dim: int = 16
    base_feature_embeddings_trainable: bool = True
    base_hidden_embeddings_trainable: bool = True
    feature_embedding_initializer: str = 'random_uniform'
    feature_embedding_initializer_seed: int = 12345
    hidden_embedding_initializer: str = 'random_uniform'
    hidden_embedding_initializer_seed: int = 67890