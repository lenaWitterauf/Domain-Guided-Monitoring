from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
from src.features.knowledge import CausalityKnowledge
from .base import BaseModel
from .knowledge_embedding import KnowledgeEmbedding
from .config import ModelConfig


class CausalityEmbedding(KnowledgeEmbedding):
    def __init__(self, causality: CausalityKnowledge, config: ModelConfig):
        super(CausalityEmbedding, self).__init__(
            causality, config, "causality_embedding"
        )


class CausalityModel(BaseModel):
    def _get_embedding_layer(
        self, metadata: SequenceMetadata, knowledge: CausalityKnowledge
    ) -> tf.keras.Model:
        return CausalityEmbedding(knowledge, self.config)
