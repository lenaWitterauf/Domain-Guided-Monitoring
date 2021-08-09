from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
from src.features.knowledge import DescriptionKnowledge
from .base import BaseModel
from .knowledge_embedding import KnowledgeEmbedding
from .config import ModelConfig


class DescriptionEmbedding(KnowledgeEmbedding):
    def __init__(self, knowledge: DescriptionKnowledge, config: ModelConfig):
        super(DescriptionEmbedding, self).__init__(
            knowledge, config, "description_embedding"
        )


class DescriptionModel(BaseModel):
    def _get_embedding_layer(
        self, metadata: SequenceMetadata, knowledge: DescriptionKnowledge
    ) -> DescriptionEmbedding:
        return DescriptionEmbedding(knowledge, self.config)
