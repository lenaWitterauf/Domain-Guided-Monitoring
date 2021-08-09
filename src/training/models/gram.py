from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
from src.features.knowledge import HierarchyKnowledge
from .base import BaseModel
from .knowledge_embedding import KnowledgeEmbedding
from .config import ModelConfig


class GramEmbedding(KnowledgeEmbedding):
    def __init__(self, knowledge: HierarchyKnowledge, config: ModelConfig):
        super(GramEmbedding, self).__init__(knowledge, config, "gram_embedding")


class GramModel(BaseModel):
    def _get_embedding_layer(
        self, metadata: SequenceMetadata, knowledge: HierarchyKnowledge
    ) -> GramEmbedding:
        return GramEmbedding(knowledge, self.config)
