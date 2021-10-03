from src.features.sequences.transformer import SequenceMetadata
from src.features.knowledge import CombinedKnowledge
from .base import BaseModel
from .knowledge_embedding import KnowledgeEmbedding
from .config import ModelConfig


class CombinedEmbedding(KnowledgeEmbedding):
    def __init__(self, knowledge: CombinedKnowledge, config: ModelConfig):
        super(CombinedEmbedding, self).__init__(knowledge, config, "combined_embedding")


class CombinedModel(BaseModel):
    def _get_embedding_layer(
        self, metadata: SequenceMetadata, knowledge: CombinedKnowledge
    ) -> CombinedEmbedding:
        return CombinedEmbedding(knowledge, self.config)
