from src.features.sequences.transformer import SequenceMetadata
import tensorflow as tf
from src.features.knowledge import FileKnowledge
from .base import BaseModel
from .knowledge_embedding import KnowledgeEmbedding
from .config import ModelConfig


class FileEmbedding(KnowledgeEmbedding):
    def __init__(self, knowledge: FileKnowledge, config: ModelConfig):
        super(FileEmbedding, self).__init__(knowledge, config, "file_embedding")


class FileModel(BaseModel):
    def _get_embedding_layer(
        self, metadata: SequenceMetadata, knowledge: FileKnowledge
    ) -> FileEmbedding:
        return FileEmbedding(knowledge, self.config)
