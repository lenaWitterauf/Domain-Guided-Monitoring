"""Code to define model structures."""
from .metrics import MulticlassTruePositiveRate, MulticlassAccuracy, MulticlassMetric, MulticlassTrueNegativeRate, PercentileSubsetMetricHelper
from .simple import SimpleModel, SimpleEmbedding
from .gram import GramEmbedding, GramModel
from .textual import DescriptionModel, DescriptionEmbedding
from .textual_paper import DescriptionPaperModel, DescriptionPaperEmbedding
from .causal import CausalityEmbedding, CausalityModel
from .base import BaseEmbedding, BaseModel
from .config import ModelConfig, TextualPaperModelConfig
