"""Code to define model structures."""
from .simple import SimpleModel, SimpleEmbedding
from .gram import GramEmbedding, GramModel
from .textual import TextualModel, DescriptionEmbedding
from .causal import CausalityEmbedding, CausalityModel
from .metrics import MulticlassAccuracy, MulticlassTrueNegativeRate, MulticlassTruePositiveRate, MulticlassMetric