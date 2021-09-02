import dataclass_cli
import dataclasses
from pathlib import Path


@dataclass_cli.add
@dataclasses.dataclass
class KnowledgeConfig:
    add_causality_prefix: bool = False
    file_knowledge: Path = Path("data/file_knowledge.json")