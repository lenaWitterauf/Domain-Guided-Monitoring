import dataclass_cli
import dataclasses
from pathlib import Path
from typing import List


@dataclass_cli.add
@dataclasses.dataclass
class KnowledgeConfig:
    add_causality_prefix: bool = False
    file_knowledge: Path = Path("data/file_knowledge.json")
    combined_knowledge_components: List[str] = dataclasses.field(
        default_factory=lambda: ["gram", "text", "causal",],
    )
    build_text_hierarchy: bool = False
