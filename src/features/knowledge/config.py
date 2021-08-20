import dataclass_cli
import dataclasses


@dataclass_cli.add
@dataclasses.dataclass
class KnowledgeConfig:
    add_causality_prefix: bool = True