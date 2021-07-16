import dataclass_cli
import dataclasses
from pathlib import Path


@dataclass_cli.add
@dataclasses.dataclass
class ExperimentConfig:
    n_epochs: int = 10
    sequence_type: str = "mimic"
    model_type: str = "simple"
    # NOISE
    noise_to_add: float = 0.0
    noise_to_remove: float = 0.0
    attention_weight_reference_file: Path = Path('data/attention.json')
    attention_noise_to_remove: float = 0.0
    # DATASET GENERATION
    max_data_size: int = -1
    use_dataset_generator: bool = True
    batch_size: int = 32
    multilabel_classification: bool = False
    # using this will cache dataset accross different runs.
    # don't use this if you change settings for creating the dataset!
    dataset_generator_cache_file: str = ""
    # SEEDING
    dataset_shuffle_buffer: int = 1000
    dataset_shuffle_seed: int = 12345
    random_seed: int = 82379498237
    tensorflow_seed: int = 7796
