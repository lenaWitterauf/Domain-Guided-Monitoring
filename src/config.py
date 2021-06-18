import dataclass_cli
import dataclasses

@dataclass_cli.add
@dataclasses.dataclass
class ExperimentConfig:
    n_epochs: int = 100
    sequence_type: str = 'mimic'
    model_type: str = 'simple'
    max_data_size: int = -1
    use_dataset_generator: bool = True
    batch_size: int = 32
    multilabel_classification: bool = False
    # using this will cache dataset accross different runs.
    # don't use this if you change settings for creating the dataset!
    dataset_generator_cache_file: str = ''
    dataset_shuffle_buffer: int = 1000
    dataset_shuffle_seed: int = 12345
    tensorflow_seed: int = 7796