import dataclass_cli
import dataclasses
from typing import List

@dataclass_cli.add
@dataclasses.dataclass
class SequenceConfig:
    test_percentage: float=0.1      # how much of the data should be used for testing
    random_test_split: bool=True    # if true, split randomly; if false, split after 1-test_percentage datapoints
    random_state: int=12345         # seed used for random test split
    flatten_x: bool=True            # if true, produces one mulit-hot encoded vector per timestamp; 
    flatten_y: bool=False           #       if false, produces multiple (number of features in timestamp) one-hot encoded vectors per timestamp
    max_window_size: int=100        # max number of timestamps per prediction input
    min_window_size: int=1          # min number of timestamps per prediction input
    window_overlap: bool=True       # if true, timestamps for different prediction inputs may overlap
    allow_subwindows: bool=False    # if true, all subsequences of a given sequence are used; if false, resembles sliding window approach
    valid_y_features: List[str] = dataclasses.field(
        default_factory=lambda: [],
    )                               # if not empty, only these features are used as prediction goals
    remove_empty_v_vecs: bool=True  # if true, removes (x,y) pairs where y is a zero vector
