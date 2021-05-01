from typing import List

def transform_to_string(list: List[List[str]]) -> str:
    return ';'.join([','.join([str(x) for x in sorted(sublist)]) for sublist in list])