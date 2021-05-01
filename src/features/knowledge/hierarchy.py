import pandas as pd
from typing import Dict, List
from tqdm import tqdm

class Node:
    label_idx: int
    label_str: str
    in_nodes: List['Node']
    out_nodes: List['Node']

    def __init__(self, label_idx, label_str):
        self.label_idx=label_idx
        self.label_str=label_str
        self.in_nodes=[]
        self.out_nodes=[]

    def is_root(self) -> bool:
        return len(self.in_nodes) == 0

    def is_leaf(self) -> bool:
        return len(self.out_nodes) == 0

    def get_ancestors(self) -> List['Node']:
        ancestors = [self]
        for node in self.in_nodes:
            ancestors = ancestors + node.get_ancestors()
        
        return list(set(ancestors))

    def get_ancestor_label_idxs(self) -> List[int]:
        ancestors = self.get_ancestors()
        return [ancestor.label_idx for ancestor in ancestors]

    def __str__(self):
        return "Node for idx " + str(self.label_idx) + " (label: " + str(self.label_str) + ")" + \
             "\n<-Parent nodes: " + ",".join([str(p.label_idx) + "(" + str(p.label_str) + ")" for p in self.in_nodes]) + \
             "\n->Child nodes: " + ",".join([str(c.label_idx) + "(" + str(c.label_str) + ")" for c in self.out_nodes]) 

class HierarchyKnowledge:
    nodes: Dict[int, Node]
    extended_vocab: Dict[str, int]
    child_col_name: str
    parent_col_name: str

    def __init__(self, child_col_name='child', parent_col_name='parent'):
        self.child_col_name = child_col_name
        self.parent_col_name = parent_col_name

    def build_hierarchy_from_df(self, hierarchy_df: pd.DataFrame, vocab: Dict[str, int]):
        self._build_extended_vocab(hierarchy_df, vocab)
        for _,row in tqdm(hierarchy_df.iterrows(), desc='Building Hierarchy from df'):
            child_node = self.nodes[self.extended_vocab[row[self.child_col_name]]]
            parent_node = self.nodes[self.extended_vocab[row[self.parent_col_name]]]

            child_node.in_nodes.append(parent_node)
            parent_node.out_nodes.append(child_node)

    def _build_extended_vocab(self, hierarchy_df: pd.DataFrame, vocab: Dict[str, int]):
        self.extended_vocab = {}
        self.nodes = {}

        labels_to_handle = list(vocab.keys()) + \
            list(set(hierarchy_df[self.child_col_name])) + \
            list(set(hierarchy_df[self.parent_col_name]))
        max_index = max(vocab.values())
        for label in labels_to_handle:
            if label in self.extended_vocab:
                continue

            if label in vocab:
                self.extended_vocab[label] = vocab[label]
            else:
                self.extended_vocab[label] = max_index + 1
                max_index = max_index + 1

            self.nodes[self.extended_vocab[label]] = Node(
                label_idx=self.extended_vocab[label], 
                label_str=label)
