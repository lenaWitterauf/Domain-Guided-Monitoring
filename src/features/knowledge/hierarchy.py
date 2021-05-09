import pandas as pd
from typing import Dict, List, Set
from tqdm import tqdm
import logging
from .node import Node

class HierarchyKnowledge:
    nodes: Dict[int, Node]
    vocab: Dict[str, int]
    extended_vocab: Dict[str, int]
    child_col_name: str
    parent_col_name: str

    def __init__(self, child_col_name='child', parent_col_name='parent'):
        self.child_col_name = child_col_name
        self.parent_col_name = parent_col_name

    def build_hierarchy_from_df(self, hierarchy_df: pd.DataFrame, vocab: Dict[str, int]):
        self.vocab = vocab
        self._build_extended_vocab(hierarchy_df, vocab)
        for _,row in tqdm(hierarchy_df.iterrows(), desc='Building Hierarchy from df'):
            child_name = row[self.child_col_name]
            if child_name not in self.extended_vocab:
                logging.debug('Ignoring node %s as not in dataset', child_name)
                continue

            child_node = self.nodes[self.extended_vocab[row[self.child_col_name]]]
            parent_node = self.nodes[self.extended_vocab[row[self.parent_col_name]]]

            child_node.in_nodes.add(parent_node)
            parent_node.out_nodes.add(child_node)
        
        logging.info('Built hierarchy with %d nodes', len(self.nodes))

    def _build_extended_vocab(self, hierarchy_df: pd.DataFrame, vocab: Dict[str, int]):
        self.extended_vocab = {}
        self.nodes = {}

        labels_to_handle = list(vocab.keys())
        max_index = max(vocab.values())
        while len(labels_to_handle) > 0:
            label = labels_to_handle.pop()
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

            parents_df = hierarchy_df[hierarchy_df[self.child_col_name] == label]
            parents = list(set(parents_df[self.parent_col_name]))
            labels_to_handle = labels_to_handle + parents

    def __str__(self):
        roots = [node for node in self.nodes.values() if node.is_root()]
        all_strings = []
        for root in roots:
            all_strings = all_strings + self._to_string_recursive(root, '')
        return '\n'.join(all_strings)

    def _to_string_recursive(self, current_node, current_prefix):
        strings = [current_prefix + current_node.label_str]
        for node in current_node.out_nodes:
            strings = strings + self._to_string_recursive(node, current_prefix + '-')

        return strings
