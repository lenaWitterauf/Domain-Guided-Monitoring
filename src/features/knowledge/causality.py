import pandas as pd
from typing import Dict, Set
from tqdm import tqdm
import logging
from .node import Node
from .base import BaseKnowledge


class CausalityKnowledge(BaseKnowledge):
    def __init__(
        self,
        child_id_col="child_id",
        parent_id_col="parent_id",
        child_name_col="child_name",
        parent_name_col="parent_name",
    ):
        self.child_id_col = child_id_col
        self.parent_id_col = parent_id_col
        self.child_name_col = child_name_col
        self.parent_name_col = parent_name_col

    def get_connections_for_idx(self, idx: int) -> Set[int]:
        return set(
            self.nodes[idx].get_neighbour_label_idxs() + [idx]
        )

    def get_description_vocab(
        self, ids: Set[int]
    ) -> Dict[int, str]:
        return {
            idx: node.label_name for idx, node in self.nodes.items() if idx in ids
        }

    def build_causality_from_df(
        self, causality_df: pd.DataFrame, vocab: Dict[str, int]
    ):
        self.vocab: Dict[str, int] = vocab
        self._build_extended_vocab(causality_df, vocab)
        for _, row in tqdm(causality_df.iterrows(), desc="Building Causality from df"):
            child_id = row[self.child_id_col]
            if child_id not in self.extended_vocab:
                logging.debug("Ignoring node %s as not in dataset", child_id)
                continue

            parent_id = row[self.parent_id_col]
            if parent_id not in self.extended_vocab:
                logging.debug("Ignoring node %s as not in dataset", parent_id)
                continue

            child_node = self.nodes[self.extended_vocab[child_id]]
            parent_node = self.nodes[self.extended_vocab[parent_id]]

            child_node.in_nodes.add(parent_node)
            parent_node.out_nodes.add(child_node)

        logging.info("Built causality with %d nodes", len(self.nodes))

    def _build_extended_vocab(self, causality_df: pd.DataFrame, vocab: Dict[str, int]):
        self.extended_vocab: Dict[str, int] = {}
        self.nodes: Dict[int, Node] = {}

        labels_to_handle = list(vocab.keys())
        max_index = max(vocab.values())
        for label in labels_to_handle:
            if label in self.extended_vocab:
                continue

            if label in vocab:
                self.extended_vocab[label] = vocab[label]

                parents_df = causality_df[causality_df[self.child_id_col] == label]
                parents = list(set(parents_df[self.parent_id_col]))
                labels_to_handle = labels_to_handle + parents

                child_df = causality_df[causality_df[self.parent_id_col] == label]
                children = list(set(child_df[self.child_id_col]))
                labels_to_handle = labels_to_handle + children
            else:
                self.extended_vocab[label] = max_index + 1
                max_index = max_index + 1

            label_names = set(
                causality_df[causality_df[self.child_id_col] == label][
                    self.child_name_col
                ]
            )
            label_names.update(
                set(
                    causality_df[causality_df[self.parent_id_col] == label][
                        self.parent_name_col
                    ]
                )
            )

            self.nodes[self.extended_vocab[label]] = Node(
                label_idx=self.extended_vocab[label],
                label_str=label,
                label_names=label_names,
            )

        if max_index == max(vocab.values()):
            logging.debug("Adding VOID node to ensure extended vocab > vocab")
            self.extended_vocab["_VOID_"] = max_index + 1
            self.nodes[self.extended_vocab["_VOID_"]] = Node(
                label_idx=self.extended_vocab["_VOID_"],
                label_str="_VOID_",
                label_names=set(["_VOID_"]),
            )

        self.extra_vocab: Dict[str, int] = {
            k: v for k, v in self.extended_vocab.items() if k not in self.vocab
        }

    def __str__(self):
        all_strings = []
        for node in self.nodes.values():
            all_strings = all_strings + self._to_neighbour_string(node)
        return "\n".join(all_strings)

    def _to_neighbour_string(self, node: Node):
        strings = [node.label_str]
        for in_node in node.in_nodes:
            strings.append("<-" + in_node.label_str)
        for out_node in node.out_nodes:
            strings.append("->" + out_node.label_str)

        return strings
