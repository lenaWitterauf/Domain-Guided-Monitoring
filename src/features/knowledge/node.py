from typing import List, Set

class Node:
    label_idx: int
    label_str: str
    in_nodes: Set['Node']
    out_nodes: Set['Node']

    def __init__(self, label_idx, label_str):
        self.label_idx=label_idx
        self.label_str=label_str
        self.in_nodes=set()
        self.out_nodes=set()

    def is_root(self) -> bool:
        return len(self.in_nodes) == 0

    def is_leaf(self) -> bool:
        return len(self.out_nodes) == 0

    def get_neighbours(self) -> List['Node']:
        return list(set(
            list(self.in_nodes) + list(self.out_nodes) + [self]
        ))

    def get_neighbour_label_idxs(self) -> List[int]:
        neighbours = self.get_neighbours()
        return [neighbour.label_idx for neighbour in neighbours]

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
