import numpy as np

from typing import List
from gymnastics.searchspace import Node, Edge, Cell
from gymnastics.searchspace.ops import (
    Conv3x3,
    Conv1x1,
    AvgPool2d,
    MaxPool2d,
    Identity,
    Zeroize,
)


class CellSpace:
    def __init__(
        self,
        ops: List,
        num_nodes: int,
        num_edges: int,
    ) -> None:
        self.ops = ops
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def generate_random_cell(self):

        nodes = {}
        edges = {}

        ## if num_nodes/edges are lists, choose from them at random
        num_nodes = (
            self.num_nodes
            if isinstance(self.num_nodes, int)
            else np.random.choice(self.num_nodes)
        )

        num_edges = (
            self.num_edges
            if isinstance(self.num_edges, int)
            else np.random.choice(self.num_edges)
        )

        # create a dictionary of empty nodes
        for node_id in num_nodes:
            nodes[node_id] = Node(node_id)

        # get the node ids, so we can choose edges to randomly connect them
        node_ids = list(nodes.keys())

        for edge_num in num_edges:

            op = np.random.choice(self.ops)
            from_node, to_node = np.random.choice(node_ids, 2, replace=False)

            edges[edge_num] = Edge(op, from_node, to_node)

        return Cell(edges, nodes)


def NASBench101CellSpace() -> CellSpace:
    return CellSpace(
        ops=[Conv3x3, Conv1x1, MaxPool2d],
        num_nodes=[1, 2, 3, 4, 5, 6, 7],
        num_edges=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )


def NASBench201CellSpace() -> CellSpace:
    return CellSpace(
        ops=[Conv3x3, Conv1x1, AvgPool2d, Identity, Zeroize], num_nodes=4, num_edges=6
    )
