import random
import numpy as np
from itertools import combinations
from typing import List
from gymnastics.searchspace.utils import Node, Edge, CellConfiguration
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

        # this is a dictionary which converts indices into op names
        self.op_encoder = {i: str(op) for i, op in enumerate(ops)}
        self.op_encoder[-1] = "None"

    def generate_random_cell_configuration(self) -> CellConfiguration:

        # store the cell in two formats
        # 1. a list of nodes and edges
        # 2. an adjacency matrix
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

        # set adj matrix to -1s to indicate no operation
        adjacency_matrix = np.zeros(shape=(num_nodes, num_nodes), dtype=int) - 1

        # create a dictionary of empty nodes
        for node_id in range(num_nodes):
            nodes[node_id] = Node(node_id)

        # get all possible pairs of nodes and shuffle them
        possible_pairs_of_nodes = list(combinations(list(nodes.keys()), 2))
        random.shuffle(possible_pairs_of_nodes)

        # randomly choose an op to connect two random nodes
        for edge_num in range(num_edges):

            # select a random op
            op_index = np.random.randint(0, len(self.ops) - 1)
            op = self.ops[op_index]

            from_node, to_node = possible_pairs_of_nodes[edge_num]

            edges[edge_num] = Edge(from_node_id=from_node, to_node_id=to_node, op=op)

            # log this to the adjacency matrix
            adjacency_matrix[from_node, to_node] = op_index

        cell_config = CellConfiguration(
            edges,
            nodes,
            input_node_ids=[0],
            output_node_id=num_nodes - 1,
            adjacency_matrix=adjacency_matrix,
        )

        return cell_config


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
