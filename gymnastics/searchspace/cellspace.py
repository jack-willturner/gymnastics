import numpy as np

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

        adjacency_matrix = np.zeros(shape=(num_nodes, num_nodes))

        # create a dictionary of empty nodes
        for node_id in range(num_nodes):
            nodes[node_id] = Node(node_id)

        # get the node ids, so we can choose edges to randomly connect them
        node_ids = list(nodes.keys())

        # randomly choose an op to connect two random nodes
        for edge_num in range(num_edges):

            # select a random op
            op_index = np.random.randint(0, len(self.ops))
            op = self.ops[op_index]

            # select two random nodes (without replacement)
            from_node, to_node = np.random.choice(node_ids, 2, replace=False)
            edges[edge_num] = Edge(op, from_node, to_node)

            # log this to the adjacency matrix
            adjacency_matrix[from_node, to_node] = op_index

        def validate_cell_config(cell_config):
            pass

            # ensure there are no cycles

            # ensure there is a path from input to output

        def breadth_first_traversal(cell_config, start: int, end: int):

            queue = [[start]]

            while queue:

                path = queue.pop(0)
                node = path[-1]

                if node == end:
                    return path

                for neighbour in adjacency_matrix[node]:

                    if neighbour in path:
                        return None

                    queue.append(path + [neighbour])

            return None

        return CellConfiguration(
            edges,
            nodes,
            input_node_ids=[0],
            output_node_id=node_ids[-1],
            adjacency_matrix=adjacency_matrix,
        )


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
