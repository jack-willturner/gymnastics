import numpy as np

from typing import List
from gymnastics.searchspace import Node, Edge, Cell


class CellSpace:
    def __init__(
        self,
        ops: List,
        num_nodes: int,
        fully_connected: bool,
    ) -> None:
        raise NotImplementedError

    def generate_random_cell(self):

        nodes = {}
        edges = {}

        for node_id in self.num_nodes:
            nodes[node_id] = Node(node_id)

        node_ids = list(nodes.keys())

        for edge_num in self.num_edges:

            op = np.random.choice(self.ops)
            from_node, to_node = np.random.choice(node_ids, 2, replace=False)

            edges[edge_num] = Edge(op, from_node, to_node)

        return Cell(edges, nodes)
