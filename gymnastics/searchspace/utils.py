import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


__all__ = ["CellConfiguration", "Node", "Edges", "Dataset"]


@dataclass
class Dataset:
    name: str
    num_classes: int
    image_size: Tuple[int, int, int]


CIFAR10 = Dataset("CIFAR-10", 10, (3, 32, 32))
CIFAR100 = Dataset("CIFAR-100", 100, (3, 32, 32))
ImageNet16_120 = Dataset("ImageNet-16-120", 10, (3, 32, 32))


@dataclass(unsafe_hash=True)
class Node:
    id: int
    channels: Optional[int] = None
    feature_map: Optional[torch.Tensor] = None
    label: Optional[str] = None


@dataclass(unsafe_hash=True, order=True)
class Edge:
    from_node_id: str
    to_node_id: str
    op: nn.Module
    connected_to_input: Optional[bool] = False
    connected_to_output: Optional[bool] = False
    label: Optional[str] = None


@dataclass
class CellConfiguration:
    edges: Dict[int, Edge]
    nodes: Dict[int, Node]
    input_node_ids: List
    output_node_id: int
    adjacency_matrix: Optional[np.ndarray]


def is_valid_cell(cell_config, start: List[int], end: int) -> bool:
    """A breadth-first traversal of the cell graph. This method will
        check for two things: (1) whether there is a path connected the
        input to the output, and (2) whether there are any cycles in the graph.

    Args:
        cell_config ([type]): A CellConfiguration containing the graph details
        start (List[int]): The input node(s)
        end (int): The output node

    Returns:
        bool: found_path indicating whether a path was found connecting the input
        to the output
    """

    queue = [[s] for s in start]
    visited = set()

    found_path = False
    while queue:

        path = queue.pop(0)
        node = path[-1]
        visited.add(node)

        if node == end:
            found_path = True

        for neighbour_index, op in enumerate(cell_config.adjacency_matrix[node]):

            if neighbour_index > node:
                if neighbour_index in path:  # then there's a cycle
                    return False

                if (neighbour_index not in visited) and (op != -1):
                    queue.append(path + [neighbour_index])

    return found_path
