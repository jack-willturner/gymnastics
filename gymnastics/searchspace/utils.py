import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


__all__ = ["CellConfiguration", "Node", "Edges"]


@dataclass(unsafe_hash=True)
class Node:
    id: int
    feature_map: Optional[torch.Tensor] = None
    label: Optional[str] = None


@dataclass(unsafe_hash=True)
class Edge:
    op: nn.Module
    from_node_id: str
    to_node_id: str
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
