import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ["Node", "Edge"]


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
    label: Optional[str] = None


class Cell(nn.Module):
    def __init__(self, edges, nodes, input_node_ids=[0], output_node_id: int = None):
        super(Cell, self).__init__()
        self.nodes: Dict = nodes
        self.edges: Dict = edges

        self.input_node_ids = input_node_ids

        if output_node_id is None:
            output_node_id = len(nodes) - 1

        self.output_node_id = output_node_id

    def forward(self, x, return_logits=False):

        # set all feature maps to zero
        for node in self.nodes.values():
            if node.feature_map is not None:
                node.feature_map = torch.zeros(node.feature_map.size())

        # accumulate the inputs
        for node_id in self.input_node_ids:
            self.nodes[node_id].feature_map = x

        # do the main forward pass of the cell
        for edge in self.edges.values():

            if self.nodes[edge.to_node_id].feature_map is not None:
                self.nodes[edge.to_node_id].feature_map += edge.op(
                    self.nodes[edge.from_node_id].feature_map
                )
            else:
                self.nodes[edge.to_node_id].feature_map = edge.op(
                    self.nodes[edge.from_node_id].feature_map
                )

        # return whatever the output is
        return self.nodes[self.output_node_id].feature_map
