import copy
import torch.nn as nn

from gymnastics.searchspace.utils import CellConfiguration
from typing import Dict

__all__ = ["Cell"]


class Cell(nn.Module):
    def __init__(
        self,
        cell_config: CellConfiguration,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: int = 4,
    ):
        super(Cell, self).__init__()
        self.nodes: Dict = copy.deepcopy(cell_config.nodes)
        self.edges: Dict = copy.deepcopy(cell_config.edges)

        self.input_node_ids = cell_config.input_node_ids

        if cell_config.output_node_id is None:
            cell_config.output_node_id = len(self.nodes) - 1

        self.output_node_id = cell_config.output_node_id

        # register which nodes are connected to input/output
        for edge in self.edges.values():
            if edge.from_node_id in self.input_node_ids:
                edge.connected_to_input = True
                # if it's connected to the input then it must have in_channels channels
                self.nodes[edge.from_node_id].channels = in_channels

            if edge.to_node_id == self.output_node_id:
                edge.connected_to_output = True

        self.expansion = expansion

        self.configure_sizes(in_channels, out_channels, stride=stride)

    def configure_sizes(self, in_channels, out_channels, **kwargs) -> None:

        for edge in self.edges.values():

            if edge.connected_to_input:
                edge.op = edge.op(in_channels, out_channels, **kwargs)
                self.nodes[edge.to_node_id].channels = edge.op.out_channels
            else:
                edge.op = edge.op(
                    self.nodes[edge.from_node_id].channels, out_channels, **kwargs
                )
                self.nodes[edge.to_node_id].channels = edge.op.out_channels

    def forward(self, x, return_logits=False):

        # set all feature maps to zero
        for node in self.nodes.values():
            if node.feature_map is not None:
                node.feature_map = None

        # accumulate the inputs
        for node_id in self.input_node_ids:
            self.nodes[node_id].feature_map = x

        # do the main forward pass of the cell
        for edge in sorted(self.edges.values()):

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

    def __repr__(self):

        s = "Cell("
        for edge in self.edges.values():
            s = (
                s
                + str(edge.from_node_id)
                + "-"
                + str(edge.op)
                + "->"
                + str(edge.to_node_id)
                + "\n"
            )
        s = s + ")"

        return s
