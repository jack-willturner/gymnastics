import torch
import torch.nn as nn
from gymnastics.searchspace import Cell, Node, Edge


def test_simple_celltype():

    nodes = {
        0: Node(0),
        1: Node(1),
        2: Node(2),
    }

    edges = {
        0: Edge(
            op=nn.Conv2d(3, 64, 3, padding=1), from_node_id=0, to_node_id=1, label="0"
        ),
        1: Edge(
            op=nn.Conv2d(64, 64, 3, padding=1), from_node_id=1, to_node_id=2, label="1"
        ),
        2: Edge(op=nn.Conv2d(3, 64, 1, padding=0), from_node_id=0, to_node_id=2),
    }

    dummy_cell = Cell(edges, nodes)

    dummy_input = torch.rand((1, 3, 32, 32))
    o = dummy_cell(dummy_input)

    print(o.size())
