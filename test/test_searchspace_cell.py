from gymnastics.searchspace.ops import Conv1x1, Conv3x3
import torch
import torch.nn as nn
from gymnastics.searchspace.utils import CellConfiguration, Node, Edge
from gymnastics.searchspace.cell import Cell


def test_simple_celltype():

    nodes = {
        0: Node(0),
        1: Node(1),
        2: Node(2),
    }

    edges = {
        0: Edge(op=Conv3x3, from_node_id=0, to_node_id=1, label="0"),
        1: Edge(op=Conv3x3, from_node_id=1, to_node_id=2, label="1"),
        2: Edge(op=Conv1x1, from_node_id=0, to_node_id=2, label="2"),
    }

    dummy_cell = Cell(
        CellConfiguration(
            edges=edges,
            nodes=nodes,
            input_node_ids=[0],
            output_node_id=2,
            adjacency_matrix=None,
        ),
        in_channels=3,
        hidden_planes=64,
        out_channels=64,
        stride=1,
    )

    dummy_input = torch.rand((1, 3, 32, 32))
    o = dummy_cell(dummy_input)

    print(o.size())


def test_edge_sorting():

    a = Edge(op=nn.Conv2d, from_node_id=3, to_node_id=4)
    b = Edge(op=nn.AvgPool2d, from_node_id=1, to_node_id=3)
    c = Edge(op=nn.Identity, from_node_id=2, to_node_id=3)

    edges = [a, b, c]

    assert sorted(edges) == [b, c, a]


def test_ops():

    data = torch.rand((1, 3, 32, 32))
    conv1x1 = Conv1x1(3, 10)

    o = conv1x1(data)
    assert o.size() == (1, 10, 32, 32)
