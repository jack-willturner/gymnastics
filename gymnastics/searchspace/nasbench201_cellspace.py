from gymnastics.searchspace import Cell, Node, Edge
from gymnastics.searchspace import CellSpace


class NASBench201CellSpace(CellSpace):
    def __init__(
        self,
        ops: List,
        num_input_nodes: int,
        num_hidden_nodes: int,
        num_output_nodes: int,
        fully_connected: bool,
    ) -> None:
        self.ops = ops
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes
        self.fully_connected = fully_connected

    def sample_random_cell(self):
        pass
