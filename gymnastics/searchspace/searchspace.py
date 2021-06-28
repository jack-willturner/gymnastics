import torch.nn as nn
from .utils import CellConfiguration, is_valid_cell


class SearchSpace:
    def __init__(self, cell_space, skeleton_generation_function):
        self.cell_space = cell_space
        self.skeleton_builder = skeleton_generation_function

    def sample_random_architecture(self) -> nn.Module:

        found_valid_cell = False

        while not found_valid_cell:
            cell_config: CellConfiguration = (
                self.cell_space.generate_random_cell_configuration()
            )

            print(cell_config.adjacency_matrix)

            found_valid_cell = is_valid_cell(
                cell_config=cell_config,
                start=cell_config.input_node_ids,
                end=cell_config.output_node_id,
            )

        return self.skeleton_builder.build_with_cell(cell_config)

    def __iter__(self):
        raise NotImplementedError

    def calc_length(self):
        raise NotImplementedError

    def __len__(self):
        if self.len is None:
            self.calc_length()

        return self.len
