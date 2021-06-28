class SearchSpace:
    def __init__(self, cell_space, skeleton_generation_function):
        self.cell_space = cell_space
        self.skeleton_builder = skeleton_generation_function

    def sample_random_architecture(self):
        cell = self.cell_space.generate_random_cell_configuration()
        return self.skeleton_builder.build_with_cell(cell)

    def __iter__(self):
        raise NotImplementedError

    def calc_length(self):
        raise NotImplementedError

    def __len__(self):
        if self.len is None:
            self.calc_length()

        return self.len
