from abc import ABC, abstractmethod


class SearchSpace(ABC):
    def __init__(self, cell_space, skeleton):
        self.cell_space = cell_space
        self.skeleton = skeleton

    @abstractmethod
    def sample_random_architecture(self):
        # get skeleton properties (n_blocks, strides)
        # generate genotype from cell_space
        # build network
        # return it
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def calc_length(self):
        raise NotImplementedError

    def __len__(self):
        if self.len is None:
            self.calc_length()

        return self.len
