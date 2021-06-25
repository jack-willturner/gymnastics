from abc import ABC, abstractmethod


class Proxy(ABC):
    @abstractmethod
    def score(self, **kwargs) -> float:
        """This function should return a scalar value which "scores" the network.
        It can take anything you like as input.

        Raises:
            NotImplementedError: your proxy must at least have this function!

        Returns:
            float: the scalar value which scores the network
        """
        raise NotImplementedError
