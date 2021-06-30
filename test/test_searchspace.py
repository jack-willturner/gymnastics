from gymnastics.datasets import CIFAR10Loader
from gymnastics.searchspace import SearchSpace, NASBench101Skeleton, CellSpace
from gymnastics.searchspace.ops import Conv3x3, Conv1x1, AvgPool2d, Identity, Zeroize


def test_search_space():

    # use the 101 skeleton with the 201 cell space
    search_space = SearchSpace(
        CellSpace(
            ops=[Conv3x3, Conv1x1, AvgPool2d, Identity, Zeroize],
            num_nodes=4,
            num_edges=6,
        ),
        NASBench101Skeleton(),
    )

    # create an accuracy predictor
    dataset = CIFAR10Loader(path="~/datasets/cifar10", download=False)

    minibatch, _ = dataset.sample_minibatch()

    # try out 10 random architectures and save the best one
    for i in range(10):

        model = search_space.sample_random_architecture()

        _ = model(minibatch)
