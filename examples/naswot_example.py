from gymnastics.datasets import CIFAR10Loader
from gymnastics.proxies import NASWOT
from gymnastics.searchspace import SearchSpace, NASBench201Skeleton, CellSpace
from gymnastics.searchspace.ops import Conv3x3, Conv1x1, AvgPool2d, Skip, Zeroize

# use the 201 skeleton with the 201 cell space
search_space = SearchSpace(
    CellSpace(
        ops=[Conv3x3, Conv1x1, AvgPool2d, Skip, Zeroize], num_nodes=4, num_edges=6
    ),
    NASBench201Skeleton(),
)

# create an accuracy predictor
proxy = NASWOT()
dataset = CIFAR10Loader(path="~/datasets/cifar10", download=False)

minibatch, _ = dataset.sample_minibatch()

best_score = 0.0
best_model = None

# try out 10 random architectures and save the best one
for i in range(1):

    model = search_space.sample_random_architecture()

    for name, mod in model.named_modules():
        print(mod)
        break

    y = model(minibatch)

    score = proxy.score(model, minibatch)

    if score > best_score:
        best_score = score
        best_model = model
