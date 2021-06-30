![gymNAStics](figures/capybara.png)

<p align="center">
  <!-- license -->
  <a href="https://tldrlegal.com/license/apache-license-2.0-%28apache-2.0%29">
      <img src="https://img.shields.io/github/license/jack-willturner/gymNAStics" alt="License" height="20">
  </a>
  <!-- CI status -->
  <a href="">
    <img src="https://img.shields.io/github/workflow/status/jack-willturner/gymNAStics/CI" alt="CI status" height="20">
  </a>
  <!-- Code analysis -->
  <img src="https://img.shields.io/lgtm/grade/python/github/jack-willturner/gymNAStics" alt="Code analysis" height="20">
  <!-- Getting started colab -->
  <a href="">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20">
  </a>
</p>

<p align="center">
  <i>A "gym" style toolkit for building lightweight Neural Architecture Search systems. I know, the name is awful. </i>
</p>

## Installation 

```bash
git clone git@github.com:jack-willturner/gymNAStics.git
cd gymNAStics
conda env create --file=environment.yml
conda activate gymnastics
```

If you want to use NAS-Bench-101, follow the instructions [here](https://github.com/google-research/nasbench).


## Overview

Over the course of the final year of my PhD I worked a lot on Neural Architecture Search (NAS) and built a bunch of tooling to make my life easier. This is an effort to standardise the various features into a single framework and provide a "gym" style toolkit for comparing various algorithms.

The key use cases for this library are:
- test out new predictors on various NAS benchmarks 
- visualise the cells/graphs of your architectures
- add new operations to NAS spaces 
- add new backbones to NAS spaces

The framework revolves around three key classes:
1. `Model` 
2. `Proxy`
3. `SearchSpace`


### Obligatory builder pattern README example

Using `gymnastics` we can very easily reconstruct NAS spaces (the goal being that it's easy to define new and exciting ones):

```python
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
for i in range(10):

    model = search_space.sample_random_architecture()

    y = model(minibatch)

    score = proxy.score(model, minibatch)

    if score > best_score:
        best_score = score
        best_model = model

best_model.show_picture()
```

Which prints:

![](figures/best_model.png)

Have a look in `examples/` for more examples.

### NAS-Benchmarks 

If you have designed a new proxy for accuracy and want to test its performance, you can use the benchmarks available in `benchmarks/`.  

The interface to the benchmarks is exactly the same as the above example for `SearchSpace`.

For example, here we score networks from the NDS ResNet space using random input data:

```python
import torch
from gymnastics.benchmarks import NDSSearchSpace
from gymnastics.proxies import Proxy, NASWOT

search_space = NDSSearchSpace(
    "~/nds/data/ResNet.json", searchspace="ResNet"
)

proxy: Proxy = NASWOT()
minibatch: torch.Tensor = torch.rand((10, 3, 32, 32))

scores = []

for _ in range(10):
    model = search_space.sample_random_architecture()
    scores.append(proxy.score(model, minibatch))
```

## Roadmap

1. Primary focus is to have (well-tested) support for at least:
   - [ ] NAS-Bench-101
   - [x] NAS-Bench-201
   - [ ] NAS-Bench-301
   - [ ] NATS-Bench
   - [x] NDS
   - [ ] NAS-Bench-NLP
2. Next stage will be to add the database connections for each 
3. Some additional tooling like the grapher, an experiment manager for benchmarks
4. Add in other proxies/predictors
5. Training pipeline (probably using some kind of hyperparameter sweep)
6. Test out more exotic NAS spaces
7. Add in standardised NAS implementations

## Additional supported operations

| Done | Tested | Op                  | Paper                                         | Notes                                                               |
| ---- | ------ | ------------------- | --------------------------------------------- | ------------------------------------------------------------------- |
| [x]  |        | conv                | -                                             | params: kernel size                                                 |
| [x]  |        | gconv               | AlexNet                                       | + params: group                                                     |
| [x]  |        | depthwise separable | [pdf](https://arxiv.org/pdf/1610.02357v3.pdf) |                                                                     |
| [x]  |        | mixconv             | [pdf](https://arxiv.org/pdf/1907.09595.pdf)   |                                                                     |
| [x]  |        | octaveconv          | [pdf](https://arxiv.org/pdf/1904.05049.pdf)   | Don't have a sensible way to include this as a single operation yet |
| [ ]  |        | shift               | [pdf](https://arxiv.org/pdf/1711.08141.pdf)   |                                                                     |
| [ ]  |        | ViT                 |                                               |                                                                     |
| [ ]  |        | Fused-MBConv        | [pdf](https://arxiv.org/pdf/2104.00298.pdf)   |                                                                     |
| [x]  |        | Lambda              | [pdf](https://arxiv.org/pdf/2102.08602.pdf)   |                                                                     |
| [ ]  |        |                     |                                               |                                                                     |
