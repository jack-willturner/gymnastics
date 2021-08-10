import time
import torch

from gymnastics.benchmarks import NDSSearchSpace
from gymnastics.proxies import Proxy, NASWOT


local_search_space_lot = "/Users/jackturner/work/nds/data/ResNet.json"
search_space_loc = "/disk/scratch_ssd/nasbench201/nds/ResNet.json"


def test_proxy_naswot():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    searchspace = NDSSearchSpace(search_space_loc, searchspace="ResNet")

    model = searchspace.sample_random_architecture(num_classes=1)

    minibatch: torch.Tensor = torch.rand(10, 3, 32, 32)

    minibatch = minibatch.to(device)
    model = model.to(device)

    proxy: Proxy = NASWOT()

    t1 = time.time()
    score = proxy.score(model, minibatch)
    t2 = time.time()

    assert (t2 - t1) < 10.0

    assert score > 0
