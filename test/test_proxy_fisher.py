import time
import torch

from gymnastics.benchmarks import NDSSearchSpace
from gymnastics.proxies import Proxy, get_proxy 


local_search_space_loc = "/Users/jackturner/work/nds/data/ResNet.json"
search_space_loc = "/disk/scratch_ssd/nasbench201/nds/"


def test_proxy_naswot():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    searchspace = NDSSearchSpace(
        search_space_loc, searchspace="ResNet"
    )

    model = searchspace.sample_random_architecture()
    
    minibatch: torch.Tensor = torch.rand(10, 3, 32, 32)
    target: torch.Tensor = torch.empty(10, dtype=torch.long).random_(10)

    minibatch, target = minibatch.to(device), target.to(device)
    model = model.to(device)

    y, acts = model(minibatch, get_ints=True)
    proxy: Proxy = get_proxy("Fisher")

    t1 = time.time()
    score = proxy.score(model, minibatch, target)
    t2 = time.time()

    assert (t2-t1) < 10.

    assert score > 0
