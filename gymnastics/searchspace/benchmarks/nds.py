import json
import torch
import random

import torch.nn as nn

from gymnastics.searchspace import SearchSpace
from gymnastics.searchspace.utils import Dataset, CIFAR10

from pycls.models.nas.nas import NetworkImageNet, NetworkCIFAR
from pycls.models.anynet import AnyNet
from pycls.models.nas.genotypes import Genotype


class ReturnFeatureLayer(torch.nn.Module):
    def __init__(self, mod):
        super(ReturnFeatureLayer, self).__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod(x), x


def return_feature_layer(network, prefix=""):

    for n, ch in list(network.named_children()):
        if isinstance(ch, torch.nn.Linear):
            setattr(network, n, ReturnFeatureLayer(ch))
        else:
            return_feature_layer(ch, prefix + "\t")


class NDSSearchSpace(SearchSpace):
    def __init__(
        self,
        path_to_api: str = None,
        dataset: Dataset = CIFAR10,
        searchspace: str = None,
    ):
        self.dataset = dataset

        api = json.load(open(path_to_api, "r"))
        try:
            api = api["top"] + api["mid"]
        except Exception as e:
            print(e)
        self.api = api

        self.searchspace = searchspace

    def sample_random_architecture(self) -> nn.Module:
        arch_id = random.randint(0, len(self) - 1)
        model = self.get_network(arch_id)
        model.arch_id = arch_id
        return model

    def get_network(self, arch_id: str) -> nn.Module:
        netinfo = self.api[arch_id]
        config = netinfo["net"]
        if "genotype" in config:
            gen = config["genotype"]
            genotype = Genotype(
                normal=gen["normal"],
                normal_concat=gen["normal_concat"],
                reduce=gen["reduce"],
                reduce_concat=gen["reduce_concat"],
            )
            if "_in" in self.searchspace:
                network = NetworkImageNet(
                    config["width"], 1, config["depth"], config["aux"], genotype
                )
            else:
                network = NetworkCIFAR(
                    config["width"], 1, config["depth"], config["aux"], genotype
                )
            network.drop_path_prob = 0.0

        else:
            if "bot_muls" in config and "bms" not in config:
                config["bms"] = config["bot_muls"]
                del config["bot_muls"]
            if "num_gs" in config and "gws" not in config:
                config["gws"] = config["num_gs"]
                del config["num_gs"]
            config["nc"] = 1
            config["se_r"] = None
            config["stem_w"] = 12
            if "ResN" in self.searchspace:
                config["stem_type"] = "res_stem_in"
            else:
                config["stem_type"] = "simple_stem_in"

            if config["block_type"] == "double_plain_block":
                config["block_type"] = "vanilla_block"
            network = AnyNet(**config)

        return_feature_layer(network)
        return network

    def random_arch(self):
        return random.randint(0, len(self.data) - 1)

    def get_final_accuracy(self, arch_id):
        return 100.0 - self.api[arch_id]["test_ep_top1"][-1]

    def get_network_config(self, arch_id):
        return self.api[arch_id]["net"]

    def get_network_optim_config(self, arch_id):
        return self.api[arch_id]["optim"]

    def __iter__(self):
        for unique_hash in range(len(self)):
            network = self.get_network(unique_hash)
            yield unique_hash, network

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.api)
