import torch
import torch.nn as nn
from gymnastics.proxies.proxy import Proxy


class Fisher(Proxy):
    def score(
        self, model: nn.Module, minibatch: torch.Tensor, target: torch.Tensor
    ) -> float:

        criterion = nn.CrossEntropyLoss()

        out, activations = model(minibatch, get_ints=True)

        for activation in activations:
            activation.retain_grad()

        loss = criterion(out, target)
        loss.backward()

        fishers = []

        for activation in activations:
            print(activation)
            fish = (
                (activation.data.detach() * activation.grad.detach())
                .sum(-1)
                .sum(-1)
                .pow(2)
                .mean(0)
                .sum()
            )
            fishers.append(fish)

        return sum(fishers)
