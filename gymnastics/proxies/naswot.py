import torch
import torch.nn as nn
from gymnastics.proxies.proxy import Proxy


class NASWOT(Proxy):
    def add_hooks(self, model) -> None:
        def counting_forward_hook(module, input, output):
            try:
                if not module.visited_backwards:
                    return

                # in case it's inputs + targets
                if isinstance(input, tuple):
                    input = input[0]

                device = input.get_device()

                # flatten inputs into vectors
                input = input.view(input.size(0), -1)

                # binary indicator for which units are switched on/off
                x = (input > 0).float()

                K = x @ x.t()  # xs that are 1 and the same
                K2 = (1.0 - x) @ (1.0 - x.t())  # xs that are 0 and the same

                K = K.to(device)
                K2 = K2.to(device)

                model.K = model.K + K.cpu() + K2.cpu()
            except Exception as e:
                pass

        def counting_backward_hook(module, input, output):
            module.visited_backwards = True

        for _, module in model.named_modules():
            if "ReLU" in str(type(module)):
                module.register_forward_hook(counting_forward_hook)
                module.register_backward_hook(counting_backward_hook)

    def get_K(self, model, minibatch) -> torch.Tensor:
        device = minibatch.get_device()
        batch_size = minibatch.size()[0]

        model.K = torch.zeros((batch_size, batch_size))

        self.add_hooks(model)

        model = model.to(device)

        # make a clone of the minibatch
        minibatch2 = torch.clone(minibatch)
        minibatch2 = minibatch2.to(device)

        # attach the forward/backward hooks
        model.zero_grad()
        y, _ = model(minibatch)
        y.backward(torch.ones_like(y))

        # push the minibatch through
        model(minibatch2)
        return model.K

    def score(
        self, model: nn.Module, minibatch: torch.Tensor, target: torch.Tensor
    ) -> float:
        K = self.get_K(model, minibatch)
        logdet = torch.logdet(K)

        return logdet
