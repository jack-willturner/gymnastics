import torch
import numpy as np
import torch.nn as nn
from proxies.proxy import Proxy


class NASWOT(Proxy):
    def get_K(self, model, minibatch):
        batch_size = minibatch.size()[0]

        model.K = np.zeros((batch_size, batch_size))

        def counting_forward_hook(module, input, output):
            try:
                if not module.visited_backwards:
                    return

                if isinstance(input, tuple):
                    input = input[0]
                input = input.view(input.size(0), -1)
                x = (input > 0).float()
                K = x @ x.t()
                K2 = (1.0 - x) @ (1.0 - x.t())

                model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()
            except:
                pass

        def counting_backward_hook(module, input, output):
            module.visited_backwards = True

            for _, module in model.named_modules():
                if "ReLU" in str(type(module)):
                    module.register_forward_hook(counting_forward_hook)
                    module.register_backward_hook(counting_backward_hook)

    def score(self, model: nn.Module, minibatch: torch.Tensor) -> float:
        K = self.get_K(model, minibatch)
        _, logdet = np.linalg.slogdet(K)

        return logdet
