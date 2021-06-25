import torch.nn as nn


class Skeleton(nn.Module):
    def __init__(
        self,
    ):
        super(Skeleton, self).__init__()

    def forward(self, x):
        out = self.stem(x)

        for layer in self.layers:
            out = layer(out)

        out = out.view(out.size(0), -1)

        return self.classifier(out)
