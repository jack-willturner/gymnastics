import torch
import torch.nn as nn


##############################################################################################
# This code is copied and modified from Hanxiao Liu's work (https://github.com/quark0/darts) #
##############################################################################################
class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            # assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False)
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 4:
            assert C_out % 4 == 0, "C_out : {:}".format(C_out)
            self.convs = nn.ModuleList()
            for i in range(4):
                self.convs.append(
                    nn.Conv2d(C_in, C_out // 4, 1, stride=stride, padding=0, bias=False)
                )
            self.pad = nn.ConstantPad2d((0, 3, 0, 3), 0)
        else:
            raise ValueError("Invalid stride : {:}".format(stride))

        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        if self.stride == 2:
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = torch.cat(
                [
                    self.convs[0](x),
                    self.convs[1](y[:, :, 1:-2, 1:-2]),
                    self.convs[2](y[:, :, 2:-1, 2:-1]),
                    self.convs[3](y[:, :, 3:, 3:]),
                ],
                dim=1,
            )
        out = self.bn(out)
        return out

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


class Skip(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=None,
        stride=1,
        bias=False,
        padding=1,
    ):
        super(Skip, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = FactorizedReduce(
                C_in=in_channels, C_out=out_channels, stride=stride, affine=True
            )

    def forward(self, x):
        return self.skip(x)

    def __str__(self):
        return f"Skip()"

    def __repr__(self):
        return f"Skip()"
