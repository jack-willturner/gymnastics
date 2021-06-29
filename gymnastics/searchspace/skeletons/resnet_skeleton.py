from gymnastics.searchspace.utils import CellConfiguration
import torch
import torch.nn as nn
from gymnastics.searchspace import Cell, Skeleton


class ResNetCIFARSkeleton(Skeleton):
    def __init__(self, cell_config, num_blocks, num_classes=10, block_expansion=4):
        super(ResNetCIFARSkeleton, self).__init__()

        self.in_channels = 64

        self.block_expansion = block_expansion

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        self.layer1 = self._make_layer(
            cell_config,
            64,
            num_blocks[0],
            stride=1,
        )
        self.layer2 = self._make_layer(
            cell_config,
            128,
            num_blocks[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            cell_config,
            256,
            num_blocks[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            cell_config,
            512,
            num_blocks[3],
            stride=2,
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(512 * block_expansion, num_classes),
        )

    def _make_layer(self, cell_config, planes, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(
                Cell(
                    cell_config,
                    in_channels=self.in_channels,
                    out_channels=planes,
                    stride=stride,
                    expansion=self.block_expansion,
                )
            )

            self.in_channels = planes * self.block_expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(out.size(0), -1)

        return self.classifier(out)


class NASBench101Skeleton:
    def build_with_cell(self, cell_config: CellConfiguration) -> Skeleton:
        return ResNetCIFARSkeleton(cell_config, [3, 3, 3, 3])


class NASBench201Skeleton:
    def build_with_cell(self, cell_config: CellConfiguration) -> Skeleton:
        return ResNetCIFARSkeleton(cell_config, [1, 1, 1, 1])
