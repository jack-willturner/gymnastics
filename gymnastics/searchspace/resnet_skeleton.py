import torch.nn as nn
from gymnastics.searchspace import Cell, Skeleton


class ResNetCIFARSkeleton(Skeleton):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCIFARSkeleton, self).__init__()

        self.cell = block

        self.in_planes = 64

        self.stem = nn.Sequential(
            [
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
            ]
        )

        self.layers = nn.Sequential(
            [
                self._make_layer(block, 64, num_blocks[0], stride=1),
                self._make_layer(block, 128, num_blocks[1], stride=2),
                self._make_layer(block, 256, num_blocks[2], stride=2),
                self._make_layer(block, 512, num_blocks[3], stride=2),
            ]
        )

        self.classifier = nn.Sequential(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Linear(512 * block.expansion, num_classes),
            ]
        )

    def _make_layer(self, block, planes, configs):

        layers = []

        for layer_config in configs:
            layers.append(block(self.in_planes, planes, configs[layer_config]))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


class NASBench101Skeleton:
    def build_with_cell(self, cell: Cell) -> Skeleton:
        return ResNetCIFARSkeleton(cell, [3, 3, 3, 3])


class NASBench201Skeleton:
    def build_with_cell(self, cell: Cell) -> Skeleton:
        return ResNetCIFARSkeleton(cell, [3, 3, 3, 3])
