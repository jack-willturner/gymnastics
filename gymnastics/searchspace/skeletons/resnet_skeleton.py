from gymnastics.searchspace.utils import CellConfiguration
import torch
import torch.nn as nn
from gymnastics.searchspace import Cell, Skeleton


class ResNetCIFARSkeleton(Skeleton):
    def __init__(
        self,
        cell_config,
        num_blocks,
        num_classes=10,
        channels_per_stage=[64, 128, 256, 512],
        strides=[1, 2, 2, 2],
        block_expansion=4,
    ):
        super(ResNetCIFARSkeleton, self).__init__()

        self.in_channels = channels_per_stage[0]

        self.block_expansion = block_expansion

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.in_channels),
        )

        stages = []
        for i, out_channels in enumerate(channels_per_stage):
            stages.append(
                self._make_layer(
                    cell_config,
                    out_channels,
                    num_blocks[i],
                    stride=strides[i],
                    expansion=self.block_expansion,
                )
            )

        stages.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.stages = nn.Sequential(*stages)

        self.classifier = nn.Sequential(
            nn.Linear(channels_per_stage[-1] * block_expansion, num_classes),
        )

    def _make_layer(self, cell_config, planes, num_blocks, stride, expansion=1):

        layers = [
            Cell(
                cell_config,
                in_channels=self.in_channels,
                out_channels=planes,
                stride=stride,
                expansion=expansion,
            )
        ]

        self.in_channels = planes * self.block_expansion

        for _ in range(1, num_blocks):
            layers.append(
                Cell(
                    cell_config,
                    in_channels=self.in_channels,
                    out_channels=planes,
                    expansion=expansion,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)

        out = self.stages(out)

        out = out.view(out.size(0), -1)

        return self.classifier(out)


class NASBench101Skeleton:
    def build_with_cell(self, cell_config: CellConfiguration) -> Skeleton:
        return ResNetCIFARSkeleton(cell_config, [2, 2, 2, 2])


class NASBench201Skeleton:
    def build_with_cell(self, cell_config: CellConfiguration) -> Skeleton:
        return ResNetCIFARSkeleton(
            cell_config,
            [5, 5, 5, 5],
            channels_per_stage=[16, 32, 64],
            strides=[1, 2, 2, 2],
            block_expansion=1,
        )
