import torch.nn as nn
from gymnastics.searchspace.cellspace import CellConfiguration


class Skeleton(nn.Module):
    def __init__(
        self,
        skeleton_type=None,
        num_blocks=[2, 2, 2, 2],
        channels_per_stage=[64, 128, 256, 512],
        strides_per_stage=[1, 2, 2, 2],
        block_expansion=1,
    ):
        super(Skeleton, self).__init__()
        self.skeleton_type = skeleton_type
        self.num_blocks = num_blocks
        self.channels_per_stage = channels_per_stage
        self.strides_per_stage = strides_per_stage
        self.block_expansion = block_expansion

    def build_with_cell(self, cell_config: CellConfiguration) -> nn.Module:
        return self.skeleton_type(
            cell_config,
            num_blocks=self.num_blocks,
            channels_per_stage=self.channels_per_stage,
            strides_per_stage=self.strides_per_stage,
            block_expansion=self.block_expansion,
        )

    def show_picture(self):
        raise NotImplementedError
