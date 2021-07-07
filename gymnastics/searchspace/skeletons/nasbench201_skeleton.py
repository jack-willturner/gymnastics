from gymnastics.searchspace import Skeleton
from .resnet_skeleton import ResNetCIFAR


def NASBench201Skeleton() -> Skeleton:
    return Skeleton(
        skeleton_type=ResNetCIFAR,
        num_blocks=[5, 5, 5],
        channels_per_stage=[16, 32, 64],
        strides_per_stage=[1, 2, 2],
        block_expansion=1,
    )
