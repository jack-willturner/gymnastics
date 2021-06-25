import yaml
import numpy as np
from typing import Dict, List, Union
from models.layers import layer_type_registry


class BasicBlockGenerator:
    def generate_block_config(
        self, conv_type: Union[List[str], str] = "Conv", stride: int = 1
    ) -> Dict:
        if not isinstance(conv_type, list):
            conv_type = [conv_type, conv_type]

        block_config = {}

        # two "layers"
        block_config["layer1"] = {
            "conv_type": conv_type[0],
            "layer_args": {
                "stride": stride,
                "kernel_size": 3,
                "padding": 1,
                "bias": False,
            },
        }
        block_config["layer2"] = {
            "conv_type": conv_type[1],
            "layer_args": {
                "stride": 1,
                "kernel_size": 3,
                "padding": 1,
                "bias": False,
            },
        }

        return block_config


class BottleneckGenerator:
    def generate_block_config(
        self, conv_type: Union[List[str], str], stride: int = 1
    ) -> Dict:

        if not isinstance(conv_type, list):
            conv_type = [conv_type, conv_type, conv_type]

        block_config = {}

        block_config["layer1"] = {
            "conv_type": conv_type[0],
            "layer_args": {
                "stride": 1,
                "kernel_size": 1,
                "padding": 0,
                "bias": False,
            },
        }

        block_config["layer2"] = {
            "conv_type": conv_type[1],
            "layer_args": {
                "stride": stride,
                "kernel_size": 3,
                "padding": 1,
                "bias": False,
            },
        }

        block_config["layer3"] = {
            "conv_type": conv_type[2],
            "layer_args": {
                "stride": 1,
                "kernel_size": 1,
                "padding": 0,
                "bias": False,
            },
        }

        return block_config


def generate_baseline_genotype(
    generator, n_blocks: List[int], strides: List[int], conv_type: str = "Conv"
) -> Dict:

    config = {}

    for s, (stage, stride) in enumerate(zip(n_blocks, strides)):

        stage_config = {}

        for b in range(stage):

            block_config = generator.generate_block_config(
                conv_type=conv_type, stride=stride if b == 0 else 1
            )

            stage_config[f"block{b+1}"] = block_config

        config[f"stage{s+1}"] = stage_config

    return config


def generate_baseline_genotype_with_random_middle_convs(
    generator: Union[BasicBlockGenerator, BottleneckGenerator],
    n_blocks: List[int],
    strides: List[int],
    conv_types: List[str] = layer_type_registry,
) -> Dict:
    """This function generates a genotype for a ResNet where the middle
       convolution of the block is substituted for a random new operation.

    Args:
        generator (Union[BasicBlockGenerator, BottleneckGenerator]): A block config generator
        n_blocks (List[int]): A list of length 4, describing how many blocks should go in each stage
        strides (List[int]): The stride for the first convolution of each stage of the network
        conv_types: (List[str]): Randomly choose a convolution from this list

    Returns:
        Dict: A config dictionary which can be passed into a ResNet to generate a network
    """

    config: Dict = {}

    for s, (stage, stride) in enumerate(zip(n_blocks, strides)):

        stage_config: Dict = {}

        for b in range(stage):

            # randomly sample a conv_type
            conv_type = np.random.choice(conv_types)

            block_config: Dict = generator.generate_block_config(
                ["Conv", conv_type, "Conv"], stride if b == 0 else 1
            )

            stage_config[f"block{b+1}"] = block_config

        config[f"stage{s+1}"] = stage_config

    return config


def generate_resnet_genotypes(conv_type: str = "Conv", save: bool = False) -> Dict:

    generated_configs = {}
    baselines = {
        "resnet18": (BasicBlockGenerator, [2, 2, 2, 2], [1, 2, 2, 2]),
        "resnet26": (BottleneckGenerator, [2, 2, 2, 2], [1, 2, 2, 2]),
        "resnet34": (BasicBlockGenerator, [3, 4, 6, 3], [1, 2, 2, 2]),
    }

    for name, (generator, n_blocks, strides) in baselines.items():
        config = generate_baseline_genotype(generator(), n_blocks, strides, conv_type)

        if save:
            with open(f"genotypes/{name}.yaml", "w") as file:
                yaml.dump(config, file)

        generated_configs[name] = config

    return generated_configs
