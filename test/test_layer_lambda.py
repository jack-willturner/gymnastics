import torch
from gymnastics.models import ResNet18, ResNet26


def test_lambda_layer_with_basic_block():
    def generate_lambda_config():

        # RESNET-18 SKELETON
        n_blocks = [2, 2, 2, 2]
        strides = [1, 2, 2, 2]

        config = {}

        for s, (stage, stride) in enumerate(zip(n_blocks, strides)):

            stage_config = {}

            for b in range(stage):

                block_config = {}

                # two "layers"
                block_config["layer1"] = {
                    "conv_type": "Conv",
                    "layer_args": {
                        "stride": stride if b == 0 else 1,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": False,
                    },
                }

                block_config["layer2"] = {
                    "conv_type": "LambdaConv",
                    "layer_args": {
                        "stride": 1,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": False,
                    },
                }

                stage_config[f"block{b+1}"] = block_config

            config[f"stage{s+1}"] = stage_config

        return config

    config = generate_lambda_config()

    net = ResNet18(config)

    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


def test_lambda_layer_with_bottleneck():
    def generate_lambda_config():

        # RESNET-26 SKELETON
        n_blocks = [2, 2, 2, 2]
        strides = [1, 2, 2, 2]

        config = {}

        for s, (stage, stride) in enumerate(zip(n_blocks, strides)):

            stage_config = {}

            for b in range(stage):

                block_config = {}

                # two "layers"
                block_config["layer1"] = {
                    "conv_type": "Conv",
                    "layer_args": {
                        "stride": 1,
                        "kernel_size": 1,
                        "padding": 0,
                        "bias": False,
                    },
                }

                block_config["layer2"] = {
                    "conv_type": "LambdaConv",
                    "layer_args": {
                        "stride": 1,
                        "kernel_size": 3,
                        "padding": 1,
                        "bias": False,
                    },
                }

                block_config["layer3"] = {
                    "conv_type": "Conv",
                    "layer_args": {
                        "stride": 1,
                        "kernel_size": 1,
                        "padding": 0,
                        "bias": False,
                    },
                }

                stage_config[f"block{b+1}"] = block_config

            config[f"stage{s+1}"] = stage_config

        return config

    config = generate_lambda_config()

    net = ResNet26(config)

    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
