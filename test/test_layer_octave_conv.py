def generate_bottleneck_config():

    # RESNET-26 SKELETON
    n_blocks = [2, 2, 2, 2]
    strides = [1, 2, 2, 2]

    config = {}

    for s, (stage, stride) in enumerate(zip(n_blocks, strides)):

        stage_config = {}

        if s == 0:
            t = "first"
        elif s == len(n_blocks) - 1:
            t = "last"
        else:
            t = "normal"

        for b in range(stage):

            block_config = {}

            block_config["layer1"] = {
                "conv_type": "OctaveConv",
                "layer_args": {
                    "stride": 1,
                    "kernel_size": 1,
                    "padding": 0,
                    "bias": False,
                    "type": t,
                },
            }

            block_config["layer2"] = {
                "conv_type": "OctaveConv",
                "layer_args": {
                    "stride": stride if b == 0 else 1,
                    "kernel_size": 3,
                    "padding": 1,
                    "bias": False,
                    "type": t,
                },
            }

            block_config["layer3"] = {
                "conv_type": "OctaveConv",
                "layer_args": {
                    "stride": 1,
                    "kernel_size": 1,
                    "padding": 0,
                    "bias": False,
                    "type": t,
                },
            }

            stage_config[f"block{b+1}"] = block_config

        config[f"stage{s+1}"] = stage_config

    return config


def test_octave_conv_resnet_26():

    """
    from models import ResNet26

    config = generate_bottleneck_config()

    net = ResNet26(config)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    """
    print("OCTAVE CONV IGNORED FOR NOW")
