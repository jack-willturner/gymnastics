import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import get_conv_bn_relu


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, layer_config):
        super(BasicBlock, self).__init__()

        """ 
        CONV -> BN -> RELU
        """
        conv, bn, relu = get_conv_bn_relu(layer_config["layer1"]["conv_type"])
        stride = layer_config["layer1"]["layer_args"]["stride"]
        self.conv1 = conv(in_planes, planes, **layer_config["layer1"]["layer_args"])
        self.bn1 = bn(planes)
        self.relu1 = relu()

        """ 
        CONV -> BN -> RELU
        """
        conv, bn, relu = get_conv_bn_relu(layer_config["layer2"]["conv_type"])
        self.conv2 = conv(planes, planes, **layer_config["layer2"]["layer_args"])
        self.bn2 = bn(planes)
        self.relu2 = relu()

        """ 
        CONV -> BN
        """
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, layer_config):
        super(Bottleneck, self).__init__()

        conv1, bn, relu = get_conv_bn_relu(layer_config["layer1"]["conv_type"])
        self.conv1 = conv1(in_planes, planes, **layer_config["layer1"]["layer_args"])
        self.bn1 = bn(planes)
        self.relu1 = relu()

        conv2, bn, relu = get_conv_bn_relu(layer_config["layer2"]["conv_type"])
        stride = layer_config["layer2"]["layer_args"]["stride"]
        self.conv2 = conv2(planes, planes, **layer_config["layer2"]["layer_args"])
        self.bn2 = bn(planes)
        self.relu2 = relu()

        conv3, bn, relu = get_conv_bn_relu(layer_config["layer3"]["conv_type"])
        self.conv3 = conv3(
            planes, self.expansion * planes, **layer_config["layer3"]["layer_args"]
        )
        self.bn3 = bn(self.expansion * planes)
        self.relu3 = relu()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, configs=None, num_classes=10):
        super(ResNet, self).__init__()

        # cache the configs for later
        self.configs = configs
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, configs["stage1"])
        self.layer2 = self._make_layer(block, 128, configs["stage2"])
        self.layer3 = self._make_layer(block, 256, configs["stage3"])
        self.layer4 = self._make_layer(block, 512, configs["stage4"])
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, configs):

        layers = []

        for layer_config in configs:
            layers.append(block(self.in_planes, planes, configs[layer_config]))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(configs):
    return ResNet(BasicBlock, [2, 2, 2, 2], configs)


def ResNet26(configs):
    return ResNet(Bottleneck, [2, 2, 2, 2], configs)


def ResNet34(configs):
    return ResNet(BasicBlock, [3, 4, 6, 3], configs)


def ResNet50(configs):
    return ResNet(Bottleneck, [3, 4, 6, 3], configs)


def ResNet101(configs):
    return ResNet(Bottleneck, [3, 4, 23, 3], configs)


def ResNet152(configs):
    return ResNet(Bottleneck, [3, 8, 36, 3], configs)
