import collections

import numpy as np
import torch
import torch.nn
import torch.nn.functional
import torch.nn.functional as F
from einops import rearrange


class SELayer(torch.nn.Module):
    """Squeeze-and-Excitation Layer"""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.channel = channel
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass of the SE layer."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CenteredBatchNorm2d(torch.nn.BatchNorm2d):
    """Only apply bias, no scale like:
    tf.layers.batch_normalization(
        center=True, scale=False,
        )
    """

    def __init__(self, channels):
        """Initialize the CenteredBatchNorm2d."""
        super().__init__(channels, affine=True, eps=1e-5, momentum=0.01)
        # #self.weight = 1 by default
        # self.weight.requires_grad = False


class ConvBlock(torch.nn.Module):
    """Convolutional block with batch norm and ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        """Initialize the ConvBlock."""
        super().__init__()
        layers = [
            (
                "conv2d",
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=bias,
                ),
            ),
            ("norm2d", CenteredBatchNorm2d(out_channels)),
            ("ReLU", torch.nn.ReLU()),
        ]
        self.seq = torch.nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x):
        """Forward pass of the ConvBlock."""
        return self.seq(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        layers = [
            (
                "conv2d_1",
                torch.nn.Conv2d(
                    channels,
                    channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            ),
            ("norm2d_1", CenteredBatchNorm2d(channels)),
            ("ReLU", torch.nn.ReLU()),
            (
                "conv2d_2",
                torch.nn.Conv2d(
                    channels,
                    channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            ),
            ("norm2d_2", CenteredBatchNorm2d(channels)),
            ("squeeze_excite", SELayer(channels, 8)),
        ]
        self.seq = torch.nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x):
        y = self.seq(x)
        y += x
        y = torch.nn.functional.relu(y, inplace=True)
        return y