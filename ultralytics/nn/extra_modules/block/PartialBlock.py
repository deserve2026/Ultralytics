"""
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/PartialBlock.png
论文链接：https://arxiv.org/pdf/2303.03667.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")

import warnings

warnings.filterwarnings("ignore")
import torch
from calflops import calculate_flops
from torch import nn

from ultralytics.nn.extra_modules.mamba.SFMB import SFMB
from ultralytics.nn.modules import Conv


class PartialBlock(nn.Module):
    def __init__(self, inc, ouc, n_div=4):
        super().__init__()

        self.partial_channels = inc // n_div
        self.identity_channels = inc - self.partial_channels

        self.partial_module = Conv(self.partial_channels, self.partial_channels, 3)
        # self.partial_module = DySnakeConv(self.partial_channels, self.partial_channels, 3)
        # self.partial_module = ADIE(self.partial_channels, self.partial_channels)
        # self.partial_module = SFMB(self.partial_channels)

        self.conv_adjust = Conv(inc, ouc, 1) if inc != ouc else nn.Identity()

    def forward(self, x):
        x1, x2 = torch.split(x, (self.partial_channels, self.identity_channels), 1)
        x1 = self.partial_module(x1)
        y = torch.cat([x1, x2], 1)
        y = self.conv_adjust(y)
        return y


class PartialBlock_SFMB(nn.Module):
    def __init__(self, inc, ouc, n_div=4):
        super().__init__()

        self.partial_channels = inc // n_div
        self.identity_channels = inc - self.partial_channels

        self.partial_module = SFMB(self.partial_channels)

        self.conv_adjust = Conv(inc, ouc, 1) if inc != ouc else nn.Identity()

    def forward(self, x):
        x1, x2 = torch.split(x, (self.partial_channels, self.identity_channels), 1)
        x1 = self.partial_module(x1)
        y = torch.cat([x1, x2], 1)
        y = self.conv_adjust(y)
        return y


if __name__ == "__main__":
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = (
        "\033[91m",
        "\033[92m",
        "\033[94m",
        "\033[93m",
        "\033[38;5;208m",
        "\033[0m",
    )
    from ultralytics.utils.torch_utils import select_device

    device_id = "0"
    batch_size, in_channel, out_channel, height, width = 1, 128, 256, 160, 160

    torch_device = select_device(device_id)
    inputs_tensor = torch.randn((batch_size, in_channel, height, width)).to(torch_device)

    module = PartialBlock(in_channel, out_channel, 4).to(torch_device)
    # module = PartialBlock_SFMB(in_channel, out_channel, 4).to(torch_device)
    module.eval()

    outputs = module(inputs_tensor)
    print(GREEN + f"inputs.size:{inputs_tensor.size()} outputs.size:{outputs.size()}" + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(
        model=module,
        input_shape=(batch_size, in_channel, height, width),
        output_as_string=True,
        output_precision=4,
        print_detailed=True,
    )
    print(RESET)
