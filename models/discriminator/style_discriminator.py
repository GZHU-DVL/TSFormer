import torch
from torch import nn
import numpy as np
from .stylegan_ops import style_function
from . import base_function


class StyleDiscriminator(nn.Module):
    def __init__(self, img_size, ndf=32, blur_kernel=[1, 3, 3, 1], use_attn=False):
        super(StyleDiscriminator, self).__init__()

        channel_multiplier = ndf / 64
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: int(512 * channel_multiplier),
            64: int(256 * channel_multiplier),
            128: int(128 * channel_multiplier),
            256: int(64 * channel_multiplier),
            512: int(32 * channel_multiplier),
            1024: int(16 * channel_multiplier),
        }

        convs = [style_function.ConvLayer(3, channels[img_size], 1)]

        log_size = int(np.log2(img_size))

        in_channel = channels[img_size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i-1)]
            if i == log_size - 3 and use_attn:
                convs.append(base_function.AttnAware(in_channel))
            convs.append(style_function.StyleBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = style_function.ConvLayer(in_channel+1, channels[4], 3)
        self.final_linear = nn.Sequential(
            style_function.EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            style_function.EqualLinear(channels[4], 1),
        )

    def forward(self, x):

        out = self.convs(x)

        b, c, h, w = out.shape
        group = min(b, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, c // self.stddev_feat, h, w)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, h, w)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(b, -1)
        out = self.final_linear(out)

        return out