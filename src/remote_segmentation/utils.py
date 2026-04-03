# -*- coding: utf-8 -*-

import os
import torch

__all__ = ["load_resnet_weights", "conv_out_size", "transpose_conv_out_size"]


def load_resnet_weights(name: str):
    path = os.path.join(os.sep.join(__file__.split(os.sep)[:-1]), "pretrained")
    files = os.listdir(path)
    fname = list(filter(lambda x: name in x, files))[0]
    fpath = os.path.join(path, fname)

    return torch.load(fpath)


def conv_out_size(
    input_size,
    kernel_size,
    padding,
    stride=1,
    dilation=1,
):
    output_size = (
        input_size + 2 * padding + stride - dilation * (kernel_size - 1) - 1
    ) / stride

    return output_size


def transpose_conv_out_size(
    input_size,
    kernel_size,
    padding,
    stride=1,
    dilation=1,
):
    output_size = (
        input_size * stride + 1 + dilation * (kernel_size - 1) - stride - 2 * padding
    )

    return output_size


class AsyncWandBLogger:
    def __init__(self, run):
        self.run = run

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)


if __name__ == "__main__":
    input_size = 224
    kernel_size = 7
    padding = 1
    stride = 2
    dilation = 1

    # input_size = 32
    # kernel_size = 5
    # padding = 0
    # stride = 1
    # dilation = 1

    out = conv_out_size(
        input_size=input_size,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
    )

    transpose_out = transpose_conv_out_size(
        input_size=out,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
    )

    print(out)
    print(transpose_out)
