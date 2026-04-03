# -*- coding: utf-8 -*-

import torch
from remote_segmentation.architectures import UNet

__all__ = ["test_unet_architecture"]


def test_unet_architecture():
    in_channels = 1
    out_channels = 32
    num_blocks = 4
    num_classes = 10

    B, H, W = 1, 572, 572

    x = torch.randn(B, in_channels, H, W).cuda()
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        num_classes=num_classes,
    ).to("cuda")

    y = model(x)

    print(f"In: {x.shape}")
    print(f"Out: {y.shape}")

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)

    s = summary(
        model,
        input_size=(1, in_channels, 512, 512),  # (batch_size, channels, height, width)
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ],
        col_width=20,
        depth=2,
        row_settings=["var_names"],
    )

    print(list(model.parameters()))

    return y.shape[-2:] == x.shape[-2:]
