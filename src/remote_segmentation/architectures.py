# -*- coding: utf-8 -*-

import torch

from typing import List, Sequence, Optional

from torch import Tensor
from torch import nn
from torch.nn import functional as F

from torchvision.transforms import CenterCrop
from torchvision.ops import DropBlock2d

from torchinfo import summary

from remote_segmentation.utils import load_resnet_weights

__all__ = ["UNetBlock", "UNetEncoder", "Bottleneck", "UNetDecoder", "UNet"]


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
        pool: bool = False,
        base: bool = False,
        downsample: bool = False,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            # stride=stride,
            stride=2 if downsample else stride,
            dilation=dilation,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = DropBlock2d(
            p=dropout,
            block_size=3,
        )  # TODO: Be sure to use an ODD block size! Even block sizes throw exceptions!
        self.buffer = None
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=2,
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(out_channels),
            )
            if downsample
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x_ = self.downsample(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.dropout(x)

        return x + x_

    def clear(self) -> None:
        self.buffer = None


class ResnetLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        num_blocks: int = 2,
        bias: bool = False,
        pool: bool = False,
        downsample: bool = False,
    ):
        super().__init__()

        # self.block1 = ResnetBlock(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=kernel_size,
        #     padding=padding,
        #     stride=stride,
        #     dilation=dilation,
        #     dropout=dropout,
        #     bias=bias,
        #     pool=pool,
        #     downsample=downsample,
        # )
        # self.block2 = ResnetBlock(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     kernel_size=kernel_size,
        #     padding=padding,
        #     stride=stride,
        #     dilation=dilation,
        #     dropout=dropout,
        #     bias=bias,
        #     pool=pool,
        #     downsample=False,
        # )

        blocks = [
            ResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                pool=pool,
                downsample=i == 0,
            )
            for i in range(num_blocks)
        ]

        self.blocks = nn.Sequential(*blocks)
        self.buffer = None

        # self.downsample = (
        #     nn.Sequential(
        #         nn.Conv2d(
        #             in_channels=in_channels,
        #             out_channels=out_channels,
        #             kernel_size=1,
        #             padding=padding,
        #             stride=stride,
        #             dilation=dilation,
        #             bias=bias,
        #         ),
        #         nn.BatchNorm2d(out_channels),
        #     )
        #     if downsample
        #     else nn.Identity()
        # )

    def forward(self, x: Tensor) -> Tensor:
        # x = self.block1(x) + self.downsample(x)
        # x = self.block1(x) + x
        # x = self.block2(x) + x

        # x = self.block1(x)
        # x = self.block2(x)

        x = self.blocks(x)
        self.buffer = x

        return x

    def clear(self) -> None:
        self.buffer = None


class Resnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        num_blocks_per_layer: List[int] = [2, 2, 2, 2],
        bias: bool = False,
        pool: bool = False,
    ):
        super().__init__()

        assert len(num_blocks_per_layer) == 4, "Model can only contain four layers!"

        num_base_channels = out_channels // 2 ** (len(num_blocks_per_layer) - 1)

        self.pre_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                num_base_channels,
                kernel_size=7,
                padding=0,
                stride=2,
                dilation=1,
            ),
            nn.BatchNorm2d(num_base_channels),
        )

        self.layer1 = ResnetLayer(
            in_channels=num_base_channels,
            out_channels=num_base_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            num_blocks=num_blocks_per_layer[0],
            bias=bias,
            pool=pool,
            downsample=False,
        )

        self.layer2 = ResnetLayer(
            in_channels=num_base_channels,
            out_channels=num_base_channels * 2,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            num_blocks=num_blocks_per_layer[1],
            bias=bias,
            pool=pool,
            downsample=True,
        )

        self.layer3 = ResnetLayer(
            in_channels=num_base_channels * 2,
            out_channels=num_base_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            num_blocks=num_blocks_per_layer[2],
            bias=bias,
            pool=pool,
            downsample=True,
        )

        self.layer4 = ResnetLayer(
            in_channels=num_base_channels * 4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            num_blocks=num_blocks_per_layer[3],
            bias=bias,
            pool=pool,
            downsample=True,
        )

    def load_weights(self, weights):
        ### Pre-block
        self.pre_block[0].weight = weights["conv1.weight"]
        self.pre_block[1].weight = weights["bn1.weight"]
        self.pre_block[1].bias = weights["bn1.bias"]
        self.pre_block[1].running_var = weights["bn1.running_var"]
        self.pre_block[1].running_mean = weights["bn1.running_mean"]

        ### Layer 1

        self.layer1.blocks[0].conv1.weight = weights["layer1.0.conv1.weight"]
        self.layer1.blocks[0].conv2.weight = weights["layer1.0.conv2.weight"]
        self.layer1.blocks[0].bn1.weight = weights["layer1.0.bn1.weight"]
        self.layer1.blocks[0].bn1.bias = weights["layer1.0.bn1.bias"]
        self.layer1.blocks[0].bn1.running_var = weights["layer1.0.bn1.running_var"]
        self.layer1.blocks[0].bn1.running_mean = weights["layer1.0.bn1.running_mean"]
        self.layer1.blocks[0].bn2.weight = weights["layer1.0.bn2.weight"]
        self.layer1.blocks[0].bn2.bias = weights["layer1.0.bn2.bias"]
        self.layer1.blocks[0].bn2.running_var = weights["layer1.0.bn2.running_var"]
        self.layer1.blocks[0].bn2.running_mean = weights["layer1.0.bn2.running_mean"]

        self.layer1.blocks[1].conv1.weight = weights["layer1.1.conv1.weight"]
        self.layer1.blocks[1].conv2.weight = weights["layer1.1.conv2.weight"]
        self.layer1.blocks[1].bn1.weight = weights["layer1.1.bn1.weight"]
        self.layer1.blocks[1].bn1.bias = weights["layer1.1.bn1.bias"]
        self.layer1.blocks[1].bn1.running_var = weights["layer1.1.bn1.running_var"]
        self.layer1.blocks[1].bn1.running_mean = weights["layer1.1.bn1.running_mean"]
        self.layer1.blocks[1].bn2.weight = weights["layer1.1.bn2.weight"]
        self.layer1.blocks[1].bn2.bias = weights["layer1.1.bn2.bias"]
        self.layer1.blocks[1].bn2.running_var = weights["layer1.1.bn2.running_var"]
        self.layer1.blocks[1].bn2.running_mean = weights["layer1.1.bn2.running_mean"]

        ### Layer 2

        self.layer2.blocks[0].conv1.weight = weights["layer2.0.conv1.weight"]
        self.layer2.blocks[0].conv2.weight = weights["layer2.0.conv2.weight"]
        self.layer2.blocks[0].bn1.weight = weights["layer2.0.bn1.weight"]
        self.layer2.blocks[0].bn1.bias = weights["layer2.0.bn1.bias"]
        self.layer2.blocks[0].bn1.running_var = weights["layer2.0.bn1.running_var"]
        self.layer2.blocks[0].bn1.running_mean = weights["layer2.0.bn1.running_mean"]
        self.layer2.blocks[0].bn2.weight = weights["layer2.0.bn2.weight"]
        self.layer2.blocks[0].bn2.bias = weights["layer2.0.bn2.bias"]
        self.layer2.blocks[0].bn2.running_var = weights["layer2.0.bn2.running_var"]
        self.layer2.blocks[0].bn2.running_mean = weights["layer2.0.bn2.running_mean"]

        self.layer2.blocks[0].downsample[0].weight = weights[
            "layer2.0.downsample.0.weight"
        ]
        self.layer2.blocks[0].downsample[1].weight = weights[
            "layer2.0.downsample.1.weight"
        ]
        self.layer2.blocks[0].downsample[1].bias = weights["layer2.0.downsample.1.bias"]
        self.layer2.blocks[0].downsample[1].running_var = weights[
            "layer2.0.downsample.1.running_var"
        ]
        self.layer2.blocks[0].downsample[1].running_mean = weights[
            "layer2.0.downsample.1.running_mean"
        ]

        self.layer2.blocks[1].conv1.weight = weights["layer2.1.conv1.weight"]
        self.layer2.blocks[1].conv2.weight = weights["layer2.1.conv2.weight"]
        self.layer2.blocks[1].bn1.weight = weights["layer2.1.bn1.weight"]
        self.layer2.blocks[1].bn1.bias = weights["layer2.1.bn1.bias"]
        self.layer2.blocks[1].bn1.running_var = weights["layer2.1.bn1.running_var"]
        self.layer2.blocks[1].bn1.running_mean = weights["layer2.1.bn1.running_mean"]
        self.layer2.blocks[1].bn2.weight = weights["layer2.1.bn2.weight"]
        self.layer2.blocks[1].bn2.bias = weights["layer2.1.bn2.bias"]
        self.layer2.blocks[1].bn2.running_var = weights["layer2.1.bn2.running_var"]
        self.layer2.blocks[1].bn2.running_mean = weights["layer2.1.bn2.running_mean"]

        ### Layer 3

        self.layer3.blocks[0].conv1.weight = weights["layer3.0.conv1.weight"]
        self.layer3.blocks[0].conv2.weight = weights["layer3.0.conv2.weight"]
        self.layer3.blocks[0].bn1.weight = weights["layer3.0.bn1.weight"]
        self.layer3.blocks[0].bn1.bias = weights["layer3.0.bn1.bias"]
        self.layer3.blocks[0].bn1.running_var = weights["layer3.0.bn1.running_var"]
        self.layer3.blocks[0].bn1.running_mean = weights["layer3.0.bn1.running_mean"]
        self.layer3.blocks[0].bn2.weight = weights["layer3.0.bn2.weight"]
        self.layer3.blocks[0].bn2.bias = weights["layer3.0.bn2.bias"]
        self.layer3.blocks[0].bn2.running_var = weights["layer3.0.bn2.running_var"]
        self.layer3.blocks[0].bn2.running_mean = weights["layer3.0.bn2.running_mean"]

        self.layer3.blocks[0].downsample[0].weight = weights[
            "layer3.0.downsample.0.weight"
        ]
        self.layer3.blocks[0].downsample[1].weight = weights[
            "layer3.0.downsample.1.weight"
        ]
        self.layer3.blocks[0].downsample[1].bias = weights["layer3.0.downsample.1.bias"]
        self.layer3.blocks[0].downsample[1].running_var = weights[
            "layer3.0.downsample.1.running_var"
        ]
        self.layer3.blocks[0].downsample[1].running_mean = weights[
            "layer3.0.downsample.1.running_mean"
        ]

        self.layer3.blocks[1].conv1.weight = weights["layer3.1.conv1.weight"]
        self.layer3.blocks[1].conv2.weight = weights["layer3.1.conv2.weight"]
        self.layer3.blocks[1].bn1.weight = weights["layer3.1.bn1.weight"]
        self.layer3.blocks[1].bn1.bias = weights["layer3.1.bn1.bias"]
        self.layer3.blocks[1].bn1.running_var = weights["layer3.1.bn1.running_var"]
        self.layer3.blocks[1].bn1.running_mean = weights["layer3.1.bn1.running_mean"]
        self.layer3.blocks[1].bn2.weight = weights["layer3.1.bn2.weight"]
        self.layer3.blocks[1].bn2.bias = weights["layer3.1.bn2.bias"]
        self.layer3.blocks[1].bn2.running_var = weights["layer3.1.bn2.running_var"]
        self.layer3.blocks[1].bn2.running_mean = weights["layer3.1.bn2.running_mean"]

        ### Layer 4

        self.layer4.blocks[0].conv1.weight = weights["layer4.0.conv1.weight"]
        self.layer4.blocks[0].conv2.weight = weights["layer4.0.conv2.weight"]
        self.layer4.blocks[0].bn1.weight = weights["layer4.0.bn1.weight"]
        self.layer4.blocks[0].bn1.bias = weights["layer4.0.bn1.bias"]
        self.layer4.blocks[0].bn1.running_var = weights["layer4.0.bn1.running_var"]
        self.layer4.blocks[0].bn1.running_mean = weights["layer4.0.bn1.running_mean"]
        self.layer4.blocks[0].bn2.weight = weights["layer4.0.bn2.weight"]
        self.layer4.blocks[0].bn2.bias = weights["layer4.0.bn2.bias"]
        self.layer4.blocks[0].bn2.running_var = weights["layer4.0.bn2.running_var"]
        self.layer4.blocks[0].bn2.running_mean = weights["layer4.0.bn2.running_mean"]

        self.layer4.blocks[0].downsample[0].weight = weights[
            "layer4.0.downsample.0.weight"
        ]
        self.layer4.blocks[0].downsample[1].weight = weights[
            "layer4.0.downsample.1.weight"
        ]
        self.layer4.blocks[0].downsample[1].bias = weights["layer4.0.downsample.1.bias"]
        self.layer4.blocks[0].downsample[1].running_var = weights[
            "layer4.0.downsample.1.running_var"
        ]
        self.layer4.blocks[0].downsample[1].running_mean = weights[
            "layer4.0.downsample.1.running_mean"
        ]

        self.layer4.blocks[1].conv1.weight = weights["layer4.1.conv1.weight"]
        self.layer4.blocks[1].conv2.weight = weights["layer4.1.conv2.weight"]
        self.layer4.blocks[1].bn1.weight = weights["layer4.1.bn1.weight"]
        self.layer4.blocks[1].bn1.bias = weights["layer4.1.bn1.bias"]
        self.layer4.blocks[1].bn1.running_var = weights["layer4.1.bn1.running_var"]
        self.layer4.blocks[1].bn1.running_mean = weights["layer4.1.bn1.running_mean"]
        self.layer4.blocks[1].bn2.weight = weights["layer4.1.bn2.weight"]
        self.layer4.blocks[1].bn2.bias = weights["layer4.1.bn2.bias"]
        self.layer4.blocks[1].bn2.running_var = weights["layer4.1.bn2.running_var"]
        self.layer4.blocks[1].bn2.running_mean = weights["layer4.1.bn2.running_mean"]

        return self

    def forward(self, x):
        x = self.pre_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def get_residuals(self):
        return [
            self.layer1.buffer,
            self.layer2.buffer,
            self.layer3.buffer,
            self.layer4.buffer,
        ]

    def clear_residuals(self):
        return [
            self.layer1.clear(),
            self.layer2.clear(),
            self.layer3.clear(),
            self.layer4.clear(),
        ]


class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
        pool: bool = False,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = DropBlock2d(
            p=dropout, block_size=3
        )  # TODO: Be sure to use an ODD block size! Even block sizes throw exceptions!
        self.buffer = None
        self.pool_layer = nn.MaxPool2d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        self.buffer = x
        x = self.pool_layer(x)

        return self.dropout(x)

    def clear(self) -> None:
        self.buffer = None


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        blocks = [
            UNetBlock(
                in_channels=in_channels if i == 0 else out_channels * 2 ** (i - 1),
                out_channels=out_channels * 2**i,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                pool=True,
            )
            for i in range(0, num_blocks)
        ]

        self.blocks = nn.ModuleList(blocks)

    def get_residuals(self) -> List[Tensor]:
        return [blk.buffer for blk in self.blocks]

    def clear_residuals(self) -> List[None]:
        return [blk.clear() for blk in self.blocks]

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.block = UNetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            bias=bias,
            pool=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        num_blocks: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        blocks = [
            UNetBlock(
                out_channels=in_channels // 2**i,
                in_channels=in_channels // 2 ** (i - 1),
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                pool=False,
            )
            for i in range(1, num_blocks + 1)
        ]

        self.blocks = nn.ModuleList(blocks)

        upsamplers = [
            nn.ConvTranspose2d(
                out_channels=in_channels // 2**i,
                in_channels=in_channels if i == 1 else in_channels // 2 ** (i - 1),
                kernel_size=2,
                padding=padding,
                stride=2,
                dilation=dilation,
                bias=bias,
            )
            for i in range(1, num_blocks + 1)
        ]
        self.upsamplers = nn.ModuleList(upsamplers)

        self.output_layer = nn.Conv2d(
            in_channels=in_channels // 2**num_blocks,
            out_channels=num_classes,
            kernel_size=1,
        )

    def crop(self, x: Tensor, size: Sequence[int]) -> Tensor:
        x = CenterCrop(size)(x)
        return x

    def forward(self, x: Tensor, residuals: List[Tensor]) -> Tensor:
        for i, block in enumerate(self.blocks):
            # print(f"UNetDecoder x: {x.shape}")
            x = self.upsamplers[i](x)
            # print(f"Upsampler x: {x.shape}")

            res_idx = len(residuals) - i - 1
            residual = residuals[res_idx]
            residual = self.crop(residual, size=tuple(x.shape[-2:]))

            x = torch.cat([residual, x], dim=1)
            # print(f"Concat decoder x: {x.shape}")
            x = block(x)
            # print(f"Final decoder x: {x.shape}\n" + "=" * 10)

        x = self.output_layer(x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        num_blocks: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        bias: bool = False,
        match_dims: bool = True,
        freeze_encoder: bool = False,
        backbone: Optional[str] = None,
    ):
        super().__init__()

        self.match_dims = match_dims
        self.freeze_encoder = freeze_encoder
        self.backbone = backbone

        base_out_channels = out_channels // (2 ** (num_blocks - 1))

        if backbone is None:
            self.encoder = UNetEncoder(
                in_channels=in_channels,
                out_channels=base_out_channels,
                kernel_size=kernel_size,
                num_blocks=num_blocks,
                padding=padding,
                stride=stride,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
            )

            self.bottleneck = Bottleneck(
                in_channels=base_out_channels * 2 ** (num_blocks - 1),
                out_channels=base_out_channels * 2 ** (num_blocks),
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
            )
        else:
            backbone_layer_map = {
                "resnet18": [2, 2, 2, 2],
                "resnet34": [3, 4, 6, 3],
                "resnet50": [3, 4, 6, 3],
            }
            # backbone_file_map = {
            #     "resnet18": "resnet18-f37072fd.pth",
            #     "resnet34": "resnet34-f37072fd.pth",
            #     "resnet50": "resnet50-f37072fd.pth",
            # }

            num_blocks_per_layer = backbone_layer_map[backbone]

            self.encoder = Resnet(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                num_blocks_per_layer=num_blocks_per_layer,
                padding=1,
                stride=1,
                dilation=1,
                dropout=dropout,
                bias=bias,
            )

            weights = load_resnet_weights(backbone)
            self.encoder.load_weights(weights)

            print("\n" + f"{backbone.capitalize()} weights successfully loaded!" + "\n")

            if freeze_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False

            self.bottleneck = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 2,
                kernel_size=1,
                padding=0,
                stride=1,
                dilation=dilation,
                bias=bias,
            )

        self.decoder = UNetDecoder(
            in_channels=base_out_channels * 2 ** (num_blocks),
            out_channels=in_channels,
            num_classes=num_classes,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            bias=bias,
        )

        self._initialize()

    def forward(self, x: Tensor) -> Tensor:
        code = self.encoder(x)
        # print(f"Code: {code.shape}\n" + "=" * 10)
        bottleneck = self.bottleneck(code)
        # print(f"Bottleneck: {bottleneck.shape}\n" + "=" * 10)
        residuals = self.encoder.get_residuals()
        y = self.decoder(bottleneck, residuals)
        # self.encoder.clear_residuals()

        if self.match_dims:
            y = F.interpolate(y, x.shape[-2:])

        return y

    def _get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _initialize(self):
        for name, param in self.named_parameters():
            if (
                not self.freeze_encoder and "encoder" not in name
            ) or "bias" not in name:
                nn.init.normal_(param, mean=0.0, std=0.02)


if __name__ == "__main__":
    in_channels = 3
    out_channels = 512
    num_blocks = 4
    num_classes = 10

    backbone = "resnet18"
    # backbone = None

    B, H, W = 1, 512, 512

    device = "cpu"

    x = torch.randn(B, in_channels, H, W).to(device)
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        num_classes=num_classes,
        backbone=backbone,
    ).to(device)

    y = model(x)

    print(f"In: {x.shape}")
    print(f"Out: {y.shape}")

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)

    s = summary(
        model,
        input_size=(1, in_channels, 224, 224),  # (batch_size, channels, height, width)
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

    print(sum(p.numel() for p in model.parameters()))
