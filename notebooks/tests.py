# -*- coding: utf-8 -*-

from remote_segmentation.models import load_unet_model
from remote_segmentation.dataset import FloodSegmentationDataset

if __name__ == "__main__":
    import torch

    encoder_name = "resnet34"
    in_channels = 3
    smp = False
    num_classes = 10

    model = load_unet_model(
        encoder_name=encoder_name,
        num_classes=num_classes,
        in_channels=in_channels,
        smp=smp,
    )
    x = torch.randn(1, in_channels, 224, 224)

    y = model(x)

    print(y.shape)

    splits = ["train", "val", "test"]

    for split in splits:
        dataset = FloodSegmentationDataset(split=split)
        print(len(dataset))

    A, B = dataset[0]

    print(A.shape)
    print(B.shape)
