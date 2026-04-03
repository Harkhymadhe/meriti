# -*- coding: utf-8 -*-

from typing import Optional

from remote_segmentation.eval import extract_losses

__all__ = ["load_unet_model_smp", "load_unet_model", "load_unet_model"]


def load_unet_model_smp(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    freeze_encoder: bool = True,
    in_channels: int = 1,
    num_classes: int = 3,
):
    from segmentation_models_pytorch import Unet

    model = Unet(
        encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,  # model output channels (number of classes in your dataset)
        activation="identity",
    )

    if freeze_encoder:
        requires_grad = False
    else:
        requires_grad = True

    for param in model.encoder.parameters():
        param.requires_grad_(requires_grad)

    return model


def load_unet_model_backbone(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    freeze_encoder: bool = True,
    in_channels: int = 1,
    num_classes: int = 3,
):
    from backbones_unet.model.unet import Unet

    model = Unet(
        # encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
        encoder_freeze=freeze_encoder,
        backbone=encoder_name,  # backbone network name
        in_channels=in_channels,  # input channels (1 for gray-scale images, 3 for RGB, etc.)
        num_classes=num_classes,  # output channels (number of classes in your dataset)
        activation="identity",
    )

    return model


def load_unet_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    freeze_encoder: bool = True,
    in_channels: int = 1,
    num_classes: int = 3,
    checkpoint: Optional[int] = None,
    smp: bool = False,
):
    if checkpoint is not None:
        encoder = checkpoint is not None
        model = load_unet_state_dict(
            experiment_num=checkpoint,
            encoder=encoder,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            freeze_encoder=freeze_encoder,
            in_channels=in_channels,
            num_classes=num_classes,
            smp=smp,
        )

        return model

    load_func = load_unet_model_smp if smp else load_unet_model_backbone

    model = load_func(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        freeze_encoder=freeze_encoder,
        in_channels=in_channels,
        num_classes=num_classes,
    )

    return model


def load_unet_state_dict(
    experiment_num,
    encoder=False,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    freeze_encoder: bool = True,
    in_channels: int = 1,
    num_classes: int = 3,
    smp: bool = False,
):
    model = load_unet_model(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        freeze_encoder=freeze_encoder,
        in_channels=in_channels,
        num_classes=num_classes,
        smp=smp,
    )

    if encoder:
        model_checkpoint = extract_losses(experiment_num)["model"]
        model.load_state_dict(model_checkpoint)
    else:
        model_ = load_unet_model(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            freeze_encoder=freeze_encoder,
            in_channels=in_channels,
            num_classes=num_classes,
            smp=smp,
        )
        model.encoder = model_.encoder

    if freeze_encoder:
        requires_grad = False
    else:
        requires_grad = True

    for param in model.encoder.parameters():
        param.requires_grad_(requires_grad)

    return model.train()


if __name__ == "__main__":
    model = load_unet_state_dict(experiment_num=2)
    print(model)
