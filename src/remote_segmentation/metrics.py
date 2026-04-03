# -*- coding: utf-8 -*-

import torch
from torch import nn

from typing import Union, Tuple, Optional

__all__ = [
    "FocalLoss",
    "PartialCrossEntropyLoss",
    "DiceLoss",
    "CompositeLoss",
    "Accuracy",
]


class FocalLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        num_classes: int = 2,
        mask_value: int = 255,
        alpha: float = 0.25,
        gamma: float = 2.0,
        device: Union[str, torch.DeviceObjType] = "cpu",
    ):
        super().__init__()

        self.device = device

        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.mask_value = mask_value

        if weights is None:
            if num_classes > 2:
                classes = [1.0 for _ in range(num_classes)]
            else:
                classes = [1.0]

            weights = torch.tensor(classes)

        weights = weights.to(self.device)

        assert weights.numel() == self.num_classes, (
            "Number of weights should match number of classes!"
        )

        self.weights = weights
        self.nll_loss = nn.NLLLoss(weight=weights, reduction="none").to(device)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape input as [num_pixels, num_classes/num_channels]
        # i.e., [B, C, H, W] -> [B, H, W, C] -> [BHW, C]
        input = (
            input.permute(0, *range(2, input.ndim), 1)
            .contiguous()
            .view(-1, self.num_classes)
            .to(self.device)
        )

        # Reshape target as [num_pixels,]
        # i.e., [B, 1, H, W] -> [BHW, ]
        target = target.contiguous().view(-1).to(self.device).to(torch.int64)

        # If mask is missing ...
        if mask is None:
            # ... generate a default mask
            mask = target != self.mask_value

        # Move mask to appropriate device
        mask = mask.to(self.device)

        # Filter input and target pixels with mask
        input = input[mask, :]
        target = target[mask]

        # print("In: ", input.shape)
        # print("Target: ", target.shape)

        # Generate probabilities from input logits
        log_probs = torch.log_softmax(input, dim=-1)
        probs = torch.exp(log_probs)

        # Calculate cross-entropy loss and focal term (i.e., class down-weighting constant)
        nll_loss = self.nll_loss(log_probs, target)
        focal_term = (1 - probs[torch.arange(len(input)), target]) ** self.gamma

        return (self.alpha * focal_term * nll_loss), mask


class PartialCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mask_value: int = 255,
        weights: Optional[torch.Tensor] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduce: bool = False,
        device: Union[str, torch.DeviceObjType] = "cpu",
    ):
        super().__init__()

        self.reduce = reduce
        self.device = device
        self.num_classes = num_classes

        self.focal_loss = FocalLoss(
            weights=weights,
            num_classes=num_classes,
            alpha=alpha,
            gamma=gamma,
            mask_value=mask_value,
            device=device,
        )

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Calculate focal loss
        # focal_loss, mask_ = self.focal_loss(input, target)
        focal_loss, mask = self.focal_loss(input, target, mask)

        # print("Generated mask")
        # print(mask.shape)
        # print(mask)

        # Ensure mask is on same device as focal loss calculated
        mask = mask.to(self.device)

        # Apply mask to calculated per-pixel focal loss
        # focal_loss = (mask * focal_loss.contiguous().view(mask.shape))

        if self.reduce:
            # Reduce focal loss tensor to give scalar loss
            focal_loss = focal_loss.sum() / mask.sum()

        return focal_loss, mask


class DiceLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        num_classes: int = 2,
        mask_value: int = 255,
        reduction: str = "mean",
        epsilon: float = 1e-8,
        device: Union[str, torch.DeviceObjType] = "cpu",
    ):
        super().__init__()

        assert reduction in ["mean", "sum"], (
            "`reduction` must be either 'mean' or 'sum'."
        )

        self.device = device
        self.num_classes = num_classes
        self.mask_value = mask_value
        self.reduction = reduction
        self.epsilon = epsilon

        ##### Option  1
        # if weights is None:
        #     if num_classes >= 2:
        #         classes = [1.0 for _ in range(num_classes)]
        #     else:
        #         classes = [1.0]

        #     weights = torch.tensor(classes)

        # weights = weights.to(self.device)

        # assert weights.numel() == self.num_classes, (
        #     "Number of weights should match number of classes!"
        # )
        # self.weights = weights

        #### Option 2
        if weights is None:
            if num_classes >= 2:
                weights = torch.ones(
                    size=(num_classes,), dtype=torch.float32, device=self.device
                )
            else:
                weights = torch.ones(size=(1,), dtype=torch.float32, device=self.device)

        assert weights.numel() == self.num_classes, (
            "Number of weights should match number of classes!"
        )

        self.weights = weights.to(self.device)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Reshape input as [num_pixels, num_classes/num_channels]
        # i.e., [B, C, H, W] -> [B, H, W, C] -> [BHW, C]
        input = (
            input.permute(0, *range(2, input.ndim), 1)
            .contiguous()
            .view(-1, self.num_classes)
            .to(self.device)
        )

        # Reshape target as [num_pixels,]
        # i.e., [B, 1, H, W] -> [BHW, ]
        target = target.contiguous().view(-1).to(self.device).to(torch.int64)

        # If mask is missing ...
        if mask is None:
            # ... generate a default mask
            mask = target != self.mask_value

        # Move mask to appropriate device
        mask = mask.to(self.device)

        # Filter input and target pixels with mask
        input = input[mask, :]
        target = target[mask]

        # print("In: ", input.shape)
        # print("Target: ", target.shape)

        # Generate probabilities from input logits
        log_probs = torch.log_softmax(input, dim=-1)

        y_pred = log_probs.argmax(dim=-1)
        dice_losses = torch.zeros_like(self.weights, device=self.device)

        for cls in range(self.num_classes):
            t = y_pred == cls
            p = target == cls

            intersection = (t * p).sum() + self.epsilon
            union = t.sum() + p.sum() + self.epsilon
            dice = 2 * intersection / union
            dice_losses[cls] = 1 - dice

        dice_losses *= self.weights

        if self.reduction == "mean":
            loss = dice_losses.mean()
        else:
            loss = dice_losses.sum()

        return loss, mask


class CompositeLoss(nn.Module):
    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        num_classes: int = 2,
        mask_value: int = 255,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pce_reduction: bool = False,
        dice_reduction: str = "mean",
        pce_weight: float = 1.0,
        dice_weight: float = 1.0,
        # reduction: str = "mean",
        device: Union[str, torch.DeviceObjType] = "cpu",
    ):
        super().__init__()

        # assert reduction in ["mean", "sum"], f"'reduction' must be one of: 'mean', 'sum'."

        self.device = device

        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.mask_value = mask_value
        self.pce_reduction = pce_reduction
        self.dice_reduction = dice_reduction
        # self.reduction = reduction

        self.dice_weight = dice_weight
        self.pce_weight = pce_weight

        self.pce_loss = PartialCrossEntropyLoss(
            num_classes=num_classes,
            mask_value=mask_value,
            weights=weights,
            alpha=alpha,
            gamma=gamma,
            reduce=pce_reduction,
            device=device,
        )

        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            mask_value=mask_value,
            weights=weights,
            reduction=dice_reduction,
            device=device,
        )

    def forward(self, input, target, mask=None) -> torch.Tensor:
        dice_loss, mask = self.dice_loss(input, target, mask=mask)
        pce_loss, _ = self.pce_loss(input, target, mask=mask)

        dice_loss = dice_loss * self.dice_weight
        pce_loss = pce_loss * self.pce_weight

        # print(f"D-Loss: {dice_loss.item():.4f}")
        # print(f"P-Loss: {pce_loss.item():.4f}")

        return dice_loss + pce_loss


class Accuracy(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mask_value: int = 255,
        device: Union[str, torch.DeviceObjType] = "cpu",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.mask_value = mask_value
        self.device = device

        self.accuracy_per_class = torch.zeros(size=(self.num_classes,), device=device)
        self.pred_class_count = torch.zeros(size=(self.num_classes,), device=device)
        self.class_count = torch.zeros(size=(self.num_classes,), device=device)

    def to_frequency(self, true_labels: bool = True) -> torch.Tensor:
        count = self.class_count if true_labels else self.pred_class_count
        return 100 * count / count.sum()

    def zero_(self):
        self.accuracy_per_class.zero_()
        self.pred_class_count.zero_()
        self.class_count.zero_()

    @torch.inference_mode()
    def forward(self, input, target, mask=None):
        # Reshape input as [num_pixels, num_classes/num_channels]
        # i.e., [B, C, H, W] -> [B, H, W, C] -> [BHW, C]
        input = (
            input.permute(0, *range(2, input.ndim), 1)
            .contiguous()
            .view(-1, self.num_classes)
            .to(self.device)
        )

        # Reshape target as [num_pixels,]
        # i.e., [B, 1, H, W] -> [BHW, ]
        target = target.contiguous().view(-1).to(self.device).to(torch.int64)

        # If mask is missing ...
        if mask is None:
            # ... generate a default mask
            mask = target != self.mask_value

        # Move mask to appropriate device
        mask = mask.to(self.device)

        # Filter input and target pixels with mask
        input = input[mask, :]
        target = target[mask]

        # Generate probabilities from input logits
        log_probs = torch.log_softmax(input, dim=-1)
        target_hat = log_probs.max(dim=-1).indices

        outputs, counts = torch.unique(
            target, return_inverse=False, return_counts=True, sorted=True
        )
        self.class_count[outputs] += counts

        outputs, counts = torch.unique(
            target_hat, return_inverse=False, return_counts=True, sorted=True
        )
        self.pred_class_count[outputs] += counts

        for cls in range(self.num_classes):
            cls_mask = target == cls
            if cls_mask.sum() > 0:
                cls_acc = (target_hat[cls_mask] == cls).float().mean()
                self.accuracy_per_class[cls] = 100 * cls_acc
        #         print(f"Class {cls} Accuracy: {100*cls_acc.item(): .4f}")
        # NOTE: Classes 6 and 9 is hogging most of the predictions, hence mode collapse!

        return (target_hat == target).float().mean()
