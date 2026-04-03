# -*- coding: utf-8 -*-

import os
import torch

from matplotlib import pyplot as plt

__all__ = ["extract_losses", "extract_losses_", "display_losses"]


def extract_losses(experiment_num):
    experiment_dir = f"../../experiments/experiment_{experiment_num}"

    checkpoint = torch.load(f"{experiment_dir}/checkpoint.pth")
    checkpoint["experiment_dir"] = experiment_dir

    return checkpoint


def extract_losses_(experiment_num):
    # experiment_dir = f"../experiments/experiment_{experiment_num}"
    experiment_dir = "."

    train_losses = torch.load(f"{experiment_dir}/train_losses.pth")
    val_losses = torch.load(f"{experiment_dir}/val_losses.pth")

    return dict(train_losses=train_losses, val_losses=val_losses)


def display_losses(losses):
    train_losses = losses["train_losses"].numpy()
    val_losses = losses["val_losses"].numpy()
    experiment_dir = losses["experiment_dir"]
    epochs = losses["epochs"]

    os.mkdir(f"{experiment_dir}/plots", exist_ok=True)

    plt.plot(range(1, 1 + epochs), train_losses, label="train_losses")
    plt.plot(range(1, 1 + epochs), val_losses, label="val_losses")

    plt.legend()
    plt.savefig(f"{experiment_dir}/plots/losses.png")
    plt.show()


if __name__ == "__main__":
    print(extract_losses_(0))

    experiment_num = 6

    checkpoint = extract_losses(experiment_num=experiment_num)
    params = {k: v for k, v in checkpoint.items() if "losses" in k}
    print(params)

    display_losses(checkpoint)
