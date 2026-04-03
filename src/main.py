# -*- coding: utf-8 -*-

import os
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import hydra
import wandb

import torch

from loguru import logger
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf

from torch import optim

from remote_segmentation.metrics import CompositeLoss, Accuracy
from remote_segmentation.architectures import UNet
from remote_segmentation.dataset import FloodSegmentationDataset

__all__ = ["main"]


@hydra.main(config_name="config", config_path="conf", version_base="1.1")
def main(cfg: DictConfig):
    #### Reduce precision for faster computation
    torch.set_float32_matmul_precision("high")

    ##### Ensure reproducibility
    torch.backends.cudnn.deterministic = cfg.reproducibility.cudnn_deterministic
    torch.backends.cudnn.benchmark = cfg.reproducibility.cudnn_benchmark

    SEED = cfg.reproducibility.seed
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)

    #### Add logger from Loguru
    logger.add(sink=cfg.logs.path, enqueue=True)

    #### Generate training and validation dataloaders

    train_ds = FloodSegmentationDataset(
        data_dir=cfg.data.path,
        split="train",
        size=(cfg.data.image.size.height, cfg.data.image.size.width),
        augment=cfg.data.augment,
        num_label_pixels=int(
            cfg.data.pixel_fraction
            * cfg.data.image.size.height
            * cfg.data.image.size.width
        ),
    )
    train_dataloader = train_ds.to_dataloader(
        batch_size=cfg.training.data.batch_size,
        num_workers=cfg.training.data.num_workers,
        prefetch_factor=cfg.training.data.prefetch_factor,
        pin_memory=cfg.training.data.pin_memory,
        persistent_workers=True,
    )
    train_dl = iter(train_dataloader)

    val_ds = FloodSegmentationDataset(
        data_dir=cfg.data.path,
        split="val",
        size=(cfg.data.image.size.height, cfg.data.image.size.width),
        augment=False,
        num_label_pixels=int(
            cfg.data.pixel_fraction
            * cfg.data.image.size.height
            * cfg.data.image.size.width
        ),
    )
    val_dataloader = val_ds.to_dataloader(
        batch_size=cfg.training.data.batch_size,
        num_workers=cfg.training.data.num_workers,
        prefetch_factor=cfg.training.data.prefetch_factor,
        pin_memory=False,
    )
    val_dl = iter(val_dataloader)

    num_train_batches = len(train_ds) // cfg.training.data.batch_size

    if len(train_ds) % cfg.training.data.batch_size != 0:
        num_train_batches += 1

    num_val_batches = len(val_ds) // cfg.training.data.batch_size

    if len(val_ds) % cfg.training.data.batch_size != 0:
        num_val_batches += 1

    ##### Device instantiation

    device = torch.device(
        "cpu" if not torch.cuda.is_available() else cfg.training.device
    )

    with wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        tags=["sweep-experiments"],
        # tags=["dry-run"],
    ) as run:
        ## Define train-step-level metrics to log

        wandb.define_metric("train/loss", step_metric="global_train_step")
        wandb.define_metric("train/accuracy", step_metric="global_train_step")
        wandb.define_metric("train/class_accuracy", step_metric="global_train_step")
        wandb.define_metric("train/class_proportion/y", step_metric="global_train_step")
        wandb.define_metric(
            "train/class_proportion/y_hat", step_metric="global_train_step"
        )
        wandb.define_metric("train/max_grad_norm", step_metric="global_train_step")

        ### Define validation-step-level metrics to log

        wandb.define_metric("val/loss", step_metric="global_val_step")
        wandb.define_metric("val/accuracy", step_metric="global_val_step")
        wandb.define_metric("val/class_accuracy", step_metric="global_val_step")
        wandb.define_metric("val/class_proportion/y", step_metric="global_val_step")
        wandb.define_metric("val/class_proportion/y_hat", step_metric="global_val_step")

        ### Define epoch-level metrics to log

        wandb.define_metric("epoch/train/loss", step_metric="epoch")
        wandb.define_metric("epoch/train/accuracy", step_metric="epoch")
        wandb.define_metric("epoch/train/class_accuracy", step_metric="epoch")
        wandb.define_metric("epoch/train/class_proportion/y", step_metric="epoch")
        wandb.define_metric("epoch/train/class_proportion/y_hat", step_metric="epoch")

        wandb.define_metric("epoch/val/loss", step_metric="epoch")
        wandb.define_metric("epoch/val/accuracy", step_metric="epoch")
        wandb.define_metric("epoch/val/class_accuracy", step_metric="epoch")
        wandb.define_metric("epoch/val/class_proportion/y", step_metric="epoch")
        wandb.define_metric("epoch/val/class_proportion/y_hat", step_metric="epoch")

        ### Assign sweep hyperparameters to Hydra config

        cfg.data.pixel_fraction = wandb.config["data.pixel_fraction"]
        cfg.training.loss.pce.gamma = wandb.config["training.loss.pce.gamma"]

        ### Update wandb config object with Hydra config
        run.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            allow_val_change=True,
        )

        ##### Instantiate model
        model = UNet(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            num_classes=cfg.data.num_classes,
            num_blocks=cfg.model.num_blocks,
            kernel_size=cfg.model.kernel_size,
            padding=cfg.model.padding,
            stride=cfg.model.stride,
            dilation=cfg.model.dilation,
            dropout=cfg.model.dropout,
            bias=cfg.model.bias,
            match_dims=cfg.model.match_dims,
            freeze_encoder=cfg.model.freeze_encoder,
            backbone=cfg.model.backbone,
        ).to(device)

        ##### Set model to training mode
        model = model.train()

        logger.info(f"Parameter count: {model._get_parameter_count(): ,}")

        ##### Log parameters and their gradients
        run.watch(
            model,
            log="all",  # log both gradients and parameters
            log_freq=len(train_dataloader) // 2,  # log twice every epoch
        )

        ##### Parameter filtering and grouping
        param_groups = []

        if not cfg.model.freeze_encoder:
            encoder_params = {
                "params": filter(lambda x: x.requires_grad, model.encoder.parameters()),
                "lr": cfg.training.lr.encoder_lr,
            }
            param_groups.append(encoder_params)
        else:
            for param in model.encoder.parameters():
                param.requires_grad_(False)

        bottleneck_params = {
            "params": filter(lambda x: x.requires_grad, model.bottleneck.parameters()),
            "lr": cfg.training.lr.bottleneck_lr,
        }

        param_groups.append(bottleneck_params)

        decoder_params = {
            "params": filter(lambda x: x.requires_grad, model.decoder.parameters()),
            "lr": cfg.training.lr.decoder_lr,
        }

        param_groups.append(decoder_params)

        ##### Instantiate optimizer
        optimizer = optim.AdamW(
            params=param_groups,
            lr=cfg.training.lr.base_lr,
            betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2),
            weight_decay=cfg.training.optimizer.weight_decay,
            fused=device.__str__() == "cuda",
        )

        ##### Instantiate objective function and accuracy module
        criterion = CompositeLoss(
            num_classes=cfg.data.num_classes,
            pce_reduction=cfg.training.loss.pce.reduce,
            dice_reduction=cfg.training.loss.dice.reduction,
            alpha=cfg.training.loss.pce.alpha,
            gamma=cfg.training.loss.pce.gamma,
            pce_weight=cfg.training.loss.pce.weight,
            dice_weight=cfg.training.loss.dice.weight,
            device=device,
        )

        train_metric = Accuracy(num_classes=cfg.data.num_classes, device=device)
        val_metric = Accuracy(num_classes=cfg.data.num_classes, device=device)

        LEAVE = False

        train_losses_ = torch.zeros(size=(cfg.training.num_epochs,))
        val_losses_ = torch.zeros(size=(cfg.training.num_epochs,))

        train_accs_ = torch.zeros(size=(cfg.training.num_epochs,))
        val_accs_ = torch.zeros(size=(cfg.training.num_epochs,))

        logger.info("Training the network...")

        start_time = time.time()

        train_class_proportion, train_class_pred_proportion = None, None
        val_class_proportion, val_class_pred_proportion = None, None

        epoch_train_losses = torch.zeros(size=(num_train_batches,), device=device)
        epoch_val_losses = torch.zeros(size=(num_val_batches,), device=device)

        epoch_train_accs = torch.zeros(size=(num_train_batches,), device=device)
        epoch_val_accs = torch.zeros(size=(num_val_batches,), device=device)

        epoch_max_norms = torch.zeros(size=(num_train_batches,), device=device)

        train_global_step = 0
        val_global_step = 0

        ### Checkpoint path

        checkpoint_dir = f"{cfg.runs.path}/{run.id}"
        # checkpoint_dir = f"{cfg.runs.path}/dry_run"

        # Training loop
        for epoch in range(1, cfg.training.num_epochs + 1):
            #### Set model to training mode

            model.train()

            #### Loop over training data

            for train_index in tqdm(
                range(num_train_batches),
                desc=f"Epoch {epoch} | Training Phase",
                position=1,
                leave=LEAVE,
            ):
                train_global_step += 1

                X, y = next(train_dl)

                X, y = (
                    X.to(device, non_blocking=cfg.training.data.pin_memory),
                    y.to(device, non_blocking=cfg.training.data.pin_memory),
                )

                y_hat = model(X)

                loss = criterion(y_hat, y)

                acc = train_metric(y_hat, y)

                loss.backward()

                ### Clip the gradient
                max_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.training.max_gradient_norm,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                epoch_train_losses[train_index] = loss.detach()
                epoch_train_accs[train_index] = acc.detach()
                epoch_max_norms[train_index] = max_norm.detach()

                ### Accumulate training metric data to log
                train_accuracy_per_class = train_metric.accuracy_per_class.cpu()

                plt.figure()
                plt.bar(
                    range(len(train_accuracy_per_class)),
                    train_accuracy_per_class,
                )
                plt.title("Pixel Classification Accuracy per Class (Training)")
                plt.ylim(0, 100)
                plt.xticks(range(len(train_accuracy_per_class)))
                plt.xlabel("Pixel Class")
                plt.ylabel("Pixel Class Accuracy")
                train_acc_bar_plot = plt.gcf()

                train_class_proportion_ = train_metric.to_frequency(true_labels=True).cpu()
                train_class_pred_proportion_ = train_metric.to_frequency(
                    true_labels=False
                ).cpu()

                # ------------------------------------------

                plt.figure()
                plt.bar(range(len(train_class_proportion_)), train_class_proportion_)
                plt.ylim(0, 100)
                plt.xticks(range(len(train_class_proportion_)))
                plt.title("True Pixel Class Proportion (Training)")
                plt.xlabel("Pixel Class")
                plt.ylabel("Pixel Class Proportion")
                train_class_proportion_bar_plot = plt.gcf()

                # ------------------------------------------

                plt.figure()
                plt.bar(
                    range(len(train_class_pred_proportion_)),
                    train_class_pred_proportion_,
                )
                plt.ylim(0, 100)
                plt.xticks(range(len(train_class_proportion_)))
                plt.title("Predicted Pixel Class Proportion (Training)")
                plt.xlabel("Pixel Class")
                plt.ylabel("Pixel Class Proportion")
                train_class_pred_proportion_bar_plot = plt.gcf()

                train_log_data = {
                    "train/loss": epoch_train_losses[train_index],
                    "train/accuracy": epoch_train_accs[train_index],
                    "train/class_accuracy": train_acc_bar_plot,
                    "train/class_proportion/y": train_class_proportion_bar_plot,
                    "train/class_proportion/y_hat": train_class_pred_proportion_bar_plot,
                    "train/max_grad_norm": max_norm.detach().mean(),
                    "global_train_step": train_global_step,
                }

                ### Log training metrics to wandb
                wandb.log(train_log_data)

            #### Set model to evaluation mode
            model.eval()

            with torch.no_grad():
                ### Loop over validation data

                for val_index in tqdm(
                    range(num_val_batches),
                    desc=f"Epoch {epoch} | Validation Phase",
                    position=1,
                    leave=LEAVE,
                ):
                    val_global_step += 1

                    X, y = next(val_dl)
                    X, y = X.to(device), y.to(device)

                    y_hat = model(X)

                    loss = criterion(y_hat, y)
                    acc = val_metric(y_hat, y)

                    epoch_val_losses[val_index] = loss.detach()
                    epoch_val_accs[val_index] = acc.detach()

                    ### Accumulate validation metric data to log
                    val_accuracy_per_class = val_metric.accuracy_per_class.cpu()

                    plt.figure()
                    plt.bar(
                        range(len(val_accuracy_per_class)),
                        val_accuracy_per_class,
                    )
                    plt.ylim(0, 100)
                    plt.xticks(range(len(val_accuracy_per_class)))
                    plt.title("Pixel Classification Accuracy per Class (Validation)")
                    plt.xlabel("Pixel Class")
                    plt.ylabel("Pixel Class Accuracy")
                    val_acc_bar_plot = plt.gcf()

                    val_class_proportion_ = val_metric.to_frequency(true_labels=True).cpu()
                    val_class_pred_proportion_ = val_metric.to_frequency(
                        true_labels=False
                    ).cpu()

                    plt.figure()
                    plt.bar(range(len(val_class_proportion_)), val_class_proportion_)
                    plt.ylim(0, 100)
                    plt.xticks(range(len(val_class_proportion_)))
                    plt.title("True Pixel Class Proportion (Training)")
                    plt.xlabel("Pixel Class")
                    plt.ylabel("Pixel Class Proportion")
                    val_class_proportion_bar_plot = plt.gcf()

                    # ------------------------------------------

                    plt.figure()
                    plt.bar(
                        range(len(val_class_pred_proportion_)),
                        val_class_pred_proportion_,
                    )
                    plt.ylim(0, 100)
                    plt.xticks(range(len(val_class_pred_proportion_)))
                    plt.title("Predicted Pixel Class Proportion (Validation)")
                    plt.xlabel("Pixel Class")
                    plt.ylabel("Pixel Class Proportion")
                    val_class_pred_proportion_bar_plot = plt.gcf()

                    val_log_data = {
                        "val/loss": epoch_val_losses[val_index],
                        "val/accuracy": epoch_val_accs[val_index],
                        "val/class_accuracy": val_acc_bar_plot,
                        "val/class_proportion/y": val_class_proportion_bar_plot,
                        "val/class_proportion/y_hat": val_class_pred_proportion_bar_plot,
                        "global_val_step": val_global_step,
                    }

                    ### Log validation metrics to wandb
                    wandb.log(val_log_data)

            # Re-instantiate dataloaders
            train_dl = iter(train_dataloader)
            val_dl = iter(val_dataloader)

            epoch_train_loss = epoch_train_losses.mean().detach()
            epoch_val_loss = epoch_val_losses.mean().detach()

            epoch_train_acc = epoch_train_accs.mean().detach()
            epoch_val_acc = epoch_val_accs.mean().detach()

            epoch_max_norm = epoch_max_norms.mean().detach().item()

            train_losses_[epoch - 1] = epoch_train_loss
            val_losses_[epoch - 1] = epoch_val_loss

            train_accs_[epoch - 1] = epoch_train_acc
            val_accs_[epoch - 1] = epoch_val_acc

            epoch_train_losses.zero_()
            epoch_val_losses.zero_()

            epoch_train_accs.zero_()
            epoch_val_accs.zero_()

            epoch_max_norms.zero_()

            #### Get class proportions for both true and predicted classes in train and val splits
            if epoch == 1:
                train_class_proportion = train_metric.to_frequency(true_labels=True).cpu()
                val_class_proportion = val_metric.to_frequency(true_labels=True).cpu()

            train_class_pred_proportion = train_metric.to_frequency(true_labels=False).cpu()
            val_class_pred_proportion = val_metric.to_frequency(true_labels=False).cpu()

            # #### -----------------------------------

            plt.figure()
            plt.bar(range(len(train_class_proportion)), train_class_proportion)
            plt.ylim(0, 100)
            plt.xticks(range(len(train_class_proportion)))
            plt.title("Epoch-level Pixel True Class Proportion (Training)")
            plt.xlabel("Pixel Class")
            plt.ylabel("Pixel Class Proportion")
            epoch_train_class_proportion_bar_plot = plt.gcf()

            #### -----------------------------------

            plt.figure()
            plt.bar(
                range(len(train_class_pred_proportion)), train_class_pred_proportion
            )
            plt.ylim(0, 100)
            plt.xticks(range(len(train_class_pred_proportion)))
            plt.title("Epoch-level Pixel Predicted Class Proportion (Training)")
            plt.xlabel("Pixel Class")
            plt.ylabel("Pixel Class Proportion")
            epoch_train_class_pred_proportion_bar_plot = plt.gcf()

            #### -----------------------------------

            plt.figure()
            plt.bar(range(len(val_class_proportion)), val_class_proportion)
            plt.ylim(0, 100)
            plt.xticks(range(len(val_class_proportion)))
            plt.title("Epoch-level Pixel True Class Proportion (Validation)")
            plt.xlabel("Pixel Class")
            plt.ylabel("Pixel Class Proportion")
            epoch_val_class_proportion_bar_plot = plt.gcf()

            #### -----------------------------------

            plt.figure()
            plt.bar(range(len(val_class_pred_proportion)), val_class_pred_proportion)
            plt.ylim(0, 100)
            plt.xticks(range(len(val_class_pred_proportion)))
            plt.title("Epoch-level Pixel Predicted Class Proportion (Validation)")
            plt.xlabel("Pixel Class")
            plt.ylabel("Pixel Class Proportion")
            epoch_val_class_pred_proportion_bar_plot = plt.gcf()

            # Log to Wandb
            epoch_logs = {
                "epoch/train/loss": epoch_train_loss,
                "epoch/val/loss": epoch_val_loss,
                "epoch/train/accuracy": epoch_train_acc,
                "epoch/val/accuracy": epoch_val_acc,
                "epoch/train/class_accuracy": epoch_train_acc,
                "epoch/val/class_accuracy": epoch_val_acc,
                "epoch/train/class_proportion/y": epoch_train_class_proportion_bar_plot,
                "epoch/train/class_proportion/y_hat": epoch_train_class_pred_proportion_bar_plot,
                "epoch/val/class_proportion/y": epoch_val_class_proportion_bar_plot,
                "epoch/val/class_proportion/y_hat": epoch_val_class_pred_proportion_bar_plot,
                "epoch": epoch - 1,
            }
            wandb.log(epoch_logs)

            # Log to loguru
            logger.info(
                f"Epoch {epoch} | Avg. max norm: {epoch_max_norm:.4f} | Train loss: {epoch_train_loss:.4f} | Validation loss: {epoch_val_loss:.4f} | Train acc: {epoch_train_acc:.4f} | Validation acc: {epoch_val_acc:.4f}"
            )

            ### Clear metric buffers
            train_metric.zero_()
            val_metric.zero_()

            ### Persist artefacts

            model_path = f"{checkpoint_dir}/epoch_{epoch}_train_loss_{epoch_train_loss:.4f}_val_loss_{epoch_val_loss:.4f}.pt"

            os.makedirs(checkpoint_dir, exist_ok=True)

            with torch.no_grad():
                checkpoint = {
                    "model": model.eval().state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train/loss": epoch_train_loss,
                    "val/loss": epoch_val_loss,
                    "train/accuracy": epoch_train_acc,
                    "val/accuracy": epoch_val_acc,
                    "epoch": epoch - 1,
                }
                torch.save(checkpoint, model_path)

    if device.__str__() == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    logger.info(
        "Total time taken to train the model for {} epochs: {:.2f}s".format(
            cfg.training.num_epochs, end_time - start_time
        )
    )

    with torch.no_grad():
        checkpoint = {
            "model": model.eval().state_dict(),
            "optimizer": optimizer.state_dict(),
            "train/losses": train_losses_,
            "val/losses": val_losses_,
            "train/accs": train_accs_,
            "val/accs": val_accs_,
            "num_epochs": cfg.training.num_epochs,
        }

        torch.save(checkpoint, f"{checkpoint_dir}/final_checkpoint.pt")
    
    plt.close("all")


if __name__ == "__main__":
    main()
