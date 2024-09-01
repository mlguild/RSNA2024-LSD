import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import PIL
import PIL.Image
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.utils import set_seed
from gate.data.image.classification.imagenet1k import StandardAugmentations
from rich import print
from rich.traceback import install
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm

import wandb

from lsd.src.lsd.data import (
    BalancedBatchSampler,
    ISIC2024Dataset,
    Importance,
    SplitNames,
)

install()


class AveragingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([model(x).squeeze() for model in self.models]).mean(
            dim=0
        )


def compute_pauc_above_80_tpr(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Find the index where TPR crosses 80%
    idx_80_tpr = np.argmax(tpr >= 0.8)

    # Slice the TPR and FPR from this point onwards
    tpr_slice = tpr[idx_80_tpr:]
    fpr_slice = fpr[idx_80_tpr:]

    # Compute pAUC
    pauc = np.trapz(tpr_slice, fpr_slice)

    # Normalize pAUC to [0, 0.2] range
    pauc_normalized = pauc / 0.2

    return pauc_normalized


import torch.nn.functional as F


def compute_metrics(
    outputs: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    with torch.no_grad():
        loss = F.binary_cross_entropy_with_logits(outputs, labels).item()
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        labels = labels.cpu().numpy()
        predictions = (probabilities > 0.5).astype(int)

        accuracy = (predictions == labels).mean()

        metrics = {
            "accuracy": accuracy,
            "loss": loss,
        }

        if np.unique(labels).size > 1:
            # Compute overall metrics
            auc_roc = roc_auc_score(labels, probabilities)
            pauc = compute_pauc_above_80_tpr(labels, probabilities)
            ap = average_precision_score(labels, probabilities)

            metrics.update(
                {
                    "auc_roc": auc_roc,
                    "pauc_above_80_tpr": pauc,
                    "average_precision": ap,
                }
            )

            # Compute per-class metrics
            for class_id in range(2):  # Assuming binary classification
                class_mask = labels == class_id
                class_labels = labels[class_mask]
                class_predictions = predictions[class_mask]

                class_accuracy = (class_predictions == class_labels).mean()

                metrics.update(
                    {
                        f"class_{class_id}_accuracy": class_accuracy,
                    }
                )

        return metrics


def log_metrics_and_images(
    metrics: Dict[str, float],
    images: List[Dict],
    mode: str,
    iter_id: int,
    accelerator: Accelerator,
):
    if accelerator.is_local_main_process:
        wandb.log({f"{mode}_{k}": v for k, v in metrics.items()})

        # Log per-class metrics in a table
        if "class_0_accuracy" in metrics:
            table = wandb.Table(columns=["Class", "Accuracy", "AUC-ROC"])
            for class_id in range(2):  # Assuming binary classification
                table.add_data(
                    class_id,
                    metrics[f"class_{class_id}_accuracy"],
                    metrics.get(f"class_{class_id}_auc_roc", "N/A"),
                )
            wandb.log({f"{mode}_per_class_metrics": table})

        # Log images
        if images:
            wandb_images = [
                wandb.Image(
                    r["image"],
                    caption=f"Predicted: {torch.sigmoid(r['output']).item():.4f}, Target: {r['label'].item()}",
                )
                for r in images[:100]
            ]
            wandb.log(
                {f"{mode}_examples": wandb_images, f"{mode}_iter": iter_id}
            )


def create_dataloaders(
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    root_dir: str,
    image_size: int,
    seed: int,
    model_transform_config: Callable,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    common_transforms = T.Compose(
        [
            T.Resize(
                image_size,
                interpolation=PIL.Image.Resampling.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(
                mean=model_transform_config["mean"],
                std=model_transform_config["std"],
            ),
        ]
    )
    train_transforms = T.Compose(
        [
            T.Resize(
                image_size,
                interpolation=PIL.Image.Resampling.BICUBIC,
            ),
            StandardAugmentations(),
            T.ToTensor(),
            T.Normalize(
                mean=model_transform_config["mean"],
                std=model_transform_config["std"],
            ),
        ]
    )

    train_dataset = ISIC2024Dataset(
        root_dir=root_dir,
        split_name=SplitNames.TRAIN,
        transform=train_transforms,
        importance_level_labels=Importance.HIGH,
        return_samples_as_dict=True,
    )

    val_dataset = ISIC2024Dataset(
        root_dir=root_dir,
        split_name=SplitNames.VAL,
        transform=common_transforms,
        importance_level_labels=Importance.HIGH,
        return_samples_as_dict=True,
    )

    test_dataset = ISIC2024Dataset(
        root_dir=root_dir,
        split_name=SplitNames.DEVTEST,
        transform=common_transforms,
        importance_level_labels=Importance.HIGH,
        return_samples_as_dict=True,
    )

    # Use BalancedBatchSampler for training
    try:
        train_sampler = BalancedBatchSampler(
            train_dataset, train_batch_size, class_ratios=[0.5, 0.5]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(
            f"Successfully created BalancedBatchSampler with {len(train_sampler)} batches"
        )
    except Exception as e:
        print(f"Error creating BalancedBatchSampler: {str(e)}")
        print("Falling back to regular DataLoader for training")
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def save_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    iter_id: int,
    metrics: Dict[str, float],
    accelerator: Accelerator,
    is_best: bool = False,
) -> None:
    if accelerator.is_local_main_process:
        checkpoint_dir = f"{checkpoint_dir}/iter_{iter_id}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "model_state_dict": accelerator.get_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "iter_id": iter_id,
            "metrics": metrics,
        }

        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, f"checkpoint_{iter_id}.pth"),
        )
        torch.save(
            checkpoint, os.path.join(checkpoint_dir, "checkpoint_latest.pth")
        )

        with open(os.path.join(checkpoint_dir, "metrics.txt"), "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        if is_best:
            torch.save(
                checkpoint, os.path.join(checkpoint_dir, "checkpoint_best.pth")
            )


def set_dropout_rate(model: nn.Module, dropout_rate: float) -> nn.Module:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
    return model


def main():
    # Hyperparameters
    TRAIN_MICRO_BATCH_SIZE = 128
    EVAL_BATCH_SIZE = 512
    NUM_WORKERS = 16
    LEARNING_RATE = 6e-6
    NUM_TRAIN_ITER = 10000
    VALIDATE_EVERY = 100
    NUM_CLASSES = 1  # Binary classification
    TOP_K_MODELS = 3
    SEED = 42
    ROOT_DIR = "/mnt/nvme-fast0/datasets/"  # Adjust this to your dataset path
    MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
    MIXED_PRECISION = "bf16"
    PROJECT_NAME = "isic2024-training"
    IMAGE_SIZE = 224
    WEIGHT_DECAY = 0.0001
    DROPOUT_RATE = 0.5
    EXPERIMENT_NAME = (
        f"{MODEL_NAME}_{LEARNING_RATE}_{WEIGHT_DECAY}_{DROPOUT_RATE}_{SEED}"
    )

    CHECKPOINT_DIR = (
        f"/mnt/nvme-fast0/experiments/{PROJECT_NAME}/{EXPERIMENT_NAME}"
    )
    LOG_EVERY = 100  # Log every 10 steps

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=1,
    )

    # Set seeds for reproducibility
    set_seed(SEED)
    accelerator.print(f"Random seed set to {SEED}")

    # Initialize wandb (only on main process)
    if accelerator.is_local_main_process:
        wandb.init(
            project=PROJECT_NAME,
            config={
                "train_batch_size": TRAIN_MICRO_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "num_train_iter": NUM_TRAIN_ITER,
                "validate_every": VALIDATE_EVERY,
                "num_classes": NUM_CLASSES,
                "top_k_models": TOP_K_MODELS,
                "seed": SEED,
                "model_name": MODEL_NAME,
                "mixed_precision": MIXED_PRECISION,
                "image_size": IMAGE_SIZE,
            },
        )

    accelerator.print("[bold green]Creating dataloaders...[/bold green]")

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=NUM_CLASSES,
        drop_rate=DROPOUT_RATE,
    )
    model = set_dropout_rate(model, DROPOUT_RATE)

    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)

    print(model)

    train_loader, val_loader, test_loader = create_dataloaders(
        TRAIN_MICRO_BATCH_SIZE,
        EVAL_BATCH_SIZE,
        NUM_WORKERS,
        ROOT_DIR,
        IMAGE_SIZE,
        SEED,
        data_cfg,
    )

    accelerator.print("[bold green]Loading model...[/bold green]")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    model, optimizer, train_loader, val_loader, test_loader = (
        accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader
        )
    )

    metric_cache = []
    image_cache = []

    accelerator.print("[bold green]Starting training...[/bold green]")

    train_iter = iter(train_loader)
    for iter_id in tqdm(range(1, NUM_TRAIN_ITER + 1), desc="Training"):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        images, labels = batch["image"], batch["labels"]["target"].float()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        # Compute metrics
        metrics = compute_metrics(outputs.detach().cpu(), labels.cpu())
        metrics["train_iter"] = iter_id
        metric_cache.append(metrics)

        # Cache images
        image_cache.extend(
            [
                {
                    "image": img.cpu(),
                    "output": out.cpu(),
                    "label": lbl.cpu(),
                }
                for img, out, lbl in zip(batch["image"], outputs, labels)
            ]
        )

        # Log metrics and images every LOG_EVERY steps
        if iter_id % LOG_EVERY == 0:
            log_metrics_and_images(
                {
                    k: np.mean([m[k] for m in metric_cache])
                    for k in metric_cache[0]
                },
                image_cache,
                "train",
                iter_id,
                accelerator,
            )
            metric_cache = []
            image_cache = []

        if iter_id % VALIDATE_EVERY == 0:
            model.eval()
            val_outputs = []
            val_labels = []
            with torch.no_grad():
                for val_batch in tqdm(val_loader):
                    val_images, val_targets = (
                        val_batch["image"],
                        val_batch["labels"]["target"].float(),
                    )
                    val_out = model(val_images).squeeze()
                    val_outputs.append(val_out.cpu())
                    val_labels.append(val_targets.cpu())

            val_outputs = torch.cat(val_outputs)
            val_labels = torch.cat(val_labels)

            # Compute and log validation metrics
            val_metrics = compute_metrics(val_outputs.cpu(), val_labels.cpu())
            val_metrics["val_iter"] = iter_id
            log_metrics_and_images(
                val_metrics,
                [
                    {
                        "image": img.cpu(),
                        "output": out.cpu(),
                        "label": lbl.cpu(),
                    }
                    for img, out, lbl in zip(
                        val_batch["image"], val_outputs, val_labels
                    )
                ],
                "val",
                iter_id,
                accelerator,
            )

            # Save checkpoint
            save_checkpoint(
                CHECKPOINT_DIR,
                model,
                optimizer,
                iter_id // len(train_loader) + 1,
                iter_id,
                val_metrics,
                accelerator,
                is_best=False,  # We'll determine this after getting metrics
            )

    accelerator.print(
        "[bold green]Training completed. Creating ensemble...[/bold green]"
    )

    # Load the best checkpoints and create an ensemble
    best_checkpoints = sorted(os.listdir(CHECKPOINT_DIR))[-TOP_K_MODELS:]
    ensemble_models = []
    for checkpoint_name in best_checkpoints:
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, checkpoint_name, "checkpoint_latest.pth"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=accelerator.device
        )
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=NUM_CLASSES,
            drop_rate=DROPOUT_RATE,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        ensemble_models.append(model)

    ensemble = AveragingEnsemble(ensemble_models)
    ensemble = accelerator.prepare(ensemble)

    accelerator.print("[bold green]Testing ensemble...[/bold green]")
    ensemble.eval()
    test_outputs = []
    test_labels = []
    with torch.no_grad():
        for test_batch in tqdm(test_loader, desc="Testing"):
            test_images, test_targets = (
                test_batch["image"],
                test_batch["labels"]["target"].float(),
            )
            test_out = ensemble(test_images).squeeze()
            test_outputs.append(test_out)
            test_labels.append(test_targets)

    test_outputs = torch.cat(test_outputs)
    test_labels = torch.cat(test_labels)

    # Compute and log test metrics
    test_metrics = compute_metrics(test_outputs.cpu(), test_labels.cpu())

    if accelerator.is_local_main_process:
        accelerator.print(
            f"[bold cyan]Ensemble Test Results:[/bold cyan] {test_metrics}"
        )
        wandb.log({"ensemble_test_" + k: v for k, v in test_metrics.items()})

        # Log per-class metrics for ensemble test results
        if "class_0_accuracy" in test_metrics:
            table = wandb.Table(
                columns=[
                    "Class",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "AUC-ROC",
                ]
            )
            for class_id in range(2):  # Assuming binary classification
                table.add_data(
                    class_id,
                    test_metrics[f"class_{class_id}_accuracy"],
                    test_metrics[f"class_{class_id}_precision"],
                    test_metrics[f"class_{class_id}_recall"],
                    test_metrics[f"class_{class_id}_f1"],
                    test_metrics.get(f"class_{class_id}_auc_roc", "N/A"),
                )
            wandb.log({"ensemble_test_per_class_metrics": table})

        log_metrics_and_images(
            test_metrics,
            [
                {
                    "image": img.cpu(),
                    "output": out.cpu(),
                    "label": lbl.cpu(),
                }
                for img, out, lbl in zip(
                    test_batch["image"], test_outputs, test_labels
                )
            ],
            "val",
            iter_id,
            accelerator,
        )

        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
