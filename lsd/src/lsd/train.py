from dataclasses import dataclass
import importlib
import os
import pathlib
import sys
from typing import Callable, Dict, List, Tuple

import fire
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


def initialize_wandb(config, accelerator):
    if accelerator.is_local_main_process:
        wandb.init(
            project=config.PROJECT_NAME,
            config={k: v for k, v in vars(config).items() if k.isupper()},
        )


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_iter = checkpoint["iter_id"] + 1
    return model, optimizer, start_iter


def find_latest_checkpoint(checkpoint_dir):
    if os.path.exists(checkpoint_dir):
        checkpoints = [
            d for d in os.listdir(checkpoint_dir) if d.startswith("iter_")
        ]
        if checkpoints:
            latest_checkpoint = max(
                checkpoints, key=lambda x: int(x.split("_")[1])
            )
            return os.path.join(
                checkpoint_dir, latest_checkpoint, "checkpoint_latest.pth"
            )
    return None


def create_model(config, pretrained):
    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=config.NUM_CLASSES,
        drop_rate=config.DROPOUT_RATE,
    )
    return set_dropout_rate(model, config.DROPOUT_RATE)


def train_epoch(
    model, train_loader, val_loader, optimizer, criterion, accelerator, config
):
    model.train()
    metric_cache = []
    image_cache = []

    train_iter = iter(train_loader)
    for iter_id in tqdm(range(1, config.NUM_TRAIN_ITER + 1), desc="Training"):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images, labels = batch["image"], batch["labels"]["target"].float()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        metrics = compute_metrics(outputs.detach().cpu(), labels.cpu())
        metrics["train_iter"] = iter_id
        metric_cache.append(metrics)

        image_cache.extend(
            [
                {"image": img.cpu(), "output": out.cpu(), "label": lbl.cpu()}
                for img, out, lbl in zip(batch["image"], outputs, labels)
            ]
        )

        if iter_id % config.LOG_EVERY == 0:
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

        if iter_id % config.VALIDATE_EVERY == 0:
            validate(
                model, val_loader, criterion, accelerator, config, iter_id
            )
            save_checkpoint(
                config.get_checkpoint_dir(),
                model,
                optimizer,
                iter_id // len(train_loader) + 1,
                iter_id,
                accelerator,
                is_best=False,
            )


def validate(model, val_loader, accelerator, iter_id):
    model.eval()
    val_outputs, val_labels = [], []
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Validating"):
            val_images, val_targets = (
                val_batch["image"],
                val_batch["labels"]["target"].float(),
            )
            val_out = model(val_images).squeeze()
            val_outputs.append(val_out.cpu())
            val_labels.append(val_targets.cpu())

    val_outputs = torch.cat(val_outputs)
    val_labels = torch.cat(val_labels)

    val_metrics = compute_metrics(val_outputs, val_labels)
    val_metrics["val_iter"] = iter_id
    log_metrics_and_images(
        val_metrics,
        [
            {"image": img.cpu(), "output": out.cpu(), "label": lbl.cpu()}
            for img, out, lbl in zip(
                val_batch["image"], val_outputs, val_labels
            )
        ],
        "val",
        iter_id,
        accelerator,
    )


def create_ensemble(config, checkpoint_dir, accelerator):
    best_checkpoints = sorted(os.listdir(checkpoint_dir))[
        -config.TOP_K_MODELS :
    ]
    ensemble_models = []
    for checkpoint_name in best_checkpoints:
        checkpoint_path = os.path.join(
            checkpoint_dir, checkpoint_name, "checkpoint_latest.pth"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=accelerator.device
        )
        model = create_model(config, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        ensemble_models.append(model)
    return AveragingEnsemble(ensemble_models)


def test_ensemble(ensemble, test_loader):
    ensemble.eval()
    test_outputs, test_labels = [], []
    with torch.no_grad():
        for test_batch in tqdm(test_loader, desc="Testing"):
            test_images, test_targets = (
                test_batch["image"],
                test_batch["labels"]["target"].float(),
            )
            test_out = ensemble(test_images).squeeze()
            test_outputs.append(test_out)
            test_labels.append(test_targets)
    return torch.cat(test_outputs), torch.cat(test_labels)


def log_test_results(
    test_metrics, test_batch, test_outputs, test_labels, accelerator, config
):
    if accelerator.is_local_main_process:
        accelerator.print(
            f"[bold cyan]Ensemble Test Results:[/bold cyan] {test_metrics}"
        )
        wandb.log({"ensemble_test_" + k: v for k, v in test_metrics.items()})

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
                    test_metrics.get(f"class_{class_id}_precision", "N/A"),
                    test_metrics.get(f"class_{class_id}_recall", "N/A"),
                    test_metrics.get(f"class_{class_id}_f1", "N/A"),
                    test_metrics.get(f"class_{class_id}_auc_roc", "N/A"),
                )
            wandb.log({"ensemble_test_per_class_metrics": table})

        log_metrics_and_images(
            test_metrics,
            [
                {"image": img.cpu(), "output": out.cpu(), "label": lbl.cpu()}
                for img, out, lbl in zip(
                    test_batch["image"], test_outputs, test_labels
                )
            ],
            "test",
            config.NUM_TRAIN_ITER,
            accelerator,
        )


@dataclass
class Config:
    TRAIN_MICRO_BATCH_SIZE: int = 128
    EVAL_BATCH_SIZE: int = 512
    NUM_WORKERS: int = 16
    LEARNING_RATE: float = 6e-6
    NUM_TRAIN_ITER: int = 10000
    VALIDATE_EVERY: int = 100
    NUM_CLASSES: int = 1
    TOP_K_MODELS: int = 3
    SEED: int = 42
    ROOT_DIR: str = "/mnt/nvme-fast0/datasets/"
    MODEL_NAME: str = "convnext_base.fb_in22k_ft_in1k"
    MIXED_PRECISION: str = "bf16"
    PROJECT_NAME: str = "isic2024-training"
    IMAGE_SIZE: int = 224
    WEIGHT_DECAY: float = 0.0001
    DROPOUT_RATE: float = 0.5
    LOG_EVERY: int = 100

    def get_experiment_name(self):
        return f"{self.MODEL_NAME}_{self.LEARNING_RATE}_{self.WEIGHT_DECAY}_{self.DROPOUT_RATE}_{self.SEED}"

    def get_checkpoint_dir(self):
        return f"/mnt/nvme-fast0/experiments/{self.PROJECT_NAME}/{self.get_experiment_name()}"


def main(config: str | pathlib.Path | Config):

    if isinstance(config, (str, pathlib.Path)):
        config_path = pathlib.Path(config).resolve()

        # Add the parent directory of the config file to sys.path
        sys.path.append(str(config_path.parent.parent))

        # Import the config module
        module_name = f"{config_path.parent.name}.{config_path.stem}"
        config = importlib.import_module(module_name)

        # Create a Config instance with values from the imported module
        config = Config(
            **{k: v for k, v in vars(config).items() if k.isupper()}
        )
    print(f"Config being used is {config}")

    accelerator = Accelerator(
        mixed_precision=config.MIXED_PRECISION, gradient_accumulation_steps=1
    )
    set_seed(config.SEED)
    accelerator.print(f"Random seed set to {config.SEED}")

    initialize_wandb(config, accelerator)

    latest_checkpoint_path = find_latest_checkpoint(
        config.get_checkpoint_dir()
    )
    model = create_model(config, pretrained=(latest_checkpoint_path is None))
    model_transforms = timm.data.resolve_data_config(model.pretrained_cfg)
    print(model, model_transforms)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_batch_size=config.TRAIN_MICRO_BATCH_SIZE,
        eval_batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        image_size=config.IMAGE_SIZE,
        root_dir=config.ROOT_DIR,
        seed=config.SEED,
        model_transform_config=model_transforms,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    start_iter = 1
    if latest_checkpoint_path:
        model, optimizer, start_iter = load_checkpoint(
            latest_checkpoint_path, model, optimizer
        )
        accelerator.print(
            f"[bold green]Resuming from iteration {start_iter}[/bold green]"
        )

    model, optimizer, train_loader, val_loader, test_loader = (
        accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader
        )
    )

    accelerator.print("[bold green]Starting/Resuming training...[/bold green]")
    train_epoch(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        accelerator,
        config,
    )

    accelerator.print(
        "[bold green]Training completed. Creating ensemble...[/bold green]"
    )
    ensemble = create_ensemble(
        config, config.get_checkpoint_dir(), accelerator
    )
    ensemble = accelerator.prepare(ensemble)

    accelerator.print("[bold green]Testing ensemble...[/bold green]")
    test_outputs, test_labels = test_ensemble(
        ensemble, test_loader, accelerator
    )
    test_metrics = compute_metrics(test_outputs.cpu(), test_labels.cpu())

    log_test_results(
        test_metrics,
        test_loader.dataset[0],
        test_outputs,
        test_labels,
        accelerator,
        config,
    )

    if accelerator.is_local_main_process:
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire(main)
