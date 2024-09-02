import importlib
import os
import pathlib
import sys
from dataclasses import dataclass

import fire
import numpy as np
import PIL
import PIL.Image
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Importance,
    ISIC2024Dataset,
    SplitNames,
)

install()


class AveragingEnsemble(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([model(x).squeeze() for model in self.models]).mean(
            dim=0
        )


def compute_pauc_above_80_tpr(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
    idx_80_tpr = np.argmax(tpr >= 0.8)
    tpr_slice = tpr[idx_80_tpr:]
    fpr_slice = fpr[idx_80_tpr:]
    pauc = np.trapz(y=tpr_slice, x=fpr_slice)
    pauc_normalized = pauc / 0.2
    return pauc_normalized


def compute_metrics(
    outputs: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    with torch.no_grad():
        loss = F.binary_cross_entropy_with_logits(
            input=outputs, target=labels
        ).item()
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        labels = labels.cpu().numpy()
        predictions = (probabilities > 0.5).astype(int)
        accuracy = (predictions == labels).mean()
        metrics = {
            "accuracy": accuracy,
            "loss": loss,
        }
        if np.unique(labels).size > 1:
            auc_roc = roc_auc_score(y_true=labels, y_score=probabilities)
            pauc = compute_pauc_above_80_tpr(
                y_true=labels, y_pred=probabilities
            )
            ap = average_precision_score(y_true=labels, y_score=probabilities)
            metrics.update(
                {
                    "auc_roc": auc_roc,
                    "pauc_above_80_tpr": pauc,
                    "average_precision": ap,
                }
            )
            for class_id in range(2):
                class_mask = labels == class_id
                class_labels = labels[class_mask]
                class_predictions = predictions[class_mask]
                class_accuracy = (class_predictions == class_labels).mean()
                metrics.update({f"class_{class_id}_accuracy": class_accuracy})
        return metrics


def log_metrics_and_images(
    metrics: dict[str, float],
    images: list[dict],
    mode: str,
    iter_id: int,
    accelerator: Accelerator,
):
    if accelerator.is_local_main_process:
        wandb.log({f"{mode}_{k}": v for k, v in metrics.items()})
        if "class_0_accuracy" in metrics:
            table = wandb.Table(columns=["Class", "Accuracy", "AUC-ROC"])
            for class_id in range(2):
                table.add_data(
                    class_id,
                    metrics[f"class_{class_id}_accuracy"],
                    metrics.get(f"class_{class_id}_auc_roc", "N/A"),
                )
            wandb.log({f"{mode}_per_class_metrics": table})
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
    model_transform_config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common_transforms = T.Compose(
        [
            T.Resize(
                size=image_size, interpolation=PIL.Image.Resampling.BICUBIC
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
                size=image_size, interpolation=PIL.Image.Resampling.BICUBIC
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

    try:
        train_sampler = BalancedBatchSampler(
            dataset=train_dataset,
            batch_size=train_batch_size,
            class_ratios=[0.5, 0.5],
        )
        train_loader = DataLoader(
            dataset=train_dataset,
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
            dataset=train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
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
    metrics: dict[str, float],
    accelerator: Accelerator,
    is_best: bool = False,
) -> None:
    if accelerator.is_local_main_process:
        checkpoint_dir = f"{checkpoint_dir}/iter_{iter_id}"
        os.makedirs(name=checkpoint_dir, exist_ok=True)
        checkpoint = {
            "model_state_dict": accelerator.get_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "iter_id": iter_id,
            "metrics": metrics,
        }
        torch.save(
            obj=checkpoint,
            f=os.path.join(checkpoint_dir, f"checkpoint_{iter_id}.pth"),
        )
        torch.save(
            obj=checkpoint,
            f=os.path.join(checkpoint_dir, "checkpoint_latest.pth"),
        )
        with open(
            file=os.path.join(checkpoint_dir, "metrics.txt"), mode="w"
        ) as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        if is_best:
            torch.save(
                obj=checkpoint,
                f=os.path.join(checkpoint_dir, "checkpoint_best.pth"),
            )


def set_dropout_rate(model: nn.Module, dropout_rate: float) -> nn.Module:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
    return model


def initialize_wandb(config, accelerator: Accelerator):
    if accelerator.is_local_main_process:
        wandb.init(
            project=config.PROJECT_NAME,
            config={k: v for k, v in vars(config).items() if k.isupper()},
        )


def load_checkpoint(
    checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer
):
    checkpoint = torch.load(f=checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    optimizer.load_state_dict(state_dict=checkpoint["optimizer_state_dict"])
    start_iter = checkpoint["iter_id"] + 1
    return model, optimizer, start_iter


def find_latest_checkpoint(checkpoint_dir: str):
    print(f"Checkpoint directory: {checkpoint_dir}")
    if os.path.exists(checkpoint_dir):
        checkpoints = [
            d for d in os.listdir(checkpoint_dir) if d.startswith("iter_")
        ]
        print(f"Checkpoints found: {checkpoints}")
        if checkpoints:
            latest_checkpoint = max(
                checkpoints, key=lambda x: int(x.split("_")[1])
            )
            print(f"Latest checkpoint: {latest_checkpoint}")
            return os.path.join(
                checkpoint_dir, latest_checkpoint, "checkpoint_latest.pth"
            )
    return None


def create_model(config, pretrained: bool):
    model = timm.create_model(
        model_name=config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=config.NUM_CLASSES,
        drop_rate=config.DROPOUT_RATE,
    )
    return set_dropout_rate(model=model, dropout_rate=config.DROPOUT_RATE)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    accelerator: Accelerator,
    start_iter: int,
    config,
):
    model.train()
    metric_cache = []
    image_cache = []
    train_iter = iter(train_loader)
    for iter_id in tqdm(
        range(start_iter, config.NUM_TRAIN_ITER + 1), desc="Training"
    ):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images, labels = batch["image"], batch["labels"]["target"].float()
        outputs = model(images).squeeze()
        loss = criterion(input=outputs, target=labels)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        metrics = compute_metrics(
            outputs=outputs.detach().cpu(), labels=labels.cpu()
        )
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
                metrics={
                    k: np.mean([m[k] for m in metric_cache])
                    for k in metric_cache[0]
                },
                images=image_cache,
                mode="train",
                iter_id=iter_id,
                accelerator=accelerator,
            )
            metric_cache = []
            image_cache = []
        if iter_id % config.VALIDATE_EVERY == 0:
            validate(
                model=model,
                val_loader=val_loader,
                accelerator=accelerator,
                iter_id=iter_id,
            )
            save_checkpoint(
                checkpoint_dir=config.get_checkpoint_dir(),
                model=model,
                optimizer=optimizer,
                epoch=iter_id // len(train_loader) + 1,
                iter_id=iter_id,
                metrics=metrics,
                accelerator=accelerator,
                is_best=False,
            )


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    accelerator: Accelerator,
    iter_id: int,
):
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
    val_metrics = compute_metrics(outputs=val_outputs, labels=val_labels)
    val_metrics["val_iter"] = iter_id
    log_metrics_and_images(
        metrics=val_metrics,
        images=[
            {"image": img.cpu(), "output": out.cpu(), "label": lbl.cpu()}
            for img, out, lbl in zip(
                val_batch["image"], val_outputs, val_labels
            )
        ],
        mode="val",
        iter_id=iter_id,
        accelerator=accelerator,
    )


def create_ensemble(config, checkpoint_dir: str, accelerator: Accelerator):
    best_checkpoints = sorted(os.listdir(checkpoint_dir))[
        -config.TOP_K_MODELS :
    ]
    ensemble_models = []
    for checkpoint_name in best_checkpoints:
        checkpoint_path = os.path.join(
            checkpoint_dir, checkpoint_name, "checkpoint_latest.pth"
        )
        checkpoint = torch.load(
            f=checkpoint_path, map_location=accelerator.device
        )
        model = create_model(config=config, pretrained=False)
        model.load_state_dict(state_dict=checkpoint["model_state_dict"])
        ensemble_models.append(model)
    return AveragingEnsemble(models=ensemble_models)


def test_ensemble(ensemble: nn.Module, test_loader: DataLoader):
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
    test_metrics: dict[str, float],
    test_batch: dict,
    test_outputs: torch.Tensor,
    test_labels: torch.Tensor,
    accelerator: Accelerator,
    config,
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
            for class_id in range(2):
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
            metrics=test_metrics,
            images=[
                {"image": img.cpu(), "output": out.cpu(), "label": lbl.cpu()}
                for img, out, lbl in zip(
                    test_batch["image"], test_outputs, test_labels
                )
            ],
            mode="test",
            iter_id=config.NUM_TRAIN_ITER,
            accelerator=accelerator,
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
        sys.path.append(str(config_path.parent.parent))
        module_name = f"{config_path.parent.name}.{config_path.stem}"
        config = importlib.import_module(name=module_name)
        print(config)
        config = Config(
            **{k: v for k, v in vars(config).items() if k.isupper()}
        )
    print(f"Config being used is {config}")

    accelerator = Accelerator(
        mixed_precision=config.MIXED_PRECISION, gradient_accumulation_steps=1
    )
    set_seed(seed=config.SEED)
    accelerator.print(f"Random seed set to {config.SEED}")

    initialize_wandb(config=config, accelerator=accelerator)

    latest_checkpoint_path = find_latest_checkpoint(
        checkpoint_dir=config.get_checkpoint_dir()
    )
    model = create_model(
        config=config, pretrained=(latest_checkpoint_path is None)
    )
    model_transforms = timm.data.resolve_data_config(model.pretrained_cfg)
    print(model, model_transforms)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_batch_size=config.TRAIN_MICRO_BATCH_SIZE,
        eval_batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        root_dir=config.ROOT_DIR,
        image_size=config.IMAGE_SIZE,
        seed=config.SEED,
        model_transform_config=model_transforms,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    start_iter = 1
    if latest_checkpoint_path:
        model, optimizer, start_iter = load_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
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
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        accelerator=accelerator,
        config=config,
        start_iter=start_iter,
    )

    accelerator.print(
        "[bold green]Training completed. Creating ensemble...[/bold green]"
    )
    ensemble = create_ensemble(
        config=config,
        checkpoint_dir=config.get_checkpoint_dir(),
        accelerator=accelerator,
    )
    ensemble = accelerator.prepare(ensemble)

    accelerator.print("[bold green]Testing ensemble...[/bold green]")
    test_outputs, test_labels = test_ensemble(
        ensemble=ensemble, test_loader=test_loader
    )
    test_metrics = compute_metrics(
        outputs=test_outputs.cpu(), labels=test_labels.cpu()
    )

    log_test_results(
        test_metrics=test_metrics,
        test_batch=test_loader.dataset[0],
        test_outputs=test_outputs,
        test_labels=test_labels,
        accelerator=accelerator,
        config=config,
    )

    if accelerator.is_local_main_process:
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire(main)
