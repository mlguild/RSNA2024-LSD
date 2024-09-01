import os
from typing import Callable, Dict, List, Tuple

import PIL
import PIL.Image
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.utils import set_seed
from gate.data.image.classification.imagenet1k import StandardAugmentations
from rich import print
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from lsd.src.lsd.data import (
    BalancedBatchSampler,
    Importance,
    ISIC2024Dataset,
    SplitNames,
)


class AveragingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([model(x).squeeze() for model in self.models]).mean(
            dim=0
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


def compute_metrics(
    outputs: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    with torch.no_grad():
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        labels = labels.cpu().numpy()
        predictions = (probabilities > 0.5).astype(int)

        accuracy = (predictions == labels).mean()
        if np.unique(labels).size > 1:
            # Compute AUC-ROC
            auc_roc = roc_auc_score(labels, probabilities)

            # Compute pAUC above 80% TPR
            pauc = compute_pauc_above_80_tpr(labels, probabilities)

            # Compute precision, recall, and F1 score
            precision, recall, _ = precision_recall_curve(
                labels, probabilities
            )
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_score = np.max(f1_scores)

            # Compute average precision
            ap = average_precision_score(labels, probabilities)

            return {
                "accuracy": accuracy,
                "auc_roc": auc_roc,
                "pauc_above_80_tpr": pauc,
                "f1_score": f1_score,
                "average_precision": ap,
            }
        else:
            return {"accuracy": accuracy}


def step(
    model: nn.Module, batch: Dict[str, torch.Tensor], criterion: nn.Module
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor]:
    images, labels = batch["image"], batch["labels"]["target"].float()
    outputs = model(images).squeeze()
    loss = criterion(outputs, labels)
    metrics = compute_metrics(outputs.detach().cpu(), labels.cpu())
    return loss, metrics, outputs, labels


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    accelerator: Accelerator,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    model.train()
    loss, metrics, outputs, labels = step(model, batch, criterion)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    metrics["loss"] = loss.item()
    return metrics, outputs.detach().cpu(), labels.cpu()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    accelerator: Accelerator,
) -> Tuple[Dict[str, float], List[Dict[str, torch.Tensor]]]:
    model.eval()
    all_outputs = []
    all_labels = []
    all_losses = []
    all_results = []

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Evaluating",
            disable=not accelerator.is_local_main_process,
        ):
            loss, _, outputs, labels = step(model, batch, criterion)
            all_outputs.append(outputs)
            all_labels.append(labels)
            all_losses.append(loss)
            all_results.extend(
                [
                    {
                        "image": img.cpu(),
                        "output": out.cpu(),
                        "label": lbl.cpu(),
                    }
                    for img, out, lbl in zip(batch["image"], outputs, labels)
                ]
            )

    avg_loss = np.mean([loss.cpu() for loss in all_losses])
    metrics = compute_metrics(torch.cat(all_outputs), torch.cat(all_labels))
    metrics["loss"] = avg_loss

    return metrics, all_results


def log_images_to_wandb(
    results: List[Dict[str, torch.Tensor]], mode: str, num_images: int = 100
):
    images = [
        wandb.Image(
            r["image"],
            caption=f"Predicted: {torch.sigmoid(r['output']).item():.4f}, Target: {r['label'].item()}",
        )
        for r in results[:num_images]
    ]
    wandb.log({f"{mode}_examples": images})


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


def main():
    from torch.nn import SyncBatchNorm

    # Hyperparameters
    TRAIN_MICRO_BATCH_SIZE = 256

    EVAL_BATCH_SIZE = 512
    NUM_WORKERS = 16
    LEARNING_RATE = 6e-6
    NUM_TRAIN_ITER = 10000
    VALIDATE_EVERY = 100
    NUM_CLASSES = 1  # Binary classification
    TOP_K_MODELS = 3
    SEED = 42
    VAL_SPLIT = 0.05
    TEST_SPLIT = 0.05
    ROOT_DIR = "/mnt/nvme-fast0/datasets/"  # Adjust this to your dataset path
    MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
    MIXED_PRECISION = "bf16"
    PROJECT_NAME = "isic2024-training"
    EXPERIMENT_NAME = (
        "{MODEL_NAME}_{LEARNING_RATE}_{WEIGHT_DECAY}_{DROPOUT_RATE}_{SEED}"
    )
    IMAGE_SIZE = 224
    WEIGHT_DECAY = 0.0001
    DROPOUT_RATE = 0.5
    CHECKPOINT_DIR = f"experiments/{PROJECT_NAME}/{EXPERIMENT_NAME}"

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
                "val_split": VAL_SPLIT,
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

    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)

    model = SyncBatchNorm.convert_sync_batchnorm(model)

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

    accelerator.print("[bold green]Starting training...[/bold green]")
    best_val_pauc = 0
    best_models = []

    train_iter = iter(train_loader)
    train_results = []
    epoch_metrics = {
        "loss": 0,
        "accuracy": 0,
        "auc_roc": 0,
        "pauc_above_80_tpr": 0,
        "f1_score": 0,
        "average_precision": 0,
    }
    num_batches = 0
    with tqdm(total=NUM_TRAIN_ITER, desc="Training") as pbar:
        for iter_id in range(1, NUM_TRAIN_ITER + 1):
            try:
                batch = next(train_iter)

            except StopIteration:
                # Log epoch metrics
                if accelerator.is_local_main_process:
                    wandb.log(
                        {
                            f"epoch_{k}": v / num_batches
                            for k, v in epoch_metrics.items()
                        }
                    )
                epoch_metrics = {
                    "loss": 0,
                    "accuracy": 0,
                    "auc_roc": 0,
                    "pauc_above_80_tpr": 0,
                    "f1_score": 0,
                    "average_precision": 0,
                }
                num_batches = 0
                train_iter = iter(train_loader)
                batch = next(train_iter)

            metrics, outputs, labels = train_step(
                model, batch, criterion, optimizer, accelerator
            )

            # Update epoch metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1

            if accelerator.is_local_main_process:
                wandb.log({f"iter_{k}": v for k, v in metrics.items()})

                train_results.extend(
                    [
                        {
                            "image": img,
                            "output": out,
                            "label": lbl,
                        }
                        for img, out, lbl in zip(
                            batch["image"], outputs, labels
                        )
                    ]
                )
                if len(train_results) >= 100:
                    log_images_to_wandb(train_results, "train")
                    train_results = []

            if accelerator.is_local_main_process:
                pbar.update(1)
                pbar.set_description(f"loss: {metrics['loss']}")

            if iter_id % VALIDATE_EVERY == 0:
                val_metrics, val_results = evaluate(
                    model, val_loader, criterion, accelerator
                )

                if accelerator.is_local_main_process:
                    wandb.log({f"val_{k}": v for k, v in val_metrics.items()})
                    log_images_to_wandb(val_results, "val")
                    accelerator.print(
                        f"[cyan]Iteration {iter_id}:[/cyan] {val_metrics}"
                    )

                is_best = val_metrics["pauc_above_80_tpr"] > best_val_pauc
                save_checkpoint(
                    CHECKPOINT_DIR,
                    model,
                    optimizer,
                    iter_id // len(train_loader) + 1,
                    iter_id,
                    val_metrics,
                    accelerator,
                    is_best,
                )

                if is_best:
                    best_val_pauc = val_metrics["pauc_above_80_tpr"]

                best_models.append((val_metrics["pauc_above_80_tpr"], iter_id))
                best_models.sort(reverse=True)
                best_models = best_models[:TOP_K_MODELS]

    accelerator.print(
        "[bold green]Training completed. Creating ensemble...[/bold green]"
    )
    ensemble_models = []
    for _, iter_id in best_models:
        checkpoint = torch.load(
            f"{CHECKPOINT_DIR}/iter_{iter_id}/checkpoint_{iter_id}.pth"
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
    test_metrics, test_results = evaluate(
        ensemble, test_loader, criterion, accelerator
    )

    if accelerator.is_local_main_process:
        accelerator.print(
            f"[bold cyan]Ensemble Test Results:[/bold cyan] {test_metrics}"
        )

        wandb.log({"ensemble_test_" + k: v for k, v in test_metrics.items()})
        log_images_to_wandb(
            test_results, "test", num_images=len(test_results)
        )  # Log all test cases

        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
