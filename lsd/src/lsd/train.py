import concurrent.futures
import io
import multiprocessing as mp
import os
import pathlib
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from queue import Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import PIL
import PIL.Image
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from accelerate import Accelerator
from accelerate.utils import set_seed
from gate.data import download_kaggle_dataset
from gate.data.image.classification.imagenet1k import StandardAugmentations
from rich import print
from rich.traceback import install
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from tqdm.auto import tqdm

import wandb

install()


class DatasetNames(StrEnum):
    ISIC_2024 = "isic-2024-challenge"


class SplitNames(StrEnum):
    TRAIN = "train"
    VAL = "val"
    DEVTEST = "devtest"
    TEST = "test"


ISIC2024_COUNT = 401064

METADATA_KEYS = set(
    [
        "isic_id",
        "target",
        "patient_id",
        "clin_size_long_diam_mm",
        "image_type",
        "tbp_tile_type",
        "tbp_lv_A",
        "tbp_lv_Aext",
        "tbp_lv_B",
        "tbp_lv_Bext",
        "tbp_lv_C",
        "tbp_lv_Cext",
        "tbp_lv_H",
        "tbp_lv_Hext",
        "tbp_lv_L",
        "tbp_lv_Lext",
        "tbp_lv_areaMM2",
        "tbp_lv_area_perim_ratio",
        "tbp_lv_color_std_mean",
        "tbp_lv_deltaA",
        "tbp_lv_deltaB",
        "tbp_lv_deltaL",
        "tbp_lv_deltaLB",
        "tbp_lv_deltaLBnorm",
        "tbp_lv_eccentricity",
        "tbp_lv_location",
        "tbp_lv_location_simple",
        "tbp_lv_minorAxisMM",
        "tbp_lv_nevi_confidence",
        "tbp_lv_norm_border",
        "tbp_lv_norm_color",
        "tbp_lv_perimeterMM",
        "tbp_lv_radial_color_std_max",
        "tbp_lv_stdL",
        "tbp_lv_stdLExt",
        "tbp_lv_symm_2axis",
        "tbp_lv_symm_2axis_angle",
        "tbp_lv_x",
        "tbp_lv_y",
        "tbp_lv_z",
        "attribution",
        "copyright_license",
        "iddx_full",
        "iddx_1",
        "tbp_lv_dnn_lesion_confidence",
    ]
)


@dataclass
class MetadataInfo:
    name: str
    importance: int


class Importance:
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    VERY_LOW = 4


METADATA_KEYS_TO_USE = [
    MetadataInfo(name="isic_id", importance=Importance.HIGH),
    MetadataInfo(name="target", importance=Importance.HIGH),
    MetadataInfo(name="patient_id", importance=Importance.VERY_LOW),
    MetadataInfo(name="clin_size_long_diam_mm", importance=Importance.MEDIUM),
    MetadataInfo(name="image_type", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_tile_type", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_A", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_Aext", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_B", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_Bext", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_C", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_Cext", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_H", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_Hext", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_L", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_Lext", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_areaMM2", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_area_perim_ratio", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_color_std_mean", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_deltaA", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_deltaB", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_deltaL", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_deltaLB", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_deltaLBnorm", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_eccentricity", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_location", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_location_simple", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_minorAxisMM", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_nevi_confidence", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_norm_border", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_norm_color", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_perimeterMM", importance=Importance.MEDIUM),
    MetadataInfo(
        name="tbp_lv_radial_color_std_max", importance=Importance.MEDIUM
    ),
    MetadataInfo(name="tbp_lv_stdL", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_stdLExt", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_symm_2axis", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_symm_2axis_angle", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_x", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_y", importance=Importance.MEDIUM),
    MetadataInfo(name="tbp_lv_z", importance=Importance.MEDIUM),
    MetadataInfo(name="attribution", importance=Importance.VERY_LOW),
    MetadataInfo(name="copyright_license", importance=Importance.VERY_LOW),
    MetadataInfo(name="iddx_full", importance=Importance.HIGH),
    MetadataInfo(name="iddx_1", importance=Importance.MEDIUM),
    MetadataInfo(
        name="tbp_lv_dnn_lesion_confidence", importance=Importance.HIGH
    ),
]
ISIC_ID = "isic_id"


@dataclass
class DataItem:
    image: PIL.Image.Image
    labels: Dict[str, Any]


class ISIC2024Dataset(Dataset):
    def __init__(
        self,
        root_dir: str | pathlib.Path,
        split_name: SplitNames | str = SplitNames.TRAIN,
        transform: Optional[Callable] = None,
        importance_level_labels: Importance = Importance.HIGH,
        return_samples_as_dict: bool = False,
    ):
        super().__init__()
        if isinstance(root_dir, str):
            root_dir = pathlib.Path(root_dir)

        self.root_dir = root_dir
        self.transform = transform
        self.download_extract_data()
        self.file_count_after_download_and_extract = ISIC2024_COUNT
        self.return_samples_as_dict = return_samples_as_dict

        self.split_name = split_name

        if split_name in {
            SplitNames.TRAIN,
            SplitNames.VAL,
            SplitNames.DEVTEST,
        }:
            target_split_name = "train"
        elif split_name == SplitNames.TEST:
            target_split_name = "test"
        else:
            raise ValueError(f"Invalid split name: {split_name}")

        self.data = (
            self.root_dir
            / DatasetNames.ISIC_2024
            / f"{target_split_name}-image.hdf5"
        )
        self.data = h5py.File(self.data, "r")
        self.metadata_path = (
            self.root_dir
            / DatasetNames.ISIC_2024
            / f"{target_split_name}-metadata.csv"
        )
        self.metadata = pd.read_csv(self.metadata_path)

        if split_name in {
            SplitNames.TRAIN,
            SplitNames.VAL,
            SplitNames.DEVTEST,
        }:
            self.metadata = self._split_metadata()

        self.label_keys_to_use = [
            item.name
            for item in METADATA_KEYS_TO_USE
            if item.importance <= importance_level_labels
        ]

        # Check if 'target' is available in the metadata
        self.has_targets = "target" in self.metadata.columns

        if self.has_targets:
            print("Creating class to index mapper...")
            self.class_indices = self.create_class_indices()
        else:
            print("No targets available. Skipping class to index mapping.")
            self.class_indices = None

    def _split_metadata(self):
        """Split the training metadata into train, val, and devtest"""
        train_ratio, val_ratio, devtest_ratio = 0.6, 0.10, 0.30
        train_end = int(train_ratio * len(self.metadata))
        val_end = int((train_ratio + val_ratio) * len(self.metadata))

        if self.split_name == SplitNames.TRAIN:
            return self.metadata[:train_end]
        elif self.split_name == SplitNames.VAL:
            return self.metadata[train_end:val_end]
        elif self.split_name == SplitNames.DEVTEST:
            return self.metadata[val_end:]
        else:
            raise ValueError(
                f"Invalid split name for splitting metadata: {self.split_name}"
            )

    def create_class_indices(self):
        if not self.has_targets:
            return None

        class_indices = defaultdict(list)

        def process_row(args):
            idx, row = args
            return int(row["target"]), idx

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=mp.cpu_count()
        ) as executor:
            future_to_row = {
                executor.submit(process_row, (idx, row)): idx
                for idx, row in self.metadata.iterrows()
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_row),
                total=len(self.metadata),
                desc="Processing rows",
            ):
                target, idx = future.result()
                class_indices[target].append(idx)

        return class_indices

    def download_extract_data(self):
        download_kaggle_dataset(
            dataset_name=DatasetNames.ISIC_2024,
            dataset_path=DatasetNames.ISIC_2024,
            target_dir_path=self.root_dir,
            is_competition=True,
            unzip=True,
            file_count_after_download_and_extract=4000,
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        labels = {
            key: self.metadata.iloc[idx][key]
            for key in self.label_keys_to_use
            if key in self.metadata.columns
        }
        image_bytes = self.data[labels[ISIC_ID]][()]
        image = PIL.Image.open(io.BytesIO(image_bytes))

        if self.transform:
            image = self.transform(image)

        output = DataItem(image=image, labels=labels)

        if self.return_samples_as_dict:
            output = asdict(output)
        return output


class BalancedBatchSampler(Sampler):
    def __init__(
        self,
        dataset: ISIC2024Dataset,
        batch_size: int,
        class_ratios: List[float],
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_ratios = class_ratios
        self.num_classes = len(class_ratios)

        print(
            f"Initializing BalancedBatchSampler with batch size {batch_size} and class ratios {class_ratios}"
        )

        if not self.dataset.has_targets:
            print(
                "Dataset has no targets. Using random sampling instead of balanced sampling."
            )
            self.indices = list(range(len(self.dataset)))
            self.num_batches_per_epoch = len(self.indices) // batch_size
            return

        # Create class indices based on the Subset
        self.class_indices = [[] for _ in range(self.num_classes)]
        for idx in range(len(self.dataset)):
            label = self.dataset.metadata.iloc[idx]["target"]
            self.class_indices[int(label)].append(idx)

        # Print class distribution
        for class_idx, indices in enumerate(self.class_indices):
            print(f"Class {class_idx}: {len(indices)} samples")

        # Calculate number of samples for each class in a batch
        self.samples_per_class = [
            max(1, int(batch_size * ratio)) for ratio in class_ratios
        ]

        # Adjust if total samples exceed batch size
        while sum(self.samples_per_class) > batch_size:
            max_class = self.samples_per_class.index(
                max(self.samples_per_class)
            )
            self.samples_per_class[max_class] -= 1

        print(f"Samples per class in each batch: {self.samples_per_class}")

        # Calculate the number of batches
        self.num_batches_per_epoch = min(
            len(indices) // samples
            for indices, samples in zip(
                self.class_indices, self.samples_per_class
            )
        )
        print(f"Number of batches per epoch: {self.num_batches_per_epoch}")
        self.batch_count = 0
        # Start off by reshuffling indices
        self._reshuffle_indices()

    def _reshuffle_indices(self):
        if not self.dataset.has_targets:
            np.random.shuffle(self.indices)
            return

        self.remaining_indices = [
            np.random.permutation(indices).tolist()
            for indices in self.class_indices
        ]

    def __iter__(self) -> Iterator[List[int]]:
        if not self.dataset.has_targets:
            for i in range(0, len(self.indices), self.batch_size):
                yield self.indices[i : i + self.batch_size]
            return

        while True:  # Infinite loop to keep yielding batches
            if self.batch_count >= self.num_batches_per_epoch:
                self._reshuffle_indices()
                self.batch_count = 0

            batch = []
            for class_idx, num_samples in enumerate(self.samples_per_class):
                class_indices = self.remaining_indices[class_idx]

                if len(class_indices) < num_samples:
                    self.remaining_indices[class_idx] = np.random.permutation(
                        self.class_indices[class_idx]
                    ).tolist()
                    class_indices = self.remaining_indices[class_idx]

                sampled_indices = class_indices[:num_samples]
                batch.extend(sampled_indices)
                self.remaining_indices[class_idx] = class_indices[
                    num_samples:
                ]  # Remove sampled indices

            np.random.shuffle(batch)
            yield batch
            self.batch_count += 1

    def __len__(self) -> int:
        return self.num_batches_per_epoch


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
            precision, recall, _ = precision_recall_curve(
                labels, probabilities
            )
            f1_scores = f1_score(labels, predictions)
            ap = average_precision_score(labels, probabilities)

            metrics.update(
                {
                    "auc_roc": auc_roc,
                    "pauc_above_80_tpr": pauc,
                    "f1_score": f1_scores,
                    "average_precision": ap,
                }
            )

            # Compute per-class metrics
            for class_id in range(2):  # Assuming binary classification
                class_mask = labels == class_id
                class_probabilities = probabilities[class_mask]
                class_labels = labels[class_mask]
                class_predictions = predictions[class_mask]

                class_accuracy = (class_predictions == class_labels).mean()
                class_precision = precision_score(
                    class_labels, class_predictions, average="binary"
                )
                class_recall = recall_score(
                    class_labels, class_predictions, average="binary"
                )
                class_f1 = f1_score(
                    class_labels, class_predictions, average="binary"
                )

                metrics.update(
                    {
                        f"class_{class_id}_accuracy": class_accuracy,
                        f"class_{class_id}_precision": class_precision,
                        f"class_{class_id}_recall": class_recall,
                        f"class_{class_id}_f1": class_f1,
                    }
                )

        return metrics


def compute_metrics_and_log(queue, accelerator):
    while True:
        item = queue.get()
        if item is None:
            break

        outputs, labels, iter_id, mode = item

        metrics = compute_metrics(outputs, labels)
        metrics[f"{mode}_iter"] = iter_id

        if accelerator.is_local_main_process:
            wandb.log({f"{mode}_{k}": v for k, v in metrics.items()})

            # Log per-class metrics in a table
            if "class_0_accuracy" in metrics:
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
                        metrics[f"class_{class_id}_accuracy"],
                        metrics[f"class_{class_id}_precision"],
                        metrics[f"class_{class_id}_recall"],
                        metrics[f"class_{class_id}_f1"],
                        metrics.get(f"class_{class_id}_auc_roc", "N/A"),
                    )
                wandb.log({f"{mode}_per_class_metrics": table})

        queue.task_done()


def log_images_thread(queue, accelerator):
    while True:
        item = queue.get()
        if item is None:
            break

        results, mode, iter_id = item

        if accelerator.is_local_main_process:
            images = [
                wandb.Image(
                    r["image"],
                    caption=f"Predicted: {torch.sigmoid(r['output']).item():.4f}, Target: {r['label'].item()}",
                )
                for r in results[:100]
            ]
            wandb.log({f"{mode}_examples": images, f"{mode}_iter": iter_id})

        queue.task_done()


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

    # Create queues and start threads
    metric_queue = Queue()
    image_queue = Queue()

    metric_thread = threading.Thread(
        target=compute_metrics_and_log, args=(metric_queue, accelerator)
    )
    image_thread = threading.Thread(
        target=log_images_thread, args=(image_queue, accelerator)
    )

    metric_thread.start()
    image_thread.start()

    accelerator.print("[bold green]Starting training...[/bold green]")
    best_val_pauc = 0
    best_models = []

    train_iter = iter(train_loader)
    train_results = []
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

        # Add metrics computation to queue
        metric_queue.put(
            (outputs.detach().cpu(), labels.cpu(), iter_id, "train")
        )

        if accelerator.is_local_main_process:
            train_results.extend(
                [
                    {
                        "image": img.cpu(),
                        "output": out.cpu(),
                        "label": lbl.cpu(),
                    }
                    for img, out, lbl in zip(batch["image"], outputs, labels)
                ]
            )
            if len(train_results) >= 100:
                image_queue.put((train_results, "train", iter_id))
                train_results = []

        if iter_id % VALIDATE_EVERY == 0:
            model.eval()
            val_outputs = []
            val_labels = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_images, val_targets = (
                        val_batch["image"],
                        val_batch["labels"]["target"].float(),
                    )
                    val_out = model(val_images).squeeze()
                    val_outputs.append(val_out)
                    val_labels.append(val_targets)

            val_outputs = torch.cat(val_outputs)
            val_labels = torch.cat(val_labels)

            # Add validation metrics computation to queue
            metric_queue.put(
                (val_outputs.cpu(), val_labels.cpu(), iter_id, "val")
            )

            if accelerator.is_local_main_process:
                val_results = [
                    {
                        "image": img,
                        "output": out,
                        "label": lbl,
                    }
                    for img, out, lbl in zip(
                        val_batch["image"], val_outputs, val_labels
                    )
                ]
                image_queue.put((val_results, "val", iter_id))

            # Save checkpoint
            save_checkpoint(
                CHECKPOINT_DIR,
                model,
                optimizer,
                iter_id // len(train_loader) + 1,
                iter_id,
                {"iter": iter_id},  # We'll update metrics later
                accelerator,
                is_best=False,  # We'll determine this after getting metrics
            )

    accelerator.print(
        "[bold green]Training completed. Creating ensemble...[/bold green]"
    )

    # Stop the threads
    metric_queue.put(None)
    image_queue.put(None)
    metric_thread.join()
    image_thread.join()

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

        # Log test images
        test_results = [
            {
                "image": img,
                "output": out,
                "label": lbl,
            }
            for img, out, lbl in zip(
                test_batch["image"], test_outputs, test_labels
            )
        ]
        log_images_thread(
            [(test_results, "test", NUM_TRAIN_ITER)], accelerator
        )

        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
