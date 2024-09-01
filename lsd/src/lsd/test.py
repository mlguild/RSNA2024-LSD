import os
import torch
import timm
import numpy as np
from typing import List, Tuple
from torch.nn import SyncBatchNorm
from accelerate import Accelerator
from torch.utils.data import DataLoader

from train import (
    ISIC2024Dataset,
    SplitNames,
    Importance,
    AveragingEnsemble,
    create_dataloaders,
    evaluate,
    compute_metrics,
)


def load_best_models(
    checkpoint_dir: str, top_k: int = 3
) -> List[Tuple[float, str]]:
    model_scores = []
    for subdir in os.listdir(checkpoint_dir):
        if subdir.startswith("iter_"):
            metrics_file = os.path.join(checkpoint_dir, subdir, "metrics.txt")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    for line in f:
                        if line.startswith("pauc_above_80_tpr:"):
                            score = float(line.split(":")[1].strip())
                            model_scores.append((score, subdir))
                            break

    return sorted(model_scores, reverse=True)[:top_k]


def main():
    # Configuration
    CHECKPOINT_DIR = "checkpoints"
    TOP_K_MODELS = 3
    EVAL_BATCH_SIZE = 512
    NUM_WORKERS = 16
    ROOT_DIR = "/mnt/nvme-fast0/datasets/"
    MODEL_NAME = "convnextv2_base"
    NUM_CLASSES = 1
    IMAGE_SIZE = 224
    MIXED_PRECISION = "bf16"

    accelerator = Accelerator(mixed_precision=MIXED_PRECISION)

    # Load best models
    best_models = load_best_models(CHECKPOINT_DIR, TOP_K_MODELS)
    print(f"Best models: {best_models}")

    # Create ensemble
    ensemble_models = []
    for _, iter_subdir in best_models:
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            iter_subdir,
            f"checkpoint_{iter_subdir.split('_')[1]}.pth",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = timm.create_model(
            MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        ensemble_models.append(model)

    ensemble = AveragingEnsemble(ensemble_models)

    # Prepare data
    data_cfg = timm.data.resolve_data_config(ensemble_models[0].pretrained_cfg)
    _, _, test_loader = create_dataloaders(
        EVAL_BATCH_SIZE,
        EVAL_BATCH_SIZE,
        NUM_WORKERS,
        ROOT_DIR,
        IMAGE_SIZE,
        42,
        data_cfg,
    )

    # Prepare model and data for accelerator
    ensemble, test_loader = accelerator.prepare(ensemble, test_loader)

    # Evaluate ensemble
    criterion = torch.nn.BCEWithLogitsLoss()
    test_metrics, test_results = evaluate(
        ensemble, test_loader, criterion, accelerator
    )

    accelerator.print(f"Ensemble Test Results: {test_metrics}")


if __name__ == "__main__":
    main()
