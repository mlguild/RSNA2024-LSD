import os
import torch
import timm
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.nn import SyncBatchNorm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import (
    ISIC2024Dataset,
    SplitNames,
    Importance,
    AveragingEnsemble,
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


def predict(
    model: torch.nn.Module, dataloader: DataLoader, accelerator: Accelerator
) -> Tuple[List[str], np.ndarray]:
    model.eval()
    all_predictions = []
    all_isic_ids = []

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Predicting",
            disable=not accelerator.is_local_main_process,
        ):
            images = batch["image"]
            isic_ids = batch["labels"]["isic_id"]
            outputs = model(images).squeeze()
            probabilities = torch.sigmoid(outputs)
            all_predictions.extend(probabilities.cpu().numpy())
            all_isic_ids.extend(isic_ids)

    return all_isic_ids, np.array(all_predictions)


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
    OUTPUT_FILE = "submission.csv"

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
    dataset = ISIC2024Dataset(
        root_dir=ROOT_DIR,
        split_name=SplitNames.TEST,  # Using the test set for predictions
        transform=data_cfg["test_transform"],
        importance_level_labels=Importance.HIGH,
        return_samples_as_dict=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Prepare model and data for accelerator
    ensemble, dataloader = accelerator.prepare(ensemble, dataloader)

    # Make predictions
    isic_ids, predictions = predict(ensemble, dataloader, accelerator)

    # Create submission dataframe
    submission_df = pd.DataFrame({"isic_id": isic_ids, "target": predictions})

    # Save submission file
    submission_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission file saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
