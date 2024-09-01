import os
import torch
import timm
import numpy as np
from typing import List, Tuple
from torch.nn import SyncBatchNorm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

from train import (
    ISIC2024Dataset,
    SplitNames,
    Importance,
    AveragingEnsemble,
    create_dataloaders,
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


def evaluate_per_class(model, dataloader, criterion, accelerator):
    model.eval()
    all_outputs = []
    all_labels = []
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Evaluating",
            disable=not accelerator.is_local_main_process,
        ):
            images, labels = batch["image"], batch["labels"]["target"].float()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            all_outputs.append(outputs)
            all_labels.append(labels)
            all_losses.append(loss)

    all_outputs = torch.cat(all_outputs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    avg_loss = np.mean([loss.cpu().item() for loss in all_losses])

    # Compute overall metrics
    overall_metrics = compute_metrics(
        torch.from_numpy(all_outputs), torch.from_numpy(all_labels)
    )
    overall_metrics["loss"] = avg_loss

    # Compute per-class metrics
    predictions = (all_outputs > 0.5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, predictions
    )

    per_class_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, predictions)

    return overall_metrics, per_class_metrics, cm


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
    overall_metrics, per_class_metrics, confusion_matrix = evaluate_per_class(
        ensemble, test_loader, criterion, accelerator
    )

    accelerator.print(f"Overall Test Results: {overall_metrics}")

    # Print per-class metrics
    class_names = ["Benign", "Malignant"]
    for i, class_name in enumerate(class_names):
        accelerator.print(f"\nMetrics for class {class_name}:")
        accelerator.print(
            f"Precision: {per_class_metrics['precision'][i]:.4f}"
        )
        accelerator.print(f"Recall: {per_class_metrics['recall'][i]:.4f}")
        accelerator.print(f"F1-score: {per_class_metrics['f1'][i]:.4f}")
        accelerator.print(f"Support: {per_class_metrics['support'][i]}")

    # Print confusion matrix
    accelerator.print("\nConfusion Matrix:")
    accelerator.print(confusion_matrix)

    # Calculate and print additional metrics to identify failure modes
    accelerator.print("\nFailure Mode Analysis:")
    false_positives = confusion_matrix[0, 1]
    false_negatives = confusion_matrix[1, 0]
    total_samples = np.sum(confusion_matrix)

    accelerator.print(
        f"False Positive Rate: {false_positives / total_samples:.4f}"
    )
    accelerator.print(
        f"False Negative Rate: {false_negatives / total_samples:.4f}"
    )
    accelerator.print(
        f"Misclassification Rate: {(false_positives + false_negatives) / total_samples:.4f}"
    )


if __name__ == "__main__":
    main()
