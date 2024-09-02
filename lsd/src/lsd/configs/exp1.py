# experiments/experiment1/config.py

TRAIN_MICRO_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 512
NUM_WORKERS = 16
LEARNING_RATE = 6e-6
NUM_TRAIN_ITER = 10000
VALIDATE_EVERY = 100
NUM_CLASSES = 1
TOP_K_MODELS = 3
SEED = 42
ROOT_DIR = "/mnt/nvme-fast0/datasets/"
MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
MIXED_PRECISION = "bf16"
PROJECT_NAME = "isic2024-training"
IMAGE_SIZE = 224
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.5
LOG_EVERY = 100


def get_experiment_name():
    return f"{MODEL_NAME}_{LEARNING_RATE}_{WEIGHT_DECAY}_{DROPOUT_RATE}_{SEED}"


def get_checkpoint_dir():
    return (
        f"/mnt/nvme-fast0/experiments/{PROJECT_NAME}/{get_experiment_name()}"
    )
