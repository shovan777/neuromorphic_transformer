import torch
import os
import numpy as np
import random

# configurations
NUM_EPOCHS = 200
INPUT_CHANNELS = 3
IMG_SIZE = 32
PATCH_SIZE = 4
DIM = 128# 512
NUM_HEADS = 4
MLP_DIM = 256  # mlp_ratio=2 (2 * DIM)
NUM_LAYERS = 4
# Keep this alias for compatibility with existing scripts; it now tracks embed_dim.
NUM_CHANNELS = 64#DIM // 4 #64
LEARNING_RATE = 8e-4
WARMUP_LR = 1e-6
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 0.05
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
USE_POISSON_ENCODING = False
# Debug mode to isolate convergence failures.
DEBUG_MODE = False
DEBUG_EPOCHS = 10
DEBUG_OVERFIT_STEPS = 200
DEBUG_OVERFIT_BATCH_SIZE = 128
RESUME_PATH = "/home/srs476/vit_exp/neuromorphic_transformer/models/checkpoint_20260426-15h/final_spikformer_dim128_heads4_mlp256_layers4_epoch200_acc0.3983.pth"  # Set to a checkpoint path to resume (e.g. "models/checkpoint_20260427-01h/best_...pth")
RESUME_EPOCH = 0   # Epoch to resume from (scheduler will be fast-forwarded to this epoch)
BATCH_SIZE = 256#128 #256
TIME_STEP = 4
NUM_WORKERS = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
SEED = 42

MODEL_CONFIG = {
    "embed_dim": DIM,
    "num_heads": NUM_HEADS,
    "mlp_dim": MLP_DIM,
    "num_layers": NUM_LAYERS,
    "patch_size": PATCH_SIZE,
}

# Previous train.py defaults (kept for quick rollback/reference):
# embed_dim=64, num_heads=4, mlp_dim=128, num_layers=2, patch_size=4

# Reference CIFAR10 recipe from official Spikformer cifar10.yml:
# https://github.com/ZK-Zhou/spikformer/blob/main/cifar10/cifar10.yml
SOTA_CIFAR10_REFERENCE = {
    "epochs": 300,
    "time_step": 4,
    "embed_dim": 384,
    "num_heads": 12,
    "patch_size": 4,
    "mlp_dim": 1536,  # mlp_ratio=4
    "num_layers": 4,
    "batch_size": 128,
    "learning_rate": 5e-4,
    "warmup_lr": 1e-5,
    "warmup_epochs": 20,
    "weight_decay": 6e-2,
    "mixup_alpha": 0.5,
    "cutmix_alpha": 0.0,
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
