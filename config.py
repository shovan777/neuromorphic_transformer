import torch
import os
import numpy as np
import random

# configurations
NUM_EPOCHS = 12
NUM_CHANNELS = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
TIME_STEP = 8
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./fmnist_data"
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
