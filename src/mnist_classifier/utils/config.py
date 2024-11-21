from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 1,
    "device": "cpu"
}

# Model configuration
MODEL_CONFIG = {
    "dropout_rate": 0.1,
    "fc_features": 10,
    "conv_channels": [8, 16, 32],
    "batch_norm": True,
    "scale_tolerance": 0.3
}

# Dataset configuration
DATASET_CONFIG = {
    "mean": (0.1307,),
    "std": (0.3081,),
    "train_batch_size": 64,
    "test_batch_size": 1000,
    "num_workers": 0,
    "shuffle_train": True,
    "shuffle_test": False
}