"""
Utility functions for MixConfig experiments and utilities.
"""
import json
import platform
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, Dataset
import matplotlib.pyplot as plt
import scienceplots


def print_environment_info():
    """Print information about the environment."""
    print("\n=== Environment Information ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Platform: {platform.platform()}")
    print("============================\n")


def get_device_info():
    """Get information about available computing devices.

    Returns
    -------
    torch.device
        The device that will be used for computations (either 'cuda' or 'cpu')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== Device Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Using device: {device}")
    print("========================\n")
    return device


def set_all_seeds(seed=42):
    """
    Set all seeds for reproducibility.
    """
    print("\n=== Setting Random Seeds ===")
    print(f"Seed value: {seed}")
    print("Setting torch CPU seed...")
    torch.manual_seed(seed)
    print("Setting torch CUDA seed...")
    torch.cuda.manual_seed_all(seed)
    print("Setting numpy seed...")
    np.random.seed(seed)
    print("Configuring CUDNN...")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Configuring PL...")
    pl.seed_everything(seed, workers=True)
    print("========================\n")
    return seed


def set_plt_style():
    """
    Set the style for matplotlib.
    """
    plt.style.use('science')
    plt.rcParams.update({
        'pdf.fonttype': 42,            # Use TrueType fonts in PDF 
        'ps.fonttype': 42,             # Use TrueType fonts in PS files
        'font.family': 'sans-serif',
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.figsize': (10, 7),
        'font.size': 13,
        'axes.labelsize': 17,
        'axes.titlesize': 17,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        # 'text.usetex': True,
    })


def train_val_split(train_set, val_ratio=0.2, seed=42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and validation sets.

    Parameters
    ----------
    train_set : Dataset
        The dataset to be split.
    val_ratio : float, optional
        The ratio of the dataset to be used for validation (default is 0.2).
    seed : int, optional
        The seed for random number generation (default is 42).

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training and validation datasets.
    """
    train_set_size = int(len(train_set) * (1 - val_ratio))
    val_set_size = len(train_set) - train_set_size
    seed = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        train_set, [train_set_size, val_set_size], generator=seed
    )
    return train_set, val_set


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists and return its Path.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(path: str, payload: Dict[str, Any]) -> None:
    """
    Save a dictionary as formatted JSON.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def expand_energy_stats(
    energy_stats: Dict[str, np.ndarray],
    n_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Expand config-level energy stats to per-sample tensor.

    Returns a tensor of shape [n_samples, n_configs, 4].
    """
    stats = np.stack(
        [
            energy_stats["entropy"],
            energy_stats["attractive"],
            energy_stats["repulsive"],
            energy_stats["energy_gap"],
        ],
        axis=-1,
    )
    stats_t = torch.tensor(stats, dtype=torch.float32, device=device)
    return stats_t.unsqueeze(0).repeat(n_samples, 1, 1)


def one_hot_configs(
    configs: np.ndarray,
    max_clusters: Optional[int] = None,
) -> np.ndarray:
    """
    One-hot encode configuration assignments and concatenate per config.
    """
    n_samples, n_configs = configs.shape
    outputs = []
    for i in range(n_configs):
        cfg = configs[:, i].astype(int)
        n_clusters = max_clusters or (cfg.max() + 1)
        n_clusters = max(2, int(n_clusters))
        one_hot = np.zeros((n_samples, n_clusters), dtype=np.float32)
        one_hot[np.arange(n_samples), cfg] = 1.0
        outputs.append(one_hot)
    return np.concatenate(outputs, axis=1)


def get_run_dir(base_dir: str, run_name: Optional[str] = None) -> Path:
    """
    Create a run directory under base_dir.
    """
    base = ensure_dir(base_dir)
    if run_name is None:
        run_name = "run"
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

