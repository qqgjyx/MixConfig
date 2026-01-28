#!/usr/bin/env python
"""
Run ablation studies for MixConfig components.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_all_seeds, get_run_dir, save_json
from src.datasets.molecular import BBBPLoader
from src.datasets.vision import CIFAR100Loader
from src.mixconfig import ConfigExtractor, EnergyAwareSelector
from src.predictors.neural import LitMLPPredictor


class ZeroContext(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.output_dim, device=x.device)


class ZeroClusterEmbedder(nn.Module):
    def __init__(self, n_configs: int, embed_dim: int):
        super().__init__()
        self.n_configs = n_configs
        self.embed_dim = embed_dim

    def forward(self, cluster_assignments: torch.Tensor) -> torch.Tensor:
        batch_size = cluster_assignments.shape[0]
        return torch.zeros(batch_size, self.n_configs, self.embed_dim, device=cluster_assignments.device)


def split_train_val(X, y, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train_eval_mlp(X_train, y_train, X_val, y_val, X_test, y_test, config, is_classifier=True):
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long if is_classifier else torch.float32),
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long if is_classifier else torch.float32),
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long if is_classifier else torch.float32),
    )

    train_config = config.get("training", {})
    batch_size = train_config.get("batch_size", 64)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model = LitMLPPredictor(
        input_dim=X_train.shape[1],
        output_dim=len(np.unique(y_train)) if is_classifier else 1,
        hidden_dims=config.get("predictor", {}).get("hidden_dims", [256, 128, 64]),
        is_classifier=is_classifier,
        learning_rate=train_config.get("learning_rate", 0.001),
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=train_config.get("early_stopping_patience", 10), mode="min"),
    ]

    trainer = pl.Trainer(
        max_epochs=train_config.get("max_epochs", 100),
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=False,
    )

    trainer.fit(model, train_loader, val_loader)
    results = trainer.test(model, test_loader, verbose=False)
    return results[0].get("test_acc", 0.0)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_ablation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_all_seeds(args.seed)
    config_path = Path(__file__).parent / args.config if args.config else None
    config = load_config(str(config_path)) if config_path and config_path.exists() else {"training": {}, "predictor": {}}

    if args.dataset == "bbbp":
        loader = BBBPLoader(data_dir=args.data_dir)
        data = loader.extract_fingerprints()
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
    else:
        loader = CIFAR100Loader(data_dir=args.data_dir)
        data = loader.extract_features(device=device)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        X_train, y_train, X_val, y_val = split_train_val(X_train, y_train, val_ratio=0.1, seed=args.seed)

    extractor = ConfigExtractor(n_neighbors=args.n_neighbors, n_configs=args.n_configs)
    configs_train, energy_stats = extractor.extract(X_train)
    configs_val = extractor.transform(X_val)
    configs_test = extractor.transform(X_test)

    selector = EnergyAwareSelector(
        input_dim=X_train.shape[1],
        n_configs=args.n_configs,
        context_dim=args.context_dim,
        cluster_embed_dim=args.cluster_embed_dim,
    ).to(device)

    energy_stats_t = torch.tensor(
        np.stack(
            [
                energy_stats["entropy"],
                energy_stats["attractive"],
                energy_stats["repulsive"],
                energy_stats["energy_gap"],
            ],
            axis=-1,
        ),
        dtype=torch.float32,
        device=device,
    )
    if args.ablation == "no_energy":
        energy_stats_t = torch.zeros(args.n_configs, 4, device=device)
    if args.ablation == "no_context":
        selector.context_encoder = ZeroContext(selector.context_dim)
    if args.ablation == "no_cluster":
        selector.cluster_embedder = ZeroClusterEmbedder(selector.n_configs, selector.cluster_embed_dim)

    with torch.no_grad():
        X_train_mixed = selector(
            torch.tensor(X_train, dtype=torch.float32, device=device),
            torch.tensor(configs_train, dtype=torch.long, device=device),
            energy_stats_t,
        ).cpu().numpy()
        X_val_mixed = selector(
            torch.tensor(X_val, dtype=torch.float32, device=device),
            torch.tensor(configs_val, dtype=torch.long, device=device),
            energy_stats_t,
        ).cpu().numpy()
        X_test_mixed = selector(
            torch.tensor(X_test, dtype=torch.float32, device=device),
            torch.tensor(configs_test, dtype=torch.long, device=device),
            energy_stats_t,
        ).cpu().numpy()

    X_train_aug = np.concatenate([X_train, X_train_mixed], axis=1)
    X_val_aug = np.concatenate([X_val, X_val_mixed], axis=1)
    X_test_aug = np.concatenate([X_test, X_test_mixed], axis=1)

    acc = train_eval_mlp(X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test, config=config, is_classifier=True)
    print(f"Ablation {args.ablation}: test_acc={acc:.4f}")

    run_dir = get_run_dir(args.output_dir, args.run_name)
    save_json(str(run_dir / "ablation_results.json"), {
        "dataset": args.dataset,
        "ablation": args.ablation,
        "test_accuracy": acc,
    })


def main():
    parser = argparse.ArgumentParser(description="Ablation studies for MixConfig")
    parser.add_argument("--dataset", type=str, default="bbbp", choices=["bbbp", "cifar100"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_energy", "no_context", "no_cluster"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--n_configs", type=int, default=8)
    parser.add_argument("--context_dim", type=int, default=64)
    parser.add_argument("--cluster_embed_dim", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run_ablation(args)


if __name__ == "__main__":
    main()
