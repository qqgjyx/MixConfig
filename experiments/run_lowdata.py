#!/usr/bin/env python
"""
Run low-data sweeps for MixConfig on selected datasets.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_all_seeds, one_hot_configs, get_run_dir, save_json
from src.datasets.molecular import BBBPLoader
from src.datasets.vision import CIFAR100Loader
from src.mixconfig import ConfigExtractor, EnergyAwareSelector
from src.predictors.neural import LitMLPPredictor


def split_train_val(X, y, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def build_features(
    mode,
    X_train,
    X_val,
    X_test,
    configs_train,
    configs_val,
    configs_test,
    selector,
    device,
    energy_stats_t=None,
):
    if mode == "base":
        return X_train, X_val, X_test
    if mode == "config":
        return (
            np.concatenate([X_train, one_hot_configs(configs_train)], axis=1),
            np.concatenate([X_val, one_hot_configs(configs_val)], axis=1),
            np.concatenate([X_test, one_hot_configs(configs_test)], axis=1),
        )

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

    return (
        np.concatenate([X_train, X_train_mixed], axis=1),
        np.concatenate([X_val, X_val_mixed], axis=1),
        np.concatenate([X_test, X_test_mixed], axis=1),
    )


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


def run_lowdata(args):
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

    results = []
    fractions = [float(x) for x in args.fractions.split(",")]

    for frac in fractions:
        frac = max(0.01, min(1.0, frac))
        n_sub = max(2, int(len(X_train) * frac))
        idx = np.random.RandomState(args.seed).choice(len(X_train), size=n_sub, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]

        if args.mode in {"config", "mixconfig"}:
            extractor = ConfigExtractor(n_neighbors=args.n_neighbors, n_configs=args.n_configs)
            configs_train, energy_stats = extractor.extract(X_sub)
            configs_val = extractor.transform(X_val)
            configs_test = extractor.transform(X_test)

            selector = None
            energy_stats_t = None
            if args.mode == "mixconfig":
                selector = EnergyAwareSelector(
                    input_dim=X_sub.shape[1],
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
        else:
            configs_train = configs_val = configs_test = None
            selector = None
            energy_stats_t = None

        X_tr, X_va, X_te = build_features(
            args.mode,
            X_sub,
            X_val,
            X_test,
            configs_train,
            configs_val,
            configs_test,
            selector,
            device,
            energy_stats_t,
        )

        acc = train_eval_mlp(X_tr, y_sub, X_va, y_val, X_te, y_test, config=config, is_classifier=True)
        results.append({"fraction": frac, "test_accuracy": acc})
        print(f"Fraction {frac:.2f}: test_acc={acc:.4f}")

    run_dir = get_run_dir(args.output_dir, args.run_name)
    save_json(str(run_dir / "lowdata_results.json"), {
        "dataset": args.dataset,
        "mode": args.mode,
        "fractions": fractions,
        "results": results,
    })


def main():
    parser = argparse.ArgumentParser(description="Low-data sweeps for MixConfig")
    parser.add_argument("--dataset", type=str, default="bbbp", choices=["bbbp", "cifar100"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, default="mixconfig", choices=["base", "config", "mixconfig"])
    parser.add_argument("--fractions", type=str, default="1.0,0.5,0.2,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--n_configs", type=int, default=8)
    parser.add_argument("--context_dim", type=int, default=64)
    parser.add_argument("--cluster_embed_dim", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run_lowdata(args)


if __name__ == "__main__":
    main()
