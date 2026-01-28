#!/usr/bin/env python
"""
Run MixConfig experiments on vision datasets (CIFAR-100, ImageNet-1K).

Usage:
    python run_vision.py --dataset cifar100 --config configs/datasets/vision.yaml
    python run_vision.py --dataset imagenet1k --data_dir /path/to/imagenet
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_all_seeds, one_hot_configs, get_run_dir, save_json
from src.datasets.vision import CIFAR100Loader, ImageNet1KLoader
from src.mixconfig import ConfigExtractor, EnergyAwareSelector
from src.predictors.neural import LitMLPPredictor
from src.predictors.classical import get_predictor


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(
    dataset_name: str,
    config: dict,
    data_dir: str = "./data",
    predictor_type: str = "mlp",
    mode: str = "mixconfig",
    device: str = "cuda",
):
    """
    Run MixConfig experiment on vision dataset.

    Args:
        dataset_name: Dataset name ('cifar100' or 'imagenet1k').
        config: Configuration dictionary.
        data_dir: Data directory.
        predictor_type: Type of predictor.
        mode: One of "base", "config", "mixconfig".
        device: Computation device.

    Returns:
        Dictionary with results.
    """
    print(f"\nDataset: {dataset_name.upper()}")

    # Load and extract features
    if dataset_name == "cifar100":
        loader = CIFAR100Loader(data_dir=data_dir)
        data = loader.extract_features(device=device)
        n_classes = 100
    elif dataset_name == "imagenet1k":
        loader = ImageNet1KLoader(data_dir=data_dir)
        data = loader.extract_features(device=device, max_samples=50000)
        n_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data.get("X_test", data.get("X_val")), data.get("y_test", data.get("y_val"))

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {n_classes}")

    # Extract configurations
    if mode in {"config", "mixconfig"}:
        mc_config = config.get("mixconfig", {})
        extractor = ConfigExtractor(
            n_neighbors=mc_config.get("n_neighbors", 30),
            n_configs=mc_config.get("n_configs", 12),
        )

        configs_train, energy_stats = extractor.extract(X_train)
        configs_test = extractor.transform(X_test)

        if mode == "mixconfig":
            selector = EnergyAwareSelector(
                input_dim=X_train.shape[1],
                n_configs=mc_config.get("n_configs", 12),
                context_dim=mc_config.get("context_dim", 128),
                cluster_embed_dim=mc_config.get("cluster_embed_dim", 64),
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

            with torch.no_grad():
                X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
                configs_train_t = torch.tensor(configs_train, dtype=torch.long, device=device)
                X_train_mixed = selector(X_train_t, configs_train_t, energy_stats_t).cpu().numpy()

                X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
                configs_test_t = torch.tensor(configs_test, dtype=torch.long, device=device)
                X_test_mixed = selector(X_test_t, configs_test_t, energy_stats_t).cpu().numpy()

            X_train_aug = np.concatenate([X_train, X_train_mixed], axis=1)
            X_test_aug = np.concatenate([X_test, X_test_mixed], axis=1)
        else:
            X_train_aug = np.concatenate([X_train, one_hot_configs(configs_train)], axis=1)
            X_test_aug = np.concatenate([X_test, one_hot_configs(configs_test)], axis=1)
    else:
        X_train_aug, X_test_aug = X_train, X_test

    # Train predictor
    if predictor_type == "mlp":
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_aug),
            torch.LongTensor(y_train)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test_aug),
            torch.LongTensor(y_test)
        )

        train_config = config.get("training", {})
        batch_size = train_config.get("batch_size", 128)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        model = LitMLPPredictor(
            input_dim=X_train_aug.shape[1],
            output_dim=n_classes,
            hidden_dims=config.get("predictor", {}).get("hidden_dims", [512, 256, 128]),
            is_classifier=True,
            learning_rate=train_config.get("learning_rate", 0.0005),
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ]

        trainer = pl.Trainer(
            max_epochs=train_config.get("max_epochs", 50),
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=False,
        )

        # Split train for validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_split, val_split = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_split, batch_size=batch_size)

        trainer.fit(model, train_loader, val_loader)
        results = trainer.test(model, test_loader, verbose=False)
        test_acc = results[0].get("test_acc", 0)

    else:
        predictor = get_predictor(predictor_type, is_classifier=True)
        predictor.fit(X_train_aug, y_train)
        test_acc = predictor.score(X_test_aug, y_test)

    print(f"  Test Accuracy: {test_acc:.4f}")

    return {
        "dataset": dataset_name,
        "test_accuracy": test_acc,
        "mode": mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Run MixConfig on vision datasets")
    parser.add_argument("--config", type=str, default="configs/datasets/vision.yaml")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "imagenet1k"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--predictor", type=str, default="mlp",
                        choices=["mlp", "xgboost", "rf", "linear"])
    parser.add_argument("--mode", type=str, default="mixconfig",
                        choices=["base", "config", "mixconfig"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    set_all_seeds(args.seed)

    config_path = Path(__file__).parent / args.config
    config = load_config(config_path) if config_path.exists() else {}

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = run_experiment(
        dataset_name=args.dataset,
        config=config,
        data_dir=args.data_dir,
        predictor_type=args.predictor,
        mode=args.mode,
        device=device,
    )

    print(f"\nFinal Test Accuracy: {results['test_accuracy']:.4f}")

    run_dir = get_run_dir(args.output_dir, args.run_name)
    save_json(str(run_dir / "vision_results.json"), {
        "mode": args.mode,
        "predictor": args.predictor,
        "dataset": args.dataset,
        "result": results,
    })


if __name__ == "__main__":
    main()
