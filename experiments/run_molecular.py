#!/usr/bin/env python
"""
Run MixConfig experiments on molecular datasets (MolHIV, BBBP, BACE).

Usage:
    python run_molecular.py --dataset molhiv --config configs/datasets/molecular.yaml
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
from src.datasets.molecular import MolHIVLoader, BBBPLoader, BACELoader
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
    device: str = "cpu",
):
    """
    Run MixConfig experiment on molecular dataset.

    Args:
        dataset_name: Dataset name.
        config: Configuration dictionary.
        data_dir: Data directory.
        predictor_type: Type of predictor.
        mode: One of "base", "config", "mixconfig".

    Returns:
        Dictionary with results.
    """
    print(f"\nDataset: {dataset_name.upper()}")

    # Load dataset and extract features
    loaders = {
        "molhiv": MolHIVLoader,
        "bbbp": BBBPLoader,
        "bace": BACELoader,
    }

    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = loaders[dataset_name.lower()](data_dir=data_dir)

    try:
        data = loader.extract_fingerprints()
    except Exception as e:
        raise RuntimeError(
            f"Fingerprint extraction failed for {dataset_name}. "
            "Ensure RDKit and OGB are installed."
        ) from e

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    is_classifier = dataset_name.lower() != "qm9"

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Task: {'Classification' if is_classifier else 'Regression'}")

    # Extract configurations
    if mode in {"config", "mixconfig"}:
        mc_config = config.get("mixconfig", {})
        extractor = ConfigExtractor(
            n_neighbors=mc_config.get("n_neighbors", 20),
            n_configs=mc_config.get("n_configs", 10),
        )

        configs_train, energy_stats = extractor.extract(X_train)
        configs_val = extractor.transform(X_val)
        configs_test = extractor.transform(X_test)

        if mode == "mixconfig":
            selector = EnergyAwareSelector(
                input_dim=X_train.shape[1],
                n_configs=mc_config.get("n_configs", 10),
                context_dim=mc_config.get("context_dim", 64),
                cluster_embed_dim=mc_config.get("cluster_embed_dim", 32),
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
        else:
            X_train_aug = np.concatenate([X_train, one_hot_configs(configs_train)], axis=1)
            X_val_aug = np.concatenate([X_val, one_hot_configs(configs_val)], axis=1)
            X_test_aug = np.concatenate([X_test, one_hot_configs(configs_test)], axis=1)
    else:
        X_train_aug, X_val_aug, X_test_aug = X_train, X_val, X_test

    # Train predictor
    if predictor_type == "mlp":
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_aug),
            torch.LongTensor(y_train) if is_classifier else torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val_aug),
            torch.LongTensor(y_val) if is_classifier else torch.FloatTensor(y_val)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test_aug),
            torch.LongTensor(y_test) if is_classifier else torch.FloatTensor(y_test)
        )

        train_config = config.get("training", {})
        batch_size = train_config.get("batch_size", 64)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        model = LitMLPPredictor(
            input_dim=X_train_aug.shape[1],
            output_dim=2 if is_classifier else 1,
            hidden_dims=config.get("predictor", {}).get("hidden_dims", [256, 128, 64]),
            is_classifier=is_classifier,
            learning_rate=train_config.get("learning_rate", 0.001),
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ]

        trainer = pl.Trainer(
            max_epochs=train_config.get("max_epochs", 100),
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=False,
        )

        trainer.fit(model, train_loader, val_loader)
        results = trainer.test(model, test_loader, verbose=False)
        test_metric = results[0].get("test_acc" if is_classifier else "test_r2", 0)

    else:
        predictor = get_predictor(predictor_type, is_classifier=is_classifier)
        predictor.fit(X_train_aug, y_train)
        test_metric = predictor.score(X_test_aug, y_test)

    metric_name = "Accuracy" if is_classifier else "R2"
    print(f"  Test {metric_name}: {test_metric:.4f}")

    return {
        "dataset": dataset_name,
        "test_metric": test_metric,
        "metric_name": metric_name,
        "mode": mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Run MixConfig on molecular datasets")
    parser.add_argument("--config", type=str, default="configs/datasets/molecular.yaml")
    parser.add_argument("--dataset", type=str, default="molhiv",
                        choices=["molhiv", "bbbp", "bace"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--predictor", type=str, default="mlp",
                        choices=["mlp", "xgboost", "rf", "linear"])
    parser.add_argument("--mode", type=str, default="mixconfig",
                        choices=["base", "config", "mixconfig"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    set_all_seeds(args.seed)

    config_path = Path(__file__).parent / args.config
    config = load_config(config_path) if config_path.exists() else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = run_experiment(
        dataset_name=args.dataset,
        config=config,
        data_dir=args.data_dir,
        predictor_type=args.predictor,
        mode=args.mode,
        device=device,
    )

    print(f"\nFinal {results['metric_name']}: {results['test_metric']:.4f}")

    run_dir = get_run_dir(args.output_dir, args.run_name)
    save_json(str(run_dir / "molecular_results.json"), {
        "mode": args.mode,
        "predictor": args.predictor,
        "dataset": args.dataset,
        "result": results,
    })


if __name__ == "__main__":
    main()
