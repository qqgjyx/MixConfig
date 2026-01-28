#!/usr/bin/env python
"""
Run MixConfig experiments on text datasets (SST-2, AG News).

Usage:
    python run_text.py --dataset sst2 --config configs/datasets/text.yaml
    python run_text.py --dataset ag_news --model_name roberta-base
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
from src.datasets.text import SST2Loader, AGNewsLoader, get_text_embeddings
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
    model_name: str = "bert-base-uncased",
    predictor_type: str = "mlp",
    mode: str = "mixconfig",
    device: str = "cuda",
    max_samples: int = None,
):
    """
    Run MixConfig experiment on text dataset.

    Args:
        dataset_name: Dataset name.
        config: Configuration dictionary.
        model_name: Pretrained model for embeddings.
        predictor_type: Type of predictor.
        mode: One of "base", "config", "mixconfig".
        device: Computation device.
        max_samples: Max samples for debugging.

    Returns:
        Dictionary with results.
    """
    print(f"\nDataset: {dataset_name.upper()}")
    print(f"Embedding model: {model_name}")

    # Extract embeddings
    try:
        data = get_text_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            pooling="cls",
            device=device,
            max_samples=max_samples,
        )
    except Exception as e:
        raise RuntimeError(
            f"Embedding extraction failed for {dataset_name}. "
            "Ensure transformers and datasets are installed."
        ) from e

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data.get("X_test", data.get("X_val"))
    y_test = data.get("y_test", data.get("y_val"))

    # Create validation split if not present
    if "X_val" in data:
        X_val, y_val = data["X_val"], data["y_val"]
    else:
        # Split train data
        n_val = len(X_train) // 10
        X_val, y_val = X_train[-n_val:], y_train[-n_val:]
        X_train, y_train = X_train[:-n_val], y_train[:-n_val]

    n_classes = len(np.unique(y_train))

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {n_classes}")

    # Extract configurations
    if mode in {"config", "mixconfig"}:
        mc_config = config.get("mixconfig", {})
        extractor = ConfigExtractor(
            n_neighbors=mc_config.get("n_neighbors", 15),
            n_configs=mc_config.get("n_configs", 8),
        )

        configs_train, energy_stats = extractor.extract(X_train)
        configs_val = extractor.transform(X_val)
        configs_test = extractor.transform(X_test)

        if mode == "mixconfig":
            selector = EnergyAwareSelector(
                input_dim=X_train.shape[1],
                n_configs=mc_config.get("n_configs", 8),
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
            torch.LongTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val_aug),
            torch.LongTensor(y_val)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test_aug),
            torch.LongTensor(y_test)
        )

        train_config = config.get("training", {})
        batch_size = train_config.get("batch_size", 32)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        model = LitMLPPredictor(
            input_dim=X_train_aug.shape[1],
            output_dim=n_classes,
            hidden_dims=config.get("predictor", {}).get("hidden_dims", [384, 192, 96]),
            is_classifier=True,
            learning_rate=train_config.get("learning_rate", 0.0001),
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ]

        trainer = pl.Trainer(
            max_epochs=train_config.get("max_epochs", 30),
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=False,
        )

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
        "model_name": model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Run MixConfig on text datasets")
    parser.add_argument("--config", type=str, default="configs/datasets/text.yaml")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "ag_news"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--predictor", type=str, default="mlp",
                        choices=["mlp", "xgboost", "rf", "linear"])
    parser.add_argument("--mode", type=str, default="mixconfig",
                        choices=["base", "config", "mixconfig"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples for debugging")
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
        model_name=args.model_name,
        predictor_type=args.predictor,
        mode=args.mode,
        device=device,
        max_samples=args.max_samples,
    )

    print(f"\nFinal Test Accuracy: {results['test_accuracy']:.4f}")

    run_dir = get_run_dir(args.output_dir, args.run_name)
    save_json(str(run_dir / "text_results.json"), {
        "mode": args.mode,
        "predictor": args.predictor,
        "dataset": args.dataset,
        "result": results,
    })


if __name__ == "__main__":
    main()
