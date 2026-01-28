#!/usr/bin/env python
"""
Run MixConfig experiments on tabular datasets (OpenML-CC18).

Usage:
    python run_tabular.py --dataset openml-cc18 --config configs/datasets/tabular.yaml
    python run_tabular.py --task_id 3 --predictor xgboost
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

from src.utils import (
    set_all_seeds,
    one_hot_configs,
    get_run_dir,
    save_json,
)
from src.datasets.tabular import OpenMLCC18Loader, load_openml_dataset
from src.mixconfig import ConfigExtractor, EnergyAwareSelector
from src.predictors.neural import LitMLPPredictor
from src.predictors.classical import get_predictor


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_single_dataset(
    task_id: int,
    config: dict,
    predictor_type: str = "mlp",
    mode: str = "mixconfig",
    device: str = "cpu",
):
    """
    Run MixConfig experiment on a single dataset.

    Args:
        task_id: OpenML task ID.
        config: Configuration dictionary.
        predictor_type: Type of predictor to use.
        mode: One of "base", "config", "mixconfig".

    Returns:
        Dictionary with results.
    """
    # Load dataset
    data = load_openml_dataset(
        dataset_id=task_id,
        test_size=config.get("dataset", {}).get("test_size", 0.2),
        val_size=config.get("dataset", {}).get("val_size", 0.1),
        normalize=config.get("dataset", {}).get("normalize", True),
    )

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    is_classifier = data["is_classification"]

    print(f"\nDataset: {data['dataset_name']} (Task {task_id})")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Task: {'Classification' if is_classifier else 'Regression'}")

    # Extract configurations if needed
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
                max_clusters=mc_config.get("max_clusters", 256),
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
                X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
                X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
                X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
                configs_train_t = torch.tensor(configs_train, dtype=torch.long, device=device)
                configs_val_t = torch.tensor(configs_val, dtype=torch.long, device=device)
                configs_test_t = torch.tensor(configs_test, dtype=torch.long, device=device)

                X_train_mixed = selector(X_train_t, configs_train_t, energy_stats_t).cpu().numpy()
                X_val_mixed = selector(X_val_t, configs_val_t, energy_stats_t).cpu().numpy()
                X_test_mixed = selector(X_test_t, configs_test_t, energy_stats_t).cpu().numpy()

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
        # Use PyTorch Lightning
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
            output_dim=data["n_classes"] if is_classifier else 1,
            hidden_dims=config.get("predictor", {}).get("hidden_dims", [256, 128, 64]),
            is_classifier=is_classifier,
            learning_rate=train_config.get("learning_rate", 0.001),
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=train_config.get("early_stopping_patience", 10),
                mode="min",
            ),
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
        # Use classical predictor
        predictor = get_predictor(predictor_type, is_classifier=is_classifier)
        predictor.fit(X_train_aug, y_train, normalize=False)
        test_metric = predictor.score(X_test_aug, y_test, normalize=False)

    metric_name = "Accuracy" if is_classifier else "R2"
    print(f"  Test {metric_name}: {test_metric:.4f}")

    return {
        "task_id": task_id,
        "dataset_name": data["dataset_name"],
        "is_classification": is_classifier,
        "test_metric": test_metric,
        "metric_name": metric_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Run MixConfig on tabular datasets")
    parser.add_argument("--config", type=str, default="configs/datasets/tabular.yaml",
                        help="Path to config file")
    parser.add_argument("--dataset", type=str, default="openml-cc18",
                        help="Dataset to use")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Specific OpenML task ID (overrides dataset)")
    parser.add_argument("--predictor", type=str, default="mlp",
                        choices=["mlp", "xgboost", "rf", "linear"],
                        help="Predictor type")
    parser.add_argument("--mode", type=str, default="mixconfig",
                        choices=["base", "config", "mixconfig"],
                        help="Representation mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run name for output directory")
    parser.add_argument("--max_tasks", type=int, default=5,
                        help="Max tasks to run for full benchmark")

    args = parser.parse_args()

    # Set seeds
    set_all_seeds(args.seed)

    # Load config
    config_path = Path(__file__).parent / args.config
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run experiment
    if args.task_id is not None:
        # Single dataset
        results = run_single_dataset(
            task_id=args.task_id,
            config=config,
            predictor_type=args.predictor,
            mode=args.mode,
            device=device,
        )
        print(f"\nFinal Result: {results['metric_name']} = {results['test_metric']:.4f}")
        run_dir = get_run_dir(args.output_dir, args.run_name)
        save_json(str(run_dir / "tabular_result_single.json"), {
            "mode": args.mode,
            "predictor": args.predictor,
            "task_id": args.task_id,
            "result": results,
        })

    else:
        # Full benchmark
        all_results = []
        loader = OpenMLCC18Loader()

        max_tasks = args.max_tasks if args.max_tasks and args.max_tasks > 0 else len(loader.task_ids)
        for i, task_id in enumerate(loader.task_ids[:max_tasks]):
            try:
                result = run_single_dataset(
                    task_id=task_id,
                    config=config,
                    predictor_type=args.predictor,
                    mode=args.mode,
                    device=device,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  Failed: {e}")

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for r in all_results:
            print(f"{r['dataset_name']}: {r['metric_name']} = {r['test_metric']:.4f}")

        avg_metric = np.mean([r["test_metric"] for r in all_results]) if all_results else 0.0
        print(f"\nAverage: {avg_metric:.4f}")

        run_dir = get_run_dir(args.output_dir, args.run_name)
        save_json(str(run_dir / "tabular_results.json"), {
            "mode": args.mode,
            "predictor": args.predictor,
            "results": all_results,
            "avg_metric": avg_metric,
        })


if __name__ == "__main__":
    main()
