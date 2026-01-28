"""
Tabular dataset loaders for MixConfig.

Supports OpenML-CC18 benchmark suite and individual OpenML datasets.
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


# OpenML-CC18 benchmark task IDs
OPENML_CC18_TASKS = [
    3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54,
    151, 182, 188, 38, 307, 300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067,
    1068, 1461, 1462, 1464, 1468, 1475, 1478, 1480, 1485, 1486, 1487, 1489,
    1494, 1497, 1501, 1510, 4134, 4534, 6332, 23381, 40499, 40668, 40670,
    40701, 40923, 40927, 40978, 40979, 40981, 40982, 40983, 40984, 40994, 41027
]


class TabularDataset(Dataset):
    """
    PyTorch Dataset wrapper for tabular data.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        configs: Optional[np.ndarray] = None,
    ):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y.dtype in [np.float32, np.float64] else torch.LongTensor(y)
        self.configs = torch.LongTensor(configs) if configs is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.configs is not None:
            return self.X[idx], self.y[idx], self.configs[idx]
        return self.X[idx], self.y[idx]


def load_openml_dataset(
    dataset_id: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Load a dataset from OpenML by ID.

    Args:
        dataset_id: OpenML dataset ID.
        test_size: Fraction for test split.
        val_size: Fraction for validation split (from training data).
        random_state: Random seed.
        normalize: Whether to standardize features.

    Returns:
        Dictionary containing:
        - X_train, X_val, X_test: Feature arrays
        - y_train, y_val, y_test: Target arrays
        - feature_names: List of feature names
        - is_classification: Whether task is classification
        - n_classes: Number of classes (if classification)
    """
    try:
        import openml
    except ImportError:
        raise ImportError("OpenML not installed. Run: pip install openml")

    # Load dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_mask, feature_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    # Convert to numpy
    X = X.values if hasattr(X, 'values') else np.array(X)
    y = y.values if hasattr(y, 'values') else np.array(y)

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)

    # Determine task type
    is_classification = len(np.unique(y)) < 50 and not np.issubdtype(y.dtype, np.floating)

    # Encode labels if classification
    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y)
        n_classes = len(le.classes_)
    else:
        y = y.astype(np.float32)
        n_classes = None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if is_classification else None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_train if is_classification else None
    )

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "is_classification": is_classification,
        "n_classes": n_classes,
        "dataset_name": dataset.name,
    }


class OpenMLCC18Loader:
    """
    Loader for OpenML-CC18 benchmark suite.

    Provides iterator over all CC18 datasets with consistent preprocessing.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        normalize: bool = True,
        batch_size: int = 64,
        task_ids: Optional[List[int]] = None,
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.normalize = normalize
        self.batch_size = batch_size
        self.task_ids = task_ids or OPENML_CC18_TASKS

    def __iter__(self):
        """Iterate over CC18 datasets."""
        for task_id in self.task_ids:
            try:
                data = load_openml_dataset(
                    dataset_id=task_id,
                    test_size=self.test_size,
                    val_size=self.val_size,
                    random_state=self.random_state,
                    normalize=self.normalize,
                )

                # Create data loaders
                train_dataset = TabularDataset(data["X_train"], data["y_train"])
                val_dataset = TabularDataset(data["X_val"], data["y_val"])
                test_dataset = TabularDataset(data["X_test"], data["y_test"])

                data["train_loader"] = DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=True
                )
                data["val_loader"] = DataLoader(
                    val_dataset, batch_size=self.batch_size, shuffle=False
                )
                data["test_loader"] = DataLoader(
                    test_dataset, batch_size=self.batch_size, shuffle=False
                )

                data["task_id"] = task_id

                yield data

            except Exception as e:
                print(f"Failed to load task {task_id}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.task_ids)

    def load_single(self, task_id: int) -> Dict[str, Any]:
        """Load a single dataset by task ID."""
        return load_openml_dataset(
            dataset_id=task_id,
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state,
            normalize=self.normalize,
        )
