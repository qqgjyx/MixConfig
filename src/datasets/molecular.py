"""
Molecular dataset loaders for MixConfig.

Supports MolHIV, QM9, BBBP, and BACE from OGB and other sources.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MolecularDataset(Dataset):
    """Base dataset class for molecular data with precomputed features."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        configs: Optional[np.ndarray] = None,
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.configs = torch.LongTensor(configs) if configs is not None else None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        if self.configs is not None:
            return self.features[idx], self.labels[idx], self.configs[idx]
        return self.features[idx], self.labels[idx]


def _extract_ogb_fingerprints(
    dataset_name: str,
    data_dir: str,
    fp_type: str = "morgan",
    n_bits: int = 2048,
    radius: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Extract RDKit fingerprints from OGB molecular datasets.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys
        from ogb.graphproppred import PygGraphPropPredDataset
    except ImportError as e:
        raise ImportError("RDKit and OGB are required for fingerprint extraction") from e

    dataset = PygGraphPropPredDataset(name=dataset_name, root=data_dir)
    split_idx = dataset.get_idx_split()

    def get_fingerprint(smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        if fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
        else:
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        return np.array(fp)

    smiles_list = dataset.smiles_list if hasattr(dataset, "smiles_list") else None
    if smiles_list is None:
        raise RuntimeError("SMILES not available in dataset")

    all_fps = np.array([get_fingerprint(s) for s in smiles_list])
    all_labels = dataset.data.y.numpy().squeeze()

    return {
        "X_train": all_fps[split_idx["train"]],
        "X_val": all_fps[split_idx["valid"]],
        "X_test": all_fps[split_idx["test"]],
        "y_train": all_labels[split_idx["train"]],
        "y_val": all_labels[split_idx["valid"]],
        "y_test": all_labels[split_idx["test"]],
        "feature_dim": n_bits,
    }


class MolHIVLoader:
    """
    Loader for OGB MolHIV dataset.

    HIV virus replication inhibition prediction (binary classification).
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        use_fingerprints: bool = True,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_fingerprints = use_fingerprints

    def load(self) -> Dict[str, Any]:
        """
        Load MolHIV dataset.

        Returns:
            Dictionary containing data and loaders.
        """
        try:
            from ogb.graphproppred import PygGraphPropPredDataset
        except ImportError:
            raise ImportError("OGB not installed. Run: pip install ogb")

        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=self.data_dir)
        split_idx = dataset.get_idx_split()

        return {
            "dataset": dataset,
            "train_idx": split_idx["train"],
            "val_idx": split_idx["valid"],
            "test_idx": split_idx["test"],
            "n_classes": 2,
            "task_type": "binary_classification",
            "metric": "rocauc",
            "dataset_name": "ogbg-molhiv",
        }

    def extract_fingerprints(
        self,
        fp_type: str = "morgan",
        n_bits: int = 2048,
        radius: int = 2,
    ) -> Dict[str, np.ndarray]:
        """
        Extract molecular fingerprints for MixConfig.

        Args:
            fp_type: Type of fingerprint ('morgan', 'maccs', 'rdkit').
            n_bits: Number of bits for fingerprint.
            radius: Radius for Morgan fingerprint.

        Returns:
            Dictionary with fingerprint features and labels.
        """
        return _extract_ogb_fingerprints(
            dataset_name="ogbg-molhiv",
            data_dir=self.data_dir,
            fp_type=fp_type,
            n_bits=n_bits,
            radius=radius,
        )


class QM9Loader:
    """
    Loader for QM9 dataset.

    Quantum mechanical properties of small molecules (regression).
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        target_idx: int = 0,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_idx = target_idx

        # QM9 target properties
        self.target_names = [
            "mu", "alpha", "homo", "lumo", "gap", "r2",
            "zpve", "U0", "U", "H", "G", "Cv"
        ]

    def load(self) -> Dict[str, Any]:
        """
        Load QM9 dataset.

        Returns:
            Dictionary containing data and metadata.
        """
        try:
            from torch_geometric.datasets import QM9
        except ImportError:
            raise ImportError("torch_geometric not installed")

        dataset = QM9(root=self.data_dir)

        # Standard split: 110k train, 10k val, 10k test
        n = len(dataset)
        train_size = 110000
        val_size = 10000

        indices = torch.randperm(n)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        return {
            "dataset": dataset,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "target_name": self.target_names[self.target_idx],
            "task_type": "regression",
            "metric": "mae",
            "dataset_name": "QM9",
        }


class BBBPLoader:
    """
    Loader for BBBP (Blood-Brain Barrier Penetration) dataset.

    Binary classification for blood-brain barrier permeability.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def load(self) -> Dict[str, Any]:
        """
        Load BBBP dataset.

        Returns:
            Dictionary containing data and metadata.
        """
        try:
            from ogb.graphproppred import PygGraphPropPredDataset
        except ImportError:
            raise ImportError("OGB not installed")

        dataset = PygGraphPropPredDataset(name="ogbg-molbbbp", root=self.data_dir)
        split_idx = dataset.get_idx_split()

        return {
            "dataset": dataset,
            "train_idx": split_idx["train"],
            "val_idx": split_idx["valid"],
            "test_idx": split_idx["test"],
            "n_classes": 2,
            "task_type": "binary_classification",
            "metric": "rocauc",
            "dataset_name": "BBBP",
        }

    def extract_fingerprints(
        self,
        fp_type: str = "morgan",
        n_bits: int = 2048,
        radius: int = 2,
    ) -> Dict[str, np.ndarray]:
        """
        Extract molecular fingerprints for BBBP.
        """
        return _extract_ogb_fingerprints(
            dataset_name="ogbg-molbbbp",
            data_dir=self.data_dir,
            fp_type=fp_type,
            n_bits=n_bits,
            radius=radius,
        )


class BACELoader:
    """
    Loader for BACE dataset.

    Binary classification for BACE-1 inhibitor activity.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def load(self) -> Dict[str, Any]:
        """
        Load BACE dataset.

        Returns:
            Dictionary containing data and metadata.
        """
        try:
            from ogb.graphproppred import PygGraphPropPredDataset
        except ImportError:
            raise ImportError("OGB not installed")

        dataset = PygGraphPropPredDataset(name="ogbg-molbace", root=self.data_dir)
        split_idx = dataset.get_idx_split()

        return {
            "dataset": dataset,
            "train_idx": split_idx["train"],
            "val_idx": split_idx["valid"],
            "test_idx": split_idx["test"],
            "n_classes": 2,
            "task_type": "binary_classification",
            "metric": "rocauc",
            "dataset_name": "BACE",
        }

    def extract_fingerprints(
        self,
        fp_type: str = "morgan",
        n_bits: int = 2048,
        radius: int = 2,
    ) -> Dict[str, np.ndarray]:
        """
        Extract molecular fingerprints for BACE.
        """
        return _extract_ogb_fingerprints(
            dataset_name="ogbg-molbace",
            data_dir=self.data_dir,
            fp_type=fp_type,
            n_bits=n_bits,
            radius=radius,
        )


def get_molecular_features(
    dataset_name: str,
    data_dir: str = "./data",
    feature_type: str = "fingerprint",
) -> Dict[str, np.ndarray]:
    """
    Convenience function to get molecular features for MixConfig.

    Args:
        dataset_name: One of 'molhiv', 'qm9', 'bbbp', 'bace'.
        data_dir: Data directory.
        feature_type: Type of features to extract.

    Returns:
        Dictionary with features and labels.
    """
    loaders = {
        "molhiv": MolHIVLoader,
        "bbbp": BBBPLoader,
        "bace": BACELoader,
    }

    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = loaders[dataset_name.lower()](data_dir=data_dir)

    if feature_type == "fingerprint" and hasattr(loader, "extract_fingerprints"):
        return loader.extract_fingerprints()

    return loader.load()
