"""
Vision dataset loaders for MixConfig.

Supports CIFAR-100 and ImageNet-1K.
"""

from typing import Tuple, Optional, Dict, Any, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CIFAR100Loader:
    """
    Loader for CIFAR-100 dataset.

    Extracts features using pretrained models for MixConfig experiments.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        feature_extractor: Optional[str] = "resnet50",
        normalize: bool = True,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_extractor = feature_extractor
        self.normalize = normalize

    def load(self) -> Dict[str, Any]:
        """
        Load CIFAR-100 with optional feature extraction.

        Returns:
            Dictionary containing data loaders and metadata.
        """
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("torchvision not installed. Run: pip install torchvision")

        # Standard CIFAR-100 transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
        ])

        # Load datasets
        train_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )

        test_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "n_classes": 100,
            "input_shape": (3, 32, 32),
            "dataset_name": "CIFAR-100",
        }

    def extract_features(
        self,
        model_name: str = "resnet50",
        device: str = "cuda",
    ) -> Dict[str, np.ndarray]:
        """
        Extract features using a pretrained model.

        Args:
            model_name: Name of pretrained model.
            device: Device for computation.

        Returns:
            Dictionary with extracted features and labels.
        """
        try:
            from torchvision import models, transforms, datasets
        except ImportError:
            raise ImportError("torchvision not installed")

        # Load pretrained model
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(device)
        model.eval()

        # Transform for feature extraction
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        train_dataset = datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=transform
        )

        def extract(dataset):
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            features = []
            labels = []

            with torch.no_grad():
                for images, targets in loader:
                    images = images.to(device)
                    feats = model(images).squeeze()
                    features.append(feats.cpu().numpy())
                    labels.append(targets.numpy())

            return np.vstack(features), np.concatenate(labels)

        X_train, y_train = extract(train_dataset)
        X_test, y_test = extract(test_dataset)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_dim": feature_dim,
        }


class ImageNet1KLoader:
    """
    Loader for ImageNet-1K dataset.

    Requires local ImageNet data directory.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 224,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def load(self) -> Dict[str, Any]:
        """
        Load ImageNet-1K dataset.

        Returns:
            Dictionary containing data loaders and metadata.
        """
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("torchvision not installed")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(
            f"{self.data_dir}/train",
            transform=transform_train,
        )

        val_dataset = datasets.ImageFolder(
            f"{self.data_dir}/val",
            transform=transform_val,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "n_classes": 1000,
            "input_shape": (3, self.image_size, self.image_size),
            "dataset_name": "ImageNet-1K",
        }

    def extract_features(
        self,
        model_name: str = "resnet50",
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from ImageNet using pretrained model.

        Args:
            model_name: Pretrained model name.
            device: Computation device.
            max_samples: Maximum samples to extract (for debugging).

        Returns:
            Dictionary with features and labels.
        """
        try:
            from torchvision import models, datasets, transforms
        except ImportError:
            raise ImportError("torchvision not installed")

        # Load model
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(device)
        model.eval()

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(
            f"{self.data_dir}/train", transform=transform
        )
        val_dataset = datasets.ImageFolder(
            f"{self.data_dir}/val", transform=transform
        )

        def extract(dataset, max_n=None):
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            features = []
            labels = []
            count = 0

            with torch.no_grad():
                for images, targets in loader:
                    if max_n and count >= max_n:
                        break
                    images = images.to(device)
                    feats = model(images).squeeze()
                    features.append(feats.cpu().numpy())
                    labels.append(targets.numpy())
                    count += len(targets)

            return np.vstack(features), np.concatenate(labels)

        X_train, y_train = extract(train_dataset, max_samples)
        X_val, y_val = extract(val_dataset, max_samples)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "feature_dim": feature_dim,
        }
