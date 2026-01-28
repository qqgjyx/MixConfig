"""
Text dataset loaders for MixConfig.

Supports SST-2 and AG News using Hugging Face transformers.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Dataset class for text data with precomputed embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        configs: Optional[np.ndarray] = None,
    ):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        self.configs = torch.LongTensor(configs) if configs is not None else None

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        if self.configs is not None:
            return self.embeddings[idx], self.labels[idx], self.configs[idx]
        return self.embeddings[idx], self.labels[idx]


class SST2Loader:
    """
    Loader for SST-2 (Stanford Sentiment Treebank) dataset.

    Binary sentiment classification.
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_length: int = 128,
        model_name: str = "bert-base-uncased",
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name

    def load(self) -> Dict[str, Any]:
        """
        Load SST-2 dataset from Hugging Face.

        Returns:
            Dictionary containing dataset info.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets not installed. Run: pip install datasets")

        dataset = load_dataset("glue", "sst2")

        return {
            "train_data": dataset["train"],
            "val_data": dataset["validation"],
            "test_data": dataset.get("test"),  # May not have labels
            "n_classes": 2,
            "task_type": "binary_classification",
            "metric": "accuracy",
            "dataset_name": "SST-2",
        }

    def extract_embeddings(
        self,
        pooling: str = "cls",
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract text embeddings using pretrained transformer.

        Args:
            pooling: Pooling strategy ('cls', 'mean').
            device: Computation device.
            max_samples: Maximum samples (for debugging).

        Returns:
            Dictionary with embeddings and labels.
        """
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("transformers/datasets not installed")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model = model.to(device)
        model.eval()

        dataset = load_dataset("glue", "sst2")

        def extract(split_data, max_n=None):
            texts = split_data["sentence"]
            labels = split_data["label"]

            if max_n:
                texts = texts[:max_n]
                labels = labels[:max_n]

            embeddings = []

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                    if pooling == "cls":
                        # Use [CLS] token embedding
                        batch_emb = outputs.last_hidden_state[:, 0, :]
                    else:
                        # Mean pooling
                        attention_mask = inputs["attention_mask"]
                        token_emb = outputs.last_hidden_state
                        mask = attention_mask.unsqueeze(-1).expand(token_emb.size())
                        batch_emb = (token_emb * mask).sum(1) / mask.sum(1)

                    embeddings.append(batch_emb.cpu().numpy())

            return np.vstack(embeddings), np.array(labels)

        X_train, y_train = extract(dataset["train"], max_samples)
        X_val, y_val = extract(dataset["validation"], max_samples)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "feature_dim": X_train.shape[1],
        }


class AGNewsLoader:
    """
    Loader for AG News dataset.

    4-class news topic classification.
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_length: int = 256,
        model_name: str = "bert-base-uncased",
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name

    def load(self) -> Dict[str, Any]:
        """
        Load AG News dataset.

        Returns:
            Dictionary containing dataset info.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets not installed")

        dataset = load_dataset("ag_news")

        return {
            "train_data": dataset["train"],
            "test_data": dataset["test"],
            "n_classes": 4,
            "class_names": ["World", "Sports", "Business", "Sci/Tech"],
            "task_type": "multiclass_classification",
            "metric": "accuracy",
            "dataset_name": "AG News",
        }

    def extract_embeddings(
        self,
        pooling: str = "cls",
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract text embeddings using pretrained transformer.

        Args:
            pooling: Pooling strategy ('cls', 'mean').
            device: Computation device.
            max_samples: Maximum samples (for debugging).

        Returns:
            Dictionary with embeddings and labels.
        """
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("transformers/datasets not installed")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model = model.to(device)
        model.eval()

        dataset = load_dataset("ag_news")

        def extract(split_data, max_n=None):
            texts = split_data["text"]
            labels = split_data["label"]

            if max_n:
                texts = texts[:max_n]
                labels = labels[:max_n]

            embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                    if pooling == "cls":
                        batch_emb = outputs.last_hidden_state[:, 0, :]
                    else:
                        attention_mask = inputs["attention_mask"]
                        token_emb = outputs.last_hidden_state
                        mask = attention_mask.unsqueeze(-1).expand(token_emb.size())
                        batch_emb = (token_emb * mask).sum(1) / mask.sum(1)

                    embeddings.append(batch_emb.cpu().numpy())

            return np.vstack(embeddings), np.array(labels)

        X_train, y_train = extract(dataset["train"], max_samples)
        X_test, y_test = extract(dataset["test"], max_samples)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_dim": X_train.shape[1],
        }


def get_text_embeddings(
    dataset_name: str,
    model_name: str = "bert-base-uncased",
    pooling: str = "cls",
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to get text embeddings for MixConfig.

    Args:
        dataset_name: One of 'sst2', 'ag_news'.
        model_name: Pretrained model name.
        pooling: Pooling strategy.
        device: Computation device.
        max_samples: Maximum samples.

    Returns:
        Dictionary with embeddings and labels.
    """
    loaders = {
        "sst2": SST2Loader,
        "ag_news": AGNewsLoader,
    }

    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = loaders[dataset_name.lower()](model_name=model_name)
    return loader.extract_embeddings(
        pooling=pooling, device=device, max_samples=max_samples
    )
