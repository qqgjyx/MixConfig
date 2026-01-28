"""
Configuration Extractor for MixConfig.

This module provides the public interface for configuration extraction.
The Parallel-DT extraction algorithm (Pitsianis et al.) is not included here.
A Python implementation will be released later.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np


class ConfigExtractor:
    """
    Configuration extraction interface (Parallel-DT backend not included).

    Expected API:
        extractor = ConfigExtractor(n_neighbors=15, n_configs=8)
        configs, energy_stats = extractor.extract(X)

    Args:
        n_neighbors: Number of nearest neighbors for k-NN graph.
        n_configs: Number of configurations to extract.
        metric: Distance metric for k-NN. Default: 'euclidean'.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_configs: int = 8,
        metric: str = "euclidean",
        random_state: Optional[int] = 42,
    ):
        self.n_neighbors = n_neighbors
        self.n_configs = n_configs
        self.metric = metric
        self.random_state = random_state

        self._is_fitted = False
        self._knn_graph = None
        self._configs = None
        self._energy_stats = None

    def _build_knn_graph(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-NN graph from input data.

        Args:
            X: Input features of shape [n_samples, n_features].

        Returns:
            Tuple of (distances, indices) arrays.
        """
        raise NotImplementedError(
            "k-NN graph construction is not exposed in the public stub. "
            "Parallel-DT extraction will be released later."
        )

    def extract(
        self,
        X: np.ndarray,
        return_graph: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract configurations and energy statistics from input data.

        Args:
            X: Input features of shape [n_samples, n_features].
            return_graph: Whether to also return the k-NN graph.

        Returns:
            Tuple of:
            - configs: Configuration assignments of shape [n_samples, n_configs]
            - energy_stats: Dictionary containing energy statistics
            - (optional) knn_graph: Tuple of (distances, indices) if return_graph=True
        """
        raise NotImplementedError(
            "ConfigExtractor.extract() is not available in this release. "
            "A Python implementation of Parallel-DT will be provided later."
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted configuration extraction to new data.

        Args:
            X: New input features of shape [n_samples, n_features].

        Returns:
            Configuration assignments of shape [n_samples, n_configs].
        """
        if not self._is_fitted:
            raise RuntimeError("ConfigExtractor must be fitted before transform.")

        raise NotImplementedError(
            "ConfigExtractor.transform() is not available in this release. "
            "A Python implementation of Parallel-DT will be provided later."
        )

    @property
    def configs(self) -> Optional[np.ndarray]:
        """Return extracted configurations."""
        return self._configs

    @property
    def energy_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Return computed energy statistics."""
        return self._energy_stats

    @property
    def n_clusters_per_config(self) -> Optional[np.ndarray]:
        """Return number of clusters in each configuration."""
        if self._configs is None:
            return None
        return np.array([len(np.unique(self._configs[:, i])) for i in range(self.n_configs)])
