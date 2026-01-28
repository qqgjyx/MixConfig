"""
Energy Statistics computation for MixConfig.

Computes energy-based features for each configuration.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import numpy as np


class EnergyStatistics(nn.Module):
    """
    Computes energy statistics for each configuration.

    For each configuration i, computes:
    - H_i: Shannon entropy of cluster assignment distribution
    - h_a^(i): Attractive energy (intra-cluster cohesion)
    - h_r^(i): Repulsive energy (inter-cluster separation)
    - delta_gamma_i: Energy gap (difference between min intra and max inter)

    e_i = [H_i, h_a^(i), h_r^(i), delta_gamma_i]

    Args:
        n_configs: Number of configurations.
        normalize: Whether to normalize energy statistics. Default: True.
    """

    def __init__(
        self,
        n_configs: int,
        normalize: bool = True,
    ):
        super().__init__()

        self.n_configs = n_configs
        self.normalize = normalize
        self.energy_dim = 4  # H, h_a, h_r, delta_gamma

        # Running statistics for normalization
        if normalize:
            self.register_buffer("running_mean", torch.zeros(n_configs, self.energy_dim))
            self.register_buffer("running_var", torch.ones(n_configs, self.energy_dim))
            self.register_buffer("num_batches", torch.tensor(0))

    def compute_entropy(self, cluster_assignments: torch.Tensor, n_clusters: int) -> torch.Tensor:
        """
        Compute Shannon entropy of cluster distribution.

        Args:
            cluster_assignments: Cluster indices of shape [batch_size].
            n_clusters: Number of possible clusters.

        Returns:
            Entropy value (scalar tensor).
        """
        # Count cluster occurrences
        counts = torch.bincount(cluster_assignments.long(), minlength=n_clusters).float()
        probs = counts / counts.sum()

        # Compute entropy (handle zeros)
        probs = probs + 1e-10
        entropy = -torch.sum(probs * torch.log(probs))

        return entropy

    def compute_energies(
        self,
        features: torch.Tensor,
        cluster_assignments: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute attractive and repulsive energies.

        Args:
            features: Sample features of shape [batch_size, feature_dim].
            cluster_assignments: Cluster indices of shape [batch_size].

        Returns:
            Tuple of (attractive_energy, repulsive_energy, energy_gap).
        """
        unique_clusters = torch.unique(cluster_assignments)
        n_clusters = len(unique_clusters)

        if n_clusters <= 1:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # Compute cluster centroids
        centroids = []
        intra_distances = []

        for c in unique_clusters:
            mask = cluster_assignments == c
            cluster_features = features[mask]

            if len(cluster_features) > 0:
                centroid = cluster_features.mean(dim=0)
                centroids.append(centroid)

                # Intra-cluster distances
                if len(cluster_features) > 1:
                    dists = torch.cdist(cluster_features.unsqueeze(0),
                                       centroid.unsqueeze(0).unsqueeze(0)).squeeze()
                    intra_distances.append(dists.mean())

        centroids = torch.stack(centroids)

        # Attractive energy: mean intra-cluster distance
        h_a = torch.stack(intra_distances).mean() if intra_distances else torch.tensor(0.0)

        # Repulsive energy: mean inter-cluster distance
        inter_dists = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze()
        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(inter_dists), diagonal=1).bool()
        h_r = inter_dists[mask].mean() if mask.sum() > 0 else torch.tensor(0.0)

        # Energy gap
        min_intra = torch.stack(intra_distances).min() if intra_distances else torch.tensor(0.0)
        max_inter = inter_dists[mask].max() if mask.sum() > 0 else torch.tensor(0.0)
        delta_gamma = max_inter - min_intra

        return h_a, h_r, delta_gamma

    def forward(
        self,
        features: torch.Tensor,
        cluster_assignments: torch.Tensor,
        n_clusters_per_config: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute energy statistics for all configurations.

        Args:
            features: Sample features of shape [batch_size, feature_dim].
            cluster_assignments: Cluster assignments of shape [batch_size, n_configs].
            n_clusters_per_config: Number of clusters per configuration.

        Returns:
            Energy statistics of shape [batch_size, n_configs, energy_dim].
        """
        batch_size = features.shape[0]
        device = features.device

        # Compute per-configuration statistics
        energies = torch.zeros(batch_size, self.n_configs, self.energy_dim, device=device)

        for i in range(self.n_configs):
            config_assignments = cluster_assignments[:, i]
            n_clusters = n_clusters_per_config[i] if n_clusters_per_config is not None else 100

            # Entropy (same for all samples in batch for this config)
            H = self.compute_entropy(config_assignments, n_clusters)

            # Attractive/repulsive energies
            h_a, h_r, delta_gamma = self.compute_energies(features, config_assignments)

            # Broadcast to all samples
            energies[:, i, 0] = H
            energies[:, i, 1] = h_a
            energies[:, i, 2] = h_r
            energies[:, i, 3] = delta_gamma

        # Normalize if enabled
        if self.normalize and self.training:
            # Update running statistics
            batch_mean = energies.mean(dim=0)
            batch_var = energies.var(dim=0)

            self.num_batches += 1
            momentum = 1.0 / self.num_batches.float()
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var

        if self.normalize:
            energies = (energies - self.running_mean) / (self.running_var.sqrt() + 1e-8)

        return energies


def precompute_energy_statistics(
    features: np.ndarray,
    all_configs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Precompute energy statistics from numpy arrays (for use before training).

    Args:
        features: Sample features of shape [n_samples, feature_dim].
        all_configs: Cluster assignments of shape [n_samples, n_configs].

    Returns:
        Dictionary containing precomputed statistics.
    """
    n_samples, n_configs = all_configs.shape

    stats = {
        "entropy": np.zeros((n_configs,)),
        "attractive": np.zeros((n_configs,)),
        "repulsive": np.zeros((n_configs,)),
        "energy_gap": np.zeros((n_configs,)),
    }

    for i in range(n_configs):
        config = all_configs[:, i]
        unique_clusters = np.unique(config)

        # Entropy
        counts = np.bincount(config.astype(int))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        stats["entropy"][i] = -np.sum(probs * np.log(probs + 1e-10))

        # Cluster centroids and intra-cluster distances
        centroids = []
        intra_dists = []

        for c in unique_clusters:
            mask = config == c
            cluster_feats = features[mask]
            centroid = cluster_feats.mean(axis=0)
            centroids.append(centroid)

            if len(cluster_feats) > 1:
                dists = np.linalg.norm(cluster_feats - centroid, axis=1)
                intra_dists.append(dists.mean())

        centroids = np.array(centroids)

        # Attractive energy
        stats["attractive"][i] = np.mean(intra_dists) if intra_dists else 0

        # Repulsive energy
        if len(centroids) > 1:
            from scipy.spatial.distance import pdist
            inter_dists = pdist(centroids)
            stats["repulsive"][i] = inter_dists.mean()

            # Energy gap
            min_intra = np.min(intra_dists) if intra_dists else 0
            max_inter = inter_dists.max()
            stats["energy_gap"][i] = max_inter - min_intra

    return stats
