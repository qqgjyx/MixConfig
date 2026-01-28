"""
Cluster Assignment Embedder for MixConfig.

Embeds discrete cluster assignments into continuous representations.
"""

from typing import List
import torch
import torch.nn as nn


class ClusterAssignmentEmbedder(nn.Module):
    """
    Embeds cluster assignments from each configuration into learnable vectors.

    c_i = Embed_i(omega_i(x))

    where omega_i(x) is the cluster assignment for sample x in configuration i.

    Args:
        n_configs: Number of configurations.
        max_clusters: Maximum number of clusters per configuration.
        embed_dim: Dimension of each cluster embedding.
    """

    def __init__(
        self,
        n_configs: int,
        max_clusters: int,
        embed_dim: int = 32,
    ):
        super().__init__()

        self.n_configs = n_configs
        self.max_clusters = max_clusters
        self.embed_dim = embed_dim

        # Separate embedding table for each configuration
        self.embeddings = nn.ModuleList([
            nn.Embedding(max_clusters, embed_dim)
            for _ in range(n_configs)
        ])

        # Initialize embeddings
        for embed in self.embeddings:
            nn.init.normal_(embed.weight, std=0.02)

    def forward(self, cluster_assignments: torch.Tensor) -> torch.Tensor:
        """
        Embed cluster assignments from all configurations.

        Args:
            cluster_assignments: Tensor of shape [batch_size, n_configs]
                containing cluster indices for each sample and configuration.

        Returns:
            Embeddings of shape [batch_size, n_configs, embed_dim].
        """
        batch_size = cluster_assignments.shape[0]

        # Embed each configuration's assignments
        embeddings = []
        for i, embed_layer in enumerate(self.embeddings):
            # Get cluster indices for configuration i
            indices = cluster_assignments[:, i].long()
            # Embed: [batch_size] -> [batch_size, embed_dim]
            emb = embed_layer(indices)
            embeddings.append(emb)

        # Stack: [batch_size, n_configs, embed_dim]
        return torch.stack(embeddings, dim=1)

    def get_all_embeddings(self, config_idx: int) -> torch.Tensor:
        """
        Get all cluster embeddings for a specific configuration.

        Args:
            config_idx: Index of the configuration.

        Returns:
            Embedding matrix of shape [max_clusters, embed_dim].
        """
        return self.embeddings[config_idx].weight
