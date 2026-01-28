"""
Energy-Aware Selector for MixConfig.

Learns sample-specific configuration weights based on sample context,
cluster embeddings, and energy statistics.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .encoder import SampleContextEncoder
from .embedder import ClusterAssignmentEmbedder
from .energy import EnergyStatistics


class EnergyAwareSelector(nn.Module):
    """
    Energy-Aware Selector for sample-specific configuration weights.

    For each sample x, computes:
    1. Sample context: h = MLP_enc(x)
    2. Cluster embedding: c_i = Embed_i(omega_i(x))
    3. Energy statistics: e_i = [H_i, h_a^(i), h_r^(i), delta_gamma_i]
    4. Compatibility score: s_i = MLP_score([h; c_i; e_i])
    5. Weights: w_i(x) = softmax(s_i)
    6. Mixed representation: z(x) = sum_i w_i(x) * c_i

    Args:
        input_dim: Dimension of input features.
        n_configs: Number of configurations.
        max_clusters: Maximum number of clusters per configuration.
        context_dim: Dimension of sample context embedding.
        cluster_embed_dim: Dimension of cluster embeddings.
        hidden_dims: Hidden dimensions for encoders.
        dropout: Dropout probability.
        temperature: Temperature for softmax (lower = sharper weights).
    """

    def __init__(
        self,
        input_dim: int,
        n_configs: int,
        max_clusters: int = 256,
        context_dim: int = 64,
        cluster_embed_dim: int = 32,
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_configs = n_configs
        self.max_clusters = max_clusters
        self.context_dim = context_dim
        self.cluster_embed_dim = cluster_embed_dim
        self.temperature = temperature

        # Sample context encoder: h = MLP_enc(x)
        self.context_encoder = SampleContextEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=context_dim,
            dropout=dropout,
        )

        # Cluster assignment embedder: c_i = Embed_i(omega_i(x))
        self.cluster_embedder = ClusterAssignmentEmbedder(
            n_configs=n_configs,
            max_clusters=max_clusters,
            embed_dim=cluster_embed_dim,
        )

        # Energy statistics module
        self.energy_stats = EnergyStatistics(
            n_configs=n_configs,
            normalize=True,
        )

        # Score MLP: s_i = MLP_score([h; c_i; e_i])
        # Input: context (context_dim) + cluster embed (cluster_embed_dim) + energy (4)
        score_input_dim = context_dim + cluster_embed_dim + 4
        self.score_mlp = nn.Sequential(
            nn.Linear(score_input_dim, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1),
        )

        # Output projection for mixed representation
        self.output_proj = nn.Linear(cluster_embed_dim, cluster_embed_dim)

    def compute_weights(
        self,
        features: torch.Tensor,
        cluster_assignments: torch.Tensor,
        energy_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sample-specific configuration weights.

        Args:
            features: Input features of shape [batch_size, input_dim].
            cluster_assignments: Cluster assignments of shape [batch_size, n_configs].
            energy_stats: Precomputed energy statistics of shape [batch_size, n_configs, 4].
                If None, will be computed from inputs.

        Returns:
            Configuration weights of shape [batch_size, n_configs].
        """
        batch_size = features.shape[0]
        device = features.device

        # 1. Sample context: h = MLP_enc(x)
        context = self.context_encoder(features)  # [B, context_dim]

        # 2. Cluster embeddings: c_i = Embed_i(omega_i(x))
        cluster_embeds = self.cluster_embedder(cluster_assignments)  # [B, n_configs, embed_dim]

        # 3. Energy statistics: e_i
        if energy_stats is None:
            energy_stats = self.energy_stats(features, cluster_assignments)  # [B, n_configs, 4]
        else:
            if energy_stats.dim() == 2:
                energy_stats = energy_stats.unsqueeze(0).expand(batch_size, -1, -1)

        # 4. Compute compatibility scores for each configuration
        scores = []
        for i in range(self.n_configs):
            # Concatenate: [h; c_i; e_i]
            c_i = cluster_embeds[:, i, :]  # [B, embed_dim]
            e_i = energy_stats[:, i, :]    # [B, 4]

            combined = torch.cat([context, c_i, e_i], dim=-1)  # [B, score_input_dim]
            score_i = self.score_mlp(combined)  # [B, 1]
            scores.append(score_i)

        # Stack scores: [B, n_configs]
        scores = torch.cat(scores, dim=-1)

        # 5. Softmax to get weights
        weights = torch.softmax(scores / self.temperature, dim=-1)

        return weights

    def get_mixed_representation(
        self,
        features: torch.Tensor,
        cluster_assignments: torch.Tensor,
        energy_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the mixed representation z(x) = sum_i w_i(x) * c_i.

        Args:
            features: Input features of shape [batch_size, input_dim].
            cluster_assignments: Cluster assignments of shape [batch_size, n_configs].
            energy_stats: Precomputed energy statistics.

        Returns:
            Mixed representation of shape [batch_size, cluster_embed_dim].
        """
        # Get weights
        weights = self.compute_weights(features, cluster_assignments, energy_stats)

        # Get cluster embeddings
        cluster_embeds = self.cluster_embedder(cluster_assignments)  # [B, n_configs, embed_dim]

        # Weighted sum: z(x) = sum_i w_i(x) * c_i
        # weights: [B, n_configs] -> [B, n_configs, 1]
        weights_expanded = weights.unsqueeze(-1)

        # [B, n_configs, embed_dim] * [B, n_configs, 1] -> sum over configs
        mixed = (cluster_embeds * weights_expanded).sum(dim=1)  # [B, embed_dim]

        # Optional projection
        mixed = self.output_proj(mixed)

        return mixed

    def forward(
        self,
        features: torch.Tensor,
        cluster_assignments: torch.Tensor,
        energy_stats: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass: compute mixed representation.

        Args:
            features: Input features of shape [batch_size, input_dim].
            cluster_assignments: Cluster assignments of shape [batch_size, n_configs].
            energy_stats: Precomputed energy statistics.
            return_weights: If True, also return configuration weights.

        Returns:
            Mixed representation of shape [batch_size, cluster_embed_dim].
            If return_weights=True, returns (mixed_repr, weights).
        """
        mixed = self.get_mixed_representation(features, cluster_assignments, energy_stats)

        if return_weights:
            weights = self.compute_weights(features, cluster_assignments, energy_stats)
            return mixed, weights

        return mixed

    def get_attention_weights(
        self,
        features: torch.Tensor,
        cluster_assignments: torch.Tensor,
        energy_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get configuration attention weights for visualization.

        This is an alias for compute_weights() for visualization purposes.

        Args:
            features: Input features of shape [batch_size, input_dim].
            cluster_assignments: Cluster assignments of shape [batch_size, n_configs].
            energy_stats: Precomputed energy statistics.

        Returns:
            Configuration weights of shape [batch_size, n_configs].
        """
        self.eval()
        with torch.no_grad():
            return self.compute_weights(features, cluster_assignments, energy_stats)


class MixConfigPredictor(nn.Module):
    """
    Complete MixConfig model with Energy-Aware Selector and downstream predictor.

    Combines configuration extraction, weight computation, and prediction.

    Args:
        input_dim: Dimension of input features.
        output_dim: Dimension of output (1 for regression, num_classes for classification).
        n_configs: Number of configurations.
        max_clusters: Maximum clusters per configuration.
        hidden_dims: Hidden dimensions for predictor MLP.
        is_classifier: Whether this is a classification task.
        **selector_kwargs: Additional arguments for EnergyAwareSelector.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        n_configs: int = 8,
        max_clusters: int = 256,
        hidden_dims: list = [256, 128, 64],
        is_classifier: bool = False,
        **selector_kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_configs = n_configs
        self.is_classifier = is_classifier

        # Energy-Aware Selector
        self.selector = EnergyAwareSelector(
            input_dim=input_dim,
            n_configs=n_configs,
            max_clusters=max_clusters,
            **selector_kwargs,
        )

        # Predictor MLP
        # Input: original features + mixed representation
        predictor_input_dim = input_dim + self.selector.cluster_embed_dim

        layers = []
        prev_dim = predictor_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.predictor = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        cluster_assignments: torch.Tensor,
        energy_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: compute prediction from mixed representation.

        Args:
            features: Input features of shape [batch_size, input_dim].
            cluster_assignments: Cluster assignments of shape [batch_size, n_configs].
            energy_stats: Precomputed energy statistics.

        Returns:
            Predictions of shape [batch_size, output_dim].
        """
        # Get mixed representation from selector
        mixed_repr = self.selector(features, cluster_assignments, energy_stats)

        # Concatenate original features with mixed representation
        combined = torch.cat([features, mixed_repr], dim=-1)

        # Predict
        output = self.predictor(combined)

        if not self.is_classifier:
            output = output.squeeze(-1)

        return output
