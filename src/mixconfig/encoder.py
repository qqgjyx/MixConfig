"""
Sample Context Encoder for MixConfig.

Encodes raw sample features into a latent context representation.
"""

from typing import List, Optional, Callable
import torch
import torch.nn as nn


class SampleContextEncoder(nn.Module):
    """
    MLP encoder that maps raw sample features to a context embedding.

    h = MLP_enc(x)

    Args:
        input_dim: Dimension of input features.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Dimension of the context embedding.
        activation: Activation function class. Default: nn.ReLU.
        dropout: Dropout probability. Default: 0.1.
        norm_layer: Optional normalization layer class.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        output_dim: int = 64,
        activation: Callable[..., nn.Module] = nn.ReLU,
        dropout: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sample features to context embedding.

        Args:
            x: Input features of shape [batch_size, input_dim].

        Returns:
            Context embedding of shape [batch_size, output_dim].
        """
        return self.encoder(x)
