"""
Models for the project

Drafted by: Juntang Wang @ Mar 5, 2025

Copyright (c) 2025, Reserved
"""
import math
from typing import List, Optional, Callable
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import MLP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


class LitAutoEncoder(pl.LightningModule):
    """
    A LightningModule for training an autoencoder model.
    as is in lightning demo
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class ConfigAttention(nn.Module):
    """
    Simple self-attention mechanism for configuration features.
    
    Args:
        config_dim (int): Number of configurations (input tokens).
        embed_dim (int): Dimension to project scalar config values into.
        num_registers (int): Number of learnable register tokens.
        use_cls_token (bool): Whether to use a learnable [CLS] token.
        dropout (float): Dropout rate.
    """
    def __init__(
        self, 
        config_dim, 
        embed_dim=32, 
        num_registers=2, 
        use_cls_token=True, 
        dropout=0.1
        ):
        super().__init__()
        self.config_dim = config_dim
        self.embed_dim = embed_dim
        self.num_registers = num_registers
        self.use_cls_token = use_cls_token

        self.token_embed = nn.Linear(1, embed_dim)
        self.register_buffer(
            'pos_encoding', 
            self._create_positional_encoding(config_dim, embed_dim)
        ) # Sinusoidal positional encodings
        
        # CLS token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)
    
        # Optional register tokens
        if num_registers > 0:
            self.registers = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
            nn.init.normal_(self.registers, std=0.02)
        else:
            self.registers = None
    
        # Attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
    
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
    
        # Output projection
        self.out_proj = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
        # Scaling factor for attention
        self.scale = embed_dim ** -0.5
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding for each position."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_enc = torch.zeros(1, max_len, d_model)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x):
        """
        Args:
            x: [B, config_dim] - raw configuration values

        Returns:
            outputs: [B, config_dim+1] - refined configs if use_cls_token=True, else [B, config_dim]
        """
        batch_size = x.shape[0]
        # Embed scalar values
        x = self.token_embed(x.unsqueeze(-1))  # [B, N, D]
        
        # Add sinusoidal positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]  # [B, N, D]
        
        tokens = [x]
        
        # Add CLS token if enabled
        if self.use_cls_token:
            cls = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
            tokens.append(cls)
            
        # Add register tokens if enabled
        if self.registers is not None:
            regs = self.registers.expand(batch_size, -1, -1)  # [B, R, D]
            tokens.append(regs)
            
        # Combine all tokens
        all_tokens = torch.cat(tokens, dim=1)  # [B, N+1+R, D]
        
        # Compute attention projections
        q = self.q_proj(all_tokens)  # [B, T, D]
        k = self.k_proj(all_tokens)  # [B, T, D]
        v = self.v_proj(all_tokens)  # [B, T, D]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, T, T]
        attn_weights = self.dropout(attn_weights)  # [B, T, T]
        attended = torch.matmul(attn_weights, v)  # [B, T, D]
        attended = self.norm(attended)  # [B, T, D]
        # Project back to scalar values
        out = self.out_proj(attended).squeeze(-1)  # [B, T]

        # Return only the config outputs (and cls token if used)
        return out[:, :-self.num_registers]  # [B, N(+1)]
        
    def compute_attention_for_visualization(self, x):
        """
        Computes attention weights for visualization.
        
        Args:
            x: Tensor of shape [B, config_dim] containing configuration values
                
        Returns:
            torch.Tensor: Attention weights with shape [B, config_dim]
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.shape[0]
            N = x.shape[1]  # Use actual input size rather than self.config_dim
            
            # Embed tokens
            x_embedded = self.token_embed(x.unsqueeze(-1))  # [B, N, D]
            x_embedded = x_embedded + self.pos_encoding[:, :N, :]  # [B, N, D]
            
            tokens = [x_embedded]
            
            if self.use_cls_token:
                cls = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
                tokens.append(cls)
                
            if self.registers is not None:
                regs = self.registers.expand(batch_size, -1, -1)  # [B, R, D]
                tokens.append(regs)
                
            # Combine all tokens
            all_tokens = torch.cat(tokens, dim=1)  # [B, N+1+R, D]
            
            # Compute attention
            q = self.q_proj(all_tokens)  # [B, T, D]
            k = self.k_proj(all_tokens)  # [B, T, D]

            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, T, T]
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, T, T]


            # Extract attention to configuration tokens
            config_attention = attn_weights[:, :N, :N]  # [B, N, N]
            # attention from cls token to configuration tokens
            cls_attention = attn_weights[:, N, :N]  # [B, N]

            # Attention Norm across query tokens to get importance of each config token
            # Calculate L2 norm (Euclidean norm) across query tokens
            # attn_norm = torch.norm(config_attention, p=2, dim=1)  # [B, N]
            attn_norm = cls_attention

            return attn_norm


class GraphSegConfig(nn.Module):
    """
    Base class for multi-resolution graph segmentation configurations.
    """
    def __init__(
        self,
        config_dim: int = 0,
        embed_dim: int = 32,
        num_registers: int = 2,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.config_dim = config_dim
        self.embed_dim = embed_dim
        self.num_registers = num_registers

        # Validation
        if config_dim <= 0:
            raise ValueError("config_dim must be positive")

        self.attention = ConfigAttention(
            config_dim=config_dim,
            embed_dim=embed_dim,
            num_registers=num_registers,
            use_cls_token=use_cls_token,
            dropout=0.2
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        # Split input into features and configuration components.
        features = x[:, :-self.config_dim]
        config = x[:, -self.config_dim:]
        attended_config = self.attention(config)
        x = torch.cat([features, attended_config], dim=1)
        return x


class LitBase(pl.LightningModule):
    """
    Base class for all LightningModules.
    """
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.train_losses.append(loss.cpu().item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)
        score = self.score(y, y_pred)
        self.val_losses.append(loss.cpu().item())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_" + self.metric_name, score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)
        score = self.score(y, y_pred)
        self.log("test_loss", loss)
        self.log("test_" + self.metric_name, score)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        raise NotImplementedError("LitBase.forward must be implemented by subclasses.")

    def score(self, y: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Score the model."""
        if self.is_classifier: y_pred = torch.argmax(y_pred, dim=-1)
        score = self.score_func(y.cpu().numpy(), y_pred.cpu().numpy())
        return score


class LitGSC(LitBase):
    """
    LightningModule for training a model w/ GSC
    """
    def __init__(
        self,
        nn_model: nn.Module,
        loss_func,
        score_func,
        metric_name,
        nn_gsc: GraphSegConfig=None,
        is_classifier: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'nn_gsc',
            'nn_model',
            'loss_func',
            'score_func',
            'metric_name'
        ])
        self.nn_gsc = nn_gsc
        self.nn_model = nn_model
        self.train_losses = []  # Store training loss
        self.val_losses = []  # Store validation loss

        self.loss_func = loss_func
        self.score_func = score_func
        self.metric_name = metric_name
        self.is_classifier = is_classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if self.nn_gsc is not None:
            x = self.nn_gsc(x)
        y_hat = self.nn_model(x)
        if not self.is_classifier:
            y_hat = y_hat.squeeze(-1)
        return y_hat


def _x4tab(x, num_continuous):
    """
    Prepare the input for the TabNet model.
    """
    x_numer = x[:, :num_continuous]
    x_categ = x[:, num_continuous:].to(torch.int)
    return x_categ, x_numer


class LitTableTransformer(LitBase):
    """
    A LightningModule for training a FT/Tab Transformer model.
    """
    def __init__(
        self,
        transformer, # either TabTransformer or FTTransformer
        loss_func,
        score_func,
        metric_name,
        is_classifier: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'transformer',
            'loss_func',
            'score_func',
            'metric_name'
        ])
        self.transformer = transformer
        self.train_losses = []  # Store training loss
        self.val_losses = []  # Store validation loss

        self.loss_func = loss_func
        self.score_func = score_func
        self.metric_name = metric_name
        self.is_classifier = is_classifier

    def forward(self, x):
        """Forward pass through the network."""
        x_categ, x_numer = _x4tab(x, self.transformer.num_continuous)
        y_hat = self.transformer(x_categ, x_numer)
        if not self.is_classifier:
            y_hat = y_hat.squeeze(-1)
        return y_hat


class CustomMLPv2(nn.Module):
    """This block implements a custom MLP that handles both classification and regression tasks.
    Based on torchvision.ops.misc.MLP implementation.

    Args:
        input_dim (int): Number of channels of the input.
        hidden_dims (List[int]): List of the hidden channel dimensions.
        output_dim (int): Number of output dimensions.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Norm layer that will be stacked on top of the linear layer. 
            If ``None`` this layer won't be used. Default: ``None``.
        activation_layer (Optional[Callable[..., nn.Module]], optional): Activation function which will be stacked on top of the 
            normalization layer (if not None), otherwise on top of the linear layer. Default: ``nn.ReLU``.
        inplace (Optional[bool], optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``.
        bias (bool): Whether to use bias in the linear layer. Default is ``True``.
        dropout (float): The probability for the dropout layer. Default: 0.2.
        is_classifier (bool): Whether this is a classification model. Default: False.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.2,
        is_classifier: bool = False,
    ):
        super().__init__()
        self.is_classifier = is_classifier

        # Create the MLP layers
        layer_dims = hidden_dims + [output_dim]  # including output layer
        self.mlp = MLP(
            in_channels=input_dim,
            hidden_channels=layer_dims,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=inplace,
            bias=bias,
            dropout=dropout
        )

        # If not a classifier, remove the last layer
        if not self.is_classifier:
            layers = list(self.mlp.children())
            del layers[-1]  # Remove the last layer
            self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

# ---------------------------------------------------------------------------- #
#                                  Deprecated:                                 #
# ---------------------------------------------------------------------------- #
class CustomMLP(nn.Module):
    """This block implements a custom MLP that handles both classification and regression tasks.
    Based on torchvision.ops.misc.MLP implementation.

    Args:
        input_dim (int): Number of channels of the input.
        hidden_dims (List[int]): List of the hidden channel dimensions.
        output_dim (int): Number of output dimensions.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Norm layer that will be stacked on top of the linear layer. 
            If ``None`` this layer won't be used. Default: ``None``.
        activation_layer (Optional[Callable[..., nn.Module]], optional): Activation function which will be stacked on top of the 
            normalization layer (if not None), otherwise on top of the linear layer. Default: ``nn.ReLU``.
        inplace (Optional[bool], optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``.
        bias (bool): Whether to use bias in the linear layer. Default is ``True``.
        dropout (float): The probability for the dropout layer. Default: 0.2.
        is_classifier (bool): Whether this is a classification model. Default: False.
        use_attention (bool): Whether to use attention mechanism on the configuration features. Default: False.
        config_dim (int): Number of configuration features at the end of the input. Default: 0.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.2,
        is_classifier: bool = False,
        use_attention: bool = False,
        config_dim: int = 0,
        embed_dim: int = 32,
        num_registers: int = 2,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.is_classifier = is_classifier
        self.use_attention = use_attention
        self.config_dim = config_dim
        self.feature_dim = input_dim - config_dim
        self.embed_dim = embed_dim
        self.num_registers = num_registers
        self.use_cls_token = use_cls_token

        # Validate input dimensions
        if use_attention and config_dim <= 0:
            raise ValueError("config_dim must be positive when use_attention is True")
        if use_attention and config_dim > input_dim:
            raise ValueError("config_dim cannot be larger than input_dim")

        # Create the MLP layers
        layer_dims = hidden_dims + [output_dim]  # including output layer
        input_dim += 1 if use_cls_token else 0
        self.mlp = MLP(
            in_channels=input_dim,
            hidden_channels=layer_dims,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=inplace,
            bias=bias,
            dropout=dropout
        )

        # If not a classifier, remove the last layer
        if not self.is_classifier:
            layers = list(self.mlp.children())
            del layers[-1]  # Remove the last layer
            self.mlp = nn.Sequential(*layers)

        # Add attention mechanism for configuration features if requested.
        # Here we use the new attention head with registers.
        if self.use_attention:
            self.attention = ConfigAttention(
                config_dim=config_dim,
                embed_dim=embed_dim,
                num_registers=num_registers,
                use_cls_token=use_cls_token,
                dropout=dropout
            )
            # Add the final output layer if not a classifier
            if not self.is_classifier:
                self.final_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        if self.use_attention:
            # Split input into features and configuration components.
            features = x[:, :self.feature_dim]
            config = x[:, self.feature_dim:]
            # Apply attention to configuration features.
            attended_config = self.attention(config)
            # Combine features and the attended configuration.
            x = torch.cat([features, attended_config], dim=1)

        # Pass through the MLP.
        x = self.mlp(x)

        # Apply final layer if not a classifier and attention is used.
        if not self.is_classifier and self.use_attention:
            x = self.final_layer(x)

        return x


class LitMLP(pl.LightningModule):
    """
    A LightningModule for training a custom MLP model.

    Args:
        mlp (CustomMLP): The MLP model to be trained.
    """
    def __init__(self, mlp, loss_func=None):
        super().__init__()
        self.save_hyperparameters(ignore=['mlp'])
        self.mlp = mlp
        self.is_classifier = mlp.is_classifier
        self.train_losses = []  # Store training loss
        self.val_losses = []  # Store validation loss

        # Define loss and score functions
        if self.is_classifier:
            self.loss_func = nn.functional.cross_entropy
            self.score_func = accuracy_score
            self.metric_name = "accuracy"
        else:
            self.loss_func = nn.MSELoss()
            self.score_func = r2_score
            self.metric_name = "r2"

        if loss_func is not None:
            if loss_func == "MAE":
                self.loss_func = nn.L1Loss()
            else:
                raise ValueError(f"Invalid loss function: {loss_func}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.mlp(x)
        if not self.is_classifier:
            y_hat = y_hat.squeeze(-1)
        loss = self.loss_func(y_hat, y)
        self.train_losses.append(loss.cpu().item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.mlp(x)
        if not self.is_classifier:
            y_pred = y_pred.squeeze(-1)
        loss = self.loss_func(y_pred, y)
        if self.is_classifier:
            y_pred = torch.argmax(y_pred, dim=-1)
        score = self.score_func(y.cpu().numpy(), y_pred.cpu().numpy())
        self.val_losses.append(loss.cpu().item())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_" + self.metric_name, score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.mlp(x)
        if not self.is_classifier:
            y_pred = y_pred.squeeze(-1)
        loss = self.loss_func(y_pred, y)
        if self.is_classifier:
            y_pred = torch.argmax(y_pred, dim=-1)
        score = self.score_func(y.cpu().numpy(), y_pred.cpu().numpy())
        self.log("test_loss", loss)
        self.log("test_" + self.metric_name, score)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer
