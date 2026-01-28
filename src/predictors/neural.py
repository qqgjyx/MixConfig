"""
Neural network predictors for MixConfig.

Includes MLP and PyTorch Lightning wrappers.
"""

from typing import List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score


class MLPPredictor(nn.Module):
    """
    Multi-Layer Perceptron for downstream prediction.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension (1 for regression, n_classes for classification).
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function class.
        dropout: Dropout probability.
        norm_layer: Normalization layer class.
        is_classifier: Whether this is a classification task.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [256, 128, 64],
        activation: Callable[..., nn.Module] = nn.ReLU,
        dropout: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.LayerNorm,
        is_classifier: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_classifier = is_classifier

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

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, input_dim].

        Returns:
            Output tensor of shape [batch_size, output_dim].
        """
        out = self.mlp(x)

        if not self.is_classifier and self.output_dim == 1:
            out = out.squeeze(-1)

        return out


class LitMLPPredictor(pl.LightningModule):
    """
    PyTorch Lightning wrapper for MLP predictor.

    Handles training, validation, and testing with appropriate metrics.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension.
        hidden_dims: Hidden layer dimensions.
        is_classifier: Whether classification task.
        learning_rate: Learning rate.
        weight_decay: L2 regularization.
        loss_fn: Optional custom loss function.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [256, 128, 64],
        is_classifier: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["loss_fn"])

        self.model = MLPPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            is_classifier=is_classifier,
        )

        self.is_classifier = is_classifier
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Set loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif is_classifier:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

        # For tracking
        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]  # Handle optional config data
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        self.train_losses.append(loss.detach().cpu().item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        # Compute metrics
        if self.is_classifier:
            preds = torch.argmax(y_hat, dim=-1)
            acc = (preds == y).float().mean()
            self.log("val_acc", acc, prog_bar=True)
        else:
            # R2 score for regression
            y_np = y.cpu().numpy()
            y_hat_np = y_hat.cpu().numpy()
            r2 = r2_score(y_np, y_hat_np)
            self.log("val_r2", r2, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        self.val_losses.append(loss.detach().cpu().item())

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        if self.is_classifier:
            preds = torch.argmax(y_hat, dim=-1)
            acc = (preds == y).float().mean()
            self.log("test_acc", acc)
        else:
            y_np = y.cpu().numpy()
            y_hat_np = y_hat.cpu().numpy()
            r2 = r2_score(y_np, y_hat_np)
            self.log("test_r2", r2)

        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def predict_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x)


class MixConfigMLPPredictor(pl.LightningModule):
    """
    MLP Predictor that accepts MixConfig representations.

    Combines original features with MixConfig mixed representations.

    Args:
        feature_dim: Original feature dimension.
        mixconfig_dim: MixConfig representation dimension.
        output_dim: Output dimension.
        hidden_dims: Hidden layer dimensions.
        is_classifier: Whether classification task.
        use_original_features: Whether to concatenate original features.
    """

    def __init__(
        self,
        feature_dim: int,
        mixconfig_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [256, 128, 64],
        is_classifier: bool = False,
        use_original_features: bool = True,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.use_original_features = use_original_features

        input_dim = mixconfig_dim
        if use_original_features:
            input_dim += feature_dim

        self.model = MLPPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            is_classifier=is_classifier,
        )

        self.is_classifier = is_classifier
        self.learning_rate = learning_rate

        if is_classifier:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def forward(
        self,
        features: torch.Tensor,
        mixconfig_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with both original features and MixConfig representation.

        Args:
            features: Original features [batch_size, feature_dim].
            mixconfig_repr: MixConfig representation [batch_size, mixconfig_dim].

        Returns:
            Predictions.
        """
        if self.use_original_features:
            x = torch.cat([features, mixconfig_repr], dim=-1)
        else:
            x = mixconfig_repr

        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, y, mixconfig_repr = batch
        y_hat = self(features, mixconfig_repr)

        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_losses.append(loss.detach().cpu().item())

        return loss

    def validation_step(self, batch, batch_idx):
        features, y, mixconfig_repr = batch
        y_hat = self(features, mixconfig_repr)

        loss = self.loss_fn(y_hat, y)

        if self.is_classifier:
            preds = torch.argmax(y_hat, dim=-1)
            acc = (preds == y).float().mean()
            self.log("val_acc", acc, prog_bar=True)
        else:
            y_np = y.cpu().numpy()
            y_hat_np = y_hat.cpu().numpy()
            r2 = r2_score(y_np, y_hat_np)
            self.log("val_r2", r2, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        self.val_losses.append(loss.detach().cpu().item())

        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
