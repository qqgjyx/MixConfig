"""
Visualization utilities.

Drafted by Juntang Wang at Mar 5th 4 the GASNN project
"""
from typing import Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def plot_decorator(title: str, xlabel: str = None, ylabel: str = None, legend: bool = True):
    """
    Decorator that sets common style elements for plot methods.
    
    Args:
        title: The plot title
        xlabel: The x-axis label
        ylabel: The y-axis label
        legend: Whether to show legend. Default True.
    """
    def decorator(plot_method):
        def wrapper(*args, **kwargs):
            # Get self from args
            self = args[0]
            
            # Get or create axis
            if 'ax' in kwargs and kwargs['ax'] is not None:
                ax = kwargs['ax']
            else:
                _, ax = plt.subplots()  # Using global figure size from rcParams
                kwargs['ax'] = ax
            
            # Call the original plotting method
            plot_method(*args, **kwargs)
            
            # Apply styling
            if self.model_name is not None:
                ax.set_title(f'{title} {self.model_name}', pad=20)
            else:
                ax.set_title(title, pad=20)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if legend:
                ax.legend()
            ax.grid(True, alpha=0.5)
            
            return ax
        return wrapper
    return decorator


class CustomVisualizer():
    """
    A custom visualizer for the project.
    """
    def __init__(
        self,
        model: Union[pl.LightningModule, sklearn.base.BaseEstimator, dict],
        model_name: str = None
    ):
        self.model = model
        self.model_name = model_name
        
        if isinstance(self.model, dict):
            self.df = pd.DataFrame(self.model)

    @plot_decorator('Training and Validation Loss Evolution', 'Epoch', 'Loss')
    def plot_epoch_loss(self, num_epochs: int = None, ax=None):
        """
        Plot the loss of the models.
        
        Args:
            num_epochs: Number of epochs to plot
            ax: Optional matplotlib axis to plot on. If None, creates a new figure.
        
        Returns:
            The matplotlib axis with the plot
        """
        # Calculate batches per epoch
        train_batches_per_epoch = len(self.model.train_losses) // num_epochs
        val_batches_per_epoch = len(self.model.val_losses) // num_epochs

        # Get final loss value of each epoch
        train_epoch_losses = [self.model.train_losses[
            min((i+1)*train_batches_per_epoch - 1, len(self.model.train_losses)-1)
        ] for i in range(num_epochs)]
        val_epoch_losses = [self.model.val_losses[
            min((i+1)*val_batches_per_epoch - 1, len(self.model.val_losses)-1)
        ] for i in range(num_epochs)]
        sns.lineplot(
            x=range(1, num_epochs+1), 
            y=train_epoch_losses, 
            label="Train Loss", 
            linewidth=2, 
            ax=ax
        )
        sns.lineplot(
            x=range(1, num_epochs+1), 
            y=val_epoch_losses, 
            label="Val Loss", 
            linewidth=2, 
            ax=ax
        )
        return ax
    
    @plot_decorator('Actual vs Predicted Values', 'Predicted', 'True', legend=False)
    def plot_actual_vs_predicted(self, data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]], ax=None, error_metric="MSE"):
        """
        Plot actual vs predicted values with performance metrics.
        
        Args:
            data: DataLoader containing the data to evaluate or a tuple of numpy arrays (X, y)
            ax: Optional matplotlib axis to plot on. If None, creates a new figure.
            
        Returns:
            The matplotlib axis with the plot
        """
        y_true = np.array([])
        y_pred = np.array([])
        if isinstance(self.model, pl.LightningModule):
            self.model.eval()
            for batch in data:
                with torch.no_grad():
                    inputs, labels = batch
                    outputs = self.model(inputs)
                    predicted = outputs.squeeze(-1)
                    y_true = np.concatenate([y_true, labels.cpu().numpy()])
                    y_pred = np.concatenate([y_pred, predicted.cpu().numpy()])
        else:  # for sklearn model
            y_pred = self.model.predict(data[0])
            y_true = data[1]

        line_min = min(y_pred.min(), y_true.min())
        line_max = max(y_pred.max(), y_true.max())
        if error_metric == "MSE": error = mean_squared_error(y_true, y_pred)
        elif error_metric == "MAE": error = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics = f'{error_metric}: {error:.4f}\nR²: {r2:.4f}'
        
        ax.scatter(
            y_pred, y_true, 
            alpha=0.6, 
            c='#2E86C1', 
            s=42, 
            edgecolor='white', 
            linewidth=0.5
        )
        ax.plot(
            [line_min, line_max], [line_min, line_max],
            '--', 
            color='#E74C3C', 
            alpha=0.8, 
            linewidth=2
        )
        ax.text(
            0.05, 0.95, 
            metrics, 
            transform=ax.transAxes,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#BDC3C7', alpha=0.7, pad=0.5)
        )
        ax.set_aspect('equal', adjustable='box')
        return ax

    @plot_decorator('Confusion Matrix', 'Predicted', 'True', legend=False)
    def plot_confusion_matrix(self, data: Union[DataLoader, Tuple[np.ndarray, np.ndarray]], ax=None):
        """
        Plot confusion matrix for model predictions.
        
        Args:
            data: DataLoader containing the data to evaluate or a tuple of numpy arrays (X, y)
            ax: Optional matplotlib axis to plot on. If None, creates a new figure.
            
        Returns:
            The matplotlib axis with the plot
        """
        y_true = []
        y_pred = []
        if isinstance(self.model, pl.LightningModule): # for pytorch model
            self.model.eval()
            for batch in data:
                with torch.no_grad():
                    inputs, labels = batch
                    outputs = self.model(inputs)
                    predicted = torch.argmax(outputs, dim=1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
        else: # for sklearn model
            y_pred = self.model.predict(data[0])
            y_true = data[1]

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax, square=True)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        return ax

    @plot_decorator('Feature Importances', 'Importance', 'Features', legend=False)
    def plot_feature_importance(self, ax=None, n_features=42, feature_names=None, show_y_labels=False):
        """
        Plot feature importance for tree-based models.
        
        Args:
            ax: Optional matplotlib axis to plot on
            n_features: Number of top features to display
            feature_names: Optional list of feature names
            show_y_labels: Whether to display y-axis labels (feature names)
            
        Returns:
            The matplotlib axis with the plot
        """
        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            raise AttributeError("Model lacks feature_importances_ or coef_ attribute")

        if feature_names is None: # Create feature names if not provided
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(n_features)
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        if not show_y_labels:
            ax.set_yticklabels([])
        
        return ax

    @plot_decorator('Ablation Study: Metric Trend', 'Feature Combinations', None, False)
    def plot_ablation_line(self, ax=None, metric="test_accuracy", highlight_index=1):
        """
        Plots a line chart showing the trend of metrics across different model variants or feature combinations.
        
        Args:
            ax: Matplotlib axis to plot on
            metric: The metric to plot (default: "test_accuracy")
            highlight_index: Optional index to highlight with a different color
        """
        df = self.df.copy()

        if highlight_index is not None and 0 <= highlight_index < len(self.df):
            highlighted_row = df.iloc[highlight_index]
            df = df.drop(highlight_index).reset_index(drop=True)
            df = pd.concat([pd.DataFrame([highlighted_row]), df])
            
            sns.lineplot(x=range(len(df)), y=df[metric], ax=ax, label='combine', marker='o')
            sns.lineplot(x=range(len(df))[:highlight_index+1], y=df[metric][:highlight_index+1], ax=ax, label='RMS', marker='o')
            
            first_value = df.iloc[highlight_index][metric]
            ax.annotate('', xy=(highlight_index, first_value), xytext=(highlight_index-0.5, first_value), arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
            ax.annotate('', xy=(highlight_index, first_value), xytext=(highlight_index+0.5, first_value), arrowprops=dict(arrowstyle='<-', color='black', lw=1.5))
        else:
            sns.lineplot(x=range(len(self.df)), y=self.df[metric], ax=ax)
            
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["model_name"], rotation=45)
        ax.set_ylabel(metric)

    @staticmethod
    def create_subplot_grid(rows=1, cols=2, figsize=(10, 4.5)):
        """
        Create a grid of subplots for multiple visualizations.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            figsize: Figure size as (width, height) tuple
            
        Returns:
            fig: The matplotlib figure
            axes: 2D array of axes objects
        """
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # If there's only one subplot, axes won't be a 2D array
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        plt.tight_layout(pad=3.0)
        return fig, axes
    
    @plot_decorator("Attention Maps", "Configuration Tokens", "Samples", legend=False)
    def plot_attention_maps(self, dataloader, save_to=None, ax=None):
        """
        Plots the attention maps for each sample horizontally.
        The samples are sorted vertically based on their configuration values,
        using lexicographical order from the first configuration to the last.

        Args:
            dataloader: DataLoader containing configuration tensors.
            ax (Optional): Matplotlib axis to plot on.
        """
        self.model.eval()
        
        # Get all attention weights and configurations
        all_attn_weights = []
        all_configs = []
        
        for batch in dataloader:
            inputs, _ = batch
            attn_weight = self.model.nn_gsc.attention.compute_attention_for_visualization(inputs)
            all_attn_weights.append(attn_weight)
            all_configs.append(inputs.cpu().numpy())
        
        # Combine all batches
        attn_weights = torch.cat(all_attn_weights, dim=0)
        config_values = np.vstack(all_configs)
        
        attn_np = attn_weights.cpu().numpy()  # [N, C]
        # normalize each row to sum to 1
        attn_np = attn_np / attn_np.sum(axis=1, keepdims=True)
        
        # Lexicographical sort: use keys in reverse order so that the first config is primary.
        keys = tuple(config_values[:, i] for i in range(config_values.shape[1] - 1, -1, -1))
        sort_idx = np.lexsort(keys)
        self.sorted_attn = attn_np[sort_idx]

        sns.heatmap(
            self.sorted_attn,
            cmap="viridis",
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            ax=ax
        )
        
        if save_to is not None:
            # save the sorted_attn as a csv file
            pd.DataFrame(self.sorted_attn).to_csv(save_to, index=False)

        return ax


def plot_lowdata_curve(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, ax=None):
    """
    Plot low-data performance curves.
    """
    if ax is None:
        _, ax = plt.subplots()
    sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, marker="o", ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.4)
    return ax


def plot_ablation_bars(df: pd.DataFrame, x_col: str, y_col: str, ax=None):
    """
    Plot ablation results as a bar chart.
    """
    if ax is None:
        _, ax = plt.subplots()
    sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, axis="y", alpha=0.4)
    return ax


def plot_weight_heatmap(weights: np.ndarray, ax=None):
    """
    Plot a heatmap of configuration weights.
    """
    if ax is None:
        _, ax = plt.subplots()
    sns.heatmap(weights, cmap="viridis", cbar=True, ax=ax)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Sample")
    return ax
