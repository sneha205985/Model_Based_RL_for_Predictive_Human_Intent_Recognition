"""
Visualization functions for human behavior analysis.

This module provides comprehensive plotting and visualization tools for
trajectory data, intent predictions, feature analysis, and model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
from pathlib import Path
import logging

from ..data.synthetic_generator import SyntheticSequence, GestureType
from ..models.human_behavior import HumanState, BehaviorPrediction, BehaviorType
from ..models.intent_predictor import IntentPrediction, IntentType
from ..data.feature_extraction import ExtractedFeatures
from ..data.validation import ValidationResult
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BehaviorVisualizer:
    """
    Comprehensive visualization toolkit for human behavior analysis.
    
    This class provides methods for plotting trajectories, intent predictions,
    feature distributions, model performance, and uncertainty visualizations.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        save_format: str = 'png'
    ):
        """
        Initialize behavior visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure resolution
            save_format: Format for saving plots ('png', 'pdf', 'svg')
        """
        self.figsize = figsize
        self.dpi = dpi
        self.save_format = save_format
        
        # Color schemes
        self.gesture_colors = {
            GestureType.WAVE: '#FF6B6B',
            GestureType.POINT: '#4ECDC4',
            GestureType.GRAB: '#45B7D1',
            GestureType.HANDOVER: '#96CEB4',
            GestureType.REACH: '#FECA57',
            GestureType.IDLE: '#BDC3C7'
        }
        
        self.behavior_colors = {
            BehaviorType.GESTURE: '#E74C3C',
            BehaviorType.HANDOVER: '#2ECC71',
            BehaviorType.REACHING: '#3498DB',
            BehaviorType.POINTING: '#F39C12',
            BehaviorType.IDLE: '#95A5A6',
            BehaviorType.UNKNOWN: '#34495E'
        }
        
        self.intent_colors = {
            IntentType.REACH_OBJECT: '#E74C3C',
            IntentType.HANDOVER_TO_ROBOT: '#2ECC71',
            IntentType.HANDOVER_TO_HUMAN: '#27AE60',
            IntentType.POINT_TO_LOCATION: '#F39C12',
            IntentType.PICK_UP_OBJECT: '#8E44AD',
            IntentType.PLACE_OBJECT: '#3498DB',
            IntentType.GESTURE_COMMUNICATION: '#E67E22',
            IntentType.IDLE_WAITING: '#95A5A6',
            IntentType.UNKNOWN: '#34495E'
        }
        
        logger.info("Initialized behavior visualizer")
    
    def plot_trajectory_3d(
        self,
        sequence: SyntheticSequence,
        show_uncertainty: bool = False,
        uncertainty_data: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        interactive: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot 3D trajectory with optional uncertainty visualization.
        
        Args:
            sequence: Synthetic sequence to plot
            show_uncertainty: Whether to show uncertainty bands
            uncertainty_data: Uncertainty values for each point [N, 3]
            save_path: Path to save the plot
            interactive: Whether to create interactive Plotly plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        trajectory = sequence.hand_trajectory
        timestamps = sequence.timestamps
        
        if interactive:
            return self._plot_trajectory_3d_plotly(
                trajectory, timestamps, sequence.gesture_type,
                uncertainty_data, save_path
            )
        else:
            return self._plot_trajectory_3d_matplotlib(
                trajectory, timestamps, sequence.gesture_type,
                show_uncertainty, uncertainty_data, save_path
            )
    
    def _plot_trajectory_3d_matplotlib(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray,
        gesture_type: GestureType,
        show_uncertainty: bool,
        uncertainty_data: Optional[np.ndarray],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Create 3D trajectory plot using matplotlib."""
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Main trajectory
        color = self.gesture_colors.get(gesture_type, '#333333')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=color, linewidth=2, alpha=0.8, label=f'{gesture_type.value}')
        
        # Start and end points
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                  c='red', s=100, marker='s', label='End')
        
        # Uncertainty visualization
        if show_uncertainty and uncertainty_data is not None:
            # Plot uncertainty as error bars at select points
            step = max(1, len(trajectory) // 20)  # Show ~20 uncertainty points
            for i in range(0, len(trajectory), step):
                if i < len(uncertainty_data):
                    unc = uncertainty_data[i]
                    pos = trajectory[i]
                    
                    # Error bars in each dimension
                    ax.plot([pos[0] - unc[0], pos[0] + unc[0]], [pos[1], pos[1]], [pos[2], pos[2]],
                           'k-', alpha=0.3, linewidth=1)
                    ax.plot([pos[0], pos[0]], [pos[1] - unc[1], pos[1] + unc[1]], [pos[2], pos[2]],
                           'k-', alpha=0.3, linewidth=1)
                    ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [pos[2] - unc[2], pos[2] + unc[2]],
                           'k-', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Trajectory - {gesture_type.value.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_trajectory_3d_plotly(
        self,
        trajectory: np.ndarray,
        timestamps: np.ndarray,
        gesture_type: GestureType,
        uncertainty_data: Optional[np.ndarray],
        save_path: Optional[str]
    ) -> go.Figure:
        """Create interactive 3D trajectory plot using Plotly."""
        color = self.gesture_colors.get(gesture_type, '#333333')
        
        fig = go.Figure()
        
        # Main trajectory
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines+markers',
            line=dict(color=color, width=4),
            marker=dict(size=3, opacity=0.6),
            name=f'{gesture_type.value}',
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<br>' +
                         'Time: %{customdata:.3f}s<extra></extra>',
            customdata=timestamps
        ))
        
        # Start and end points
        fig.add_trace(go.Scatter3d(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            z=[trajectory[0, 2]],
            mode='markers',
            marker=dict(color='green', size=8),
            name='Start'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[trajectory[-1, 0]],
            y=[trajectory[-1, 1]],
            z=[trajectory[-1, 2]],
            mode='markers',
            marker=dict(color='red', size=8, symbol='square'),
            name='End'
        ))
        
        # Uncertainty visualization
        if uncertainty_data is not None:
            # Add uncertainty as error bars (simplified)
            step = max(1, len(trajectory) // 10)
            for i in range(0, len(trajectory), step):
                if i < len(uncertainty_data):
                    unc = uncertainty_data[i]
                    pos = trajectory[i]
                    
                    # Uncertainty ellipsoid (simplified as lines)
                    for dim, (delta, axis) in enumerate(zip(unc, ['x', 'y', 'z'])):
                        line_data = np.zeros((2, 3))
                        line_data[:, :] = pos
                        line_data[0, dim] -= delta
                        line_data[1, dim] += delta
                        
                        fig.add_trace(go.Scatter3d(
                            x=line_data[:, 0],
                            y=line_data[:, 1],
                            z=line_data[:, 2],
                            mode='lines',
                            line=dict(color='black', width=1),
                            showlegend=False,
                            opacity=0.3
                        ))
        
        # Layout
        fig.update_layout(
            title=f'3D Trajectory - {gesture_type.value.capitalize()}',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='cube'
            ),
            showlegend=True,
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
        
        return fig
    
    def plot_trajectory_time_series(
        self,
        sequence: SyntheticSequence,
        show_derivatives: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot trajectory components over time with derivatives.
        
        Args:
            sequence: Synthetic sequence to plot
            show_derivatives: Whether to show velocity and acceleration
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        trajectory = sequence.hand_trajectory
        timestamps = sequence.timestamps
        
        # Compute derivatives
        dt = np.mean(np.diff(timestamps))
        velocities = np.gradient(trajectory, dt, axis=0)
        accelerations = np.gradient(velocities, dt, axis=0)
        
        n_plots = 3 if show_derivatives else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(self.figsize[0], self.figsize[1] * n_plots // 2), dpi=self.dpi)
        
        if n_plots == 1:
            axes = [axes]
        
        dims = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']
        
        # Position plot
        for i, (dim, color) in enumerate(zip(dims, colors)):
            axes[0].plot(timestamps, trajectory[:, i], label=f'{dim} Position', color=color, linewidth=2)
        
        axes[0].set_ylabel('Position (m)')
        axes[0].set_title(f'Trajectory Components - {sequence.gesture_type.value.capitalize()}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if show_derivatives:
            # Velocity plot
            for i, (dim, color) in enumerate(zip(dims, colors)):
                axes[1].plot(timestamps, velocities[:, i], label=f'{dim} Velocity', color=color, linewidth=2)
            
            axes[1].set_ylabel('Velocity (m/s)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Acceleration plot
            for i, (dim, color) in enumerate(zip(dims, colors)):
                axes[2].plot(timestamps, accelerations[:, i], label=f'{dim} Acceleration', color=color, linewidth=2)
            
            axes[2].set_ylabel('Acceleration (m/s²)')
            axes[2].set_xlabel('Time (s)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[0].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_intent_probabilities(
        self,
        predictions: List[IntentPrediction],
        show_uncertainty: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot intent probability distributions with uncertainty.
        
        Args:
            predictions: List of intent predictions
            show_uncertainty: Whether to show uncertainty bars
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Prepare data
        intent_names = [pred.intent_type.value.replace('_', ' ').title() for pred in predictions]
        probabilities = [pred.probability for pred in predictions]
        confidences = [pred.confidence for pred in predictions]
        
        # Get colors
        colors = [self.intent_colors.get(pred.intent_type, '#333333') for pred in predictions]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] // 2), dpi=self.dpi)
        
        # Probability bar plot
        bars = ax1.barh(intent_names, probabilities, color=colors, alpha=0.7)
        
        # Add uncertainty error bars if available
        if show_uncertainty:
            uncertainties = []
            for pred in predictions:
                total_unc = pred.uncertainty.get('total', 0.1)
                uncertainties.append(total_unc * pred.probability)  # Scale by probability
            
            ax1.errorbar(probabilities, range(len(probabilities)), 
                        xerr=uncertainties, fmt='none', color='black', alpha=0.5, capsize=3)
        
        # Add probability values
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax1.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', va='center', fontsize=10)
        
        ax1.set_xlabel('Probability')
        ax1.set_title('Intent Probabilities')
        ax1.set_xlim(0, 1)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Confidence plot
        conf_bars = ax2.barh(intent_names, confidences, color=colors, alpha=0.5)
        
        for i, (bar, conf) in enumerate(zip(conf_bars, confidences)):
            ax2.text(conf + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.3f}', va='center', fontsize=10)
        
        ax2.set_xlabel('Confidence')
        ax2.set_title('Prediction Confidence')
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_breakdown(
        self,
        predictions: List[IntentPrediction],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot breakdown of uncertainty types for intent predictions.
        
        Args:
            predictions: List of intent predictions
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Extract uncertainty data
        intent_names = [pred.intent_type.value.replace('_', ' ').title() for pred in predictions]
        
        epistemic_unc = []
        aleatoric_unc = []
        total_unc = []
        
        for pred in predictions:
            epistemic_unc.append(pred.uncertainty.get('epistemic', 0.0))
            aleatoric_unc.append(pred.uncertainty.get('aleatoric', 0.0))
            total_unc.append(pred.uncertainty.get('total', 0.0))
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        bar_width = 0.6
        indices = np.arange(len(intent_names))
        
        # Stacked bars
        p1 = ax.bar(indices, epistemic_unc, bar_width, label='Epistemic (Model)', 
                   color='#E74C3C', alpha=0.8)
        p2 = ax.bar(indices, aleatoric_unc, bar_width, bottom=epistemic_unc,
                   label='Aleatoric (Data)', color='#3498DB', alpha=0.8)
        
        # Total uncertainty line
        ax.plot(indices, total_unc, 'ko-', linewidth=2, markersize=6, label='Total')
        
        ax.set_xlabel('Intent Type')
        ax.set_ylabel('Uncertainty')
        ax.set_title('Uncertainty Breakdown by Intent Type')
        ax.set_xticks(indices)
        ax.set_xticklabels(intent_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores for each feature
            top_n: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        top_indices = sorted_indices[:top_n]
        
        top_features = [feature_names[i] for i in top_indices]
        top_scores = importance_scores[top_indices]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_scores, color='steelblue', alpha=0.7)
        
        # Add score values
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(score + 0.01 * max(top_scores), bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', fontsize=9)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_distributions(
        self,
        features_df: pd.DataFrame,
        labels: List[str],
        feature_subset: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature distributions by class label.
        
        Args:
            features_df: DataFrame with features
            labels: Class labels for each sample
            feature_subset: Subset of features to plot (default: first 9)
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        if feature_subset is None:
            feature_subset = features_df.columns[:9]  # Plot first 9 features
        
        n_features = len(feature_subset)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows // 3), dpi=self.dpi)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        # Unique labels and colors
        unique_labels = list(set(labels))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, feature in enumerate(feature_subset):
            ax = axes[i]
            
            # Plot distribution for each class
            for label, color in zip(unique_labels, colors):
                mask = np.array(labels) == label
                if np.any(mask):
                    data = features_df[feature][mask]
                    ax.hist(data, bins=20, alpha=0.6, label=label, color=color, density=True)
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution: {feature}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_training_curves(
        self,
        training_history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training curves (loss, accuracy, etc.).
        
        Args:
            training_history: Dictionary with training metrics over epochs
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        metrics = list(training_history.keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            logger.warning("No training metrics to plot")
            return plt.figure()
        
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows // 2), dpi=self.dpi)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (metric, values) in enumerate(training_history.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, 'o-', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Training {metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            # Add trend line for long training
            if len(values) > 10:
                z = np.polyfit(epochs, values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), '--', alpha=0.7, color='red')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix for classification results.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            class_names: List of class names
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        from sklearn.metrics import confusion_matrix
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Set ticks and labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_validation_results(
        self,
        validation_results: List[ValidationResult],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot data validation results summary.
        
        Args:
            validation_results: List of validation results
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        # Extract data
        quality_scores = [result.quality_score for result in validation_results]
        is_valid = [result.is_valid for result in validation_results]
        
        # Count issues
        all_issues = []
        for result in validation_results:
            all_issues.extend([issue.value for issue in result.issues])
        
        issue_counts = pd.Series(all_issues).value_counts()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1]), dpi=self.dpi)
        
        # Quality score distribution
        axes[0, 0].hist(quality_scores, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(quality_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(quality_scores):.3f}')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Quality Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Valid vs Invalid
        valid_counts = [sum(is_valid), len(is_valid) - sum(is_valid)]
        valid_labels = ['Valid', 'Invalid']
        axes[0, 1].pie(valid_counts, labels=valid_labels, autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral'])
        axes[0, 1].set_title('Data Validity')
        
        # Issue frequency
        if not issue_counts.empty:
            top_issues = issue_counts.head(10)
            axes[1, 0].barh(range(len(top_issues)), top_issues.values, color='orange', alpha=0.7)
            axes[1, 0].set_yticks(range(len(top_issues)))
            axes[1, 0].set_yticklabels(top_issues.index)
            axes[1, 0].set_xlabel('Frequency')
            axes[1, 0].set_title('Most Common Issues')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
        else:
            axes[1, 0].text(0.5, 0.5, 'No issues detected', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Issues')
        
        # Quality vs Validity scatter
        colors = ['green' if valid else 'red' for valid in is_valid]
        axes[1, 1].scatter(quality_scores, [int(v) for v in is_valid], 
                          c=colors, alpha=0.6, s=50)
        axes[1, 1].set_xlabel('Quality Score')
        axes[1, 1].set_ylabel('Is Valid')
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_yticklabels(['Invalid', 'Valid'])
        axes[1, 1].set_title('Quality vs Validity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.save_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(
        self,
        sequence: SyntheticSequence,
        predictions: List[IntentPrediction],
        features: Optional[ExtractedFeatures] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive dashboard with multiple visualizations.
        
        Args:
            sequence: Synthetic sequence data
            predictions: Intent predictions
            features: Extracted features (optional)
            save_path: Path to save the dashboard
            
        Returns:
            Plotly figure object
        """
        # Create subplot layout
        specs = [
            [{"type": "scatter3d", "rowspan": 2}, {"type": "bar"}],
            [None, {"type": "bar"}],
            [{"type": "scatter", "colspan": 2}, None]
        ]
        
        fig = make_subplots(
            rows=3, cols=2,
            specs=specs,
            subplot_titles=(
                '3D Trajectory',
                'Intent Probabilities',
                'Prediction Confidence',
                'Trajectory Time Series'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 3D trajectory
        trajectory = sequence.hand_trajectory
        color = self.gesture_colors.get(sequence.gesture_type, '#333333')
        
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=3),
                name=f'{sequence.gesture_type.value}',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Intent probabilities
        intent_names = [pred.intent_type.value.replace('_', ' ') for pred in predictions]
        probabilities = [pred.probability for pred in predictions]
        colors_bar = [self.intent_colors.get(pred.intent_type, '#333333') for pred in predictions]
        
        fig.add_trace(
            go.Bar(
                x=probabilities,
                y=intent_names,
                orientation='h',
                marker_color=colors_bar,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Prediction confidence
        confidences = [pred.confidence for pred in predictions]
        
        fig.add_trace(
            go.Bar(
                x=confidences,
                y=intent_names,
                orientation='h',
                marker_color=colors_bar,
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Time series
        timestamps = sequence.timestamps
        
        for i, dim in enumerate(['X', 'Y', 'Z']):
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=trajectory[:, i],
                    mode='lines',
                    name=f'{dim} Position',
                    line=dict(color=['red', 'green', 'blue'][i])
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'Behavior Analysis Dashboard - {sequence.gesture_type.value.capitalize()}',
            height=900,
            showlegend=True
        )
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="cube"
        )
        
        # Update bar chart axes
        fig.update_xaxes(title_text="Probability", range=[0, 1], row=1, col=2)
        fig.update_xaxes(title_text="Confidence", range=[0, 1], row=2, col=2)
        
        # Update time series axes
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Position (m)", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def save_all_plots(
        self,
        sequence: SyntheticSequence,
        predictions: List[IntentPrediction],
        features: Optional[ExtractedFeatures] = None,
        output_dir: str = "plots"
    ) -> None:
        """
        Save all visualization plots to directory.
        
        Args:
            sequence: Synthetic sequence data
            predictions: Intent predictions
            features: Extracted features (optional)
            output_dir: Output directory for plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        seq_id = sequence.sequence_id
        
        # 3D trajectory
        fig_3d = self.plot_trajectory_3d(
            sequence, save_path=str(output_path / f"{seq_id}_trajectory_3d.{self.save_format}")
        )
        plt.close(fig_3d)
        
        # Time series
        fig_ts = self.plot_trajectory_time_series(
            sequence, save_path=str(output_path / f"{seq_id}_time_series.{self.save_format}")
        )
        plt.close(fig_ts)
        
        # Intent probabilities
        if predictions:
            fig_intent = self.plot_intent_probabilities(
                predictions, save_path=str(output_path / f"{seq_id}_intent_probs.{self.save_format}")
            )
            plt.close(fig_intent)
            
            # Uncertainty breakdown
            fig_unc = self.plot_uncertainty_breakdown(
                predictions, save_path=str(output_path / f"{seq_id}_uncertainty.{self.save_format}")
            )
            plt.close(fig_unc)
        
        # Interactive dashboard
        fig_dash = self.create_dashboard(
            sequence, predictions, features,
            save_path=str(output_path / f"{seq_id}_dashboard.html")
        )
        
        logger.info(f"All plots saved to {output_dir}")


# Convenience functions for quick plotting
def quick_plot_trajectory(sequence: SyntheticSequence, save_path: Optional[str] = None) -> plt.Figure:
    """Quick 3D trajectory plot."""
    visualizer = BehaviorVisualizer()
    return visualizer.plot_trajectory_3d(sequence, save_path=save_path)


def quick_plot_intents(predictions: List[IntentPrediction], save_path: Optional[str] = None) -> plt.Figure:
    """Quick intent probability plot."""
    visualizer = BehaviorVisualizer()
    return visualizer.plot_intent_probabilities(predictions, save_path=save_path)