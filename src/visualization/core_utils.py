"""
Core visualization utilities and base classes.

This module provides the foundation for all visualization components including:
- Base visualization classes
- Configuration management
- Color schemes and styling
- Common plotting utilities
- Error handling and validation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod
import warnings
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PlotType(Enum):
    """Enumeration of supported plot types."""
    STATIC_2D = "static_2d"
    STATIC_3D = "static_3d"
    INTERACTIVE = "interactive"
    ANIMATED = "animated"
    DASHBOARD = "dashboard"


class ColorScheme(Enum):
    """Predefined color schemes."""
    DEFAULT = "default"
    PUBLICATION = "publication"
    COLORBLIND = "colorblind"
    HIGH_CONTRAST = "high_contrast"
    PASTEL = "pastel"
    SCIENTIFIC = "scientific"


@dataclass
class PlotConfig:
    """Configuration for plot appearance and behavior."""
    
    # Figure settings
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    facecolor: str = 'white'
    edgecolor: str = 'black'
    
    # Style settings
    style: str = 'seaborn-v0_8-darkgrid'
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    font_family: str = 'DejaVu Sans'
    font_size: int = 12
    title_font_size: int = 14
    label_font_size: int = 11
    
    # Layout settings
    tight_layout: bool = True
    grid: bool = True
    grid_alpha: float = 0.3
    legend: bool = True
    legend_loc: str = 'best'
    
    # Export settings
    save_format: str = 'png'
    bbox_inches: str = 'tight'
    transparent: bool = False
    
    # Interactive settings
    interactive_engine: str = 'plotly'
    show_toolbar: bool = True
    
    # Animation settings
    animation_interval: int = 100
    animation_repeat: bool = True
    
    # Publication settings
    publication_ready: bool = False
    include_metadata: bool = True


@dataclass
class ColorPalette:
    """Color palette for consistent visualization."""
    
    # Primary colors
    primary: str = '#2E86C1'
    secondary: str = '#28B463'
    accent: str = '#F39C12'
    warning: str = '#E74C3C'
    info: str = '#8E44AD'
    success: str = '#27AE60'
    danger: str = '#C0392B'
    
    # Gradient colors
    gradient_start: str = '#3498DB'
    gradient_end: str = '#E74C3C'
    
    # Categorical colors
    categorical: List[str] = field(default_factory=lambda: [
        '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
        '#1ABC9C', '#34495E', '#E67E22', '#95A5A6', '#16A085'
    ])
    
    # Sequential colors
    sequential: List[str] = field(default_factory=lambda: [
        '#FEF9E7', '#FCF3CF', '#F9E79F', '#F7DC6F', '#F4D03F',
        '#F1C40F', '#D4AC0D', '#B7950B', '#9A7D0A', '#7D6608'
    ])
    
    # Diverging colors
    diverging: List[str] = field(default_factory=lambda: [
        '#D73027', '#F46D43', '#FDAE61', '#FEE08B', '#FFFFBF',
        '#E6F598', '#ABDDA4', '#66C2A5', '#3288BD', '#5E4FA2'
    ])


class VisualizationError(Exception):
    """Custom exception for visualization-related errors."""
    pass


class BaseVisualizer(ABC):
    """
    Abstract base class for all visualizers.
    
    Provides common functionality including:
    - Configuration management
    - Error handling
    - Styling
    - Save/export functionality
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize base visualizer.
        
        Args:
            config: Plot configuration object
        """
        self.config = config or PlotConfig()
        self.palette = ColorPalette()
        self._setup_style()
        
        logger.debug(f"Initialized {self.__class__.__name__} visualizer")
    
    def _setup_style(self) -> None:
        """Setup matplotlib style and seaborn settings."""
        try:
            plt.style.use(self.config.style)
        except OSError:
            logger.warning(f"Style '{self.config.style}' not found, using default")
            plt.style.use('default')
        
        # Configure matplotlib
        plt.rcParams.update({
            'figure.figsize': self.config.figsize,
            'figure.dpi': self.config.dpi,
            'figure.facecolor': self.config.facecolor,
            'figure.edgecolor': self.config.edgecolor,
            'font.family': self.config.font_family,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_font_size,
            'axes.labelsize': self.config.label_font_size,
            'xtick.labelsize': self.config.label_font_size,
            'ytick.labelsize': self.config.label_font_size,
            'legend.fontsize': self.config.label_font_size,
            'axes.grid': self.config.grid,
            'grid.alpha': self.config.grid_alpha,
        })
        
        # Configure seaborn
        sns.set_palette(self.palette.categorical)
    
    def _validate_data(self, data: Any, expected_shape: Optional[Tuple] = None) -> None:
        """
        Validate input data.
        
        Args:
            data: Data to validate
            expected_shape: Expected shape (optional)
            
        Raises:
            VisualizationError: If data validation fails
        """
        if data is None:
            raise VisualizationError("Input data cannot be None")
        
        if isinstance(data, (np.ndarray, list)):
            data_array = np.array(data)
            if data_array.size == 0:
                raise VisualizationError("Input data cannot be empty")
            
            if expected_shape and data_array.shape != expected_shape:
                raise VisualizationError(
                    f"Data shape {data_array.shape} does not match expected {expected_shape}"
                )
        
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise VisualizationError("Input DataFrame cannot be empty")
    
    def _create_figure(self, subplot_spec: Optional[Dict] = None) -> Union[plt.Figure, go.Figure]:
        """
        Create figure based on plot type.
        
        Args:
            subplot_spec: Subplot specification for complex layouts
            
        Returns:
            Figure object
        """
        if self.config.interactive_engine == 'plotly':
            if subplot_spec:
                return make_subplots(**subplot_spec)
            else:
                return go.Figure()
        else:
            if subplot_spec:
                return plt.subplots(**subplot_spec)
            else:
                return plt.figure(
                    figsize=self.config.figsize,
                    dpi=self.config.dpi,
                    facecolor=self.config.facecolor,
                    edgecolor=self.config.edgecolor
                )
    
    def _apply_styling(self, ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
        """
        Apply consistent styling to matplotlib axes.
        
        Args:
            ax: Matplotlib axes object
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if title:
            ax.set_title(title, fontsize=self.config.title_font_size)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.config.label_font_size)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.config.label_font_size)
        
        ax.grid(self.config.grid, alpha=self.config.grid_alpha)
        
        if self.config.legend:
            ax.legend(loc=self.config.legend_loc)
        
        if self.config.tight_layout:
            plt.tight_layout()
    
    def _save_figure(self, fig: Union[plt.Figure, go.Figure], save_path: str, **kwargs) -> None:
        """
        Save figure to file with appropriate format.
        
        Args:
            fig: Figure object to save
            save_path: Output file path
            **kwargs: Additional save parameters
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(fig, go.Figure):
            # Plotly figure
            if save_path.suffix.lower() == '.html':
                fig.write_html(str(save_path))
            elif save_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
                fig.write_image(str(save_path), **kwargs)
            else:
                fig.write_html(str(save_path.with_suffix('.html')))
        else:
            # Matplotlib figure
            fig.savefig(
                save_path,
                format=self.config.save_format,
                dpi=self.config.dpi,
                bbox_inches=self.config.bbox_inches,
                transparent=self.config.transparent,
                facecolor=self.config.facecolor,
                edgecolor=self.config.edgecolor,
                **kwargs
            )
        
        logger.info(f"Saved figure to {save_path}")
    
    def get_color_palette(self, n_colors: int, palette_type: str = 'categorical') -> List[str]:
        """
        Get color palette with specified number of colors.
        
        Args:
            n_colors: Number of colors needed
            palette_type: Type of palette ('categorical', 'sequential', 'diverging')
            
        Returns:
            List of color strings
        """
        if palette_type == 'categorical':
            colors = self.palette.categorical
        elif palette_type == 'sequential':
            colors = self.palette.sequential
        elif palette_type == 'diverging':
            colors = self.palette.diverging
        else:
            colors = self.palette.categorical
        
        if n_colors <= len(colors):
            return colors[:n_colors]
        else:
            # Generate additional colors if needed
            return colors + sns.color_palette("husl", n_colors - len(colors)).as_hex()
    
    def create_custom_colormap(self, colors: List[str], name: str = "custom") -> LinearSegmentedColormap:
        """
        Create custom colormap from color list.
        
        Args:
            colors: List of color strings
            name: Colormap name
            
        Returns:
            Custom colormap
        """
        return LinearSegmentedColormap.from_list(name, colors)
    
    @abstractmethod
    def plot(self, *args, **kwargs) -> Union[plt.Figure, go.Figure]:
        """
        Abstract plot method to be implemented by subclasses.
        """
        pass


class StatisticsCalculator:
    """Utility class for statistical calculations used in visualizations."""
    
    @staticmethod
    def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for data.
        
        Args:
            data: Input data array
            confidence: Confidence level (default 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy import stats
        
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        
        return mean - h, mean + h
    
    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray, 
                                    statistic: Callable = np.mean,
                                    confidence: float = 0.95,
                                    n_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Input data array
            statistic: Statistic function to apply
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    @staticmethod
    def effect_size_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1: First group data
            group2: Second group data
            
        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std


class PlotAnnotator:
    """Utility class for adding annotations and statistical information to plots."""
    
    @staticmethod
    def add_significance_bars(ax: plt.Axes, 
                            x_positions: List[float], 
                            y_position: float,
                            significance_levels: List[str],
                            bar_height: float = 0.02) -> None:
        """
        Add significance bars to plot.
        
        Args:
            ax: Matplotlib axes object
            x_positions: X positions for bars
            y_position: Y position for bars
            significance_levels: Significance level labels
            bar_height: Height of significance bars
        """
        for i in range(len(x_positions) - 1):
            x1, x2 = x_positions[i], x_positions[i + 1]
            
            # Draw horizontal line
            ax.plot([x1, x2], [y_position, y_position], 'k-', linewidth=1)
            
            # Draw vertical lines
            ax.plot([x1, x1], [y_position - bar_height/2, y_position + bar_height/2], 'k-', linewidth=1)
            ax.plot([x2, x2], [y_position - bar_height/2, y_position + bar_height/2], 'k-', linewidth=1)
            
            # Add significance text
            if i < len(significance_levels):
                ax.text((x1 + x2) / 2, y_position + bar_height, significance_levels[i],
                       ha='center', va='bottom', fontsize=10)
    
    @staticmethod
    def add_statistical_summary(ax: plt.Axes, 
                              data: np.ndarray,
                              x_position: float,
                              include_outliers: bool = True) -> None:
        """
        Add statistical summary box to plot.
        
        Args:
            ax: Matplotlib axes object
            data: Data for statistics
            x_position: X position for summary box
            include_outliers: Whether to include outlier information
        """
        stats_text = []
        
        # Basic statistics
        stats_text.append(f"Mean: {np.mean(data):.3f}")
        stats_text.append(f"Median: {np.median(data):.3f}")
        stats_text.append(f"Std: {np.std(data):.3f}")
        stats_text.append(f"N: {len(data)}")
        
        # Quartiles
        q25, q75 = np.percentile(data, [25, 75])
        stats_text.append(f"Q1-Q3: {q25:.3f}-{q75:.3f}")
        
        # Outliers
        if include_outliers:
            iqr = q75 - q25
            lower_fence = q25 - 1.5 * iqr
            upper_fence = q75 + 1.5 * iqr
            outliers = data[(data < lower_fence) | (data > upper_fence)]
            stats_text.append(f"Outliers: {len(outliers)}")
        
        # Create text box
        text = '\n'.join(stats_text)
        
        # Position the text box
        ax.text(x_position, 0.95, text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)


class ValidationUtils:
    """Utility functions for data validation in visualizations."""
    
    @staticmethod
    def validate_array_shapes(*arrays: np.ndarray) -> None:
        """
        Validate that arrays have compatible shapes.
        
        Args:
            *arrays: Variable number of arrays to validate
            
        Raises:
            VisualizationError: If arrays have incompatible shapes
        """
        if len(arrays) < 2:
            return
        
        first_shape = arrays[0].shape[0]
        for i, array in enumerate(arrays[1:], 1):
            if array.shape[0] != first_shape:
                raise VisualizationError(
                    f"Array {i} has shape {array.shape[0]} but expected {first_shape}"
                )
    
    @staticmethod
    def validate_finite_data(data: np.ndarray, name: str = "data") -> None:
        """
        Validate that data contains only finite values.
        
        Args:
            data: Data array to validate
            name: Name of the data for error messages
            
        Raises:
            VisualizationError: If data contains non-finite values
        """
        if not np.all(np.isfinite(data)):
            non_finite_count = np.sum(~np.isfinite(data))
            raise VisualizationError(
                f"{name} contains {non_finite_count} non-finite values"
            )
    
    @staticmethod
    def validate_probability_data(data: np.ndarray, name: str = "probabilities") -> None:
        """
        Validate that data represents valid probabilities.
        
        Args:
            data: Probability data to validate
            name: Name of the data for error messages
            
        Raises:
            VisualizationError: If data is not valid probabilities
        """
        if np.any(data < 0) or np.any(data > 1):
            raise VisualizationError(
                f"{name} must be between 0 and 1"
            )


# Convenience functions for common operations
def create_publication_config() -> PlotConfig:
    """Create configuration optimized for publication-ready plots."""
    return PlotConfig(
        figsize=(10, 6),
        dpi=300,
        style='classic',
        color_scheme=ColorScheme.PUBLICATION,
        font_family='Times New Roman',
        font_size=12,
        title_font_size=14,
        publication_ready=True,
        save_format='pdf',
        transparent=False
    )


def create_presentation_config() -> PlotConfig:
    """Create configuration optimized for presentations."""
    return PlotConfig(
        figsize=(12, 8),
        dpi=150,
        style='seaborn-v0_8-talk',
        color_scheme=ColorScheme.HIGH_CONTRAST,
        font_size=14,
        title_font_size=18,
        save_format='png'
    )


def create_web_config() -> PlotConfig:
    """Create configuration optimized for web display."""
    return PlotConfig(
        figsize=(10, 6),
        dpi=96,
        style='seaborn-v0_8-whitegrid',
        color_scheme=ColorScheme.DEFAULT,
        interactive_engine='plotly',
        save_format='html'
    )


def setup_matplotlib_for_notebook() -> None:
    """Setup matplotlib for optimal notebook display."""
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')
    except NameError:
        # Not in a notebook environment
        pass


def create_color_palette_preview(palette: ColorPalette, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a preview of the color palette.
    
    Args:
        palette: Color palette to preview
        save_path: Optional save path
        
    Returns:
        Figure showing color palette
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Categorical colors
    axes[0, 0].bar(range(len(palette.categorical)), [1] * len(palette.categorical), 
                  color=palette.categorical)
    axes[0, 0].set_title('Categorical Colors')
    axes[0, 0].set_xticks(range(len(palette.categorical)))
    
    # Sequential colors
    axes[0, 1].bar(range(len(palette.sequential)), [1] * len(palette.sequential), 
                  color=palette.sequential)
    axes[0, 1].set_title('Sequential Colors')
    
    # Diverging colors
    axes[1, 0].bar(range(len(palette.diverging)), [1] * len(palette.diverging), 
                  color=palette.diverging)
    axes[1, 0].set_title('Diverging Colors')
    
    # Primary colors
    primary_colors = [palette.primary, palette.secondary, palette.accent, 
                     palette.warning, palette.info, palette.success, palette.danger]
    primary_labels = ['Primary', 'Secondary', 'Accent', 'Warning', 'Info', 'Success', 'Danger']
    
    axes[1, 1].bar(range(len(primary_colors)), [1] * len(primary_colors), 
                  color=primary_colors)
    axes[1, 1].set_title('Primary Colors')
    axes[1, 1].set_xticks(range(len(primary_colors)))
    axes[1, 1].set_xticklabels(primary_labels, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


logger.info("Core visualization utilities loaded successfully")