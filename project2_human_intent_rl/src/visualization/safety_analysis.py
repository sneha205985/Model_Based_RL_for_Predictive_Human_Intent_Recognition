"""
Safety Analysis Visualization Suite.

This module provides comprehensive visualization tools for analyzing system safety
including distance-to-human tracking, safety violations, risk assessments,
and safety constraint monitoring.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Ellipse
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime, timedelta

from .core_utils import BaseVisualizer, PlotConfig, StatisticsCalculator, ValidationUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SafetyEvent:
    """Container for safety event data."""
    
    timestamp: float
    event_type: str
    severity: str
    distance: float
    position: np.ndarray
    human_position: np.ndarray
    robot_velocity: float
    description: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class SafetyMetrics:
    """Container for safety analysis metrics."""
    
    min_distances: List[float]
    violation_count: int
    violation_duration: float
    risk_scores: List[float]
    safety_margins: List[float]
    timestamps: List[float]
    events: List[SafetyEvent] = None


class SafetyViolationType(Enum):
    """Types of safety violations."""
    DISTANCE_VIOLATION = "distance_violation"
    VELOCITY_VIOLATION = "velocity_violation"
    ACCELERATION_VIOLATION = "acceleration_violation"
    WORKSPACE_VIOLATION = "workspace_violation"
    COLLISION_RISK = "collision_risk"


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyAnalyzer(BaseVisualizer):
    """
    Comprehensive safety analysis visualization toolkit.
    
    Provides methods for:
    - Distance-to-human tracking
    - Safety violation analysis
    - Risk assessment visualization
    - Safety zone monitoring
    - Collision risk assessment
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize safety analyzer.
        
        Args:
            config: Visualization configuration
        """
        super().__init__(config)
        
        # Safety-specific color scheme
        self.safety_colors = {
            RiskLevel.LOW: '#2ECC71',
            RiskLevel.MODERATE: '#F39C12',
            RiskLevel.HIGH: '#E67E22',
            RiskLevel.CRITICAL: '#E74C3C',
            'safe': '#2ECC71',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'violation': '#C0392B'
        }
        
        logger.info("Initialized safety analyzer")
    
    def plot_distance_over_time(self,
                              distances: List[float],
                              timestamps: List[float],
                              safety_threshold: float,
                              save_path: Optional[str] = None,
                              include_violations: bool = True,
                              show_risk_zones: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Plot distance-to-human over time with safety thresholds.
        
        Args:
            distances: List of distances to human
            timestamps: Corresponding timestamps
            safety_threshold: Minimum safe distance
            save_path: Path to save the plot
            include_violations: Whether to highlight violations
            show_risk_zones: Whether to show risk zones
            
        Returns:
            Figure object
        """
        self._validate_data(distances)
        ValidationUtils.validate_array_shapes(np.array(distances), np.array(timestamps))
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_distance_plotly(distances, timestamps, safety_threshold, 
                                            save_path, include_violations, show_risk_zones)
        else:
            return self._plot_distance_matplotlib(distances, timestamps, safety_threshold,
                                                save_path, include_violations, show_risk_zones)
    
    def _plot_distance_matplotlib(self,
                                distances: List[float],
                                timestamps: List[float],
                                safety_threshold: float,
                                save_path: Optional[str],
                                include_violations: bool,
                                show_risk_zones: bool) -> plt.Figure:
        """Create distance plot using matplotlib."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.figsize[0], self.config.figsize[1]),
                                      height_ratios=[3, 1])
        
        distances = np.array(distances)
        timestamps = np.array(timestamps)
        
        # Main distance plot
        ax1.plot(timestamps, distances, 'b-', linewidth=2, label='Distance to Human')
        
        # Safety threshold line
        ax1.axhline(y=safety_threshold, color=self.safety_colors['danger'], 
                   linestyle='--', linewidth=2, label=f'Safety Threshold ({safety_threshold:.2f}m)')
        
        # Risk zones
        if show_risk_zones:
            # Critical zone (below safety threshold)
            ax1.fill_between(timestamps, 0, safety_threshold, alpha=0.3, 
                           color=self.safety_colors[RiskLevel.CRITICAL], 
                           label='Critical Zone')
            
            # Warning zone (safety threshold to 1.5x threshold)
            warning_threshold = safety_threshold * 1.5
            ax1.fill_between(timestamps, safety_threshold, warning_threshold, alpha=0.2, 
                           color=self.safety_colors[RiskLevel.HIGH], 
                           label='Warning Zone')
        
        # Highlight violations
        if include_violations:
            violations = distances < safety_threshold
            if np.any(violations):
                # Create segments for violation periods
                violation_segments = []
                violation_distances = []
                
                in_violation = False
                start_idx = 0
                
                for i, is_violation in enumerate(violations):
                    if is_violation and not in_violation:
                        start_idx = i
                        in_violation = True
                    elif not is_violation and in_violation:
                        # End of violation period
                        violation_segments.append([timestamps[start_idx], timestamps[i-1]])
                        violation_distances.append([distances[start_idx], distances[i-1]])
                        in_violation = False
                
                # Handle case where violation continues to end
                if in_violation:
                    violation_segments.append([timestamps[start_idx], timestamps[-1]])
                    violation_distances.append([distances[start_idx], distances[-1]])
                
                # Plot violation periods with different style
                for segment_times, segment_distances in zip(violation_segments, violation_distances):
                    ax1.plot(segment_times, segment_distances, 'r-', linewidth=4, alpha=0.8)
                
                # Add violation markers
                violation_times = timestamps[violations]
                violation_distances = distances[violations]
                ax1.scatter(violation_times, violation_distances, color='red', s=20, 
                           zorder=5, alpha=0.7, label='Violations')
        
        ax1.set_ylabel('Distance (m)')
        ax1.set_title('Distance to Human Over Time')
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Risk indicator subplot
        risk_levels = self._compute_risk_levels(distances, safety_threshold)
        risk_colors = [self.safety_colors[level] for level in risk_levels]
        
        # Create colored segments
        for i in range(len(timestamps)-1):
            ax2.fill_between([timestamps[i], timestamps[i+1]], 0, 1, 
                           color=risk_colors[i], alpha=0.8)
        
        ax2.set_ylabel('Risk Level')
        ax2.set_xlabel('Time (s)')
        ax2.set_yticks([0.25, 0.5, 0.75])
        ax2.set_yticklabels(['Low', 'Moderate', 'High'])
        ax2.set_title('Risk Level Indicator')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _plot_distance_plotly(self,
                            distances: List[float],
                            timestamps: List[float],
                            safety_threshold: float,
                            save_path: Optional[str],
                            include_violations: bool,
                            show_risk_zones: bool) -> go.Figure:
        """Create interactive distance plot using plotly."""
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Distance to Human Over Time', 'Risk Level Indicator'),
            vertical_spacing=0.1
        )
        
        distances = np.array(distances)
        timestamps = np.array(timestamps)
        
        # Main distance trace
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=distances,
                mode='lines',
                name='Distance to Human',
                line=dict(color='blue', width=2),
                hovertemplate='Time: %{x:.2f}s<br>Distance: %{y:.3f}m<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Safety threshold line
        fig.add_hline(
            y=safety_threshold,
            line_dash="dash",
            line_color=self.safety_colors['danger'],
            annotation_text=f"Safety Threshold ({safety_threshold:.2f}m)",
            row=1, col=1
        )
        
        # Risk zones
        if show_risk_zones:
            # Critical zone
            fig.add_hrect(
                y0=0, y1=safety_threshold,
                fillcolor=self.safety_colors[RiskLevel.CRITICAL],
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text="Critical Zone",
                row=1, col=1
            )
            
            # Warning zone
            warning_threshold = safety_threshold * 1.5
            fig.add_hrect(
                y0=safety_threshold, y1=warning_threshold,
                fillcolor=self.safety_colors[RiskLevel.HIGH],
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text="Warning Zone",
                row=1, col=1
            )
        
        # Violations
        if include_violations:
            violations = distances < safety_threshold
            if np.any(violations):
                violation_times = timestamps[violations]
                violation_distances = distances[violations]
                
                fig.add_trace(
                    go.Scatter(
                        x=violation_times,
                        y=violation_distances,
                        mode='markers',
                        name='Violations',
                        marker=dict(color='red', size=8, symbol='x'),
                        hovertemplate='Violation<br>Time: %{x:.2f}s<br>Distance: %{y:.3f}m<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Risk level indicator
        risk_levels = self._compute_risk_levels(distances, safety_threshold)
        risk_numeric = [self._risk_level_to_numeric(level) for level in risk_levels]
        risk_colors = [self.safety_colors[level] for level in risk_levels]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=risk_numeric,
                mode='lines',
                name='Risk Level',
                line=dict(color=risk_colors[0], width=10),
                fill='tonexty',
                showlegend=False,
                hovertemplate='Time: %{x:.2f}s<br>Risk: %{text}<extra></extra>',
                text=[level.value for level in risk_levels]
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Safety Analysis: Distance Monitoring',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Distance (m)", row=1, col=1)
        fig.update_yaxes(title_text="Risk Level", row=2, col=1)
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_safety_violations_analysis(self,
                                      violations: List[SafetyEvent],
                                      save_path: Optional[str] = None,
                                      group_by: str = 'type') -> Union[plt.Figure, go.Figure]:
        """
        Plot safety violations analysis.
        
        Args:
            violations: List of safety violation events
            save_path: Path to save the plot
            group_by: How to group violations ('type', 'severity', 'time')
            
        Returns:
            Figure object
        """
        self._validate_data(violations)
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_violations_plotly(violations, save_path, group_by)
        else:
            return self._plot_violations_matplotlib(violations, save_path, group_by)
    
    def _plot_violations_matplotlib(self,
                                  violations: List[SafetyEvent],
                                  save_path: Optional[str],
                                  group_by: str) -> plt.Figure:
        """Create violations analysis using matplotlib."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract violation data
        violation_types = [v.event_type for v in violations]
        severities = [v.severity for v in violations]
        timestamps = [v.timestamp for v in violations]
        distances = [v.distance for v in violations]
        
        # Violation count by type
        type_counts = pd.Series(violation_types).value_counts()
        colors = self.get_color_palette(len(type_counts), 'categorical')
        
        bars = ax1.bar(type_counts.index, type_counts.values, color=colors, alpha=0.8)
        ax1.set_title('Violations by Type')
        ax1.set_xlabel('Violation Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, type_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # Severity distribution
        severity_counts = pd.Series(severities).value_counts()
        severity_colors = [self.safety_colors.get(RiskLevel(sev), '#888888') for sev in severity_counts.index]
        
        ax2.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
               colors=severity_colors, startangle=90)
        ax2.set_title('Severity Distribution')
        
        # Violations over time
        if timestamps:
            violation_times = np.array(timestamps)
            time_bins = np.linspace(min(violation_times), max(violation_times), 20)
            counts, bins = np.histogram(violation_times, bins=time_bins)
            
            ax3.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.7, 
                   color=self.safety_colors['danger'])
            ax3.set_title('Violations Over Time')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Violation Count')
        
        # Distance at violation
        if distances:
            ax4.hist(distances, bins=15, alpha=0.7, color=self.safety_colors['warning'])
            ax4.axvline(np.mean(distances), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(distances):.3f}m')
            ax4.set_title('Distance Distribution at Violations')
            ax4.set_xlabel('Distance (m)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        
        # Apply grid to all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_risk_assessment(self,
                           risk_scores: List[float],
                           timestamps: List[float],
                           risk_factors: Optional[Dict[str, List[float]]] = None,
                           save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """
        Plot risk assessment over time.
        
        Args:
            risk_scores: Overall risk scores over time
            timestamps: Corresponding timestamps
            risk_factors: Dictionary of individual risk factor contributions
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        self._validate_data(risk_scores)
        ValidationUtils.validate_array_shapes(np.array(risk_scores), np.array(timestamps))
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_risk_plotly(risk_scores, timestamps, risk_factors, save_path)
        else:
            return self._plot_risk_matplotlib(risk_scores, timestamps, risk_factors, save_path)
    
    def _plot_risk_matplotlib(self,
                            risk_scores: List[float],
                            timestamps: List[float],
                            risk_factors: Optional[Dict[str, List[float]]],
                            save_path: Optional[str]) -> plt.Figure:
        """Create risk assessment plot using matplotlib."""
        
        if risk_factors:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.figsize[0], self.config.figsize[1]))
        else:
            fig, ax1 = plt.subplots(figsize=self.config.figsize)
            ax2 = None
        
        risk_scores = np.array(risk_scores)
        timestamps = np.array(timestamps)
        
        # Overall risk score
        ax1.plot(timestamps, risk_scores, 'b-', linewidth=2, label='Overall Risk')
        
        # Risk level zones
        ax1.fill_between(timestamps, 0, 0.3, alpha=0.2, color=self.safety_colors[RiskLevel.LOW], 
                        label='Low Risk')
        ax1.fill_between(timestamps, 0.3, 0.6, alpha=0.2, color=self.safety_colors[RiskLevel.MODERATE],
                        label='Moderate Risk')
        ax1.fill_between(timestamps, 0.6, 0.8, alpha=0.2, color=self.safety_colors[RiskLevel.HIGH],
                        label='High Risk')
        ax1.fill_between(timestamps, 0.8, 1.0, alpha=0.2, color=self.safety_colors[RiskLevel.CRITICAL],
                        label='Critical Risk')
        
        # Highlight high risk periods
        high_risk_mask = risk_scores > 0.7
        if np.any(high_risk_mask):
            ax1.fill_between(timestamps, 0, risk_scores, where=high_risk_mask, alpha=0.5,
                           color=self.safety_colors[RiskLevel.CRITICAL], interpolate=True)
        
        ax1.set_ylabel('Risk Score')
        ax1.set_title('Risk Assessment Over Time')
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        ax1.set_ylim(0, 1)
        
        # Risk factors breakdown
        if risk_factors and ax2 is not None:
            colors = self.get_color_palette(len(risk_factors), 'categorical')
            
            # Stack plot for risk factor contributions
            factor_data = np.array([risk_factors[factor] for factor in risk_factors.keys()])
            ax2.stackplot(timestamps, *factor_data, labels=list(risk_factors.keys()),
                         colors=colors, alpha=0.8)
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Risk Contribution')
            ax2.set_title('Risk Factors Breakdown')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=self.config.grid_alpha)
        
        if ax2 is None:
            ax1.set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_safety_zones_3d(self,
                            robot_positions: np.ndarray,
                            human_positions: np.ndarray,
                            safety_zones: List[Dict[str, Any]],
                            save_path: Optional[str] = None,
                            show_violations: bool = True) -> Union[plt.Figure, go.Figure]:
        """
        Plot 3D visualization of safety zones and trajectories.
        
        Args:
            robot_positions: Robot end-effector positions [N, 3]
            human_positions: Human positions [N, 3] or [3] if static
            safety_zones: List of safety zone specifications
            save_path: Path to save the plot
            show_violations: Whether to highlight zone violations
            
        Returns:
            Figure object
        """
        self._validate_data(robot_positions)
        self._validate_data(human_positions)
        
        if self.config.interactive_engine == 'plotly':
            return self._plot_safety_zones_plotly(robot_positions, human_positions, 
                                                safety_zones, save_path, show_violations)
        else:
            return self._plot_safety_zones_matplotlib(robot_positions, human_positions,
                                                    safety_zones, save_path, show_violations)
    
    def _plot_safety_zones_matplotlib(self,
                                    robot_positions: np.ndarray,
                                    human_positions: np.ndarray,
                                    safety_zones: List[Dict[str, Any]],
                                    save_path: Optional[str],
                                    show_violations: bool) -> plt.Figure:
        """Create 3D safety zones plot using matplotlib."""
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        robot_positions = np.array(robot_positions)
        human_positions = np.array(human_positions)
        
        # Ensure human_positions is 2D
        if human_positions.ndim == 1:
            human_positions = human_positions.reshape(1, -1)
        elif human_positions.ndim == 2 and human_positions.shape[0] == 1:
            # Static human position, repeat for all timesteps
            human_positions = np.repeat(human_positions, robot_positions.shape[0], axis=0)
        
        # Robot trajectory
        ax.plot(robot_positions[:, 0], robot_positions[:, 1], robot_positions[:, 2],
               'b-', linewidth=2, label='Robot Trajectory', alpha=0.8)
        
        # Start and end points
        ax.scatter(robot_positions[0, 0], robot_positions[0, 1], robot_positions[0, 2],
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(robot_positions[-1, 0], robot_positions[-1, 1], robot_positions[-1, 2],
                  c='red', s=100, marker='s', label='End')
        
        # Human trajectory/position
        if human_positions.shape[0] > 1:
            ax.plot(human_positions[:, 0], human_positions[:, 1], human_positions[:, 2],
                   'orange', linewidth=2, label='Human Trajectory', alpha=0.8)
        else:
            ax.scatter(human_positions[0, 0], human_positions[0, 1], human_positions[0, 2],
                      c='orange', s=200, marker='^', label='Human')
        
        # Safety zones
        for i, zone in enumerate(safety_zones):
            if zone['type'] == 'sphere':
                center = zone['center']
                radius = zone['radius']
                color = zone.get('color', self.safety_colors['danger'])
                
                # Draw sphere
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax.plot_surface(x, y, z, alpha=0.3, color=color)
                
            elif zone['type'] == 'cylinder':
                # Simplified cylinder representation
                center = zone['center']
                radius = zone['radius']
                height = zone.get('height', 2.0)
                color = zone.get('color', self.safety_colors['danger'])
                
                # Draw cylinder as wireframe
                theta = np.linspace(0, 2*np.pi, 20)
                z_cyl = np.linspace(center[2] - height/2, center[2] + height/2, 10)
                theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
                x_cyl = center[0] + radius * np.cos(theta_mesh)
                y_cyl = center[1] + radius * np.sin(theta_mesh)
                
                ax.plot_wireframe(x_cyl, y_cyl, z_mesh, alpha=0.3, color=color)
        
        # Highlight violations
        if show_violations:
            for zone in safety_zones:
                if zone['type'] == 'sphere':
                    center = zone['center']
                    radius = zone['radius']
                    
                    # Check which robot positions violate this zone
                    distances = np.linalg.norm(robot_positions - center, axis=1)
                    violations = distances < radius
                    
                    if np.any(violations):
                        violation_positions = robot_positions[violations]
                        ax.scatter(violation_positions[:, 0], violation_positions[:, 1], 
                                 violation_positions[:, 2], c='red', s=50, marker='x',
                                 alpha=0.8, label=f'Zone {i+1} Violations')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Safety Zones Visualization')
        ax.legend()
        
        # Equal aspect ratio
        max_range = 0.5 * np.max([
            robot_positions.max() - robot_positions.min(),
            human_positions.max() - human_positions.min()
        ])
        
        mid_x = 0.5 * (robot_positions[:, 0].max() + robot_positions[:, 0].min())
        mid_y = 0.5 * (robot_positions[:, 1].max() + robot_positions[:, 1].min())
        mid_z = 0.5 * (robot_positions[:, 2].max() + robot_positions[:, 2].min())
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _compute_risk_levels(self, distances: np.ndarray, safety_threshold: float) -> List[RiskLevel]:
        """Compute risk levels based on distances."""
        risk_levels = []
        
        for distance in distances:
            if distance < safety_threshold:
                risk_levels.append(RiskLevel.CRITICAL)
            elif distance < safety_threshold * 1.2:
                risk_levels.append(RiskLevel.HIGH)
            elif distance < safety_threshold * 1.5:
                risk_levels.append(RiskLevel.MODERATE)
            else:
                risk_levels.append(RiskLevel.LOW)
        
        return risk_levels
    
    def _risk_level_to_numeric(self, risk_level: RiskLevel) -> float:
        """Convert risk level to numeric value for plotting."""
        mapping = {
            RiskLevel.LOW: 0.25,
            RiskLevel.MODERATE: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 1.0
        }
        return mapping[risk_level]
    
    def plot(self, *args, **kwargs) -> Union[plt.Figure, go.Figure]:
        """Main plot method - delegates to appropriate visualization."""
        if 'plot_type' in kwargs:
            plot_type = kwargs.pop('plot_type')
            
            if plot_type == 'distance':
                return self.plot_distance_over_time(*args, **kwargs)
            elif plot_type == 'violations':
                return self.plot_safety_violations_analysis(*args, **kwargs)
            elif plot_type == 'risk':
                return self.plot_risk_assessment(*args, **kwargs)
            elif plot_type == 'safety_zones_3d':
                return self.plot_safety_zones_3d(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
        else:
            return self.plot_distance_over_time(*args, **kwargs)


logger.info("Safety analysis visualization suite loaded successfully")