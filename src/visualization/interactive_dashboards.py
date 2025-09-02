"""
Interactive Dashboards with Plotly.

This module provides comprehensive interactive dashboard capabilities for
real-time data exploration, parameter sensitivity analysis, and system monitoring.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly.graph_objs import Figure
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import time
from datetime import datetime, timedelta
import queue

from .core_utils import BaseVisualizer, PlotConfig, ColorPalette
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DashboardData:
    """Container for dashboard data."""
    
    timestamps: List[float] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class DashboardConfig:
    """Configuration for dashboard appearance and behavior."""
    
    # Layout settings
    title: str = "Model-Based RL System Dashboard"
    theme: str = "bootstrap"
    update_interval: int = 1000  # milliseconds
    max_data_points: int = 1000
    
    # Component settings
    show_controls: bool = True
    show_statistics: bool = True
    show_export: bool = True
    
    # Real-time settings
    enable_real_time: bool = True
    data_buffer_size: int = 10000
    
    # Styling
    height: int = 800
    color_scheme: str = "plotly"


class DashboardComponent(Enum):
    """Types of dashboard components."""
    TIME_SERIES = "time_series"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    GAUGE = "gauge"
    INDICATOR = "indicator"
    TABLE = "table"
    CONTROL_PANEL = "control_panel"


class InteractiveDashboard(BaseVisualizer):
    """
    Comprehensive interactive dashboard system.
    
    Provides:
    - Real-time data visualization
    - Parameter sensitivity analysis
    - Interactive exploration tools
    - Export capabilities
    - Responsive design
    """
    
    def __init__(self, config: Optional[PlotConfig] = None, 
                 dashboard_config: Optional[DashboardConfig] = None):
        """
        Initialize interactive dashboard.
        
        Args:
            config: Visualization configuration
            dashboard_config: Dashboard-specific configuration
        """
        super().__init__(config)
        self.dashboard_config = dashboard_config or DashboardConfig()
        
        # Dashboard data storage
        self.data_buffer = DashboardData()
        self.data_queue = queue.Queue(maxsize=self.dashboard_config.data_buffer_size)
        
        # Dashboard app
        self.app = None
        self.server_thread = None
        self.is_running = False
        
        logger.info("Initialized interactive dashboard system")
    
    def create_comprehensive_dashboard(self,
                                     layout_config: Dict[str, Any],
                                     save_path: Optional[str] = None) -> Figure:
        """
        Create comprehensive multi-panel dashboard.
        
        Args:
            layout_config: Configuration for dashboard layout and components
            save_path: Path to save dashboard HTML
            
        Returns:
            Plotly figure object
        """
        # Parse layout configuration
        components = layout_config.get('components', [])
        layout_type = layout_config.get('layout', 'grid')
        
        if layout_type == 'grid':
            return self._create_grid_dashboard(components, save_path)
        elif layout_type == 'tabs':
            return self._create_tabbed_dashboard(components, save_path)
        else:
            raise ValueError(f"Unsupported layout type: {layout_type}")
    
    def _create_grid_dashboard(self,
                             components: List[Dict[str, Any]],
                             save_path: Optional[str]) -> Figure:
        """Create grid-based dashboard layout."""
        
        n_components = len(components)
        n_cols = min(3, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols
        
        # Create subplot specifications
        subplot_specs = []
        subplot_titles = []
        
        for i in range(n_rows):
            row_specs = []
            for j in range(n_cols):
                comp_idx = i * n_cols + j
                if comp_idx < n_components:
                    component = components[comp_idx]
                    comp_type = component['type']
                    
                    if comp_type in ['time_series', 'scatter', 'bar_chart']:
                        spec = {"secondary_y": component.get('secondary_y', False)}\n                    elif comp_type == 'heatmap':
                        spec = {"type": "heatmap"}
                    elif comp_type in ['histogram', 'box_plot']:
                        spec = {}
                    elif comp_type == '3d_scatter':
                        spec = {"type": "scatter3d"}
                    else:
                        spec = {}
                    
                    row_specs.append(spec)
                    subplot_titles.append(component.get('title', f'Component {comp_idx+1}'))
                else:
                    row_specs.append(None)
            subplot_specs.append(row_specs)
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            specs=subplot_specs,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Add components to subplots
        for i, component in enumerate(components):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            self._add_component_to_subplot(fig, component, row, col)
        
        # Update layout
        fig.update_layout(
            title=self.dashboard_config.title,
            height=self.dashboard_config.height,
            showlegend=True,
            template=self.dashboard_config.color_scheme
        )
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _add_component_to_subplot(self,
                                fig: Figure,
                                component: Dict[str, Any],
                                row: int,
                                col: int) -> None:
        """Add a component to a subplot."""
        
        comp_type = component['type']
        data = component.get('data', {})
        style = component.get('style', {})
        
        if comp_type == 'time_series':
            self._add_time_series_component(fig, data, style, row, col)
        elif comp_type == 'scatter':
            self._add_scatter_component(fig, data, style, row, col)
        elif comp_type == 'histogram':
            self._add_histogram_component(fig, data, style, row, col)
        elif comp_type == 'heatmap':
            self._add_heatmap_component(fig, data, style, row, col)
        elif comp_type == 'bar_chart':
            self._add_bar_chart_component(fig, data, style, row, col)
        elif comp_type == 'gauge':
            self._add_gauge_component(fig, data, style, row, col)
        else:
            logger.warning(f"Unknown component type: {comp_type}")
    
    def _add_time_series_component(self,
                                 fig: Figure,
                                 data: Dict[str, Any],
                                 style: Dict[str, Any],
                                 row: int,
                                 col: int) -> None:
        """Add time series component to subplot."""
        
        x_data = data.get('x', [])
        y_series = data.get('y', {})
        
        colors = self.get_color_palette(len(y_series), 'categorical')
        
        for i, (series_name, y_data) in enumerate(y_series.items()):
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers',
                    name=series_name,
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4),
                    showlegend=True
                ),
                row=row, col=col
            )
        
        # Add uncertainty bands if available
        if 'uncertainty' in data:
            uncertainty = data['uncertainty']
            for series_name in y_series.keys():
                if series_name in uncertainty:
                    y_data = y_series[series_name]
                    y_lower = np.array(y_data) - np.array(uncertainty[series_name])
                    y_upper = np.array(y_data) + np.array(uncertainty[series_name])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_data + x_data[::-1],
                            y=list(y_upper) + list(y_lower[::-1]),
                            fill='toself',
                            fillcolor=colors[i % len(colors)],
                            opacity=0.2,
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            name=f'{series_name} Uncertainty'
                        ),
                        row=row, col=col
                    )
    
    def _add_scatter_component(self,
                             fig: Figure,
                             data: Dict[str, Any],
                             style: Dict[str, Any],
                             row: int,
                             col: int) -> None:
        """Add scatter plot component to subplot."""
        
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        color_data = data.get('color', None)
        size_data = data.get('size', None)
        
        marker_dict = dict(size=8)
        if color_data is not None:
            marker_dict['color'] = color_data
            marker_dict['colorscale'] = 'Viridis'
            marker_dict['showscale'] = True
        if size_data is not None:
            marker_dict['size'] = size_data
        
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker=marker_dict,
                name=data.get('name', 'Scatter'),
                text=data.get('hover_text', None),
                hovertemplate=data.get('hovertemplate', None)
            ),
            row=row, col=col
        )
    
    def _add_histogram_component(self,
                               fig: Figure,
                               data: Dict[str, Any],
                               style: Dict[str, Any],
                               row: int,
                               col: int) -> None:
        """Add histogram component to subplot."""
        
        x_data = data.get('x', [])
        
        fig.add_trace(
            go.Histogram(
                x=x_data,
                nbinsx=data.get('bins', 30),
                name=data.get('name', 'Histogram'),
                opacity=0.8,
                marker_color=style.get('color', 'blue')
            ),
            row=row, col=col
        )
    
    def _add_heatmap_component(self,
                             fig: Figure,
                             data: Dict[str, Any],
                             style: Dict[str, Any],
                             row: int,
                             col: int) -> None:
        """Add heatmap component to subplot."""
        
        z_data = data.get('z', [])
        x_labels = data.get('x_labels', None)
        y_labels = data.get('y_labels', None)
        
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=x_labels,
                y=y_labels,
                colorscale=style.get('colorscale', 'Viridis'),
                showscale=True
            ),
            row=row, col=col
        )
    
    def _add_bar_chart_component(self,
                               fig: Figure,
                               data: Dict[str, Any],
                               style: Dict[str, Any],
                               row: int,
                               col: int) -> None:
        """Add bar chart component to subplot."""
        
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        
        fig.add_trace(
            go.Bar(
                x=x_data,
                y=y_data,
                name=data.get('name', 'Bar Chart'),
                marker_color=style.get('color', 'blue'),
                text=y_data,
                textposition='outside'
            ),
            row=row, col=col
        )
    
    def _add_gauge_component(self,
                           fig: Figure,
                           data: Dict[str, Any],
                           style: Dict[str, Any],
                           row: int,
                           col: int) -> None:
        """Add gauge component to subplot."""
        
        value = data.get('value', 0)
        min_val = data.get('min', 0)
        max_val = data.get('max', 100)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': data.get('title', 'Gauge')},
                delta={'reference': data.get('reference', 0)},
                gauge={
                    'axis': {'range': [min_val, max_val]},
                    'bar': {'color': style.get('color', 'darkblue')},
                    'steps': [
                        {'range': [min_val, max_val * 0.5], 'color': 'lightgray'},
                        {'range': [max_val * 0.5, max_val], 'color': 'gray'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': data.get('threshold', max_val * 0.9)
                    }
                }
            ),
            row=row, col=col
        )
    
    def create_parameter_sensitivity_dashboard(self,
                                             parameter_ranges: Dict[str, Tuple[float, float]],
                                             sensitivity_function: Callable,
                                             save_path: Optional[str] = None) -> Figure:
        """
        Create parameter sensitivity analysis dashboard.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to (min, max) ranges
            sensitivity_function: Function that computes metrics given parameter values
            save_path: Path to save dashboard
            
        Returns:
            Interactive sensitivity dashboard
        """
        # Create parameter grid
        n_params = len(parameter_ranges)
        param_names = list(parameter_ranges.keys())
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=n_params,
            subplot_titles=[f'{param} Sensitivity' for param in param_names] + 
                          [f'{param} Distribution' for param in param_names],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        colors = self.get_color_palette(n_params, 'categorical')
        
        for i, (param_name, param_range) in enumerate(parameter_ranges.items()):
            # Generate parameter values
            param_values = np.linspace(param_range[0], param_range[1], 50)
            
            # Compute sensitivity (simplified - would need actual implementation)
            # This is a placeholder that would call the sensitivity_function
            metric_values = []
            for param_val in param_values:
                # Create parameter dict with current parameter varied
                params = {name: (ranges[0] + ranges[1]) / 2 
                         for name, ranges in parameter_ranges.items()}
                params[param_name] = param_val
                
                # Compute metric (placeholder)
                try:
                    metric = sensitivity_function(params)
                    metric_values.append(metric)
                except Exception as e:
                    logger.warning(f"Error computing sensitivity for {param_name}={param_val}: {e}")
                    metric_values.append(np.nan)
            
            # Sensitivity curve
            fig.add_trace(
                go.Scatter(
                    x=param_values,
                    y=metric_values,
                    mode='lines+markers',
                    name=f'{param_name} Sensitivity',
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=6)
                ),
                row=1, col=i+1
            )
            
            # Parameter distribution (histogram)
            # Generate sample parameter values (normally distributed around center)
            center = (param_range[0] + param_range[1]) / 2
            spread = (param_range[1] - param_range[0]) / 6  # 3-sigma rule
            sample_values = np.random.normal(center, spread, 1000)
            sample_values = np.clip(sample_values, param_range[0], param_range[1])
            
            fig.add_trace(
                go.Histogram(
                    x=sample_values,
                    nbinsx=30,
                    name=f'{param_name} Distribution',
                    marker_color=colors[i],
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=i+1
            )
            
            # Update axes labels
            fig.update_xaxes(title_text=param_name, row=1, col=i+1)
            fig.update_yaxes(title_text='Metric Value', row=1, col=i+1)
            fig.update_xaxes(title_text=param_name, row=2, col=i+1)
            fig.update_yaxes(title_text='Frequency', row=2, col=i+1)
        
        # Update layout
        fig.update_layout(
            title='Parameter Sensitivity Analysis Dashboard',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Add interactivity with sliders (would require Dash for full implementation)
        sliders = []
        for i, (param_name, param_range) in enumerate(parameter_ranges.items()):
            slider = dict(
                active=25,  # Middle value
                currentvalue={"prefix": f"{param_name}: "},
                pad={"t": 50},
                steps=[
                    dict(
                        label=f"{val:.2f}",
                        method="restyle",
                        args=[{"x": [np.linspace(param_range[0], param_range[1], 50)]}]
                    )
                    for val in np.linspace(param_range[0], param_range[1], 50)
                ]
            )
            sliders.append(slider)
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_real_time_dashboard(self,
                                 data_callback: Callable[[], Dict[str, Any]],
                                 update_interval: int = 1000) -> dash.Dash:
        """
        Create real-time monitoring dashboard using Dash.
        
        Args:
            data_callback: Function that returns current data
            update_interval: Update interval in milliseconds
            
        Returns:
            Dash app instance
        """
        # Initialize Dash app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Define layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Real-Time System Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Controls", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Update Interval (ms):"),
                                    dbc.Input(
                                        id="update-interval",
                                        type="number",
                                        value=update_interval,
                                        min=100,
                                        max=10000,
                                        step=100
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Max Data Points:"),
                                    dbc.Input(
                                        id="max-points",
                                        type="number",
                                        value=1000,
                                        min=100,
                                        max=10000,
                                        step=100
                                    )
                                ], width=6)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Start/Stop", id="toggle-button", 
                                             color="primary", className="me-2"),
                                    dbc.Button("Reset", id="reset-button", 
                                             color="secondary", className="me-2"),
                                    dbc.Button("Export", id="export-button", 
                                             color="success")
                                ])
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Main dashboard
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="main-time-series", style={'height': '400px'})
                ], width=8),
                dbc.Col([
                    dcc.Graph(id="metric-gauges", style={'height': '400px'})
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="histogram-plot", style={'height': '350px'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="correlation-heatmap", style={'height': '350px'})
                ], width=6)
            ], className="mb-4"),
            
            # Status and statistics
            dbc.Row([
                dbc.Col([
                    html.Div(id="status-info")
                ])
            ]),
            
            # Hidden div to store data
            html.Div(id="data-store", style={'display': 'none'}),
            
            # Interval component for updates
            dcc.Interval(
                id="interval-component",
                interval=update_interval,
                n_intervals=0,
                disabled=False
            )
        ], fluid=True)
        
        # Define callbacks
        @app.callback(
            [Output("main-time-series", "figure"),
             Output("metric-gauges", "figure"),
             Output("histogram-plot", "figure"),
             Output("correlation-heatmap", "figure"),
             Output("status-info", "children"),
             Output("data-store", "children")],
            [Input("interval-component", "n_intervals")],
            [State("max-points", "value")]
        )
        def update_dashboard(n_intervals, max_points):
            # Get fresh data
            try:
                current_data = data_callback()
                
                # Update data buffer
                self._update_data_buffer(current_data, max_points)
                
                # Create visualizations
                time_series_fig = self._create_real_time_time_series()
                gauges_fig = self._create_real_time_gauges(current_data)
                histogram_fig = self._create_real_time_histogram()
                heatmap_fig = self._create_real_time_heatmap()
                
                # Status information
                status_info = self._create_status_info(current_data)
                
                # Store data (serialized)
                data_store = json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'n_points': len(self.data_buffer.timestamps),
                    'metrics': list(self.data_buffer.metrics.keys())
                })
                
                return (time_series_fig, gauges_fig, histogram_fig, 
                       heatmap_fig, status_info, data_store)
                
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                return ({}, {}, {}, {}, html.Div(f"Error: {e}"), "{}")
        
        @app.callback(
            Output("interval-component", "disabled"),
            [Input("toggle-button", "n_clicks")],
            [State("interval-component", "disabled")]
        )
        def toggle_updates(n_clicks, is_disabled):
            if n_clicks:
                return not is_disabled
            return is_disabled
        
        @app.callback(
            Output("interval-component", "interval"),
            [Input("update-interval", "value")]
        )
        def update_interval_setting(interval_value):
            return interval_value or 1000
        
        self.app = app
        return app
    
    def _update_data_buffer(self, new_data: Dict[str, Any], max_points: int) -> None:
        """Update the data buffer with new data."""
        current_time = time.time()
        
        # Add timestamp
        self.data_buffer.timestamps.append(current_time)
        
        # Add metrics
        for metric_name, metric_value in new_data.items():
            if metric_name not in self.data_buffer.metrics:
                self.data_buffer.metrics[metric_name] = []
            self.data_buffer.metrics[metric_name].append(metric_value)
        
        # Trim data if necessary
        if len(self.data_buffer.timestamps) > max_points:
            excess = len(self.data_buffer.timestamps) - max_points
            self.data_buffer.timestamps = self.data_buffer.timestamps[excess:]
            for metric_name in self.data_buffer.metrics:
                self.data_buffer.metrics[metric_name] = \
                    self.data_buffer.metrics[metric_name][excess:]
        
        self.data_buffer.last_updated = datetime.now()
    
    def _create_real_time_time_series(self) -> Figure:
        """Create real-time time series plot."""
        fig = go.Figure()
        
        colors = self.get_color_palette(len(self.data_buffer.metrics), 'categorical')
        
        for i, (metric_name, values) in enumerate(self.data_buffer.metrics.items()):
            fig.add_trace(
                go.Scatter(
                    x=self.data_buffer.timestamps,
                    y=values,
                    mode='lines+markers',
                    name=metric_name,
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4)
                )
            )
        
        fig.update_layout(
            title="Real-Time Metrics",
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_real_time_gauges(self, current_data: Dict[str, Any]) -> Figure:
        """Create gauge plots for current metric values."""
        n_metrics = min(4, len(current_data))  # Limit to 4 gauges
        
        if n_metrics == 0:
            return go.Figure()
        
        # Create subplot layout for gauges
        specs = [[{"type": "indicator"}] * min(2, n_metrics)]
        if n_metrics > 2:
            specs.append([{"type": "indicator"}] * min(2, n_metrics - 2))
        
        rows = len(specs)
        cols = len(specs[0])
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            subplot_titles=list(current_data.keys())[:n_metrics]
        )
        
        metric_items = list(current_data.items())[:n_metrics]
        
        for i, (metric_name, current_value) in enumerate(metric_items):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Estimate range from historical data
            if metric_name in self.data_buffer.metrics:
                historical_values = self.data_buffer.metrics[metric_name]
                if historical_values:
                    min_val = min(historical_values) * 0.8
                    max_val = max(historical_values) * 1.2
                else:
                    min_val, max_val = 0, 100
            else:
                min_val, max_val = 0, 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=current_value,
                    title={'text': metric_name},
                    gauge={
                        'axis': {'range': [min_val, max_val]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [min_val, max_val * 0.5], 'color': 'lightgray'},
                            {'range': [max_val * 0.5, max_val], 'color': 'gray'}
                        ]
                    }
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=400, title="Current Metrics")
        
        return fig
    
    def _create_real_time_histogram(self) -> Figure:
        """Create histogram of recent metric values."""
        if not self.data_buffer.metrics:
            return go.Figure()
        
        # Use the first metric for histogram
        metric_name = list(self.data_buffer.metrics.keys())[0]
        values = self.data_buffer.metrics[metric_name]
        
        if not values:
            return go.Figure()
        
        fig = go.Figure(data=[
            go.Histogram(
                x=values,
                nbinsx=30,
                marker_color='blue',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title=f"Distribution: {metric_name}",
            xaxis_title=metric_name,
            yaxis_title="Frequency",
            template='plotly_white'
        )
        
        return fig
    
    def _create_real_time_heatmap(self) -> Figure:
        """Create correlation heatmap of metrics."""
        if len(self.data_buffer.metrics) < 2:
            return go.Figure()
        
        # Create correlation matrix
        df = pd.DataFrame(self.data_buffer.metrics)
        correlation_matrix = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Metric Correlations",
            template='plotly_white'
        )
        
        return fig
    
    def _create_status_info(self, current_data: Dict[str, Any]) -> html.Div:
        """Create status information display."""
        
        status_cards = []
        
        # Data status
        status_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Data Status", className="card-title"),
                    html.P([
                        f"Data Points: {len(self.data_buffer.timestamps)}",
                        html.Br(),
                        f"Metrics: {len(self.data_buffer.metrics)}",
                        html.Br(),
                        f"Last Update: {self.data_buffer.last_updated.strftime('%H:%M:%S')}"
                    ])
                ])
            ])
        )
        
        # Current values
        if current_data:
            current_values_text = []
            for metric, value in current_data.items():
                current_values_text.extend([
                    f"{metric}: {value:.3f}",
                    html.Br()
                ])
            
            status_cards.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Current Values", className="card-title"),
                        html.P(current_values_text)
                    ])
                ])
            )
        
        return dbc.Row([
            dbc.Col(card, width=6) for card in status_cards
        ])
    
    def start_server(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
        """
        Start the dashboard server.
        
        Args:
            host: Host address
            port: Port number
            debug: Enable debug mode
        """
        if self.app is None:
            raise ValueError("No dashboard app created. Call create_real_time_dashboard first.")
        
        def run_server():
            self.app.run_server(host=host, port=port, debug=debug, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        
        logger.info(f"Dashboard server started at http://{host}:{port}")
    
    def stop_server(self) -> None:
        """Stop the dashboard server."""
        self.is_running = False
        if self.server_thread:
            self.server_thread.join(timeout=1)
        
        logger.info("Dashboard server stopped")
    
    def plot(self, *args, **kwargs) -> Figure:
        """Main plot method - delegates to appropriate dashboard creation."""
        dashboard_type = kwargs.pop('dashboard_type', 'comprehensive')
        
        if dashboard_type == 'comprehensive':
            return self.create_comprehensive_dashboard(*args, **kwargs)
        elif dashboard_type == 'sensitivity':
            return self.create_parameter_sensitivity_dashboard(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported dashboard type: {dashboard_type}")


logger.info("Interactive dashboards module loaded successfully")