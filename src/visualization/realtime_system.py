"""
Real-Time Visualization System.

This module provides comprehensive real-time visualization capabilities for
live system monitoring, streaming data visualization, and dynamic updates.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import queue
import collections
from datetime import datetime, timedelta
import json
import asyncio
import websockets

from .core_utils import BaseVisualizer, PlotConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RealTimeData:
    """Container for real-time data streams."""
    
    timestamp: float
    robot_state: Dict[str, Any] = field(default_factory=dict)
    human_state: Dict[str, Any] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)
    safety_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    system_status: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamConfig:
    """Configuration for real-time data streams."""
    
    buffer_size: int = 1000
    update_rate: float = 10.0  # Hz
    max_display_points: int = 500
    enable_smoothing: bool = True
    smoothing_window: int = 5
    auto_scale: bool = True
    time_window: float = 60.0  # seconds


class UpdateStrategy(Enum):
    """Strategies for updating real-time visualizations."""
    REPLACE = "replace"
    APPEND = "append"
    SLIDING_WINDOW = "sliding_window"
    TRIGGERED = "triggered"


class RealTimeVisualizer(BaseVisualizer):
    """
    Comprehensive real-time visualization system.
    
    Provides:
    - Live system dashboard
    - Streaming data plots
    - Dynamic safety monitoring
    - Performance tracking
    - WebSocket integration
    """
    
    def __init__(self, config: Optional[PlotConfig] = None, 
                 stream_config: Optional[StreamConfig] = None):
        """
        Initialize real-time visualizer.
        
        Args:
            config: Visualization configuration
            stream_config: Stream-specific configuration
        """
        super().__init__(config)
        self.stream_config = stream_config or StreamConfig()
        
        # Data storage
        self.data_buffer = collections.deque(maxlen=self.stream_config.buffer_size)
        self.data_queue = queue.Queue()
        
        # Animation and threading
        self.animation = None
        self.update_thread = None
        self.websocket_server = None
        self.is_running = False
        
        # Figure and axes storage
        self.fig = None
        self.axes = {}
        
        logger.info("Initialized real-time visualization system")
    
    def create_live_dashboard(self,
                            layout: Dict[str, Any],
                            data_callback: Optional[Callable[[], RealTimeData]] = None) -> plt.Figure:
        """
        Create live monitoring dashboard.
        
        Args:
            layout: Dashboard layout configuration
            data_callback: Function to get current data
            
        Returns:
            Matplotlib figure with animation
        """
        # Create figure and subplots
        self.fig, self.axes = self._create_dashboard_layout(layout)
        
        # Initialize plots
        self._initialize_dashboard_plots(layout)
        
        # Set up animation
        if data_callback:
            self.animation = animation.FuncAnimation(
                self.fig,
                lambda frame: self._update_dashboard(data_callback()),
                interval=int(1000 / self.stream_config.update_rate),
                blit=False,
                cache_frame_data=False
            )
        
        return self.fig
    
    def _create_dashboard_layout(self, layout: Dict[str, Any]) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """Create dashboard layout based on configuration."""
        
        layout_type = layout.get('type', 'grid')
        components = layout.get('components', [])
        
        if layout_type == 'grid':
            n_components = len(components)
            n_cols = layout.get('cols', min(3, n_components))
            n_rows = (n_components + n_cols - 1) // n_cols
            
            fig, axes_array = plt.subplots(
                n_rows, n_cols,
                figsize=layout.get('figsize', (15, 10)),
                constrained_layout=True
            )
            
            # Flatten axes array for easier access
            if n_components == 1:
                axes_array = [axes_array]
            elif n_rows == 1:
                axes_array = axes_array
            else:
                axes_array = axes_array.flatten()
            
            # Create axes dictionary
            axes = {}
            for i, component in enumerate(components):
                if i < len(axes_array):
                    axes[component['name']] = axes_array[i]
                    axes_array[i].set_title(component.get('title', component['name']))
            
            # Hide unused subplots
            for i in range(len(components), len(axes_array)):
                axes_array[i].set_visible(False)
        
        else:
            raise ValueError(f"Unsupported layout type: {layout_type}")
        
        return fig, axes
    
    def _initialize_dashboard_plots(self, layout: Dict[str, Any]) -> None:
        """Initialize dashboard plots with empty data."""
        
        components = layout.get('components', [])
        
        for component in components:
            name = component['name']
            plot_type = component['type']
            
            if name not in self.axes:
                continue
            
            ax = self.axes[name]
            
            if plot_type == 'time_series':
                # Initialize empty line plots
                metrics = component.get('metrics', ['default'])
                colors = self.get_color_palette(len(metrics), 'categorical')
                
                for i, metric in enumerate(metrics):
                    ax.plot([], [], color=colors[i], label=metric, linewidth=2)
                
                ax.legend()
                ax.grid(True, alpha=self.config.grid_alpha)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(component.get('ylabel', 'Value'))
                
            elif plot_type == 'gauge':
                # Initialize gauge plot
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.set_aspect('equal')
                ax.axis('off')
                
                # Draw gauge background
                circle = Circle((0, 0), 1, fill=False, color='gray', linewidth=2)
                ax.add_patch(circle)
                
                # Add gauge ticks
                for angle in np.linspace(0, np.pi, 6):
                    x1, y1 = 0.9 * np.cos(angle), 0.9 * np.sin(angle)
                    x2, y2 = np.cos(angle), np.sin(angle)
                    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
                
            elif plot_type == 'trajectory_2d':
                # Initialize 2D trajectory plot
                ax.set_aspect('equal')
                ax.grid(True, alpha=self.config.grid_alpha)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                
                # Initialize empty trajectory line
                ax.plot([], [], 'b-', linewidth=2, label='Robot Trajectory')
                ax.plot([], [], 'ro', markersize=8, label='Current Position')
                ax.legend()
                
            elif plot_type == 'safety_zones':
                # Initialize safety zone visualization
                ax.set_aspect('equal')
                ax.grid(True, alpha=self.config.grid_alpha)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
    
    def _update_dashboard(self, data: RealTimeData) -> None:
        """Update dashboard with new data."""
        
        # Add data to buffer
        self.data_buffer.append(data)
        
        # Update each component
        for name, ax in self.axes.items():
            try:
                self._update_component(name, ax, data)
            except Exception as e:
                logger.warning(f"Error updating component {name}: {e}")
    
    def _update_component(self, name: str, ax: plt.Axes, data: RealTimeData) -> None:
        """Update a specific dashboard component."""
        
        # Get recent data for time series
        recent_data = list(self.data_buffer)
        if not recent_data:
            return
        
        timestamps = [d.timestamp for d in recent_data]
        
        # Update based on component type
        if 'time_series' in name.lower():
            self._update_time_series(ax, recent_data, timestamps)
            
        elif 'gauge' in name.lower():
            self._update_gauge(ax, data)
            
        elif 'trajectory' in name.lower():
            self._update_trajectory(ax, recent_data)
            
        elif 'safety' in name.lower():
            self._update_safety_zones(ax, data)
        
        # Auto-scale if enabled
        if self.stream_config.auto_scale:
            ax.relim()
            ax.autoscale_view(True, True, True)
    
    def _update_time_series(self, ax: plt.Axes, data_history: List[RealTimeData], 
                           timestamps: List[float]) -> None:
        """Update time series plot."""
        
        lines = ax.get_lines()
        if not lines:
            return
        
        # Extract metrics from data history
        metrics_data = {}
        for data_point in data_history:
            for category in ['performance_metrics', 'safety_metrics', 'system_status']:
                if hasattr(data_point, category):
                    category_data = getattr(data_point, category)
                    for metric_name, value in category_data.items():
                        if metric_name not in metrics_data:
                            metrics_data[metric_name] = []
                        metrics_data[metric_name].append(value)
        
        # Update line data
        for i, line in enumerate(lines):
            metric_name = line.get_label()
            if metric_name in metrics_data:
                values = metrics_data[metric_name]
                
                # Apply smoothing if enabled
                if self.stream_config.enable_smoothing and len(values) > self.stream_config.smoothing_window:
                    smoothed = np.convolve(
                        values, 
                        np.ones(self.stream_config.smoothing_window) / self.stream_config.smoothing_window,
                        mode='valid'
                    )
                    smooth_timestamps = timestamps[self.stream_config.smoothing_window-1:]
                    line.set_data(smooth_timestamps, smoothed)
                else:
                    line.set_data(timestamps, values)
        
        # Update time window
        if timestamps:
            current_time = timestamps[-1]
            time_window = self.stream_config.time_window
            ax.set_xlim(current_time - time_window, current_time)
    
    def _update_gauge(self, ax: plt.Axes, data: RealTimeData) -> None:
        """Update gauge display."""
        
        # Clear previous gauge needle
        for line in ax.get_lines():
            if line.get_label() == 'needle':
                line.remove()
        
        # Get gauge value (example: use first performance metric)
        gauge_value = 0.5  # Default
        if data.performance_metrics:
            first_metric = list(data.performance_metrics.values())[0]
            gauge_value = np.clip(first_metric / 100.0, 0, 1)  # Normalize to 0-1
        
        # Draw needle
        angle = np.pi * gauge_value  # 0 to pi radians
        needle_x = [0, 0.8 * np.cos(angle)]
        needle_y = [0, 0.8 * np.sin(angle)]
        
        ax.plot(needle_x, needle_y, 'r-', linewidth=4, label='needle')
        
        # Add value text
        ax.text(0, -0.3, f'{gauge_value:.2f}', ha='center', va='center', 
                fontsize=16, fontweight='bold')
    
    def _update_trajectory(self, ax: plt.Axes, data_history: List[RealTimeData]) -> None:
        """Update trajectory plot."""
        
        lines = ax.get_lines()
        if not lines:
            return
        
        # Extract robot positions
        positions = []
        for data_point in data_history:
            if 'position' in data_point.robot_state:
                pos = data_point.robot_state['position']
                if len(pos) >= 2:
                    positions.append(pos[:2])  # Take X, Y coordinates
        
        if not positions:
            return
        
        positions = np.array(positions)
        
        # Update trajectory line
        if len(lines) > 0:
            lines[0].set_data(positions[:, 0], positions[:, 1])
        
        # Update current position marker
        if len(lines) > 1 and len(positions) > 0:
            lines[1].set_data([positions[-1, 0]], [positions[-1, 1]])
    
    def _update_safety_zones(self, ax: plt.Axes, data: RealTimeData) -> None:
        """Update safety zones visualization."""
        
        # Clear previous patches
        for patch in ax.patches[:]:
            patch.remove()
        
        # Add safety zones
        if 'safety_zones' in data.safety_metrics:
            zones = data.safety_metrics['safety_zones']
            for zone in zones:
                if zone['type'] == 'circle':
                    circle = Circle(
                        zone['center'][:2], 
                        zone['radius'],
                        fill=False,
                        color='red' if zone.get('violated', False) else 'orange',
                        linewidth=2,
                        alpha=0.7
                    )
                    ax.add_patch(circle)
                elif zone['type'] == 'rectangle':
                    rect = Rectangle(
                        zone['corner'][:2],
                        zone['width'],
                        zone['height'],
                        fill=False,
                        color='red' if zone.get('violated', False) else 'orange',
                        linewidth=2,
                        alpha=0.7
                    )
                    ax.add_patch(rect)
        
        # Add robot and human positions
        if 'position' in data.robot_state:
            robot_pos = data.robot_state['position'][:2]
            ax.plot(robot_pos[0], robot_pos[1], 'bo', markersize=10, label='Robot')
        
        if 'position' in data.human_state:
            human_pos = data.human_state['position'][:2]
            ax.plot(human_pos[0], human_pos[1], 'ro', markersize=10, label='Human')
    
    def create_streaming_plot(self,
                            metrics: List[str],
                            data_source: Callable[[], Dict[str, float]],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create streaming plot for specified metrics.
        
        Args:
            metrics: List of metric names to plot
            data_source: Function that returns current metric values
            save_path: Optional path to save animation
            
        Returns:
            Matplotlib figure with animation
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Initialize data storage
        time_data = collections.deque(maxlen=self.stream_config.max_display_points)
        metric_data = {metric: collections.deque(maxlen=self.stream_config.max_display_points) 
                      for metric in metrics}
        
        # Initialize lines
        colors = self.get_color_palette(len(metrics), 'categorical')
        lines = {}
        for i, metric in enumerate(metrics):
            line, = ax.plot([], [], color=colors[i], label=metric, linewidth=2)
            lines[metric] = line
        
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.set_title('Real-Time Metrics Stream')
        
        def animate(frame):
            try:
                # Get new data
                current_data = data_source()
                current_time = time.time()
                
                time_data.append(current_time)
                
                # Update metric data
                for metric in metrics:
                    value = current_data.get(metric, 0)
                    metric_data[metric].append(value)
                
                # Update plots
                for metric in metrics:
                    if metric in lines:
                        lines[metric].set_data(list(time_data), list(metric_data[metric]))
                
                # Auto-scale
                if self.stream_config.auto_scale and time_data:
                    time_window = self.stream_config.time_window
                    current_time = max(time_data)
                    ax.set_xlim(current_time - time_window, current_time)
                    
                    # Y-axis scaling
                    all_values = []
                    for values in metric_data.values():
                        all_values.extend(values)
                    
                    if all_values:
                        y_min, y_max = min(all_values), max(all_values)
                        y_range = y_max - y_min
                        if y_range > 0:
                            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                
                return list(lines.values())
                
            except Exception as e:
                logger.error(f"Error in streaming plot animation: {e}")
                return []
        
        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            interval=int(1000 / self.stream_config.update_rate),
            blit=True,
            cache_frame_data=False
        )
        
        if save_path:
            writer = animation.PillowWriter(fps=self.stream_config.update_rate)
            anim.save(save_path, writer=writer)
        
        return fig
    
    def start_websocket_server(self,
                              host: str = "localhost",
                              port: int = 8765) -> None:
        """
        Start WebSocket server for real-time data streaming.
        
        Args:
            host: Server host address
            port: Server port
        """
        async def handle_client(websocket, path):
            logger.info(f"Client connected: {websocket.remote_address}")
            
            try:
                while True:
                    # Get data from queue (non-blocking)
                    try:
                        data = self.data_queue.get_nowait()
                        # Send data as JSON
                        await websocket.send(json.dumps(data, default=str))
                    except queue.Empty:
                        pass
                    
                    # Small delay to prevent overwhelming clients
                    await asyncio.sleep(1.0 / self.stream_config.update_rate)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client disconnected: {websocket.remote_address}")
            except Exception as e:
                logger.error(f"Error in WebSocket handler: {e}")
        
        async def start_server():
            self.websocket_server = await websockets.serve(handle_client, host, port)
            logger.info(f"WebSocket server started on {host}:{port}")
            await self.websocket_server.wait_closed()
        
        # Run server in separate thread
        def run_server():
            asyncio.run(start_server())
        
        self.update_thread = threading.Thread(target=run_server, daemon=True)
        self.update_thread.start()
        self.is_running = True
    
    def add_data(self, data: Union[RealTimeData, Dict[str, Any]]) -> None:
        """
        Add data to the real-time system.
        
        Args:
            data: Data to add (RealTimeData object or dictionary)
        """
        if isinstance(data, dict):
            # Convert dict to RealTimeData
            rt_data = RealTimeData(
                timestamp=data.get('timestamp', time.time()),
                robot_state=data.get('robot_state', {}),
                human_state=data.get('human_state', {}),
                predictions=data.get('predictions', {}),
                safety_metrics=data.get('safety_metrics', {}),
                performance_metrics=data.get('performance_metrics', {}),
                system_status=data.get('system_status', {})
            )
        else:
            rt_data = data
        
        # Add to buffer and queue
        self.data_buffer.append(rt_data)
        
        # Add to queue for WebSocket clients (non-blocking)
        try:
            self.data_queue.put_nowait(rt_data.__dict__)
        except queue.Full:
            # Remove oldest item and try again
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(rt_data.__dict__)
            except queue.Empty:
                pass
    
    def stop_visualization(self) -> None:
        """Stop real-time visualization."""
        self.is_running = False
        
        if self.animation:
            self.animation.event_source.stop()
        
        if self.websocket_server:
            self.websocket_server.close()
        
        if self.update_thread:
            self.update_thread.join(timeout=1)
        
        logger.info("Real-time visualization stopped")
    
    def export_data(self, filepath: str, format: str = 'json') -> None:
        """
        Export buffered data to file.
        
        Args:
            filepath: Output file path
            format: Export format ('json', 'csv', 'hdf5')
        """
        data_list = [data.__dict__ for data in self.data_buffer]
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data_list, f, indent=2, default=str)
        
        elif format == 'csv':
            # Flatten nested dictionaries for CSV export
            flattened_data = []
            for data in data_list:
                flat_data = {'timestamp': data['timestamp']}
                
                for category in ['robot_state', 'human_state', 'predictions', 
                               'safety_metrics', 'performance_metrics', 'system_status']:
                    if category in data:
                        for key, value in data[category].items():
                            flat_data[f'{category}_{key}'] = value
                
                flattened_data.append(flat_data)
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(filepath, index=False)
        
        elif format == 'hdf5':
            df = pd.DataFrame(data_list)
            df.to_hdf(filepath, key='realtime_data', mode='w')
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(data_list)} data points to {filepath}")
    
    def plot(self, *args, **kwargs) -> plt.Figure:
        """Main plot method - delegates to appropriate visualization."""
        plot_type = kwargs.pop('plot_type', 'dashboard')
        
        if plot_type == 'dashboard':
            return self.create_live_dashboard(*args, **kwargs)
        elif plot_type == 'streaming':
            return self.create_streaming_plot(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")


class RealTimeDataGenerator:
    """Utility class for generating synthetic real-time data for testing."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.start_time = time.time()
        self.iteration = 0
    
    def generate_sample_data(self) -> RealTimeData:
        """Generate synthetic real-time data."""
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Robot state
        robot_state = {
            'position': [
                2 * np.sin(0.1 * elapsed) + np.random.normal(0, 0.05),
                2 * np.cos(0.1 * elapsed) + np.random.normal(0, 0.05),
                1.0 + 0.5 * np.sin(0.2 * elapsed)
            ],
            'velocity': np.random.normal(0, 0.1, 3).tolist(),
            'joint_angles': np.random.uniform(-np.pi, np.pi, 6).tolist()
        }
        
        # Human state
        human_state = {
            'position': [
                1 + np.random.normal(0, 0.1),
                1 + np.random.normal(0, 0.1),
                0.0
            ],
            'velocity': np.random.normal(0, 0.05, 3).tolist()
        }
        
        # Predictions
        predictions = {
            'intent_probabilities': {
                'reach': max(0, min(1, 0.5 + 0.3 * np.sin(0.05 * elapsed) + np.random.normal(0, 0.1))),
                'handover': max(0, min(1, 0.3 + 0.2 * np.cos(0.08 * elapsed) + np.random.normal(0, 0.1))),
                'idle': max(0, min(1, 0.2 + np.random.normal(0, 0.05)))
            },
            'uncertainty': max(0, min(1, 0.1 + 0.05 * np.random.random()))
        }
        
        # Safety metrics
        distance = np.linalg.norm(
            np.array(robot_state['position']) - np.array(human_state['position'])
        )
        
        safety_metrics = {
            'human_robot_distance': distance,
            'safety_violations': int(distance < 0.5),
            'risk_score': max(0, min(1, 0.8 / (distance + 0.1))),
            'safety_zones': [
                {
                    'type': 'circle',
                    'center': human_state['position'],
                    'radius': 0.5,
                    'violated': distance < 0.5
                }
            ]
        }
        
        # Performance metrics
        performance_metrics = {
            'success_rate': max(0, min(1, 0.85 + 0.1 * np.sin(0.02 * elapsed) + np.random.normal(0, 0.02))),
            'completion_time': max(0, 10 + 2 * np.sin(0.03 * elapsed) + np.random.normal(0, 0.5)),
            'efficiency': max(0, min(1, 0.9 + 0.05 * np.cos(0.04 * elapsed) + np.random.normal(0, 0.01))),
            'mpc_solve_time': max(0, 0.01 + 0.005 * np.random.random())
        }
        
        # System status
        system_status = {
            'cpu_usage': max(0, min(100, 45 + 10 * np.sin(0.1 * elapsed) + np.random.normal(0, 2))),
            'memory_usage': max(0, min(100, 60 + 5 * np.cos(0.15 * elapsed) + np.random.normal(0, 1))),
            'temperature': max(0, 35 + 5 * np.sin(0.05 * elapsed) + np.random.normal(0, 0.5))
        }
        
        self.iteration += 1
        
        return RealTimeData(
            timestamp=current_time,
            robot_state=robot_state,
            human_state=human_state,
            predictions=predictions,
            safety_metrics=safety_metrics,
            performance_metrics=performance_metrics,
            system_status=system_status
        )


logger.info("Real-time visualization system loaded successfully")