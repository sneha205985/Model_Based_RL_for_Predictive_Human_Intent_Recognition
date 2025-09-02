"""
Visualization tools for Model Predictive Control.

This module provides comprehensive visualization capabilities for MPC
including trajectory plots, constraint satisfaction monitoring, cost
function evolution, and real-time control visualization.

The visualizations help in:
- Understanding MPC behavior and performance
- Debugging controller issues
- Analyzing safety constraint satisfaction
- Monitoring real-time performance
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from ..control.mpc_controller import MPCResult, MPCStatus
from ..control.hri_mpc import HRIMPCController, InteractionPhase
from ..models.robotics.robot_dynamics import Robot6DOF, RobotState
from ..control.safety_constraints import SafetyConstraint, SafetyMonitor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MPCVisualizer:
    """
    Comprehensive visualization toolkit for MPC analysis.
    
    Provides static and interactive visualizations for:
    - 3D robot trajectories
    - Joint space trajectories  
    - Control effort analysis
    - Constraint satisfaction monitoring
    - Cost function evolution
    - Safety zone visualization
    - Performance benchmarking
    """
    
    def __init__(self, 
                 robot_model: Optional[Robot6DOF] = None,
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize MPC visualizer.
        
        Args:
            robot_model: Robot model for kinematic visualization
            figsize: Default figure size for matplotlib plots
        """
        self.robot_model = robot_model
        self.figsize = figsize
        
        # Plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info("Initialized MPC visualizer")
    
    def plot_3d_trajectory(self,
                          predicted_states: np.ndarray,
                          reference_trajectory: Optional[np.ndarray] = None,
                          human_position: Optional[np.ndarray] = None,
                          safety_zones: Optional[List[Dict]] = None,
                          save_path: Optional[str] = None,
                          interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Plot 3D end-effector trajectory with safety zones.
        
        Args:
            predicted_states: Predicted state trajectory (N x state_dim)
            reference_trajectory: Reference trajectory (N x state_dim)
            human_position: Human position [x, y, z] if available
            safety_zones: List of safety zone specifications
            save_path: Path to save figure
            interactive: Use Plotly for interactive visualization
        
        Returns:
            Figure object (matplotlib or plotly)
        """
        if self.robot_model is None:
            logger.error("Robot model required for trajectory plotting")
            return None
        
        # Compute end-effector positions from joint states
        ee_positions = []
        for state in predicted_states:
            joint_pos = state[0:6] if len(state) >= 6 else state
            ee_pos, _ = self.robot_model.forward_kinematics(joint_pos)
            ee_positions.append(ee_pos)
        ee_positions = np.array(ee_positions)
        
        if interactive:
            return self._plot_3d_trajectory_plotly(
                ee_positions, reference_trajectory, human_position, 
                safety_zones, save_path
            )
        else:
            return self._plot_3d_trajectory_matplotlib(
                ee_positions, reference_trajectory, human_position,
                safety_zones, save_path
            )
    
    def _plot_3d_trajectory_matplotlib(self,
                                     ee_positions: np.ndarray,
                                     reference_trajectory: Optional[np.ndarray],
                                     human_position: Optional[np.ndarray],
                                     safety_zones: Optional[List[Dict]],
                                     save_path: Optional[str]) -> plt.Figure:
        """Create 3D trajectory plot using matplotlib."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Robot trajectory
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                'b-', linewidth=2, label='Robot Trajectory')
        ax.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2],
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2],
                  c='red', s=100, marker='s', label='End')
        
        # Reference trajectory
        if reference_trajectory is not None:
            # Compute reference end-effector positions
            ref_ee_positions = []
            for state in reference_trajectory:
                joint_pos = state[0:6] if len(state) >= 6 else state
                try:
                    ee_pos, _ = self.robot_model.forward_kinematics(joint_pos)
                    ref_ee_positions.append(ee_pos)
                except:
                    continue
            
            if ref_ee_positions:
                ref_ee_positions = np.array(ref_ee_positions)
                ax.plot(ref_ee_positions[:, 0], ref_ee_positions[:, 1], ref_ee_positions[:, 2],
                       'r--', linewidth=1, alpha=0.7, label='Reference')
        
        # Human position
        if human_position is not None:
            ax.scatter(human_position[0], human_position[1], human_position[2],
                      c='orange', s=200, marker='^', label='Human')
        
        # Safety zones
        if safety_zones:
            for zone in safety_zones:
                if zone.get('type') == 'sphere' and 'center' in zone and 'radius' in zone:
                    self._draw_sphere(ax, zone['center'], zone['radius'], 
                                    alpha=0.2, color='red', label='Safety Zone')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot End-Effector Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        max_range = np.array([
            ee_positions[:, 0].max() - ee_positions[:, 0].min(),
            ee_positions[:, 1].max() - ee_positions[:, 1].min(),
            ee_positions[:, 2].max() - ee_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (ee_positions[:, 0].max() + ee_positions[:, 0].min()) * 0.5
        mid_y = (ee_positions[:, 1].max() + ee_positions[:, 1].min()) * 0.5
        mid_z = (ee_positions[:, 2].max() + ee_positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 3D trajectory plot to {save_path}")
        
        return fig
    
    def _plot_3d_trajectory_plotly(self,
                                 ee_positions: np.ndarray,
                                 reference_trajectory: Optional[np.ndarray],
                                 human_position: Optional[np.ndarray],
                                 safety_zones: Optional[List[Dict]],
                                 save_path: Optional[str]) -> go.Figure:
        """Create interactive 3D trajectory plot using Plotly."""
        fig = go.Figure()
        
        # Robot trajectory
        fig.add_trace(go.Scatter3d(
            x=ee_positions[:, 0],
            y=ee_positions[:, 1],
            z=ee_positions[:, 2],
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=3),
            name='Robot Trajectory'
        ))
        
        # Start and end points
        fig.add_trace(go.Scatter3d(
            x=[ee_positions[0, 0]],
            y=[ee_positions[0, 1]],
            z=[ee_positions[0, 2]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='circle'),
            name='Start'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[ee_positions[-1, 0]],
            y=[ee_positions[-1, 1]],
            z=[ee_positions[-1, 2]],
            mode='markers',
            marker=dict(color='red', size=10, symbol='square'),
            name='End'
        ))
        
        # Human position
        if human_position is not None:
            fig.add_trace(go.Scatter3d(
                x=[human_position[0]],
                y=[human_position[1]],
                z=[human_position[2]],
                mode='markers',
                marker=dict(color='orange', size=15, symbol='diamond'),
                name='Human'
            ))
        
        # Safety zones (spheres)
        if safety_zones:
            for i, zone in enumerate(safety_zones):
                if zone.get('type') == 'sphere':
                    center = zone['center']
                    radius = zone['radius']
                    
                    # Create sphere mesh
                    phi, theta = np.mgrid[0:np.pi:20j, 0:2*np.pi:20j]
                    x_sphere = center[0] + radius * np.sin(phi) * np.cos(theta)
                    y_sphere = center[1] + radius * np.sin(phi) * np.sin(theta)
                    z_sphere = center[2] + radius * np.cos(phi)
                    
                    fig.add_trace(go.Surface(
                        x=x_sphere, y=y_sphere, z=z_sphere,
                        opacity=0.3,
                        colorscale='Reds',
                        showscale=False,
                        name=f'Safety Zone {i+1}'
                    ))
        
        fig.update_layout(
            title='Robot End-Effector Trajectory (Interactive)',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive 3D trajectory plot to {save_path}")
        
        return fig
    
    def plot_joint_trajectories(self,
                              predicted_states: np.ndarray,
                              reference_states: Optional[np.ndarray] = None,
                              joint_limits: Optional[Dict] = None,
                              sampling_time: float = 0.1,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot joint position and velocity trajectories.
        
        Args:
            predicted_states: Predicted states (N x 12)
            reference_states: Reference states (N x 12)  
            joint_limits: Joint limits dictionary
            sampling_time: Time step between samples
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        n_steps = len(predicted_states)
        time_vector = np.arange(n_steps) * sampling_time
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        joint_names = [f'Joint {i+1}' for i in range(6)]
        
        # Joint positions
        for i in range(6):
            ax = axes[i]
            
            # Predicted trajectory
            ax.plot(time_vector, predicted_states[:, i], 'b-', linewidth=2,
                   label='Predicted')
            
            # Reference trajectory
            if reference_states is not None:
                ax.plot(time_vector, reference_states[:, i], 'r--', linewidth=1,
                       label='Reference')
            
            # Joint limits
            if joint_limits and 'position_min' in joint_limits:
                ax.axhline(joint_limits['position_min'][i], color='red',
                          linestyle=':', alpha=0.7, label='Limits')
                ax.axhline(joint_limits['position_max'][i], color='red',
                          linestyle=':', alpha=0.7)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (rad)')
            ax.set_title(f'{joint_names[i]} Position')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved joint trajectories plot to {save_path}")
        
        return fig
    
    def plot_control_effort(self,
                           control_sequence: np.ndarray,
                           control_limits: Optional[np.ndarray] = None,
                           sampling_time: float = 0.1,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot control effort over time.
        
        Args:
            control_sequence: Control commands (N x control_dim)
            control_limits: Control limits (control_dim,)
            sampling_time: Time step between samples
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        n_steps, n_controls = control_sequence.shape
        time_vector = np.arange(n_steps) * sampling_time
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        control_names = [f'Control {i+1}' for i in range(n_controls)]
        
        for i in range(min(6, n_controls)):  # Plot up to 6 controls
            ax = axes[i]
            
            ax.plot(time_vector, control_sequence[:, i], 'b-', linewidth=2)
            
            # Control limits
            if control_limits is not None:
                ax.axhline(control_limits[i], color='red', linestyle='--',
                          alpha=0.7, label='Limit')
                ax.axhline(-control_limits[i], color='red', linestyle='--',
                          alpha=0.7)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Control Input')
            ax.set_title(f'{control_names[i]} Effort')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved control effort plot to {save_path}")
        
        return fig
    
    def plot_constraint_satisfaction(self,
                                   constraint_history: List[Dict],
                                   time_vector: Optional[np.ndarray] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot constraint satisfaction over time.
        
        Args:
            constraint_history: List of constraint evaluation results
            time_vector: Time vector (optional)
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        if not constraint_history:
            logger.warning("No constraint history provided")
            return plt.figure()
        
        if time_vector is None:
            time_vector = np.arange(len(constraint_history))
        
        # Organize data by constraint type
        constraint_data = {}
        for i, constraints in enumerate(constraint_history):
            for constraint_name, value in constraints.items():
                if constraint_name not in constraint_data:
                    constraint_data[constraint_name] = []
                constraint_data[constraint_name].append((time_vector[i], value))
        
        n_constraints = len(constraint_data)
        n_cols = min(3, n_constraints)
        n_rows = (n_constraints + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_constraints == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (constraint_name, data) in enumerate(constraint_data.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            times, values = zip(*data)
            
            ax.plot(times, values, 'b-', linewidth=2)
            ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='Constraint Boundary')
            ax.fill_between(times, values, 0, where=np.array(values) >= 0,
                           alpha=0.3, color='red', interpolate=True, label='Violation')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Constraint Value')
            ax.set_title(f'{constraint_name} Constraint')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_constraints, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved constraint satisfaction plot to {save_path}")
        
        return fig
    
    def plot_cost_evolution(self,
                           cost_history: List[float],
                           cost_components: Optional[List[Dict]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot MPC cost function evolution over time.
        
        Args:
            cost_history: History of total costs
            cost_components: History of cost component breakdowns
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Total cost evolution
        ax1.plot(cost_history, 'b-', linewidth=2, label='Total Cost')
        ax1.set_xlabel('MPC Iteration')
        ax1.set_ylabel('Cost Value')
        ax1.set_title('MPC Cost Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Cost components breakdown
        if cost_components:
            component_names = list(cost_components[0].keys())
            for component in component_names:
                values = [costs.get(component, 0) for costs in cost_components]
                ax2.plot(values, linewidth=2, label=component)
            
            ax2.set_xlabel('MPC Iteration')
            ax2.set_ylabel('Component Cost')
            ax2.set_title('Cost Components Breakdown')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            # Show cost gradient if components not available
            cost_gradient = np.gradient(cost_history)
            ax2.plot(cost_gradient, 'r-', linewidth=2, label='Cost Gradient')
            ax2.set_xlabel('MPC Iteration')
            ax2.set_ylabel('Cost Change')
            ax2.set_title('Cost Gradient')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cost evolution plot to {save_path}")
        
        return fig
    
    def plot_performance_metrics(self,
                               performance_data: Dict[str, List[float]],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot MPC performance metrics.
        
        Args:
            performance_data: Dictionary of performance metric histories
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        n_metrics = len(performance_data)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(performance_data.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            ax.plot(values, 'b-', linewidth=2)
            
            # Add statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axhline(mean_val, color='red', linestyle='--', alpha=0.7,
                      label=f'Mean: {mean_val:.4f}')
            ax.fill_between(range(len(values)), mean_val - std_val, mean_val + std_val,
                           alpha=0.2, color='red', label=f'±1σ: {std_val:.4f}')
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance metrics plot to {save_path}")
        
        return fig
    
    def create_real_time_dashboard(self,
                                 update_function: callable,
                                 interval: int = 100) -> FuncAnimation:
        """
        Create real-time MPC monitoring dashboard.
        
        Args:
            update_function: Function to update plot data
            interval: Update interval in milliseconds
        
        Returns:
            Animation object for real-time updates
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        def animate(frame):
            # Clear axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # Get updated data
            data = update_function()
            
            if 'trajectory' in data:
                trajectory = data['trajectory']
                ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-')
                ax1.set_title('End-Effector XY Trajectory')
                ax1.set_xlabel('X (m)')
                ax1.set_ylabel('Y (m)')
                ax1.grid(True)
            
            if 'joint_positions' in data:
                joint_pos = data['joint_positions']
                ax2.plot(joint_pos)
                ax2.set_title('Joint Positions')
                ax2.set_xlabel('Joint Index')
                ax2.set_ylabel('Position (rad)')
                ax2.grid(True)
            
            if 'control_effort' in data:
                control = data['control_effort']
                ax3.plot(control)
                ax3.set_title('Control Effort')
                ax3.set_xlabel('Control Index')
                ax3.set_ylabel('Control Value')
                ax3.grid(True)
            
            if 'cost_history' in data:
                costs = data['cost_history']
                ax4.plot(costs, 'r-')
                ax4.set_title('Cost Evolution')
                ax4.set_xlabel('Time Step')
                ax4.set_ylabel('Cost')
                ax4.grid(True)
        
        animation = FuncAnimation(fig, animate, interval=interval, blit=False)
        return animation
    
    def _draw_sphere(self, ax, center: np.ndarray, radius: float,
                    alpha: float = 0.3, color: str = 'red',
                    label: Optional[str] = None) -> None:
        """Draw a sphere on 3D axis."""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, alpha=alpha, color=color, label=label)


class HRIVisualizationDashboard:
    """
    Specialized dashboard for Human-Robot Interaction MPC visualization.
    
    Provides HRI-specific visualizations including:
    - Human-robot proximity monitoring
    - Intent prediction visualization
    - Safety zone monitoring
    - Interaction phase tracking
    """
    
    def __init__(self, hri_controller: HRIMPCController):
        """
        Initialize HRI visualization dashboard.
        
        Args:
            hri_controller: HRI MPC controller instance
        """
        self.hri_controller = hri_controller
        self.visualizer = MPCVisualizer(hri_controller.robot_model)
        
        # Data storage for dashboard
        self.interaction_history = []
        self.intent_history = []
        self.safety_history = []
        
        logger.info("Initialized HRI visualization dashboard")
    
    def update_hri_data(self,
                       robot_state: RobotState,
                       mpc_result: MPCResult,
                       human_state: Optional[Dict] = None) -> None:
        """
        Update dashboard with new HRI data.
        
        Args:
            robot_state: Current robot state
            mpc_result: Latest MPC result
            human_state: Current human state
        """
        timestamp = robot_state.timestamp
        
        # Store interaction data
        interaction_data = {
            'timestamp': timestamp,
            'phase': self.hri_controller.current_phase.value,
            'robot_position': robot_state.end_effector_pose[0:3],
            'robot_velocity': np.linalg.norm(robot_state.end_effector_velocity[0:3]),
            'solve_time': mpc_result.solve_time,
            'cost': mpc_result.optimal_cost
        }
        
        if human_state:
            interaction_data['human_position'] = human_state.get('position', np.zeros(3))
            interaction_data['intent_uncertainty'] = human_state.get('uncertainty', 0.0)
            interaction_data['dominant_intent'] = max(
                human_state.get('intent_probabilities', {}).items(),
                key=lambda x: x[1], default=('unknown', 0.0)
            )[0]
        
        self.interaction_history.append(interaction_data)
        
        # Maintain history size
        max_history = 1000
        if len(self.interaction_history) > max_history:
            self.interaction_history = self.interaction_history[-max_history//2:]
    
    def create_hri_dashboard(self) -> go.Figure:
        """
        Create comprehensive HRI monitoring dashboard.
        
        Returns:
            Plotly dashboard figure
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Robot-Human Distance', 'Intent Uncertainty',
                'Interaction Phases', 'MPC Solve Times',
                'Safety Violations', 'Control Performance'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        if not self.interaction_history:
            logger.warning("No interaction history available for dashboard")
            return fig
        
        timestamps = [d['timestamp'] for d in self.interaction_history]
        
        # Robot-Human Distance
        if 'human_position' in self.interaction_history[0]:
            distances = []
            for data in self.interaction_history:
                robot_pos = data['robot_position']
                human_pos = data.get('human_position', np.zeros(3))
                distance = np.linalg.norm(robot_pos - human_pos)
                distances.append(distance)
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=distances, name='Distance',
                          line=dict(color='blue')),
                row=1, col=1
            )
            
            # Safety threshold
            safety_threshold = self.hri_controller.hri_config.min_safety_distance
            fig.add_hline(y=safety_threshold, line_dash="dash", line_color="red",
                         row=1, col=1)
        
        # Intent Uncertainty
        if 'intent_uncertainty' in self.interaction_history[0]:
            uncertainties = [d.get('intent_uncertainty', 0.0) for d in self.interaction_history]
            fig.add_trace(
                go.Scatter(x=timestamps, y=uncertainties, name='Uncertainty',
                          line=dict(color='orange')),
                row=1, col=2
            )
        
        # Interaction Phases
        phases = [d['phase'] for d in self.interaction_history]
        phase_numeric = [hash(p) % 10 for p in phases]  # Simple numeric encoding
        fig.add_trace(
            go.Scatter(x=timestamps, y=phase_numeric, mode='markers+lines',
                      name='Phase', line=dict(color='green')),
            row=2, col=1
        )
        
        # MPC Solve Times
        solve_times = [d.get('solve_time', 0.0) for d in self.interaction_history]
        fig.add_trace(
            go.Scatter(x=timestamps, y=solve_times, name='Solve Time',
                      line=dict(color='purple')),
            row=2, col=2
        )
        
        # Real-time constraint
        real_time_limit = self.hri_controller.config.max_solve_time
        fig.add_hline(y=real_time_limit, line_dash="dash", line_color="red",
                     row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='HRI MPC Monitoring Dashboard',
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Distance (m)", row=1, col=1)
        fig.update_yaxes(title_text="Uncertainty", row=1, col=2)
        fig.update_yaxes(title_text="Phase", row=2, col=1)
        fig.update_yaxes(title_text="Solve Time (s)", row=2, col=2)
        
        return fig
    
    def save_hri_summary_report(self, save_path: str) -> None:
        """
        Generate and save HRI performance summary report.
        
        Args:
            save_path: Path to save report
        """
        if not self.interaction_history:
            logger.warning("No interaction history to summarize")
            return
        
        # Create summary visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Extract data
        timestamps = [d['timestamp'] for d in self.interaction_history]
        phases = [d['phase'] for d in self.interaction_history]
        solve_times = [d.get('solve_time', 0.0) for d in self.interaction_history]
        uncertainties = [d.get('intent_uncertainty', 0.0) for d in self.interaction_history]
        
        # Phase distribution
        phase_counts = {phase: phases.count(phase) for phase in set(phases)}
        axes[0].pie(phase_counts.values(), labels=phase_counts.keys(), autopct='%1.1f%%')
        axes[0].set_title('Interaction Phase Distribution')
        
        # Solve time histogram
        axes[1].hist(solve_times, bins=30, alpha=0.7, color='purple')
        axes[1].axvline(self.hri_controller.config.max_solve_time, color='red',
                       linestyle='--', label='Real-time Limit')
        axes[1].set_xlabel('Solve Time (s)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('MPC Solve Time Distribution')
        axes[1].legend()
        
        # Uncertainty over time
        axes[2].plot(timestamps, uncertainties, 'o-', alpha=0.7)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Intent Uncertainty')
        axes[2].set_title('Intent Uncertainty Evolution')
        axes[2].grid(True, alpha=0.3)
        
        # Performance statistics
        axes[3].text(0.1, 0.9, f"Total Interactions: {len(self.interaction_history)}", 
                    transform=axes[3].transAxes, fontsize=12)
        axes[3].text(0.1, 0.8, f"Mean Solve Time: {np.mean(solve_times):.4f}s",
                    transform=axes[3].transAxes, fontsize=12)
        axes[3].text(0.1, 0.7, f"Max Solve Time: {np.max(solve_times):.4f}s",
                    transform=axes[3].transAxes, fontsize=12)
        axes[3].text(0.1, 0.6, f"Real-time Violations: {sum(1 for t in solve_times if t > self.hri_controller.config.max_solve_time)}",
                    transform=axes[3].transAxes, fontsize=12)
        axes[3].text(0.1, 0.5, f"Mean Uncertainty: {np.mean(uncertainties):.3f}",
                    transform=axes[3].transAxes, fontsize=12)
        axes[3].set_title('Performance Summary')
        axes[3].axis('off')
        
        # Hide unused subplots
        for i in range(4, 6):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved HRI summary report to {save_path}")
        plt.close()


def create_mpc_visualization_suite(robot_model: Robot6DOF,
                                 hri_controller: Optional[HRIMPCController] = None) -> Dict:
    """
    Create complete MPC visualization suite.
    
    Args:
        robot_model: Robot dynamics model
        hri_controller: HRI MPC controller (optional)
    
    Returns:
        Dictionary of visualization tools
    """
    suite = {
        'visualizer': MPCVisualizer(robot_model),
        'dashboard': None
    }
    
    if hri_controller:
        suite['dashboard'] = HRIVisualizationDashboard(hri_controller)
    
    logger.info("Created MPC visualization suite")
    return suite