"""
Performance benchmarking script for MPC Controller.

This script provides comprehensive performance analysis of the MPC controller
including:
- Solve time vs horizon length analysis
- Scalability testing
- Memory usage profiling
- Real-time performance validation
- Constraint satisfaction benchmarking
- Comparison of solver methods
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import gc
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path
import argparse
import logging

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.control.mpc_controller import MPCController, MPCConfiguration, SolverType
from src.control.hri_mpc import HRIMPCController, HRIConfiguration, create_default_hri_mpc
from src.models.robotics.robot_dynamics import create_default_6dof_robot
from src.control.safety_constraints import create_default_safety_constraints, SafetyMonitor, SafetyParameters
from src.visualization.mpc_plots import MPCVisualizer
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


class MPCBenchmark:
    """
    Comprehensive MPC performance benchmarking suite.
    
    Provides various benchmarks for analyzing MPC controller performance
    including timing, scalability, memory usage, and solution quality.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize MPC benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Benchmark data storage
        self.results = {}
        
        # Create robot model
        self.robot = create_default_6dof_robot()
        
        logger.info(f"Initialized MPC benchmark suite, output: {output_dir}")
    
    def benchmark_solve_time_vs_horizon(self,
                                      horizon_range: List[int] = None,
                                      num_trials: int = 10) -> Dict:
        """
        Benchmark solve time vs prediction horizon length.
        
        Args:
            horizon_range: List of horizon lengths to test
            num_trials: Number of trials per horizon length
        
        Returns:
            Benchmark results dictionary
        """
        if horizon_range is None:
            horizon_range = [5, 10, 15, 20, 25, 30, 40, 50]
        
        logger.info(f"Benchmarking solve time vs horizon: {horizon_range}")
        
        results = {
            'horizons': [],
            'mean_solve_times': [],
            'std_solve_times': [],
            'min_solve_times': [],
            'max_solve_times': [],
            'success_rates': [],
            'solver_type': 'scipy_minimize'
        }
        
        for horizon in horizon_range:
            logger.info(f"Testing horizon length: {horizon}")
            
            # Create MPC controller
            config = MPCConfiguration(
                prediction_horizon=horizon,
                control_horizon=min(horizon, 20),
                solver_type=SolverType.SCIPY_MINIMIZE,
                max_solve_time=5.0  # Generous for benchmarking
            )
            
            controller = MPCController(
                config=config,
                state_dim=12,
                control_dim=6,
                dynamics_model=self._get_robot_dynamics_wrapper()
            )
            
            # Set up objective and constraints
            self._setup_controller(controller)
            
            # Run multiple trials
            solve_times = []
            successes = 0
            
            for trial in range(num_trials):
                # Random initial state
                initial_state = self._generate_random_state()
                
                start_time = time.time()
                result = controller.solve_mpc(initial_state)
                solve_time = time.time() - start_time
                
                solve_times.append(solve_time)
                
                if result.status.name in ['OPTIMAL', 'FEASIBLE']:
                    successes += 1
                
                # Clear any warm start to ensure independent trials
                controller.reset_warm_start()
            
            # Store results
            results['horizons'].append(horizon)
            results['mean_solve_times'].append(np.mean(solve_times))
            results['std_solve_times'].append(np.std(solve_times))
            results['min_solve_times'].append(np.min(solve_times))
            results['max_solve_times'].append(np.max(solve_times))
            results['success_rates'].append(successes / num_trials)
            
            logger.info(f"Horizon {horizon}: mean={np.mean(solve_times):.4f}s, "
                       f"success_rate={successes/num_trials:.2%}")
        
        # Save results
        self.results['horizon_benchmark'] = results
        self._save_horizon_benchmark_plot(results)
        
        return results
    
    def benchmark_solver_comparison(self,
                                  horizon: int = 20,
                                  num_trials: int = 20) -> Dict:
        """
        Compare performance of different solvers.
        
        Args:
            horizon: Prediction horizon to use
            num_trials: Number of trials per solver
        
        Returns:
            Solver comparison results
        """
        logger.info(f"Benchmarking solver comparison with horizon={horizon}")
        
        # Only test solvers that are likely to work
        solvers_to_test = [SolverType.SCIPY_MINIMIZE]
        
        # Try CVXPY if available
        try:
            import cvxpy as cp
            solvers_to_test.append(SolverType.CVXPY_QP)
        except ImportError:
            logger.warning("CVXPY not available, skipping CVXPY_QP solver")
        
        results = {
            'solvers': [],
            'mean_solve_times': [],
            'std_solve_times': [],
            'success_rates': [],
            'mean_costs': [],
            'memory_usage': []
        }
        
        for solver_type in solvers_to_test:
            logger.info(f"Testing solver: {solver_type.value}")
            
            config = MPCConfiguration(
                prediction_horizon=horizon,
                solver_type=solver_type,
                max_solve_time=10.0
            )
            
            controller = MPCController(
                config=config,
                state_dim=12,
                control_dim=6,
                dynamics_model=self._get_robot_dynamics_wrapper()
            )
            
            self._setup_controller(controller)
            
            # Run trials
            solve_times = []
            costs = []
            successes = 0
            memory_usages = []
            
            for trial in range(num_trials):
                # Monitor memory usage
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                initial_state = self._generate_random_state()
                
                start_time = time.time()
                result = controller.solve_mpc(initial_state)
                solve_time = time.time() - start_time
                
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_usages.append(mem_after - mem_before)
                
                solve_times.append(solve_time)
                
                if result.status.name in ['OPTIMAL', 'FEASIBLE']:
                    successes += 1
                    if result.optimal_cost is not None:
                        costs.append(result.optimal_cost)
                
                controller.reset_warm_start()
                gc.collect()  # Clean up memory
            
            # Store results
            results['solvers'].append(solver_type.value)
            results['mean_solve_times'].append(np.mean(solve_times))
            results['std_solve_times'].append(np.std(solve_times))
            results['success_rates'].append(successes / num_trials)
            results['mean_costs'].append(np.mean(costs) if costs else 0.0)
            results['memory_usage'].append(np.mean(memory_usages))
            
            logger.info(f"Solver {solver_type.value}: mean_time={np.mean(solve_times):.4f}s, "
                       f"success_rate={successes/num_trials:.2%}, "
                       f"memory={np.mean(memory_usages):.2f}MB")
        
        self.results['solver_comparison'] = results
        self._save_solver_comparison_plot(results)
        
        return results
    
    def benchmark_real_time_performance(self,
                                      target_frequency: float = 10.0,
                                      duration: float = 30.0) -> Dict:
        """
        Benchmark real-time performance at target control frequency.
        
        Args:
            target_frequency: Target control frequency (Hz)
            duration: Test duration (seconds)
        
        Returns:
            Real-time performance results
        """
        logger.info(f"Benchmarking real-time performance at {target_frequency}Hz for {duration}s")
        
        # Setup controller for real-time
        config = MPCConfiguration(
            prediction_horizon=15,
            control_horizon=10,
            sampling_time=1.0 / target_frequency,
            max_solve_time=0.8 / target_frequency,  # 80% of available time
            use_warm_start=True
        )
        
        controller = MPCController(
            config=config,
            state_dim=12,
            control_dim=6,
            dynamics_model=self._get_robot_dynamics_wrapper()
        )
        
        self._setup_controller(controller)
        
        # Simulation variables
        current_state = self._generate_random_state()
        target_dt = 1.0 / target_frequency
        
        # Data collection
        solve_times = []
        costs = []
        real_time_violations = 0
        total_iterations = 0
        
        start_time = time.time()
        last_time = start_time
        
        while time.time() - start_time < duration:
            iteration_start = time.time()
            
            # Solve MPC
            result = controller.solve_mpc(current_state)
            solve_time = result.solve_time
            
            solve_times.append(solve_time)
            if result.optimal_cost is not None:
                costs.append(result.optimal_cost)
            
            # Check real-time constraint
            if solve_time > config.max_solve_time:
                real_time_violations += 1
            
            # Simulate system forward (simple integration)
            if result.optimal_control is not None and len(result.optimal_control) > 0:
                control = result.optimal_control[0]
                current_state = self.robot.dynamics(current_state, control)
            
            total_iterations += 1
            
            # Sleep to maintain target frequency
            iteration_time = time.time() - iteration_start
            sleep_time = max(0, target_dt - iteration_time)
            time.sleep(sleep_time)
            
            if total_iterations % int(target_frequency) == 0:
                elapsed = time.time() - start_time
                logger.info(f"Real-time test progress: {elapsed:.1f}s / {duration:.1f}s")
        
        # Compute results
        results = {
            'target_frequency': target_frequency,
            'actual_frequency': total_iterations / duration,
            'mean_solve_time': np.mean(solve_times),
            'max_solve_time': np.max(solve_times),
            'std_solve_time': np.std(solve_times),
            'real_time_violations': real_time_violations,
            'violation_rate': real_time_violations / total_iterations,
            'mean_cost': np.mean(costs) if costs else 0.0,
            'total_iterations': total_iterations
        }
        
        logger.info(f"Real-time benchmark completed:")
        logger.info(f"  Target frequency: {target_frequency:.1f}Hz")
        logger.info(f"  Actual frequency: {results['actual_frequency']:.1f}Hz")
        logger.info(f"  Mean solve time: {results['mean_solve_time']:.4f}s")
        logger.info(f"  Violations: {real_time_violations}/{total_iterations} ({results['violation_rate']:.1%})")
        
        self.results['real_time_performance'] = results
        self._save_real_time_performance_plot(results, solve_times)
        
        return results
    
    def benchmark_hri_performance(self,
                                num_scenarios: int = 10,
                                scenario_duration: float = 10.0) -> Dict:
        """
        Benchmark Human-Robot Interaction MPC performance.
        
        Args:
            num_scenarios: Number of HRI scenarios to test
            scenario_duration: Duration of each scenario (seconds)
        
        Returns:
            HRI performance results
        """
        logger.info(f"Benchmarking HRI MPC with {num_scenarios} scenarios")
        
        # Create HRI controller
        hri_controller = create_default_hri_mpc(self.robot)
        
        # Setup cost matrices
        Q = np.eye(12)
        Q[0:6, 0:6] *= 10  # Weight joint positions
        R = np.eye(6) * 0.01
        hri_controller.set_objective_function(Q, R)
        
        # Set constraints
        hri_controller.set_constraints(
            state_bounds=(
                np.concatenate([self.robot.joint_limits.position_min, 
                               -self.robot.joint_limits.velocity_max]),
                np.concatenate([self.robot.joint_limits.position_max,
                               self.robot.joint_limits.velocity_max])
            ),
            control_bounds=(
                -self.robot.joint_limits.torque_max * 0.5,  # Conservative
                self.robot.joint_limits.torque_max * 0.5
            )
        )
        
        # Benchmark data
        scenario_results = []
        
        for scenario in range(num_scenarios):
            logger.info(f"Running HRI scenario {scenario + 1}/{num_scenarios}")
            
            # Generate random scenario
            human_trajectory = self._generate_human_trajectory(scenario_duration)
            robot_initial_state = self._generate_robot_state()
            target_pose = np.array([0.5, 0.2, 0.8, 0.0, 0.0, 0.0])
            
            # Run scenario
            scenario_result = self._run_hri_scenario(
                hri_controller, robot_initial_state, target_pose,
                human_trajectory, scenario_duration
            )
            
            scenario_results.append(scenario_result)
        
        # Aggregate results
        results = {
            'num_scenarios': num_scenarios,
            'mean_solve_time': np.mean([r['mean_solve_time'] for r in scenario_results]),
            'max_solve_time': np.max([r['max_solve_time'] for r in scenario_results]),
            'mean_safety_distance': np.mean([r['mean_safety_distance'] for r in scenario_results]),
            'min_safety_distance': np.min([r['min_safety_distance'] for r in scenario_results]),
            'interaction_phases': {},
            'intent_adaptation_events': sum(r['intent_changes'] for r in scenario_results),
            'emergency_stops': sum(r['emergency_stops'] for r in scenario_results)
        }
        
        # Aggregate interaction phases
        all_phases = []
        for r in scenario_results:
            all_phases.extend(r['phases'])
        
        phase_counts = {}
        for phase in all_phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        results['interaction_phases'] = phase_counts
        
        logger.info(f"HRI benchmark completed:")
        logger.info(f"  Mean solve time: {results['mean_solve_time']:.4f}s")
        logger.info(f"  Min safety distance: {results['min_safety_distance']:.3f}m")
        logger.info(f"  Emergency stops: {results['emergency_stops']}")
        logger.info(f"  Intent adaptations: {results['intent_adaptation_events']}")
        
        self.results['hri_performance'] = results
        self._save_hri_performance_plot(results, scenario_results)
        
        return results
    
    def generate_benchmark_report(self) -> str:
        """
        Generate comprehensive benchmark report.
        
        Returns:
            Path to generated report
        """
        logger.info("Generating comprehensive benchmark report")
        
        report_path = self.output_dir / "mpc_benchmark_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# MPC Controller Performance Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Horizon benchmark
            if 'horizon_benchmark' in self.results:
                f.write("## Solve Time vs Horizon Length\n\n")
                horizon_results = self.results['horizon_benchmark']
                
                f.write("| Horizon | Mean Time (s) | Std (s) | Success Rate |\n")
                f.write("|---------|---------------|---------|---------------|\n")
                
                for i, horizon in enumerate(horizon_results['horizons']):
                    mean_time = horizon_results['mean_solve_times'][i]
                    std_time = horizon_results['std_solve_times'][i]
                    success_rate = horizon_results['success_rates'][i]
                    f.write(f"| {horizon:2d} | {mean_time:.4f} | {std_time:.4f} | {success_rate:.1%} |\n")
                
                f.write("\n")
            
            # Solver comparison
            if 'solver_comparison' in self.results:
                f.write("## Solver Comparison\n\n")
                solver_results = self.results['solver_comparison']
                
                f.write("| Solver | Mean Time (s) | Success Rate | Memory (MB) |\n")
                f.write("|--------|---------------|--------------|-------------|\n")
                
                for i, solver in enumerate(solver_results['solvers']):
                    mean_time = solver_results['mean_solve_times'][i]
                    success_rate = solver_results['success_rates'][i]
                    memory = solver_results['memory_usage'][i]
                    f.write(f"| {solver} | {mean_time:.4f} | {success_rate:.1%} | {memory:.2f} |\n")
                
                f.write("\n")
            
            # Real-time performance
            if 'real_time_performance' in self.results:
                f.write("## Real-Time Performance\n\n")
                rt_results = self.results['real_time_performance']
                
                f.write(f"- Target Frequency: {rt_results['target_frequency']:.1f} Hz\n")
                f.write(f"- Actual Frequency: {rt_results['actual_frequency']:.1f} Hz\n")
                f.write(f"- Mean Solve Time: {rt_results['mean_solve_time']:.4f} s\n")
                f.write(f"- Max Solve Time: {rt_results['max_solve_time']:.4f} s\n")
                f.write(f"- Real-time Violations: {rt_results['violation_rate']:.1%}\n\n")
            
            # HRI performance
            if 'hri_performance' in self.results:
                f.write("## Human-Robot Interaction Performance\n\n")
                hri_results = self.results['hri_performance']
                
                f.write(f"- Number of Scenarios: {hri_results['num_scenarios']}\n")
                f.write(f"- Mean Solve Time: {hri_results['mean_solve_time']:.4f} s\n")
                f.write(f"- Min Safety Distance: {hri_results['min_safety_distance']:.3f} m\n")
                f.write(f"- Emergency Stops: {hri_results['emergency_stops']}\n")
                f.write(f"- Intent Adaptations: {hri_results['intent_adaptation_events']}\n\n")
                
                f.write("### Interaction Phase Distribution\n\n")
                for phase, count in hri_results['interaction_phases'].items():
                    f.write(f"- {phase}: {count}\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Performance Recommendations\n\n")
            
            if 'horizon_benchmark' in self.results:
                horizon_results = self.results['horizon_benchmark']
                optimal_horizon = None
                min_time_increase = float('inf')
                
                for i in range(1, len(horizon_results['horizons'])):
                    time_increase = (horizon_results['mean_solve_times'][i] - 
                                   horizon_results['mean_solve_times'][i-1])
                    if time_increase < min_time_increase and horizon_results['success_rates'][i] > 0.9:
                        min_time_increase = time_increase
                        optimal_horizon = horizon_results['horizons'][i-1]
                
                if optimal_horizon:
                    f.write(f"- Recommended horizon length: {optimal_horizon} (good performance/time tradeoff)\n")
            
            if 'real_time_performance' in self.results:
                rt_results = self.results['real_time_performance']
                if rt_results['violation_rate'] > 0.1:
                    f.write("- Consider reducing prediction horizon for better real-time performance\n")
                if rt_results['violation_rate'] < 0.01:
                    f.write("- Real-time performance is excellent, horizon could potentially be increased\n")
            
            f.write("\n## Visualization Files\n\n")
            f.write("The following plots have been generated:\n\n")
            plot_files = list(self.output_dir.glob("*.png"))
            for plot_file in plot_files:
                f.write(f"- [{plot_file.name}]({plot_file.name})\n")
        
        logger.info(f"Benchmark report saved to {report_path}")
        return str(report_path)
    
    def _get_robot_dynamics_wrapper(self):
        """Get robot dynamics wrapper for MPC."""
        def dynamics_wrapper(state, control):
            return self.robot.dynamics(state, control)
        return dynamics_wrapper
    
    def _setup_controller(self, controller: MPCController):
        """Setup controller with standard objective and constraints."""
        Q = np.eye(12)
        Q[0:6, 0:6] *= 10  # Weight joint positions more
        R = np.eye(6) * 0.01
        controller.set_objective_function(Q, R)
        
        # Set reasonable constraints
        controller.set_constraints(
            state_bounds=(
                np.concatenate([self.robot.joint_limits.position_min,
                               -self.robot.joint_limits.velocity_max]),
                np.concatenate([self.robot.joint_limits.position_max,
                               self.robot.joint_limits.velocity_max])
            ),
            control_bounds=(
                -self.robot.joint_limits.torque_max * 0.3,  # Conservative limits
                self.robot.joint_limits.torque_max * 0.3
            )
        )
    
    def _generate_random_state(self) -> np.ndarray:
        """Generate random feasible state."""
        # Random joint positions within limits
        joint_pos = np.random.uniform(
            self.robot.joint_limits.position_min * 0.5,
            self.robot.joint_limits.position_max * 0.5
        )
        
        # Small random joint velocities
        joint_vel = np.random.uniform(
            -self.robot.joint_limits.velocity_max * 0.1,
            self.robot.joint_limits.velocity_max * 0.1
        )
        
        return np.concatenate([joint_pos, joint_vel])
    
    def _generate_robot_state(self):
        """Generate robot state for HRI testing."""
        from src.models.robotics.robot_dynamics import RobotState
        
        joint_pos = self._generate_random_state()[0:6]
        joint_vel = np.zeros(6)
        
        return self.robot.get_state_from_measurements(joint_pos, joint_vel)
    
    def _generate_human_trajectory(self, duration: float):
        """Generate synthetic human trajectory for HRI testing."""
        dt = 0.1
        n_steps = int(duration / dt)
        
        # Simple sinusoidal human movement
        t = np.linspace(0, duration, n_steps)
        
        trajectory = []
        for i in range(n_steps):
            human_pos = np.array([
                0.5 + 0.2 * np.sin(2 * np.pi * t[i] / 5),  # x
                0.3 + 0.1 * np.cos(2 * np.pi * t[i] / 5),  # y
                0.8 + 0.1 * np.sin(2 * np.pi * t[i] / 3)   # z
            ])
            
            # Simple intent model
            if t[i] < duration / 3:
                intent = {'reach': 0.8, 'grab': 0.2}
                uncertainty = 0.2
            elif t[i] < 2 * duration / 3:
                intent = {'handover': 0.9, 'reach': 0.1}
                uncertainty = 0.1
            else:
                intent = {'wave': 0.7, 'idle': 0.3}
                uncertainty = 0.3
            
            trajectory.append({
                'position': human_pos,
                'intent_probabilities': intent,
                'uncertainty': uncertainty,
                'timestamp': t[i]
            })
        
        return trajectory
    
    def _run_hri_scenario(self, hri_controller, robot_state, target_pose,
                         human_trajectory, duration):
        """Run single HRI scenario."""
        from src.control.hri_mpc import HumanState
        
        dt = 0.1
        solve_times = []
        safety_distances = []
        phases = []
        intent_changes = 0
        emergency_stops = 0
        
        current_robot_state = robot_state
        last_intent = None
        
        for step_data in human_trajectory:
            # Create human state
            human_state = HumanState(
                position=step_data['position'],
                velocity=np.zeros(3),  # Simplified
                intent_probabilities=step_data['intent_probabilities'],
                uncertainty=step_data['uncertainty'],
                timestamp=step_data['timestamp']
            )
            
            # Solve HRI MPC
            result = hri_controller.solve_hri_mpc(
                current_robot_state, target_pose, human_state
            )
            
            solve_times.append(result.solve_time)
            phases.append(hri_controller.current_phase.value)
            
            # Check for intent changes
            current_intent = max(step_data['intent_probabilities'].items(),
                               key=lambda x: x[1])[0]
            if last_intent and last_intent != current_intent:
                intent_changes += 1
            last_intent = current_intent
            
            # Compute safety distance
            robot_ee_pos = current_robot_state.end_effector_pose[0:3]
            safety_distance = np.linalg.norm(robot_ee_pos - step_data['position'])
            safety_distances.append(safety_distance)
            
            # Check for emergency stops
            if hri_controller.current_phase.value == 'emergency':
                emergency_stops += 1
            
            # Simulate robot forward (simplified)
            if (result.optimal_control is not None and 
                len(result.optimal_control) > 0 and
                result.status.name in ['OPTIMAL', 'FEASIBLE']):
                
                control = result.optimal_control[0]
                next_state_vec = hri_controller.robot_model.dynamics(
                    np.concatenate([current_robot_state.joint_positions,
                                   current_robot_state.joint_velocities]),
                    control
                )
                
                # Update robot state
                current_robot_state = hri_controller.robot_model.get_state_from_measurements(
                    next_state_vec[0:6], next_state_vec[6:12]
                )
                current_robot_state.timestamp += dt
        
        return {
            'mean_solve_time': np.mean(solve_times),
            'max_solve_time': np.max(solve_times),
            'mean_safety_distance': np.mean(safety_distances),
            'min_safety_distance': np.min(safety_distances),
            'phases': phases,
            'intent_changes': intent_changes,
            'emergency_stops': emergency_stops
        }
    
    def _save_horizon_benchmark_plot(self, results):
        """Save horizon benchmark visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        horizons = results['horizons']
        mean_times = results['mean_solve_times']
        std_times = results['std_solve_times']
        success_rates = results['success_rates']
        
        # Solve time plot
        ax1.errorbar(horizons, mean_times, yerr=std_times, 
                    marker='o', capsize=3, capthick=2)
        ax1.set_xlabel('Prediction Horizon')
        ax1.set_ylabel('Solve Time (s)')
        ax1.set_title('MPC Solve Time vs Horizon Length')
        ax1.grid(True, alpha=0.3)
        
        # Success rate plot
        ax2.plot(horizons, success_rates, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Prediction Horizon')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('MPC Success Rate vs Horizon Length')
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'horizon_benchmark.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_solver_comparison_plot(self, results):
        """Save solver comparison visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        solvers = results['solvers']
        
        # Solve time comparison
        ax1.bar(solvers, results['mean_solve_times'])
        ax1.set_ylabel('Mean Solve Time (s)')
        ax1.set_title('Solver Performance Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        ax2.bar(solvers, results['success_rates'])
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Solver Success Rate')
        ax2.set_ylim([0, 1.1])
        ax2.tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        ax3.bar(solvers, results['memory_usage'])
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Solver Memory Usage')
        ax3.tick_params(axis='x', rotation=45)
        
        # Cost comparison (if available)
        if any(cost > 0 for cost in results['mean_costs']):
            ax4.bar(solvers, results['mean_costs'])
            ax4.set_ylabel('Mean Cost')
            ax4.set_title('Solution Quality Comparison')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Cost data not available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Solution Quality')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'solver_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_real_time_performance_plot(self, results, solve_times):
        """Save real-time performance visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Solve time history
        ax1.plot(solve_times, 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(results['target_frequency'] and 1.0/results['target_frequency'] * 0.8, 
                   color='red', linestyle='--', label='Real-time Limit')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Solve Time (s)')
        ax1.set_title('Real-time MPC Solve Times')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Solve time histogram
        ax2.hist(solve_times, bins=30, alpha=0.7, density=True)
        ax2.axvline(results['mean_solve_time'], color='red', linestyle='-', 
                   label=f'Mean: {results["mean_solve_time"]:.4f}s')
        if results['target_frequency']:
            limit = 1.0/results['target_frequency'] * 0.8
            ax2.axvline(limit, color='orange', linestyle='--', 
                       label='Real-time Limit')
        ax2.set_xlabel('Solve Time (s)')
        ax2.set_ylabel('Density')
        ax2.set_title('Solve Time Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'real_time_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_hri_performance_plot(self, results, scenario_results):
        """Save HRI performance visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Solve times across scenarios
        mean_solve_times = [r['mean_solve_time'] for r in scenario_results]
        ax1.plot(mean_solve_times, 'bo-')
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Mean Solve Time (s)')
        ax1.set_title('HRI MPC Solve Times')
        ax1.grid(True, alpha=0.3)
        
        # Safety distances
        min_safety_distances = [r['min_safety_distance'] for r in scenario_results]
        ax2.plot(min_safety_distances, 'ro-')
        ax2.axhline(0.3, color='orange', linestyle='--', label='Safety Threshold')
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Min Safety Distance (m)')
        ax2.set_title('Minimum Safety Distances')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Interaction phase distribution
        if results['interaction_phases']:
            phases = list(results['interaction_phases'].keys())
            counts = list(results['interaction_phases'].values())
            ax3.pie(counts, labels=phases, autopct='%1.1f%%')
            ax3.set_title('Interaction Phase Distribution')
        
        # Intent adaptations per scenario
        intent_changes = [r['intent_changes'] for r in scenario_results]
        ax4.bar(range(len(intent_changes)), intent_changes)
        ax4.set_xlabel('Scenario')
        ax4.set_ylabel('Intent Adaptations')
        ax4.set_title('Intent Adaptation Events')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hri_performance.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Run MPC benchmarking suite."""
    parser = argparse.ArgumentParser(description="MPC Performance Benchmarking")
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--horizon-range', nargs='+', type=int,
                       default=[5, 10, 15, 20, 25, 30],
                       help='Horizon lengths to benchmark')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of trials per test')
    parser.add_argument('--real-time-freq', type=float, default=10.0,
                       help='Target frequency for real-time test (Hz)')
    parser.add_argument('--real-time-duration', type=float, default=30.0,
                       help='Duration for real-time test (s)')
    parser.add_argument('--hri-scenarios', type=int, default=5,
                       help='Number of HRI scenarios to test')
    parser.add_argument('--skip-horizon', action='store_true',
                       help='Skip horizon benchmarking')
    parser.add_argument('--skip-solver', action='store_true',
                       help='Skip solver comparison')
    parser.add_argument('--skip-realtime', action='store_true',
                       help='Skip real-time benchmarking')
    parser.add_argument('--skip-hri', action='store_true',
                       help='Skip HRI benchmarking')
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = MPCBenchmark(args.output_dir)
    
    # Run benchmarks
    if not args.skip_horizon:
        logger.info("Running horizon length benchmark...")
        benchmark.benchmark_solve_time_vs_horizon(
            horizon_range=args.horizon_range,
            num_trials=args.trials
        )
    
    if not args.skip_solver:
        logger.info("Running solver comparison benchmark...")
        benchmark.benchmark_solver_comparison(num_trials=args.trials)
    
    if not args.skip_realtime:
        logger.info("Running real-time performance benchmark...")
        benchmark.benchmark_real_time_performance(
            target_frequency=args.real_time_freq,
            duration=args.real_time_duration
        )
    
    if not args.skip_hri:
        logger.info("Running HRI performance benchmark...")
        benchmark.benchmark_hri_performance(num_scenarios=args.hri_scenarios)
    
    # Generate report
    report_path = benchmark.generate_benchmark_report()
    logger.info(f"Benchmark completed! Report saved to: {report_path}")


if __name__ == "__main__":
    main()