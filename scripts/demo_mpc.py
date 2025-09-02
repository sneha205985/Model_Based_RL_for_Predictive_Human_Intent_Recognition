"""
Comprehensive demonstration of MPC Controller implementation.

This script demonstrates the complete MPC system including:
- Basic MPC controller with robot dynamics
- Human-Robot Interaction MPC
- Safety constraint enforcement
- Real-time performance analysis
- Visualization capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.control import (
    MPCController, MPCConfiguration, SolverType,
    HRIMPCController, HRIConfiguration, HumanState, create_default_hri_mpc
)
from src.models.robotics import create_default_6dof_robot
from src.control.safety_constraints import create_default_safety_constraints, SafetyMonitor
from src.visualization.mpc_plots import MPCVisualizer, create_mpc_visualization_suite
from src.data.synthetic_generator import SyntheticHumanBehaviorGenerator, GestureType
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


class MPCDemo:
    """Comprehensive MPC demonstration suite."""
    
    def __init__(self, output_dir: str = "demo_results"):
        """Initialize demo suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create robot model
        self.robot = create_default_6dof_robot()
        
        # Create visualization tools
        self.viz_suite = create_mpc_visualization_suite(self.robot)
        
        logger.info(f"Initialized MPC demo suite, output: {output_dir}")
    
    def demo_basic_mpc(self):
        """Demonstrate basic MPC controller functionality."""
        logger.info("=" * 60)
        logger.info("DEMO 1: Basic MPC Controller")
        logger.info("=" * 60)
        
        # Create basic MPC controller
        config = MPCConfiguration(
            prediction_horizon=15,
            control_horizon=12,
            sampling_time=0.1,
            solver_type=SolverType.SCIPY_MINIMIZE,
            max_solve_time=0.5
        )
        
        mpc = MPCController(
            config=config,
            state_dim=self.robot.state_dim,
            control_dim=self.robot.control_dim,
            dynamics_model=lambda x, u: self.robot.dynamics(x, u)
        )
        
        # Set up objective function
        Q = np.eye(12)
        Q[0:6, 0:6] *= 10  # Weight joint positions more
        R = np.eye(6) * 0.01
        mpc.set_objective_function(Q, R)
        
        # Set constraints
        mpc.set_constraints(
            state_bounds=(
                np.concatenate([self.robot.joint_limits.position_min,
                               -self.robot.joint_limits.velocity_max]),
                np.concatenate([self.robot.joint_limits.position_max,
                               self.robot.joint_limits.velocity_max])
            ),
            control_bounds=(
                -self.robot.joint_limits.torque_max * 0.5,
                self.robot.joint_limits.torque_max * 0.5
            )
        )
        
        # Demonstration: Point-to-point motion
        logger.info("Testing point-to-point motion...")
        
        initial_state = np.concatenate([
            np.array([0.0, -0.5, 0.8, 0.0, 0.3, 0.0]),  # joint positions
            np.zeros(6)  # joint velocities
        ])
        
        target_state = np.concatenate([
            np.array([0.3, -0.2, 0.5, 0.1, 0.4, -0.1]),  # target positions
            np.zeros(6)  # target velocities
        ])
        
        # Create reference trajectory
        N = config.prediction_horizon
        reference = np.tile(target_state, (N + 1, 1))
        
        # Solve MPC
        start_time = time.time()
        result = mpc.solve_mpc(initial_state, reference_trajectory=reference)
        solve_time = time.time() - start_time
        
        logger.info(f"MPC solve status: {result.status}")
        logger.info(f"Solve time: {solve_time:.4f} seconds")
        logger.info(f"Optimal cost: {result.optimal_cost}")
        
        if result.optimal_control is not None:
            logger.info(f"Control sequence shape: {result.optimal_control.shape}")
            logger.info(f"First control input: {result.optimal_control[0]}")
            
            # Simulate closed-loop for a few steps
            current_state = initial_state.copy()
            trajectory = [current_state.copy()]
            
            for step in range(5):
                # Apply control
                control = result.optimal_control[min(step, len(result.optimal_control)-1)]
                current_state = self.robot.dynamics(current_state, control)
                trajectory.append(current_state.copy())
                
                logger.info(f"Step {step+1} - Joint positions: {current_state[0:6]}")
            
            # Visualize trajectory if possible
            try:
                trajectory_array = np.array(trajectory)
                fig = self.viz_suite['visualizer'].plot_3d_trajectory(
                    predicted_states=trajectory_array,
                    save_path=str(self.output_dir / "basic_mpc_trajectory.png")
                )
                logger.info("Saved trajectory visualization")
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
        
        else:
            logger.warning("MPC failed to find optimal control sequence")
        
        # Performance metrics
        metrics = mpc.get_performance_metrics()
        logger.info(f"Performance metrics: {metrics}")
        
        logger.info("Basic MPC demo completed.\n")
    
    def demo_hri_mpc(self):
        """Demonstrate Human-Robot Interaction MPC."""
        logger.info("=" * 60)
        logger.info("DEMO 2: Human-Robot Interaction MPC")
        logger.info("=" * 60)
        
        # Create HRI MPC controller
        hri_controller = create_default_hri_mpc(self.robot)
        
        # Set up cost matrices
        Q = np.eye(12)
        Q[0:6, 0:6] *= 10
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
                -self.robot.joint_limits.torque_max * 0.3,
                self.robot.joint_limits.torque_max * 0.3
            )
        )
        
        # Create synthetic human behavior
        workspace_bounds = np.array([-1, 1, -1, 1, 0, 2])
        behavior_gen = SyntheticHumanBehaviorGenerator(
            workspace_bounds=workspace_bounds,
            random_seed=42
        )
        
        # Generate human sequence
        logger.info("Generating synthetic human behavior sequence...")
        human_sequence = behavior_gen.generate_sequence(
            gesture_type=GestureType.HANDOVER,
            duration=3.0,
            noise_level=0.02
        )
        
        if human_sequence:
            logger.info(f"Generated human sequence with {len(human_sequence.trajectory)} points")
            logger.info(f"Gesture type: {human_sequence.gesture_type}")
            logger.info(f"Duration: {human_sequence.duration:.2f} seconds")
        else:
            logger.warning("Could not generate human sequence, using synthetic data")
            
        # Initial robot state
        robot_state = self.robot.get_state_from_measurements(
            np.array([0.1, -0.3, 0.6, 0.0, 0.2, 0.0]),
            np.zeros(6)
        )
        
        # Target pose for robot
        target_pose = np.array([0.5, 0.3, 0.8, 0.0, 0.0, 0.0])
        
        # Demonstration scenarios
        scenarios = [
            {
                'name': 'Collaborative Handover',
                'human_pos': np.array([0.6, 0.25, 0.85]),
                'intent': {'handover': 0.9, 'reach': 0.1},
                'uncertainty': 0.1
            },
            {
                'name': 'Human Reaching',
                'human_pos': np.array([0.7, 0.2, 0.8]),
                'intent': {'reach': 0.7, 'grab': 0.3},
                'uncertainty': 0.2
            },
            {
                'name': 'Human Waving (Retreat)',
                'human_pos': np.array([0.8, 0.3, 0.9]),
                'intent': {'wave': 0.8, 'idle': 0.2},
                'uncertainty': 0.3
            },
            {
                'name': 'High Uncertainty Scenario',
                'human_pos': np.array([0.5, 0.4, 0.7]),
                'intent': {'reach': 0.4, 'grab': 0.3, 'wave': 0.3},
                'uncertainty': 0.8
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"\nTesting scenario: {scenario['name']}")
            
            # Create human state
            human_state = HumanState(
                position=scenario['human_pos'],
                velocity=np.zeros(3),
                intent_probabilities=scenario['intent'],
                uncertainty=scenario['uncertainty'],
                timestamp=float(i)
            )
            
            # Solve HRI MPC
            start_time = time.time()
            result = hri_controller.solve_hri_mpc(
                robot_state, target_pose, human_state
            )
            solve_time = time.time() - start_time
            
            logger.info(f"  Interaction phase: {hri_controller.current_phase.value}")
            logger.info(f"  MPC status: {result.status}")
            logger.info(f"  Solve time: {solve_time:.4f}s")
            
            if result.optimal_control is not None:
                control_magnitude = np.linalg.norm(result.optimal_control[0])
                logger.info(f"  Control magnitude: {control_magnitude:.4f}")
                
                # Check proximity factor (speed reduction)
                distance_to_human = np.linalg.norm(
                    robot_state.end_effector_pose[0:3] - scenario['human_pos']
                )
                logger.info(f"  Distance to human: {distance_to_human:.3f}m")
            
            results.append({
                'scenario': scenario['name'],
                'phase': hri_controller.current_phase.value,
                'solve_time': solve_time,
                'status': result.status.name,
                'uncertainty': scenario['uncertainty']
            })
        
        # Performance analysis
        logger.info("\nHRI MPC Performance Analysis:")
        successful_results = [r for r in results if r['status'] in ['OPTIMAL', 'FEASIBLE']]
        
        if successful_results:
            mean_solve_time = np.mean([r['solve_time'] for r in successful_results])
            logger.info(f"  Success rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results):.1%})")
            logger.info(f"  Mean solve time: {mean_solve_time:.4f}s")
            
            # Analyze phase distribution
            phase_counts = {}
            for r in results:
                phase = r['phase']
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
            logger.info("  Interaction phases:")
            for phase, count in phase_counts.items():
                logger.info(f"    {phase}: {count}")
        
        # Get HRI-specific metrics
        hri_metrics = hri_controller.get_interaction_metrics()
        logger.info(f"  HRI metrics: {hri_metrics}")
        
        logger.info("HRI MPC demo completed.\n")
    
    def demo_safety_constraints(self):
        """Demonstrate safety constraint enforcement."""
        logger.info("=" * 60)
        logger.info("DEMO 3: Safety Constraint Enforcement")
        logger.info("=" * 60)
        
        # Create safety constraints
        safety_constraints = create_default_safety_constraints(self.robot)
        logger.info(f"Created {len(safety_constraints)} safety constraints")
        
        # Create safety monitor
        from src.control.safety_constraints import SafetyParameters
        safety_params = SafetyParameters(
            min_distance=0.3,
            collision_buffer=0.1,
            workspace_bounds=self.robot.workspace_bounds
        )
        
        safety_monitor = SafetyMonitor(
            constraints=safety_constraints,
            safety_params=safety_params
        )
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Safe Configuration',
                'state': np.concatenate([
                    np.array([0.1, -0.2, 0.3, 0.0, 0.1, 0.0]),  # positions
                    np.array([0.1, -0.1, 0.05, 0.2, -0.1, 0.0])  # velocities
                ]),
                'human_pos': np.array([0.8, 0.5, 1.0])
            },
            {
                'name': 'Near Joint Limits',
                'state': np.concatenate([
                    self.robot.joint_limits.position_max * 0.95,  # Near upper limits
                    np.zeros(6)
                ]),
                'human_pos': np.array([0.8, 0.5, 1.0])
            },
            {
                'name': 'High Velocities',
                'state': np.concatenate([
                    np.zeros(6),
                    self.robot.joint_limits.velocity_max * 0.8  # High velocities
                ]),
                'human_pos': np.array([0.8, 0.5, 1.0])
            },
            {
                'name': 'Human Too Close',
                'state': np.zeros(12),
                'human_pos': np.array([0.2, 0.1, 0.5])  # Very close to robot base
            }
        ]
        
        for scenario in test_scenarios:
            logger.info(f"\nTesting: {scenario['name']}")
            
            # Check safety
            context = {
                'human_position': scenario['human_pos'],
                'human_uncertainty': 0.2,
                'timestamp': time.time()
            }
            
            # Simulate predicted trajectory (simple forward integration)
            current_state = scenario['state']
            predicted_trajectory = [current_state]
            for _ in range(5):
                # Simple forward integration with zero control
                next_state = self.robot.dynamics(current_state, np.zeros(6))
                predicted_trajectory.append(next_state)
                current_state = next_state
            
            safety_result = safety_monitor.check_safety(
                current_state=scenario['state'],
                predicted_trajectory=np.array(predicted_trajectory),
                context=context
            )
            
            logger.info(f"  Safe: {safety_result['is_safe']}")
            logger.info(f"  Warning: {safety_result['is_warning']}")
            logger.info(f"  Emergency: {safety_result['is_emergency']}")
            logger.info(f"  Safety margin: {safety_result['safety_margin']:.4f}")
            
            if safety_result['violations']:
                logger.info("  Violations:")
                for violation in safety_result['violations']:
                    logger.info(f"    {violation['constraint']}: {violation['violation']:.4f}")
        
        # Safety metrics
        safety_metrics = safety_monitor.get_safety_metrics()
        logger.info(f"\nSafety monitoring metrics: {safety_metrics}")
        
        logger.info("Safety constraint demo completed.\n")
    
    def demo_real_time_performance(self):
        """Demonstrate real-time performance analysis."""
        logger.info("=" * 60)
        logger.info("DEMO 4: Real-Time Performance Analysis")
        logger.info("=" * 60)
        
        # Create MPC controller optimized for real-time
        config = MPCConfiguration(
            prediction_horizon=10,  # Shorter horizon for speed
            control_horizon=8,
            sampling_time=0.1,
            solver_type=SolverType.SCIPY_MINIMIZE,
            max_solve_time=0.05,  # 50ms real-time constraint
            use_warm_start=True
        )
        
        mpc = MPCController(
            config=config,
            state_dim=12,
            control_dim=6,
            dynamics_model=lambda x, u: self.robot.dynamics(x, u)
        )
        
        # Set up controller
        Q = np.eye(12)
        R = np.eye(6) * 0.01
        mpc.set_objective_function(Q, R)
        
        mpc.set_constraints(
            state_bounds=(
                np.concatenate([self.robot.joint_limits.position_min,
                               -self.robot.joint_limits.velocity_max]),
                np.concatenate([self.robot.joint_limits.position_max,
                               self.robot.joint_limits.velocity_max])
            ),
            control_bounds=(
                -self.robot.joint_limits.torque_max * 0.3,
                self.robot.joint_limits.torque_max * 0.3
            )
        )
        
        # Real-time simulation
        logger.info("Running real-time simulation...")
        
        sim_duration = 5.0  # seconds
        target_frequency = 10.0  # Hz
        dt = 1.0 / target_frequency
        n_steps = int(sim_duration / dt)
        
        # Initial state and target
        current_state = np.concatenate([
            np.array([0.0, -0.3, 0.5, 0.0, 0.2, 0.0]),
            np.zeros(6)
        ])
        
        target_state = np.concatenate([
            np.array([0.2, -0.1, 0.3, 0.1, 0.3, -0.1]),
            np.zeros(6)
        ])
        
        # Performance tracking
        solve_times = []
        costs = []
        real_time_violations = 0
        
        logger.info(f"Target: {target_frequency}Hz ({dt:.3f}s per cycle)")
        logger.info(f"Real-time limit: {config.max_solve_time:.3f}s")
        
        start_time = time.time()
        
        for step in range(n_steps):
            step_start = time.time()
            
            # Create reference trajectory
            N = config.prediction_horizon
            reference = np.tile(target_state, (N + 1, 1))
            
            # Solve MPC
            solve_start = time.time()
            result = mpc.solve_mpc(current_state, reference_trajectory=reference)
            solve_time = time.time() - solve_start
            
            solve_times.append(solve_time)
            
            if result.optimal_cost is not None:
                costs.append(result.optimal_cost)
            
            # Check real-time constraint
            if solve_time > config.max_solve_time:
                real_time_violations += 1
            
            # Apply control and simulate
            if result.optimal_control is not None and len(result.optimal_control) > 0:
                control = result.optimal_control[0]
                current_state = self.robot.dynamics(current_state, control)
            
            # Maintain timing
            step_time = time.time() - step_start
            sleep_time = max(0, dt - step_time)
            time.sleep(sleep_time)
            
            if (step + 1) % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  Step {step+1}/{n_steps}, elapsed: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        actual_frequency = n_steps / total_time
        
        # Performance analysis
        logger.info("\nReal-Time Performance Results:")
        logger.info(f"  Target frequency: {target_frequency:.1f} Hz")
        logger.info(f"  Actual frequency: {actual_frequency:.1f} Hz")
        logger.info(f"  Mean solve time: {np.mean(solve_times):.4f}s")
        logger.info(f"  Max solve time: {np.max(solve_times):.4f}s")
        logger.info(f"  Std solve time: {np.std(solve_times):.4f}s")
        logger.info(f"  Real-time violations: {real_time_violations}/{n_steps} ({real_time_violations/n_steps:.1%})")
        
        if costs:
            logger.info(f"  Mean cost: {np.mean(costs):.2f}")
            logger.info(f"  Final cost: {costs[-1]:.2f}")
        
        # Error analysis
        final_error = np.linalg.norm(current_state[0:6] - target_state[0:6])
        logger.info(f"  Final tracking error: {final_error:.4f} rad")
        
        # Visualize performance if possible
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Solve times
            time_vec = np.arange(len(solve_times)) * dt
            ax1.plot(time_vec, solve_times, 'b-', alpha=0.7)
            ax1.axhline(config.max_solve_time, color='red', linestyle='--', 
                       label='Real-time Limit')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Solve Time (s)')
            ax1.set_title('MPC Real-Time Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Cost evolution
            if costs:
                cost_time_vec = np.arange(len(costs)) * dt
                ax2.plot(cost_time_vec, costs, 'g-', alpha=0.7)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Cost')
                ax2.set_title('Cost Evolution')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "real_time_performance.png", dpi=300, bbox_inches='tight')
            logger.info("Saved real-time performance plot")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Performance visualization failed: {e}")
        
        logger.info("Real-time performance demo completed.\n")
    
    def run_all_demos(self):
        """Run all demonstration scenarios."""
        logger.info("Starting comprehensive MPC demonstration...")
        logger.info(f"Output directory: {self.output_dir}")
        
        try:
            self.demo_basic_mpc()
            self.demo_hri_mpc()
            self.demo_safety_constraints()
            self.demo_real_time_performance()
            
            logger.info("=" * 60)
            logger.info("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {self.output_dir}")
            
            # List generated files
            generated_files = list(self.output_dir.glob("*"))
            if generated_files:
                logger.info("\nGenerated files:")
                for file_path in generated_files:
                    logger.info(f"  {file_path.name}")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            import traceback
            logger.error(traceback.format_exc())


def main():
    """Run MPC demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MPC Controller Demonstration")
    parser.add_argument('--output-dir', default='demo_results',
                       help='Output directory for demo results')
    parser.add_argument('--demo', choices=['basic', 'hri', 'safety', 'realtime', 'all'],
                       default='all', help='Which demo to run')
    
    args = parser.parse_args()
    
    # Create demo suite
    demo = MPCDemo(args.output_dir)
    
    # Run requested demo
    if args.demo == 'basic':
        demo.demo_basic_mpc()
    elif args.demo == 'hri':
        demo.demo_hri_mpc()
    elif args.demo == 'safety':
        demo.demo_safety_constraints()
    elif args.demo == 'realtime':
        demo.demo_real_time_performance()
    elif args.demo == 'all':
        demo.run_all_demos()


if __name__ == "__main__":
    main()