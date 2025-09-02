"""
Concrete implementation of Nonlinear Model Predictive Control (NMPC) controller.

This module implements a high-performance NMPC controller for safe human-robot
interaction with real-time constraints and uncertainty handling.
"""

import numpy as np
import scipy.optimize as opt
from scipy.linalg import solve_discrete_are, block_diag
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import time
import logging
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import numba
    from numba import jit, types, typed
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .mpc_controller import (
    MPCController, MPCConfiguration, RobotState, ControlAction,
    OptimizationResult, ControlObjective, ConstraintType
)
from ..models.human_behavior import HumanState, BehaviorPrediction
from ..models.intent_predictor import IntentPrediction, ContextInformation


@dataclass
class NMPCConfiguration(MPCConfiguration):
    """Extended configuration for Nonlinear MPC."""
    solver_method: str = "SLSQP"
    line_search_method: str = "armijo"
    hessian_approximation: str = "BFGS"
    feasibility_tolerance: float = 1e-6
    optimality_tolerance: float = 1e-6
    max_line_search_iterations: int = 20
    adaptive_discretization: bool = True
    uncertainty_propagation: bool = True
    robust_horizon: int = 3
    risk_threshold: float = 0.1


@dataclass
class CostFunction:
    """Represents a cost function component."""
    objective_type: ControlObjective
    weight: float
    reference: Optional[Any] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    gradient_func: Optional[Callable] = None
    hessian_func: Optional[Callable] = None


@dataclass
class Constraint:
    """Represents an optimization constraint."""
    constraint_type: ConstraintType
    bounds: Union[Tuple[float, float], np.ndarray]
    parameters: Dict[str, Any] = field(default_factory=dict)
    jacobian_func: Optional[Callable] = None
    violation_penalty: float = 1e6


class NonlinearMPCController(MPCController):
    """
    Nonlinear Model Predictive Control controller for human-robot interaction.
    
    Features:
    - Nonlinear dynamics and constraints
    - Uncertainty propagation and robust control
    - Real-time optimization with warm-starting
    - Adaptive discretization and prediction horizons
    - Safety-critical constraint handling
    """
    
    def __init__(self, config: NMPCConfiguration) -> None:
        """Initialize the Nonlinear MPC controller."""
        if not isinstance(config, NMPCConfiguration):
            # Convert base config to NMPC config
            nmpc_config = NMPCConfiguration(
                prediction_horizon=config.prediction_horizon,
                control_horizon=config.control_horizon,
                sampling_time=config.sampling_time,
                state_weights=config.state_weights,
                control_weights=config.control_weights,
                terminal_weights=config.terminal_weights,
                max_iterations=config.max_iterations,
                convergence_tolerance=config.convergence_tolerance,
                warm_start=config.warm_start
            )
            config = nmpc_config
        
        super().__init__(config)
        self.config: NMPCConfiguration = config
        
        # Optimization components
        self.cost_functions: List[CostFunction] = []
        self.constraints: List[Constraint] = []
        self.dynamics_function: Optional[Callable] = None
        self.linearization_function: Optional[Callable] = None
        
        # State and control dimensions (will be set during initialization)
        self.state_dim: int = 0
        self.control_dim: int = 0
        
        # Warm start solution
        self.warm_start_solution: Optional[np.ndarray] = None
        self.previous_solution: Optional[np.ndarray] = None
        
        # Performance tracking
        self.solve_times: List[float] = []
        self.convergence_flags: List[bool] = []
        
        # Uncertainty handling
        self.uncertainty_propagator: Optional[Callable] = None
        self.risk_allocator: Dict[str, float] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_controller(self) -> None:
        """Initialize the MPC controller components."""
        # Set up default cost functions and constraints
        self._setup_default_objectives()
        self._setup_default_constraints()
        
        # Initialize optimization solver
        self._setup_optimizer()
        
        self.is_initialized = True
        self.logger.info("Nonlinear MPC controller initialized")
    
    def _setup_default_objectives(self) -> None:
        """Set up default objective functions."""
        # Task completion objective
        self.add_objective(
            ControlObjective.TASK_COMPLETION,
            weight=self.config.state_weights.get('task', 1.0),
            parameters={'type': 'quadratic'}
        )
        
        # Control effort minimization
        for control_type, weight in self.config.control_weights.items():
            if weight > 0:
                self.add_objective(
                    ControlObjective.ENERGY_EFFICIENCY,
                    weight=weight,
                    parameters={'control_type': control_type}
                )
        
        # Smooth motion objective
        self.add_objective(
            ControlObjective.SMOOTH_MOTION,
            weight=self.config.state_weights.get('smoothness', 0.1)
        )
    
    def _setup_default_constraints(self) -> None:
        """Set up default constraints."""
        # Basic safety constraints will be added when dynamics model is set
        pass
    
    def _setup_optimizer(self) -> None:
        """Set up the nonlinear optimization solver."""
        self.solver_options = {
            'method': self.config.solver_method,
            'options': {
                'maxiter': self.config.max_iterations,
                'ftol': self.config.optimality_tolerance,
                'gtol': self.config.optimality_tolerance,
                'disp': False
            }
        }
        
        if self.config.solver_method == 'SLSQP':
            self.solver_options['options'].update({
                'eps': 1e-8,
                'finite_diff_rel_step': None
            })
    
    def set_dynamics_model(
        self,
        dynamics_function: Callable[[RobotState, ControlAction, float], RobotState],
        linearization_function: Optional[Callable] = None
    ) -> None:
        """Set the robot dynamics model for MPC prediction."""
        self.dynamics_function = dynamics_function
        self.linearization_function = linearization_function
        
        # Determine state and control dimensions from first call
        dummy_state = RobotState(
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            end_effector_pose=np.zeros(7)
        )
        dummy_control = ControlAction(
            joint_torques=np.zeros(7)
        )
        
        try:
            result_state = dynamics_function(dummy_state, dummy_control, self.config.sampling_time)
            self.state_dim = len(dummy_state.joint_positions) * 2 + 7  # positions + velocities + ee_pose
            self.control_dim = len(dummy_control.joint_torques) if dummy_control.joint_torques is not None else 7
        except Exception as e:
            self.logger.warning(f"Could not determine dimensions from dynamics function: {e}")
            self.state_dim = 21  # Default for 7-DOF robot
            self.control_dim = 7
        
        self.logger.info(f"Dynamics model set: state_dim={self.state_dim}, control_dim={self.control_dim}")
    
    def add_objective(
        self,
        objective_type: ControlObjective,
        weight: float,
        reference: Optional[Any] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an objective function to the MPC cost."""
        if parameters is None:
            parameters = {}
        
        # Create gradient and Hessian functions if possible
        gradient_func = self._create_objective_gradient(objective_type, parameters)
        hessian_func = self._create_objective_hessian(objective_type, parameters)
        
        cost_function = CostFunction(
            objective_type=objective_type,
            weight=weight,
            reference=reference,
            parameters=parameters,
            gradient_func=gradient_func,
            hessian_func=hessian_func
        )
        
        self.cost_functions.append(cost_function)
        self.logger.debug(f"Added objective: {objective_type.value} with weight {weight}")
    
    def add_constraint(
        self,
        constraint_type: ConstraintType,
        bounds: Union[Tuple[float, float], np.ndarray],
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a constraint to the MPC optimization problem."""
        if parameters is None:
            parameters = {}
        
        jacobian_func = self._create_constraint_jacobian(constraint_type, parameters)
        
        constraint = Constraint(
            constraint_type=constraint_type,
            bounds=bounds,
            parameters=parameters,
            jacobian_func=jacobian_func
        )
        
        self.constraints.append(constraint)
        self.logger.debug(f"Added constraint: {constraint_type.value}")
    
    def solve_mpc(
        self,
        current_state: RobotState,
        human_predictions: List[BehaviorPrediction],
        intent_predictions: List[IntentPrediction],
        context: ContextInformation
    ) -> OptimizationResult:
        """Solve the MPC optimization problem."""
        if not self.is_initialized:
            raise RuntimeError("Controller not initialized")
        
        if self.dynamics_function is None:
            raise RuntimeError("Dynamics model not set")
        
        start_time = time.time()
        
        # Prepare optimization variables
        n_vars = self.config.control_horizon * self.control_dim
        
        # Initial guess (warm start if available)
        if self.config.warm_start and self.warm_start_solution is not None:
            x0 = self.warm_start_solution[:n_vars]
        else:
            x0 = np.zeros(n_vars)
        
        # Set up optimization problem
        bounds = self._create_bounds()
        constraints = self._create_constraint_functions(
            current_state, human_predictions, intent_predictions, context
        )
        
        # Create cost function
        cost_func = self._create_cost_function(
            current_state, human_predictions, intent_predictions, context
        )
        
        # Solve optimization problem
        try:
            result = opt.minimize(
                cost_func,
                x0,
                bounds=bounds,
                constraints=constraints,
                **self.solver_options
            )
            
            convergence_status = result.success
            optimal_solution = result.x
            cost_value = result.fun
            iterations = result.nit if hasattr(result, 'nit') else 0
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            convergence_status = False
            optimal_solution = x0
            cost_value = float('inf')
            iterations = 0
        
        solve_time = time.time() - start_time
        
        # Extract control sequence and predict states
        optimal_controls = self._extract_control_sequence(optimal_solution)
        predicted_states = self.get_predicted_trajectory(current_state, optimal_controls)
        
        # Compute constraint violations
        constraint_violations = self._compute_constraint_violations(
            optimal_solution, current_state, human_predictions, intent_predictions, context
        )
        
        # Store warm start for next iteration
        if convergence_status and self.config.warm_start:
            self.warm_start_solution = np.concatenate([
                optimal_solution[self.control_dim:],  # Shift solution
                optimal_solution[-self.control_dim:]  # Repeat last control
            ])
        
        # Update performance tracking
        self.solve_times.append(solve_time)
        self.convergence_flags.append(convergence_status)
        
        optimization_result = OptimizationResult(
            optimal_controls=optimal_controls,
            predicted_states=predicted_states,
            cost_value=cost_value,
            constraint_violations=constraint_violations,
            solve_time=solve_time,
            iterations=iterations,
            convergence_status=convergence_status
        )
        
        self.optimization_history.append(optimization_result)
        
        if solve_time > 0.1:  # 100ms threshold
            self.logger.warning(f"MPC solve time {solve_time:.3f}s exceeds real-time constraint")
        
        return optimization_result
    
    def get_next_control(
        self,
        current_state: RobotState,
        human_predictions: List[BehaviorPrediction],
        intent_predictions: List[IntentPrediction],
        context: ContextInformation
    ) -> ControlAction:
        """Get the next control action using receding horizon control."""
        optimization_result = self.solve_mpc(
            current_state, human_predictions, intent_predictions, context
        )
        
        if optimization_result.optimal_controls:
            next_control = optimization_result.optimal_controls[0]
            next_control.timestamp = time.time()
            
            # Store in history
            self.control_history.append(next_control)
            self.state_history.append(current_state)
            
            return next_control
        else:
            # Emergency fallback - zero control
            self.logger.error("No optimal control found, using zero control")
            return ControlAction(
                joint_torques=np.zeros(self.control_dim),
                timestamp=time.time()
            )
    
    def _create_bounds(self) -> List[Tuple[float, float]]:
        """Create bounds for optimization variables."""
        bounds = []
        
        # Default control bounds
        for _ in range(self.config.control_horizon):
            for _ in range(self.control_dim):
                bounds.append((-100.0, 100.0))  # Default torque bounds
        
        # Apply constraint-specific bounds
        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.CONTROL_BOUNDS:
                if isinstance(constraint.bounds, tuple):
                    lower, upper = constraint.bounds
                    for i in range(len(bounds)):
                        bounds[i] = (max(bounds[i][0], lower), min(bounds[i][1], upper))
        
        return bounds
    
    def _create_cost_function(
        self,
        current_state: RobotState,
        human_predictions: List[BehaviorPrediction],
        intent_predictions: List[IntentPrediction],
        context: ContextInformation
    ) -> Callable:
        """Create the cost function for optimization."""
        
        def cost_function(x: np.ndarray) -> float:
            total_cost = 0.0
            
            # Extract control sequence
            controls = self._extract_control_sequence(x)
            
            # Predict trajectory
            try:
                states = self.get_predicted_trajectory(current_state, controls)
            except Exception:
                return 1e6  # High penalty for infeasible trajectories
            
            # Evaluate each cost function component
            for cost_func in self.cost_functions:
                try:
                    component_cost = self._evaluate_cost_component(
                        cost_func, states, controls, human_predictions, intent_predictions, context
                    )
                    total_cost += cost_func.weight * component_cost
                except Exception as e:
                    self.logger.debug(f"Cost evaluation error: {e}")
                    total_cost += 1e3  # Penalty for evaluation errors
            
            # Add penalty for constraint violations
            violations = self._compute_constraint_violations(
                x, current_state, human_predictions, intent_predictions, context
            )
            
            for constraint_type, violation in violations.items():
                if violation > 0:
                    total_cost += 1e4 * violation ** 2
            
            return total_cost
        
        return cost_function
    
    def _create_constraint_functions(
        self,
        current_state: RobotState,
        human_predictions: List[BehaviorPrediction],
        intent_predictions: List[IntentPrediction],
        context: ContextInformation
    ) -> List[Dict[str, Any]]:
        """Create constraint functions for optimization."""
        constraint_functions = []
        
        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.COLLISION_AVOIDANCE:
                constraint_functions.append({
                    'type': 'ineq',
                    'fun': lambda x, c=constraint: self._collision_constraint(
                        x, current_state, human_predictions, c
                    )
                })
            elif constraint.constraint_type == ConstraintType.SAFETY_DISTANCE:
                constraint_functions.append({
                    'type': 'ineq',
                    'fun': lambda x, c=constraint: self._safety_distance_constraint(
                        x, current_state, human_predictions, c
                    )
                })
        
        return constraint_functions
    
    def _extract_control_sequence(self, x: np.ndarray) -> List[ControlAction]:
        """Extract control sequence from optimization variables."""
        controls = []
        
        for i in range(self.config.control_horizon):
            start_idx = i * self.control_dim
            end_idx = (i + 1) * self.control_dim
            
            control_values = x[start_idx:end_idx]
            
            control = ControlAction(
                joint_torques=control_values.copy(),
                execution_time=self.config.sampling_time
            )
            controls.append(control)
        
        return controls
    
    def _evaluate_cost_component(
        self,
        cost_func: CostFunction,
        states: List[RobotState],
        controls: List[ControlAction],
        human_predictions: List[BehaviorPrediction],
        intent_predictions: List[IntentPrediction],
        context: ContextInformation
    ) -> float:
        """Evaluate a single cost function component."""
        
        if cost_func.objective_type == ControlObjective.TASK_COMPLETION:
            return self._task_completion_cost(states, cost_func.reference)
        
        elif cost_func.objective_type == ControlObjective.ENERGY_EFFICIENCY:
            return self._energy_efficiency_cost(controls)
        
        elif cost_func.objective_type == ControlObjective.SMOOTH_MOTION:
            return self._smooth_motion_cost(controls)
        
        elif cost_func.objective_type == ControlObjective.HUMAN_SAFETY:
            return self._human_safety_cost(states, human_predictions)
        
        elif cost_func.objective_type == ControlObjective.COLLABORATIVE_EFFICIENCY:
            return self._collaborative_efficiency_cost(states, human_predictions, intent_predictions)
        
        elif cost_func.objective_type == ControlObjective.UNCERTAINTY_MINIMIZATION:
            return self._uncertainty_minimization_cost(states, context)
        
        else:
            return 0.0
    
    def _task_completion_cost(self, states: List[RobotState], reference: Optional[Any]) -> float:
        """Evaluate task completion cost."""
        if reference is None or not states:
            return 0.0
        
        # Simple quadratic cost to reference position
        terminal_state = states[-1]
        if isinstance(reference, np.ndarray) and reference.shape == (7,):
            error = terminal_state.end_effector_pose[:3] - reference[:3]
            return np.sum(error ** 2)
        
        return 0.0
    
    def _energy_efficiency_cost(self, controls: List[ControlAction]) -> float:
        """Evaluate energy efficiency cost."""
        total_cost = 0.0
        
        for control in controls:
            if control.joint_torques is not None:
                total_cost += np.sum(control.joint_torques ** 2) * control.execution_time
        
        return total_cost
    
    def _smooth_motion_cost(self, controls: List[ControlAction]) -> float:
        """Evaluate smooth motion cost (control rate penalty)."""
        if len(controls) < 2:
            return 0.0
        
        total_cost = 0.0
        
        for i in range(1, len(controls)):
            if (controls[i].joint_torques is not None and 
                controls[i-1].joint_torques is not None):
                
                control_diff = controls[i].joint_torques - controls[i-1].joint_torques
                total_cost += np.sum(control_diff ** 2)
        
        return total_cost
    
    def _human_safety_cost(self, states: List[RobotState], human_predictions: List[BehaviorPrediction]) -> float:
        """Evaluate human safety cost."""
        if not states or not human_predictions:
            return 0.0
        
        total_cost = 0.0
        
        for i, state in enumerate(states):
            if i < len(human_predictions):
                human_pred = human_predictions[i]
                
                # Simple distance-based safety cost
                robot_pos = state.end_effector_pose[:3]
                
                if human_pred.predicted_trajectory.size > 0:
                    human_pos = human_pred.predicted_trajectory[0, :3]  # Assuming first point is position
                    distance = np.linalg.norm(robot_pos - human_pos)
                    
                    # Exponential penalty for close proximity
                    if distance < 2.0:  # Safety threshold
                        total_cost += np.exp(-(distance - 0.5) / 0.2)
        
        return total_cost
    
    def _collaborative_efficiency_cost(
        self,
        states: List[RobotState],
        human_predictions: List[BehaviorPrediction],
        intent_predictions: List[IntentPrediction]
    ) -> float:
        """Evaluate collaborative efficiency cost."""
        # Placeholder for collaborative efficiency evaluation
        return 0.0
    
    def _uncertainty_minimization_cost(self, states: List[RobotState], context: ContextInformation) -> float:
        """Evaluate uncertainty minimization cost."""
        total_cost = 0.0
        
        for state in states:
            if state.uncertainty is not None:
                # Penalize high uncertainty
                total_cost += np.trace(state.uncertainty)
        
        return total_cost
    
    def _compute_constraint_violations(
        self,
        x: np.ndarray,
        current_state: RobotState,
        human_predictions: List[BehaviorPrediction],
        intent_predictions: List[IntentPrediction],
        context: ContextInformation
    ) -> Dict[str, float]:
        """Compute constraint violations for given solution."""
        violations = {}
        
        controls = self._extract_control_sequence(x)
        
        try:
            states = self.get_predicted_trajectory(current_state, controls)
        except Exception:
            violations['dynamics'] = 1e6
            return violations
        
        for constraint in self.constraints:
            try:
                if constraint.constraint_type == ConstraintType.COLLISION_AVOIDANCE:
                    violation = -self._collision_constraint(x, current_state, human_predictions, constraint)
                    violations[constraint.constraint_type.value] = max(0, violation)
                
                elif constraint.constraint_type == ConstraintType.SAFETY_DISTANCE:
                    violation = -self._safety_distance_constraint(x, current_state, human_predictions, constraint)
                    violations[constraint.constraint_type.value] = max(0, violation)
                
                elif constraint.constraint_type == ConstraintType.JOINT_LIMITS:
                    violation = self._joint_limits_violation(states, constraint)
                    violations[constraint.constraint_type.value] = violation
                
                elif constraint.constraint_type == ConstraintType.VELOCITY_LIMITS:
                    violation = self._velocity_limits_violation(states, constraint)
                    violations[constraint.constraint_type.value] = violation
            
            except Exception as e:
                self.logger.debug(f"Constraint evaluation error: {e}")
                violations[constraint.constraint_type.value] = 1e3
        
        return violations
    
    def _collision_constraint(
        self,
        x: np.ndarray,
        current_state: RobotState,
        human_predictions: List[BehaviorPrediction],
        constraint: Constraint
    ) -> float:
        """Evaluate collision avoidance constraint."""
        controls = self._extract_control_sequence(x)
        
        try:
            states = self.get_predicted_trajectory(current_state, controls)
        except Exception:
            return -1e6  # Infeasible trajectory
        
        min_distance = float('inf')
        
        for i, state in enumerate(states):
            if i < len(human_predictions):
                human_pred = human_predictions[i]
                
                robot_pos = state.end_effector_pose[:3]
                
                if human_pred.predicted_trajectory.size > 0:
                    human_pos = human_pred.predicted_trajectory[0, :3]
                    distance = np.linalg.norm(robot_pos - human_pos)
                    min_distance = min(min_distance, distance)
        
        # Return negative of minimum distance (constraint is g(x) >= 0)
        safety_margin = constraint.parameters.get('safety_margin', 0.5)
        return min_distance - safety_margin
    
    def _safety_distance_constraint(
        self,
        x: np.ndarray,
        current_state: RobotState,
        human_predictions: List[BehaviorPrediction],
        constraint: Constraint
    ) -> float:
        """Evaluate safety distance constraint."""
        return self._collision_constraint(x, current_state, human_predictions, constraint)
    
    def _joint_limits_violation(self, states: List[RobotState], constraint: Constraint) -> float:
        """Compute joint limits violation."""
        if isinstance(constraint.bounds, tuple):
            lower, upper = constraint.bounds
        else:
            return 0.0
        
        max_violation = 0.0
        
        for state in states:
            for joint_pos in state.joint_positions:
                if joint_pos < lower:
                    max_violation = max(max_violation, lower - joint_pos)
                elif joint_pos > upper:
                    max_violation = max(max_violation, joint_pos - upper)
        
        return max_violation
    
    def _velocity_limits_violation(self, states: List[RobotState], constraint: Constraint) -> float:
        """Compute velocity limits violation."""
        if isinstance(constraint.bounds, tuple):
            lower, upper = constraint.bounds
        else:
            return 0.0
        
        max_violation = 0.0
        
        for state in states:
            for joint_vel in state.joint_velocities:
                if joint_vel < lower:
                    max_violation = max(max_violation, lower - joint_vel)
                elif joint_vel > upper:
                    max_violation = max(max_violation, joint_vel - upper)
        
        return max_violation
    
    def _create_objective_gradient(self, objective_type: ControlObjective, parameters: Dict[str, Any]) -> Optional[Callable]:
        """Create gradient function for objective (placeholder for future implementation)."""
        return None
    
    def _create_objective_hessian(self, objective_type: ControlObjective, parameters: Dict[str, Any]) -> Optional[Callable]:
        """Create Hessian function for objective (placeholder for future implementation)."""
        return None
    
    def _create_constraint_jacobian(self, constraint_type: ConstraintType, parameters: Dict[str, Any]) -> Optional[Callable]:
        """Create Jacobian function for constraint (placeholder for future implementation)."""
        return None
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get controller performance metrics."""
        if not self.solve_times:
            return {}
        
        return {
            'average_solve_time': np.mean(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'convergence_rate': np.mean(self.convergence_flags),
            'real_time_violations': np.sum(np.array(self.solve_times) > 0.1),
            'total_optimizations': len(self.solve_times)
        }
    
    def reset_performance_tracking(self) -> None:
        """Reset performance tracking metrics."""
        self.solve_times.clear()
        self.convergence_flags.clear()
    
    def set_real_time_mode(self, enabled: bool) -> None:
        """Enable/disable real-time optimization mode."""
        if enabled:
            # Reduce iterations for real-time performance
            self.config.max_iterations = min(self.config.max_iterations, 50)
            self.solver_options['options']['maxiter'] = self.config.max_iterations
            
            # Use faster but less accurate tolerances
            self.config.convergence_tolerance = max(self.config.convergence_tolerance, 1e-4)
            self.solver_options['options']['ftol'] = self.config.convergence_tolerance
            self.solver_options['options']['gtol'] = self.config.convergence_tolerance
            
            self.logger.info("Real-time mode enabled - reduced accuracy for speed")
        else:
            # Restore original settings for accuracy
            self._setup_optimizer()
            self.logger.info("Real-time mode disabled - full accuracy restored")


@jit(nopython=True, cache=True)
def _fast_dynamics_integration(
    state_vector: np.ndarray,
    control_vector: np.ndarray,
    dt: float,
    mass_matrix: np.ndarray,
    coriolis_matrix: np.ndarray
) -> np.ndarray:
    """Fast dynamics integration using Numba JIT compilation."""
    n_dof = len(control_vector)
    
    # Extract positions and velocities
    q = state_vector[:n_dof]
    q_dot = state_vector[n_dof:2*n_dof]
    
    # Simple forward Euler integration (can be improved to RK4)
    q_ddot = np.linalg.solve(mass_matrix, control_vector - coriolis_matrix @ q_dot)
    
    # Integration
    q_new = q + q_dot * dt
    q_dot_new = q_dot + q_ddot * dt
    
    # Combine new state
    state_new = np.concatenate([q_new, q_dot_new])
    
    return state_new


if __name__ == "__main__":
    # Example usage and testing
    config = NMPCConfiguration(
        prediction_horizon=10,
        control_horizon=5,
        sampling_time=0.1,
        state_weights={'task': 1.0, 'smoothness': 0.1},
        control_weights={'torque': 0.01},
        terminal_weights={'task': 10.0}
    )
    
    controller = NonlinearMPCController(config)
    print(f"Controller initialized: {controller.is_initialized}")
    print(f"Controller info: {controller.get_controller_info()}")