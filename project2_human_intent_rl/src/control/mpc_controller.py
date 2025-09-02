"""
Model Predictive Control implementation for human-robot interaction.

This module implements a comprehensive MPC controller for robotic systems
operating in human-robot interaction scenarios. The controller incorporates
human intent prediction, safety constraints, and real-time optimization.

Mathematical Formulation:
========================

The MPC problem is formulated as a finite horizon optimal control problem:

    minimize     ∑(k=0 to N-1) [x_k^T Q x_k + u_k^T R u_k] + x_N^T P x_N
    subject to   x_{k+1} = f(x_k, u_k)  ∀k ∈ [0, N-1]
                 u_min ≤ u_k ≤ u_max     ∀k ∈ [0, N-1]
                 x_min ≤ x_k ≤ x_max     ∀k ∈ [0, N]
                 g(x_k, u_k) ≤ 0         ∀k ∈ [0, N-1]  (safety constraints)

Where:
- x_k ∈ ℝ^n: system state at time k
- u_k ∈ ℝ^m: control input at time k
- f(·): discretized system dynamics
- Q ∈ ℝ^{n×n}: state cost matrix (positive semi-definite)
- R ∈ ℝ^{m×m}: control cost matrix (positive definite)
- P ∈ ℝ^{n×n}: terminal cost matrix
- g(·): nonlinear safety constraints
- N: prediction horizon
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.linalg import solve_discrete_are, inv, pinv
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
from collections import deque

try:
    from ..utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SolverType(Enum):
    """Available MPC solver types."""
    SCIPY_MINIMIZE = "scipy_minimize"
    CVXPY_QP = "cvxpy_qp"
    CVXPY_SOCP = "cvxpy_socp"
    CUSTOM_QP = "custom_qp"


class MPCStatus(Enum):
    """MPC solver status codes."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"
    CONSTRAINT_VIOLATION_RECOVERY = "constraint_violation_recovery"


@dataclass
class MPCResult:
    """Results from MPC optimization."""
    status: MPCStatus
    optimal_control: Optional[np.ndarray] = None
    predicted_states: Optional[np.ndarray] = None
    optimal_cost: Optional[float] = None
    solve_time: float = 0.0
    iterations: int = 0
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    solver_info: Dict = field(default_factory=dict)
    lyapunov_decrease: Optional[float] = None  # Lyapunov function decrease
    stability_margin: Optional[float] = None   # Stability margin
    feasibility_recovery_used: bool = False    # Whether recovery was triggered


@dataclass
class MPCConfiguration:
    """Configuration parameters for MPC controller."""
    # Horizon parameters
    prediction_horizon: int = 20
    control_horizon: int = 15
    sampling_time: float = 0.1
    
    # Solver parameters
    solver_type: SolverType = SolverType.CVXPY_QP
    max_solve_time: float = 0.05  # Real-time constraint
    solver_tolerance: float = 1e-6
    max_iterations: int = 100
    
    # Warm start
    use_warm_start: bool = True
    warm_start_alpha: float = 0.8
    
    # Safety parameters
    safety_margin: float = 0.1
    collision_threshold: float = 0.05
    emergency_stop_enabled: bool = True
    
    # Feasibility recovery parameters
    enable_feasibility_recovery: bool = True
    recovery_slack_penalty: float = 1e6
    max_recovery_iterations: int = 3
    
    # Stability parameters
    enable_lyapunov_constraints: bool = True
    lyapunov_decrease_rate: float = 0.01  # Minimum required decrease
    terminal_set_constraint: bool = True
    
    # Cost weights
    state_weight_factor: float = 1.0
    control_weight_factor: float = 0.1
    terminal_weight_factor: float = 10.0
    safety_weight_factor: float = 1000.0


class MPCController:
    """
    Model Predictive Control controller for human-robot interaction.
    
    This controller implements a receding horizon optimal control strategy
    that incorporates human intent prediction, safety constraints, and
    real-time optimization requirements.
    
    The controller solves the following optimization problem at each time step:
    
    minimize     J(x₀, U) = ∑(k=0 to N-1) ℓ(x_k, u_k) + V_f(x_N)
    subject to   x_{k+1} = f(x_k, u_k)
                 (x_k, u_k) ∈ X × U
                 x_N ∈ X_f
                 Safety constraints for human-robot interaction
    
    Where U = [u_0, u_1, ..., u_{N-1}] is the control sequence.
    """
    
    def __init__(self, 
                 config: MPCConfiguration,
                 state_dim: int,
                 control_dim: int,
                 dynamics_model: Optional[Callable] = None):
        """
        Initialize MPC controller.
        
        Args:
            config: MPC configuration parameters
            state_dim: Dimension of system state
            control_dim: Dimension of control input
            dynamics_model: System dynamics function f(x, u)
        """
        self.config = config
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dynamics_model = dynamics_model
        
        # Cost matrices
        self.Q: Optional[np.ndarray] = None  # State cost matrix
        self.R: Optional[np.ndarray] = None  # Control cost matrix
        self.P: Optional[np.ndarray] = None  # Terminal cost matrix
        
        # Constraint matrices
        self.state_bounds: Optional[Bounds] = None
        self.control_bounds: Optional[Bounds] = None
        self.safety_constraints: List[NonlinearConstraint] = []
        
        # Warm start data
        self.previous_solution: Optional[np.ndarray] = None
        self.previous_states: Optional[np.ndarray] = None
        
        # Performance tracking
        self.solve_times: List[float] = []
        self.constraint_violations_history: List[Dict[str, float]] = []
        
        # Human intent integration
        self.human_intent_predictor: Optional[Callable] = None
        self.adaptive_weights: bool = False
        
        # State-space model parameters
        self.A_matrix: Optional[np.ndarray] = None  # Discrete-time A matrix
        self.B_matrix: Optional[np.ndarray] = None  # Discrete-time B matrix
        self.C_matrix: Optional[np.ndarray] = None  # Output matrix
        self.D_matrix: Optional[np.ndarray] = None  # Feedthrough matrix
        
        # Stability and recovery
        self.lyapunov_function: Optional[Callable] = None
        self.terminal_invariant_set: Optional[np.ndarray] = None
        self.constraint_violation_history: deque = deque(maxlen=10)
        self.recovery_mode: bool = False
        
        # Initialize discrete-time state-space model if linear
        self._initialize_state_space_model()
        
        logger.info(f"Initialized MPC controller with state_dim={state_dim}, "
                   f"control_dim={control_dim}, horizon={config.prediction_horizon}")
    
    def set_objective_function(self,
                             Q: np.ndarray,
                             R: np.ndarray,
                             P: Optional[np.ndarray] = None,
                             cost_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Set the quadratic cost function matrices.
        
        The cost function is:
        J = ∑(k=0 to N-1) [x_k^T Q x_k + u_k^T R u_k] + x_N^T P x_N
        
        Args:
            Q: State cost matrix (n x n, positive semi-definite)
            R: Control cost matrix (m x m, positive definite)
            P: Terminal cost matrix (n x n, positive semi-definite)
            cost_weights: Additional cost weight factors
        """
        # Validate dimensions
        if Q.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Q matrix must be {self.state_dim}x{self.state_dim}")
        if R.shape != (self.control_dim, self.control_dim):
            raise ValueError(f"R matrix must be {self.control_dim}x{self.control_dim}")
        
        # Validate positive definiteness
        if not np.all(np.linalg.eigvals(Q) >= -1e-10):  # Allow small numerical errors
            warnings.warn("Q matrix is not positive semi-definite")
        if not np.all(np.linalg.eigvals(R) > 1e-10):
            raise ValueError("R matrix must be positive definite")
        
        self.Q = Q.copy()
        self.R = R.copy()
        
        # Set terminal cost matrix
        if P is not None:
            if P.shape != (self.state_dim, self.state_dim):
                raise ValueError(f"P matrix must be {self.state_dim}x{self.state_dim}")
            if not np.all(np.linalg.eigvals(P) >= -1e-10):
                warnings.warn("P matrix is not positive semi-definite")
            self.P = P.copy()
        else:
            # Use algebraic Riccati equation solution if available
            # For now, use scaled Q matrix as default
            self.P = self.config.terminal_weight_factor * Q
        
        # Apply cost weight factors
        if cost_weights:
            if 'state_weight' in cost_weights:
                self.Q *= cost_weights['state_weight']
            if 'control_weight' in cost_weights:
                self.R *= cost_weights['control_weight']
            if 'terminal_weight' in cost_weights and self.P is not None:
                self.P *= cost_weights['terminal_weight']
        
        logger.info("Set objective function matrices")
        logger.debug(f"Q condition number: {np.linalg.cond(Q):.2e}")
        logger.debug(f"R condition number: {np.linalg.cond(R):.2e}")
        
        # Compute terminal cost using discrete algebraic Riccati equation if possible
        if self.A_matrix is not None and self.B_matrix is not None:
            self._compute_terminal_cost_are()
    
    def update_dynamics_model(self,
                            dynamics_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                            jacobian_function: Optional[Callable] = None) -> None:
        """
        Update the system dynamics model.
        
        Args:
            dynamics_function: Function f(x, u) returning x_{k+1}
            jacobian_function: Function returning (∂f/∂x, ∂f/∂u) if available
        """
        self.dynamics_model = dynamics_function
        self.jacobian_function = jacobian_function
        
        logger.info("Updated dynamics model")
    
    def set_constraints(self,
                       state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                       control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                       safety_constraints: Optional[List[NonlinearConstraint]] = None) -> None:
        """
        Set system constraints.
        
        Args:
            state_bounds: (lower, upper) bounds for states
            control_bounds: (lower, upper) bounds for controls
            safety_constraints: List of nonlinear safety constraints
        """
        if state_bounds:
            self.state_bounds = Bounds(state_bounds[0], state_bounds[1])
            logger.debug(f"Set state bounds: [{state_bounds[0].min():.3f}, {state_bounds[1].max():.3f}]")
        
        if control_bounds:
            self.control_bounds = Bounds(control_bounds[0], control_bounds[1])
            logger.debug(f"Set control bounds: [{control_bounds[0].min():.3f}, {control_bounds[1].max():.3f}]")
        
        if safety_constraints:
            self.safety_constraints = safety_constraints.copy()
            logger.info(f"Set {len(safety_constraints)} safety constraints")
    
    def solve_mpc(self,
                  current_state: np.ndarray,
                  reference_trajectory: Optional[np.ndarray] = None,
                  predicted_human_intent: Optional[Dict] = None,
                  horizon_steps: Optional[int] = None) -> MPCResult:
        """
        Solve the MPC optimization problem.
        
        This method formulates and solves the finite horizon optimal control problem:
        
        minimize     ∑(k=0 to N-1) [‖x_k - x_ref‖²_Q + ‖u_k‖²_R] + ‖x_N - x_ref‖²_P
        subject to   x_{k+1} = f(x_k, u_k)
                     u_min ≤ u_k ≤ u_max
                     x_min ≤ x_k ≤ x_max
                     g_safety(x_k, u_k) ≤ 0
        
        Args:
            current_state: Current system state x₀
            reference_trajectory: Desired state trajectory (N+1 x n)
            predicted_human_intent: Human intent prediction data
            horizon_steps: Prediction horizon (overrides config if provided)
        
        Returns:
            MPCResult containing optimal control sequence and diagnostics
        """
        start_time = time.time()
        
        # Validate inputs
        if current_state.shape[0] != self.state_dim:
            raise ValueError(f"Current state must have dimension {self.state_dim}")
        if self.Q is None or self.R is None:
            raise ValueError("Objective function matrices must be set before solving")
        if self.dynamics_model is None and self.A_matrix is None:
            raise ValueError("Either dynamics model or state-space matrices must be set before solving")
        
        # Set horizon
        N = horizon_steps if horizon_steps is not None else self.config.prediction_horizon
        
        # Prepare reference trajectory
        if reference_trajectory is None:
            # Use current state as constant reference
            x_ref = np.tile(current_state.reshape(-1, 1), (1, N + 1)).T
        else:
            if reference_trajectory.shape != (N + 1, self.state_dim):
                raise ValueError(f"Reference trajectory must be {N+1} x {self.state_dim}")
            x_ref = reference_trajectory
        
        # Adapt cost weights based on human intent uncertainty if enabled
        Q_adapted, R_adapted = self._adapt_cost_weights(predicted_human_intent)
        
        # Select solver and solve
        try:
            if self.config.solver_type == SolverType.CVXPY_QP:
                result = self._solve_cvxpy_qp(current_state, x_ref, Q_adapted, R_adapted, N)
            elif self.config.solver_type == SolverType.SCIPY_MINIMIZE:
                result = self._solve_scipy_minimize(current_state, x_ref, Q_adapted, R_adapted, N)
            else:
                raise NotImplementedError(f"Solver {self.config.solver_type} not implemented")
            
            # Check if infeasible and recovery is enabled
            if (result.status == MPCStatus.INFEASIBLE and 
                self.config.enable_feasibility_recovery and 
                not self.recovery_mode):
                
                logger.warning("Primary solver infeasible, attempting recovery")
                self.recovery_mode = True
                
                # Attempt feasibility recovery
                recovery_result = self._feasibility_recovery_solve(
                    current_state, x_ref, Q_adapted, R_adapted, N)
                
                self.recovery_mode = False
                
                if recovery_result.status != MPCStatus.EMERGENCY_STOP:
                    result = recovery_result
                
            # Check for constraint violations if solution exists
            if result.optimal_control is not None and result.predicted_states is not None:
                violations = self._check_constraint_violations(
                    result.predicted_states, result.optimal_control)
                result.constraint_violations = violations
                
                # Store violation history
                self.constraint_violation_history.append(violations)
                
                # Compute Lyapunov metrics if applicable
                if self.config.enable_lyapunov_constraints and len(result.predicted_states) > 1:
                    V_current = self._compute_lyapunov_function(result.predicted_states[0])
                    V_next = self._compute_lyapunov_function(result.predicted_states[1]) 
                    result.lyapunov_decrease = V_current - V_next
                    result.stability_margin = min(0, result.lyapunov_decrease)
            
            # Update warm start data if solution is feasible
            if result.status in [MPCStatus.OPTIMAL, MPCStatus.FEASIBLE, 
                               MPCStatus.CONSTRAINT_VIOLATION_RECOVERY]:
                self.previous_solution = result.optimal_control
                self.previous_states = result.predicted_states
            
            # Track performance
            result.solve_time = time.time() - start_time
            self.solve_times.append(result.solve_time)
            
            # Check real-time constraint
            if result.solve_time > self.config.max_solve_time:
                logger.warning(f"MPC solve time {result.solve_time:.3f}s exceeds real-time limit "
                             f"{self.config.max_solve_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"MPC solve failed: {e}")
            return MPCResult(status=MPCStatus.ERROR, solve_time=time.time() - start_time)
    
    def get_control_sequence(self, mpc_result: MPCResult, num_controls: Optional[int] = None) -> np.ndarray:
        """
        Extract control sequence from MPC result.
        
        Args:
            mpc_result: Result from solve_mpc()
            num_controls: Number of control steps to return (default: control horizon)
        
        Returns:
            Control sequence array (num_controls x control_dim)
        """
        if mpc_result.optimal_control is None:
            raise ValueError("No optimal control available in MPC result")
        
        if num_controls is None:
            num_controls = min(self.config.control_horizon, len(mpc_result.optimal_control))
        
        return mpc_result.optimal_control[:num_controls]
    
    def _adapt_cost_weights(self, predicted_human_intent: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adapt cost weights based on human intent uncertainty.
        
        Args:
            predicted_human_intent: Dict with 'intent_probs' and 'uncertainty'
        
        Returns:
            Adapted (Q, R) matrices
        """
        Q_adapted = self.Q.copy()
        R_adapted = self.R.copy()
        
        if predicted_human_intent and self.adaptive_weights:
            uncertainty = predicted_human_intent.get('uncertainty', 0.0)
            
            # Increase safety weight with higher uncertainty
            safety_factor = 1.0 + uncertainty * self.config.safety_weight_factor
            Q_adapted *= safety_factor
            
            # Reduce control aggressiveness with higher uncertainty
            control_factor = 1.0 + uncertainty * 0.5
            R_adapted *= control_factor
            
            logger.debug(f"Adapted weights for uncertainty {uncertainty:.3f}: "
                        f"safety_factor={safety_factor:.3f}, control_factor={control_factor:.3f}")
        
        return Q_adapted, R_adapted
    
    def _solve_cvxpy_qp(self,
                       current_state: np.ndarray,
                       x_ref: np.ndarray,
                       Q: np.ndarray,
                       R: np.ndarray,
                       N: int) -> MPCResult:
        """
        Solve MPC using CVXPY quadratic programming.
        
        This formulates the MPC problem as a QP:
        minimize     z^T H z + f^T z
        subject to   A_eq z = b_eq    (dynamics)
                     A_ineq z ≤ b_ineq (box constraints)
        
        where z = [u_0; u_1; ...; u_{N-1}; x_1; x_2; ...; x_N]
        """
        try:
            # Decision variables
            u = cp.Variable((N, self.control_dim))  # Control sequence
            x = cp.Variable((N + 1, self.state_dim))  # State sequence
            
            # Cost function
            cost = 0
            for k in range(N):
                # Stage cost: (x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k
                cost += cp.quad_form(x[k] - x_ref[k], Q)
                cost += cp.quad_form(u[k], R)
            
            # Terminal cost
            if self.P is not None:
                cost += cp.quad_form(x[N] - x_ref[N], self.P)
            
            # Constraints
            constraints = []
            
            # Initial condition
            constraints.append(x[0] == current_state)
            
            # System dynamics
            for k in range(N):
                if self.A_matrix is not None and self.B_matrix is not None:
                    # Use discrete-time state-space model: x[k+1] = A*x[k] + B*u[k]
                    constraints.append(x[k+1] == self.A_matrix @ x[k] + self.B_matrix @ u[k])
                elif hasattr(self, '_linearize_dynamics'):
                    # Use linearized dynamics around reference
                    A, B = self._linearize_dynamics(x_ref[k], np.zeros(self.control_dim))
                    constraints.append(x[k+1] == A @ x[k] + B @ u[k])
                else:
                    # Simple integrator fallback: x[k+1] = x[k] + Ts*u[k]
                    constraints.append(x[k+1] == x[k] + self.config.sampling_time * u[k])
            
            # Box constraints
            if self.control_bounds:
                for k in range(N):
                    constraints.append(u[k] >= self.control_bounds.lb)
                    constraints.append(u[k] <= self.control_bounds.ub)
            
            if self.state_bounds:
                for k in range(N + 1):
                    constraints.append(x[k] >= self.state_bounds.lb)
                    constraints.append(x[k] <= self.state_bounds.ub)
            
            # Add Lyapunov stability constraints
            self._add_lyapunov_stability_constraints(x, constraints)
            
            # Formulate and solve problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            
            # Solver settings
            solver_kwargs = {
                'max_iters': self.config.max_iterations,
                'eps': self.config.solver_tolerance,
                'verbose': False
            }
            
            # Add warm start if available
            if self.config.use_warm_start and self.previous_solution is not None:
                # Shift previous solution for warm start
                u_warm = np.roll(self.previous_solution, -1, axis=0)
                u_warm[-1] = self.previous_solution[-1]  # Repeat last control
                u.value = u_warm[:N]
            
            # Solve with timeout
            problem.solve(solver=cp.OSQP, **solver_kwargs)
            
            # Process results
            if problem.status == cp.OPTIMAL:
                status = MPCStatus.OPTIMAL
            elif problem.status == cp.OPTIMAL_INACCURATE:
                status = MPCStatus.FEASIBLE
                logger.warning("CVXPY solver returned inaccurate solution")
            else:
                status = MPCStatus.INFEASIBLE
                logger.warning(f"CVXPY solver status: {problem.status}")
                return MPCResult(status=status, solver_info={'cvxpy_status': problem.status})
            
            # Extract solution
            optimal_control = u.value if u.value is not None else None
            predicted_states = x.value if x.value is not None else None
            optimal_cost = problem.value if problem.value is not None else None
            
            return MPCResult(
                status=status,
                optimal_control=optimal_control,
                predicted_states=predicted_states,
                optimal_cost=optimal_cost,
                solver_info={
                    'cvxpy_status': problem.status,
                    'solver_stats': problem.solver_stats
                }
            )
            
        except Exception as e:
            logger.error(f"CVXPY solver failed: {e}")
            return MPCResult(status=MPCStatus.ERROR, solver_info={'error': str(e)})
    
    def _solve_scipy_minimize(self,
                            current_state: np.ndarray,
                            x_ref: np.ndarray,
                            Q: np.ndarray,
                            R: np.ndarray,
                            N: int) -> MPCResult:
        """
        Solve MPC using scipy.optimize.minimize for nonlinear MPC.
        
        This method can handle nonlinear dynamics and constraints.
        """
        try:
            # Decision variables: [u_0, u_1, ..., u_{N-1}]
            decision_var_dim = N * self.control_dim
            
            # Initial guess
            if self.config.use_warm_start and self.previous_solution is not None:
                # Shift and pad previous solution
                u0 = np.roll(self.previous_solution.flatten(), -self.control_dim)
                u0[-self.control_dim:] = self.previous_solution[-1]  # Repeat last control
                u0 = u0[:decision_var_dim]
            else:
                u0 = np.zeros(decision_var_dim)
            
            # Objective function
            def objective(u_flat):
                u_sequence = u_flat.reshape(N, self.control_dim)
                
                # Simulate forward dynamics
                x_sequence = np.zeros((N + 1, self.state_dim))
                x_sequence[0] = current_state
                
                for k in range(N):
                    x_sequence[k+1] = self.dynamics_model(x_sequence[k], u_sequence[k])
                
                # Compute cost
                cost = 0.0
                for k in range(N):
                    x_err = x_sequence[k] - x_ref[k]
                    cost += x_err.T @ Q @ x_err + u_sequence[k].T @ R @ u_sequence[k]
                
                # Terminal cost
                if self.P is not None:
                    x_err_N = x_sequence[N] - x_ref[N]
                    cost += x_err_N.T @ self.P @ x_err_N
                
                return cost
            
            # Bounds
            bounds = None
            if self.control_bounds:
                bounds = Bounds(
                    np.tile(self.control_bounds.lb, N),
                    np.tile(self.control_bounds.ub, N)
                )
            
            # Solve optimization
            result = minimize(
                objective,
                u0,
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.solver_tolerance,
                    'disp': False
                }
            )
            
            if result.success:
                status = MPCStatus.OPTIMAL
            elif result.fun < np.inf:
                status = MPCStatus.FEASIBLE
                logger.warning("Scipy solver did not fully converge")
            else:
                status = MPCStatus.INFEASIBLE
            
            # Extract solution
            optimal_control = result.x.reshape(N, self.control_dim)
            
            # Reconstruct state trajectory
            predicted_states = np.zeros((N + 1, self.state_dim))
            predicted_states[0] = current_state
            for k in range(N):
                predicted_states[k+1] = self.dynamics_model(predicted_states[k], optimal_control[k])
            
            return MPCResult(
                status=status,
                optimal_control=optimal_control,
                predicted_states=predicted_states,
                optimal_cost=result.fun,
                iterations=result.nit,
                solver_info={
                    'scipy_result': result,
                    'message': result.message
                }
            )
            
        except Exception as e:
            logger.error(f"Scipy solver failed: {e}")
            return MPCResult(status=MPCStatus.ERROR, solver_info={'error': str(e)})
    
    def emergency_stop(self) -> np.ndarray:
        """
        Generate emergency stop control command.
        
        Returns:
            Zero control input for immediate stop
        """
        logger.warning("Emergency stop activated!")
        return np.zeros(self.control_dim)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get controller performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.solve_times:
            return {}
        
        return {
            'mean_solve_time': np.mean(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'min_solve_time': np.min(self.solve_times),
            'std_solve_time': np.std(self.solve_times),
            'real_time_violations': sum(1 for t in self.solve_times if t > self.config.max_solve_time),
            'success_rate': sum(1 for _ in self.solve_times) / len(self.solve_times) if self.solve_times else 0
        }
    
    def reset_warm_start(self) -> None:
        """Reset warm start data."""
        self.previous_solution = None
        self.previous_states = None
        logger.info("Reset warm start data")
    
    def set_human_intent_predictor(self, predictor: Callable) -> None:
        """
        Set human intent predictor function.
        
        Args:
            predictor: Function that returns intent prediction dict
        """
        self.human_intent_predictor = predictor
        self.adaptive_weights = True
        logger.info("Set human intent predictor for adaptive MPC")
    
    def _initialize_state_space_model(self) -> None:
        """
        Initialize discrete-time state-space model matrices.
        
        For a continuous-time system: ẋ = Ac*x + Bc*u + w
        Discretize to: x[k+1] = A*x[k] + B*u[k] + w[k]
        
        Where A = exp(Ac*Ts), B = ∫[0,Ts] exp(Ac*τ) dτ * Bc
        """
        # Default to identity dynamics if no specific model is provided
        # This will be overridden when set_state_space_model is called
        self.A_matrix = np.eye(self.state_dim)
        self.B_matrix = np.eye(self.state_dim, self.control_dim) * self.config.sampling_time
        self.C_matrix = np.eye(self.state_dim)  # Full state observation
        self.D_matrix = np.zeros((self.state_dim, self.control_dim))
        
        logger.debug("Initialized default identity state-space model")
    
    def set_state_space_model(self,
                             A: np.ndarray,
                             B: np.ndarray,
                             C: Optional[np.ndarray] = None,
                             D: Optional[np.ndarray] = None) -> None:
        """
        Set discrete-time state-space model matrices.
        
        System dynamics: x[k+1] = A*x[k] + B*u[k] + w[k]
        Output equation: y[k] = C*x[k] + D*u[k] + v[k]
        
        Args:
            A: State transition matrix (n × n)
            B: Control input matrix (n × m)  
            C: Output matrix (p × n), defaults to identity
            D: Feedthrough matrix (p × m), defaults to zero
        """
        # Validate dimensions
        if A.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"A matrix must be {self.state_dim}×{self.state_dim}")
        if B.shape != (self.state_dim, self.control_dim):
            raise ValueError(f"B matrix must be {self.state_dim}×{self.control_dim}")
        
        # Check controllability (Kalman rank condition)
        controllability_matrix = self._compute_controllability_matrix(A, B)
        if np.linalg.matrix_rank(controllability_matrix) < self.state_dim:
            warnings.warn("System may not be completely controllable")
        
        # Check stability of A matrix
        eigenvalues = np.linalg.eigvals(A)
        if np.any(np.abs(eigenvalues) >= 1.0):
            warnings.warn("Open-loop system is unstable (eigenvalues outside unit circle)")
            logger.warning(f"Max eigenvalue magnitude: {np.max(np.abs(eigenvalues)):.3f}")
        
        self.A_matrix = A.copy()
        self.B_matrix = B.copy()
        
        if C is not None:
            if C.shape[1] != self.state_dim:
                raise ValueError(f"C matrix must have {self.state_dim} columns")
            self.C_matrix = C.copy()
        else:
            self.C_matrix = np.eye(self.state_dim)
        
        if D is not None:
            if D.shape != (self.C_matrix.shape[0], self.control_dim):
                raise ValueError(f"D matrix must be {self.C_matrix.shape[0]}×{self.control_dim}")
            self.D_matrix = D.copy()
        else:
            self.D_matrix = np.zeros((self.C_matrix.shape[0], self.control_dim))
        
        logger.info("Set discrete-time state-space model")
        logger.debug(f"A matrix condition number: {np.linalg.cond(A):.2e}")
        logger.debug(f"Controllability matrix rank: {np.linalg.matrix_rank(controllability_matrix)}/{self.state_dim}")
        
        # Recompute terminal cost with new model
        if self.Q is not None and self.R is not None:
            self._compute_terminal_cost_are()
    
    def _compute_controllability_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute controllability matrix: [B, AB, A²B, ..., A^(n-1)B]
        
        Args:
            A: State transition matrix
            B: Control input matrix
            
        Returns:
            Controllability matrix
        """
        n = A.shape[0]
        controllability = B.copy()
        
        A_power = A.copy()
        for i in range(1, n):
            controllability = np.hstack([controllability, A_power @ B])
            A_power = A_power @ A
        
        return controllability
    
    def _compute_terminal_cost_are(self) -> None:
        """
        Compute terminal cost matrix P using discrete algebraic Riccati equation (DARE).
        
        DARE: P = A^T P A - A^T P B (R + B^T P B)^(-1) B^T P A + Q
        
        This ensures closed-loop stability and optimality.
        """
        if self.A_matrix is None or self.B_matrix is None:
            logger.warning("Cannot compute ARE solution without state-space model")
            return
        
        try:
            # Solve discrete algebraic Riccati equation
            P_are = solve_discrete_are(self.A_matrix, self.B_matrix, self.Q, self.R)
            
            # Verify solution is positive semi-definite
            eigenvals = np.linalg.eigvals(P_are)
            if np.all(eigenvals >= -1e-10):
                self.P = P_are
                
                # Compute optimal feedback gain K = (R + B^T P B)^(-1) B^T P A
                K_optimal = inv(self.R + self.B_matrix.T @ P_are @ self.B_matrix) @ \
                           self.B_matrix.T @ P_are @ self.A_matrix
                
                # Check closed-loop stability
                A_cl = self.A_matrix - self.B_matrix @ K_optimal
                cl_eigenvals = np.linalg.eigvals(A_cl)
                
                if np.all(np.abs(cl_eigenvals) < 1.0):
                    logger.info("Computed stabilizing terminal cost from DARE")
                    logger.debug(f"Max closed-loop eigenvalue: {np.max(np.abs(cl_eigenvals)):.3f}")
                else:
                    logger.warning("DARE solution does not yield stable closed-loop system")
                    
            else:
                logger.warning("DARE solution is not positive semi-definite")
                
        except Exception as e:
            logger.warning(f"Failed to solve DARE: {e}. Using scaled Q matrix.")
            self.P = self.config.terminal_weight_factor * self.Q
    
    def _check_constraint_violations(self, 
                                   states: np.ndarray, 
                                   controls: np.ndarray) -> Dict[str, float]:
        """
        Check for constraint violations in predicted trajectory.
        
        Args:
            states: State trajectory (N+1 × n)
            controls: Control sequence (N × m)
            
        Returns:
            Dictionary of constraint violations
        """
        violations = {}
        
        # State bound violations
        if self.state_bounds is not None:
            state_lb_viol = np.maximum(0, self.state_bounds.lb - states).max()
            state_ub_viol = np.maximum(0, states - self.state_bounds.ub).max()
            violations['state_lower'] = float(state_lb_viol)
            violations['state_upper'] = float(state_ub_viol)
        
        # Control bound violations  
        if self.control_bounds is not None:
            control_lb_viol = np.maximum(0, self.control_bounds.lb - controls).max()
            control_ub_viol = np.maximum(0, controls - self.control_bounds.ub).max()
            violations['control_lower'] = float(control_lb_viol)
            violations['control_upper'] = float(control_ub_viol)
        
        # Safety constraint violations
        for i, constraint in enumerate(self.safety_constraints):
            try:
                violation = constraint.fun(states, controls)
                if hasattr(violation, '__iter__'):
                    violations[f'safety_{i}'] = float(np.maximum(0, violation).max())
                else:
                    violations[f'safety_{i}'] = float(max(0, violation))
            except Exception as e:
                logger.warning(f"Error evaluating safety constraint {i}: {e}")
        
        return violations
    
    def _feasibility_recovery_solve(self,
                                  current_state: np.ndarray,
                                  x_ref: np.ndarray, 
                                  Q: np.ndarray,
                                  R: np.ndarray,
                                  N: int) -> MPCResult:
        """
        Solve MPC with constraint violation recovery using slack variables.
        
        The relaxed problem becomes:
        minimize     J_original + ρ * ||ε||₁
        subject to   x_{k+1} = f(x_k, u_k)
                     u_min ≤ u_k ≤ u_max
                     x_min - ε_x ≤ x_k ≤ x_max + ε_x
                     g_safety(x_k, u_k) ≤ ε_g
                     ε_x, ε_g ≥ 0
        
        Args:
            current_state: Current system state
            x_ref: Reference trajectory 
            Q, R: Cost matrices
            N: Prediction horizon
            
        Returns:
            Recovery MPC result
        """
        logger.warning("Activating feasibility recovery mode")
        
        try:
            if self.config.solver_type == SolverType.CVXPY_QP:
                return self._solve_recovery_cvxpy(current_state, x_ref, Q, R, N)
            else:
                # Fallback to emergency stop if recovery not implemented for solver
                logger.error("Feasibility recovery not implemented for current solver")
                return MPCResult(
                    status=MPCStatus.EMERGENCY_STOP,
                    optimal_control=np.zeros((1, self.control_dim)),
                    feasibility_recovery_used=True
                )
    
    def _solve_recovery_cvxpy(self,
                            current_state: np.ndarray,
                            x_ref: np.ndarray,
                            Q: np.ndarray, 
                            R: np.ndarray,
                            N: int) -> MPCResult:
        """
        Solve feasibility recovery using CVXPY with slack variables.
        """
        # Decision variables
        u = cp.Variable((N, self.control_dim))
        x = cp.Variable((N + 1, self.state_dim))
        
        # Slack variables for constraint relaxation
        eps_x_lower = cp.Variable((N + 1, self.state_dim), nonneg=True)
        eps_x_upper = cp.Variable((N + 1, self.state_dim), nonneg=True) 
        eps_u_lower = cp.Variable((N, self.control_dim), nonneg=True)
        eps_u_upper = cp.Variable((N, self.control_dim), nonneg=True)
        
        # Original cost
        cost = 0
        for k in range(N):
            cost += cp.quad_form(x[k] - x_ref[k], Q)
            cost += cp.quad_form(u[k], R)
        
        if self.P is not None:
            cost += cp.quad_form(x[N] - x_ref[N], self.P)
        
        # Slack penalty (L1 norm)
        slack_penalty = self.config.recovery_slack_penalty
        cost += slack_penalty * (cp.sum(eps_x_lower) + cp.sum(eps_x_upper) + 
                                cp.sum(eps_u_lower) + cp.sum(eps_u_upper))
        
        # Constraints
        constraints = [x[0] == current_state]
        
        # Dynamics
        for k in range(N):
            if self.A_matrix is not None:
                constraints.append(x[k+1] == self.A_matrix @ x[k] + self.B_matrix @ u[k])
            else:
                # Simple integrator fallback
                constraints.append(x[k+1] == x[k] + self.config.sampling_time * u[k])
        
        # Relaxed state bounds
        if self.state_bounds is not None:
            for k in range(N + 1):
                constraints.append(x[k] >= self.state_bounds.lb - eps_x_lower[k])
                constraints.append(x[k] <= self.state_bounds.ub + eps_x_upper[k])
        
        # Relaxed control bounds  
        if self.control_bounds is not None:
            for k in range(N):
                constraints.append(u[k] >= self.control_bounds.lb - eps_u_lower[k])
                constraints.append(u[k] <= self.control_bounds.ub + eps_u_upper[k])
        
        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            status = MPCStatus.CONSTRAINT_VIOLATION_RECOVERY
            
            # Check if slack variables were used
            slack_used = (cp.sum(eps_x_lower).value + cp.sum(eps_x_upper).value + 
                         cp.sum(eps_u_lower).value + cp.sum(eps_u_upper).value) > 1e-6
            
            if slack_used:
                logger.warning(f"Constraint relaxation used, total slack: "
                              f"{cp.sum(eps_x_lower).value + cp.sum(eps_x_upper).value:.3e}")
                
            return MPCResult(
                status=status,
                optimal_control=u.value,
                predicted_states=x.value, 
                optimal_cost=problem.value,
                feasibility_recovery_used=True,
                solver_info={'cvxpy_status': problem.status}
            )
        else:
            logger.error(f"Recovery solver failed with status: {problem.status}")
            return MPCResult(
                status=MPCStatus.EMERGENCY_STOP,
                optimal_control=np.zeros((1, self.control_dim)),
                feasibility_recovery_used=True
            )
    
    def _compute_lyapunov_function(self, state: np.ndarray) -> float:
        """
        Compute Lyapunov function value V(x) = x^T P x.
        
        Args:
            state: System state
            
        Returns:
            Lyapunov function value
        """
        if self.P is None:
            # Use quadratic form with Q matrix as fallback
            return float(state.T @ self.Q @ state)
        return float(state.T @ self.P @ state)
    
    def _add_lyapunov_stability_constraints(self,
                                          x: cp.Variable,
                                          constraints: List) -> None:
        """
        Add Lyapunov stability constraints to ensure decrease condition:
        V(x_{k+1}) - V(x_k) ≤ -α||x_k||²
        
        Args:
            x: State variable sequence
            constraints: List to append constraints to
        """
        if not self.config.enable_lyapunov_constraints or self.P is None:
            return
        
        alpha = self.config.lyapunov_decrease_rate
        
        for k in range(len(x) - 1):
            # V(x_{k+1}) - V(x_k) ≤ -α||x_k||²
            V_k = cp.quad_form(x[k], self.P)  
            V_k1 = cp.quad_form(x[k+1], self.P)
            constraints.append(V_k1 - V_k <= -alpha * cp.sum_squares(x[k]))
        
        # Terminal set constraint: x_N ∈ X_f
        if self.config.terminal_set_constraint and self.terminal_invariant_set is not None:
            # For simplicity, use ellipsoidal terminal set: x_N^T P x_N ≤ β
            terminal_level = 1.0  # This should be computed based on control constraints
            constraints.append(cp.quad_form(x[-1], self.P) <= terminal_level)