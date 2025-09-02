"""
Algorithm Optimization Module
Model-Based RL Human Intent Recognition System

This module provides optimized implementations for computational intensive algorithms
including Gaussian Process inference, Model Predictive Control, and Bayesian RL components.
"""

import numpy as np
import scipy.linalg
import scipy.optimize
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import logging
from abc import ABC, abstractmethod
import numba
from numba import jit, njit, prange
import warnings

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class OptimizationConfig:
    """Configuration for algorithm optimization."""
    use_parallel: bool = True
    num_workers: int = mp.cpu_count()
    use_gpu: bool = TORCH_AVAILABLE
    use_numba: bool = True
    cache_size: int = 1000
    approximation_threshold: float = 1e-6
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-8


class GPInferenceOptimizer:
    """Optimized Gaussian Process inference."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.kernel_cache = {}
        self.cholesky_cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU support if available
        self.device = None
        if self.config.use_gpu and TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray, 
                              kernel_params: Dict[str, float]) -> np.ndarray:
        """Optimized kernel matrix computation."""
        cache_key = self._get_cache_key(X1, X2, kernel_params)
        
        if cache_key in self.kernel_cache:
            return self.kernel_cache[cache_key]
        
        if self.config.use_gpu and TORCH_AVAILABLE:
            K = self._compute_kernel_gpu(X1, X2, kernel_params)
        elif self.config.use_numba:
            K = self._compute_kernel_numba(X1, X2, kernel_params)
        else:
            K = self._compute_kernel_standard(X1, X2, kernel_params)
        
        # Cache result if cache not full
        if len(self.kernel_cache) < self.config.cache_size:
            self.kernel_cache[cache_key] = K
        
        return K
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _compute_rbf_kernel_numba(X1: np.ndarray, X2: np.ndarray, 
                                 length_scale: float, variance: float) -> np.ndarray:
        """Numba-optimized RBF kernel computation."""
        n1, d = X1.shape
        n2 = X2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in prange(n1):
            for j in prange(n2):
                dist_sq = 0.0
                for k in range(d):
                    diff = X1[i, k] - X2[j, k]
                    dist_sq += diff * diff
                
                K[i, j] = variance * np.exp(-0.5 * dist_sq / (length_scale * length_scale))
        
        return K
    
    def _compute_kernel_gpu(self, X1: np.ndarray, X2: np.ndarray, 
                           kernel_params: Dict[str, float]) -> np.ndarray:
        """GPU-accelerated kernel computation."""
        X1_torch = torch.tensor(X1, dtype=torch.float32, device=self.device)
        X2_torch = torch.tensor(X2, dtype=torch.float32, device=self.device)
        
        # Compute squared distances
        dist_sq = torch.cdist(X1_torch, X2_torch, p=2) ** 2
        
        # RBF kernel
        length_scale = kernel_params.get('length_scale', 1.0)
        variance = kernel_params.get('variance', 1.0)
        
        K = variance * torch.exp(-0.5 * dist_sq / (length_scale ** 2))
        
        return K.cpu().numpy()
    
    def _compute_kernel_numba(self, X1: np.ndarray, X2: np.ndarray, 
                            kernel_params: Dict[str, float]) -> np.ndarray:
        """Numba-optimized kernel computation."""
        length_scale = kernel_params.get('length_scale', 1.0)
        variance = kernel_params.get('variance', 1.0)
        
        return self._compute_rbf_kernel_numba(X1, X2, length_scale, variance)
    
    def _compute_kernel_standard(self, X1: np.ndarray, X2: np.ndarray, 
                               kernel_params: Dict[str, float]) -> np.ndarray:
        """Standard kernel computation."""
        from scipy.spatial.distance import cdist
        
        length_scale = kernel_params.get('length_scale', 1.0)
        variance = kernel_params.get('variance', 1.0)
        
        dist = cdist(X1, X2)
        K = variance * np.exp(-0.5 * (dist / length_scale) ** 2)
        
        return K
    
    def _get_cache_key(self, X1: np.ndarray, X2: np.ndarray, 
                      kernel_params: Dict[str, float]) -> str:
        """Generate cache key for kernel computation."""
        X1_hash = hash(X1.data.tobytes())
        X2_hash = hash(X2.data.tobytes())
        params_hash = hash(tuple(sorted(kernel_params.items())))
        
        return f"{X1_hash}_{X2_hash}_{params_hash}"
    
    def _stable_cholesky(self, K: np.ndarray, noise_var: float) -> np.ndarray:
        """Numerically stable Cholesky decomposition."""
        cache_key = hash((K.data.tobytes(), noise_var))
        
        if cache_key in self.cholesky_cache:
            return self.cholesky_cache[cache_key]
        
        try:
            # Add noise for numerical stability
            K_noisy = K + (noise_var + 1e-6) * np.eye(K.shape[0])
            L = scipy.linalg.cholesky(K_noisy, lower=True)
        except scipy.linalg.LinAlgError:
            # Fallback to eigenvalue regularization
            eigenvals, eigenvecs = scipy.linalg.eigh(K)
            eigenvals = np.maximum(eigenvals, 1e-6)
            K_reg = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            K_noisy = K_reg + noise_var * np.eye(K.shape[0])
            L = scipy.linalg.cholesky(K_noisy, lower=True)
        
        # Cache result
        if len(self.cholesky_cache) < self.config.cache_size:
            self.cholesky_cache[cache_key] = L
        
        return L
    
    def predict(self, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, kernel_params: Dict[str, float],
                noise_var: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized GP prediction."""
        start_time = time.time()
        
        # Compute kernel matrices
        K_train = self._compute_kernel_matrix(X_train, X_train, kernel_params)
        K_test_train = self._compute_kernel_matrix(X_test, X_train, kernel_params)
        K_test = self._compute_kernel_matrix(X_test, X_test, kernel_params)
        
        # Stable Cholesky decomposition
        L = self._stable_cholesky(K_train, noise_var)
        
        # Solve for predictive mean
        alpha = scipy.linalg.solve_triangular(L, y_train, lower=True)
        mean = K_test_train @ scipy.linalg.solve_triangular(L.T, alpha, lower=False)
        
        # Solve for predictive variance
        v = scipy.linalg.solve_triangular(L, K_test_train.T, lower=True)
        var = np.diag(K_test) - np.sum(v**2, axis=0)
        
        # Ensure positive variance
        var = np.maximum(var, 1e-10)
        
        computation_time = time.time() - start_time
        self.logger.debug(f"GP prediction completed in {computation_time:.3f}s")
        
        return mean, var
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                initial_params: Dict[str, float]) -> Dict[str, float]:
        """Optimized hyperparameter optimization."""
        def objective(theta):
            params = {'length_scale': np.exp(theta[0]), 'variance': np.exp(theta[1])}
            noise_var = np.exp(theta[2])
            
            K = self._compute_kernel_matrix(X_train, X_train, params)
            
            try:
                L = self._stable_cholesky(K, noise_var)
                alpha = scipy.linalg.solve_triangular(L, y_train, lower=True)
                
                # Log marginal likelihood
                log_likelihood = -0.5 * np.sum(alpha**2) - np.sum(np.log(np.diag(L))) - \
                                0.5 * len(y_train) * np.log(2 * np.pi)
                
                return -log_likelihood  # Minimize negative log likelihood
            except:
                return np.inf
        
        # Initial parameter transformation
        theta0 = [
            np.log(initial_params.get('length_scale', 1.0)),
            np.log(initial_params.get('variance', 1.0)),
            np.log(1e-6)  # noise variance
        ]
        
        # Optimize
        result = scipy.optimize.minimize(
            objective, theta0, method='L-BFGS-B',
            options={'maxiter': self.config.max_iterations}
        )
        
        if result.success:
            optimal_params = {
                'length_scale': np.exp(result.x[0]),
                'variance': np.exp(result.x[1]),
                'noise_var': np.exp(result.x[2])
            }
        else:
            optimal_params = initial_params
            self.logger.warning("Hyperparameter optimization failed, using initial parameters")
        
        return optimal_params


class MPCOptimizer:
    """Optimized Model Predictive Control."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Warm-start storage
        self.previous_solution = None
        self.constraint_jacobian_cache = {}
    
    @staticmethod
    @njit(fastmath=True)
    def _compute_quadratic_cost_numba(states: np.ndarray, controls: np.ndarray,
                                    Q: np.ndarray, R: np.ndarray,
                                    reference: np.ndarray) -> float:
        """Numba-optimized quadratic cost computation."""
        cost = 0.0
        
        # State cost
        for t in range(states.shape[0]):
            error = states[t] - reference[t]
            cost += 0.5 * np.dot(error, Q @ error)
        
        # Control cost
        for t in range(controls.shape[0]):
            cost += 0.5 * np.dot(controls[t], R @ controls[t])
        
        return cost
    
    def _compute_system_matrices(self, dt: float, state_dim: int, 
                               control_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute discretized system matrices."""
        # Simple double integrator model (can be extended)
        A = np.eye(state_dim)
        B = np.zeros((state_dim, control_dim))
        
        # Position-velocity dynamics
        if state_dim >= 6 and control_dim >= 3:  # 3D position + velocity
            A[0:3, 3:6] = dt * np.eye(3)  # position += velocity * dt
            B[3:6, 0:3] = dt * np.eye(3)  # velocity += acceleration * dt
        
        return A, B
    
    def _setup_optimization_problem(self, initial_state: np.ndarray,
                                  reference_trajectory: np.ndarray,
                                  horizon: int, dt: float,
                                  Q: np.ndarray, R: np.ndarray,
                                  constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Setup MPC optimization problem."""
        state_dim = len(initial_state)
        control_dim = R.shape[0]
        
        # System matrices
        A, B = self._compute_system_matrices(dt, state_dim, control_dim)
        
        # Decision variables: [u_0, u_1, ..., u_{N-1}, x_1, x_2, ..., x_N]
        num_controls = horizon * control_dim
        num_states = horizon * state_dim
        num_vars = num_controls + num_states
        
        # Objective function (quadratic form)
        H = np.zeros((num_vars, num_vars))
        f = np.zeros(num_vars)
        
        # Control cost
        for t in range(horizon):
            u_idx = t * control_dim
            H[u_idx:u_idx+control_dim, u_idx:u_idx+control_dim] = R
        
        # State cost
        for t in range(horizon):
            x_idx = num_controls + t * state_dim
            H[x_idx:x_idx+state_dim, x_idx:x_idx+state_dim] = Q
            
            # Reference tracking
            if t < len(reference_trajectory):
                f[x_idx:x_idx+state_dim] = -Q @ reference_trajectory[t]
        
        # Equality constraints (system dynamics)
        A_eq = []
        b_eq = []
        
        # Initial state constraint
        x_0_constraint = np.zeros(num_vars)
        x_0_constraint[num_controls:num_controls+state_dim] = 1.0
        A_eq.append(x_0_constraint)
        b_eq.append(initial_state)
        
        # Dynamics constraints
        for t in range(horizon-1):
            dynamics_constraint = np.zeros(num_vars)
            
            # Current state: -A*x_t
            x_t_idx = num_controls + t * state_dim
            dynamics_constraint[x_t_idx:x_t_idx+state_dim] = -A
            
            # Current control: -B*u_t
            u_t_idx = t * control_dim
            dynamics_constraint[u_t_idx:u_t_idx+control_dim] = -B
            
            # Next state: x_{t+1}
            x_t1_idx = num_controls + (t+1) * state_dim
            dynamics_constraint[x_t1_idx:x_t1_idx+state_dim] = np.eye(state_dim)
            
            A_eq.append(dynamics_constraint)
            b_eq.append(np.zeros(state_dim))
        
        A_eq = np.array(A_eq) if A_eq else np.zeros((0, num_vars))
        b_eq = np.array(b_eq) if b_eq else np.zeros(0)
        
        # Inequality constraints
        A_ineq = []
        b_ineq = []
        
        # Control bounds
        if 'u_min' in constraints and 'u_max' in constraints:
            u_min = constraints['u_min']
            u_max = constraints['u_max']
            
            for t in range(horizon):
                u_idx = t * control_dim
                
                # Lower bounds: -u >= -u_max
                u_upper_constraint = np.zeros(num_vars)
                u_upper_constraint[u_idx:u_idx+control_dim] = -np.eye(control_dim)
                A_ineq.append(u_upper_constraint)
                b_ineq.append(-u_max)
                
                # Upper bounds: u >= u_min
                u_lower_constraint = np.zeros(num_vars)
                u_lower_constraint[u_idx:u_idx+control_dim] = np.eye(control_dim)
                A_ineq.append(u_lower_constraint)
                b_ineq.append(u_min)
        
        A_ineq = np.array(A_ineq) if A_ineq else np.zeros((0, num_vars))
        b_ineq = np.array(b_ineq) if b_ineq else np.zeros(0)
        
        return {
            'H': H, 'f': f,
            'A_eq': A_eq, 'b_eq': b_eq,
            'A_ineq': A_ineq, 'b_ineq': b_ineq,
            'num_controls': num_controls,
            'num_states': num_states,
            'control_dim': control_dim,
            'state_dim': state_dim,
            'horizon': horizon
        }
    
    def solve(self, initial_state: np.ndarray, reference_trajectory: np.ndarray,
              horizon: int, dt: float, Q: np.ndarray, R: np.ndarray,
              constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Solve MPC optimization problem."""
        start_time = time.time()
        
        constraints = constraints or {}
        
        # Setup optimization problem
        problem = self._setup_optimization_problem(
            initial_state, reference_trajectory, horizon, dt, Q, R, constraints
        )
        
        # Solve quadratic program
        try:
            if self.config.use_parallel:
                solution = self._solve_parallel(problem)
            else:
                solution = self._solve_sequential(problem)
        except Exception as e:
            self.logger.error(f"MPC optimization failed: {e}")
            # Return emergency stop solution
            control_dim = problem['control_dim']
            solution = np.zeros(horizon * control_dim)
        
        # Extract control sequence
        control_sequence = solution[:problem['num_controls']].reshape(
            (horizon, problem['control_dim'])
        )
        
        # Extract predicted states
        predicted_states = solution[problem['num_controls']:].reshape(
            (horizon, problem['state_dim'])
        )
        
        computation_time = time.time() - start_time
        
        # Store for warm start
        self.previous_solution = solution
        
        return {
            'optimal_controls': control_sequence,
            'predicted_states': predicted_states,
            'computation_time': computation_time,
            'success': True
        }
    
    def _solve_sequential(self, problem: Dict[str, Any]) -> np.ndarray:
        """Solve optimization problem sequentially."""
        from scipy.optimize import minimize
        
        def objective(x):
            return 0.5 * x.T @ problem['H'] @ x + problem['f'].T @ x
        
        def jacobian(x):
            return problem['H'] @ x + problem['f']
        
        # Constraints
        constraints_list = []
        
        if problem['A_eq'].size > 0:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: problem['A_eq'] @ x - problem['b_eq'],
                'jac': lambda x: problem['A_eq']
            })
        
        if problem['A_ineq'].size > 0:
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda x: problem['b_ineq'] - problem['A_ineq'] @ x,
                'jac': lambda x: -problem['A_ineq']
            })
        
        # Initial guess (warm start if available)
        x0 = self.previous_solution if self.previous_solution is not None else \
             np.zeros(problem['num_controls'] + problem['num_states'])
        
        # Solve
        result = minimize(
            objective, x0, method='SLSQP',
            jac=jacobian, constraints=constraints_list,
            options={'maxiter': self.config.max_iterations,
                    'ftol': self.config.convergence_tolerance}
        )
        
        if not result.success:
            self.logger.warning(f"MPC optimization warning: {result.message}")
        
        return result.x
    
    def _solve_parallel(self, problem: Dict[str, Any]) -> np.ndarray:
        """Solve optimization problem with parallel processing."""
        # For now, fall back to sequential (can implement ADMM or other parallel methods)
        return self._solve_sequential(problem)


class BayesianRLOptimizer:
    """Optimized Bayesian Reinforcement Learning components."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Thompson sampling cache
        self.posterior_samples_cache = {}
        
    def optimize_acquisition(self, gp_mean: np.ndarray, gp_var: np.ndarray,
                           actions: np.ndarray, acquisition_type: str = "ucb",
                           beta: float = 2.0) -> int:
        """Optimized acquisition function optimization."""
        if acquisition_type == "ucb":
            return self._optimize_ucb(gp_mean, gp_var, beta)
        elif acquisition_type == "thompson":
            return self._optimize_thompson_sampling(gp_mean, gp_var)
        elif acquisition_type == "ei":
            return self._optimize_expected_improvement(gp_mean, gp_var)
        else:
            raise ValueError(f"Unknown acquisition type: {acquisition_type}")
    
    @staticmethod
    @njit(fastmath=True)
    def _compute_ucb_numba(mean: np.ndarray, std: np.ndarray, beta: float) -> np.ndarray:
        """Numba-optimized UCB computation."""
        return mean + beta * std
    
    def _optimize_ucb(self, gp_mean: np.ndarray, gp_var: np.ndarray, beta: float) -> int:
        """Optimize Upper Confidence Bound acquisition function."""
        if self.config.use_numba:
            ucb_values = self._compute_ucb_numba(gp_mean, np.sqrt(gp_var), beta)
        else:
            ucb_values = gp_mean + beta * np.sqrt(gp_var)
        
        return np.argmax(ucb_values)
    
    def _optimize_thompson_sampling(self, gp_mean: np.ndarray, gp_var: np.ndarray) -> int:
        """Optimize Thompson Sampling acquisition function."""
        cache_key = hash((gp_mean.data.tobytes(), gp_var.data.tobytes()))
        
        if cache_key in self.posterior_samples_cache:
            samples = self.posterior_samples_cache[cache_key]
        else:
            # Sample from posterior
            samples = np.random.normal(gp_mean, np.sqrt(gp_var))
            
            # Cache if not full
            if len(self.posterior_samples_cache) < self.config.cache_size:
                self.posterior_samples_cache[cache_key] = samples
        
        return np.argmax(samples)
    
    def _optimize_expected_improvement(self, gp_mean: np.ndarray, gp_var: np.ndarray) -> int:
        """Optimize Expected Improvement acquisition function."""
        if len(gp_mean) == 0:
            return 0
        
        # Current best
        f_max = np.max(gp_mean)
        
        # Expected improvement
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            std = np.sqrt(gp_var)
            z = (gp_mean - f_max) / std
            z = np.clip(z, -10, 10)  # Numerical stability
            
            from scipy.stats import norm
            ei = (gp_mean - f_max) * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0] = 0
        
        return np.argmax(ei)
    
    def update_belief_state(self, prior_belief: Dict[str, np.ndarray],
                          observation: np.ndarray, action: int,
                          reward: float) -> Dict[str, np.ndarray]:
        """Optimized belief state update."""
        if self.config.use_parallel:
            return self._update_belief_parallel(prior_belief, observation, action, reward)
        else:
            return self._update_belief_sequential(prior_belief, observation, action, reward)
    
    def _update_belief_sequential(self, prior_belief: Dict[str, np.ndarray],
                                observation: np.ndarray, action: int,
                                reward: float) -> Dict[str, np.ndarray]:
        """Sequential belief update."""
        # Implement particle filter or other belief update
        # Simplified implementation
        
        posterior_belief = prior_belief.copy()
        
        # Update particles based on observation likelihood
        if 'particles' in prior_belief:
            particles = prior_belief['particles']
            weights = prior_belief.get('weights', np.ones(len(particles)) / len(particles))
            
            # Compute likelihood for each particle
            likelihoods = self._compute_observation_likelihood(particles, observation, action)
            
            # Update weights
            weights = weights * likelihoods
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            # Resample if effective sample size is low
            eff_sample_size = 1.0 / np.sum(weights**2)
            if eff_sample_size < len(particles) / 2:
                indices = np.random.choice(len(particles), len(particles), p=weights)
                particles = particles[indices]
                weights = np.ones(len(particles)) / len(particles)
            
            posterior_belief['particles'] = particles
            posterior_belief['weights'] = weights
        
        return posterior_belief
    
    def _update_belief_parallel(self, prior_belief: Dict[str, np.ndarray],
                              observation: np.ndarray, action: int,
                              reward: float) -> Dict[str, np.ndarray]:
        """Parallel belief update."""
        # For now, use sequential (can implement parallel particle filtering)
        return self._update_belief_sequential(prior_belief, observation, action, reward)
    
    def _compute_observation_likelihood(self, particles: np.ndarray,
                                     observation: np.ndarray, action: int) -> np.ndarray:
        """Compute observation likelihood for particles."""
        # Simplified Gaussian likelihood
        likelihoods = np.ones(len(particles))
        
        for i, particle in enumerate(particles):
            # Predict observation from particle state
            predicted_obs = particle  # Simplified: assume direct observation
            
            # Gaussian likelihood
            diff = observation - predicted_obs
            likelihood = np.exp(-0.5 * np.sum(diff**2))
            likelihoods[i] = likelihood
        
        return likelihoods


class AlgorithmOptimizerSuite:
    """Main suite combining all algorithm optimizers."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.gp_optimizer = GPInferenceOptimizer(config)
        self.mpc_optimizer = MPCOptimizer(config)
        self.bayes_rl_optimizer = BayesianRLOptimizer(config)
        
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_history = {
            'gp_times': [],
            'mpc_times': [],
            'bayes_rl_times': []
        }
    
    def benchmark_algorithms(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark all algorithms with test data."""
        results = {}
        
        # Benchmark GP optimization
        if 'gp_data' in test_data:
            gp_data = test_data['gp_data']
            start_time = time.time()
            
            mean, var = self.gp_optimizer.predict(
                gp_data['X_train'], gp_data['y_train'],
                gp_data['X_test'], gp_data['kernel_params']
            )
            
            gp_time = time.time() - start_time
            self.performance_history['gp_times'].append(gp_time)
            
            results['gp'] = {
                'computation_time': gp_time,
                'mean_prediction': mean,
                'variance_prediction': var,
                'cache_hits': len(self.gp_optimizer.kernel_cache)
            }
        
        # Benchmark MPC optimization
        if 'mpc_data' in test_data:
            mpc_data = test_data['mpc_data']
            start_time = time.time()
            
            mpc_result = self.mpc_optimizer.solve(
                mpc_data['initial_state'], mpc_data['reference'],
                mpc_data['horizon'], mpc_data['dt'],
                mpc_data['Q'], mpc_data['R'],
                mpc_data.get('constraints', {})
            )
            
            mpc_time = time.time() - start_time
            self.performance_history['mpc_times'].append(mpc_time)
            
            results['mpc'] = {
                'computation_time': mpc_time,
                'optimal_controls': mpc_result['optimal_controls'],
                'success': mpc_result['success']
            }
        
        # Benchmark Bayesian RL
        if 'bayes_rl_data' in test_data:
            bayes_data = test_data['bayes_rl_data']
            start_time = time.time()
            
            action = self.bayes_rl_optimizer.optimize_acquisition(
                bayes_data['gp_mean'], bayes_data['gp_var'],
                bayes_data['actions'], bayes_data['acquisition_type']
            )
            
            bayes_time = time.time() - start_time
            self.performance_history['bayes_rl_times'].append(bayes_time)
            
            results['bayes_rl'] = {
                'computation_time': bayes_time,
                'selected_action': action,
                'cache_hits': len(self.bayes_rl_optimizer.posterior_samples_cache)
            }
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {}
        
        for algorithm, times in self.performance_history.items():
            if times:
                summary[algorithm] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_calls': len(times)
                }
            else:
                summary[algorithm] = {'total_calls': 0}
        
        return summary
    
    def clear_caches(self):
        """Clear all algorithm caches."""
        self.gp_optimizer.kernel_cache.clear()
        self.gp_optimizer.cholesky_cache.clear()
        self.bayes_rl_optimizer.posterior_samples_cache.clear()
        
        self.logger.info("All algorithm caches cleared")
    
    def optimize_system_wide(self, optimization_target: str = "speed") -> Dict[str, Any]:
        """Perform system-wide optimization."""
        optimizations = []
        
        if optimization_target == "speed":
            # Enable all speed optimizations
            if not self.config.use_numba:
                self.config.use_numba = True
                optimizations.append("Enabled Numba compilation")
            
            if not self.config.use_parallel:
                self.config.use_parallel = True
                optimizations.append("Enabled parallel processing")
                
            if TORCH_AVAILABLE and not self.config.use_gpu:
                self.config.use_gpu = True
                optimizations.append("Enabled GPU acceleration")
        
        elif optimization_target == "memory":
            # Reduce cache sizes
            self.config.cache_size = max(100, self.config.cache_size // 2)
            optimizations.append(f"Reduced cache size to {self.config.cache_size}")
            
            # Clear existing caches
            self.clear_caches()
            optimizations.append("Cleared existing caches")
        
        elif optimization_target == "accuracy":
            # Increase precision
            self.config.convergence_tolerance = self.config.convergence_tolerance / 10
            self.config.max_iterations = min(5000, self.config.max_iterations * 2)
            optimizations.append("Increased numerical precision")
        
        return {
            'target': optimization_target,
            'optimizations_applied': optimizations,
            'new_config': self.config
        }


if __name__ == "__main__":
    # Example usage and testing
    config = OptimizationConfig(use_numba=True, use_parallel=True)
    optimizer_suite = AlgorithmOptimizerSuite(config)
    
    # Create test data
    np.random.seed(42)
    test_data = {
        'gp_data': {
            'X_train': np.random.randn(100, 3),
            'y_train': np.random.randn(100),
            'X_test': np.random.randn(20, 3),
            'kernel_params': {'length_scale': 1.0, 'variance': 1.0}
        },
        'mpc_data': {
            'initial_state': np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),  # 3D pos + vel
            'reference': np.zeros((10, 6)),  # Zero reference
            'horizon': 10,
            'dt': 0.1,
            'Q': np.eye(6),
            'R': np.eye(3),
            'constraints': {
                'u_min': np.array([-5.0, -5.0, -5.0]),
                'u_max': np.array([5.0, 5.0, 5.0])
            }
        },
        'bayes_rl_data': {
            'gp_mean': np.random.randn(50),
            'gp_var': np.random.rand(50),
            'actions': np.arange(50),
            'acquisition_type': 'ucb'
        }
    }
    
    # Run benchmark
    print("Running algorithm optimization benchmark...")
    results = optimizer_suite.benchmark_algorithms(test_data)
    
    print("\nBenchmark Results:")
    for algorithm, result in results.items():
        print(f"{algorithm.upper()}: {result['computation_time']:.4f}s")
    
    # Get performance summary
    summary = optimizer_suite.get_performance_summary()
    print(f"\nPerformance Summary: {summary}")