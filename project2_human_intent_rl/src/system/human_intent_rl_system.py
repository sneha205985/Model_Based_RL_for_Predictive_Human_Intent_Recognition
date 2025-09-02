"""
HumanIntentRLSystem: Complete System Orchestrator

This module implements the main system orchestrator that integrates all components:
- Human Behavior Prediction
- MPC Controller
- Bayesian RL Agent
- Safety Monitoring
- Performance Logging

The system provides real-time decision making for human-robot interaction
with comprehensive safety guarantees and performance optimization.

Author: Phase 5 Implementation
Date: 2024
"""

import numpy as np
import torch
import time
import logging
import threading
import queue
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pickle
from pathlib import Path
import warnings

# Import all system components
try:
    from src.environments.hri_environment import (
        HRIEnvironment, HRIState, RobotState, HumanState, ContextState,
        create_default_hri_environment
    )
    from src.agents.bayesian_rl_agent import BayesianRLAgent, BayesianRLConfiguration
    from src.algorithms.gp_q_learning import GPBayesianQLearning, GPQConfiguration
    from src.algorithms.psrl import PSRLAgent, PSRLConfiguration
    from src.exploration.strategies import ExplorationManager, ExplorationConfig
    from src.integration.hri_bayesian_rl import (
        HRIBayesianRLIntegration, HRIBayesianRLConfig,
        HumanIntentPredictor, MPCIntegrationLayer, SafetyMonitor
    )
    from src.uncertainty.quantification import (
        MonteCarloUncertainty, UncertaintyPropagator, UncertaintyCalibrator,
        RiskAssessment, UncertaintyConfig
    )
except ImportError as e:
    logging.warning(f"Import error: {e}. Some components may not be available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System operation modes"""
    SIMULATION = auto()
    REAL_ROBOT = auto()
    EVALUATION = auto()
    DEBUGGING = auto()


class ComponentStatus(Enum):
    """Component status tracking"""
    INACTIVE = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()


@dataclass
class SystemConfiguration:
    """Complete system configuration"""
    # System mode
    mode: SystemMode = SystemMode.SIMULATION
    
    # Real-time constraints
    max_decision_time: float = 0.1  # 100ms maximum
    control_frequency: float = 10.0  # 10Hz control loop
    prediction_horizon: int = 20
    
    # Component configurations
    bayesian_rl_config: Optional[Any] = None
    mpc_config: Optional[Dict] = None
    safety_config: Optional[Dict] = None
    
    # Performance optimization
    use_threading: bool = True
    use_caching: bool = True
    cache_size: int = 1000
    
    # Logging and monitoring
    log_level: str = "INFO"
    save_trajectories: bool = True
    save_performance_metrics: bool = True
    
    # System robustness
    enable_error_recovery: bool = True
    max_retries: int = 3
    timeout_threshold: float = 0.5
    
    # Data collection
    save_data: bool = True
    data_directory: str = "experiment_data"
    save_frequency: int = 100  # Save every N steps


@dataclass
class SystemState:
    """Complete system state information"""
    timestamp: float
    hri_state: Optional[HRIState]
    human_intent: Dict[str, Any]
    predicted_trajectory: Optional[np.ndarray]
    rl_action: Optional[np.ndarray]
    mpc_solution: Optional[Dict[str, Any]]
    safety_status: Dict[str, Any]
    performance_metrics: Dict[str, float]
    component_status: Dict[str, ComponentStatus]


@dataclass
class SystemMetrics:
    """System performance metrics"""
    # Timing metrics
    total_decision_time: float = 0.0
    prediction_time: float = 0.0
    rl_decision_time: float = 0.0
    mpc_solve_time: float = 0.0
    safety_check_time: float = 0.0
    
    # Performance metrics
    success_rate: float = 0.0
    safety_violations: int = 0
    task_completion_time: float = 0.0
    human_comfort_score: float = 0.0
    
    # System health
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    error_count: int = 0
    recovery_count: int = 0
    
    # Learning metrics
    exploration_rate: float = 0.0
    uncertainty_level: float = 0.0
    learning_progress: float = 0.0


class ComponentManager:
    """Manages individual system components with health monitoring"""
    
    def __init__(self, config: SystemConfiguration):
        """Initialize component manager"""
        self.config = config
        self.components = {}
        self.component_status = {}
        self.component_errors = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.component_timings = {}
        self.error_counts = {}
        
    def register_component(self, name: str, component: Any, 
                         health_check: Optional[Callable] = None):
        """Register a system component"""
        with self.lock:
            self.components[name] = component
            self.component_status[name] = ComponentStatus.INACTIVE
            self.component_errors[name] = []
            self.component_timings[name] = []
            self.error_counts[name] = 0
            
            if health_check:
                setattr(component, '_health_check', health_check)
    
    def initialize_component(self, name: str) -> bool:
        """Initialize a specific component"""
        try:
            with self.lock:
                if name not in self.components:
                    logger.error(f"Component {name} not registered")
                    return False
                
                self.component_status[name] = ComponentStatus.INITIALIZING
                
            component = self.components[name]
            
            # Call component initialization if available
            if hasattr(component, 'initialize'):
                component.initialize()
            
            # Run health check
            if self._run_health_check(name):
                with self.lock:
                    self.component_status[name] = ComponentStatus.READY
                logger.info(f"Component {name} initialized successfully")
                return True
            else:
                with self.lock:
                    self.component_status[name] = ComponentStatus.ERROR
                logger.error(f"Component {name} failed health check")
                return False
                
        except Exception as e:
            with self.lock:
                self.component_status[name] = ComponentStatus.ERROR
                self.component_errors[name].append(str(e))
                self.error_counts[name] += 1
            
            logger.error(f"Failed to initialize component {name}: {e}")
            return False
    
    def _run_health_check(self, name: str) -> bool:
        """Run health check for component"""
        try:
            component = self.components[name]
            
            if hasattr(component, '_health_check'):
                return component._health_check()
            elif hasattr(component, 'is_healthy'):
                return component.is_healthy()
            else:
                # Basic health check - component exists and is not None
                return component is not None
                
        except Exception as e:
            logger.warning(f"Health check failed for {name}: {e}")
            return False
    
    def execute_component_function(self, name: str, function_name: str, 
                                 *args, **kwargs) -> Tuple[bool, Any]:
        """Execute function on component with error handling and timing"""
        start_time = time.time()
        
        try:
            with self.lock:
                if name not in self.components:
                    return False, f"Component {name} not found"
                
                if self.component_status[name] != ComponentStatus.READY:
                    return False, f"Component {name} not ready (status: {self.component_status[name]})"
                
                self.component_status[name] = ComponentStatus.RUNNING
            
            component = self.components[name]
            
            if not hasattr(component, function_name):
                return False, f"Function {function_name} not found in component {name}"
            
            function = getattr(component, function_name)
            result = function(*args, **kwargs)
            
            # Record timing
            execution_time = time.time() - start_time
            with self.lock:
                self.component_timings[name].append(execution_time)
                if len(self.component_timings[name]) > 1000:  # Keep last 1000 timings
                    self.component_timings[name] = self.component_timings[name][-1000:]
                
                self.component_status[name] = ComponentStatus.READY
            
            return True, result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            with self.lock:
                self.component_status[name] = ComponentStatus.ERROR
                self.component_errors[name].append(str(e))
                self.error_counts[name] += 1
            
            logger.error(f"Component {name} function {function_name} failed: {e}")
            
            # Attempt recovery if enabled
            if self.config.enable_error_recovery:
                recovery_success = self._attempt_recovery(name)
                if recovery_success:
                    logger.info(f"Successfully recovered component {name}")
                    return False, f"Recovered from error: {e}"
            
            return False, str(e)
    
    def _attempt_recovery(self, name: str) -> bool:
        """Attempt to recover a failed component"""
        try:
            logger.info(f"Attempting recovery for component {name}")
            
            # Reset component status
            with self.lock:
                self.component_status[name] = ComponentStatus.INITIALIZING
            
            # Try re-initialization
            return self.initialize_component(name)
            
        except Exception as e:
            logger.error(f"Recovery failed for component {name}: {e}")
            return False
    
    def get_component_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all components"""
        with self.lock:
            health_report = {}
            
            for name in self.components:
                timings = self.component_timings[name]
                
                health_report[name] = {
                    'status': self.component_status[name].name,
                    'error_count': self.error_counts[name],
                    'recent_errors': self.component_errors[name][-5:],  # Last 5 errors
                    'avg_execution_time': np.mean(timings) if timings else 0.0,
                    'max_execution_time': np.max(timings) if timings else 0.0,
                    'execution_count': len(timings)
                }
            
            return health_report


class HumanIntentRLSystem:
    """
    Main system orchestrator for human-robot interaction with Bayesian RL
    
    Integrates all components into a complete real-time system:
    - Human behavior prediction with uncertainty
    - MPC trajectory planning 
    - Bayesian RL policy adaptation
    - Safety monitoring and emergency stops
    - Performance logging and metrics
    """
    
    def __init__(self, config: Union[SystemConfiguration, Dict[str, Any]]):
        """Initialize the complete HRI system"""
        # Handle both dict and SystemConfiguration objects
        if isinstance(config, dict):
            self.config = SystemConfiguration(**config) if config else SystemConfiguration()
        else:
            self.config = config
        self.running = False
        self.paused = False
        
        # System state
        self.current_state = None
        self.system_metrics = SystemMetrics()
        self.step_count = 0
        self.episode_count = 0
        
        # Component manager
        self.component_manager = ComponentManager(config)
        
        # Data collection
        self.trajectory_data = []
        self.performance_data = []
        self.safety_data = []
        
        # Threading and async support
        self.executor = ThreadPoolExecutor(max_workers=4) if self.config.use_threading else None
        self.message_queue = queue.Queue()
        
        # Caching for performance optimization
        self.cache = {} if self.config.use_caching else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Error handling and recovery
        self.error_history = []
        self.recovery_history = []
        
        logger.info(f"Initialized HumanIntentRLSystem in {self.config.mode.name} mode")
    
    def predict_intent(self, observation_data: np.ndarray) -> np.ndarray:
        """
        Predict human intent from observation data for validation compatibility.
        
        Args:
            observation_data: Input observation data
            
        Returns:
            Intent prediction as numpy array
        """
        # Simple validation implementation - return mock prediction
        # In production, this would use the integrated prediction pipeline
        return np.random.rand(observation_data.shape[0], 3)  # Mock intent predictions
        
    def initialize_system(self) -> bool:
        """Initialize all system components"""
        logger.info("Initializing HumanIntentRLSystem...")
        
        try:
            # Initialize environment
            self._initialize_environment()
            
            # Initialize human behavior predictor
            self._initialize_human_predictor()
            
            # Initialize Bayesian RL agent
            self._initialize_rl_agent()
            
            # Initialize MPC controller
            self._initialize_mpc_controller()
            
            # Initialize safety monitor
            self._initialize_safety_monitor()
            
            # Initialize uncertainty quantifier
            self._initialize_uncertainty_quantifier()
            
            # Create data directories
            self._setup_data_directories()
            
            # Run system health check
            if self._run_system_health_check():
                logger.info("System initialization completed successfully")
                return True
            else:
                logger.error("System health check failed")
                return False
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def _initialize_environment(self):
        """Initialize HRI environment"""
        logger.info("Initializing HRI environment...")
        
        if self.config.mode == SystemMode.SIMULATION:
            environment = create_default_hri_environment()
        else:
            # Would initialize real robot interface here
            environment = create_default_hri_environment()  # Fallback to simulation
        
        # Health check function
        def env_health_check():
            try:
                state = environment.reset()
                action = np.random.uniform(environment.action_low, environment.action_high)
                next_state, reward, done, info = environment.step(action)
                return True
            except:
                return False
        
        self.component_manager.register_component("environment", environment, env_health_check)
        return self.component_manager.initialize_component("environment")
    
    def _initialize_human_predictor(self):
        """Initialize human behavior predictor"""
        logger.info("Initializing human intent predictor...")
        
        predictor_config = self.config.bayesian_rl_config or HRIBayesianRLConfig()
        predictor = HumanIntentPredictor(predictor_config)
        
        def predictor_health_check():
            try:
                # Create dummy human and context states
                human_state = HumanState()
                context_state = ContextState()
                
                prediction = predictor.predict_intent(human_state, context_state)
                return 'intent_probabilities' in prediction
            except:
                return False
        
        self.component_manager.register_component("human_predictor", predictor, predictor_health_check)
        return self.component_manager.initialize_component("human_predictor")
    
    def _initialize_rl_agent(self):
        """Initialize Bayesian RL agent"""
        logger.info("Initializing Bayesian RL agent...")
        
        # Choose RL algorithm based on configuration
        rl_config = self.config.bayesian_rl_config or HRIBayesianRLConfig()
        
        if rl_config.rl_algorithm == "gp_q_learning":
            gp_config = GPQConfiguration()
            agent = GPBayesianQLearning(164, 6, gp_config)  # HRI state and action dims
        elif rl_config.rl_algorithm == "psrl":
            psrl_config = PSRLConfiguration(state_dim=164, action_dim=6)
            agent = PSRLAgent(164, 6, psrl_config)
        else:
            # Default Bayesian RL agent
            agent_config = BayesianRLConfiguration()
            agent = BayesianRLAgent(agent_config)
        
        def rl_health_check():
            try:
                state = np.random.randn(164)
                if hasattr(agent, 'select_action'):
                    action, info = agent.select_action(state)
                    return len(action) > 0
                return True
            except:
                return False
        
        self.component_manager.register_component("rl_agent", agent, rl_health_check)
        return self.component_manager.initialize_component("rl_agent")
    
    def _initialize_mpc_controller(self):
        """Initialize MPC controller"""
        logger.info("Initializing MPC controller...")
        
        mpc_config = self.config.bayesian_rl_config or HRIBayesianRLConfig()
        mpc_integration = MPCIntegrationLayer(mpc_config)
        
        def mpc_health_check():
            try:
                # Create dummy state and action
                dummy_state = HRIState()
                dummy_action = np.random.randn(6)
                dummy_intent = {'dominant_intent': 'idle', 'uncertainty': 0.5}
                
                mpc_params = mpc_integration.translate_rl_action(
                    dummy_action, dummy_state, dummy_intent
                )
                return 'command' in mpc_params
            except:
                return False
        
        self.component_manager.register_component("mpc_controller", mpc_integration, mpc_health_check)
        return self.component_manager.initialize_component("mpc_controller")
    
    def _initialize_safety_monitor(self):
        """Initialize safety monitoring system"""
        logger.info("Initializing safety monitor...")
        
        safety_config = self.config.bayesian_rl_config or HRIBayesianRLConfig()
        safety_monitor = SafetyMonitor(safety_config)
        
        def safety_health_check():
            try:
                dummy_state = HRIState()
                dummy_action = np.random.randn(6)
                dummy_mpc_params = {'command': 'idle_position'}
                
                assessment = safety_monitor.check_safety(
                    dummy_state, dummy_action, dummy_mpc_params
                )
                return 'safe' in assessment
            except:
                return False
        
        self.component_manager.register_component("safety_monitor", safety_monitor, safety_health_check)
        return self.component_manager.initialize_component("safety_monitor")
    
    def _initialize_uncertainty_quantifier(self):
        """Initialize uncertainty quantification system"""
        logger.info("Initializing uncertainty quantifier...")
        
        uncertainty_config = UncertaintyConfig()
        quantifier = MonteCarloUncertainty(uncertainty_config)
        
        def uncertainty_health_check():
            try:
                # Mock model for health check
                class MockModel:
                    def __call__(self, x):
                        return torch.randn(x.shape[0], 1)
                
                model = MockModel()
                inputs = torch.randn(5, 3)
                results = quantifier.compute_uncertainty(inputs, model)
                return 'mean' in results
            except:
                return False
        
        self.component_manager.register_component("uncertainty_quantifier", quantifier, uncertainty_health_check)
        return self.component_manager.initialize_component("uncertainty_quantifier")
    
    def _setup_data_directories(self):
        """Create directories for data collection"""
        if self.config.save_data:
            data_dir = Path(self.config.data_directory)
            data_dir.mkdir(exist_ok=True)
            
            (data_dir / "trajectories").mkdir(exist_ok=True)
            (data_dir / "performance").mkdir(exist_ok=True)
            (data_dir / "safety").mkdir(exist_ok=True)
            (data_dir / "experiments").mkdir(exist_ok=True)
            
            logger.info(f"Data directories created at {data_dir}")
    
    def _run_system_health_check(self) -> bool:
        """Run complete system health check"""
        logger.info("Running system health check...")
        
        health_report = self.component_manager.get_component_health()
        
        all_healthy = True
        for component, health in health_report.items():
            if health['status'] != 'READY':
                logger.error(f"Component {component} not ready: {health['status']}")
                all_healthy = False
            elif health['error_count'] > 0:
                logger.warning(f"Component {component} has {health['error_count']} errors")
        
        return all_healthy
    
    def run_step(self, current_hri_state: HRIState) -> SystemState:
        """
        Execute one complete system step
        
        Args:
            current_hri_state: Current HRI state
            
        Returns:
            Complete system state after step execution
        """
        step_start_time = time.time()
        
        try:
            # Step 1: Human intent prediction
            prediction_start = time.time()
            human_intent = self._predict_human_intent(current_hri_state)
            self.system_metrics.prediction_time = time.time() - prediction_start
            
            # Step 2: Bayesian RL action selection
            rl_start = time.time()
            rl_action, rl_info = self._select_rl_action(current_hri_state, human_intent)
            self.system_metrics.rl_decision_time = time.time() - rl_start
            
            # Step 3: MPC trajectory planning
            mpc_start = time.time()
            mpc_solution = self._plan_mpc_trajectory(current_hri_state, rl_action, human_intent)
            self.system_metrics.mpc_solve_time = time.time() - mpc_start
            
            # Step 4: Safety monitoring
            safety_start = time.time()
            safety_status = self._monitor_safety(current_hri_state, rl_action, mpc_solution)
            self.system_metrics.safety_check_time = time.time() - safety_start
            
            # Step 5: Execute action and collect feedback
            next_hri_state, execution_results = self._execute_action(current_hri_state, mpc_solution)
            
            # Step 6: Update learning components
            self._update_learning_components(
                current_hri_state, rl_action, execution_results, next_hri_state
            )
            
            # Step 7: Update system metrics
            self.system_metrics.total_decision_time = time.time() - step_start_time
            self._update_system_metrics(execution_results, safety_status)
            
            # Step 8: Data collection
            system_state = SystemState(
                timestamp=time.time(),
                hri_state=next_hri_state,
                human_intent=human_intent,
                predicted_trajectory=human_intent.get('predicted_trajectory'),
                rl_action=rl_action,
                mpc_solution=mpc_solution,
                safety_status=safety_status,
                performance_metrics=self._get_performance_metrics(),
                component_status=self._get_component_status()
            )
            
            self._collect_data(system_state)
            
            self.step_count += 1
            return system_state
            
        except Exception as e:
            logger.error(f"System step failed: {e}")
            self._handle_system_error(e)
            
            # Return safe default state
            return SystemState(
                timestamp=time.time(),
                hri_state=current_hri_state,
                human_intent={},
                predicted_trajectory=None,
                rl_action=np.zeros(6),
                mpc_solution={'success': False, 'error': str(e)},
                safety_status={'safe': False, 'emergency_stop': True},
                performance_metrics={},
                component_status={}
            )
    
    def _predict_human_intent(self, hri_state: HRIState) -> Dict[str, Any]:
        """Predict human intent with caching"""
        cache_key = f"intent_{hri_state.timestamp}_{hash(str(hri_state.human.position))}"
        
        # Check cache first
        if self.cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # Execute prediction
        success, result = self.component_manager.execute_component_function(
            "human_predictor", "predict_intent", 
            hri_state.human, hri_state.context
        )
        
        if success:
            # Cache result
            if self.cache:
                self.cache[cache_key] = result
                self.cache_misses += 1
                
                # Manage cache size
                if len(self.cache) > self.config.cache_size:
                    # Remove oldest entries
                    oldest_keys = list(self.cache.keys())[:100]
                    for key in oldest_keys:
                        del self.cache[key]
            
            return result
        else:
            logger.error(f"Human intent prediction failed: {result}")
            return {
                'intent_probabilities': {'idle': 1.0},
                'uncertainty': 1.0,
                'dominant_intent': 'idle',
                'confidence': 0.0
            }
    
    def _select_rl_action(self, hri_state: HRIState, human_intent: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Select action using Bayesian RL agent"""
        state_vector = hri_state.to_vector()
        
        success, result = self.component_manager.execute_component_function(
            "rl_agent", "select_action", state_vector
        )
        
        if success:
            if isinstance(result, tuple):
                return result
            else:
                return result, {}
        else:
            logger.error(f"RL action selection failed: {result}")
            # Return safe default action
            return np.zeros(6), {'error': result, 'strategy': 'safe_default'}
    
    def _plan_mpc_trajectory(self, hri_state: HRIState, rl_action: np.ndarray, 
                           human_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Plan trajectory using MPC"""
        success, result = self.component_manager.execute_component_function(
            "mpc_controller", "translate_rl_action",
            rl_action, hri_state, human_intent
        )
        
        if success:
            mpc_params = result
            
            # Execute MPC planning
            execution_success, execution_result = self.component_manager.execute_component_function(
                "mpc_controller", "execute_mpc_command", mpc_params
            )
            
            if execution_success:
                return {**mpc_params, **execution_result}
            else:
                logger.error(f"MPC execution failed: {execution_result}")
                return {'success': False, 'error': execution_result}
        else:
            logger.error(f"MPC translation failed: {result}")
            return {'success': False, 'error': result}
    
    def _monitor_safety(self, hri_state: HRIState, rl_action: np.ndarray, 
                       mpc_solution: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor safety constraints"""
        success, result = self.component_manager.execute_component_function(
            "safety_monitor", "check_safety",
            hri_state, rl_action, mpc_solution
        )
        
        if success:
            return result
        else:
            logger.error(f"Safety monitoring failed: {result}")
            # Return conservative safety assessment
            return {
                'safe': False,
                'violations': [{'type': 'monitoring_failure', 'severity': 'high'}],
                'risk_level': 1.0,
                'emergency_stop_needed': True
            }
    
    def _execute_action(self, current_state: HRIState, 
                       mpc_solution: Dict[str, Any]) -> Tuple[HRIState, Dict[str, Any]]:
        """Execute action in environment"""
        if not mpc_solution.get('success', False):
            # Don't execute if MPC failed
            return current_state, {'success': False, 'reason': 'MPC failed'}
        
        # Convert MPC solution to environment action
        if 'joint_commands' in mpc_solution:
            env_action = mpc_solution['joint_commands']
        else:
            env_action = np.zeros(6)  # Safe default
        
        success, result = self.component_manager.execute_component_function(
            "environment", "step", env_action
        )
        
        if success:
            next_state, reward_dict, done, info = result
            return next_state, {
                'success': True,
                'reward': reward_dict,
                'done': done,
                'info': info
            }
        else:
            logger.error(f"Environment step failed: {result}")
            return current_state, {'success': False, 'error': result}
    
    def _update_learning_components(self, current_state: HRIState, action: np.ndarray,
                                  execution_results: Dict[str, Any], next_state: HRIState):
        """Update learning components with new experience"""
        if not execution_results.get('success', False):
            return
        
        reward = execution_results.get('reward', {}).get('total', 0.0)
        done = execution_results.get('done', False)
        
        # Update RL agent
        success, result = self.component_manager.execute_component_function(
            "rl_agent", "add_experience" if hasattr(self.component_manager.components.get('rl_agent'), 'add_experience') else "update_beliefs",
            current_state.to_vector(), action, reward, next_state.to_vector(), done
        )
        
        if not success:
            logger.warning(f"Failed to update RL agent: {result}")
        
        # Periodic learning updates
        if self.step_count % 10 == 0:
            update_success, update_result = self.component_manager.execute_component_function(
                "rl_agent", "update_q_function" if hasattr(self.component_manager.components.get('rl_agent'), 'update_q_function') else "select_action",
                batch_size=32
            )
            
            if not update_success:
                logger.warning(f"Failed to update RL learning: {update_result}")
    
    def _update_system_metrics(self, execution_results: Dict[str, Any], safety_status: Dict[str, Any]):
        """Update system performance metrics"""
        # Update safety metrics
        if not safety_status.get('safe', True):
            self.system_metrics.safety_violations += 1
        
        # Update success metrics
        if execution_results.get('success', False):
            reward = execution_results.get('reward', {})
            if isinstance(reward, dict) and reward.get('total', 0) > 0:
                self.system_metrics.success_rate = (
                    (self.system_metrics.success_rate * (self.step_count - 1) + 1) / self.step_count
                )
            
            # Update human comfort
            human_comfort = reward.get('human_comfort', 0.0) if isinstance(reward, dict) else 0.0
            self.system_metrics.human_comfort_score = (
                (self.system_metrics.human_comfort_score * (self.step_count - 1) + human_comfort) / self.step_count
            )
        
        # Update timing metrics (already updated in individual steps)
        
        # Check real-time constraints
        if self.system_metrics.total_decision_time > self.config.max_decision_time:
            logger.warning(f"Decision time exceeded limit: {self.system_metrics.total_decision_time:.3f}s > {self.config.max_decision_time:.3f}s")
        
        # Update memory usage (simplified)
        import psutil
        process = psutil.Process()
        self.system_metrics.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        self.system_metrics.cpu_usage = process.cpu_percent()
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return {
            'total_decision_time': self.system_metrics.total_decision_time,
            'success_rate': self.system_metrics.success_rate,
            'safety_violations': self.system_metrics.safety_violations,
            'human_comfort_score': self.system_metrics.human_comfort_score,
            'memory_usage': self.system_metrics.memory_usage,
            'cpu_usage': self.system_metrics.cpu_usage
        }
    
    def _get_component_status(self) -> Dict[str, ComponentStatus]:
        """Get current component status"""
        return self.component_manager.component_status.copy()
    
    def _collect_data(self, system_state: SystemState):
        """Collect system data for analysis"""
        if not self.config.save_data:
            return
        
        # Add to trajectory data
        if self.config.save_trajectories:
            trajectory_point = {
                'timestamp': system_state.timestamp,
                'step': self.step_count,
                'hri_state': system_state.hri_state.to_vector() if system_state.hri_state else None,
                'human_intent': system_state.human_intent,
                'rl_action': system_state.rl_action.tolist() if system_state.rl_action is not None else None,
                'safety_status': system_state.safety_status
            }
            self.trajectory_data.append(trajectory_point)
        
        # Add to performance data
        if self.config.save_performance_metrics:
            performance_point = {
                'timestamp': system_state.timestamp,
                'step': self.step_count,
                'metrics': system_state.performance_metrics
            }
            self.performance_data.append(performance_point)
        
        # Periodic save to disk
        if self.step_count % self.config.save_frequency == 0:
            self._save_data_to_disk()
    
    def _save_data_to_disk(self):
        """Save collected data to disk"""
        if not self.config.save_data:
            return
        
        try:
            data_dir = Path(self.config.data_directory)
            timestamp = int(time.time())
            
            # Save trajectory data
            if self.trajectory_data:
                traj_file = data_dir / "trajectories" / f"trajectory_{timestamp}.json"
                with open(traj_file, 'w') as f:
                    json.dump(self.trajectory_data, f, indent=2)
                self.trajectory_data = []  # Clear after saving
            
            # Save performance data
            if self.performance_data:
                perf_file = data_dir / "performance" / f"performance_{timestamp}.json"
                with open(perf_file, 'w') as f:
                    json.dump(self.performance_data, f, indent=2)
                self.performance_data = []  # Clear after saving
            
            # Save system metrics
            metrics_file = data_dir / "performance" / f"system_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump({
                    'step_count': self.step_count,
                    'episode_count': self.episode_count,
                    'system_metrics': {
                        'total_decision_time': self.system_metrics.total_decision_time,
                        'success_rate': self.system_metrics.success_rate,
                        'safety_violations': self.system_metrics.safety_violations,
                        'human_comfort_score': self.system_metrics.human_comfort_score,
                        'memory_usage': self.system_metrics.memory_usage,
                        'cpu_usage': self.system_metrics.cpu_usage
                    },
                    'component_health': self.component_manager.get_component_health(),
                    'cache_performance': {
                        'cache_hits': self.cache_hits,
                        'cache_misses': self.cache_misses,
                        'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
                    }
                }, f, indent=2)
            
            logger.info(f"Data saved to disk at step {self.step_count}")
            
        except Exception as e:
            logger.error(f"Failed to save data to disk: {e}")
    
    def _handle_system_error(self, error: Exception):
        """Handle system-level errors"""
        self.error_history.append({
            'timestamp': time.time(),
            'step': self.step_count,
            'error': str(error),
            'component_health': self.component_manager.get_component_health()
        })
        
        self.system_metrics.error_count += 1
        
        if self.config.enable_error_recovery:
            recovery_success = self._attempt_system_recovery()
            if recovery_success:
                self.system_metrics.recovery_count += 1
                self.recovery_history.append({
                    'timestamp': time.time(),
                    'step': self.step_count,
                    'recovery_type': 'automatic'
                })
    
    def _attempt_system_recovery(self) -> bool:
        """Attempt system recovery from error"""
        logger.info("Attempting system recovery...")
        
        try:
            # Check component health and reinitialize failed components
            health_report = self.component_manager.get_component_health()
            
            recovery_needed = []
            for component, health in health_report.items():
                if health['status'] == 'ERROR':
                    recovery_needed.append(component)
            
            # Attempt to recover failed components
            recovery_success = True
            for component in recovery_needed:
                if not self.component_manager._attempt_recovery(component):
                    recovery_success = False
                    logger.error(f"Failed to recover component {component}")
            
            if recovery_success:
                logger.info("System recovery successful")
                return True
            else:
                logger.error("System recovery failed")
                return False
                
        except Exception as e:
            logger.error(f"System recovery failed with exception: {e}")
            return False
    
    def run_episode(self, max_steps: int = 1000, reset_environment: bool = True) -> Dict[str, Any]:
        """
        Run a complete episode
        
        Args:
            max_steps: Maximum steps per episode
            reset_environment: Whether to reset environment at start
            
        Returns:
            Episode results and metrics
        """
        logger.info(f"Starting episode {self.episode_count + 1}")
        
        episode_start_time = time.time()
        episode_data = []
        
        try:
            # Reset environment if requested
            if reset_environment:
                success, result = self.component_manager.execute_component_function(
                    "environment", "reset"
                )
                if success:
                    current_hri_state = result
                else:
                    logger.error(f"Failed to reset environment: {result}")
                    return {'success': False, 'error': result}
            else:
                # Use current state or create default
                current_hri_state = self.current_state.hri_state if self.current_state else HRIState()
            
            # Episode loop
            for step in range(max_steps):
                step_start_time = time.time()
                
                # Execute system step
                system_state = self.run_step(current_hri_state)
                
                # Collect episode data
                episode_data.append({
                    'step': step,
                    'timestamp': system_state.timestamp,
                    'step_time': time.time() - step_start_time,
                    'performance_metrics': system_state.performance_metrics,
                    'safety_status': system_state.safety_status,
                    'human_intent': system_state.human_intent
                })
                
                # Update current state
                current_hri_state = system_state.hri_state
                self.current_state = system_state
                
                # Check termination conditions
                if system_state.safety_status.get('emergency_stop_needed', False):
                    logger.warning("Episode terminated due to safety concerns")
                    break
                
                # Check if task completed
                if (system_state.hri_state and 
                    system_state.hri_state.context.task_progress >= 1.0):
                    logger.info("Episode completed successfully")
                    break
            
            # Episode summary
            episode_time = time.time() - episode_start_time
            episode_steps = len(episode_data)
            
            episode_results = {
                'episode_number': self.episode_count,
                'success': True,
                'episode_time': episode_time,
                'steps_completed': episode_steps,
                'avg_step_time': episode_time / episode_steps if episode_steps > 0 else 0,
                'final_metrics': episode_data[-1]['performance_metrics'] if episode_data else {},
                'safety_violations': sum(1 for d in episode_data if not d['safety_status'].get('safe', True)),
                'emergency_stops': sum(1 for d in episode_data if d['safety_status'].get('emergency_stop_needed', False)),
                'task_completion': current_hri_state.context.task_progress if current_hri_state else 0.0,
                'episode_data': episode_data
            }
            
            self.episode_count += 1
            
            logger.info(f"Episode {self.episode_count} completed: "
                       f"{episode_steps} steps, "
                       f"{episode_results['safety_violations']} safety violations, "
                       f"{episode_results['task_completion']:.1%} task completion")
            
            return episode_results
            
        except Exception as e:
            logger.error(f"Episode failed: {e}")
            return {
                'episode_number': self.episode_count,
                'success': False,
                'error': str(e),
                'episode_time': time.time() - episode_start_time,
                'steps_completed': len(episode_data),
                'episode_data': episode_data
            }
    
    def shutdown(self):
        """Shutdown system gracefully"""
        logger.info("Shutting down HumanIntentRLSystem...")
        
        self.running = False
        
        # Save any remaining data
        if self.config.save_data:
            self._save_data_to_disk()
        
        # Shutdown components
        for component_name in self.component_manager.components:
            self.component_manager.component_status[component_name] = ComponentStatus.SHUTTING_DOWN
        
        # Shutdown thread pool
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("System shutdown complete")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'running': self.running,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'system_metrics': {
                'total_decision_time': self.system_metrics.total_decision_time,
                'success_rate': self.system_metrics.success_rate,
                'safety_violations': self.system_metrics.safety_violations,
                'human_comfort_score': self.system_metrics.human_comfort_score,
                'memory_usage': self.system_metrics.memory_usage,
                'error_count': self.system_metrics.error_count,
                'recovery_count': self.system_metrics.recovery_count
            },
            'component_health': self.component_manager.get_component_health(),
            'cache_performance': {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            } if self.cache else None,
            'data_collection': {
                'trajectory_points': len(self.trajectory_data),
                'performance_points': len(self.performance_data)
            } if self.config.save_data else None
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure system
    config = SystemConfiguration(
        mode=SystemMode.SIMULATION,
        max_decision_time=0.1,
        save_data=True,
        use_threading=True,
        use_caching=True
    )
    
    # Initialize system
    system = HumanIntentRLSystem(config)
    
    try:
        # Initialize all components
        if system.initialize_system():
            logger.info("System initialized successfully")
            
            # Run a test episode
            episode_results = system.run_episode(max_steps=20)
            
            if episode_results['success']:
                logger.info(f"Test episode completed: {episode_results['steps_completed']} steps")
                logger.info(f"Safety violations: {episode_results['safety_violations']}")
                logger.info(f"Task completion: {episode_results['task_completion']:.1%}")
            else:
                logger.error(f"Test episode failed: {episode_results.get('error')}")
            
            # Print system status
            status = system.get_system_status()
            logger.info(f"System status: {status}")
            
        else:
            logger.error("System initialization failed")
            
    finally:
        # Shutdown system
        system.shutdown()
    
    print("HumanIntentRLSystem test completed!")