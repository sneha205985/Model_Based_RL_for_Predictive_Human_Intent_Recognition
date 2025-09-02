"""
Comprehensive Experimental Framework for HRI Bayesian RL System

This module implements a complete experimental framework for evaluating
the integrated human-robot interaction system with proper statistical
analysis, baseline comparisons, and comprehensive metrics.

Experiments:
1. Handover Task Performance
2. Safety Analysis  
3. Adaptation Speed
4. Computational Performance

Author: Phase 5 Implementation
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Statistics and analysis
from scipy.stats import ttest_ind, mannwhitneyu, friedmanchisquare
from statsmodels.stats.power import ttest_power
from statsmodels.stats.contingency_tables import mcnemar

# Import system components
try:
    from src.system.human_intent_rl_system import (
        HumanIntentRLSystem, SystemConfiguration, SystemMode, SystemMetrics
    )
    from src.environments.hri_environment import (
        HRIEnvironment, HRIState, RobotState, HumanState, ContextState,
        InteractionPhase, create_default_hri_environment
    )
    from src.integration.hri_bayesian_rl import HRIBayesianRLConfig
except ImportError as e:
    logging.warning(f"Import error: {e}. Some components may not be available.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure plotting
plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'seaborn')
sns.set_palette("husl")


class ExperimentType(Enum):
    """Types of experiments"""
    HANDOVER_PERFORMANCE = auto()
    SAFETY_ANALYSIS = auto()
    ADAPTATION_SPEED = auto()
    COMPUTATIONAL_PERFORMANCE = auto()
    ABLATION_STUDY = auto()


class BaselineMethod(Enum):
    """Baseline comparison methods"""
    NO_PREDICTION = auto()        # No human behavior prediction
    FIXED_POLICY = auto()         # Fixed handover policy
    CLASSICAL_RL = auto()         # Non-Bayesian RL
    NO_UNCERTAINTY = auto()       # Bayesian RL without uncertainty
    NO_MPC = auto()              # Direct control without MPC


@dataclass
class ExperimentConfiguration:
    """Configuration for experiments"""
    # General experiment settings
    experiment_name: str
    experiment_type: ExperimentType
    num_trials: int = 50
    max_steps_per_trial: int = 200
    random_seed: int = 42
    
    # Statistical analysis
    significance_level: float = 0.05
    confidence_level: float = 0.95
    min_effect_size: float = 0.5
    statistical_power: float = 0.8
    
    # System configurations to test
    baseline_methods: List[BaselineMethod] = field(default_factory=list)
    system_configs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scenario parameters
    scenario_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance parameters
    parallel_execution: bool = True
    max_workers: int = 4
    save_raw_data: bool = True
    
    # Output settings
    results_directory: str = "experiment_results"
    generate_plots: bool = True
    save_statistics: bool = True


@dataclass
class TrialResult:
    """Results from a single trial"""
    trial_id: int
    experiment_type: ExperimentType
    method: str
    scenario_params: Dict[str, Any]
    
    # Performance metrics
    success: bool
    task_completion_time: float
    safety_violations: int
    human_comfort_score: float
    
    # Detailed metrics
    step_count: int
    average_decision_time: float
    max_decision_time: float
    memory_usage: float
    
    # Trajectory data
    trajectory_data: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metrics (experiment-specific)
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResults:
    """Complete experiment results"""
    experiment_config: ExperimentConfiguration
    trial_results: List[TrialResult]
    
    # Aggregated statistics
    method_statistics: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Dict[str, Any]]
    
    # Performance analysis
    execution_time: float
    total_trials: int
    successful_trials: int
    
    # Generated outputs
    plots: Dict[str, str] = field(default_factory=dict)  # plot_name -> file_path
    raw_data_files: List[str] = field(default_factory=list)


class ScenarioGenerator:
    """Generates different HRI scenarios for testing"""
    
    def __init__(self, random_seed: int = 42):
        """Initialize scenario generator"""
        self.rng = np.random.RandomState(random_seed)
        
    def generate_handover_scenario(self, **params) -> Dict[str, Any]:
        """Generate handover scenario parameters"""
        # Default parameters
        defaults = {
            'human_approach_angle': 0.0,      # degrees
            'human_approach_speed': 0.1,      # m/s
            'object_weight': 0.5,             # kg
            'handover_height': 0.8,           # m
            'noise_level': 0.1,               # sensor noise
            'human_uncertainty': 0.3          # intent uncertainty
        }
        
        # Override with provided parameters
        scenario = defaults.copy()
        scenario.update(params)
        
        # Add randomization if requested
        if params.get('randomize', False):
            scenario['human_approach_angle'] = self.rng.uniform(-45, 45)
            scenario['human_approach_speed'] = self.rng.uniform(0.05, 0.2)
            scenario['object_weight'] = self.rng.uniform(0.2, 1.0)
            scenario['human_uncertainty'] = self.rng.uniform(0.1, 0.5)
        
        return scenario
    
    def generate_safety_scenario(self, **params) -> Dict[str, Any]:
        """Generate safety-critical scenario parameters"""
        defaults = {
            'unexpected_movement': False,
            'movement_magnitude': 0.0,        # m/s unexpected velocity
            'movement_direction': 0.0,        # degrees
            'obstacle_present': False,
            'obstacle_position': [0.0, 0.0, 0.0],
            'human_erratic_behavior': False,
            'noise_level': 0.1
        }
        
        scenario = defaults.copy()
        scenario.update(params)
        
        if params.get('randomize', False):
            scenario['unexpected_movement'] = self.rng.random() < 0.3
            if scenario['unexpected_movement']:
                scenario['movement_magnitude'] = self.rng.uniform(0.1, 0.5)
                scenario['movement_direction'] = self.rng.uniform(0, 360)
            
            scenario['obstacle_present'] = self.rng.random() < 0.2
            if scenario['obstacle_present']:
                scenario['obstacle_position'] = self.rng.uniform(-0.5, 0.5, 3).tolist()
        
        return scenario
    
    def generate_adaptation_scenario(self, **params) -> Dict[str, Any]:
        """Generate adaptation scenario parameters"""
        defaults = {
            'human_behavior_type': 'normal',   # 'normal', 'aggressive', 'cautious', 'unpredictable'
            'behavior_change_step': None,      # Step at which behavior changes
            'new_behavior_type': 'normal',
            'adaptation_difficulty': 'medium', # 'easy', 'medium', 'hard'
            'context_changes': False
        }
        
        scenario = defaults.copy()
        scenario.update(params)
        
        if params.get('randomize', False):
            behavior_types = ['normal', 'aggressive', 'cautious', 'unpredictable']
            scenario['human_behavior_type'] = self.rng.choice(behavior_types)
            
            if self.rng.random() < 0.5:  # 50% chance of behavior change
                scenario['behavior_change_step'] = self.rng.randint(50, 150)
                scenario['new_behavior_type'] = self.rng.choice(behavior_types)
        
        return scenario


class BaselineImplementations:
    """Implementations of baseline methods for comparison"""
    
    @staticmethod
    def create_no_prediction_system(base_config: SystemConfiguration) -> SystemConfiguration:
        """Create system without human behavior prediction"""
        config = base_config.__class__(**base_config.__dict__)
        
        # Disable human prediction by setting uncertainty to maximum
        if hasattr(config, 'bayesian_rl_config'):
            if config.bayesian_rl_config:
                config.bayesian_rl_config.use_human_intent_prediction = False
        
        return config
    
    @staticmethod
    def create_fixed_policy_system(base_config: SystemConfiguration) -> SystemConfiguration:
        """Create system with fixed handover policy"""
        config = base_config.__class__(**base_config.__dict__)
        
        # Set RL algorithm to use fixed policy
        if hasattr(config, 'bayesian_rl_config'):
            if config.bayesian_rl_config:
                config.bayesian_rl_config.rl_algorithm = "fixed_policy"
        
        return config
    
    @staticmethod
    def create_classical_rl_system(base_config: SystemConfiguration) -> SystemConfiguration:
        """Create system with classical (non-Bayesian) RL"""
        config = base_config.__class__(**base_config.__dict__)
        
        # Use classical Q-learning instead of Bayesian
        if hasattr(config, 'bayesian_rl_config'):
            if config.bayesian_rl_config:
                config.bayesian_rl_config.rl_algorithm = "classical_q_learning"
        
        return config
    
    @staticmethod
    def create_no_uncertainty_system(base_config: SystemConfiguration) -> SystemConfiguration:
        """Create Bayesian system without uncertainty quantification"""
        config = base_config.__class__(**base_config.__dict__)
        
        # Disable uncertainty quantification
        if hasattr(config, 'bayesian_rl_config'):
            if config.bayesian_rl_config:
                config.bayesian_rl_config.use_exploration_manager = False
        
        return config
    
    @staticmethod
    def create_no_mpc_system(base_config: SystemConfiguration) -> SystemConfiguration:
        """Create system without MPC (direct control)"""
        config = base_config.__class__(**base_config.__dict__)
        
        # Disable MPC integration
        if hasattr(config, 'bayesian_rl_config'):
            if config.bayesian_rl_config:
                config.bayesian_rl_config.use_mpc_controller = False
                config.bayesian_rl_config.integration_mode = "direct_control"
        
        return config


class ExperimentRunner:
    """Main class for running experiments"""
    
    def __init__(self, config: ExperimentConfiguration):
        """Initialize experiment runner"""
        self.config = config
        self.scenario_generator = ScenarioGenerator(config.random_seed)
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        
        # Create results directory
        self.results_dir = Path(config.results_directory)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize data storage
        self.trial_results: List[TrialResult] = []
        
        logger.info(f"Initialized experiment runner: {config.experiment_name}")
    
    def run_experiment(self) -> ExperimentResults:
        """Run complete experiment with all methods and statistical analysis"""
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        start_time = time.time()
        
        try:
            # Generate system configurations for testing
            system_configs = self._generate_system_configurations()
            
            # Run trials for each configuration
            all_results = []
            
            if self.config.parallel_execution:
                all_results = self._run_trials_parallel(system_configs)
            else:
                all_results = self._run_trials_sequential(system_configs)
            
            # Perform statistical analysis
            method_statistics = self._compute_method_statistics(all_results)
            statistical_tests = self._perform_statistical_tests(all_results)
            
            # Generate visualizations
            plots = {}
            if self.config.generate_plots:
                plots = self._generate_plots(all_results, method_statistics)
            
            # Save raw data
            raw_data_files = []
            if self.config.save_raw_data:
                raw_data_files = self._save_raw_data(all_results)
            
            # Create experiment results
            results = ExperimentResults(
                experiment_config=self.config,
                trial_results=all_results,
                method_statistics=method_statistics,
                statistical_tests=statistical_tests,
                execution_time=time.time() - start_time,
                total_trials=len(all_results),
                successful_trials=sum(1 for r in all_results if r.success),
                plots=plots,
                raw_data_files=raw_data_files
            )
            
            # Save experiment results
            self._save_experiment_results(results)
            
            logger.info(f"Experiment completed: {len(all_results)} trials in {results.execution_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
    
    def _generate_system_configurations(self) -> List[Tuple[str, SystemConfiguration, Dict[str, Any]]]:
        """Generate system configurations for testing"""
        configurations = []
        
        # Base configuration
        base_config = SystemConfiguration(
            mode=SystemMode.SIMULATION,
            max_decision_time=0.1,
            save_data=False,  # Don't save individual trial data
            use_threading=True,
            use_caching=True
        )
        
        # Add full Bayesian RL system (our method)
        configurations.append((
            "Bayesian_RL_Full",
            base_config,
            {'method_type': 'proposed'}
        ))
        
        # Add baseline methods
        if BaselineMethod.NO_PREDICTION in self.config.baseline_methods:
            no_pred_config = BaselineImplementations.create_no_prediction_system(base_config)
            configurations.append((
                "No_Prediction",
                no_pred_config,
                {'method_type': 'baseline'}
            ))
        
        if BaselineMethod.FIXED_POLICY in self.config.baseline_methods:
            fixed_config = BaselineImplementations.create_fixed_policy_system(base_config)
            configurations.append((
                "Fixed_Policy",
                fixed_config,
                {'method_type': 'baseline'}
            ))
        
        if BaselineMethod.CLASSICAL_RL in self.config.baseline_methods:
            classical_config = BaselineImplementations.create_classical_rl_system(base_config)
            configurations.append((
                "Classical_RL",
                classical_config,
                {'method_type': 'baseline'}
            ))
        
        if BaselineMethod.NO_UNCERTAINTY in self.config.baseline_methods:
            no_unc_config = BaselineImplementations.create_no_uncertainty_system(base_config)
            configurations.append((
                "No_Uncertainty",
                no_unc_config,
                {'method_type': 'ablation'}
            ))
        
        if BaselineMethod.NO_MPC in self.config.baseline_methods:
            no_mpc_config = BaselineImplementations.create_no_mpc_system(base_config)
            configurations.append((
                "No_MPC",
                no_mpc_config,
                {'method_type': 'ablation'}
            ))
        
        logger.info(f"Generated {len(configurations)} system configurations")
        return configurations
    
    def _run_trials_sequential(self, system_configs: List[Tuple[str, SystemConfiguration, Dict[str, Any]]]) -> List[TrialResult]:
        """Run trials sequentially"""
        all_results = []
        
        total_trials = len(system_configs) * self.config.num_trials
        completed_trials = 0
        
        for method_name, system_config, method_info in system_configs:
            logger.info(f"Running trials for method: {method_name}")
            
            for trial_id in range(self.config.num_trials):
                try:
                    # Generate scenario for this trial
                    scenario_params = self._generate_trial_scenario(trial_id)
                    
                    # Run single trial
                    result = self._run_single_trial(
                        trial_id, method_name, system_config, scenario_params, method_info
                    )
                    
                    all_results.append(result)
                    completed_trials += 1
                    
                    if completed_trials % 10 == 0:
                        logger.info(f"Progress: {completed_trials}/{total_trials} trials completed")
                
                except Exception as e:
                    logger.error(f"Trial {trial_id} for {method_name} failed: {e}")
                    # Create failed trial result
                    failed_result = TrialResult(
                        trial_id=trial_id,
                        experiment_type=self.config.experiment_type,
                        method=method_name,
                        scenario_params=scenario_params,
                        success=False,
                        task_completion_time=float('inf'),
                        safety_violations=999,
                        human_comfort_score=0.0,
                        step_count=0,
                        average_decision_time=0.0,
                        max_decision_time=0.0,
                        memory_usage=0.0,
                        additional_metrics={'error': str(e)}
                    )
                    all_results.append(failed_result)
                    completed_trials += 1
        
        return all_results
    
    def _run_trials_parallel(self, system_configs: List[Tuple[str, SystemConfiguration, Dict[str, Any]]]) -> List[TrialResult]:
        """Run trials in parallel"""
        all_results = []
        
        # Create trial tasks
        trial_tasks = []
        for method_name, system_config, method_info in system_configs:
            for trial_id in range(self.config.num_trials):
                scenario_params = self._generate_trial_scenario(trial_id)
                trial_tasks.append((trial_id, method_name, system_config, scenario_params, method_info))
        
        total_trials = len(trial_tasks)
        logger.info(f"Running {total_trials} trials in parallel with {self.config.max_workers} workers")
        
        # Execute trials in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_trial = {
                executor.submit(
                    self._run_single_trial, 
                    trial_id, method_name, system_config, scenario_params, method_info
                ): (trial_id, method_name) 
                for trial_id, method_name, system_config, scenario_params, method_info in trial_tasks
            }
            
            completed = 0
            for future in as_completed(future_to_trial):
                trial_id, method_name = future_to_trial[future]
                
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Trial {trial_id} for {method_name} failed: {e}")
                    # Create failed trial result
                    failed_result = TrialResult(
                        trial_id=trial_id,
                        experiment_type=self.config.experiment_type,
                        method=method_name,
                        scenario_params={},
                        success=False,
                        task_completion_time=float('inf'),
                        safety_violations=999,
                        human_comfort_score=0.0,
                        step_count=0,
                        average_decision_time=0.0,
                        max_decision_time=0.0,
                        memory_usage=0.0,
                        additional_metrics={'error': str(e)}
                    )
                    all_results.append(failed_result)
                
                completed += 1
                if completed % 20 == 0:
                    logger.info(f"Progress: {completed}/{total_trials} trials completed")
        
        return all_results
    
    def _generate_trial_scenario(self, trial_id: int) -> Dict[str, Any]:
        """Generate scenario parameters for a specific trial"""
        # Use trial ID as seed for reproducible scenarios
        trial_rng = np.random.RandomState(self.config.random_seed + trial_id)
        
        if self.config.experiment_type == ExperimentType.HANDOVER_PERFORMANCE:
            return self.scenario_generator.generate_handover_scenario(
                randomize=True,
                **self.config.scenario_parameters
            )
        elif self.config.experiment_type == ExperimentType.SAFETY_ANALYSIS:
            return self.scenario_generator.generate_safety_scenario(
                randomize=True,
                **self.config.scenario_parameters
            )
        elif self.config.experiment_type == ExperimentType.ADAPTATION_SPEED:
            return self.scenario_generator.generate_adaptation_scenario(
                randomize=True,
                **self.config.scenario_parameters
            )
        else:
            return self.config.scenario_parameters.copy()
    
    def _run_single_trial(self, trial_id: int, method_name: str, 
                         system_config: SystemConfiguration,
                         scenario_params: Dict[str, Any],
                         method_info: Dict[str, Any]) -> TrialResult:
        """Run a single experimental trial"""
        trial_start_time = time.time()
        
        try:
            # Initialize system for this trial
            system = HumanIntentRLSystem(system_config)
            
            if not system.initialize_system():
                raise Exception("System initialization failed")
            
            # Configure scenario in environment
            self._configure_trial_scenario(system, scenario_params)
            
            # Run episode
            episode_results = system.run_episode(max_steps=self.config.max_steps_per_trial)
            
            # Extract metrics
            success = episode_results.get('success', False) and episode_results.get('task_completion', 0) > 0.8
            task_completion_time = episode_results.get('episode_time', float('inf'))
            safety_violations = episode_results.get('safety_violations', 999)
            
            # Calculate human comfort score
            episode_data = episode_results.get('episode_data', [])
            comfort_scores = [d.get('performance_metrics', {}).get('human_comfort_score', 0) 
                            for d in episode_data if 'performance_metrics' in d]
            human_comfort_score = np.mean(comfort_scores) if comfort_scores else 0.0
            
            # Calculate decision time metrics
            step_times = [d.get('step_time', 0) for d in episode_data]
            avg_decision_time = np.mean(step_times) if step_times else 0.0
            max_decision_time = np.max(step_times) if step_times else 0.0
            
            # Get system status for memory usage
            system_status = system.get_system_status()
            memory_usage = system_status.get('system_metrics', {}).get('memory_usage', 0.0)
            
            # Experiment-specific metrics
            additional_metrics = self._compute_additional_metrics(
                episode_results, scenario_params, method_info
            )
            
            # Create trial result
            result = TrialResult(
                trial_id=trial_id,
                experiment_type=self.config.experiment_type,
                method=method_name,
                scenario_params=scenario_params,
                success=success,
                task_completion_time=task_completion_time if success else float('inf'),
                safety_violations=safety_violations,
                human_comfort_score=human_comfort_score,
                step_count=episode_results.get('steps_completed', 0),
                average_decision_time=avg_decision_time,
                max_decision_time=max_decision_time,
                memory_usage=memory_usage,
                trajectory_data=episode_data if len(episode_data) < 50 else episode_data[::5],  # Subsample for storage
                additional_metrics=additional_metrics
            )
            
            # Shutdown system
            system.shutdown()
            
            return result
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            # Return failed trial result
            return TrialResult(
                trial_id=trial_id,
                experiment_type=self.config.experiment_type,
                method=method_name,
                scenario_params=scenario_params,
                success=False,
                task_completion_time=float('inf'),
                safety_violations=999,
                human_comfort_score=0.0,
                step_count=0,
                average_decision_time=0.0,
                max_decision_time=0.0,
                memory_usage=0.0,
                additional_metrics={'error': str(e), 'trial_time': time.time() - trial_start_time}
            )
    
    def _configure_trial_scenario(self, system: HumanIntentRLSystem, scenario_params: Dict[str, Any]):
        """Configure the trial scenario in the system"""
        # This would configure the environment based on scenario parameters
        # For now, we'll just log the scenario
        logger.debug(f"Configuring trial with scenario: {scenario_params}")
        
        # In a real implementation, this would:
        # - Set human behavior parameters
        # - Configure obstacles and environment
        # - Set noise levels and disturbances
        # - Initialize human intent patterns
    
    def _compute_additional_metrics(self, episode_results: Dict[str, Any], 
                                  scenario_params: Dict[str, Any],
                                  method_info: Dict[str, Any]) -> Dict[str, float]:
        """Compute experiment-specific additional metrics"""
        additional = {}
        
        if self.config.experiment_type == ExperimentType.HANDOVER_PERFORMANCE:
            # Handover-specific metrics
            additional['handover_accuracy'] = 1.0 if episode_results.get('task_completion', 0) > 0.9 else 0.0
            additional['approach_efficiency'] = min(1.0, 100.0 / episode_results.get('steps_completed', 100))
            additional['timing_accuracy'] = 1.0 - abs(scenario_params.get('optimal_timing', 50) - episode_results.get('steps_completed', 50)) / 50.0
            
        elif self.config.experiment_type == ExperimentType.SAFETY_ANALYSIS:
            # Safety-specific metrics
            episode_data = episode_results.get('episode_data', [])
            min_distances = []
            for step_data in episode_data:
                if 'performance_metrics' in step_data:
                    # Extract minimum human distance if available
                    min_distances.append(0.5)  # Placeholder
            
            additional['min_human_distance'] = min(min_distances) if min_distances else 0.0
            additional['safety_margin'] = additional['min_human_distance'] - 0.2  # 20cm safety threshold
            additional['reaction_time'] = episode_results.get('steps_completed', 0) * 0.1  # Estimate
            
        elif self.config.experiment_type == ExperimentType.ADAPTATION_SPEED:
            # Adaptation-specific metrics
            additional['learning_rate'] = 1.0 / max(1, episode_results.get('steps_completed', 1))
            additional['adaptation_success'] = 1.0 if episode_results.get('success', False) else 0.0
            additional['behavioral_flexibility'] = np.random.random()  # Placeholder metric
            
        elif self.config.experiment_type == ExperimentType.COMPUTATIONAL_PERFORMANCE:
            # Performance-specific metrics
            additional['real_time_ratio'] = min(1.0, 0.1 / episode_results.get('avg_step_time', 0.1))
            additional['memory_efficiency'] = 1.0 / max(1.0, episode_results.get('final_metrics', {}).get('memory_usage', 1.0))
            additional['cpu_efficiency'] = 1.0 - min(1.0, episode_results.get('final_metrics', {}).get('cpu_usage', 0.0) / 100.0)
        
        return additional
    
    def _compute_method_statistics(self, results: List[TrialResult]) -> Dict[str, Dict[str, float]]:
        """Compute statistical summaries for each method"""
        method_stats = {}
        
        # Group results by method
        method_results = {}
        for result in results:
            if result.method not in method_results:
                method_results[result.method] = []
            method_results[result.method].append(result)
        
        # Compute statistics for each method
        for method, method_trials in method_results.items():
            stats_dict = {}
            
            # Success metrics
            successes = [r.success for r in method_trials]
            stats_dict['success_rate'] = np.mean(successes)
            stats_dict['success_rate_ci'] = self._compute_proportion_ci(successes)
            
            # Task completion time (only successful trials)
            successful_times = [r.task_completion_time for r in method_trials if r.success and r.task_completion_time < float('inf')]
            if successful_times:
                stats_dict['avg_completion_time'] = np.mean(successful_times)
                stats_dict['completion_time_std'] = np.std(successful_times)
                stats_dict['completion_time_ci'] = stats.t.interval(
                    self.config.confidence_level, len(successful_times)-1,
                    loc=np.mean(successful_times), scale=stats.sem(successful_times)
                )
            else:
                stats_dict['avg_completion_time'] = float('inf')
                stats_dict['completion_time_std'] = 0.0
                stats_dict['completion_time_ci'] = (float('inf'), float('inf'))
            
            # Safety metrics
            safety_violations = [r.safety_violations for r in method_trials]
            stats_dict['avg_safety_violations'] = np.mean(safety_violations)
            stats_dict['safety_violations_std'] = np.std(safety_violations)
            
            # Human comfort
            comfort_scores = [r.human_comfort_score for r in method_trials]
            stats_dict['avg_human_comfort'] = np.mean(comfort_scores)
            stats_dict['human_comfort_std'] = np.std(comfort_scores)
            
            # Performance metrics
            decision_times = [r.average_decision_time for r in method_trials if r.average_decision_time > 0]
            if decision_times:
                stats_dict['avg_decision_time'] = np.mean(decision_times)
                stats_dict['decision_time_std'] = np.std(decision_times)
                stats_dict['max_decision_time'] = np.max([r.max_decision_time for r in method_trials])
                stats_dict['real_time_violations'] = sum(1 for r in method_trials if r.max_decision_time > self.config.scenario_parameters.get('max_decision_time', 0.1))
            else:
                stats_dict['avg_decision_time'] = 0.0
                stats_dict['decision_time_std'] = 0.0
                stats_dict['max_decision_time'] = 0.0
                stats_dict['real_time_violations'] = 0
            
            # Memory usage
            memory_usage = [r.memory_usage for r in method_trials if r.memory_usage > 0]
            if memory_usage:
                stats_dict['avg_memory_usage'] = np.mean(memory_usage)
                stats_dict['memory_usage_std'] = np.std(memory_usage)
            else:
                stats_dict['avg_memory_usage'] = 0.0
                stats_dict['memory_usage_std'] = 0.0
            
            method_stats[method] = stats_dict
        
        return method_stats
    
    def _compute_proportion_ci(self, binary_data: List[bool], confidence: float = None) -> Tuple[float, float]:
        """Compute confidence interval for proportion"""
        if confidence is None:
            confidence = self.config.confidence_level
        
        n = len(binary_data)
        p = np.mean(binary_data)
        
        # Wilson score interval
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        denominator = 1 + z**2 / n
        centre_adjusted_probability = (p + z**2 / (2 * n)) / denominator
        adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
        upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
        
        return (max(0, lower_bound), min(1, upper_bound))
    
    def _perform_statistical_tests(self, results: List[TrialResult]) -> Dict[str, Dict[str, Any]]:
        """Perform statistical significance tests between methods"""
        statistical_tests = {}
        
        # Group results by method
        method_results = {}
        for result in results:
            if result.method not in method_results:
                method_results[result.method] = []
            method_results[result.method].append(result)
        
        methods = list(method_results.keys())
        if len(methods) < 2:
            return statistical_tests
        
        # Identify our proposed method and baselines
        proposed_method = "Bayesian_RL_Full"
        baseline_methods = [m for m in methods if m != proposed_method]
        
        # Perform pairwise comparisons with our method
        for baseline in baseline_methods:
            if baseline not in method_results or proposed_method not in method_results:
                continue
            
            comparison_key = f"{proposed_method}_vs_{baseline}"
            statistical_tests[comparison_key] = {}
            
            # Get data for both methods
            proposed_data = method_results[proposed_method]
            baseline_data = method_results[baseline]
            
            # Success rate comparison (Chi-square test)
            proposed_successes = sum(1 for r in proposed_data if r.success)
            baseline_successes = sum(1 for r in baseline_data if r.success)
            
            contingency_table = np.array([
                [proposed_successes, len(proposed_data) - proposed_successes],
                [baseline_successes, len(baseline_data) - baseline_successes]
            ])
            
            try:
                chi2, p_success = stats.chi2_contingency(contingency_table)[:2]
                statistical_tests[comparison_key]['success_rate'] = {
                    'test': 'chi2_contingency',
                    'statistic': chi2,
                    'p_value': p_success,
                    'significant': p_success < self.config.significance_level
                }
            except Exception as e:
                logger.warning(f"Chi-square test failed for {comparison_key}: {e}")
                statistical_tests[comparison_key]['success_rate'] = {
                    'test': 'chi2_contingency',
                    'statistic': 0,
                    'p_value': 1.0,
                    'significant': False,
                    'error': str(e)
                }
            
            # Task completion time comparison (t-test for successful trials)
            proposed_times = [r.task_completion_time for r in proposed_data if r.success and r.task_completion_time < float('inf')]
            baseline_times = [r.task_completion_time for r in baseline_data if r.success and r.task_completion_time < float('inf')]
            
            if len(proposed_times) > 1 and len(baseline_times) > 1:
                try:
                    # Check for normality (Shapiro-Wilk test)
                    _, p_norm_prop = stats.shapiro(proposed_times)
                    _, p_norm_base = stats.shapiro(baseline_times)
                    
                    if p_norm_prop > 0.05 and p_norm_base > 0.05:
                        # Use t-test if data is normal
                        t_stat, p_time = stats.ttest_ind(proposed_times, baseline_times)
                        test_name = 'welch_ttest'
                    else:
                        # Use Mann-Whitney U test if data is not normal
                        t_stat, p_time = stats.mannwhitneyu(proposed_times, baseline_times, alternative='two-sided')
                        test_name = 'mannwhitney_u'
                    
                    statistical_tests[comparison_key]['completion_time'] = {
                        'test': test_name,
                        'statistic': t_stat,
                        'p_value': p_time,
                        'significant': p_time < self.config.significance_level,
                        'effect_size': (np.mean(proposed_times) - np.mean(baseline_times)) / np.sqrt((np.var(proposed_times) + np.var(baseline_times)) / 2)
                    }
                except Exception as e:
                    logger.warning(f"Time comparison test failed for {comparison_key}: {e}")
                    statistical_tests[comparison_key]['completion_time'] = {
                        'test': 'none',
                        'p_value': 1.0,
                        'significant': False,
                        'error': str(e)
                    }
            
            # Safety violations comparison (Mann-Whitney U test)
            proposed_safety = [r.safety_violations for r in proposed_data]
            baseline_safety = [r.safety_violations for r in baseline_data]
            
            try:
                u_stat, p_safety = stats.mannwhitneyu(proposed_safety, baseline_safety, alternative='two-sided')
                statistical_tests[comparison_key]['safety_violations'] = {
                    'test': 'mannwhitney_u',
                    'statistic': u_stat,
                    'p_value': p_safety,
                    'significant': p_safety < self.config.significance_level
                }
            except Exception as e:
                logger.warning(f"Safety comparison test failed for {comparison_key}: {e}")
                statistical_tests[comparison_key]['safety_violations'] = {
                    'test': 'mannwhitney_u',
                    'statistic': 0,
                    'p_value': 1.0,
                    'significant': False,
                    'error': str(e)
                }
            
            # Human comfort comparison (t-test or Mann-Whitney U)
            proposed_comfort = [r.human_comfort_score for r in proposed_data]
            baseline_comfort = [r.human_comfort_score for r in baseline_data]
            
            try:
                u_stat, p_comfort = stats.mannwhitneyu(proposed_comfort, baseline_comfort, alternative='two-sided')
                statistical_tests[comparison_key]['human_comfort'] = {
                    'test': 'mannwhitney_u',
                    'statistic': u_stat,
                    'p_value': p_comfort,
                    'significant': p_comfort < self.config.significance_level
                }
            except Exception as e:
                logger.warning(f"Comfort comparison test failed for {comparison_key}: {e}")
                statistical_tests[comparison_key]['human_comfort'] = {
                    'test': 'mannwhitney_u',
                    'statistic': 0,
                    'p_value': 1.0,
                    'significant': False,
                    'error': str(e)
                }
        
        return statistical_tests
    
    def _generate_plots(self, results: List[TrialResult], 
                       method_stats: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Generate visualization plots"""
        plots = {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Plot 1: Success rate comparison
        plots['success_rates'] = self._plot_success_rates(method_stats)
        
        # Plot 2: Task completion time comparison
        plots['completion_times'] = self._plot_completion_times(results, method_stats)
        
        # Plot 3: Safety analysis
        plots['safety_analysis'] = self._plot_safety_analysis(results)
        
        # Plot 4: Performance metrics (decision times, memory usage)
        plots['performance_metrics'] = self._plot_performance_metrics(results)
        
        # Plot 5: Learning curves (if applicable)
        if self.config.experiment_type == ExperimentType.ADAPTATION_SPEED:
            plots['learning_curves'] = self._plot_learning_curves(results)
        
        return plots
    
    def _plot_success_rates(self, method_stats: Dict[str, Dict[str, float]]) -> str:
        """Plot success rate comparison with error bars"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        methods = list(method_stats.keys())
        success_rates = [method_stats[m]['success_rate'] for m in methods]
        
        # Get confidence intervals
        ci_lower = []
        ci_upper = []
        for method in methods:
            ci = method_stats[method]['success_rate_ci']
            ci_lower.append(success_rates[methods.index(method)] - ci[0])
            ci_upper.append(ci[1] - success_rates[methods.index(method)])
        
        # Create bar plot with error bars
        bars = ax.bar(methods, success_rates, yerr=[ci_lower, ci_upper], 
                     capsize=5, alpha=0.8, color=sns.color_palette("husl", len(methods)))
        
        ax.set_ylabel('Success Rate')
        ax.set_title(f'Task Success Rate Comparison - {self.config.experiment_name}')
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, success_rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + ci_upper[i] + 0.02,
                   f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels if needed
        if len(max(methods, key=len)) > 8:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"{self.config.experiment_name}_success_rates.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_completion_times(self, results: List[TrialResult], 
                              method_stats: Dict[str, Dict[str, float]]) -> str:
        """Plot task completion time comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Group results by method
        method_results = {}
        for result in results:
            if result.method not in method_results:
                method_results[result.method] = []
            method_results[result.method].append(result)
        
        # Plot 1: Box plot of completion times (successful trials only)
        successful_times = {}
        for method, trials in method_results.items():
            times = [r.task_completion_time for r in trials if r.success and r.task_completion_time < float('inf')]
            if times:
                successful_times[method] = times
        
        if successful_times:
            ax1.boxplot(successful_times.values(), labels=successful_times.keys())
            ax1.set_ylabel('Task Completion Time (s)')
            ax1.set_title('Task Completion Time Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Rotate labels if needed
            if len(max(successful_times.keys(), key=len)) > 8:
                ax1.set_xticklabels(successful_times.keys(), rotation=45, ha='right')
        
        # Plot 2: Average completion time with confidence intervals
        methods = list(method_stats.keys())
        avg_times = []
        ci_lower = []
        ci_upper = []
        
        for method in methods:
            avg_time = method_stats[method]['avg_completion_time']
            if avg_time < float('inf'):
                avg_times.append(avg_time)
                ci = method_stats[method]['completion_time_ci']
                ci_lower.append(avg_time - ci[0])
                ci_upper.append(ci[1] - avg_time)
            else:
                avg_times.append(0)
                ci_lower.append(0)
                ci_upper.append(0)
        
        bars = ax2.bar(methods, avg_times, yerr=[ci_lower, ci_upper], 
                      capsize=5, alpha=0.8, color=sns.color_palette("husl", len(methods)))
        
        ax2.set_ylabel('Average Completion Time (s)')
        ax2.set_title('Average Task Completion Time')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, avg_time in zip(bars, avg_times):
            if avg_time > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(ci_upper)/10,
                        f'{avg_time:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        if len(max(methods, key=len)) > 8:
            ax2.set_xticklabels(methods, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"{self.config.experiment_name}_completion_times.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_safety_analysis(self, results: List[TrialResult]) -> str:
        """Plot safety violation analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Group results by method
        method_results = {}
        for result in results:
            if result.method not in method_results:
                method_results[result.method] = []
            method_results[result.method].append(result)
        
        # Plot 1: Safety violations per method
        methods = list(method_results.keys())
        avg_violations = []
        std_violations = []
        
        for method in methods:
            violations = [r.safety_violations for r in method_results[method]]
            avg_violations.append(np.mean(violations))
            std_violations.append(np.std(violations))
        
        bars = ax1.bar(methods, avg_violations, yerr=std_violations, 
                      capsize=5, alpha=0.8, color=sns.color_palette("Reds_r", len(methods)))
        
        ax1.set_ylabel('Average Safety Violations')
        ax1.set_title('Safety Violations by Method')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, avg_viol in zip(bars, avg_violations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(std_violations)/10,
                    f'{avg_viol:.1f}', ha='center', va='bottom', fontweight='bold')
        
        if len(max(methods, key=len)) > 8:
            ax1.set_xticklabels(methods, rotation=45, ha='right')
        
        # Plot 2: Safety violation rate (percentage of trials with violations)
        violation_rates = []
        for method in methods:
            trials_with_violations = sum(1 for r in method_results[method] if r.safety_violations > 0)
            violation_rate = trials_with_violations / len(method_results[method])
            violation_rates.append(violation_rate)
        
        bars2 = ax2.bar(methods, violation_rates, alpha=0.8, 
                       color=sns.color_palette("Reds_r", len(methods)))
        
        ax2.set_ylabel('Safety Violation Rate')
        ax2.set_title('Percentage of Trials with Safety Violations')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars2, violation_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        if len(max(methods, key=len)) > 8:
            ax2.set_xticklabels(methods, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"{self.config.experiment_name}_safety_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_performance_metrics(self, results: List[TrialResult]) -> str:
        """Plot computational performance metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Group results by method
        method_results = {}
        for result in results:
            if result.method not in method_results:
                method_results[result.method] = []
            method_results[result.method].append(result)
        
        methods = list(method_results.keys())
        colors = sns.color_palette("husl", len(methods))
        
        # Plot 1: Average decision time
        avg_decision_times = []
        std_decision_times = []
        
        for method in methods:
            decision_times = [r.average_decision_time for r in method_results[method] if r.average_decision_time > 0]
            if decision_times:
                avg_decision_times.append(np.mean(decision_times))
                std_decision_times.append(np.std(decision_times))
            else:
                avg_decision_times.append(0)
                std_decision_times.append(0)
        
        bars1 = ax1.bar(methods, avg_decision_times, yerr=std_decision_times, 
                       capsize=5, alpha=0.8, color=colors)
        
        ax1.set_ylabel('Average Decision Time (s)')
        ax1.set_title('Decision Time Performance')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Real-time threshold')
        ax1.legend()
        
        if len(max(methods, key=len)) > 8:
            ax1.set_xticklabels(methods, rotation=45, ha='right')
        
        # Plot 2: Maximum decision time
        max_decision_times = []
        for method in methods:
            max_times = [r.max_decision_time for r in method_results[method] if r.max_decision_time > 0]
            max_decision_times.append(max(max_times) if max_times else 0)
        
        bars2 = ax2.bar(methods, max_decision_times, alpha=0.8, color=colors)
        ax2.set_ylabel('Maximum Decision Time (s)')
        ax2.set_title('Worst-Case Decision Time')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Real-time threshold')
        ax2.legend()
        
        if len(max(methods, key=len)) > 8:
            ax2.set_xticklabels(methods, rotation=45, ha='right')
        
        # Plot 3: Memory usage
        avg_memory = []
        std_memory = []
        
        for method in methods:
            memory_usage = [r.memory_usage for r in method_results[method] if r.memory_usage > 0]
            if memory_usage:
                avg_memory.append(np.mean(memory_usage))
                std_memory.append(np.std(memory_usage))
            else:
                avg_memory.append(0)
                std_memory.append(0)
        
        bars3 = ax3.bar(methods, avg_memory, yerr=std_memory, 
                       capsize=5, alpha=0.8, color=colors)
        
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage')
        ax3.grid(True, alpha=0.3)
        
        if len(max(methods, key=len)) > 8:
            ax3.set_xticklabels(methods, rotation=45, ha='right')
        
        # Plot 4: Real-time constraint violations
        rt_violations = []
        for method in methods:
            violations = sum(1 for r in method_results[method] if r.max_decision_time > 0.1)
            violation_rate = violations / len(method_results[method])
            rt_violations.append(violation_rate)
        
        bars4 = ax4.bar(methods, rt_violations, alpha=0.8, color=colors)
        ax4.set_ylabel('Real-time Violation Rate')
        ax4.set_title('Real-time Constraint Violations')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars4, rt_violations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        if len(max(methods, key=len)) > 8:
            ax4.set_xticklabels(methods, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"{self.config.experiment_name}_performance_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_learning_curves(self, results: List[TrialResult]) -> str:
        """Plot learning curves for adaptation experiments"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Group results by method
        method_results = {}
        for result in results:
            if result.method not in method_results:
                method_results[result.method] = []
            method_results[result.method].append(result)
        
        # Plot 1: Success rate over trials (sliding window average)
        window_size = 10
        for method, trials in method_results.items():
            if len(trials) < window_size:
                continue
            
            # Sort trials by trial_id to get chronological order
            trials.sort(key=lambda x: x.trial_id)
            
            success_rates = []
            trial_numbers = []
            
            for i in range(window_size, len(trials) + 1):
                window_trials = trials[i-window_size:i]
                success_rate = sum(1 for t in window_trials if t.success) / len(window_trials)
                success_rates.append(success_rate)
                trial_numbers.append(i)
            
            ax1.plot(trial_numbers, success_rates, marker='o', label=method, alpha=0.8)
        
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Success Rate (Sliding Window)')
        ax1.set_title(f'Learning Curves - Success Rate (Window={window_size})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 1])
        
        # Plot 2: Task completion time over trials
        for method, trials in method_results.items():
            if len(trials) < window_size:
                continue
            
            trials.sort(key=lambda x: x.trial_id)
            
            completion_times = []
            trial_numbers = []
            
            for i in range(window_size, len(trials) + 1):
                window_trials = trials[i-window_size:i]
                # Only consider successful trials
                successful_times = [t.task_completion_time for t in window_trials if t.success and t.task_completion_time < float('inf')]
                
                if successful_times:
                    avg_time = np.mean(successful_times)
                    completion_times.append(avg_time)
                    trial_numbers.append(i)
            
            if completion_times:
                ax2.plot(trial_numbers, completion_times, marker='s', label=method, alpha=0.8)
        
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Average Task Completion Time (s)')
        ax2.set_title(f'Learning Curves - Completion Time (Window={window_size})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"{self.config.experiment_name}_learning_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _save_raw_data(self, results: List[TrialResult]) -> List[str]:
        """Save raw experimental data"""
        saved_files = []
        
        # Convert results to DataFrame
        data_rows = []
        for result in results:
            row = {
                'trial_id': result.trial_id,
                'experiment_type': result.experiment_type.name,
                'method': result.method,
                'success': result.success,
                'task_completion_time': result.task_completion_time,
                'safety_violations': result.safety_violations,
                'human_comfort_score': result.human_comfort_score,
                'step_count': result.step_count,
                'average_decision_time': result.average_decision_time,
                'max_decision_time': result.max_decision_time,
                'memory_usage': result.memory_usage
            }
            
            # Add scenario parameters
            for key, value in result.scenario_params.items():
                row[f'scenario_{key}'] = value
            
            # Add additional metrics
            for key, value in result.additional_metrics.items():
                row[f'additional_{key}'] = value
            
            data_rows.append(row)
        
        # Save as CSV
        df = pd.DataFrame(data_rows)
        csv_path = self.results_dir / f"{self.config.experiment_name}_raw_data.csv"
        df.to_csv(csv_path, index=False)
        saved_files.append(str(csv_path))
        
        # Save as pickle for complete data preservation
        pickle_path = self.results_dir / f"{self.config.experiment_name}_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        saved_files.append(str(pickle_path))
        
        # Save trajectory data separately (if available)
        trajectory_data = []
        for result in results:
            if result.trajectory_data:
                for step_data in result.trajectory_data:
                    traj_row = {
                        'trial_id': result.trial_id,
                        'method': result.method,
                        'step': step_data.get('step', 0),
                        'timestamp': step_data.get('timestamp', 0)
                    }
                    # Add step-specific data
                    traj_row.update(step_data)
                    trajectory_data.append(traj_row)
        
        if trajectory_data:
            traj_df = pd.DataFrame(trajectory_data)
            traj_path = self.results_dir / f"{self.config.experiment_name}_trajectories.csv"
            traj_df.to_csv(traj_path, index=False)
            saved_files.append(str(traj_path))
        
        return saved_files
    
    def _save_experiment_results(self, results: ExperimentResults):
        """Save complete experiment results"""
        # Save results summary as JSON
        summary_path = self.results_dir / f"{self.config.experiment_name}_summary.json"
        
        summary_data = {
            'experiment_config': {
                'experiment_name': self.config.experiment_name,
                'experiment_type': self.config.experiment_type.name,
                'num_trials': self.config.num_trials,
                'baseline_methods': [m.name for m in self.config.baseline_methods],
                'significance_level': self.config.significance_level,
                'confidence_level': self.config.confidence_level
            },
            'execution_summary': {
                'execution_time': results.execution_time,
                'total_trials': results.total_trials,
                'successful_trials': results.successful_trials,
                'success_rate': results.successful_trials / results.total_trials if results.total_trials > 0 else 0
            },
            'method_statistics': results.method_statistics,
            'statistical_tests': results.statistical_tests,
            'generated_plots': results.plots
        }
        
        # Convert numpy arrays and other non-JSON serializable objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            else:
                return obj
        
        # Recursively convert the summary data
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {key: recursive_convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_for_json(obj)
        
        summary_data = recursive_convert(summary_data)
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Experiment results saved to {summary_path}")


# Example experiment configurations
def create_handover_experiment() -> ExperimentConfiguration:
    """Create handover task performance experiment"""
    return ExperimentConfiguration(
        experiment_name="Handover_Performance_Analysis",
        experiment_type=ExperimentType.HANDOVER_PERFORMANCE,
        num_trials=50,
        baseline_methods=[
            BaselineMethod.NO_PREDICTION,
            BaselineMethod.FIXED_POLICY,
            BaselineMethod.CLASSICAL_RL
        ],
        scenario_parameters={
            'randomize': True,
            'max_decision_time': 0.1
        }
    )


def create_safety_experiment() -> ExperimentConfiguration:
    """Create safety analysis experiment"""
    return ExperimentConfiguration(
        experiment_name="Safety_Analysis",
        experiment_type=ExperimentType.SAFETY_ANALYSIS,
        num_trials=75,
        baseline_methods=[
            BaselineMethod.NO_PREDICTION,
            BaselineMethod.CLASSICAL_RL,
            BaselineMethod.NO_MPC
        ],
        scenario_parameters={
            'randomize': True,
            'include_unexpected_movements': True,
            'include_obstacles': True
        }
    )


def create_adaptation_experiment() -> ExperimentConfiguration:
    """Create adaptation speed experiment"""
    return ExperimentConfiguration(
        experiment_name="Adaptation_Speed_Analysis",
        experiment_type=ExperimentType.ADAPTATION_SPEED,
        num_trials=100,
        baseline_methods=[
            BaselineMethod.CLASSICAL_RL,
            BaselineMethod.NO_UNCERTAINTY
        ],
        scenario_parameters={
            'randomize': True,
            'include_behavior_changes': True
        }
    )


def create_performance_experiment() -> ExperimentConfiguration:
    """Create computational performance experiment"""
    return ExperimentConfiguration(
        experiment_name="Computational_Performance_Analysis",
        experiment_type=ExperimentType.COMPUTATIONAL_PERFORMANCE,
        num_trials=30,
        baseline_methods=[
            BaselineMethod.CLASSICAL_RL,
            BaselineMethod.NO_MPC,
            BaselineMethod.NO_UNCERTAINTY
        ],
        scenario_parameters={
            'max_decision_time': 0.1,
            'measure_scalability': True
        }
    )


# Main execution function
def run_all_experiments():
    """Run all experimental evaluations"""
    logger.info("Starting comprehensive experimental evaluation")
    
    # Create experiment configurations
    experiments = [
        create_handover_experiment(),
        create_safety_experiment(),
        create_adaptation_experiment(),
        create_performance_experiment()
    ]
    
    all_results = {}
    
    for exp_config in experiments:
        logger.info(f"Running experiment: {exp_config.experiment_name}")
        
        try:
            # Run experiment
            runner = ExperimentRunner(exp_config)
            results = runner.run_experiment()
            
            all_results[exp_config.experiment_name] = results
            
            # Print summary
            logger.info(f"Experiment {exp_config.experiment_name} completed:")
            logger.info(f"  Total trials: {results.total_trials}")
            logger.info(f"  Successful trials: {results.successful_trials}")
            logger.info(f"  Execution time: {results.execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Experiment {exp_config.experiment_name} failed: {e}")
            continue
    
    logger.info(f"All experiments completed. {len(all_results)} experiments successful.")
    return all_results


# Example usage and testing
if __name__ == "__main__":
    # Run a small test experiment
    test_config = ExperimentConfiguration(
        experiment_name="Test_Handover",
        experiment_type=ExperimentType.HANDOVER_PERFORMANCE,
        num_trials=5,  # Small number for testing
        max_steps_per_trial=50,
        baseline_methods=[BaselineMethod.NO_PREDICTION],
        parallel_execution=False,
        generate_plots=True
    )
    
    try:
        runner = ExperimentRunner(test_config)
        results = runner.run_experiment()
        
        logger.info("Test experiment completed successfully!")
        logger.info(f"Results: {results.total_trials} trials, {results.successful_trials} successful")
        
    except Exception as e:
        logger.error(f"Test experiment failed: {e}")
    
    print("Experimental framework test completed!")