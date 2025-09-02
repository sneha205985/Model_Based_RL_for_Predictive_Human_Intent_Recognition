"""
Comprehensive Experimental Scenarios and Metrics
==============================================

Defines standardized experimental scenarios and comprehensive metrics collection
for rigorous evaluation of Model-Based RL Human Intent Recognition:

1. Handover Tasks (various objects, approaches, interruptions)
2. Collaborative Assembly (tool passing, coordinated manipulation) 
3. Gesture Following (pointing, reaching, dynamic sequences)
4. Safety Critical Scenarios (unexpected movements, sensor failures)

Each scenario includes:
- Primary metrics (success rate, safety, efficiency, comfort)
- Secondary metrics (prediction accuracy, learning efficiency, performance)
- Robustness metrics (noise resistance, adaptation, recovery)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import logging
from datetime import datetime
import time
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import warnings

# For trajectory and geometric calculations
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import scipy.stats as stats


class ScenarioType(Enum):
    """Types of experimental scenarios"""
    HANDOVER = "handover"
    COLLABORATIVE_ASSEMBLY = "collaborative_assembly"
    GESTURE_FOLLOWING = "gesture_following"
    SAFETY_CRITICAL = "safety_critical"
    MULTI_HUMAN = "multi_human"
    ADAPTIVE_LEARNING = "adaptive_learning"


class MetricType(Enum):
    """Types of metrics"""
    PRIMARY = "primary"           # Task success, safety, efficiency, comfort
    SECONDARY = "secondary"       # Prediction accuracy, learning efficiency
    ROBUSTNESS = "robustness"    # Performance under stress
    COMPUTATIONAL = "computational"  # Latency, memory, throughput


@dataclass
class ObjectProperties:
    """Properties of objects used in scenarios"""
    name: str
    mass: float  # kg
    dimensions: Tuple[float, float, float]  # length, width, height (m)
    fragility: float  # 0-1, 1 = very fragile
    grip_difficulty: float  # 0-1, 1 = very difficult to grip
    shape_category: str  # box, cylinder, sphere, irregular
    material_type: str  # metal, plastic, glass, fabric
    center_of_mass_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class HumanBehaviorProfile:
    """Model of human behavior characteristics"""
    name: str
    movement_speed_factor: float = 1.0  # Multiplier for normal speeds
    noise_level: float = 0.1  # Amount of randomness in movements
    predictability: float = 0.8  # How predictable the human is (0-1)
    cooperation_level: float = 0.9  # How cooperative (0-1)
    gesture_clarity: float = 0.8  # How clear gestures are (0-1)
    adaptation_rate: float = 0.5  # How quickly they adapt (0-1)
    interruption_frequency: float = 0.1  # Probability of interrupting actions
    error_recovery_skill: float = 0.7  # How well they recover from errors


@dataclass
class EnvironmentConditions:
    """Environmental conditions for scenarios"""
    lighting_quality: float = 1.0  # 0-1, 1 = perfect lighting
    noise_level: float = 0.0  # 0-1, 1 = very noisy
    space_constraints: float = 0.0  # 0-1, 1 = very constrained
    distraction_level: float = 0.0  # 0-1, 1 = many distractions
    sensor_reliability: float = 1.0  # 0-1, 1 = perfect sensors
    communication_clarity: float = 1.0  # 0-1, 1 = perfect communication


@dataclass
class MetricResult:
    """Container for individual metric results"""
    metric_name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Results from executing a scenario"""
    scenario_name: str
    scenario_type: ScenarioType
    duration: float
    success: bool
    primary_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    secondary_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    robustness_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    computational_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    trajectory_data: Dict[str, List[Tuple[float, float, float]]] = field(default_factory=dict)
    event_log: List[Dict[str, Any]] = field(default_factory=list)
    failure_reason: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)


class BaseScenario(ABC):
    """Abstract base class for experimental scenarios"""
    
    def __init__(self, scenario_name: str, scenario_type: ScenarioType,
                 config: Dict[str, Any] = None):
        self.scenario_name = scenario_name
        self.scenario_type = scenario_type
        self.config = config or {}
        self.metrics_collector = MetricsCollector()
        self.start_time = None
        self.end_time = None
        self.event_log = []
        
    @abstractmethod
    def setup(self, **kwargs) -> bool:
        """Setup the scenario"""
        pass
    
    @abstractmethod
    def execute(self, agent, environment) -> ScenarioResult:
        """Execute the scenario"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup after scenario execution"""
        pass
    
    def log_event(self, event_type: str, description: str, **kwargs):
        """Log an event during scenario execution"""
        self.event_log.append({
            'timestamp': time.time() - (self.start_time or time.time()),
            'event_type': event_type,
            'description': description,
            **kwargs
        })
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get scenario configuration"""
        return {
            'scenario_name': self.scenario_name,
            'scenario_type': self.scenario_type.value,
            'config': self.config.copy()
        }


class HandoverScenario(BaseScenario):
    """Handover task scenarios with various objects and approaches"""
    
    def __init__(self, scenario_name: str, config: Dict[str, Any] = None):
        super().__init__(scenario_name, ScenarioType.HANDOVER, config)
        
        # Default configuration
        default_config = {
            'object_type': 'cup',
            'approach_angle': 0.0,  # radians
            'approach_speed': 0.2,  # m/s
            'handover_height': 1.0,  # m
            'interruption_probability': 0.0,
            'timeout': 30.0  # seconds
        }
        default_config.update(self.config)
        self.config = default_config
        
        # Define available objects
        self.objects = {
            'cup': ObjectProperties(
                name='coffee_cup', mass=0.3, dimensions=(0.08, 0.08, 0.10),
                fragility=0.7, grip_difficulty=0.3, shape_category='cylinder',
                material_type='ceramic'
            ),
            'bottle': ObjectProperties(
                name='water_bottle', mass=0.5, dimensions=(0.06, 0.06, 0.20),
                fragility=0.4, grip_difficulty=0.2, shape_category='cylinder',
                material_type='plastic'
            ),
            'box': ObjectProperties(
                name='small_box', mass=0.8, dimensions=(0.15, 0.10, 0.08),
                fragility=0.2, grip_difficulty=0.4, shape_category='box',
                material_type='cardboard'
            ),
            'sphere': ObjectProperties(
                name='ball', mass=0.2, dimensions=(0.10, 0.10, 0.10),
                fragility=0.1, grip_difficulty=0.6, shape_category='sphere',
                material_type='rubber'
            ),
            'tool': ObjectProperties(
                name='screwdriver', mass=0.15, dimensions=(0.20, 0.02, 0.02),
                fragility=0.1, grip_difficulty=0.3, shape_category='irregular',
                material_type='metal'
            )
        }
        
        self.current_object = None
        self.human_position = np.array([0.5, 0.0, 1.0])
        self.robot_start_position = np.array([0.0, 0.0, 1.0])
        
    def setup(self, **kwargs) -> bool:
        """Setup handover scenario"""
        try:
            object_type = self.config.get('object_type', 'cup')
            self.current_object = self.objects.get(object_type)
            
            if not self.current_object:
                raise ValueError(f"Unknown object type: {object_type}")
            
            # Setup positions based on approach angle
            angle = self.config.get('approach_angle', 0.0)
            distance = 0.8  # Base distance between human and robot
            
            self.human_position = np.array([
                distance * np.cos(angle),
                distance * np.sin(angle), 
                self.config.get('handover_height', 1.0)
            ])
            
            self.log_event('setup', f'Handover scenario setup complete', 
                          object=object_type, approach_angle=angle)
            return True
            
        except Exception as e:
            self.log_event('error', f'Setup failed: {str(e)}')
            return False
    
    def execute(self, agent, environment) -> ScenarioResult:
        """Execute handover scenario"""
        self.start_time = time.time()
        result = ScenarioResult(
            scenario_name=self.scenario_name,
            scenario_type=self.scenario_type,
            duration=0.0,
            success=False,
            configuration=self.get_configuration()
        )
        
        try:
            # Phase 1: Approach phase
            self.log_event('phase_start', 'Approach phase started')
            approach_success = self._execute_approach_phase(agent, environment, result)
            
            if not approach_success:
                result.failure_reason = "Failed during approach phase"
                return self._finalize_result(result)
            
            # Phase 2: Handover execution
            self.log_event('phase_start', 'Handover execution started')
            handover_success = self._execute_handover_phase(agent, environment, result)
            
            if not handover_success:
                result.failure_reason = "Failed during handover execution"
                return self._finalize_result(result)
            
            # Phase 3: Completion verification
            self.log_event('phase_start', 'Completion verification started')
            completion_success = self._verify_completion(agent, environment, result)
            
            result.success = completion_success
            if not completion_success:
                result.failure_reason = "Failed completion verification"
            
        except Exception as e:
            result.failure_reason = f"Exception during execution: {str(e)}"
            self.log_event('error', str(e))
        
        return self._finalize_result(result)
    
    def _execute_approach_phase(self, agent, environment, result: ScenarioResult) -> bool:
        """Execute the approach phase"""
        phase_start = time.time()
        approach_timeout = 10.0  # seconds
        
        robot_position = self.robot_start_position.copy()
        target_position = self.human_position.copy()
        target_position[0] -= 0.3  # Stop 30cm away from human
        
        approach_trajectory = []
        
        while time.time() - phase_start < approach_timeout:
            current_time = time.time() - phase_start
            
            # Get agent action
            observation = self._get_observation(robot_position, self.human_position, environment)
            action = agent.select_action(observation, training=False)
            
            # Simulate robot movement
            robot_position += np.array(action[:3]) * 0.1  # 10cm steps
            approach_trajectory.append((current_time, *robot_position))
            
            # Check if close enough to target
            distance_to_target = np.linalg.norm(robot_position - target_position)
            if distance_to_target < 0.05:  # 5cm tolerance
                break
            
            # Simulate environment step
            time.sleep(0.1)  # 10Hz update rate
        
        # Calculate approach metrics
        final_distance = np.linalg.norm(robot_position - target_position)
        approach_time = time.time() - phase_start
        
        # Path efficiency (straight line vs actual path)
        if len(approach_trajectory) > 1:
            actual_path_length = sum(
                np.linalg.norm(np.array(approach_trajectory[i][1:4]) - 
                             np.array(approach_trajectory[i-1][1:4]))
                for i in range(1, len(approach_trajectory))
            )
            straight_line_distance = np.linalg.norm(target_position - self.robot_start_position)
            path_efficiency = straight_line_distance / (actual_path_length + 1e-8)
        else:
            path_efficiency = 0.0
        
        # Store trajectory data
        result.trajectory_data['approach'] = approach_trajectory
        
        # Store approach metrics
        result.secondary_metrics['approach_time'] = MetricResult(
            'approach_time', approach_time, 'seconds', time.time()
        )
        result.secondary_metrics['approach_accuracy'] = MetricResult(
            'approach_accuracy', 1.0 - final_distance, 'normalized', time.time()
        )
        result.secondary_metrics['path_efficiency'] = MetricResult(
            'path_efficiency', path_efficiency, 'ratio', time.time()
        )
        
        success = final_distance < 0.1  # 10cm tolerance
        self.log_event('phase_complete', 'Approach phase complete', 
                      success=success, final_distance=final_distance)
        return success
    
    def _execute_handover_phase(self, agent, environment, result: ScenarioResult) -> bool:
        """Execute the handover phase"""
        phase_start = time.time()
        handover_timeout = 15.0  # seconds
        
        # Simulate human extending hand with object
        human_hand_position = self.human_position.copy()
        human_hand_position[0] -= 0.2  # Extend hand 20cm toward robot
        
        robot_position = result.trajectory_data['approach'][-1][1:4]  # Final approach position
        handover_trajectory = []
        
        # Handover state variables
        object_grasped = False
        handover_complete = False
        min_safety_distance = float('inf')
        
        while time.time() - phase_start < handover_timeout and not handover_complete:
            current_time = time.time() - phase_start
            
            # Simulate potential interruption
            if (self.config.get('interruption_probability', 0.0) > 0 and 
                np.random.random() < self.config['interruption_probability'] * 0.1):
                self.log_event('interruption', 'Human interrupted handover')
                # Simulate human pulling back briefly
                human_hand_position[0] += 0.1
                time.sleep(0.5)
                human_hand_position[0] -= 0.1
            
            # Get observation including object and human hand positions
            observation = self._get_handover_observation(
                robot_position, human_hand_position, self.current_object, environment
            )
            action = agent.select_action(observation, training=False)
            
            # Simulate robot movement toward object
            robot_position += np.array(action[:3]) * 0.05  # Slower, more precise movements
            handover_trajectory.append((current_time, *robot_position))
            
            # Calculate safety distance (robot to human)
            safety_distance = np.linalg.norm(robot_position - self.human_position)
            min_safety_distance = min(min_safety_distance, safety_distance)
            
            # Check if robot is close enough to grasp object
            distance_to_object = np.linalg.norm(robot_position - human_hand_position)
            
            if not object_grasped and distance_to_object < 0.08:  # 8cm grasp threshold
                object_grasped = True
                self.log_event('grasp', 'Object grasped by robot')
            
            # Check for handover completion
            if object_grasped and distance_to_object < 0.05:  # 5cm for stable grasp
                # Simulate human releasing object
                handover_complete = True
                self.log_event('handover_complete', 'Handover successfully completed')
            
            time.sleep(0.05)  # 20Hz update rate for precise control
        
        # Calculate handover metrics
        handover_time = time.time() - phase_start
        success_rate = 1.0 if handover_complete else 0.0
        
        # Safety metrics
        safety_violation = min_safety_distance < 0.2  # 20cm minimum safety distance
        safety_score = min(1.0, min_safety_distance / 0.4)  # Normalize to 40cm optimal distance
        
        # Smoothness metrics (from trajectory)
        if len(handover_trajectory) > 2:
            velocities = []
            for i in range(1, len(handover_trajectory)):
                dt = handover_trajectory[i][0] - handover_trajectory[i-1][0]
                if dt > 0:
                    dp = np.linalg.norm(np.array(handover_trajectory[i][1:4]) - 
                                       np.array(handover_trajectory[i-1][1:4]))
                    velocities.append(dp / dt)
            
            if velocities:
                velocity_smoothness = 1.0 / (1.0 + np.std(velocities))
            else:
                velocity_smoothness = 0.0
        else:
            velocity_smoothness = 0.0
        
        # Store trajectory
        result.trajectory_data['handover'] = handover_trajectory
        
        # Store primary metrics
        result.primary_metrics['task_success_rate'] = MetricResult(
            'task_success_rate', success_rate, 'ratio', time.time()
        )
        result.primary_metrics['safety_score'] = MetricResult(
            'safety_score', safety_score, 'normalized', time.time()
        )
        result.primary_metrics['task_efficiency'] = MetricResult(
            'task_efficiency', 1.0 / (handover_time + 1.0), 'inverse_seconds', time.time()
        )
        result.primary_metrics['human_comfort'] = MetricResult(
            'human_comfort', velocity_smoothness, 'normalized', time.time()
        )
        
        # Store secondary metrics
        result.secondary_metrics['handover_time'] = MetricResult(
            'handover_time', handover_time, 'seconds', time.time()
        )
        result.secondary_metrics['min_safety_distance'] = MetricResult(
            'min_safety_distance', min_safety_distance, 'meters', time.time()
        )
        result.secondary_metrics['object_grasped'] = MetricResult(
            'object_grasped', 1.0 if object_grasped else 0.0, 'binary', time.time()
        )
        
        self.log_event('phase_complete', 'Handover phase complete', 
                      success=handover_complete, safety_violation=safety_violation)
        return handover_complete
    
    def _verify_completion(self, agent, environment, result: ScenarioResult) -> bool:
        """Verify successful completion of handover"""
        # Simple completion check - in real scenario this would involve
        # verifying object transfer, stability, etc.
        
        task_success = result.primary_metrics.get('task_success_rate')
        safety_ok = not result.secondary_metrics.get('min_safety_distance', MetricResult('', 0.0, '', 0)).value < 0.15
        
        completion_success = (task_success is not None and 
                            task_success.value > 0.9 and 
                            safety_ok)
        
        self.log_event('completion_verified', f'Completion verification: {completion_success}')
        return completion_success
    
    def _get_observation(self, robot_pos: np.ndarray, human_pos: np.ndarray, 
                        environment) -> np.ndarray:
        """Get observation for approach phase"""
        # Construct observation vector
        obs = np.concatenate([
            robot_pos,                    # Robot position (3)
            human_pos,                    # Human position (3)
            human_pos - robot_pos,        # Relative position (3)
            [np.linalg.norm(human_pos - robot_pos)],  # Distance (1)
            [time.time() - self.start_time]  # Time elapsed (1)
        ])
        return obs
    
    def _get_handover_observation(self, robot_pos: np.ndarray, object_pos: np.ndarray,
                                 obj_properties: ObjectProperties, environment) -> np.ndarray:
        """Get observation for handover phase"""
        obs = np.concatenate([
            robot_pos,                    # Robot position (3)
            object_pos,                   # Object position (3) 
            self.human_position,          # Human position (3)
            object_pos - robot_pos,       # Relative to object (3)
            [obj_properties.mass],        # Object mass (1)
            [obj_properties.fragility],   # Object fragility (1)
            [obj_properties.grip_difficulty],  # Grip difficulty (1)
            [time.time() - self.start_time]    # Time elapsed (1)
        ])
        return obs
    
    def _finalize_result(self, result: ScenarioResult) -> ScenarioResult:
        """Finalize scenario result"""
        self.end_time = time.time()
        result.duration = self.end_time - self.start_time
        result.event_log = self.event_log.copy()
        
        self.log_event('scenario_complete', f'Scenario completed', 
                      success=result.success, duration=result.duration)
        return result
    
    def cleanup(self):
        """Cleanup handover scenario"""
        self.current_object = None
        self.event_log.clear()


class CollaborativeAssemblyScenario(BaseScenario):
    """Collaborative assembly scenarios with tool passing and coordination"""
    
    def __init__(self, scenario_name: str, config: Dict[str, Any] = None):
        super().__init__(scenario_name, ScenarioType.COLLABORATIVE_ASSEMBLY, config)
        
        default_config = {
            'assembly_type': 'simple_construction',
            'num_components': 5,
            'coordination_complexity': 0.5,  # 0-1
            'error_injection_probability': 0.1,
            'timeout': 120.0
        }
        default_config.update(self.config)
        self.config = default_config
        
        # Define assembly components and tools
        self.tools = {
            'screwdriver': ObjectProperties(
                name='screwdriver', mass=0.15, dimensions=(0.20, 0.02, 0.02),
                fragility=0.1, grip_difficulty=0.3, shape_category='tool',
                material_type='metal'
            ),
            'wrench': ObjectProperties(
                name='wrench', mass=0.3, dimensions=(0.25, 0.03, 0.015),
                fragility=0.1, grip_difficulty=0.4, shape_category='tool', 
                material_type='metal'
            ),
            'pliers': ObjectProperties(
                name='pliers', mass=0.2, dimensions=(0.18, 0.05, 0.02),
                fragility=0.1, grip_difficulty=0.5, shape_category='tool',
                material_type='metal'
            )
        }
        
        self.components = []  # Will be populated in setup
        self.assembly_sequence = []
        self.current_step = 0
        
    def setup(self, **kwargs) -> bool:
        """Setup collaborative assembly scenario"""
        try:
            num_components = self.config.get('num_components', 5)
            
            # Create assembly components
            for i in range(num_components):
                component = {
                    'id': i,
                    'type': f'component_{i}',
                    'position': np.random.uniform(-0.5, 0.5, 3),
                    'required_tool': list(self.tools.keys())[i % len(self.tools)],
                    'assembly_difficulty': np.random.uniform(0.3, 0.8)
                }
                self.components.append(component)
            
            # Create assembly sequence
            self.assembly_sequence = list(range(num_components))
            if self.config.get('coordination_complexity', 0.5) > 0.5:
                # Shuffle for more complex coordination
                np.random.shuffle(self.assembly_sequence)
            
            self.log_event('setup', 'Assembly scenario setup complete',
                          num_components=num_components)
            return True
            
        except Exception as e:
            self.log_event('error', f'Setup failed: {str(e)}')
            return False
    
    def execute(self, agent, environment) -> ScenarioResult:
        """Execute collaborative assembly scenario"""
        self.start_time = time.time()
        result = ScenarioResult(
            scenario_name=self.scenario_name,
            scenario_type=self.scenario_type,
            duration=0.0,
            success=False,
            configuration=self.get_configuration()
        )
        
        try:
            completed_components = 0
            coordination_events = 0
            tool_exchanges = 0
            
            for step_idx in self.assembly_sequence:
                component = self.components[step_idx]
                step_start = time.time()
                
                self.log_event('assembly_step', f'Starting component {component["id"]}',
                              component_type=component['type'])
                
                # Execute assembly step
                step_success = self._execute_assembly_step(agent, environment, component, result)
                
                step_duration = time.time() - step_start
                
                if step_success:
                    completed_components += 1
                    self.log_event('step_complete', f'Component {component["id"]} completed',
                                  duration=step_duration)
                else:
                    self.log_event('step_failed', f'Component {component["id"]} failed')
                    break
                
                # Track coordination and tool exchanges
                coordination_events += self._count_coordination_events(component)
                tool_exchanges += 1  # Each step requires tool exchange
                
                # Check timeout
                if time.time() - self.start_time > self.config.get('timeout', 120.0):
                    result.failure_reason = "Assembly timeout exceeded"
                    break
            
            # Calculate overall success
            completion_rate = completed_components / len(self.assembly_sequence)
            result.success = completion_rate >= 0.8  # 80% completion threshold
            
            # Store primary metrics
            result.primary_metrics['task_success_rate'] = MetricResult(
                'task_success_rate', completion_rate, 'ratio', time.time()
            )
            
            # Calculate efficiency
            assembly_time = time.time() - self.start_time
            expected_time = len(self.assembly_sequence) * 20.0  # 20s per component baseline
            efficiency = expected_time / (assembly_time + 1.0)
            
            result.primary_metrics['task_efficiency'] = MetricResult(
                'task_efficiency', efficiency, 'ratio', time.time()
            )
            
            # Store secondary metrics
            result.secondary_metrics['completion_rate'] = MetricResult(
                'completion_rate', completion_rate, 'ratio', time.time()
            )
            result.secondary_metrics['coordination_events'] = MetricResult(
                'coordination_events', coordination_events, 'count', time.time()
            )
            result.secondary_metrics['tool_exchanges'] = MetricResult(
                'tool_exchanges', tool_exchanges, 'count', time.time()
            )
            result.secondary_metrics['assembly_time'] = MetricResult(
                'assembly_time', assembly_time, 'seconds', time.time()
            )
            
        except Exception as e:
            result.failure_reason = f"Exception during assembly: {str(e)}"
            self.log_event('error', str(e))
        
        return self._finalize_result(result)
    
    def _execute_assembly_step(self, agent, environment, component: Dict, 
                             result: ScenarioResult) -> bool:
        """Execute a single assembly step"""
        step_timeout = 25.0  # seconds per step
        step_start = time.time()
        
        required_tool = component['required_tool']
        tool_properties = self.tools[required_tool]
        
        # Phase 1: Tool request and acquisition
        tool_acquired = self._simulate_tool_exchange(agent, environment, required_tool)
        
        if not tool_acquired:
            self.log_event('tool_acquisition_failed', f'Failed to acquire {required_tool}')
            return False
        
        # Phase 2: Component positioning and assembly
        assembly_success = self._simulate_component_assembly(
            agent, environment, component, tool_properties, result
        )
        
        # Phase 3: Tool return
        tool_returned = self._simulate_tool_return(agent, environment, required_tool)
        
        step_time = time.time() - step_start
        success = tool_acquired and assembly_success and tool_returned and step_time < step_timeout
        
        # Track step metrics
        step_metrics = {
            'step_duration': step_time,
            'tool_acquired': tool_acquired,
            'assembly_success': assembly_success, 
            'tool_returned': tool_returned,
            'timeout_exceeded': step_time >= step_timeout
        }
        
        result.event_log.append({
            'timestamp': step_time,
            'event_type': 'assembly_step_complete',
            'component_id': component['id'],
            'success': success,
            'metrics': step_metrics
        })
        
        return success
    
    def _simulate_tool_exchange(self, agent, environment, tool_name: str) -> bool:
        """Simulate tool exchange between human and robot"""
        exchange_timeout = 8.0
        start_time = time.time()
        
        # Simulate human offering tool
        human_pos = np.array([0.6, 0.0, 1.0])
        robot_pos = np.array([0.0, 0.0, 1.0])
        
        while time.time() - start_time < exchange_timeout:
            # Simple simulation - robot moves toward human
            observation = np.concatenate([
                robot_pos, human_pos, [time.time() - start_time]
            ])
            
            action = agent.select_action(observation, training=False)
            robot_pos += np.array(action[:3]) * 0.1
            
            # Check if close enough for tool exchange
            if np.linalg.norm(robot_pos - human_pos) < 0.15:
                self.log_event('tool_acquired', f'Tool {tool_name} acquired')
                return True
            
            time.sleep(0.1)
        
        return False
    
    def _simulate_component_assembly(self, agent, environment, component: Dict,
                                   tool_properties: ObjectProperties, 
                                   result: ScenarioResult) -> bool:
        """Simulate component assembly process"""
        assembly_timeout = 15.0
        start_time = time.time()
        
        component_pos = np.array(component['position'])
        robot_pos = np.array([0.0, 0.0, 1.0])  # Assume robot has tool
        
        assembly_trajectory = []
        precision_required = component['assembly_difficulty']
        
        while time.time() - start_time < assembly_timeout:
            current_time = time.time() - start_time
            
            # Create observation with component and tool information
            observation = np.concatenate([
                robot_pos,
                component_pos,
                [tool_properties.mass],
                [precision_required],
                [current_time]
            ])
            
            action = agent.select_action(observation, training=False)
            robot_pos += np.array(action[:3]) * 0.05  # Precise movements
            
            assembly_trajectory.append((current_time, *robot_pos))
            
            # Check assembly completion (proximity and precision)
            distance_to_component = np.linalg.norm(robot_pos - component_pos)
            
            if distance_to_component < 0.03:  # 3cm precision required
                # Simulate assembly action completion
                assembly_duration = current_time
                break
            
            time.sleep(0.05)
        else:
            # Timeout occurred
            return False
        
        # Calculate assembly quality metrics
        if assembly_trajectory:
            # Path smoothness
            positions = np.array([traj[1:4] for traj in assembly_trajectory])
            if len(positions) > 2:
                velocities = np.diff(positions, axis=0)
                accelerations = np.diff(velocities, axis=0) 
                smoothness = 1.0 / (1.0 + np.mean(np.linalg.norm(accelerations, axis=1)))
            else:
                smoothness = 0.5
                
            # Precision score
            final_distance = np.linalg.norm(robot_pos - component_pos)
            precision_score = max(0.0, 1.0 - final_distance / 0.1)  # 10cm tolerance
            
            # Store assembly quality metrics
            quality_score = (smoothness + precision_score) / 2.0
            
            result.secondary_metrics[f'assembly_quality_{component["id"]}'] = MetricResult(
                f'assembly_quality_{component["id"]}', quality_score, 'normalized', time.time()
            )
            
            result.trajectory_data[f'assembly_{component["id"]}'] = assembly_trajectory
        
        self.log_event('component_assembled', f'Component {component["id"]} assembled',
                      assembly_duration=assembly_duration)
        return True
    
    def _simulate_tool_return(self, agent, environment, tool_name: str) -> bool:
        """Simulate returning tool to human"""
        # Simplified - assume successful return
        self.log_event('tool_returned', f'Tool {tool_name} returned')
        return True
    
    def _count_coordination_events(self, component: Dict) -> int:
        """Count coordination events for this component"""
        # Simulate coordination complexity
        base_events = 2  # Tool request and return
        complexity_events = int(component['assembly_difficulty'] * 3)
        return base_events + complexity_events
    
    def _finalize_result(self, result: ScenarioResult) -> ScenarioResult:
        """Finalize assembly scenario result"""
        self.end_time = time.time()
        result.duration = self.end_time - self.start_time
        result.event_log.extend(self.event_log)
        return result
    
    def cleanup(self):
        """Cleanup assembly scenario"""
        self.components.clear()
        self.assembly_sequence.clear()
        self.current_step = 0
        self.event_log.clear()


class GestureFollowingScenario(BaseScenario):
    """Gesture following scenarios with pointing and reaching instructions"""
    
    def __init__(self, scenario_name: str, config: Dict[str, Any] = None):
        super().__init__(scenario_name, ScenarioType.GESTURE_FOLLOWING, config)
        
        default_config = {
            'gesture_sequence_length': 5,
            'gesture_types': ['pointing', 'reaching', 'stop', 'come_here'],
            'ambiguity_level': 0.2,  # 0-1, probability of ambiguous gestures
            'dynamic_sequence': True,  # Gestures change during execution
            'timeout': 60.0
        }
        default_config.update(self.config)
        self.config = default_config
        
        self.gesture_sequence = []
        self.current_gesture_idx = 0
        
    def setup(self, **kwargs) -> bool:
        """Setup gesture following scenario"""
        try:
            gesture_types = self.config.get('gesture_types', ['pointing'])
            sequence_length = self.config.get('gesture_sequence_length', 5)
            
            # Generate gesture sequence
            for i in range(sequence_length):
                gesture_type = np.random.choice(gesture_types)
                
                gesture = {
                    'type': gesture_type,
                    'target_position': np.random.uniform(-1.0, 1.0, 3),
                    'clarity': 1.0 - self.config.get('ambiguity_level', 0.2) * np.random.random(),
                    'duration': np.random.uniform(2.0, 5.0),
                    'sequence_index': i
                }
                
                self.gesture_sequence.append(gesture)
            
            self.log_event('setup', 'Gesture following scenario setup complete',
                          sequence_length=sequence_length)
            return True
            
        except Exception as e:
            self.log_event('error', f'Setup failed: {str(e)}')
            return False
    
    def execute(self, agent, environment) -> ScenarioResult:
        """Execute gesture following scenario"""
        self.start_time = time.time()
        result = ScenarioResult(
            scenario_name=self.scenario_name,
            scenario_type=self.scenario_type,
            duration=0.0,
            success=False,
            configuration=self.get_configuration()
        )
        
        try:
            correct_responses = 0
            total_gestures = len(self.gesture_sequence)
            response_times = []
            gesture_recognition_accuracy = []
            
            for gesture_idx, gesture in enumerate(self.gesture_sequence):
                self.current_gesture_idx = gesture_idx
                gesture_start = time.time()
                
                self.log_event('gesture_start', f'Gesture {gesture_idx}: {gesture["type"]}',
                              gesture_type=gesture['type'], target_position=gesture['target_position'])
                
                # Execute gesture recognition and response
                gesture_success, response_time, recognition_confidence = self._execute_gesture_response(
                    agent, environment, gesture, result
                )
                
                if gesture_success:
                    correct_responses += 1
                
                response_times.append(response_time)
                gesture_recognition_accuracy.append(recognition_confidence)
                
                self.log_event('gesture_complete', f'Gesture {gesture_idx} completed',
                              success=gesture_success, response_time=response_time)
                
                # Check for dynamic sequence changes
                if self.config.get('dynamic_sequence', False) and np.random.random() < 0.3:
                    self._modify_gesture_sequence(gesture_idx + 1)
                
                # Brief pause between gestures
                time.sleep(0.5)
                
                # Check timeout
                if time.time() - self.start_time > self.config.get('timeout', 60.0):
                    result.failure_reason = "Gesture following timeout exceeded"
                    break
            
            # Calculate overall performance
            success_rate = correct_responses / total_gestures if total_gestures > 0 else 0.0
            avg_response_time = np.mean(response_times) if response_times else 0.0
            avg_recognition_accuracy = np.mean(gesture_recognition_accuracy) if gesture_recognition_accuracy else 0.0
            
            result.success = success_rate >= 0.8
            
            # Store primary metrics
            result.primary_metrics['task_success_rate'] = MetricResult(
                'task_success_rate', success_rate, 'ratio', time.time()
            )
            result.primary_metrics['task_efficiency'] = MetricResult(
                'task_efficiency', 1.0 / (avg_response_time + 0.1), 'inverse_seconds', time.time()
            )
            
            # Human comfort based on response predictability
            response_time_consistency = 1.0 / (1.0 + np.std(response_times)) if len(response_times) > 1 else 0.5
            result.primary_metrics['human_comfort'] = MetricResult(
                'human_comfort', response_time_consistency, 'normalized', time.time()
            )
            
            # Store secondary metrics
            result.secondary_metrics['gesture_recognition_accuracy'] = MetricResult(
                'gesture_recognition_accuracy', avg_recognition_accuracy, 'ratio', time.time()
            )
            result.secondary_metrics['average_response_time'] = MetricResult(
                'average_response_time', avg_response_time, 'seconds', time.time()
            )
            result.secondary_metrics['correct_responses'] = MetricResult(
                'correct_responses', correct_responses, 'count', time.time()
            )
            
        except Exception as e:
            result.failure_reason = f"Exception during gesture following: {str(e)}"
            self.log_event('error', str(e))
        
        return self._finalize_result(result)
    
    def _execute_gesture_response(self, agent, environment, gesture: Dict,
                                result: ScenarioResult) -> Tuple[bool, float, float]:
        """Execute response to a single gesture"""
        response_start = time.time()
        response_timeout = 10.0
        
        robot_position = np.array([0.0, 0.0, 1.0])
        human_position = np.array([0.8, 0.0, 1.0])
        
        # Simulate gesture recognition phase
        recognition_time = np.random.uniform(0.5, 2.0)  # Time to recognize gesture
        time.sleep(recognition_time)
        
        # Gesture recognition confidence (affected by clarity)
        recognition_confidence = gesture['clarity'] + np.random.normal(0, 0.1)
        recognition_confidence = np.clip(recognition_confidence, 0.0, 1.0)
        
        # Simulate potential misrecognition
        gesture_correctly_recognized = recognition_confidence > 0.7
        
        if not gesture_correctly_recognized:
            self.log_event('gesture_misrecognized', f'Gesture misrecognized',
                          actual_type=gesture['type'], confidence=recognition_confidence)
            return False, time.time() - response_start, recognition_confidence
        
        # Execute appropriate response based on gesture type
        response_trajectory = []
        target_reached = False
        
        if gesture['type'] == 'pointing':
            # Move toward pointed location
            target_pos = gesture['target_position']
            
            while time.time() - response_start < response_timeout and not target_reached:
                current_time = time.time() - response_start
                
                # Create observation with gesture information
                observation = np.concatenate([
                    robot_position,
                    target_pos,
                    human_position,
                    [recognition_confidence],
                    [current_time]
                ])
                
                action = agent.select_action(observation, training=False)
                robot_position += np.array(action[:3]) * 0.1
                
                response_trajectory.append((current_time, *robot_position))
                
                # Check if target reached
                distance_to_target = np.linalg.norm(robot_position - target_pos)
                if distance_to_target < 0.2:  # 20cm tolerance
                    target_reached = True
                    break
                
                time.sleep(0.1)
        
        elif gesture['type'] == 'reaching':
            # Extend toward human's reaching direction
            reach_direction = gesture['target_position'] / np.linalg.norm(gesture['target_position'])
            target_pos = human_position + reach_direction * 0.3
            
            while time.time() - response_start < response_timeout and not target_reached:
                current_time = time.time() - response_start
                
                observation = np.concatenate([
                    robot_position,
                    target_pos,
                    human_position,
                    reach_direction,
                    [current_time]
                ])
                
                action = agent.select_action(observation, training=False)
                robot_position += np.array(action[:3]) * 0.08  # Slower for reaching
                
                response_trajectory.append((current_time, *robot_position))
                
                if np.linalg.norm(robot_position - target_pos) < 0.15:
                    target_reached = True
                    break
                
                time.sleep(0.1)
        
        elif gesture['type'] == 'stop':
            # Stop current movement (simulate by staying still)
            time.sleep(2.0)  # Stay still for 2 seconds
            target_reached = True
        
        elif gesture['type'] == 'come_here':
            # Move toward human
            target_pos = human_position.copy()
            target_pos[0] -= 0.4  # Stop 40cm away from human
            
            while time.time() - response_start < response_timeout and not target_reached:
                current_time = time.time() - response_start
                
                observation = np.concatenate([
                    robot_position,
                    target_pos,
                    human_position,
                    [current_time]
                ])
                
                action = agent.select_action(observation, training=False)
                robot_position += np.array(action[:3]) * 0.12  # Slightly faster approach
                
                response_trajectory.append((current_time, *robot_position))
                
                if np.linalg.norm(robot_position - target_pos) < 0.1:
                    target_reached = True
                    break
                
                time.sleep(0.1)
        
        response_time = time.time() - response_start
        
        # Store trajectory if available
        if response_trajectory:
            result.trajectory_data[f'gesture_{gesture["sequence_index"]}'] = response_trajectory
        
        # Calculate response quality
        response_success = target_reached and response_time < response_timeout
        
        return response_success, response_time, recognition_confidence
    
    def _modify_gesture_sequence(self, from_index: int):
        """Dynamically modify gesture sequence"""
        if from_index < len(self.gesture_sequence):
            # Change one upcoming gesture
            new_gesture_type = np.random.choice(self.config.get('gesture_types', ['pointing']))
            self.gesture_sequence[from_index]['type'] = new_gesture_type
            self.gesture_sequence[from_index]['target_position'] = np.random.uniform(-1.0, 1.0, 3)
            
            self.log_event('sequence_modified', f'Gesture {from_index} changed to {new_gesture_type}')
    
    def _finalize_result(self, result: ScenarioResult) -> ScenarioResult:
        """Finalize gesture following scenario result"""
        self.end_time = time.time()
        result.duration = self.end_time - self.start_time
        result.event_log.extend(self.event_log)
        return result
    
    def cleanup(self):
        """Cleanup gesture following scenario"""
        self.gesture_sequence.clear()
        self.current_gesture_idx = 0
        self.event_log.clear()


class SafetyCriticalScenario(BaseScenario):
    """Safety critical scenarios with unexpected movements and sensor failures"""
    
    def __init__(self, scenario_name: str, config: Dict[str, Any] = None):
        super().__init__(scenario_name, ScenarioType.SAFETY_CRITICAL, config)
        
        default_config = {
            'unexpected_movement_probability': 0.3,
            'sensor_failure_probability': 0.2,
            'emergency_scenarios': ['sudden_approach', 'sensor_occlusion', 'communication_loss'],
            'recovery_time_limit': 5.0,  # seconds
            'safety_distance_threshold': 0.3,  # meters
            'timeout': 45.0
        }
        default_config.update(self.config)
        self.config = default_config
        
        self.emergency_events = []
        self.safety_violations = []
        
    def setup(self, **kwargs) -> bool:
        """Setup safety critical scenario"""
        try:
            # Plan emergency events
            emergency_types = self.config.get('emergency_scenarios', ['sudden_approach'])
            num_emergencies = np.random.randint(1, 4)  # 1-3 emergency events
            
            for i in range(num_emergencies):
                emergency_time = np.random.uniform(5.0, 30.0)  # 5-30 seconds into scenario
                emergency_type = np.random.choice(emergency_types)
                
                emergency_event = {
                    'type': emergency_type,
                    'trigger_time': emergency_time,
                    'duration': np.random.uniform(2.0, 8.0),
                    'severity': np.random.uniform(0.5, 1.0)
                }
                
                self.emergency_events.append(emergency_event)
            
            # Sort by trigger time
            self.emergency_events.sort(key=lambda x: x['trigger_time'])
            
            self.log_event('setup', 'Safety critical scenario setup complete',
                          num_emergencies=num_emergencies)
            return True
            
        except Exception as e:
            self.log_event('error', f'Setup failed: {str(e)}')
            return False
    
    def execute(self, agent, environment) -> ScenarioResult:
        """Execute safety critical scenario"""
        self.start_time = time.time()
        result = ScenarioResult(
            scenario_name=self.scenario_name,
            scenario_type=self.scenario_type,
            duration=0.0,
            success=False,
            configuration=self.get_configuration()
        )
        
        try:
            # Normal operation with emergency injection
            robot_position = np.array([0.0, 0.0, 1.0])
            human_position = np.array([1.0, 0.0, 1.0])
            
            emergency_index = 0
            safety_violations_count = 0
            successful_recoveries = 0
            total_emergencies = len(self.emergency_events)
            
            min_human_distance = float('inf')
            emergency_response_times = []
            
            # Main execution loop
            timeout = self.config.get('timeout', 45.0)
            
            while time.time() - self.start_time < timeout:
                current_time = time.time() - self.start_time
                
                # Check for emergency triggers
                while (emergency_index < len(self.emergency_events) and 
                       current_time >= self.emergency_events[emergency_index]['trigger_time']):
                    
                    emergency = self.emergency_events[emergency_index]
                    self.log_event('emergency_triggered', f'Emergency: {emergency["type"]}',
                                  emergency_type=emergency['type'], severity=emergency['severity'])
                    
                    # Execute emergency scenario
                    emergency_handled, response_time = self._execute_emergency_scenario(
                        agent, environment, emergency, robot_position, human_position, result
                    )
                    
                    if emergency_handled:
                        successful_recoveries += 1
                    else:
                        safety_violations_count += 1
                    
                    emergency_response_times.append(response_time)
                    emergency_index += 1
                
                # Normal operation step
                observation = self._get_safety_observation(robot_position, human_position, current_time)
                action = agent.select_action(observation, training=False)
                robot_position += np.array(action[:3]) * 0.05
                
                # Track minimum distance to human
                distance_to_human = np.linalg.norm(robot_position - human_position)
                min_human_distance = min(min_human_distance, distance_to_human)
                
                # Check safety violations
                if distance_to_human < self.config.get('safety_distance_threshold', 0.3):
                    self.safety_violations.append({
                        'timestamp': current_time,
                        'distance': distance_to_human,
                        'type': 'distance_violation'
                    })
                    safety_violations_count += 1
                
                # Simulate some human movement
                human_position += np.random.normal(0, 0.01, 3)  # Small random movements
                
                time.sleep(0.1)  # 10Hz update
            
            # Calculate safety performance metrics
            safety_success_rate = successful_recoveries / total_emergencies if total_emergencies > 0 else 1.0
            violation_rate = safety_violations_count / (current_time * 10)  # violations per second
            
            avg_response_time = np.mean(emergency_response_times) if emergency_response_times else 0.0
            
            # Overall success criteria
            result.success = (safety_success_rate >= 0.8 and 
                            violation_rate < 0.1 and 
                            avg_response_time < self.config.get('recovery_time_limit', 5.0))
            
            # Store primary metrics (safety is primary concern)
            result.primary_metrics['safety_violations'] = MetricResult(
                'safety_violations', float(safety_violations_count), 'count', time.time()
            )
            result.primary_metrics['safety_score'] = MetricResult(
                'safety_score', max(0.0, 1.0 - violation_rate), 'normalized', time.time()
            )
            result.primary_metrics['task_success_rate'] = MetricResult(
                'task_success_rate', safety_success_rate, 'ratio', time.time()
            )
            
            # Store secondary metrics
            result.secondary_metrics['emergency_response_time'] = MetricResult(
                'emergency_response_time', avg_response_time, 'seconds', time.time()
            )
            result.secondary_metrics['min_human_distance'] = MetricResult(
                'min_human_distance', min_human_distance, 'meters', time.time()
            )
            result.secondary_metrics['successful_recoveries'] = MetricResult(
                'successful_recoveries', successful_recoveries, 'count', time.time()
            )
            result.secondary_metrics['total_emergencies'] = MetricResult(
                'total_emergencies', total_emergencies, 'count', time.time()
            )
            
            # Store robustness metrics
            result.robustness_metrics['violation_rate'] = MetricResult(
                'violation_rate', violation_rate, 'per_second', time.time()
            )
            
            if not result.success:
                if safety_success_rate < 0.8:
                    result.failure_reason = "Low emergency recovery success rate"
                elif violation_rate >= 0.1:
                    result.failure_reason = "High safety violation rate"
                else:
                    result.failure_reason = "Slow emergency response time"
                    
        except Exception as e:
            result.failure_reason = f"Exception during safety scenario: {str(e)}"
            self.log_event('error', str(e))
        
        return self._finalize_result(result)
    
    def _execute_emergency_scenario(self, agent, environment, emergency: Dict,
                                  robot_position: np.ndarray, human_position: np.ndarray,
                                  result: ScenarioResult) -> Tuple[bool, float]:
        """Execute specific emergency scenario"""
        emergency_start = time.time()
        recovery_limit = self.config.get('recovery_time_limit', 5.0)
        
        emergency_type = emergency['type']
        severity = emergency['severity']
        
        if emergency_type == 'sudden_approach':
            return self._handle_sudden_approach(agent, environment, robot_position, 
                                              human_position, severity, recovery_limit)
        
        elif emergency_type == 'sensor_occlusion':
            return self._handle_sensor_failure(agent, environment, robot_position,
                                             human_position, severity, recovery_limit)
        
        elif emergency_type == 'communication_loss':
            return self._handle_communication_loss(agent, environment, robot_position,
                                                 human_position, severity, recovery_limit)
        
        else:
            # Unknown emergency type
            return False, recovery_limit
    
    def _handle_sudden_approach(self, agent, environment, robot_position: np.ndarray,
                               human_position: np.ndarray, severity: float,
                               recovery_limit: float) -> Tuple[bool, float]:
        """Handle sudden human approach emergency"""
        start_time = time.time()
        
        # Simulate human suddenly moving toward robot
        approach_speed = 0.5 * severity  # m/s
        original_human_pos = human_position.copy()
        
        emergency_handled = False
        
        while time.time() - start_time < recovery_limit:
            current_time = time.time() - start_time
            
            # Human approaches robot
            direction_to_robot = (robot_position - human_position) / np.linalg.norm(robot_position - human_position)
            human_position += direction_to_robot * approach_speed * 0.1  # 10Hz update
            
            # Robot must respond to avoid collision
            observation = np.concatenate([
                robot_position,
                human_position,
                direction_to_robot,
                [approach_speed],
                [current_time]
            ])
            
            action = agent.select_action(observation, training=False)
            robot_position += np.array(action[:3]) * 0.1
            
            # Check if safe distance maintained
            distance = np.linalg.norm(robot_position - human_position)
            
            if distance > self.config.get('safety_distance_threshold', 0.3):
                emergency_handled = True
                break
            
            time.sleep(0.1)
        
        response_time = time.time() - start_time
        
        self.log_event('sudden_approach_handled', f'Sudden approach emergency',
                      handled=emergency_handled, response_time=response_time)
        
        return emergency_handled, response_time
    
    def _handle_sensor_failure(self, agent, environment, robot_position: np.ndarray,
                             human_position: np.ndarray, severity: float,
                             recovery_limit: float) -> Tuple[bool, float]:
        """Handle sensor failure emergency"""
        start_time = time.time()
        
        # Simulate sensor failure by adding noise/occlusion to observations
        noise_level = severity
        
        emergency_handled = False
        
        while time.time() - start_time < recovery_limit:
            current_time = time.time() - start_time
            
            # Create degraded observation (noisy/missing data)
            base_observation = np.concatenate([
                robot_position,
                human_position,
                [current_time]
            ])
            
            # Add noise or zero out parts of observation
            if np.random.random() < noise_level:
                # Sensor occlusion - zero out human position
                degraded_observation = base_observation.copy()
                degraded_observation[3:6] = 0.0  # Zero out human position
            else:
                # Sensor noise
                noise = np.random.normal(0, noise_level * 0.2, len(base_observation))
                degraded_observation = base_observation + noise
            
            action = agent.select_action(degraded_observation, training=False)
            robot_position += np.array(action[:3]) * 0.05  # Slower movement under uncertainty
            
            # Check if robot maintains safe operation despite sensor issues
            distance_to_human = np.linalg.norm(robot_position - human_position)
            
            # Consider emergency handled if robot maintains safe distance and doesn't move erratically
            if (distance_to_human > 0.5 and  # Conservative safety distance
                np.linalg.norm(action[:3]) < 0.2):  # Not moving too fast
                emergency_handled = True
                break
            
            time.sleep(0.1)
        
        response_time = time.time() - start_time
        
        self.log_event('sensor_failure_handled', f'Sensor failure emergency',
                      handled=emergency_handled, response_time=response_time)
        
        return emergency_handled, response_time
    
    def _handle_communication_loss(self, agent, environment, robot_position: np.ndarray,
                                 human_position: np.ndarray, severity: float,
                                 recovery_limit: float) -> Tuple[bool, float]:
        """Handle communication loss emergency"""
        start_time = time.time()
        
        # Simulate communication loss - robot must operate autonomously
        emergency_handled = False
        
        while time.time() - start_time < recovery_limit:
            current_time = time.time() - start_time
            
            # Limited observation - no direct human communication/feedback
            limited_observation = np.concatenate([
                robot_position,
                human_position,  # Visual only, no communication
                [current_time]
            ])
            
            action = agent.select_action(limited_observation, training=False)
            
            # Robot should adopt conservative behavior during communication loss
            conservative_action = action[:3] * 0.5  # Reduce movement speed
            robot_position += conservative_action * 0.05
            
            # Human may continue moving
            human_position += np.random.normal(0, 0.02, 3)
            
            # Check if robot maintains safe, predictable behavior
            distance_to_human = np.linalg.norm(robot_position - human_position)
            
            if (distance_to_human > 0.4 and  # Safe distance
                np.linalg.norm(conservative_action) < 0.1):  # Conservative movement
                emergency_handled = True
                break
            
            time.sleep(0.1)
        
        response_time = time.time() - start_time
        
        self.log_event('communication_loss_handled', f'Communication loss emergency',
                      handled=emergency_handled, response_time=response_time)
        
        return emergency_handled, response_time
    
    def _get_safety_observation(self, robot_pos: np.ndarray, human_pos: np.ndarray,
                              current_time: float) -> np.ndarray:
        """Get observation for safety-critical operation"""
        distance = np.linalg.norm(robot_pos - human_pos)
        relative_pos = human_pos - robot_pos
        
        observation = np.concatenate([
            robot_pos,           # Robot position (3)
            human_pos,           # Human position (3) 
            relative_pos,        # Relative position (3)
            [distance],          # Distance to human (1)
            [current_time]       # Time elapsed (1)
        ])
        
        return observation
    
    def _finalize_result(self, result: ScenarioResult) -> ScenarioResult:
        """Finalize safety critical scenario result"""
        self.end_time = time.time()
        result.duration = self.end_time - self.start_time
        result.event_log.extend(self.event_log)
        
        # Add safety violations to event log
        for violation in self.safety_violations:
            result.event_log.append({
                'timestamp': violation['timestamp'],
                'event_type': 'safety_violation',
                'violation_type': violation['type'],
                'distance': violation['distance']
            })
        
        return result
    
    def cleanup(self):
        """Cleanup safety critical scenario"""
        self.emergency_events.clear()
        self.safety_violations.clear()
        self.event_log.clear()


class MetricsCollector:
    """Centralized metrics collection system"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None
        
    def start_collection(self):
        """Start metrics collection"""
        self.start_time = time.time()
        
    def collect_metric(self, name: str, value: float, unit: str = "", metadata: Dict = None):
        """Collect a single metric"""
        if self.start_time is None:
            self.start_collection()
            
        metric = MetricResult(
            metric_name=name,
            value=value,
            unit=unit,
            timestamp=time.time() - self.start_time,
            metadata=metadata or {}
        )
        
        self.metrics[name].append(metric)
        
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all collected metrics"""
        summary = {}
        
        for metric_name, metric_list in self.metrics.items():
            if not metric_list:
                continue
                
            values = [m.value for m in metric_list]
            
            summary[metric_name] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
            
            if len(values) > 1:
                summary[metric_name]['p25'] = np.percentile(values, 25)
                summary[metric_name]['p75'] = np.percentile(values, 75)
                summary[metric_name]['p95'] = np.percentile(values, 95)
        
        return summary
    
    def clear(self):
        """Clear all collected metrics"""
        self.metrics.clear()
        self.start_time = None


def create_standard_scenario_suite() -> List[BaseScenario]:
    """Create standard suite of experimental scenarios"""
    
    scenarios = []
    
    # Handover scenarios
    handover_configs = [
        {'object_type': 'cup', 'approach_angle': 0.0},
        {'object_type': 'bottle', 'approach_angle': np.pi/4},
        {'object_type': 'box', 'approach_angle': -np.pi/4},
        {'object_type': 'tool', 'interruption_probability': 0.2},
        {'object_type': 'sphere', 'approach_angle': np.pi/2}
    ]
    
    for i, config in enumerate(handover_configs):
        scenario = HandoverScenario(f"handover_{i+1}", config)
        scenarios.append(scenario)
    
    # Collaborative assembly scenarios
    assembly_configs = [
        {'num_components': 3, 'coordination_complexity': 0.3},
        {'num_components': 5, 'coordination_complexity': 0.5},
        {'num_components': 7, 'coordination_complexity': 0.8, 'error_injection_probability': 0.15}
    ]
    
    for i, config in enumerate(assembly_configs):
        scenario = CollaborativeAssemblyScenario(f"assembly_{i+1}", config)
        scenarios.append(scenario)
    
    # Gesture following scenarios
    gesture_configs = [
        {'gesture_sequence_length': 3, 'ambiguity_level': 0.1},
        {'gesture_sequence_length': 5, 'ambiguity_level': 0.2, 'dynamic_sequence': True},
        {'gesture_sequence_length': 8, 'ambiguity_level': 0.3, 'dynamic_sequence': True}
    ]
    
    for i, config in enumerate(gesture_configs):
        scenario = GestureFollowingScenario(f"gesture_{i+1}", config)
        scenarios.append(scenario)
    
    # Safety critical scenarios
    safety_configs = [
        {'unexpected_movement_probability': 0.2, 'sensor_failure_probability': 0.1},
        {'unexpected_movement_probability': 0.4, 'sensor_failure_probability': 0.2},
        {'unexpected_movement_probability': 0.3, 'sensor_failure_probability': 0.3}
    ]
    
    for i, config in enumerate(safety_configs):
        scenario = SafetyCriticalScenario(f"safety_{i+1}", config)
        scenarios.append(scenario)
    
    return scenarios


if __name__ == "__main__":
    # Example usage
    scenarios = create_standard_scenario_suite()
    print(f"Created {len(scenarios)} standard scenarios:")
    
    for scenario in scenarios:
        print(f"  - {scenario.scenario_name} ({scenario.scenario_type.value})")
    
    # Example of running a single scenario (would need actual agent and environment)
    print("\nScenario suite ready for experimental validation.")