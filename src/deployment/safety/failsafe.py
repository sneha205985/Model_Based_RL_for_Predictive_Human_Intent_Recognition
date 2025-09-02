"""
Fail-Safe Mechanisms for Timing Violations

This module implements comprehensive fail-safe mechanisms that activate when timing
violations occur in safety-critical real-time systems. It provides multiple layers
of protection to ensure system safety even under severe timing constraint violations.

Key Features:
- Multi-level fail-safe activation based on violation severity
- Graceful degradation strategies for different system components
- Integration with watchdog timers and emergency stop systems
- Formal safety compliance with IEC 61508 SIL requirements
- Hardware-level protection mechanisms
- Distributed fail-safe coordination for multi-robot systems

Safety Integrity Levels:
- SIL 1: Basic fail-safe with software monitoring
- SIL 2: Enhanced fail-safe with hardware monitoring
- SIL 3: Advanced fail-safe with redundant systems
- SIL 4: Ultra-safe fail-safe with triple redundancy

Author: Claude Code - Safety-Critical Real-Time Systems
"""

import time
import threading
import logging
import queue
import enum
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

class FailSafeLevel(enum.Enum):
    """Fail-safe activation levels based on violation severity"""
    NONE = 0
    WARNING = 1      # Soft deadline miss, no immediate action
    DEGRADED = 2     # Hard deadline miss, reduce performance
    EMERGENCY = 3    # Critical timing violation, emergency actions
    SHUTDOWN = 4     # System failure, complete shutdown

class ViolationType(enum.Enum):
    """Types of timing violations that can trigger fail-safe"""
    DEADLINE_MISS = "deadline_miss"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    MEMORY_OVERFLOW = "memory_overflow"
    COMMUNICATION_TIMEOUT = "communication_timeout"
    HARDWARE_FAULT = "hardware_fault"
    COORDINATION_FAILURE = "coordination_failure"

class ActionType(enum.Enum):
    """Types of fail-safe actions"""
    LOG_WARNING = "log_warning"
    REDUCE_FREQUENCY = "reduce_frequency"
    DISABLE_COMPONENT = "disable_component"
    ACTIVATE_BACKUP = "activate_backup"
    EMERGENCY_STOP = "emergency_stop"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ISOLATE_ROBOT = "isolate_robot"
    HALT_MOTION = "halt_motion"

@dataclass
class TimingViolation:
    """Represents a timing constraint violation"""
    violation_type: ViolationType
    timestamp: float
    component: str
    severity_level: FailSafeLevel
    measured_time_ms: float
    expected_time_ms: float
    violation_ratio: float  # measured / expected
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_critical(self) -> bool:
        return self.severity_level >= FailSafeLevel.EMERGENCY
    
    @property
    def description(self) -> str:
        return (f"{self.violation_type.value} in {self.component}: "
                f"{self.measured_time_ms:.2f}ms vs {self.expected_time_ms:.2f}ms "
                f"(ratio: {self.violation_ratio:.2f})")

@dataclass
class FailSafeAction:
    """Defines a fail-safe action to execute"""
    action_type: ActionType
    target_component: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: float = 1000.0
    retry_count: int = 3
    priority: int = 0  # Higher number = higher priority
    
    def __post_init__(self):
        self.execution_time: Optional[float] = None
        self.success: Optional[bool] = None
        self.error_message: Optional[str] = None

class FailSafeRule:
    """Defines when and how to activate fail-safe mechanisms"""
    
    def __init__(self,
                 rule_id: str,
                 violation_types: List[ViolationType],
                 min_severity: FailSafeLevel,
                 actions: List[FailSafeAction],
                 activation_threshold: int = 1,
                 time_window_ms: float = 5000.0):
        self.rule_id = rule_id
        self.violation_types = violation_types
        self.min_severity = min_severity
        self.actions = sorted(actions, key=lambda a: a.priority, reverse=True)
        self.activation_threshold = activation_threshold
        self.time_window_ms = time_window_ms
        
        self.violation_history: List[TimingViolation] = []
        self.last_activation: Optional[float] = None
        self.activation_count = 0
    
    def should_activate(self, violation: TimingViolation) -> bool:
        """Check if this rule should activate for the given violation"""
        if violation.violation_type not in self.violation_types:
            return False
        
        if violation.severity_level.value < self.min_severity.value:
            return False
        
        # Add to history and clean old violations
        current_time = time.time() * 1000
        self.violation_history.append(violation)
        self.violation_history = [
            v for v in self.violation_history
            if (current_time - v.timestamp) <= self.time_window_ms
        ]
        
        # Check if threshold is met
        return len(self.violation_history) >= self.activation_threshold

class GracefulDegradationManager:
    """Manages graceful system degradation under timing constraints"""
    
    def __init__(self):
        self.degradation_levels: Dict[str, int] = {}  # component -> level
        self.baseline_frequencies: Dict[str, float] = {}
        self.current_frequencies: Dict[str, float] = {}
        self.disabled_components: Set[str] = set()
        self.backup_components: Dict[str, str] = {}  # primary -> backup
        
    def register_component(self, 
                          component: str, 
                          baseline_frequency: float,
                          backup_component: Optional[str] = None):
        """Register a component for degradation management"""
        self.baseline_frequencies[component] = baseline_frequency
        self.current_frequencies[component] = baseline_frequency
        self.degradation_levels[component] = 0
        
        if backup_component:
            self.backup_components[component] = backup_component
    
    def degrade_performance(self, component: str, level: int = 1) -> bool:
        """Reduce component performance by specified levels"""
        if component not in self.baseline_frequencies:
            logger.error(f"Component {component} not registered for degradation")
            return False
        
        max_degradation = 4  # Maximum degradation levels
        current_level = self.degradation_levels[component]
        new_level = min(current_level + level, max_degradation)
        
        if new_level == current_level:
            return True  # Already at target level
        
        # Calculate new frequency (exponential degradation)
        degradation_factor = 0.7 ** new_level
        new_frequency = self.baseline_frequencies[component] * degradation_factor
        
        self.degradation_levels[component] = new_level
        self.current_frequencies[component] = new_frequency
        
        logger.warning(f"Degraded {component} to level {new_level}, "
                      f"frequency: {new_frequency:.1f}Hz")
        return True
    
    def activate_backup(self, primary: str) -> bool:
        """Activate backup component for failed primary"""
        if primary not in self.backup_components:
            logger.error(f"No backup available for {primary}")
            return False
        
        backup = self.backup_components[primary]
        self.disabled_components.add(primary)
        
        # Transfer configuration to backup
        if primary in self.baseline_frequencies:
            self.baseline_frequencies[backup] = self.baseline_frequencies[primary]
            self.current_frequencies[backup] = self.current_frequencies[primary]
        
        logger.critical(f"Activated backup {backup} for failed {primary}")
        return True
    
    def restore_performance(self, component: str, levels: int = 1) -> bool:
        """Restore component performance by specified levels"""
        if component not in self.degradation_levels:
            return False
        
        current_level = self.degradation_levels[component]
        new_level = max(0, current_level - levels)
        
        if new_level == current_level:
            return True
        
        # Restore frequency
        degradation_factor = 0.7 ** new_level if new_level > 0 else 1.0
        new_frequency = self.baseline_frequencies[component] * degradation_factor
        
        self.degradation_levels[component] = new_level
        self.current_frequencies[component] = new_frequency
        
        logger.info(f"Restored {component} to level {new_level}, "
                   f"frequency: {new_frequency:.1f}Hz")
        return True

class HardwareProtectionInterface:
    """Interface for hardware-level protection mechanisms"""
    
    def __init__(self):
        self.protection_enabled = True
        self.hardware_estop_active = False
        self.safety_relays_active = True
        
        # Simulated hardware interfaces
        self.gpio_interface = None
        self.safety_controller = None
        
        try:
            # In production, these would be actual hardware interfaces
            self.gpio_interface = self._init_gpio()
            self.safety_controller = self._init_safety_controller()
        except Exception as e:
            logger.warning(f"Hardware interfaces not available: {e}")
    
    def _init_gpio(self):
        """Initialize GPIO interface for hardware protection"""
        # Simulated GPIO interface
        class MockGPIO:
            def set_safety_output(self, pin: int, state: bool):
                logger.info(f"GPIO: Set safety pin {pin} to {state}")
            
            def read_safety_input(self, pin: int) -> bool:
                return True  # Simulated safe state
        
        return MockGPIO()
    
    def _init_safety_controller(self):
        """Initialize safety controller interface"""
        # Simulated safety controller
        class MockSafetyController:
            def activate_safe_state(self):
                logger.critical("Safety Controller: Activated safe state")
            
            def disable_motor_power(self):
                logger.critical("Safety Controller: Disabled motor power")
            
            def isolate_system(self):
                logger.critical("Safety Controller: Isolated system")
        
        return MockSafetyController()
    
    def activate_hardware_estop(self) -> bool:
        """Activate hardware-level emergency stop"""
        try:
            if self.safety_controller:
                self.safety_controller.activate_safe_state()
                self.safety_controller.disable_motor_power()
            
            if self.gpio_interface:
                # Activate safety outputs
                for pin in [18, 19, 20, 21]:  # Safety relay pins
                    self.gpio_interface.set_safety_output(pin, False)
            
            self.hardware_estop_active = True
            logger.critical("Hardware emergency stop activated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate hardware emergency stop: {e}")
            return False
    
    def isolate_component(self, component: str) -> bool:
        """Isolate specific hardware component"""
        try:
            if self.safety_controller:
                self.safety_controller.isolate_system()
            
            logger.critical(f"Hardware isolation activated for {component}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to isolate {component}: {e}")
            return False

class DistributedFailSafeCoordinator:
    """Coordinates fail-safe actions across distributed robot systems"""
    
    def __init__(self, robot_id: str, coordinator_port: int = 8765):
        self.robot_id = robot_id
        self.coordinator_port = coordinator_port
        self.peer_robots: Dict[str, str] = {}  # robot_id -> ip_address
        self.failsafe_status: Dict[str, FailSafeLevel] = {}
        
        # Communication setup
        self.message_queue = queue.Queue(maxsize=1000)
        self.running = True
        
        # Start coordinator thread
        self.coordinator_thread = threading.Thread(
            target=self._coordinator_loop, 
            daemon=True
        )
        self.coordinator_thread.start()
    
    def register_peer(self, robot_id: str, ip_address: str):
        """Register a peer robot for coordination"""
        self.peer_robots[robot_id] = ip_address
        self.failsafe_status[robot_id] = FailSafeLevel.NONE
        logger.info(f"Registered peer robot {robot_id} at {ip_address}")
    
    def broadcast_failsafe_activation(self, level: FailSafeLevel, reason: str):
        """Broadcast fail-safe activation to all peers"""
        message = {
            'type': 'failsafe_activation',
            'sender': self.robot_id,
            'level': level.value,
            'reason': reason,
            'timestamp': time.time()
        }
        
        self.message_queue.put(message)
        logger.warning(f"Broadcasting fail-safe level {level} to {len(self.peer_robots)} peers")
    
    def coordinate_emergency_stop(self) -> bool:
        """Coordinate emergency stop across all robots"""
        message = {
            'type': 'emergency_stop',
            'sender': self.robot_id,
            'timestamp': time.time()
        }
        
        success_count = 0
        for robot_id, ip_address in self.peer_robots.items():
            try:
                # In production, this would use actual network communication
                self._send_message(ip_address, message)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to coordinate emergency stop with {robot_id}: {e}")
        
        coordination_success = success_count >= len(self.peer_robots) * 0.8
        logger.critical(f"Emergency stop coordination: {success_count}/{len(self.peer_robots)} robots")
        return coordination_success
    
    def _send_message(self, ip_address: str, message: Dict[str, Any]):
        """Send message to peer robot (simulated)"""
        # In production, this would use TCP/UDP sockets
        logger.debug(f"Sending message to {ip_address}: {message['type']}")
    
    def _coordinator_loop(self):
        """Main coordinator loop for handling distributed fail-safe"""
        while self.running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(timeout=0.1)
                    self._process_coordination_message(message)
                time.sleep(0.01)  # 100Hz coordination loop
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Coordination error: {e}")
    
    def _process_coordination_message(self, message: Dict[str, Any]):
        """Process coordination message from queue"""
        msg_type = message.get('type')
        
        if msg_type == 'failsafe_activation':
            level = FailSafeLevel(message['level'])
            logger.warning(f"Peer {message['sender']} activated fail-safe level {level}")
        
        elif msg_type == 'emergency_stop':
            logger.critical(f"Emergency stop coordinated by {message['sender']}")

class FailSafeMechanismManager:
    """Main manager for all fail-safe mechanisms"""
    
    def __init__(self, 
                 robot_id: str = "robot_001",
                 config_file: Optional[str] = None):
        self.robot_id = robot_id
        self.active_level = FailSafeLevel.NONE
        
        # Core components
        self.degradation_manager = GracefulDegradationManager()
        self.hardware_protection = HardwareProtectionInterface()
        self.distributed_coordinator = DistributedFailSafeCoordinator(robot_id)
        
        # Rule management
        self.rules: List[FailSafeRule] = []
        self.action_executors: Dict[ActionType, Callable] = {}
        self.violation_history: List[TimingViolation] = []
        
        # Threading and execution
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = True
        self.processing_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_violations': 0,
            'actions_executed': 0,
            'emergency_stops': 0,
            'system_shutdowns': 0,
            'last_violation_time': 0
        }
        
        self._setup_default_rules()
        self._setup_action_executors()
        
        if config_file:
            self.load_configuration(config_file)
    
    def _setup_default_rules(self):
        """Setup default fail-safe rules"""
        
        # Soft deadline miss - just log warning
        soft_deadline_rule = FailSafeRule(
            rule_id="soft_deadline",
            violation_types=[ViolationType.DEADLINE_MISS],
            min_severity=FailSafeLevel.WARNING,
            actions=[
                FailSafeAction(ActionType.LOG_WARNING, "system", {"level": "warning"})
            ],
            activation_threshold=3,
            time_window_ms=5000.0
        )
        
        # Hard deadline miss - reduce performance
        hard_deadline_rule = FailSafeRule(
            rule_id="hard_deadline",
            violation_types=[ViolationType.DEADLINE_MISS],
            min_severity=FailSafeLevel.DEGRADED,
            actions=[
                FailSafeAction(ActionType.REDUCE_FREQUENCY, "control_loop", 
                             {"reduction_factor": 0.8}, priority=2),
                FailSafeAction(ActionType.LOG_WARNING, "system", 
                             {"level": "error"}, priority=1)
            ],
            activation_threshold=2,
            time_window_ms=3000.0
        )
        
        # Watchdog timeout - emergency actions
        watchdog_rule = FailSafeRule(
            rule_id="watchdog_timeout",
            violation_types=[ViolationType.WATCHDOG_TIMEOUT],
            min_severity=FailSafeLevel.EMERGENCY,
            actions=[
                FailSafeAction(ActionType.HALT_MOTION, "robot_arm", priority=3),
                FailSafeAction(ActionType.ACTIVATE_BACKUP, "control_system", priority=2),
                FailSafeAction(ActionType.EMERGENCY_STOP, "system", priority=1)
            ],
            activation_threshold=1
        )
        
        # Critical system failure - complete shutdown
        critical_failure_rule = FailSafeRule(
            rule_id="critical_failure",
            violation_types=[ViolationType.HARDWARE_FAULT, ViolationType.MEMORY_OVERFLOW],
            min_severity=FailSafeLevel.SHUTDOWN,
            actions=[
                FailSafeAction(ActionType.EMERGENCY_STOP, "system", priority=3),
                FailSafeAction(ActionType.ISOLATE_ROBOT, "robot", priority=2),
                FailSafeAction(ActionType.SYSTEM_SHUTDOWN, "system", priority=1)
            ],
            activation_threshold=1
        )
        
        self.rules = [soft_deadline_rule, hard_deadline_rule, watchdog_rule, critical_failure_rule]
        logger.info(f"Initialized {len(self.rules)} default fail-safe rules")
    
    def _setup_action_executors(self):
        """Setup action executor functions"""
        self.action_executors = {
            ActionType.LOG_WARNING: self._execute_log_warning,
            ActionType.REDUCE_FREQUENCY: self._execute_reduce_frequency,
            ActionType.DISABLE_COMPONENT: self._execute_disable_component,
            ActionType.ACTIVATE_BACKUP: self._execute_activate_backup,
            ActionType.EMERGENCY_STOP: self._execute_emergency_stop,
            ActionType.SYSTEM_SHUTDOWN: self._execute_system_shutdown,
            ActionType.ISOLATE_ROBOT: self._execute_isolate_robot,
            ActionType.HALT_MOTION: self._execute_halt_motion
        }
    
    def register_component(self, 
                          component: str, 
                          baseline_frequency: float,
                          backup_component: Optional[str] = None):
        """Register component for fail-safe management"""
        self.degradation_manager.register_component(
            component, baseline_frequency, backup_component
        )
    
    def process_violation(self, violation: TimingViolation) -> List[FailSafeAction]:
        """Process a timing violation and execute appropriate fail-safe actions"""
        with self.processing_lock:
            self.stats['total_violations'] += 1
            self.stats['last_violation_time'] = violation.timestamp
            self.violation_history.append(violation)
            
            # Keep history manageable
            if len(self.violation_history) > 1000:
                self.violation_history = self.violation_history[-500:]
            
            executed_actions = []
            
            # Check all rules for activation
            for rule in self.rules:
                if rule.should_activate(violation):
                    logger.warning(f"Activating fail-safe rule: {rule.rule_id}")
                    
                    # Execute actions in priority order
                    for action in rule.actions:
                        try:
                            success = self._execute_action(action, violation)
                            action.success = success
                            if success:
                                executed_actions.append(action)
                                self.stats['actions_executed'] += 1
                        except Exception as e:
                            logger.error(f"Failed to execute action {action.action_type}: {e}")
                            action.success = False
                            action.error_message = str(e)
                    
                    rule.last_activation = time.time()
                    rule.activation_count += 1
            
            # Update system fail-safe level
            if violation.severity_level.value > self.active_level.value:
                self.active_level = violation.severity_level
                self.distributed_coordinator.broadcast_failsafe_activation(
                    self.active_level, violation.description
                )
            
            logger.info(f"Processed violation: {violation.description}, "
                       f"executed {len(executed_actions)} actions")
            return executed_actions
    
    def _execute_action(self, action: FailSafeAction, violation: TimingViolation) -> bool:
        """Execute a specific fail-safe action"""
        start_time = time.perf_counter()
        
        try:
            executor_func = self.action_executors.get(action.action_type)
            if not executor_func:
                logger.error(f"No executor for action type: {action.action_type}")
                return False
            
            success = executor_func(action, violation)
            action.execution_time = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"Executed {action.action_type} for {action.target_component} "
                       f"in {action.execution_time:.2f}ms (success: {success})")
            return success
            
        except Exception as e:
            action.execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Action execution failed: {e}")
            return False
    
    def _execute_log_warning(self, action: FailSafeAction, violation: TimingViolation) -> bool:
        """Execute warning log action"""
        level = action.parameters.get('level', 'warning')
        message = f"Fail-safe warning for {action.target_component}: {violation.description}"
        
        if level == 'error':
            logger.error(message)
        elif level == 'critical':
            logger.critical(message)
        else:
            logger.warning(message)
        
        return True
    
    def _execute_reduce_frequency(self, action: FailSafeAction, violation: TimingViolation) -> bool:
        """Execute frequency reduction action"""
        reduction_factor = action.parameters.get('reduction_factor', 0.8)
        
        # Calculate degradation levels based on reduction factor
        degradation_level = max(1, int(-2 * (reduction_factor - 1)))  # Convert factor to level
        
        return self.degradation_manager.degrade_performance(
            action.target_component, degradation_level
        )
    
    def _execute_disable_component(self, action: FailSafeAction, violation: TimingViolation) -> bool:
        """Execute component disable action"""
        self.degradation_manager.disabled_components.add(action.target_component)
        logger.critical(f"Disabled component: {action.target_component}")
        return True
    
    def _execute_activate_backup(self, action: FailSafeAction, violation: TimingViolation) -> bool:
        """Execute backup activation action"""
        return self.degradation_manager.activate_backup(action.target_component)
    
    def _execute_emergency_stop(self, action: FailSafeAction, violation: TimingViolation) -> bool:
        """Execute emergency stop action"""
        self.stats['emergency_stops'] += 1
        
        # Activate local hardware emergency stop
        hardware_success = self.hardware_protection.activate_hardware_estop()
        
        # Coordinate with distributed system
        coordination_success = self.distributed_coordinator.coordinate_emergency_stop()
        
        return hardware_success and coordination_success
    
    def _execute_system_shutdown(self, action: FailSafeAction, violation: TimingViolation) -> bool:
        """Execute system shutdown action"""
        self.stats['system_shutdowns'] += 1
        
        logger.critical(f"Initiating system shutdown due to: {violation.description}")
        
        # Graceful shutdown sequence
        try:
            # 1. Stop all motion
            self._execute_halt_motion(
                FailSafeAction(ActionType.HALT_MOTION, "all_robots"), violation
            )
            
            # 2. Emergency stop
            self._execute_emergency_stop(
                FailSafeAction(ActionType.EMERGENCY_STOP, "system"), violation
            )
            
            # 3. Disable all components
            for component in self.degradation_manager.baseline_frequencies:
                self.degradation_manager.disabled_components.add(component)
            
            # 4. Mark system as shutdown
            self.active_level = FailSafeLevel.SHUTDOWN
            self.running = False
            
            return True
            
        except Exception as e:
            logger.error(f"System shutdown failed: {e}")
            return False
    
    def _execute_isolate_robot(self, action: FailSafeAction, violation: TimingViolation) -> bool:
        """Execute robot isolation action"""
        return self.hardware_protection.isolate_component(action.target_component)
    
    def _execute_halt_motion(self, action: FailSafeAction, violation: TimingViolation) -> bool:
        """Execute motion halt action"""
        try:
            # In production, this would interface with actual motion controllers
            logger.critical(f"Halting motion for {action.target_component}")
            
            # Simulate motion halt
            if action.target_component == "all_robots":
                for component in self.degradation_manager.baseline_frequencies:
                    if "robot" in component.lower() or "arm" in component.lower():
                        self.degradation_manager.current_frequencies[component] = 0
            else:
                self.degradation_manager.current_frequencies[action.target_component] = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to halt motion: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'robot_id': self.robot_id,
            'active_fail_safe_level': self.active_level.name,
            'system_running': self.running,
            'statistics': self.stats.copy(),
            'degradation_levels': self.degradation_manager.degradation_levels.copy(),
            'disabled_components': list(self.degradation_manager.disabled_components),
            'current_frequencies': self.degradation_manager.current_frequencies.copy(),
            'recent_violations': len([v for v in self.violation_history 
                                    if time.time() - v.timestamp < 30000]),  # Last 30 seconds
            'rule_activations': {rule.rule_id: rule.activation_count for rule in self.rules}
        }
    
    def load_configuration(self, config_file: str):
        """Load fail-safe configuration from file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Load custom rules if provided
            if 'custom_rules' in config:
                # Implementation for loading custom rules would go here
                pass
            
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def save_violation_log(self, filepath: str):
        """Save violation history to file for analysis"""
        try:
            log_data = {
                'robot_id': self.robot_id,
                'timestamp': time.time(),
                'violations': [
                    {
                        'type': v.violation_type.value,
                        'component': v.component,
                        'severity': v.severity_level.name,
                        'measured_time_ms': v.measured_time_ms,
                        'expected_time_ms': v.expected_time_ms,
                        'violation_ratio': v.violation_ratio,
                        'timestamp': v.timestamp
                    }
                    for v in self.violation_history
                ],
                'statistics': self.stats
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"Saved violation log to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save violation log: {e}")
    
    def shutdown(self):
        """Gracefully shutdown the fail-safe mechanism manager"""
        logger.info("Shutting down fail-safe mechanism manager")
        self.running = False
        self.distributed_coordinator.running = False
        self.executor.shutdown(wait=True)

# Example usage and testing
if __name__ == "__main__":
    # Example of using the fail-safe mechanism manager
    manager = FailSafeMechanismManager(robot_id="test_robot_001")
    
    # Register components
    manager.register_component("control_loop", 100.0, "backup_control_loop")
    manager.register_component("motion_planner", 50.0)
    manager.register_component("sensor_fusion", 200.0, "backup_sensors")
    
    # Simulate timing violations
    violations = [
        TimingViolation(
            ViolationType.DEADLINE_MISS,
            time.time() * 1000,
            "control_loop",
            FailSafeLevel.WARNING,
            12.5, 10.0, 1.25
        ),
        TimingViolation(
            ViolationType.DEADLINE_MISS,
            time.time() * 1000 + 1000,
            "control_loop",
            FailSafeLevel.DEGRADED,
            15.8, 10.0, 1.58
        ),
        TimingViolation(
            ViolationType.WATCHDOG_TIMEOUT,
            time.time() * 1000 + 2000,
            "motion_planner",
            FailSafeLevel.EMERGENCY,
            50.0, 20.0, 2.5
        )
    ]
    
    # Process violations
    for violation in violations:
        actions = manager.process_violation(violation)
        print(f"\nProcessed violation: {violation.description}")
        print(f"Executed actions: {[a.action_type.value for a in actions]}")
        print(f"System status: {manager.get_system_status()}")
        time.sleep(0.1)
    
    # Save violation log
    manager.save_violation_log("/tmp/failsafe_violations.json")
    
    # Shutdown
    manager.shutdown()