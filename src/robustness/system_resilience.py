"""
System Resilience and Fault Tolerance

This module provides advanced resilience mechanisms including
fault detection, isolation, recovery, and adaptive system behavior
to maintain operation under adverse conditions.

Features:
- Fault detection and isolation
- Adaptive system reconfiguration
- Resource pooling and failover
- State backup and recovery
- Component health monitoring
- Graceful degradation strategies
- System self-healing capabilities

Author: Phase 5 Implementation
Date: 2024
"""

import numpy as np
import time
import threading
import logging
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from pathlib import Path
import queue
import copy
import weakref
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of system faults"""
    HARDWARE = auto()      # Hardware component failures
    SOFTWARE = auto()      # Software bugs and errors
    NETWORK = auto()       # Network connectivity issues
    RESOURCE = auto()      # Resource exhaustion
    CONFIGURATION = auto() # Configuration errors
    EXTERNAL = auto()      # External system failures
    PERFORMANCE = auto()   # Performance degradation


class SystemState(Enum):
    """System operational states"""
    HEALTHY = auto()       # All systems functioning normally
    DEGRADED = auto()      # Some functionality reduced
    CRITICAL = auto()      # Major functionality impaired
    RECOVERY = auto()      # System in recovery mode
    MAINTENANCE = auto()   # System in maintenance mode
    SHUTDOWN = auto()      # System shutting down


class ComponentStatus(Enum):
    """Status of individual components"""
    ACTIVE = auto()        # Component active and healthy
    INACTIVE = auto()      # Component inactive but available
    DEGRADED = auto()      # Component functioning with reduced capability
    FAILED = auto()        # Component has failed
    RECOVERING = auto()    # Component is recovering
    MAINTENANCE = auto()   # Component in maintenance mode


@dataclass
class ComponentInfo:
    """Information about a system component"""
    component_id: str
    component_type: str
    status: ComponentStatus = ComponentStatus.INACTIVE
    health_score: float = 1.0
    last_heartbeat: float = 0.0
    error_count: int = 0
    restart_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    backup_available: bool = False
    
    def update_health(self, new_score: float):
        """Update component health score"""
        self.health_score = max(0.0, min(1.0, new_score))
        self.last_heartbeat = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'status': self.status.name,
            'health_score': self.health_score,
            'last_heartbeat': self.last_heartbeat,
            'error_count': self.error_count,
            'restart_count': self.restart_count,
            'performance_metrics': self.performance_metrics,
            'dependencies': self.dependencies,
            'backup_available': self.backup_available
        }


@dataclass
class FaultEvent:
    """Information about a detected fault"""
    fault_id: str
    timestamp: float
    fault_type: FaultType
    component_id: str
    severity: float  # 0.0 to 1.0
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None
    impact_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResilienceConfig:
    """Configuration for system resilience"""
    # Fault detection
    enable_fault_detection: bool = True
    fault_detection_interval: float = 1.0
    health_check_timeout: float = 5.0
    heartbeat_timeout: float = 30.0
    
    # Component management
    max_restart_attempts: int = 3
    restart_cooldown: float = 60.0
    health_threshold: float = 0.7
    critical_health_threshold: float = 0.3
    
    # State management
    enable_state_backup: bool = True
    backup_interval: float = 300.0  # 5 minutes
    max_backup_files: int = 10
    
    # Recovery settings
    enable_auto_recovery: bool = True
    recovery_timeout: float = 120.0
    graceful_degradation: bool = True
    
    # Resource management
    enable_resource_pooling: bool = True
    pool_overflow_threshold: float = 0.8
    resource_reallocation: bool = True
    
    # Performance monitoring
    performance_monitoring: bool = True
    performance_window: float = 60.0
    performance_degradation_threshold: float = 0.5


class ComponentManager:
    """Manages system components and their health"""
    
    def __init__(self, config: ResilienceConfig):
        """Initialize component manager"""
        self.config = config
        self.components = {}  # component_id -> ComponentInfo
        self.component_callbacks = {}  # component_id -> callback functions
        self.health_history = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.RLock()
        
    def register_component(self, component_info: ComponentInfo, 
                          health_callback: Callable = None):
        """Register a component for monitoring"""
        with self._lock:
            self.components[component_info.component_id] = component_info
            if health_callback:
                self.component_callbacks[component_info.component_id] = health_callback
            
            logger.info(f"Registered component: {component_info.component_id}")
    
    def unregister_component(self, component_id: str):
        """Unregister a component"""
        with self._lock:
            if component_id in self.components:
                del self.components[component_id]
            if component_id in self.component_callbacks:
                del self.component_callbacks[component_id]
            
            logger.info(f"Unregistered component: {component_id}")
    
    def update_component_health(self, component_id: str, health_score: float,
                              performance_metrics: Dict[str, float] = None):
        """Update component health and performance metrics"""
        with self._lock:
            if component_id not in self.components:
                logger.warning(f"Unknown component: {component_id}")
                return
            
            component = self.components[component_id]
            component.update_health(health_score)
            
            if performance_metrics:
                component.performance_metrics.update(performance_metrics)
            
            # Store health history
            self.health_history[component_id].append({
                'timestamp': time.time(),
                'health_score': health_score,
                'metrics': performance_metrics or {}
            })
            
            # Update component status based on health
            self._update_component_status(component)
    
    def _update_component_status(self, component: ComponentInfo):
        """Update component status based on health score"""
        if component.health_score >= self.config.health_threshold:
            if component.status in [ComponentStatus.DEGRADED, ComponentStatus.RECOVERING]:
                component.status = ComponentStatus.ACTIVE
        elif component.health_score >= self.config.critical_health_threshold:
            if component.status == ComponentStatus.ACTIVE:
                component.status = ComponentStatus.DEGRADED
        else:
            if component.status != ComponentStatus.FAILED:
                component.status = ComponentStatus.FAILED
                component.error_count += 1
                logger.warning(f"Component {component.component_id} failed (health: {component.health_score})")
    
    def get_component_health(self, component_id: str) -> Optional[float]:
        """Get current health score of a component"""
        with self._lock:
            if component_id in self.components:
                return self.components[component_id].health_score
            return None
    
    def get_failed_components(self) -> List[str]:
        """Get list of failed components"""
        with self._lock:
            return [
                comp_id for comp_id, comp in self.components.items()
                if comp.status == ComponentStatus.FAILED
            ]
    
    def get_degraded_components(self) -> List[str]:
        """Get list of degraded components"""
        with self._lock:
            return [
                comp_id for comp_id, comp in self.components.items()
                if comp.status == ComponentStatus.DEGRADED
            ]
    
    def check_component_dependencies(self, component_id: str) -> Dict[str, bool]:
        """Check if component dependencies are healthy"""
        with self._lock:
            if component_id not in self.components:
                return {}
            
            component = self.components[component_id]
            dependency_status = {}
            
            for dep_id in component.dependencies:
                if dep_id in self.components:
                    dep_component = self.components[dep_id]
                    dependency_status[dep_id] = (
                        dep_component.status in [ComponentStatus.ACTIVE, ComponentStatus.DEGRADED] and
                        dep_component.health_score > self.config.critical_health_threshold
                    )
                else:
                    dependency_status[dep_id] = False
            
            return dependency_status
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get overview of all components"""
        with self._lock:
            overview = {
                'total_components': len(self.components),
                'active_components': 0,
                'degraded_components': 0,
                'failed_components': 0,
                'average_health': 0.0,
                'components': {}
            }
            
            total_health = 0.0
            
            for comp_id, comp in self.components.items():
                overview['components'][comp_id] = comp.to_dict()
                total_health += comp.health_score
                
                if comp.status == ComponentStatus.ACTIVE:
                    overview['active_components'] += 1
                elif comp.status == ComponentStatus.DEGRADED:
                    overview['degraded_components'] += 1
                elif comp.status == ComponentStatus.FAILED:
                    overview['failed_components'] += 1
            
            if self.components:
                overview['average_health'] = total_health / len(self.components)
            
            return overview


class FaultDetector:
    """Detects and classifies system faults"""
    
    def __init__(self, config: ResilienceConfig):
        """Initialize fault detector"""
        self.config = config
        self.detected_faults = deque(maxlen=1000)
        self.fault_patterns = {}
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self, component_manager: ComponentManager):
        """Start fault monitoring"""
        self.component_manager = component_manager
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Fault detection started")
    
    def stop_monitoring(self):
        """Stop fault monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Fault detection stopped")
    
    def _monitoring_loop(self):
        """Fault detection monitoring loop"""
        while self.monitoring_active:
            try:
                self._detect_component_faults()
                self._detect_system_faults()
                self._detect_performance_faults()
                
                time.sleep(self.config.fault_detection_interval)
                
            except Exception as e:
                logger.error(f"Fault detection error: {e}")
                time.sleep(5.0)
    
    def _detect_component_faults(self):
        """Detect component-level faults"""
        failed_components = self.component_manager.get_failed_components()
        
        for comp_id in failed_components:
            component = self.component_manager.components[comp_id]
            
            # Check if this is a new fault
            recent_faults = [
                f for f in self.detected_faults
                if f.component_id == comp_id and not f.resolved and
                time.time() - f.timestamp < 300  # Last 5 minutes
            ]
            
            if not recent_faults:
                fault = FaultEvent(
                    fault_id=f"fault_{comp_id}_{int(time.time())}",
                    timestamp=time.time(),
                    fault_type=FaultType.SOFTWARE,
                    component_id=comp_id,
                    severity=1.0 - component.health_score,
                    description=f"Component {comp_id} health below critical threshold",
                    context={'health_score': component.health_score}
                )
                
                self.detected_faults.append(fault)
                logger.warning(f"Detected fault: {fault.description}")
    
    def _detect_system_faults(self):
        """Detect system-level faults"""
        # Check system resources
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory fault detection
            if memory.percent > 90:
                fault = FaultEvent(
                    fault_id=f"memory_fault_{int(time.time())}",
                    timestamp=time.time(),
                    fault_type=FaultType.RESOURCE,
                    component_id="system",
                    severity=min(1.0, (memory.percent - 80) / 20),
                    description=f"High memory usage: {memory.percent:.1f}%",
                    context={'memory_percent': memory.percent}
                )
                
                # Check if this is a new fault
                if not self._is_duplicate_fault(fault):
                    self.detected_faults.append(fault)
                    logger.warning(f"Detected system fault: {fault.description}")
            
            # CPU fault detection
            if cpu_percent > 95:
                fault = FaultEvent(
                    fault_id=f"cpu_fault_{int(time.time())}",
                    timestamp=time.time(),
                    fault_type=FaultType.RESOURCE,
                    component_id="system",
                    severity=min(1.0, (cpu_percent - 80) / 20),
                    description=f"High CPU usage: {cpu_percent:.1f}%",
                    context={'cpu_percent': cpu_percent}
                )
                
                if not self._is_duplicate_fault(fault):
                    self.detected_faults.append(fault)
                    logger.warning(f"Detected system fault: {fault.description}")
                    
        except Exception as e:
            logger.error(f"System fault detection error: {e}")
    
    def _detect_performance_faults(self):
        """Detect performance-related faults"""
        overview = self.component_manager.get_system_overview()
        
        # Check for overall system performance degradation
        if overview['average_health'] < self.config.performance_degradation_threshold:
            fault = FaultEvent(
                fault_id=f"performance_fault_{int(time.time())}",
                timestamp=time.time(),
                fault_type=FaultType.PERFORMANCE,
                component_id="system",
                severity=1.0 - overview['average_health'],
                description=f"System performance degraded: {overview['average_health']:.2f}",
                context={'average_health': overview['average_health']}
            )
            
            if not self._is_duplicate_fault(fault):
                self.detected_faults.append(fault)
                logger.warning(f"Detected performance fault: {fault.description}")
    
    def _is_duplicate_fault(self, new_fault: FaultEvent) -> bool:
        """Check if fault is a duplicate of recent fault"""
        recent_time = time.time() - 60  # Last minute
        
        for fault in self.detected_faults:
            if (fault.timestamp > recent_time and
                fault.fault_type == new_fault.fault_type and
                fault.component_id == new_fault.component_id and
                not fault.resolved):
                return True
        
        return False
    
    def get_active_faults(self) -> List[FaultEvent]:
        """Get list of active (unresolved) faults"""
        return [f for f in self.detected_faults if not f.resolved]
    
    def resolve_fault(self, fault_id: str):
        """Mark fault as resolved"""
        for fault in self.detected_faults:
            if fault.fault_id == fault_id:
                fault.resolved = True
                fault.resolution_time = time.time()
                logger.info(f"Fault resolved: {fault_id}")
                break


class StateManager:
    """Manages system state backup and recovery"""
    
    def __init__(self, config: ResilienceConfig):
        """Initialize state manager"""
        self.config = config
        self.backup_dir = Path("system_backups")
        self.backup_dir.mkdir(exist_ok=True, parents=True)
        
        self.state_snapshots = {}
        self.backup_active = False
        self.backup_thread = None
        
    def start_backup_service(self):
        """Start automatic state backup service"""
        if not self.config.enable_state_backup:
            return
        
        self.backup_active = True
        self.backup_thread = threading.Thread(
            target=self._backup_loop, daemon=True
        )
        self.backup_thread.start()
        logger.info("State backup service started")
    
    def stop_backup_service(self):
        """Stop state backup service"""
        self.backup_active = False
        if self.backup_thread:
            self.backup_thread.join(timeout=10)
        logger.info("State backup service stopped")
    
    def _backup_loop(self):
        """Backup loop for automatic state saving"""
        while self.backup_active:
            try:
                self._create_system_backup()
                time.sleep(self.config.backup_interval)
            except Exception as e:
                logger.error(f"Backup error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def register_component_state(self, component_id: str, 
                                get_state_func: Callable,
                                set_state_func: Callable):
        """Register component for state backup/recovery"""
        self.state_snapshots[component_id] = {
            'get_state': get_state_func,
            'set_state': set_state_func,
            'last_backup': 0
        }
        logger.info(f"Registered state management for: {component_id}")
    
    def backup_component_state(self, component_id: str) -> bool:
        """Backup state of specific component"""
        if component_id not in self.state_snapshots:
            return False
        
        try:
            state_info = self.state_snapshots[component_id]
            current_state = state_info['get_state']()
            
            # Save state to file
            timestamp = int(time.time())
            filename = f"{component_id}_state_{timestamp}.pkl"
            filepath = self.backup_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(current_state, f)
            
            state_info['last_backup'] = time.time()
            
            # Clean up old backups
            self._cleanup_old_backups(component_id)
            
            logger.info(f"Backed up state for {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"State backup failed for {component_id}: {e}")
            return False
    
    def restore_component_state(self, component_id: str, 
                               backup_timestamp: int = None) -> bool:
        """Restore component state from backup"""
        if component_id not in self.state_snapshots:
            return False
        
        try:
            # Find backup file
            if backup_timestamp:
                filename = f"{component_id}_state_{backup_timestamp}.pkl"
                filepath = self.backup_dir / filename
            else:
                # Find most recent backup
                pattern = f"{component_id}_state_*.pkl"
                backup_files = list(self.backup_dir.glob(pattern))
                if not backup_files:
                    logger.error(f"No backup files found for {component_id}")
                    return False
                
                # Sort by timestamp (newest first)
                backup_files.sort(key=lambda x: int(x.stem.split('_')[-1]), reverse=True)
                filepath = backup_files[0]
            
            # Load and restore state
            with open(filepath, 'rb') as f:
                saved_state = pickle.load(f)
            
            state_info = self.state_snapshots[component_id]
            state_info['set_state'](saved_state)
            
            logger.info(f"Restored state for {component_id} from {filepath.name}")
            return True
            
        except Exception as e:
            logger.error(f"State restore failed for {component_id}: {e}")
            return False
    
    def _create_system_backup(self):
        """Create backup of all registered components"""
        for component_id in self.state_snapshots:
            self.backup_component_state(component_id)
    
    def _cleanup_old_backups(self, component_id: str):
        """Remove old backup files"""
        pattern = f"{component_id}_state_*.pkl"
        backup_files = list(self.backup_dir.glob(pattern))
        
        if len(backup_files) > self.config.max_backup_files:
            # Sort by timestamp and remove oldest
            backup_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            for old_file in backup_files[:-self.config.max_backup_files]:
                old_file.unlink()
                logger.debug(f"Removed old backup: {old_file.name}")
    
    def get_backup_info(self) -> Dict[str, Any]:
        """Get information about available backups"""
        backup_info = {}
        
        for component_id in self.state_snapshots:
            pattern = f"{component_id}_state_*.pkl"
            backup_files = list(self.backup_dir.glob(pattern))
            
            if backup_files:
                timestamps = [int(f.stem.split('_')[-1]) for f in backup_files]
                backup_info[component_id] = {
                    'backup_count': len(backup_files),
                    'latest_backup': max(timestamps),
                    'oldest_backup': min(timestamps),
                    'last_backup_time': self.state_snapshots[component_id]['last_backup']
                }
            else:
                backup_info[component_id] = {
                    'backup_count': 0,
                    'latest_backup': None,
                    'oldest_backup': None,
                    'last_backup_time': 0
                }
        
        return backup_info


class ResilienceOrchestrator:
    """Main orchestrator for system resilience"""
    
    def __init__(self, config: ResilienceConfig):
        """Initialize resilience orchestrator"""
        self.config = config
        
        # Initialize components
        self.component_manager = ComponentManager(config)
        self.fault_detector = FaultDetector(config)
        self.state_manager = StateManager(config)
        
        # System state
        self.system_state = SystemState.HEALTHY
        self.resilience_active = False
        
        # Recovery tracking
        self.recovery_attempts = defaultdict(int)
        self.last_recovery_time = defaultdict(float)
        
        logger.info("Resilience orchestrator initialized")
    
    def start(self):
        """Start resilience system"""
        self.resilience_active = True
        
        # Start monitoring components
        self.fault_detector.start_monitoring(self.component_manager)
        self.state_manager.start_backup_service()
        
        logger.info("Resilience system started")
    
    def stop(self):
        """Stop resilience system"""
        self.resilience_active = False
        
        # Stop monitoring
        self.fault_detector.stop_monitoring()
        self.state_manager.stop_backup_service()
        
        logger.info("Resilience system stopped")
    
    def register_component(self, component_info: ComponentInfo,
                          health_callback: Callable = None,
                          get_state_func: Callable = None,
                          set_state_func: Callable = None):
        """Register component for resilience management"""
        # Register with component manager
        self.component_manager.register_component(component_info, health_callback)
        
        # Register for state management if functions provided
        if get_state_func and set_state_func:
            self.state_manager.register_component_state(
                component_info.component_id, get_state_func, set_state_func
            )
    
    def handle_component_failure(self, component_id: str) -> bool:
        """Handle component failure with recovery strategies"""
        if not self.resilience_active:
            return False
        
        logger.warning(f"Handling failure for component: {component_id}")
        
        # Check recovery attempts
        if self.recovery_attempts[component_id] >= self.config.max_restart_attempts:
            logger.error(f"Max recovery attempts reached for {component_id}")
            return False
        
        # Check cooldown period
        if (time.time() - self.last_recovery_time[component_id] < 
            self.config.restart_cooldown):
            logger.info(f"Recovery cooldown active for {component_id}")
            return False
        
        # Attempt recovery
        recovery_success = False
        
        try:
            # Try state restoration first
            if self.state_manager.restore_component_state(component_id):
                logger.info(f"State restored for {component_id}")
                recovery_success = True
            
            # Update recovery tracking
            self.recovery_attempts[component_id] += 1
            self.last_recovery_time[component_id] = time.time()
            
            # Update component status
            if recovery_success:
                component = self.component_manager.components.get(component_id)
                if component:
                    component.status = ComponentStatus.RECOVERING
                    component.restart_count += 1
        
        except Exception as e:
            logger.error(f"Recovery failed for {component_id}: {e}")
            recovery_success = False
        
        return recovery_success
    
    def assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        overview = self.component_manager.get_system_overview()
        active_faults = self.fault_detector.get_active_faults()
        
        # Determine system state
        if overview['failed_components'] > 0:
            if overview['failed_components'] >= overview['total_components'] * 0.5:
                self.system_state = SystemState.CRITICAL
            else:
                self.system_state = SystemState.DEGRADED
        elif overview['degraded_components'] > 0:
            self.system_state = SystemState.DEGRADED
        else:
            self.system_state = SystemState.HEALTHY
        
        # Calculate health metrics
        health_assessment = {
            'system_state': self.system_state.name,
            'overall_health': overview['average_health'],
            'component_overview': overview,
            'active_faults': len(active_faults),
            'critical_faults': len([f for f in active_faults if f.severity > 0.7]),
            'recovery_statistics': {
                'total_attempts': sum(self.recovery_attempts.values()),
                'components_recovered': len([
                    comp_id for comp_id, attempts in self.recovery_attempts.items()
                    if attempts > 0
                ])
            },
            'backup_status': self.state_manager.get_backup_info()
        }
        
        return health_assessment
    
    def trigger_graceful_degradation(self, affected_components: List[str] = None):
        """Trigger graceful system degradation"""
        if not self.config.graceful_degradation:
            return
        
        logger.warning("Triggering graceful system degradation")
        
        # Identify non-critical components to disable
        if affected_components is None:
            affected_components = self.component_manager.get_degraded_components()
        
        for comp_id in affected_components:
            component = self.component_manager.components.get(comp_id)
            if component and component.status != ComponentStatus.FAILED:
                component.status = ComponentStatus.INACTIVE
                logger.info(f"Component {comp_id} set to inactive for degradation")
        
        self.system_state = SystemState.DEGRADED
    
    def export_resilience_report(self, filepath: str = None) -> str:
        """Export comprehensive resilience report"""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"resilience_report_{timestamp}.json"
        
        # Gather comprehensive data
        health_assessment = self.assess_system_health()
        active_faults = self.fault_detector.get_active_faults()
        all_faults = list(self.fault_detector.detected_faults)
        
        report = {
            'report_metadata': {
                'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'report_type': 'resilience_analysis',
                'resilience_active': self.resilience_active
            },
            'system_health': health_assessment,
            'fault_analysis': {
                'total_faults_detected': len(all_faults),
                'active_faults': len(active_faults),
                'resolved_faults': len([f for f in all_faults if f.resolved]),
                'fault_types': defaultdict(int),
                'component_fault_counts': defaultdict(int)
            },
            'component_details': health_assessment['component_overview']['components'],
            'recovery_history': {
                'recovery_attempts': dict(self.recovery_attempts),
                'last_recovery_times': dict(self.last_recovery_time)
            }
        }
        
        # Analyze fault patterns
        for fault in all_faults:
            report['fault_analysis']['fault_types'][fault.fault_type.name] += 1
            report['fault_analysis']['component_fault_counts'][fault.component_id] += 1
        
        # Convert defaultdict to regular dict for JSON serialization
        report['fault_analysis']['fault_types'] = dict(report['fault_analysis']['fault_types'])
        report['fault_analysis']['component_fault_counts'] = dict(report['fault_analysis']['component_fault_counts'])
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Resilience report exported to {filepath}")
        return filepath


# Convenience functions
def create_resilience_system(config: ResilienceConfig = None) -> ResilienceOrchestrator:
    """Create resilience system with default or custom configuration"""
    if config is None:
        config = ResilienceConfig()
    return ResilienceOrchestrator(config)


# Global resilience system
_global_resilience_system = None


def get_global_resilience_system() -> ResilienceOrchestrator:
    """Get global resilience system"""
    global _global_resilience_system
    if _global_resilience_system is None:
        _global_resilience_system = create_resilience_system()
    return _global_resilience_system


def initialize_global_resilience(config: ResilienceConfig = None):
    """Initialize and start global resilience system"""
    global _global_resilience_system
    _global_resilience_system = create_resilience_system(config)
    _global_resilience_system.start()


def shutdown_global_resilience():
    """Shutdown global resilience system"""
    global _global_resilience_system
    if _global_resilience_system:
        _global_resilience_system.stop()


# Example usage and testing
if __name__ == "__main__":
    # Test resilience system
    logger.info("Testing System Resilience")
    
    # Create resilience system
    config = ResilienceConfig(
        fault_detection_interval=2.0,
        enable_auto_recovery=True,
        backup_interval=30.0
    )
    
    resilience = ResilienceOrchestrator(config)
    resilience.start()
    
    try:
        # Create test components
        comp1 = ComponentInfo(
            component_id="test_component_1",
            component_type="computation",
            dependencies=[]
        )
        
        comp2 = ComponentInfo(
            component_id="test_component_2", 
            component_type="sensor",
            dependencies=["test_component_1"]
        )
        
        # Test state functions
        test_state = {"value": 42, "status": "active"}
        
        def get_state():
            return test_state.copy()
        
        def set_state(new_state):
            test_state.update(new_state)
        
        # Register components
        resilience.register_component(comp1, get_state_func=get_state, set_state_func=set_state)
        resilience.register_component(comp2)
        
        # Simulate normal operation
        for i in range(10):
            # Update component health with some variation
            health1 = 0.9 + 0.1 * np.sin(i * 0.5)
            health2 = 0.8 + 0.2 * np.cos(i * 0.3)
            
            resilience.component_manager.update_component_health(
                "test_component_1", health1, {"cpu_usage": 30 + i}
            )
            resilience.component_manager.update_component_health(
                "test_component_2", health2, {"accuracy": 0.95 - i * 0.01}
            )
            
            time.sleep(1)
        
        # Simulate component failure
        logger.info("Simulating component failure...")
        resilience.component_manager.update_component_health("test_component_1", 0.1)
        
        # Wait for fault detection
        time.sleep(5)
        
        # Attempt recovery
        recovery_success = resilience.handle_component_failure("test_component_1")
        logger.info(f"Recovery attempt: {'successful' if recovery_success else 'failed'}")
        
        # Wait a bit more for monitoring
        time.sleep(5)
        
        # Get system health assessment
        health_assessment = resilience.assess_system_health()
        logger.info("System Health Assessment:")
        logger.info(f"  System State: {health_assessment['system_state']}")
        logger.info(f"  Overall Health: {health_assessment['overall_health']:.2f}")
        logger.info(f"  Active Faults: {health_assessment['active_faults']}")
        logger.info(f"  Failed Components: {health_assessment['component_overview']['failed_components']}")
        
        # Export resilience report
        report_file = resilience.export_resilience_report("test_resilience_report.json")
        logger.info(f"Resilience report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Resilience test failed: {e}")
    finally:
        resilience.stop()
    
    print("System resilience test completed!")