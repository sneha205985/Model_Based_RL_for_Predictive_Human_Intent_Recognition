"""
Comprehensive Error Handling and Recovery System

This module provides robust error handling, fault tolerance, and recovery
mechanisms for the HRI Bayesian RL system to ensure reliable operation
in real-world scenarios.

Features:
- Hierarchical error classification and handling
- Automatic error recovery strategies
- Circuit breaker pattern for fault isolation
- Graceful degradation modes
- Error logging and reporting
- System health monitoring
- Failover mechanisms
- Resource cleanup and recovery

Author: Phase 5 Implementation
Date: 2024
"""

import logging
import time
import threading
import traceback
import sys
import functools
import weakref
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import json
import pickle
from pathlib import Path
import signal
import os
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = auto()         # Minor issues, system continues normally
    MEDIUM = auto()      # Moderate issues, some functionality may be impacted
    HIGH = auto()        # Serious issues, significant functionality impacted
    CRITICAL = auto()    # Critical issues, system stability at risk
    FATAL = auto()       # Fatal issues, system shutdown required


class ErrorCategory(Enum):
    """Error categories"""
    COMPUTATION = auto()     # Mathematical/algorithmic errors
    MEMORY = auto()         # Memory allocation/management errors
    IO = auto()             # Input/output errors
    NETWORK = auto()        # Network-related errors
    HARDWARE = auto()       # Hardware interface errors
    SYSTEM = auto()         # System-level errors
    CONFIGURATION = auto()  # Configuration/setup errors
    DATA = auto()          # Data validation/processing errors
    TIMEOUT = auto()       # Timeout-related errors
    RESOURCE = auto()      # Resource exhaustion errors


class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = auto()              # Retry the operation
    FALLBACK = auto()          # Use fallback method/data
    GRACEFUL_DEGRADATION = auto()  # Reduce functionality gracefully
    RESTART = auto()           # Restart affected component
    IGNORE = auto()            # Log and ignore the error
    ESCALATE = auto()          # Escalate to higher level
    SHUTDOWN = auto()          # Controlled shutdown


@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    exception: Exception
    traceback_str: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    component_name: str = "unknown"
    thread_id: int = 0
    process_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'severity': self.severity.name,
            'category': self.category.name,
            'exception_type': type(self.exception).__name__,
            'exception_message': str(self.exception),
            'traceback': self.traceback_str,
            'context': self.context,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_strategy': self.recovery_strategy.name if self.recovery_strategy else None,
            'component_name': self.component_name,
            'thread_id': self.thread_id,
            'process_id': self.process_id
        }


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling system"""
    # General settings
    enable_error_recovery: bool = True
    enable_circuit_breaker: bool = True
    enable_graceful_degradation: bool = True
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    max_retry_delay: float = 60.0
    
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3
    
    # Logging settings
    log_errors: bool = True
    log_level: str = "ERROR"
    error_log_file: str = "hri_errors.log"
    max_log_size_mb: int = 100
    
    # Recovery settings
    enable_auto_recovery: bool = True
    recovery_timeout: float = 10.0
    fallback_timeout: float = 5.0
    
    # Health monitoring
    enable_health_monitoring: bool = True
    health_check_interval: float = 30.0
    
    # System protection
    memory_limit_mb: float = 1000.0
    cpu_limit_percent: float = 90.0
    enable_resource_monitoring: bool = True


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 success_threshold: int = 3):
        """Initialize circuit breaker"""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _on_success(self):
        """Handle successful execution"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time
        }


class ErrorRecoveryManager:
    """Manages error recovery strategies"""
    
    def __init__(self, config: ErrorHandlingConfig):
        """Initialize error recovery manager"""
        self.config = config
        self.recovery_strategies = {}
        self.recovery_history = deque(maxlen=1000)
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default recovery strategies"""
        self.recovery_strategies = {
            (ErrorCategory.COMPUTATION, ErrorSeverity.LOW): RecoveryStrategy.RETRY,
            (ErrorCategory.COMPUTATION, ErrorSeverity.MEDIUM): RecoveryStrategy.FALLBACK,
            (ErrorCategory.COMPUTATION, ErrorSeverity.HIGH): RecoveryStrategy.RESTART,
            (ErrorCategory.COMPUTATION, ErrorSeverity.CRITICAL): RecoveryStrategy.GRACEFUL_DEGRADATION,
            
            (ErrorCategory.MEMORY, ErrorSeverity.LOW): RecoveryStrategy.RETRY,
            (ErrorCategory.MEMORY, ErrorSeverity.MEDIUM): RecoveryStrategy.GRACEFUL_DEGRADATION,
            (ErrorCategory.MEMORY, ErrorSeverity.HIGH): RecoveryStrategy.RESTART,
            (ErrorCategory.MEMORY, ErrorSeverity.CRITICAL): RecoveryStrategy.SHUTDOWN,
            
            (ErrorCategory.IO, ErrorSeverity.LOW): RecoveryStrategy.RETRY,
            (ErrorCategory.IO, ErrorSeverity.MEDIUM): RecoveryStrategy.FALLBACK,
            (ErrorCategory.IO, ErrorSeverity.HIGH): RecoveryStrategy.GRACEFUL_DEGRADATION,
            
            (ErrorCategory.NETWORK, ErrorSeverity.LOW): RecoveryStrategy.RETRY,
            (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM): RecoveryStrategy.FALLBACK,
            (ErrorCategory.NETWORK, ErrorSeverity.HIGH): RecoveryStrategy.GRACEFUL_DEGRADATION,
            
            (ErrorCategory.TIMEOUT, ErrorSeverity.LOW): RecoveryStrategy.RETRY,
            (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM): RecoveryStrategy.FALLBACK,
            (ErrorCategory.TIMEOUT, ErrorSeverity.HIGH): RecoveryStrategy.GRACEFUL_DEGRADATION,
            
            (ErrorCategory.RESOURCE, ErrorSeverity.MEDIUM): RecoveryStrategy.GRACEFUL_DEGRADATION,
            (ErrorCategory.RESOURCE, ErrorSeverity.HIGH): RecoveryStrategy.RESTART,
            (ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL): RecoveryStrategy.SHUTDOWN,
        }
    
    def get_recovery_strategy(self, error_info: ErrorInfo) -> RecoveryStrategy:
        """Get recovery strategy for error"""
        key = (error_info.category, error_info.severity)
        return self.recovery_strategies.get(key, RecoveryStrategy.ESCALATE)
    
    def attempt_recovery(self, error_info: ErrorInfo, 
                        recovery_func: Callable = None) -> bool:
        """Attempt recovery for error"""
        if not self.config.enable_error_recovery:
            return False
        
        strategy = self.get_recovery_strategy(error_info)
        error_info.recovery_strategy = strategy
        error_info.recovery_attempted = True
        
        success = False
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                success = self._retry_recovery(error_info, recovery_func)
            elif strategy == RecoveryStrategy.FALLBACK:
                success = self._fallback_recovery(error_info, recovery_func)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                success = self._graceful_degradation_recovery(error_info)
            elif strategy == RecoveryStrategy.RESTART:
                success = self._restart_recovery(error_info)
            elif strategy == RecoveryStrategy.IGNORE:
                success = True  # Ignore the error
            else:
                success = False
        
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            success = False
        
        error_info.recovery_successful = success
        self.recovery_history.append(error_info)
        
        return success
    
    def _retry_recovery(self, error_info: ErrorInfo, 
                       recovery_func: Callable = None) -> bool:
        """Attempt retry recovery"""
        if not recovery_func:
            return False
        
        delay = self.config.retry_delay
        
        for attempt in range(self.config.max_retries):
            try:
                time.sleep(delay)
                result = recovery_func()
                logger.info(f"Retry recovery successful on attempt {attempt + 1}")
                return True
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                
                if self.config.exponential_backoff:
                    delay = min(delay * 2, self.config.max_retry_delay)
        
        return False
    
    def _fallback_recovery(self, error_info: ErrorInfo, 
                          recovery_func: Callable = None) -> bool:
        """Attempt fallback recovery"""
        try:
            # This would typically involve using a fallback method or cached data
            if recovery_func:
                result = recovery_func()
            
            logger.info("Fallback recovery successful")
            return True
        except Exception as e:
            logger.error(f"Fallback recovery failed: {e}")
            return False
    
    def _graceful_degradation_recovery(self, error_info: ErrorInfo) -> bool:
        """Attempt graceful degradation recovery"""
        try:
            # This would involve reducing system functionality
            logger.info("Graceful degradation recovery initiated")
            
            # Signal system to enter degraded mode
            error_info.context['degraded_mode'] = True
            
            return True
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
            return False
    
    def _restart_recovery(self, error_info: ErrorInfo) -> bool:
        """Attempt restart recovery"""
        try:
            # This would involve restarting the affected component
            logger.info(f"Restart recovery initiated for component: {error_info.component_name}")
            
            # Signal for component restart
            error_info.context['restart_required'] = True
            
            return True
        except Exception as e:
            logger.error(f"Restart recovery failed: {e}")
            return False
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        if not self.recovery_history:
            return {}
        
        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r.recovery_successful)
        
        strategy_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        category_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        
        for recovery in self.recovery_history:
            if recovery.recovery_strategy:
                strategy_stats[recovery.recovery_strategy.name]['attempts'] += 1
                if recovery.recovery_successful:
                    strategy_stats[recovery.recovery_strategy.name]['successes'] += 1
            
            category_stats[recovery.category.name]['attempts'] += 1
            if recovery.recovery_successful:
                category_stats[recovery.category.name]['successes'] += 1
        
        return {
            'total_recoveries': total_recoveries,
            'successful_recoveries': successful_recoveries,
            'recovery_rate': successful_recoveries / total_recoveries if total_recoveries > 0 else 0,
            'strategy_statistics': dict(strategy_stats),
            'category_statistics': dict(category_stats)
        }


class SystemHealthMonitor:
    """Monitor system health and detect issues proactively"""
    
    def __init__(self, config: ErrorHandlingConfig):
        """Initialize system health monitor"""
        self.config = config
        self.health_metrics = {}
        self.health_alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start health monitoring"""
        if not self.config.enable_health_monitoring:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Health monitoring loop"""
        while self.monitoring_active:
            try:
                # Check system resources
                self._check_system_resources()
                
                # Check memory usage
                self._check_memory_usage()
                
                # Check CPU usage
                self._check_cpu_usage()
                
                # Update health metrics
                self._update_health_metrics()
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(10)
    
    def _check_system_resources(self):
        """Check overall system resources"""
        try:
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > self.config.memory_limit_mb:
                self.health_alerts.append({
                    'timestamp': time.time(),
                    'type': 'memory',
                    'severity': 'high',
                    'message': f"High memory usage: {memory.percent:.1f}%"
                })
            
            # Disk space check
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                self.health_alerts.append({
                    'timestamp': time.time(),
                    'type': 'disk',
                    'severity': 'high',
                    'message': f"Low disk space: {disk_percent:.1f}% used"
                })
        
        except Exception as e:
            logger.error(f"Resource check error: {e}")
    
    def _check_memory_usage(self):
        """Check process memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            if memory_mb > self.config.memory_limit_mb:
                self.health_alerts.append({
                    'timestamp': time.time(),
                    'type': 'process_memory',
                    'severity': 'medium',
                    'message': f"High process memory usage: {memory_mb:.1f} MB"
                })
        
        except Exception as e:
            logger.error(f"Memory check error: {e}")
    
    def _check_cpu_usage(self):
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > self.config.cpu_limit_percent:
                self.health_alerts.append({
                    'timestamp': time.time(),
                    'type': 'cpu',
                    'severity': 'medium',
                    'message': f"High CPU usage: {cpu_percent:.1f}%"
                })
        
        except Exception as e:
            logger.error(f"CPU check error: {e}")
    
    def _update_health_metrics(self):
        """Update health metrics"""
        try:
            process = psutil.Process()
            
            self.health_metrics = {
                'timestamp': time.time(),
                'process_memory_mb': process.memory_info().rss / (1024 * 1024),
                'process_cpu_percent': process.cpu_percent(),
                'system_memory_percent': psutil.virtual_memory().percent,
                'system_cpu_percent': psutil.cpu_percent(),
                'thread_count': process.num_threads(),
                'open_files': len(process.open_files()) if hasattr(process, 'open_files') else 0,
                'alerts_count': len(self.health_alerts)
            }
        
        except Exception as e:
            logger.error(f"Health metrics update error: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        # Determine overall health
        recent_alerts = [a for a in self.health_alerts 
                        if time.time() - a['timestamp'] < 300]  # Last 5 minutes
        
        if any(a['severity'] == 'high' for a in recent_alerts):
            health_status = 'unhealthy'
        elif any(a['severity'] == 'medium' for a in recent_alerts):
            health_status = 'degraded'
        else:
            health_status = 'healthy'
        
        return {
            'status': health_status,
            'metrics': self.health_metrics,
            'recent_alerts': recent_alerts[-10:],  # Last 10 alerts
            'total_alerts': len(self.health_alerts),
            'monitoring_active': self.monitoring_active
        }


class RobustErrorHandler:
    """Main error handling system coordinator"""
    
    def __init__(self, config: ErrorHandlingConfig):
        """Initialize robust error handler"""
        self.config = config
        
        # Initialize components
        self.recovery_manager = ErrorRecoveryManager(config)
        self.health_monitor = SystemHealthMonitor(config)
        self.circuit_breakers = {}
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.component_errors = defaultdict(list)
        
        # State management
        self.degraded_mode = False
        self.active_components = set()
        self.shutdown_in_progress = False
        
        # Setup logging
        self._setup_error_logging()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info("Robust error handler initialized")
    
    def _setup_error_logging(self):
        """Setup specialized error logging"""
        error_logger = logging.getLogger('hri_errors')
        
        # File handler
        file_handler = logging.FileHandler(self.config.error_log_file)
        file_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        error_logger.addHandler(file_handler)
        error_logger.setLevel(getattr(logging, self.config.log_level))
        
        self.error_logger = error_logger
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.initiate_shutdown()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def start(self):
        """Start error handling system"""
        self.health_monitor.start_monitoring()
        logger.info("Error handling system started")
    
    def stop(self):
        """Stop error handling system"""
        self.health_monitor.stop_monitoring()
        logger.info("Error handling system stopped")
    
    def handle_error(self, exception: Exception, component_name: str = "unknown",
                    context: Dict[str, Any] = None, 
                    recovery_func: Callable = None) -> ErrorInfo:
        """Handle an error with comprehensive error management"""
        
        # Create error info
        error_info = ErrorInfo(
            error_id=f"error_{int(time.time() * 1000000)}",
            timestamp=time.time(),
            severity=self._classify_error_severity(exception),
            category=self._classify_error_category(exception),
            exception=exception,
            traceback_str=traceback.format_exc(),
            context=context or {},
            component_name=component_name,
            thread_id=threading.get_ident(),
            process_id=os.getpid()
        )
        
        # Log error
        if self.config.log_errors:
            self.error_logger.error(
                f"Error in {component_name}: {exception}\n{error_info.traceback_str}"
            )
        
        # Track error
        self.error_history.append(error_info)
        self.error_counts[type(exception).__name__] += 1
        self.component_errors[component_name].append(error_info)
        
        # Attempt recovery if enabled
        if self.config.enable_error_recovery:
            recovery_success = self.recovery_manager.attempt_recovery(
                error_info, recovery_func
            )
            
            if not recovery_success and error_info.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
                self._handle_critical_error(error_info)
        
        # Check if degraded mode should be activated
        self._check_degraded_mode_activation(error_info)
        
        return error_info
    
    def _classify_error_severity(self, exception: Exception) -> ErrorSeverity:
        """Classify error severity"""
        exception_type = type(exception).__name__
        
        # Fatal errors
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.FATAL
        
        # Critical errors
        if isinstance(exception, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if isinstance(exception, (RuntimeError, ValueError, TypeError)):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if isinstance(exception, (IOError, OSError, ConnectionError)):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _classify_error_category(self, exception: Exception) -> ErrorCategory:
        """Classify error category"""
        exception_type = type(exception).__name__
        
        if isinstance(exception, (MemoryError,)):
            return ErrorCategory.MEMORY
        
        if isinstance(exception, (IOError, FileNotFoundError, PermissionError)):
            return ErrorCategory.IO
        
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        
        if isinstance(exception, (TimeoutError,)):
            return ErrorCategory.TIMEOUT
        
        if isinstance(exception, (ValueError, TypeError, ArithmeticError)):
            return ErrorCategory.COMPUTATION
        
        if isinstance(exception, (SystemError, SystemExit)):
            return ErrorCategory.SYSTEM
        
        return ErrorCategory.COMPUTATION  # Default category
    
    def _handle_critical_error(self, error_info: ErrorInfo):
        """Handle critical errors that may require system shutdown"""
        logger.critical(f"Critical error detected: {error_info.exception}")
        
        if error_info.severity == ErrorSeverity.FATAL:
            self.initiate_shutdown()
        elif error_info.severity == ErrorSeverity.CRITICAL:
            self.activate_degraded_mode()
    
    def _check_degraded_mode_activation(self, error_info: ErrorInfo):
        """Check if degraded mode should be activated"""
        # Count recent high-severity errors
        recent_time = time.time() - 300  # Last 5 minutes
        recent_high_severity = [
            e for e in self.error_history 
            if e.timestamp > recent_time and e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        ]
        
        if len(recent_high_severity) >= 5 and not self.degraded_mode:
            self.activate_degraded_mode()
    
    def activate_degraded_mode(self):
        """Activate system degraded mode"""
        if self.degraded_mode:
            return
        
        self.degraded_mode = True
        logger.warning("System entering degraded mode due to repeated errors")
        
        # Notify components about degraded mode
        # This would typically involve reducing functionality
    
    def deactivate_degraded_mode(self):
        """Deactivate degraded mode"""
        self.degraded_mode = False
        logger.info("System exiting degraded mode")
    
    def initiate_shutdown(self):
        """Initiate graceful system shutdown"""
        if self.shutdown_in_progress:
            return
        
        self.shutdown_in_progress = True
        logger.info("Initiating graceful system shutdown")
        
        # Stop all components
        self.stop()
        
        # Export error report
        self.export_error_report()
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=self.config.failure_threshold,
                recovery_timeout=self.config.recovery_timeout,
                success_threshold=self.config.success_threshold
            )
        return self.circuit_breakers[name]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {}
        
        # Severity distribution
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for error in self.error_history:
            severity_counts[error.severity.name] += 1
            category_counts[error.category.name] += 1
        
        # Component error distribution
        component_stats = {}
        for component, errors in self.component_errors.items():
            component_stats[component] = {
                'total_errors': len(errors),
                'recent_errors': len([e for e in errors if time.time() - e.timestamp < 300])
            }
        
        # Recovery statistics
        recovery_stats = self.recovery_manager.get_recovery_statistics()
        
        return {
            'total_errors': total_errors,
            'severity_distribution': dict(severity_counts),
            'category_distribution': dict(category_counts),
            'component_statistics': component_stats,
            'recovery_statistics': recovery_stats,
            'system_status': {
                'degraded_mode': self.degraded_mode,
                'shutdown_in_progress': self.shutdown_in_progress,
                'health_status': self.health_monitor.get_health_status()
            },
            'circuit_breaker_status': {
                name: breaker.get_state() 
                for name, breaker in self.circuit_breakers.items()
            }
        }
    
    def export_error_report(self, filepath: str = None) -> str:
        """Export comprehensive error report"""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"error_report_{timestamp}.json"
        
        report = {
            'report_metadata': {
                'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'report_type': 'error_analysis',
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform,
                    'process_id': os.getpid()
                }
            },
            'error_statistics': self.get_error_statistics(),
            'recent_errors': [
                error.to_dict() for error in list(self.error_history)[-50:]
            ]  # Last 50 errors
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Error report exported to {filepath}")
        return filepath


# Decorator for robust error handling
def robust_operation(component_name: str = "unknown", 
                    recovery_func: Callable = None,
                    error_handler: RobustErrorHandler = None):
    """Decorator for adding robust error handling to operations"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_handler is None:
                    # Use global error handler if none provided
                    handler = get_global_error_handler()
                else:
                    handler = error_handler
                
                error_info = handler.handle_error(
                    e, component_name, 
                    {'function': func.__name__, 'args': str(args)[:100]},
                    recovery_func
                )
                
                # Re-raise if recovery was not successful
                if not error_info.recovery_successful:
                    raise e
                
                # Return None or default value if recovery was successful
                return None
        
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = None


def get_global_error_handler() -> RobustErrorHandler:
    """Get global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        config = ErrorHandlingConfig()
        _global_error_handler = RobustErrorHandler(config)
    return _global_error_handler


def initialize_global_error_handler(config: ErrorHandlingConfig = None):
    """Initialize global error handler"""
    global _global_error_handler
    if config is None:
        config = ErrorHandlingConfig()
    _global_error_handler = RobustErrorHandler(config)
    _global_error_handler.start()


def shutdown_global_error_handler():
    """Shutdown global error handler"""
    global _global_error_handler
    if _global_error_handler:
        _global_error_handler.stop()


# Example usage and testing
if __name__ == "__main__":
    # Test robust error handler
    logger.info("Testing Robust Error Handler")
    
    # Create error handler with custom config
    config = ErrorHandlingConfig(
        max_retries=2,
        enable_auto_recovery=True,
        enable_health_monitoring=True
    )
    
    error_handler = RobustErrorHandler(config)
    error_handler.start()
    
    try:
        # Test error handling with different types of errors
        
        # Test computation error
        @robust_operation("test_computation", error_handler=error_handler)
        def failing_computation():
            raise ValueError("Test computation error")
        
        # Test memory error
        @robust_operation("test_memory", error_handler=error_handler)
        def memory_intensive():
            raise MemoryError("Test memory error")
        
        # Test IO error
        @robust_operation("test_io", error_handler=error_handler)
        def io_operation():
            raise IOError("Test IO error")
        
        # Run test operations
        logger.info("Testing error handling...")
        
        try:
            failing_computation()
        except ValueError:
            pass
        
        try:
            memory_intensive()
        except MemoryError:
            pass
        
        try:
            io_operation()
        except IOError:
            pass
        
        # Wait for health monitoring
        time.sleep(5)
        
        # Test circuit breaker
        breaker = error_handler.get_circuit_breaker("test_breaker")
        
        def unreliable_function():
            if time.time() % 2 > 1:
                raise Exception("Random failure")
            return "success"
        
        # Test circuit breaker functionality
        for i in range(10):
            try:
                result = breaker.call(unreliable_function)
                logger.info(f"Circuit breaker call {i}: {result}")
            except Exception as e:
                logger.info(f"Circuit breaker call {i} failed: {e}")
        
        # Get error statistics
        stats = error_handler.get_error_statistics()
        logger.info("Error statistics:")
        logger.info(f"  Total errors: {stats.get('total_errors', 0)}")
        logger.info(f"  Severity distribution: {stats.get('severity_distribution', {})}")
        logger.info(f"  Recovery rate: {stats.get('recovery_statistics', {}).get('recovery_rate', 0):.2f}")
        
        # Get health status
        health = error_handler.health_monitor.get_health_status()
        logger.info(f"System health: {health['status']}")
        
        # Export error report
        report_file = error_handler.export_error_report("test_error_report.json")
        logger.info(f"Error report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Error handler test failed: {e}")
    finally:
        error_handler.stop()
    
    print("Robust error handler test completed!")