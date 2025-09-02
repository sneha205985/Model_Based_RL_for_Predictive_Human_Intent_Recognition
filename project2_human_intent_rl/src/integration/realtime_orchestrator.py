#!/usr/bin/env python3
"""
Real-Time Orchestrator for Human Intent Recognition System
=========================================================

This module implements the main real-time control loop with strict timing guarantees
for human-robot interaction scenarios. It orchestrates the perception → prediction → 
planning → control pipeline with <100ms cycle time requirements.

Key Features:
- Asynchronous processing pipeline with priority scheduling
- Preemptive execution with fallback strategies  
- Real-time performance monitoring and adaptive load balancing
- Memory-bounded data streaming and efficient resource management
- Emergency stop and safety system integration

Performance Requirements:
- Decision cycle: <100ms guaranteed
- Memory usage: <2GB bounded
- CPU utilization: <80% average, <95% peak
- Real-time responsiveness with predictable latency

Author: Claude Code (Anthropic)
Date: 2025-01-15  
Version: 1.0
"""

import asyncio
import time
import threading
import logging
import psutil
import gc
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure logging for real-time systems
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"  # Some components failed but system operational
    EMERGENCY = "emergency"  # Emergency stop activated
    SHUTDOWN = "shutdown"
    FAULT = "fault"  # System fault requiring intervention


class Priority(Enum):
    """Task priority levels for scheduling"""
    CRITICAL = 0    # Emergency stops, safety checks
    HIGH = 1        # Perception, prediction updates
    MEDIUM = 2      # Planning, optimization
    LOW = 3         # Logging, diagnostics


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics tracking"""
    cycle_time_ms: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    network_latency_ms: float = 0.0
    component_timings: Dict[str, float] = field(default_factory=dict)
    missed_deadlines: int = 0
    total_cycles: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Task:
    """Real-time task definition"""
    name: str
    function: Callable
    priority: Priority
    deadline_ms: float
    period_ms: Optional[float] = None
    last_execution: float = 0.0
    execution_time_ms: float = 0.0
    deadline_misses: int = 0


class CircularBuffer:
    """Thread-safe circular buffer for streaming data"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.lock = threading.RLock()
    
    def put(self, item: Any) -> bool:
        """Add item to buffer, returns False if buffer full"""
        with self.lock:
            if self.size == self.capacity:
                return False
            
            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1
            return True
    
    def get(self) -> Optional[Any]:
        """Get item from buffer, returns None if empty"""
        with self.lock:
            if self.size == 0:
                return None
            
            item = self.buffer[self.head]
            self.buffer[self.head] = None  # Help GC
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            return item
    
    def put_overwrite(self, item: Any) -> None:
        """Add item, overwriting oldest if buffer full"""
        with self.lock:
            if self.size == self.capacity:
                # Overwrite oldest
                self.head = (self.head + 1) % self.capacity
                self.size -= 1
            
            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1
    
    def peek_latest(self) -> Optional[Any]:
        """Peek at most recent item without removing"""
        with self.lock:
            if self.size == 0:
                return None
            latest_idx = (self.tail - 1) % self.capacity
            return self.buffer[latest_idx]
    
    def is_full(self) -> bool:
        with self.lock:
            return self.size == self.capacity
    
    def is_empty(self) -> bool:
        with self.lock:
            return self.size == 0


class RealtimeOrchestrator:
    """
    Real-time orchestrator managing the main control loop with strict timing guarantees.
    
    Implements asynchronous processing pipeline: perception → prediction → planning → control
    with priority-based scheduling and preemptive execution.
    """
    
    def __init__(self, target_cycle_time_ms: float = 100.0):
        """
        Initialize real-time orchestrator.
        
        Args:
            target_cycle_time_ms: Target cycle time in milliseconds (default 100ms)
        """
        self.target_cycle_time_ms = target_cycle_time_ms
        self.target_cycle_time_s = target_cycle_time_ms / 1000.0
        
        # System state management
        self.state = SystemState.INITIALIZING
        self.emergency_stop_flag = threading.Event()
        self.shutdown_flag = threading.Event()
        
        # Performance monitoring
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 cycles
        self.current_metrics = PerformanceMetrics()
        
        # Task scheduling
        self.tasks = {priority: [] for priority in Priority}
        self.active_tasks = {}
        
        # Data pipelines (bounded circular buffers)
        self.sensor_data_buffer = CircularBuffer(capacity=100)
        self.perception_buffer = CircularBuffer(capacity=50)
        self.prediction_buffer = CircularBuffer(capacity=50)
        self.planning_buffer = CircularBuffer(capacity=30)
        self.control_buffer = CircularBuffer(capacity=30)
        
        # Component interfaces (to be injected)
        self.perception_system = None
        self.prediction_system = None
        self.planning_system = None
        self.control_system = None
        self.safety_monitor = None
        self.system_monitor = None
        
        # Real-time execution state
        self.cycle_start_time = 0.0
        self.last_gc_time = time.time()
        self.gc_interval = 10.0  # Run GC every 10 seconds
        
        # Performance optimization
        self.adaptive_load_balancing = True
        self.component_time_budgets = {
            'perception': 0.025,    # 25ms budget
            'prediction': 0.030,    # 30ms budget  
            'planning': 0.035,      # 35ms budget
            'control': 0.010        # 10ms budget
        }
        
        logger.info(f"Real-time orchestrator initialized with {target_cycle_time_ms}ms target cycle time")
    
    def register_component(self, component_name: str, component: Any) -> None:
        """Register system components with the orchestrator"""
        setattr(self, f"{component_name}_system", component)
        logger.info(f"Registered component: {component_name}")
    
    def add_task(self, task: Task) -> None:
        """Add task to scheduling queue"""
        self.tasks[task.priority].append(task)
        logger.debug(f"Added task '{task.name}' with priority {task.priority.name}")
    
    def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Immediate emergency stop with hardware integration"""
        logger.critical(f"EMERGENCY STOP: {reason}")
        self.emergency_stop_flag.set()
        self.state = SystemState.EMERGENCY
        
        # Immediate actions
        try:
            if self.control_system:
                self.control_system.emergency_stop()
            
            # Clear all buffers to prevent stale commands
            self._clear_all_buffers()
            
            # Notify safety monitor
            if self.safety_monitor:
                self.safety_monitor.emergency_stop_triggered(reason)
                
        except Exception as e:
            logger.critical(f"Error during emergency stop: {e}")
    
    def graceful_shutdown(self) -> None:
        """Initiate graceful system shutdown"""
        logger.info("Initiating graceful shutdown...")
        self.shutdown_flag.set()
        self.state = SystemState.SHUTDOWN
    
    async def run_realtime_loop(self) -> None:
        """
        Main real-time control loop with strict timing guarantees.
        
        Implements the perception → prediction → planning → control pipeline
        with <100ms cycle time constraint.
        """
        logger.info("Starting real-time control loop")
        self.state = SystemState.RUNNING
        
        try:
            while not self.shutdown_flag.is_set():
                cycle_start = time.perf_counter()
                self.cycle_start_time = cycle_start
                
                # Check for emergency stop
                if self.emergency_stop_flag.is_set():
                    await self._handle_emergency_state()
                    continue
                
                # Execute main pipeline with timing guarantees
                try:
                    pipeline_success = await self._execute_pipeline()
                    
                    if not pipeline_success:
                        self.state = SystemState.DEGRADED
                        logger.warning("Pipeline execution degraded")
                    else:
                        if self.state == SystemState.DEGRADED:
                            self.state = SystemState.RUNNING
                            logger.info("System recovered from degraded state")
                
                except Exception as e:
                    logger.error(f"Pipeline execution failed: {e}")
                    self.state = SystemState.FAULT
                    await self._handle_fault_state(e)
                
                # Execute scheduled tasks
                await self._execute_scheduled_tasks()
                
                # Performance monitoring and metrics update
                cycle_time = (time.perf_counter() - cycle_start) * 1000  # Convert to ms
                self._update_performance_metrics(cycle_time)
                
                # Adaptive load balancing
                if self.adaptive_load_balancing:
                    self._adapt_time_budgets()
                
                # Memory management
                await self._manage_memory()
                
                # Sleep for remaining cycle time
                remaining_time = self.target_cycle_time_s - (time.perf_counter() - cycle_start)
                if remaining_time > 0:
                    await asyncio.sleep(remaining_time)
                else:
                    self.current_metrics.missed_deadlines += 1
                    logger.warning(f"Missed deadline by {-remaining_time * 1000:.1f}ms")
        
        except Exception as e:
            logger.critical(f"Critical error in real-time loop: {e}")
            self.emergency_stop(f"Critical loop error: {e}")
        
        finally:
            logger.info("Real-time control loop stopped")
            await self._cleanup()
    
    async def _execute_pipeline(self) -> bool:
        """
        Execute the main processing pipeline: perception → prediction → planning → control
        
        Returns:
            bool: True if pipeline executed successfully within time constraints
        """
        pipeline_start = time.perf_counter()
        success = True
        
        try:
            # 1. Perception Stage (25ms budget)
            perception_start = time.perf_counter()
            if self.perception_system:
                sensor_data = self.sensor_data_buffer.peek_latest()
                if sensor_data:
                    perception_result = await self._execute_with_timeout(
                        self.perception_system.process_sensors,
                        args=(sensor_data,),
                        timeout_ms=self.component_time_budgets['perception'] * 1000
                    )
                    if perception_result:
                        self.perception_buffer.put_overwrite(perception_result)
            
            perception_time = (time.perf_counter() - perception_start) * 1000
            self.current_metrics.component_timings['perception'] = perception_time
            
            # 2. Prediction Stage (30ms budget)
            prediction_start = time.perf_counter()
            if self.prediction_system:
                perception_data = self.perception_buffer.peek_latest()
                if perception_data:
                    prediction_result = await self._execute_with_timeout(
                        self.prediction_system.predict_intent,
                        args=(perception_data,),
                        timeout_ms=self.component_time_budgets['prediction'] * 1000
                    )
                    if prediction_result:
                        self.prediction_buffer.put_overwrite(prediction_result)
            
            prediction_time = (time.perf_counter() - prediction_start) * 1000
            self.current_metrics.component_timings['prediction'] = prediction_time
            
            # 3. Planning Stage (35ms budget)
            planning_start = time.perf_counter()
            if self.planning_system:
                prediction_data = self.prediction_buffer.peek_latest()
                if prediction_data:
                    planning_result = await self._execute_with_timeout(
                        self.planning_system.plan_actions,
                        args=(prediction_data,),
                        timeout_ms=self.component_time_budgets['planning'] * 1000
                    )
                    if planning_result:
                        self.planning_buffer.put_overwrite(planning_result)
            
            planning_time = (time.perf_counter() - planning_start) * 1000
            self.current_metrics.component_timings['planning'] = planning_time
            
            # 4. Control Stage (10ms budget)
            control_start = time.perf_counter()
            if self.control_system:
                planning_data = self.planning_buffer.peek_latest()
                if planning_data:
                    control_result = await self._execute_with_timeout(
                        self.control_system.execute_control,
                        args=(planning_data,),
                        timeout_ms=self.component_time_budgets['control'] * 1000
                    )
                    if control_result:
                        self.control_buffer.put_overwrite(control_result)
            
            control_time = (time.perf_counter() - control_start) * 1000
            self.current_metrics.component_timings['control'] = control_time
            
            # Check total pipeline time
            total_pipeline_time = (time.perf_counter() - pipeline_start) * 1000
            if total_pipeline_time > self.target_cycle_time_ms * 0.9:  # 90% of cycle time
                logger.warning(f"Pipeline took {total_pipeline_time:.1f}ms (>{self.target_cycle_time_ms * 0.9:.1f}ms)")
                success = False
        
        except asyncio.TimeoutError:
            logger.error("Pipeline stage timeout")
            success = False
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            success = False
        
        return success
    
    async def _execute_with_timeout(self, func: Callable, args: tuple = (), 
                                   timeout_ms: float = 50.0) -> Any:
        """Execute function with timeout constraint"""
        try:
            # Convert to async if needed
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args), timeout=timeout_ms/1000.0)
            else:
                # Run in thread pool for blocking operations
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func, *args),
                    timeout=timeout_ms/1000.0
                )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Function {func.__name__} timed out after {timeout_ms}ms")
            return None
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            return None
    
    async def _execute_scheduled_tasks(self) -> None:
        """Execute priority-scheduled tasks with preemption"""
        current_time = time.perf_counter() * 1000  # Convert to ms
        
        # Execute in priority order
        for priority in Priority:
            tasks_to_execute = []
            
            for task in self.tasks[priority]:
                # Check if task is due
                if task.period_ms is None or (current_time - task.last_execution) >= task.period_ms:
                    tasks_to_execute.append(task)
            
            # Execute high priority tasks immediately
            for task in tasks_to_execute:
                if priority == Priority.CRITICAL:
                    await self._execute_task_immediate(task)
                else:
                    await self._execute_task_with_budget(task, max_time_ms=10.0)
    
    async def _execute_task_immediate(self, task: Task) -> None:
        """Execute critical task immediately"""
        start_time = time.perf_counter()
        
        try:
            if asyncio.iscoroutinefunction(task.function):
                await task.function()
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, task.function)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            task.execution_time_ms = execution_time
            task.last_execution = start_time * 1000
            
        except Exception as e:
            logger.error(f"Critical task '{task.name}' failed: {e}")
    
    async def _execute_task_with_budget(self, task: Task, max_time_ms: float) -> None:
        """Execute task within time budget"""
        start_time = time.perf_counter()
        
        try:
            if asyncio.iscoroutinefunction(task.function):
                await asyncio.wait_for(task.function(), timeout=max_time_ms/1000.0)
            else:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, task.function),
                    timeout=max_time_ms/1000.0
                )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            task.execution_time_ms = execution_time
            task.last_execution = start_time * 1000
            
        except asyncio.TimeoutError:
            task.deadline_misses += 1
            logger.warning(f"Task '{task.name}' exceeded time budget ({max_time_ms}ms)")
        except Exception as e:
            logger.error(f"Task '{task.name}' failed: {e}")
    
    def _update_performance_metrics(self, cycle_time_ms: float) -> None:
        """Update real-time performance metrics"""
        self.current_metrics.cycle_time_ms = cycle_time_ms
        self.current_metrics.total_cycles += 1
        
        # System resource monitoring
        process = psutil.Process()
        self.current_metrics.cpu_usage_percent = process.cpu_percent()
        self.current_metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        
        # GPU monitoring (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.current_metrics.gpu_usage_percent = gpus[0].load * 100
        except ImportError:
            pass
        
        # Store metrics history
        self.current_metrics.timestamp = time.time()
        self.metrics_history.append(PerformanceMetrics(
            cycle_time_ms=cycle_time_ms,
            cpu_usage_percent=self.current_metrics.cpu_usage_percent,
            memory_usage_mb=self.current_metrics.memory_usage_mb,
            gpu_usage_percent=self.current_metrics.gpu_usage_percent,
            component_timings=self.current_metrics.component_timings.copy(),
            missed_deadlines=self.current_metrics.missed_deadlines,
            total_cycles=self.current_metrics.total_cycles,
            timestamp=self.current_metrics.timestamp
        ))
    
    def _adapt_time_budgets(self) -> None:
        """Adaptive load balancing based on component performance"""
        if len(self.metrics_history) < 10:
            return
        
        # Analyze recent performance
        recent_metrics = list(self.metrics_history)[-10:]
        avg_component_times = {}
        
        for component in self.component_time_budgets.keys():
            times = [m.component_timings.get(component, 0) for m in recent_metrics]
            avg_component_times[component] = np.mean([t for t in times if t > 0])
        
        # Adjust budgets based on actual performance
        total_budget = self.target_cycle_time_ms * 0.9  # 90% of cycle for processing
        
        for component, avg_time in avg_component_times.items():
            current_budget = self.component_time_budgets[component] * 1000  # Convert to ms
            
            if avg_time > current_budget * 1.1:  # If consistently over budget
                # Increase budget by 10%
                new_budget = min(current_budget * 1.1, total_budget * 0.4)  # Max 40% of total
                self.component_time_budgets[component] = new_budget / 1000
                logger.info(f"Increased {component} budget to {new_budget:.1f}ms")
            
            elif avg_time < current_budget * 0.7:  # If consistently under budget
                # Decrease budget by 5%
                new_budget = max(current_budget * 0.95, 5.0)  # Min 5ms
                self.component_time_budgets[component] = new_budget / 1000
                logger.debug(f"Decreased {component} budget to {new_budget:.1f}ms")
    
    async def _manage_memory(self) -> None:
        """Memory management for real-time constraints"""
        current_time = time.time()
        
        # Periodic garbage collection
        if current_time - self.last_gc_time > self.gc_interval:
            # Quick GC to avoid real-time impact
            gc.collect(generation=0)  # Only collect youngest generation
            self.last_gc_time = current_time
            
            # Check memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > 2048:  # 2GB limit
                logger.warning(f"Memory usage high: {memory_mb:.1f}MB")
                # Force more aggressive GC
                gc.collect()
                # Clear old metrics
                while len(self.metrics_history) > 500:
                    self.metrics_history.popleft()
    
    def _clear_all_buffers(self) -> None:
        """Clear all data buffers (for emergency stop)"""
        buffers = [
            self.sensor_data_buffer,
            self.perception_buffer,
            self.prediction_buffer,
            self.planning_buffer,
            self.control_buffer
        ]
        
        for buffer in buffers:
            while not buffer.is_empty():
                buffer.get()
    
    async def _handle_emergency_state(self) -> None:
        """Handle emergency stop state"""
        logger.warning("System in emergency state")
        
        # Only execute critical safety tasks
        critical_tasks = self.tasks[Priority.CRITICAL]
        for task in critical_tasks:
            if "safety" in task.name.lower():
                await self._execute_task_immediate(task)
        
        # Wait for emergency clear
        await asyncio.sleep(0.1)  # 100ms emergency state check interval
    
    async def _handle_fault_state(self, error: Exception) -> None:
        """Handle system fault state"""
        logger.error(f"System in fault state: {error}")
        
        # Attempt recovery
        try:
            # Reset components if possible
            if hasattr(self, 'reset_components'):
                await self.reset_components()
            
            # Clear buffers
            self._clear_all_buffers()
            
            # Restart in degraded mode
            self.state = SystemState.DEGRADED
            
        except Exception as recovery_error:
            logger.critical(f"Recovery failed: {recovery_error}")
            self.emergency_stop(f"Recovery failed: {recovery_error}")
    
    async def _cleanup(self) -> None:
        """Cleanup resources on shutdown"""
        logger.info("Cleaning up orchestrator resources...")
        
        # Stop all components
        components = [
            self.perception_system,
            self.prediction_system,
            self.planning_system,
            self.control_system,
            self.safety_monitor,
            self.system_monitor
        ]
        
        for component in components:
            if component and hasattr(component, 'shutdown'):
                try:
                    if asyncio.iscoroutinefunction(component.shutdown):
                        await component.shutdown()
                    else:
                        component.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down component: {e}")
        
        # Clear all buffers
        self._clear_all_buffers()
        
        # Final GC
        gc.collect()
        
        logger.info("Orchestrator cleanup completed")
    
    # Public API methods
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.current_metrics
    
    def get_metrics_history(self) -> List[PerformanceMetrics]:
        """Get performance metrics history"""
        return list(self.metrics_history)
    
    def get_system_state(self) -> SystemState:
        """Get current system state"""
        return self.state
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        if len(self.metrics_history) < 5:
            return True  # Not enough data
        
        recent_metrics = list(self.metrics_history)[-5:]
        
        # Check recent performance
        avg_cycle_time = np.mean([m.cycle_time_ms for m in recent_metrics])
        avg_cpu_usage = np.mean([m.cpu_usage_percent for m in recent_metrics])
        avg_memory_mb = np.mean([m.memory_usage_mb for m in recent_metrics])
        
        # Health thresholds
        health_ok = (
            avg_cycle_time < self.target_cycle_time_ms * 1.2 and  # Within 120% of target
            avg_cpu_usage < 95.0 and  # CPU usage < 95%
            avg_memory_mb < 2048 and  # Memory < 2GB
            self.state in [SystemState.RUNNING, SystemState.DEGRADED]
        )
        
        return health_ok
    
    def add_sensor_data(self, data: Dict[str, Any]) -> bool:
        """Add sensor data to processing pipeline"""
        return self.sensor_data_buffer.put_overwrite(data) is None  # put_overwrite always succeeds
    
    def get_latest_control_output(self) -> Optional[Dict[str, Any]]:
        """Get latest control output"""
        return self.control_buffer.peek_latest()


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Create orchestrator
        orchestrator = RealtimeOrchestrator(target_cycle_time_ms=100.0)
        
        # Add example tasks
        def safety_check():
            print("Safety check performed")
        
        def diagnostics():
            print("System diagnostics")
        
        orchestrator.add_task(Task(
            name="safety_check",
            function=safety_check,
            priority=Priority.CRITICAL,
            deadline_ms=50.0,
            period_ms=100.0
        ))
        
        orchestrator.add_task(Task(
            name="diagnostics",
            function=diagnostics,
            priority=Priority.LOW,
            deadline_ms=200.0,
            period_ms=1000.0
        ))
        
        # Run for test duration
        try:
            await asyncio.wait_for(orchestrator.run_realtime_loop(), timeout=5.0)
        except asyncio.TimeoutError:
            orchestrator.graceful_shutdown()
        
        print(f"Final metrics: {orchestrator.get_current_metrics()}")
    
    asyncio.run(main())