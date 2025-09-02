#!/usr/bin/env python3
"""
Example: Real-Time System Usage
==============================

This example demonstrates how to use the complete real-time system integration
for Model-Based RL Human Intent Recognition with all components working together.

This shows:
1. System initialization with all components
2. Real-time orchestration with timing guarantees  
3. Memory management and optimization
4. Safety system integration
5. Performance monitoring and health checking
6. Graceful shutdown and cleanup
"""

import asyncio
import sys
from pathlib import Path
import logging
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from integration.realtime_orchestrator import RealtimeOrchestrator
from integration.memory_manager import MemoryManager
from integration.performance_optimizer import PerformanceOptimizer
from robustness.safety_system import SafetySystem
from robustness.system_monitor import HealthMonitor


async def simulate_human_intent_recognition_task():
    """Simulate a complete human intent recognition workflow"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing Real-Time Human Intent Recognition System...")
    
    # Initialize all system components
    memory_manager = MemoryManager(max_memory_mb=2048)
    performance_optimizer = PerformanceOptimizer()
    safety_system = SafetySystem()
    health_monitor = HealthMonitor()
    
    # Initialize the main orchestrator
    orchestrator = RealtimeOrchestrator(
        memory_manager=memory_manager,
        safety_system=safety_system,
        health_monitor=health_monitor,
        performance_optimizer=performance_optimizer
    )
    
    try:
        # Initialize the system
        logger.info("Starting system initialization...")
        await orchestrator.initialize()
        logger.info("System initialization completed successfully")
        
        # Start the real-time loop
        logger.info("Starting real-time decision loop...")
        loop_task = asyncio.create_task(orchestrator.run_realtime_loop())
        
        # Simulate various scenarios
        await simulate_normal_operation(orchestrator, logger)
        await simulate_high_load_scenario(orchestrator, logger)
        await simulate_emergency_scenario(orchestrator, safety_system, logger)
        
        # Stop the real-time loop
        logger.info("Stopping real-time loop...")
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
        
        # Display performance metrics
        display_performance_summary(orchestrator, logger)
        
    except Exception as e:
        logger.error(f"System error: {e}")
        
    finally:
        # Graceful shutdown
        logger.info("Shutting down system...")
        await orchestrator.shutdown()
        logger.info("System shutdown completed")


async def simulate_normal_operation(orchestrator, logger):
    """Simulate normal operation for 30 seconds"""
    logger.info("=== Simulating Normal Operation (30 seconds) ===")
    
    start_time = time.time()
    while time.time() - start_time < 30.0:
        # Simulate sensor data processing
        sensor_data = {
            'camera_frame': f"frame_{int(time.time()*1000)}",
            'lidar_points': list(range(1000)),  # Simulated point cloud
            'imu_data': {'accel': [0.1, 0.2, 9.8], 'gyro': [0.01, 0.02, 0.01]}
        }
        
        # Process through the system
        await orchestrator.process_sensor_data(sensor_data)
        
        await asyncio.sleep(0.1)  # 10Hz processing rate
    
    logger.info("Normal operation simulation completed")


async def simulate_high_load_scenario(orchestrator, logger):
    """Simulate high computational load scenario"""
    logger.info("=== Simulating High Load Scenario (20 seconds) ===")
    
    start_time = time.time()
    while time.time() - start_time < 20.0:
        # Generate high-frequency, high-volume data
        tasks = []
        for i in range(5):  # Process 5 concurrent streams
            sensor_data = {
                'camera_frame': f"high_load_frame_{i}_{int(time.time()*1000)}",
                'lidar_points': list(range(5000)),  # Larger point cloud
                'complex_computation': True
            }
            
            task = asyncio.create_task(orchestrator.process_sensor_data(sensor_data))
            tasks.append(task)
        
        # Wait for all concurrent tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        await asyncio.sleep(0.05)  # 20Hz processing rate
    
    logger.info("High load scenario simulation completed")


async def simulate_emergency_scenario(orchestrator, safety_system, logger):
    """Simulate emergency detection and response"""
    logger.info("=== Simulating Emergency Scenario ===")
    
    # Simulate emergency condition detection
    emergency_conditions = [
        "Human detected in safety zone",
        "Sensor failure detected",  
        "System overheating detected"
    ]
    
    for condition in emergency_conditions:
        logger.info(f"Simulating emergency: {condition}")
        
        # Trigger emergency stop
        start_time = time.perf_counter()
        await safety_system.emergency_stop.trigger_emergency_stop(condition)
        response_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Emergency response time: {response_time:.2f}ms")
        
        # Wait for system to stabilize
        await asyncio.sleep(2.0)
        
        # Reset emergency condition
        await safety_system.emergency_stop.reset_emergency_stop()
        logger.info(f"Emergency condition '{condition}' resolved")
        
        await asyncio.sleep(1.0)  # Brief pause between emergencies
    
    logger.info("Emergency scenario simulation completed")


def display_performance_summary(orchestrator, logger):
    """Display comprehensive performance summary"""
    logger.info("=== PERFORMANCE SUMMARY ===")
    
    # Get performance metrics
    metrics = orchestrator.get_performance_metrics()
    
    # Timing metrics
    if 'cycle_times' in metrics and metrics['cycle_times']:
        cycle_times = metrics['cycle_times']
        avg_cycle = sum(cycle_times) / len(cycle_times)
        max_cycle = max(cycle_times)
        p95_cycle = sorted(cycle_times)[int(0.95 * len(cycle_times))]
        
        logger.info(f"Decision Cycle Timing:")
        logger.info(f"  Average: {avg_cycle:.1f}ms")
        logger.info(f"  Maximum: {max_cycle:.1f}ms") 
        logger.info(f"  P95: {p95_cycle:.1f}ms")
        logger.info(f"  Requirement: < 100ms {'✓ PASSED' if max_cycle < 100 else '✗ FAILED'}")
    
    # Memory metrics
    memory_stats = orchestrator.memory_manager.get_memory_stats()
    logger.info(f"Memory Usage:")
    logger.info(f"  Current: {memory_stats.get('current_usage_mb', 0):.1f}MB")
    logger.info(f"  Peak: {memory_stats.get('peak_usage_mb', 0):.1f}MB")
    logger.info(f"  Requirement: < 2048MB {'✓ PASSED' if memory_stats.get('peak_usage_mb', 0) < 2048 else '✗ FAILED'}")
    
    # Safety metrics
    safety_stats = orchestrator.safety_system.get_safety_status()
    logger.info(f"Safety System:")
    logger.info(f"  Emergency stops: {safety_stats.get('total_emergency_stops', 0)}")
    logger.info(f"  Average response time: {safety_stats.get('avg_response_time_ms', 0):.1f}ms")
    logger.info(f"  Requirement: < 10ms {'✓ PASSED' if safety_stats.get('avg_response_time_ms', 0) < 10 else '✗ FAILED'}")
    
    # Health monitoring
    health_stats = orchestrator.health_monitor.get_health_summary()
    logger.info(f"System Health:")
    logger.info(f"  Overall status: {health_stats.get('overall_status', 'unknown')}")
    logger.info(f"  Anomalies detected: {health_stats.get('anomaly_count', 0)}")
    logger.info(f"  Uptime: {health_stats.get('uptime_seconds', 0):.1f}s")
    
    logger.info("=== END PERFORMANCE SUMMARY ===")


async def run_quick_demo():
    """Run a quick 2-minute demonstration"""
    print("Starting Quick Real-Time System Demo (2 minutes)...")
    print("This demonstrates real-time orchestration, memory management, safety, and monitoring.")
    print("=" * 70)
    
    await simulate_human_intent_recognition_task()
    
    print("=" * 70)
    print("Demo completed! Check the logs above for detailed performance metrics.")
    print("For comprehensive testing, run: python run_performance_tests.py --all")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick demo mode
        asyncio.run(run_quick_demo())
    else:
        # Full demonstration
        asyncio.run(simulate_human_intent_recognition_task())