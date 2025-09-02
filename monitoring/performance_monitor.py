#!/usr/bin/env python3
"""
Real-time Performance Monitor
Model-Based RL Human Intent Recognition System

Continuous monitoring of decision cycle performance with alerting
for violations of <10ms target performance.

Author: Monitoring Team
Date: September 2025
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import queue
import statistics

# Try imports with graceful fallbacks
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

class PerformanceMonitor:
    """
    Real-time performance monitoring system for production validation
    of <10ms decision cycle requirements.
    """
    
    def __init__(self, target_ms: float = 10.0, alert_threshold: float = 12.0):
        """
        Initialize performance monitor.
        
        Args:
            target_ms: Target performance threshold
            alert_threshold: Threshold for triggering alerts
        """
        self.target_ms = target_ms
        self.alert_threshold = alert_threshold
        self.monitoring_active = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/monitoring/logs/performance_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics storage
        self.metrics_queue = queue.Queue(maxsize=10000)
        self.recent_metrics = []
        self.alerts = []
        
        # Prometheus metrics (if available)
        if HAS_PROMETHEUS:
            self.decision_time_histogram = Histogram(
                'decision_cycle_seconds', 
                'Time spent in decision cycles'
            )
            self.alert_counter = Counter(
                'performance_alerts_total',
                'Total performance alerts triggered'
            )
            self.compliance_gauge = Gauge(
                'target_compliance_ratio',
                'Ratio of cycles meeting performance target'
            )
        
        self.logger.info(f"Performance monitor initialized (target: {target_ms}ms)")

    def start_monitoring(self, port: int = 8051):
        """Start the monitoring service."""
        self.monitoring_active = True
        
        # Start Prometheus metrics server if available
        if HAS_PROMETHEUS:
            try:
                start_http_server(port)
                self.logger.info(f"Prometheus metrics available on port {port}")
            except Exception as e:
                self.logger.warning(f"Could not start Prometheus server: {e}")
        
        # Start monitoring threads
        threading.Thread(target=self._metrics_processor, daemon=True).start()
        threading.Thread(target=self._system_monitor, daemon=True).start()
        threading.Thread(target=self._alert_processor, daemon=True).start()
        
        self.logger.info("Performance monitoring started")
        
        try:
            self._run_monitor_loop()
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        finally:
            self.stop_monitoring()

    def stop_monitoring(self):
        """Stop the monitoring service."""
        self.monitoring_active = False
        self.logger.info("Performance monitoring stopped")

    def record_metric(self, decision_time_ms: float, metadata: Dict[str, Any] = None):
        """
        Record a performance metric.
        
        Args:
            decision_time_ms: Decision cycle time in milliseconds
            metadata: Additional context information
        """
        metric = {
            'timestamp': datetime.now().isoformat(),
            'decision_time_ms': decision_time_ms,
            'metadata': metadata or {}
        }
        
        try:
            self.metrics_queue.put_nowait(metric)
        except queue.Full:
            self.logger.warning("Metrics queue full, dropping metric")
        
        # Update Prometheus metrics
        if HAS_PROMETHEUS:
            self.decision_time_histogram.observe(decision_time_ms / 1000.0)

    def _metrics_processor(self):
        """Process incoming metrics and maintain rolling statistics."""
        while self.monitoring_active:
            try:
                # Process metrics batch
                batch = []
                while len(batch) < 100:  # Process in batches of 100
                    try:
                        metric = self.metrics_queue.get(timeout=1.0)
                        batch.append(metric)
                    except queue.Empty:
                        break
                
                if batch:
                    self._process_metrics_batch(batch)
                    
            except Exception as e:
                self.logger.error(f"Metrics processing error: {e}")
                time.sleep(1)

    def _process_metrics_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of metrics."""
        # Add to recent metrics (maintain rolling window)
        self.recent_metrics.extend(batch)
        self.recent_metrics = self.recent_metrics[-1000:]  # Keep last 1000 metrics
        
        # Check for performance violations
        for metric in batch:
            decision_time = metric['decision_time_ms']
            if decision_time > self.alert_threshold:
                self._trigger_alert('PERFORMANCE_VIOLATION', {
                    'decision_time_ms': decision_time,
                    'threshold_ms': self.alert_threshold,
                    'timestamp': metric['timestamp']
                })
        
        # Update compliance metrics
        if self.recent_metrics:
            compliant_count = sum(1 for m in self.recent_metrics 
                                if m['decision_time_ms'] < self.target_ms)
            compliance_ratio = compliant_count / len(self.recent_metrics)
            
            if HAS_PROMETHEUS:
                self.compliance_gauge.set(compliance_ratio)

    def _system_monitor(self):
        """Monitor system resources."""
        if not HAS_PSUTIL:
            self.logger.warning("System monitoring disabled - psutil not available")
            return
        
        while self.monitoring_active:
            try:
                # CPU and memory monitoring
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                if cpu_percent > 80:
                    self._trigger_alert('HIGH_CPU', {
                        'cpu_percent': cpu_percent,
                        'threshold': 80
                    })
                
                if memory.percent > 85:
                    self._trigger_alert('HIGH_MEMORY', {
                        'memory_percent': memory.percent,
                        'threshold': 85
                    })
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                time.sleep(5)

    def _alert_processor(self):
        """Process and handle alerts."""
        while self.monitoring_active:
            try:
                # Check for sustained performance issues
                if len(self.recent_metrics) >= 100:
                    recent_times = [m['decision_time_ms'] for m in self.recent_metrics[-100:]]
                    avg_time = statistics.mean(recent_times)
                    
                    if avg_time > self.target_ms:
                        violations = sum(1 for t in recent_times if t > self.target_ms)
                        violation_rate = violations / len(recent_times)
                        
                        if violation_rate > 0.1:  # More than 10% violations
                            self._trigger_alert('SUSTAINED_PERFORMANCE_DEGRADATION', {
                                'avg_time_ms': round(avg_time, 2),
                                'violation_rate': round(violation_rate * 100, 1),
                                'sample_size': len(recent_times)
                            })
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                time.sleep(30)

    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger a performance alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'details': details,
            'severity': self._get_alert_severity(alert_type, details)
        }
        
        self.alerts.append(alert)
        self.alerts = self.alerts[-100:]  # Keep last 100 alerts
        
        # Log alert
        severity = alert['severity']
        self.logger.warning(f"ALERT [{severity}] {alert_type}: {details}")
        
        # Update Prometheus counter
        if HAS_PROMETHEUS:
            self.alert_counter.inc()
        
        # Save alert to file
        self._save_alert(alert)

    def _get_alert_severity(self, alert_type: str, details: Dict[str, Any]) -> str:
        """Determine alert severity."""
        if alert_type == 'PERFORMANCE_VIOLATION':
            if details.get('decision_time_ms', 0) > self.target_ms * 2:
                return 'CRITICAL'
            else:
                return 'WARNING'
        elif alert_type in ['HIGH_CPU', 'HIGH_MEMORY']:
            return 'WARNING'
        elif alert_type == 'SUSTAINED_PERFORMANCE_DEGRADATION':
            return 'CRITICAL'
        else:
            return 'INFO'

    def _save_alert(self, alert: Dict[str, Any]):
        """Save alert to file."""
        alert_file = Path('/monitoring/alerts/alerts.jsonl')
        alert_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with alert_file.open('a') as f:
                json.dump(alert, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to save alert: {e}")

    def _run_monitor_loop(self):
        """Main monitoring loop with periodic reporting."""
        report_interval = 60  # Report every minute
        last_report = time.time()
        
        while self.monitoring_active:
            current_time = time.time()
            
            if current_time - last_report >= report_interval:
                self._generate_periodic_report()
                last_report = current_time
            
            time.sleep(1)

    def _generate_periodic_report(self):
        """Generate and log periodic performance report."""
        if not self.recent_metrics:
            return
        
        recent_times = [m['decision_time_ms'] for m in self.recent_metrics[-100:]]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics_count': len(recent_times),
            'avg_ms': round(statistics.mean(recent_times), 2),
            'p95_ms': round(sorted(recent_times)[int(len(recent_times) * 0.95)], 2),
            'compliance_rate': f"{(sum(1 for t in recent_times if t < self.target_ms)/len(recent_times))*100:.1f}%",
            'recent_alerts': len([a for a in self.alerts if 
                                datetime.fromisoformat(a['timestamp']) > 
                                datetime.now() - timedelta(minutes=5)])
        }
        
        self.logger.info(f"Performance Report: {report['avg_ms']}ms avg, "
                        f"{report['compliance_rate']} compliant")
        
        # Save report
        report_file = Path('/monitoring/reports/performance_reports.jsonl')
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with report_file.open('a') as f:
                json.dump(report, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        if not self.recent_metrics:
            return {'status': 'No metrics available'}
        
        recent_times = [m['decision_time_ms'] for m in self.recent_metrics[-100:]]
        
        return {
            'monitoring_active': self.monitoring_active,
            'target_ms': self.target_ms,
            'recent_metrics_count': len(recent_times),
            'avg_performance_ms': round(statistics.mean(recent_times), 2),
            'compliance_rate': f"{(sum(1 for t in recent_times if t < self.target_ms)/len(recent_times))*100:.1f}%",
            'recent_alerts_count': len([a for a in self.alerts if 
                                     datetime.fromisoformat(a['timestamp']) > 
                                     datetime.now() - timedelta(minutes=10)]),
            'system_resources': self._get_system_status() if HAS_PSUTIL else 'unavailable'
        }

    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system resource status."""
        if not HAS_PSUTIL:
            return {'status': 'monitoring unavailable'}
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }


def main():
    """Main monitoring service entry point."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Performance Monitor Service')
    parser.add_argument('--target', type=float, default=10.0,
                       help='Performance target in milliseconds')
    parser.add_argument('--alert-threshold', type=float, default=12.0,
                       help='Alert threshold in milliseconds')
    parser.add_argument('--port', type=int, default=8051,
                       help='Monitoring service port')
    
    args = parser.parse_args()
    
    # Create monitoring directories
    Path('/monitoring/logs').mkdir(parents=True, exist_ok=True)
    Path('/monitoring/alerts').mkdir(parents=True, exist_ok=True)
    Path('/monitoring/reports').mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Real-time Performance Monitor")
    print("Model-Based RL Human Intent Recognition")
    print("=" * 60)
    print(f"Performance Target: <{args.target}ms")
    print(f"Alert Threshold: {args.alert_threshold}ms")
    print(f"Monitoring Port: {args.port}")
    print("=" * 60)
    
    monitor = PerformanceMonitor(
        target_ms=args.target,
        alert_threshold=args.alert_threshold
    )
    
    try:
        monitor.start_monitoring(port=args.port)
    except KeyboardInterrupt:
        print("\nShutting down monitor...")
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()