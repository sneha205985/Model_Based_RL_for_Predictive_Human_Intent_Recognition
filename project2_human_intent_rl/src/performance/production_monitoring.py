"""
Production Performance Monitoring and Alerting System
Model-Based RL Human Intent Recognition System

This module provides real-time performance monitoring and alerting for
production deployment with automated anomaly detection and performance
regression alerts to maintain <10ms decision cycles and >95% safety rate.

Monitoring Capabilities:
1. Real-time latency monitoring with statistical process control
2. Safety performance tracking with automatic violation detection
3. Resource utilization monitoring and alerting
4. Performance regression detection using statistical models
5. Automated alerting and escalation system

Mathematical Foundation:
- Statistical Process Control (SPC) with control charts
- Exponentially Weighted Moving Averages (EWMA) for trend detection
- Anomaly detection using z-scores and percentile thresholds
- Performance regression analysis with significance testing

Author: Production Performance Monitoring System
"""

import numpy as np
import pandas as pd
import time
import threading
import queue
import json
import logging
import psutil
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from scipy import stats
from abc import ABC, abstractmethod
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for production performance monitoring"""
    # Performance thresholds
    latency_threshold_ms: float = 10.0
    latency_p95_threshold_ms: float = 15.0
    safety_rate_threshold: float = 0.95
    memory_threshold_mb: float = 500.0
    cpu_threshold_percent: float = 80.0
    
    # Statistical process control parameters
    spc_window_size: int = 100
    spc_sigma_level: float = 3.0  # 3-sigma control limits
    ewma_alpha: float = 0.2  # EWMA smoothing parameter
    
    # Anomaly detection parameters
    anomaly_z_threshold: float = 3.0
    anomaly_percentile_threshold: float = 99.0
    min_samples_for_detection: int = 30
    
    # Alerting configuration
    alert_cooldown_minutes: int = 5
    escalation_threshold: int = 3  # Number of alerts before escalation
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
    
    # Data retention
    metric_history_hours: int = 24
    detailed_history_hours: int = 1
    
    # Monitoring intervals
    sampling_interval_s: float = 0.1
    analysis_interval_s: float = 5.0
    reporting_interval_s: float = 60.0


class MetricCollector:
    """
    Real-time metric collection with efficient data structures.
    
    Collects performance metrics from system components with
    minimal overhead and automatic data rotation.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Thread-safe data structures
        self.metrics_queue = queue.Queue(maxsize=10000)
        self.current_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.metrics_lock = threading.Lock()
        
        # Collection state
        self.collecting = False
        self.collection_thread = None
        self.last_cleanup = time.time()
        
        # Metric statistics
        self.metric_stats = defaultdict(dict)
        
    def start_collection(self):
        """Start metric collection thread"""
        if self.collecting:
            return
            
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        logger.info("📊 Performance metric collection started")
    
    def stop_collection(self):
        """Stop metric collection"""
        if not self.collecting:
            return
            
        self.collecting = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        
        logger.info("📊 Performance metric collection stopped")
    
    def record_metric(self, metric_name: str, value: float, 
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        try:
            metric_data = {
                'name': metric_name,
                'value': value,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            # Non-blocking queue put
            try:
                self.metrics_queue.put_nowait(metric_data)
            except queue.Full:
                # Drop oldest metrics if queue is full
                try:
                    self.metrics_queue.get_nowait()
                    self.metrics_queue.put_nowait(metric_data)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.warning(f"Failed to record metric {metric_name}: {e}")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.collecting:
            try:
                # Process queued metrics
                processed_count = 0
                while not self.metrics_queue.empty() and processed_count < 100:
                    try:
                        metric_data = self.metrics_queue.get_nowait()
                        self._process_metric(metric_data)
                        processed_count += 1
                    except queue.Empty:
                        break
                
                # Periodic cleanup
                if time.time() - self.last_cleanup > 300:  # 5 minutes
                    self._cleanup_old_metrics()
                    self.last_cleanup = time.time()
                
                # Brief sleep to prevent high CPU usage
                time.sleep(self.config.sampling_interval_s)
                
            except Exception as e:
                logger.warning(f"Metric collection error: {e}")
    
    def _process_metric(self, metric_data: Dict[str, Any]):
        """Process a single metric"""
        with self.metrics_lock:
            metric_name = metric_data['name']
            value = metric_data['value']
            timestamp = metric_data['timestamp']
            
            # Store in current metrics
            self.current_metrics[metric_name].append({
                'value': value,
                'timestamp': timestamp,
                'metadata': metric_data.get('metadata', {})
            })
            
            # Update statistics
            self._update_metric_stats(metric_name, value)
    
    def _update_metric_stats(self, metric_name: str, value: float):
        """Update running statistics for a metric"""
        stats = self.metric_stats[metric_name]
        
        if 'count' not in stats:
            stats.update({
                'count': 0,
                'sum': 0.0,
                'sum_sq': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'ewma': value,
                'ewma_var': 0.0
            })
        
        # Update basic statistics
        stats['count'] += 1
        stats['sum'] += value
        stats['sum_sq'] += value * value
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)
        
        # Update EWMA and variance
        alpha = self.config.ewma_alpha
        if stats['count'] == 1:
            stats['ewma'] = value
            stats['ewma_var'] = 0.0
        else:
            diff = value - stats['ewma']
            stats['ewma'] += alpha * diff
            stats['ewma_var'] = (1 - alpha) * (stats['ewma_var'] + alpha * diff * diff)
    
    def _cleanup_old_metrics(self):
        """Remove old metrics to manage memory usage"""
        cutoff_time = time.time() - (self.config.metric_history_hours * 3600)
        
        with self.metrics_lock:
            for metric_name in list(self.current_metrics.keys()):
                metric_data = self.current_metrics[metric_name]
                
                # Remove old entries
                while metric_data and metric_data[0]['timestamp'] < cutoff_time:
                    metric_data.popleft()
                
                # Remove empty metrics
                if not metric_data:
                    del self.current_metrics[metric_name]
                    if metric_name in self.metric_stats:
                        del self.metric_stats[metric_name]
    
    def get_metric_summary(self, metric_name: str, 
                          time_window_s: Optional[float] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        with self.metrics_lock:
            if metric_name not in self.current_metrics:
                return {'error': 'Metric not found'}
            
            metric_data = self.current_metrics[metric_name]
            if not metric_data:
                return {'error': 'No data available'}
            
            # Filter by time window if specified
            if time_window_s:
                cutoff_time = time.time() - time_window_s
                filtered_data = [d for d in metric_data if d['timestamp'] >= cutoff_time]
            else:
                filtered_data = list(metric_data)
            
            if not filtered_data:
                return {'error': 'No data in time window'}
            
            # Extract values
            values = [d['value'] for d in filtered_data]
            
            # Compute summary statistics
            return {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
                'latest': values[-1] if values else None,
                'time_span_s': filtered_data[-1]['timestamp'] - filtered_data[0]['timestamp'],
                'ewma': self.metric_stats[metric_name].get('ewma', np.mean(values)),
                'ewma_std': np.sqrt(self.metric_stats[metric_name].get('ewma_var', np.var(values)))
            }


class AnomalyDetector:
    """
    Statistical anomaly detection for performance monitoring.
    
    Uses multiple detection methods including z-score analysis,
    percentile thresholds, and statistical process control.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metric_baselines = {}
        self.anomaly_history = defaultdict(list)
    
    def detect_anomalies(self, metric_name: str, 
                        metric_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in metric data using multiple methods.
        
        Returns anomaly detection results with severity levels.
        """
        if 'error' in metric_summary or metric_summary['count'] < self.config.min_samples_for_detection:
            return {'anomalies_detected': False, 'reason': 'insufficient_data'}
        
        anomalies = []
        current_value = metric_summary['latest']
        
        # Method 1: Z-score based detection
        z_score_anomaly = self._detect_z_score_anomaly(metric_name, metric_summary)
        if z_score_anomaly:
            anomalies.append(z_score_anomaly)
        
        # Method 2: Percentile threshold detection
        percentile_anomaly = self._detect_percentile_anomaly(metric_name, metric_summary)
        if percentile_anomaly:
            anomalies.append(percentile_anomaly)
        
        # Method 3: Statistical Process Control
        spc_anomaly = self._detect_spc_anomaly(metric_name, metric_summary)
        if spc_anomaly:
            anomalies.append(spc_anomaly)
        
        # Method 4: EWMA trend detection
        trend_anomaly = self._detect_trend_anomaly(metric_name, metric_summary)
        if trend_anomaly:
            anomalies.append(trend_anomaly)
        
        # Update baseline if no anomalies detected
        if not anomalies:
            self._update_baseline(metric_name, metric_summary)
        
        # Record anomaly history
        if anomalies:
            self.anomaly_history[metric_name].append({
                'timestamp': time.time(),
                'anomalies': anomalies,
                'metric_value': current_value
            })
        
        return {
            'anomalies_detected': bool(anomalies),
            'anomalies': anomalies,
            'severity': self._determine_severity(anomalies),
            'current_value': current_value,
            'baseline_mean': self.metric_baselines.get(metric_name, {}).get('mean'),
            'baseline_std': self.metric_baselines.get(metric_name, {}).get('std')
        }
    
    def _detect_z_score_anomaly(self, metric_name: str, 
                               metric_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomaly using z-score analysis"""
        if metric_name not in self.metric_baselines:
            return None
        
        baseline = self.metric_baselines[metric_name]
        current_value = metric_summary['latest']
        baseline_mean = baseline['mean']
        baseline_std = baseline['std']
        
        if baseline_std == 0:
            return None
        
        z_score = abs(current_value - baseline_mean) / baseline_std
        
        if z_score > self.config.anomaly_z_threshold:
            return {
                'method': 'z_score',
                'z_score': float(z_score),
                'threshold': self.config.anomaly_z_threshold,
                'description': f'Value {current_value:.3f} is {z_score:.2f} standard deviations from baseline mean {baseline_mean:.3f}'
            }
        
        return None
    
    def _detect_percentile_anomaly(self, metric_name: str,
                                  metric_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomaly using percentile thresholds"""
        if metric_name not in self.metric_baselines:
            return None
        
        baseline = self.metric_baselines[metric_name]
        current_value = metric_summary['latest']
        
        # Check if current value exceeds historical percentile
        percentile_threshold = baseline.get('percentile_99', float('inf'))
        
        if current_value > percentile_threshold:
            return {
                'method': 'percentile',
                'percentile_threshold': float(percentile_threshold),
                'current_value': float(current_value),
                'description': f'Value {current_value:.3f} exceeds 99th percentile threshold {percentile_threshold:.3f}'
            }
        
        return None
    
    def _detect_spc_anomaly(self, metric_name: str,
                           metric_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomaly using Statistical Process Control"""
        if metric_name not in self.metric_baselines:
            return None
        
        baseline = self.metric_baselines[metric_name]
        current_value = metric_summary['latest']
        
        # Control limits (3-sigma)
        center_line = baseline['mean']
        std = baseline['std']
        ucl = center_line + self.config.spc_sigma_level * std  # Upper Control Limit
        lcl = center_line - self.config.spc_sigma_level * std  # Lower Control Limit
        
        if current_value > ucl:
            return {
                'method': 'spc',
                'control_limit': 'upper',
                'ucl': float(ucl),
                'current_value': float(current_value),
                'description': f'Value {current_value:.3f} exceeds upper control limit {ucl:.3f}'
            }
        elif current_value < lcl:
            return {
                'method': 'spc',
                'control_limit': 'lower',
                'lcl': float(lcl),
                'current_value': float(current_value),
                'description': f'Value {current_value:.3f} below lower control limit {lcl:.3f}'
            }
        
        return None
    
    def _detect_trend_anomaly(self, metric_name: str,
                             metric_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomaly in EWMA trend"""
        current_ewma = metric_summary.get('ewma')
        current_ewma_std = metric_summary.get('ewma_std')
        
        if current_ewma is None or current_ewma_std is None:
            return None
        
        if metric_name not in self.metric_baselines:
            return None
        
        baseline = self.metric_baselines[metric_name]
        baseline_ewma = baseline.get('ewma')
        baseline_ewma_std = baseline.get('ewma_std')
        
        if baseline_ewma is None or baseline_ewma_std is None or baseline_ewma_std == 0:
            return None
        
        # Detect significant deviation in EWMA
        ewma_z_score = abs(current_ewma - baseline_ewma) / baseline_ewma_std
        
        if ewma_z_score > 2.0:  # Lower threshold for trend detection
            return {
                'method': 'ewma_trend',
                'ewma_z_score': float(ewma_z_score),
                'current_ewma': float(current_ewma),
                'baseline_ewma': float(baseline_ewma),
                'description': f'EWMA trend deviation: {ewma_z_score:.2f} sigma from baseline'
            }
        
        return None
    
    def _update_baseline(self, metric_name: str, metric_summary: Dict[str, Any]):
        """Update baseline statistics for metric"""
        self.metric_baselines[metric_name] = {
            'mean': metric_summary['mean'],
            'std': metric_summary['std'],
            'median': metric_summary['median'],
            'p95': metric_summary['p95'],
            'percentile_99': metric_summary['p99'],
            'ewma': metric_summary.get('ewma'),
            'ewma_std': metric_summary.get('ewma_std'),
            'last_updated': time.time(),
            'sample_count': metric_summary['count']
        }
    
    def _determine_severity(self, anomalies: List[Dict[str, Any]]) -> str:
        """Determine overall severity level from detected anomalies"""
        if not anomalies:
            return 'none'
        
        # Count by method
        method_counts = defaultdict(int)
        max_z_score = 0
        
        for anomaly in anomalies:
            method_counts[anomaly['method']] += 1
            if 'z_score' in anomaly:
                max_z_score = max(max_z_score, anomaly['z_score'])
            elif 'ewma_z_score' in anomaly:
                max_z_score = max(max_z_score, anomaly['ewma_z_score'])
        
        # Determine severity
        if len(anomalies) >= 3 or max_z_score > 5.0:
            return 'critical'
        elif len(anomalies) >= 2 or max_z_score > 4.0:
            return 'high'
        elif max_z_score > 3.0:
            return 'medium'
        else:
            return 'low'


class AlertManager:
    """
    Alert management system with configurable thresholds and escalation.
    
    Manages alert generation, deduplication, escalation, and notification
    across multiple channels (email, Slack, logs).
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Alert state tracking
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_counts = defaultdict(int)
        self.last_alert_times = {}
        
        # Escalation tracking
        self.escalation_counts = defaultdict(int)
        
    def process_alert(self, alert_type: str, severity: str, 
                     message: str, details: Dict[str, Any]) -> bool:
        """
        Process and potentially send an alert.
        
        Returns True if alert was sent, False if suppressed.
        """
        alert_key = f"{alert_type}_{severity}"
        current_time = time.time()
        
        # Check cooldown period
        last_alert_time = self.last_alert_times.get(alert_key, 0)
        cooldown_seconds = self.config.alert_cooldown_minutes * 60
        
        if current_time - last_alert_time < cooldown_seconds:
            return False  # Alert suppressed due to cooldown
        
        # Create alert
        alert = {
            'timestamp': current_time,
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'details': details,
            'alert_key': alert_key
        }
        
        # Update tracking
        self.alert_counts[alert_key] += 1
        self.last_alert_times[alert_key] = current_time
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
        # Check for escalation
        if self.alert_counts[alert_key] >= self.config.escalation_threshold:
            alert['escalated'] = True
            self.escalation_counts[alert_key] += 1
        
        # Send alert
        self._send_alert(alert)
        
        return True
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels"""
        
        # Always log the alert
        severity = alert['severity'].upper()
        message = alert['message']
        logger.error(f"🚨 ALERT [{severity}]: {message}")
        
        # Format detailed message
        details_str = self._format_alert_details(alert['details'])
        full_message = f"{message}\n\nDetails:\n{details_str}"
        
        # Send email alert (if configured)
        if self.config.enable_email_alerts:
            try:
                self._send_email_alert(alert, full_message)
            except Exception as e:
                logger.warning(f"Failed to send email alert: {e}")
        
        # Send Slack alert (if configured)
        if self.config.enable_slack_alerts:
            try:
                self._send_slack_alert(alert, full_message)
            except Exception as e:
                logger.warning(f"Failed to send Slack alert: {e}")
    
    def _format_alert_details(self, details: Dict[str, Any]) -> str:
        """Format alert details for human readability"""
        formatted_lines = []
        
        for key, value in details.items():
            if isinstance(value, float):
                formatted_lines.append(f"  {key}: {value:.3f}")
            elif isinstance(value, dict):
                formatted_lines.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        formatted_lines.append(f"    {sub_key}: {sub_value:.3f}")
                    else:
                        formatted_lines.append(f"    {sub_key}: {sub_value}")
            else:
                formatted_lines.append(f"  {key}: {value}")
        
        return "\n".join(formatted_lines)
    
    def _send_email_alert(self, alert: Dict[str, Any], message: str):
        """Send email alert (placeholder implementation)"""
        # This would integrate with actual email service
        logger.info(f"📧 Email alert would be sent: {alert['alert_type']}")
    
    def _send_slack_alert(self, alert: Dict[str, Any], message: str):
        """Send Slack alert (placeholder implementation)"""
        # This would integrate with Slack API
        logger.info(f"💬 Slack alert would be sent: {alert['alert_type']}")
    
    def clear_alert(self, alert_key: str):
        """Clear an active alert"""
        if alert_key in self.active_alerts:
            del self.active_alerts[alert_key]
            logger.info(f"✅ Alert cleared: {alert_key}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alert status"""
        current_time = time.time()
        
        # Count active alerts by severity
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert['severity']] += 1
        
        # Recent alerts (last hour)
        recent_alerts = [
            alert for alert in self.alert_history
            if current_time - alert['timestamp'] < 3600
        ]
        
        return {
            'active_alerts': len(self.active_alerts),
            'severity_breakdown': dict(severity_counts),
            'recent_alerts_1h': len(recent_alerts),
            'total_alerts_today': len([a for a in self.alert_history 
                                     if current_time - a['timestamp'] < 86400]),
            'escalated_alerts': len([a for a in self.active_alerts.values() 
                                   if a.get('escalated', False)])
        }


class PerformanceMonitor:
    """
    Main production performance monitoring system.
    
    Orchestrates metric collection, anomaly detection, and alerting
    with comprehensive performance analysis and reporting.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """Initialize production performance monitor"""
        self.config = config or MonitoringConfig()
        
        # Initialize subsystems
        self.metric_collector = MetricCollector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Monitoring state
        self.monitoring_active = False
        self.analysis_thread = None
        self.system_monitor_thread = None
        
        # Performance tracking
        self.performance_dashboard = {
            'system_status': 'unknown',
            'last_update': time.time(),
            'key_metrics': {},
            'alerts': {},
            'trends': {}
        }
        
        logger.info("🔍 Production Performance Monitor initialized")
    
    def start_monitoring(self, system_components: Optional[Dict[str, Any]] = None):
        """Start comprehensive performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.system_components = system_components or {}
        
        # Start metric collection
        self.metric_collector.start_collection()
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._monitoring_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        
        # Start system resource monitoring
        self.system_monitor_thread = threading.Thread(target=self._system_monitoring_loop)
        self.system_monitor_thread.daemon = True
        self.system_monitor_thread.start()
        
        logger.info("🚀 Production performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        # Stop metric collection
        self.metric_collector.stop_collection()
        
        # Wait for threads to stop
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=10.0)
        
        if self.system_monitor_thread and self.system_monitor_thread.is_alive():
            self.system_monitor_thread.join(timeout=5.0)
        
        logger.info("🛑 Production performance monitoring stopped")
    
    def record_decision_cycle(self, component_name: str, latency_ms: float, 
                             success: bool, metadata: Optional[Dict[str, Any]] = None):
        """Record a decision cycle performance metric"""
        self.metric_collector.record_metric(
            f"decision_cycle_latency_{component_name}",
            latency_ms,
            {**(metadata or {}), 'success': success, 'component': component_name}
        )
        
        # Record success rate
        self.metric_collector.record_metric(
            f"decision_success_{component_name}",
            1.0 if success else 0.0,
            {**(metadata or {}), 'component': component_name}
        )
    
    def record_safety_event(self, event_type: str, is_safe: bool, 
                           distance_to_human: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Record a safety-related event"""
        self.metric_collector.record_metric(
            f"safety_event_{event_type}",
            1.0 if is_safe else 0.0,
            {
                **(metadata or {}),
                'event_type': event_type,
                'distance_to_human': distance_to_human,
                'safe': is_safe
            }
        )
        
        # Record minimum distance if provided
        if distance_to_human is not None:
            self.metric_collector.record_metric(
                "min_distance_to_human",
                distance_to_human,
                {**(metadata or {}), 'event_type': event_type}
            )
    
    def _monitoring_loop(self):
        """Main monitoring and analysis loop"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Analyze all collected metrics
                self._analyze_performance_metrics()
                
                # Update dashboard
                self._update_performance_dashboard()
                
                # Calculate sleep time to maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.analysis_interval_s - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.analysis_interval_s)
    
    def _system_monitoring_loop(self):
        """System resource monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.metric_collector.record_metric("system_cpu_percent", cpu_percent)
                
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.metric_collector.record_metric("system_memory_mb", memory_mb)
                
                # System-wide metrics
                system_cpu = psutil.cpu_percent()
                system_memory = psutil.virtual_memory()
                
                self.metric_collector.record_metric("host_cpu_percent", system_cpu)
                self.metric_collector.record_metric("host_memory_percent", system_memory.percent)
                
                time.sleep(self.config.sampling_interval_s * 10)  # Less frequent system monitoring
                
            except Exception as e:
                logger.warning(f"System monitoring error: {e}")
                time.sleep(5.0)
    
    def _analyze_performance_metrics(self):
        """Analyze all performance metrics for anomalies"""
        
        # Get all metric names
        with self.metric_collector.metrics_lock:
            metric_names = list(self.metric_collector.current_metrics.keys())
        
        for metric_name in metric_names:
            try:
                # Get metric summary
                summary = self.metric_collector.get_metric_summary(
                    metric_name, 
                    time_window_s=300  # 5 minutes
                )
                
                if 'error' in summary:
                    continue
                
                # Detect anomalies
                anomaly_result = self.anomaly_detector.detect_anomalies(metric_name, summary)
                
                # Process alerts if anomalies detected
                if anomaly_result['anomalies_detected']:
                    self._handle_performance_anomaly(metric_name, anomaly_result, summary)
                
            except Exception as e:
                logger.warning(f"Failed to analyze metric {metric_name}: {e}")
    
    def _handle_performance_anomaly(self, metric_name: str, 
                                   anomaly_result: Dict[str, Any],
                                   metric_summary: Dict[str, Any]):
        """Handle detected performance anomaly"""
        
        severity = anomaly_result['severity']
        current_value = anomaly_result['current_value']
        anomalies = anomaly_result['anomalies']
        
        # Determine alert type and message based on metric
        if 'decision_cycle_latency' in metric_name:
            alert_type = 'performance_degradation'
            message = f"Decision cycle latency anomaly: {current_value:.2f}ms (target: <{self.config.latency_threshold_ms}ms)"
            
        elif 'safety_event' in metric_name:
            alert_type = 'safety_anomaly'
            safety_rate = current_value if current_value <= 1.0 else 1.0
            message = f"Safety performance anomaly: {safety_rate:.1%} success rate (target: >{self.config.safety_rate_threshold:.1%})"
            
        elif 'memory' in metric_name:
            alert_type = 'resource_anomaly'
            message = f"Memory usage anomaly: {current_value:.0f}MB (threshold: {self.config.memory_threshold_mb}MB)"
            
        elif 'cpu' in metric_name:
            alert_type = 'resource_anomaly'
            message = f"CPU usage anomaly: {current_value:.1f}% (threshold: {self.config.cpu_threshold_percent}%)"
            
        else:
            alert_type = 'general_anomaly'
            message = f"Performance anomaly detected in {metric_name}: {current_value:.3f}"
        
        # Prepare alert details
        alert_details = {
            'metric_name': metric_name,
            'current_value': current_value,
            'metric_summary': {k: v for k, v in metric_summary.items() if k != 'error'},
            'anomalies': anomalies,
            'detection_methods': [a['method'] for a in anomalies]
        }
        
        # Send alert
        self.alert_manager.process_alert(alert_type, severity, message, alert_details)
    
    def _update_performance_dashboard(self):
        """Update performance dashboard with current status"""
        current_time = time.time()
        
        # Get key performance metrics
        key_metrics = {}
        
        # Latency metrics
        latency_metrics = [name for name in self.metric_collector.current_metrics.keys() 
                          if 'decision_cycle_latency' in name]
        
        if latency_metrics:
            latest_latencies = []
            for metric_name in latency_metrics:
                summary = self.metric_collector.get_metric_summary(metric_name, time_window_s=60)
                if 'error' not in summary:
                    latest_latencies.append(summary['latest'])
            
            if latest_latencies:
                key_metrics['avg_decision_cycle_ms'] = float(np.mean(latest_latencies))
                key_metrics['max_decision_cycle_ms'] = float(np.max(latest_latencies))
        
        # Safety metrics
        safety_metrics = [name for name in self.metric_collector.current_metrics.keys() 
                         if 'safety_event' in name]
        
        if safety_metrics:
            safety_rates = []
            for metric_name in safety_metrics:
                summary = self.metric_collector.get_metric_summary(metric_name, time_window_s=300)
                if 'error' not in summary:
                    safety_rates.append(summary['mean'])
            
            if safety_rates:
                key_metrics['avg_safety_rate'] = float(np.mean(safety_rates))
        
        # Resource metrics
        resource_metrics = {
            'system_memory_mb': 'memory_usage_mb',
            'system_cpu_percent': 'cpu_usage_percent'
        }
        
        for metric_name, display_name in resource_metrics.items():
            summary = self.metric_collector.get_metric_summary(metric_name, time_window_s=60)
            if 'error' not in summary:
                key_metrics[display_name] = summary['latest']
        
        # Determine overall system status
        system_status = self._determine_system_status(key_metrics)
        
        # Update dashboard
        self.performance_dashboard.update({
            'system_status': system_status,
            'last_update': current_time,
            'key_metrics': key_metrics,
            'alerts': self.alert_manager.get_alert_summary(),
            'uptime_s': current_time - getattr(self, 'start_time', current_time)
        })
    
    def _determine_system_status(self, key_metrics: Dict[str, Any]) -> str:
        """Determine overall system status from key metrics"""
        
        issues = []
        
        # Check decision cycle performance
        avg_latency = key_metrics.get('avg_decision_cycle_ms')
        if avg_latency and avg_latency > self.config.latency_threshold_ms:
            issues.append('latency')
        
        # Check safety performance
        avg_safety = key_metrics.get('avg_safety_rate')
        if avg_safety and avg_safety < self.config.safety_rate_threshold:
            issues.append('safety')
        
        # Check resource usage
        memory_usage = key_metrics.get('memory_usage_mb')
        if memory_usage and memory_usage > self.config.memory_threshold_mb:
            issues.append('memory')
        
        cpu_usage = key_metrics.get('cpu_usage_percent')
        if cpu_usage and cpu_usage > self.config.cpu_threshold_percent:
            issues.append('cpu')
        
        # Check for active alerts
        alert_summary = self.alert_manager.get_alert_summary()
        active_alerts = alert_summary.get('active_alerts', 0)
        
        # Determine status
        if not issues and active_alerts == 0:
            return 'healthy'
        elif len(issues) <= 1 and active_alerts <= 2:
            return 'warning'
        else:
            return 'critical'
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        report_time = time.time()
        
        # Get summaries for all key metrics
        metric_summaries = {}
        important_metrics = [
            'decision_cycle_latency',
            'safety_event',
            'system_memory_mb',
            'system_cpu_percent'
        ]
        
        with self.metric_collector.metrics_lock:
            all_metrics = list(self.metric_collector.current_metrics.keys())
        
        for metric_pattern in important_metrics:
            matching_metrics = [m for m in all_metrics if metric_pattern in m]
            for metric_name in matching_metrics:
                summary = self.metric_collector.get_metric_summary(metric_name, time_window_s=3600)  # 1 hour
                if 'error' not in summary:
                    metric_summaries[metric_name] = summary
        
        # Performance targets assessment
        performance_assessment = {
            'decision_cycle_target_met': True,
            'safety_target_met': True,
            'resource_targets_met': True
        }
        
        # Check decision cycle targets
        decision_metrics = {k: v for k, v in metric_summaries.items() if 'decision_cycle_latency' in k}
        if decision_metrics:
            max_mean_latency = max(summary['mean'] for summary in decision_metrics.values())
            performance_assessment['decision_cycle_target_met'] = max_mean_latency <= self.config.latency_threshold_ms
        
        # Check safety targets
        safety_metrics = {k: v for k, v in metric_summaries.items() if 'safety_event' in k}
        if safety_metrics:
            min_safety_rate = min(summary['mean'] for summary in safety_metrics.values())
            performance_assessment['safety_target_met'] = min_safety_rate >= self.config.safety_rate_threshold
        
        return {
            'report_timestamp': report_time,
            'monitoring_duration_s': report_time - getattr(self, 'start_time', report_time),
            'system_status': self.performance_dashboard['system_status'],
            'key_metrics': self.performance_dashboard['key_metrics'],
            'metric_summaries': metric_summaries,
            'performance_assessment': performance_assessment,
            'alert_summary': self.alert_manager.get_alert_summary(),
            'recommendations': self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        key_metrics = self.performance_dashboard.get('key_metrics', {})
        
        # Latency recommendations
        avg_latency = key_metrics.get('avg_decision_cycle_ms')
        if avg_latency and avg_latency > self.config.latency_threshold_ms * 0.8:
            recommendations.append(
                f"Decision cycle latency ({avg_latency:.1f}ms) approaching threshold. Consider optimization."
            )
        
        # Memory recommendations
        memory_usage = key_metrics.get('memory_usage_mb')
        if memory_usage and memory_usage > self.config.memory_threshold_mb * 0.8:
            recommendations.append(
                f"Memory usage ({memory_usage:.0f}MB) high. Consider memory optimization or scaling."
            )
        
        # CPU recommendations
        cpu_usage = key_metrics.get('cpu_usage_percent')
        if cpu_usage and cpu_usage > self.config.cpu_threshold_percent * 0.8:
            recommendations.append(
                f"CPU usage ({cpu_usage:.1f}%) high. Consider performance optimization or scaling."
            )
        
        return recommendations


# Production monitoring context manager for easy integration
class ProductionMonitoringContext:
    """
    Context manager for production performance monitoring.
    
    Provides easy integration with existing systems for automatic
    performance monitoring during operations.
    """
    
    def __init__(self, system_components: Dict[str, Any], 
                 config: Optional[MonitoringConfig] = None):
        self.monitor = PerformanceMonitor(config)
        self.system_components = system_components
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.monitor.start_monitoring(self.system_components)
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop_monitoring()
        
        if exc_type is not None:
            # Log any exceptions that occurred during monitoring
            self.monitor.alert_manager.process_alert(
                'system_error', 'critical',
                f"System exception during monitoring: {exc_type.__name__}",
                {'exception': str(exc_val), 'duration_s': time.time() - self.start_time}
            )


if __name__ == "__main__":
    print("🔍 Production Performance Monitoring System Ready")
    print("   Use ProductionMonitoringContext for automatic monitoring")
    print("   Provides real-time performance validation with statistical alerts")