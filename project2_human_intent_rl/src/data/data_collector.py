"""
Comprehensive Data Collection System for HRI Bayesian RL

This module provides automated data collection, preprocessing, and management
for the human-robot interaction Bayesian reinforcement learning system.

Features:
- Real-time data collection during experiments
- Automated data validation and quality checks
- Multi-format data export (CSV, JSON, HDF5, Parquet)
- Data versioning and metadata management
- Performance monitoring and logging
- Integration with experimental framework

Author: Phase 5 Implementation
Date: 2024
"""

import numpy as np
import pandas as pd
import json
import pickle
import logging
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import hashlib
import uuid
from datetime import datetime
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional dependency with graceful fallback
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    h5py = None
    HAS_H5PY = False
    logger.warning("h5py not available - HDF5 data storage disabled. Install with: pip install h5py")


class DataType(Enum):
    """Types of data collected"""
    EXPERIMENTAL_TRIAL = auto()
    SYSTEM_PERFORMANCE = auto()
    HUMAN_BEHAVIOR = auto()
    ROBOT_STATE = auto()
    ENVIRONMENT_STATE = auto()
    ALGORITHM_METRICS = auto()
    SAFETY_EVENTS = auto()
    USER_INTERACTION = auto()


class DataFormat(Enum):
    """Supported data export formats"""
    CSV = "csv"
    JSON = "json"
    HDF5 = "h5"
    PARQUET = "parquet"
    PICKLE = "pkl"
    XLSX = "xlsx"


@dataclass
class DataCollectionConfig:
    """Configuration for data collection system"""
    # Collection settings
    collection_enabled: bool = True
    real_time_collection: bool = True
    buffer_size: int = 1000
    flush_interval: float = 10.0  # seconds
    
    # Storage settings
    output_directory: str = "collected_data"
    auto_backup: bool = True
    backup_interval: float = 300.0  # 5 minutes
    max_file_size_mb: float = 100.0
    
    # Data validation
    validate_data: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Export settings
    export_formats: List[DataFormat] = field(default_factory=lambda: [DataFormat.CSV, DataFormat.JSON])
    compress_exports: bool = True
    
    # Metadata
    experiment_id: str = ""
    session_id: str = ""
    researcher_info: Dict[str, str] = field(default_factory=dict)
    
    # Performance
    enable_monitoring: bool = True
    memory_limit_mb: float = 1000.0


@dataclass
class DataRecord:
    """Single data record with metadata"""
    record_id: str
    data_type: DataType
    timestamp: float
    session_id: str
    experiment_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'record_id': self.record_id,
            'data_type': self.data_type.name,
            'timestamp': self.timestamp,
            'session_id': self.session_id,
            'experiment_id': self.experiment_id,
            'data': self.data,
            'metadata': self.metadata
        }


class DataBuffer:
    """Thread-safe data buffer for real-time collection"""
    
    def __init__(self, max_size: int = 1000):
        """Initialize data buffer"""
        self.max_size = max_size
        self._buffer = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._total_records = 0
        self._dropped_records = 0
        
    def add_record(self, record: DataRecord) -> bool:
        """Add record to buffer"""
        try:
            with self._lock:
                self._buffer.put_nowait(record)
                self._total_records += 1
                return True
        except queue.Full:
            self._dropped_records += 1
            logger.warning(f"Data buffer full. Dropped record {record.record_id}")
            return False
    
    def get_records(self, max_count: Optional[int] = None) -> List[DataRecord]:
        """Get records from buffer"""
        records = []
        count = 0
        
        with self._lock:
            while not self._buffer.empty() and (max_count is None or count < max_count):
                try:
                    record = self._buffer.get_nowait()
                    records.append(record)
                    count += 1
                except queue.Empty:
                    break
        
        return records
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics"""
        with self._lock:
            return {
                'current_size': self._buffer.qsize(),
                'max_size': self.max_size,
                'total_records': self._total_records,
                'dropped_records': self._dropped_records
            }
    
    def clear(self):
        """Clear buffer"""
        with self._lock:
            while not self._buffer.empty():
                try:
                    self._buffer.get_nowait()
                except queue.Empty:
                    break


class DataValidator:
    """Data validation and quality checks"""
    
    def __init__(self, config: DataCollectionConfig):
        """Initialize data validator"""
        self.config = config
        self.validation_rules = config.validation_rules
        self.validation_stats = {
            'total_validated': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'validation_errors': []
        }
    
    def validate_record(self, record: DataRecord) -> Tuple[bool, List[str]]:
        """Validate a single data record"""
        if not self.config.validate_data:
            return True, []
        
        errors = []
        
        # Basic validation
        if not record.record_id:
            errors.append("Missing record ID")
        
        if not record.session_id:
            errors.append("Missing session ID")
        
        if record.timestamp <= 0:
            errors.append("Invalid timestamp")
        
        if not record.data:
            errors.append("Empty data field")
        
        # Type-specific validation
        errors.extend(self._validate_by_type(record))
        
        # Custom validation rules
        errors.extend(self._apply_custom_rules(record))
        
        # Update statistics
        self.validation_stats['total_validated'] += 1
        if errors:
            self.validation_stats['validation_failed'] += 1
            self.validation_stats['validation_errors'].extend(errors)
        else:
            self.validation_stats['validation_passed'] += 1
        
        return len(errors) == 0, errors
    
    def _validate_by_type(self, record: DataRecord) -> List[str]:
        """Type-specific validation"""
        errors = []
        
        if record.data_type == DataType.EXPERIMENTAL_TRIAL:
            required_fields = ['success', 'completion_time', 'method']
            for field in required_fields:
                if field not in record.data:
                    errors.append(f"Missing required field: {field}")
        
        elif record.data_type == DataType.SYSTEM_PERFORMANCE:
            if 'cpu_usage' in record.data and not (0 <= record.data['cpu_usage'] <= 100):
                errors.append("CPU usage must be between 0 and 100")
            
            if 'memory_usage' in record.data and record.data['memory_usage'] < 0:
                errors.append("Memory usage cannot be negative")
        
        elif record.data_type == DataType.HUMAN_BEHAVIOR:
            if 'position' in record.data:
                pos = record.data['position']
                if not isinstance(pos, (list, np.ndarray)) or len(pos) != 3:
                    errors.append("Human position must be 3D coordinates")
        
        return errors
    
    def _apply_custom_rules(self, record: DataRecord) -> List[str]:
        """Apply custom validation rules"""
        errors = []
        
        for rule_name, rule_config in self.validation_rules.items():
            try:
                if not self._check_rule(record, rule_config):
                    errors.append(f"Custom rule failed: {rule_name}")
            except Exception as e:
                errors.append(f"Error in custom rule {rule_name}: {e}")
        
        return errors
    
    def _check_rule(self, record: DataRecord, rule_config: Dict[str, Any]) -> bool:
        """Check individual validation rule"""
        # Simple rule implementation
        if 'field' in rule_config and 'condition' in rule_config:
            field = rule_config['field']
            condition = rule_config['condition']
            
            if field in record.data:
                value = record.data[field]
                
                if condition == 'not_null':
                    return value is not None
                elif condition == 'positive':
                    return value > 0
                elif condition == 'non_negative':
                    return value >= 0
                elif 'range' in condition:
                    min_val, max_val = condition['range']
                    return min_val <= value <= max_val
        
        return True
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.validation_stats.copy()
        if stats['total_validated'] > 0:
            stats['success_rate'] = stats['validation_passed'] / stats['total_validated']
        else:
            stats['success_rate'] = 0.0
        return stats


class DataExporter:
    """Data export to various formats"""
    
    def __init__(self, config: DataCollectionConfig):
        """Initialize data exporter"""
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def export_data(self, records: List[DataRecord], 
                   filename_prefix: str = "data") -> Dict[DataFormat, str]:
        """Export data to configured formats"""
        exported_files = {}
        
        # Convert to DataFrame for easier export
        df = self._records_to_dataframe(records)
        
        # Export to each configured format
        for format_type in self.config.export_formats:
            try:
                filepath = self._export_format(df, format_type, filename_prefix)
                exported_files[format_type] = str(filepath)
                logger.info(f"Exported {len(records)} records to {filepath}")
            except Exception as e:
                logger.error(f"Failed to export to {format_type.value}: {e}")
        
        return exported_files
    
    def _records_to_dataframe(self, records: List[DataRecord]) -> pd.DataFrame:
        """Convert records to pandas DataFrame"""
        rows = []
        
        for record in records:
            row = {
                'record_id': record.record_id,
                'data_type': record.data_type.name,
                'timestamp': record.timestamp,
                'session_id': record.session_id,
                'experiment_id': record.experiment_id
            }
            
            # Flatten data fields
            for key, value in record.data.items():
                if isinstance(value, (list, np.ndarray)):
                    if len(value) <= 10:  # Only flatten small arrays
                        for i, v in enumerate(value):
                            row[f"{key}_{i}"] = v
                    else:
                        row[key] = str(value)  # Convert to string for large arrays
                else:
                    row[key] = value
            
            # Add metadata
            for key, value in record.metadata.items():
                row[f"meta_{key}"] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _export_format(self, df: pd.DataFrame, format_type: DataFormat, 
                      filename_prefix: str) -> Path:
        """Export DataFrame to specific format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == DataFormat.CSV:
            filepath = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
            df.to_csv(filepath, index=False, compression='gzip' if self.config.compress_exports else None)
            
        elif format_type == DataFormat.JSON:
            filepath = self.output_dir / f"{filename_prefix}_{timestamp}.json"
            df.to_json(filepath, orient='records', indent=2)
            if self.config.compress_exports:
                import gzip
                with open(filepath, 'rb') as f_in:
                    filepath_gz = Path(str(filepath) + '.gz')
                    with gzip.open(filepath_gz, 'wb') as f_out:
                        f_out.writelines(f_in)
                filepath.unlink()  # Remove uncompressed file
                filepath = filepath_gz
                
        elif format_type == DataFormat.HDF5:
            if not HAS_H5PY:
                logger.warning("HDF5 format requested but h5py not available. Falling back to CSV format.")
                # Fallback to CSV
                filepath = self.output_dir / f"{filename_prefix}_{timestamp}.csv"
                df.to_csv(filepath, index=False)
            else:
                filepath = self.output_dir / f"{filename_prefix}_{timestamp}.h5"
                df.to_hdf(filepath, key='data', mode='w', complib='zlib' if self.config.compress_exports else None)
            
        elif format_type == DataFormat.PARQUET:
            filepath = self.output_dir / f"{filename_prefix}_{timestamp}.parquet"
            df.to_parquet(filepath, compression='gzip' if self.config.compress_exports else None)
            
        elif format_type == DataFormat.PICKLE:
            filepath = self.output_dir / f"{filename_prefix}_{timestamp}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(df, f)
                
        elif format_type == DataFormat.XLSX:
            filepath = self.output_dir / f"{filename_prefix}_{timestamp}.xlsx"
            df.to_excel(filepath, index=False)
            
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return filepath


class PerformanceMonitor:
    """Monitor data collection performance"""
    
    def __init__(self, config: DataCollectionConfig):
        """Initialize performance monitor"""
        self.config = config
        self.start_time = time.time()
        self.metrics = {
            'records_collected': 0,
            'records_exported': 0,
            'bytes_written': 0,
            'collection_errors': 0,
            'export_errors': 0,
            'memory_usage_peak': 0.0,
            'cpu_usage_peak': 0.0
        }
        
        self._monitoring_active = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.config.enable_monitoring and not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Performance monitoring loop"""
        while self._monitoring_active:
            try:
                # Get system metrics
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent()
                
                # Update peak values
                self.metrics['memory_usage_peak'] = max(self.metrics['memory_usage_peak'], memory_mb)
                self.metrics['cpu_usage_peak'] = max(self.metrics['cpu_usage_peak'], cpu_percent)
                
                # Check memory limit
                if memory_mb > self.config.memory_limit_mb:
                    logger.warning(f"Memory usage ({memory_mb:.1f} MB) exceeds limit ({self.config.memory_limit_mb} MB)")
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(5.0)
    
    def record_metric(self, metric_name: str, value: Union[int, float]):
        """Record a performance metric"""
        if metric_name in self.metrics:
            if isinstance(self.metrics[metric_name], (int, float)):
                if metric_name.endswith('_peak'):
                    self.metrics[metric_name] = max(self.metrics[metric_name], value)
                else:
                    self.metrics[metric_name] += value
            else:
                self.metrics[metric_name] = value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        runtime = time.time() - self.start_time
        
        summary = {
            'runtime_seconds': runtime,
            'records_per_second': self.metrics['records_collected'] / runtime if runtime > 0 else 0,
            'export_rate': self.metrics['records_exported'] / runtime if runtime > 0 else 0,
            'error_rate': (self.metrics['collection_errors'] + self.metrics['export_errors']) / 
                         max(1, self.metrics['records_collected']),
            **self.metrics
        }
        
        return summary


class HRIDataCollector:
    """Main HRI data collection system"""
    
    def __init__(self, config: DataCollectionConfig):
        """Initialize HRI data collector"""
        self.config = config
        
        # Initialize components
        self.buffer = DataBuffer(config.buffer_size)
        self.validator = DataValidator(config)
        self.exporter = DataExporter(config)
        self.monitor = PerformanceMonitor(config)
        
        # State management
        self.session_id = config.session_id or str(uuid.uuid4())
        self.experiment_id = config.experiment_id or f"exp_{int(time.time())}"
        self.is_collecting = False
        
        # Threading
        self._collection_thread = None
        self._export_thread = None
        self._stop_event = threading.Event()
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initialized HRI data collector (Session: {self.session_id})")
    
    def start_collection(self):
        """Start data collection system"""
        if self.is_collecting:
            logger.warning("Data collection already started")
            return
        
        self.is_collecting = True
        self._stop_event.clear()
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start background threads
        if self.config.real_time_collection:
            self._export_thread = threading.Thread(target=self._export_loop, daemon=True)
            self._export_thread.start()
        
        logger.info("Data collection started")
    
    def stop_collection(self):
        """Stop data collection system"""
        if not self.is_collecting:
            return
        
        logger.info("Stopping data collection...")
        
        self.is_collecting = False
        self._stop_event.set()
        
        # Wait for threads to finish
        if self._export_thread:
            self._export_thread.join(timeout=10)
        
        # Final data export
        remaining_records = self.buffer.get_records()
        if remaining_records:
            self.export_collected_data(remaining_records, "final_export")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        logger.info("Data collection stopped")
    
    def collect_trial_data(self, trial_result: Any) -> bool:
        """Collect data from experimental trial"""
        try:
            # Extract trial data
            data = {
                'trial_id': getattr(trial_result, 'trial_id', None),
                'method': getattr(trial_result, 'method', None),
                'success': getattr(trial_result, 'success', None),
                'task_completion_time': getattr(trial_result, 'task_completion_time', None),
                'safety_violations': getattr(trial_result, 'safety_violations', None),
                'human_comfort_score': getattr(trial_result, 'human_comfort_score', None),
                'step_count': getattr(trial_result, 'step_count', None),
                'average_decision_time': getattr(trial_result, 'average_decision_time', None),
                'max_decision_time': getattr(trial_result, 'max_decision_time', None),
                'memory_usage': getattr(trial_result, 'memory_usage', None)
            }
            
            # Add scenario parameters if available
            if hasattr(trial_result, 'scenario_params'):
                data.update({f"scenario_{k}": v for k, v in trial_result.scenario_params.items()})
            
            # Add additional metrics if available
            if hasattr(trial_result, 'additional_metrics'):
                data.update({f"additional_{k}": v for k, v in trial_result.additional_metrics.items()})
            
            # Create data record
            record = DataRecord(
                record_id=f"trial_{self.session_id}_{int(time.time() * 1000000)}",
                data_type=DataType.EXPERIMENTAL_TRIAL,
                timestamp=time.time(),
                session_id=self.session_id,
                experiment_id=self.experiment_id,
                data=data,
                metadata={'source': 'experimental_trial'}
            )
            
            return self._add_record(record)
            
        except Exception as e:
            logger.error(f"Failed to collect trial data: {e}")
            self.monitor.record_metric('collection_errors', 1)
            return False
    
    def collect_system_performance(self, performance_data: Dict[str, Any]) -> bool:
        """Collect system performance data"""
        try:
            record = DataRecord(
                record_id=f"perf_{self.session_id}_{int(time.time() * 1000000)}",
                data_type=DataType.SYSTEM_PERFORMANCE,
                timestamp=time.time(),
                session_id=self.session_id,
                experiment_id=self.experiment_id,
                data=performance_data,
                metadata={'source': 'system_monitor'}
            )
            
            return self._add_record(record)
            
        except Exception as e:
            logger.error(f"Failed to collect performance data: {e}")
            self.monitor.record_metric('collection_errors', 1)
            return False
    
    def collect_human_behavior(self, behavior_data: Dict[str, Any]) -> bool:
        """Collect human behavior data"""
        try:
            record = DataRecord(
                record_id=f"behavior_{self.session_id}_{int(time.time() * 1000000)}",
                data_type=DataType.HUMAN_BEHAVIOR,
                timestamp=time.time(),
                session_id=self.session_id,
                experiment_id=self.experiment_id,
                data=behavior_data,
                metadata={'source': 'human_behavior_tracker'}
            )
            
            return self._add_record(record)
            
        except Exception as e:
            logger.error(f"Failed to collect behavior data: {e}")
            self.monitor.record_metric('collection_errors', 1)
            return False
    
    def collect_safety_event(self, event_data: Dict[str, Any]) -> bool:
        """Collect safety event data"""
        try:
            record = DataRecord(
                record_id=f"safety_{self.session_id}_{int(time.time() * 1000000)}",
                data_type=DataType.SAFETY_EVENTS,
                timestamp=time.time(),
                session_id=self.session_id,
                experiment_id=self.experiment_id,
                data=event_data,
                metadata={'source': 'safety_monitor', 'priority': 'high'}
            )
            
            return self._add_record(record)
            
        except Exception as e:
            logger.error(f"Failed to collect safety event: {e}")
            self.monitor.record_metric('collection_errors', 1)
            return False
    
    def export_collected_data(self, records: List[DataRecord] = None, 
                            filename_prefix: str = "hri_data") -> Dict[DataFormat, str]:
        """Export collected data"""
        if records is None:
            records = self.buffer.get_records()
        
        if not records:
            logger.warning("No data to export")
            return {}
        
        try:
            exported_files = self.exporter.export_data(records, filename_prefix)
            self.monitor.record_metric('records_exported', len(records))
            
            # Calculate exported file sizes
            total_bytes = 0
            for filepath in exported_files.values():
                if os.path.exists(filepath):
                    total_bytes += os.path.getsize(filepath)
            self.monitor.record_metric('bytes_written', total_bytes)
            
            return exported_files
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            self.monitor.record_metric('export_errors', 1)
            return {}
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get data collection statistics"""
        buffer_stats = self.buffer.get_stats()
        validation_stats = self.validator.get_validation_stats()
        performance_stats = self.monitor.get_performance_summary()
        
        return {
            'session_id': self.session_id,
            'experiment_id': self.experiment_id,
            'is_collecting': self.is_collecting,
            'buffer_stats': buffer_stats,
            'validation_stats': validation_stats,
            'performance_stats': performance_stats
        }
    
    def _add_record(self, record: DataRecord) -> bool:
        """Add record to buffer with validation"""
        if not self.is_collecting:
            return False
        
        # Validate record
        is_valid, errors = self.validator.validate_record(record)
        if not is_valid:
            logger.warning(f"Record validation failed: {errors}")
            return False
        
        # Add to buffer
        success = self.buffer.add_record(record)
        if success:
            self.monitor.record_metric('records_collected', 1)
        
        return success
    
    def _export_loop(self):
        """Background export loop"""
        while not self._stop_event.wait(self.config.flush_interval):
            try:
                records = self.buffer.get_records()
                if records:
                    self.export_collected_data(records, f"realtime_{int(time.time())}")
            except Exception as e:
                logger.error(f"Error in export loop: {e}")


# Convenience functions
def create_default_collector(experiment_name: str = "hri_experiment", 
                           output_dir: str = "data_collection") -> HRIDataCollector:
    """Create data collector with default configuration"""
    config = DataCollectionConfig(
        output_directory=output_dir,
        experiment_id=experiment_name,
        export_formats=[DataFormat.CSV, DataFormat.JSON],
        real_time_collection=True,
        validate_data=True
    )
    return HRIDataCollector(config)


def collect_experimental_session(experiment_results: List[Any], 
                                output_dir: str = "experimental_data") -> str:
    """Collect and export complete experimental session data"""
    collector = create_default_collector("experimental_session", output_dir)
    collector.start_collection()
    
    try:
        # Collect all trial data
        for result in experiment_results:
            collector.collect_trial_data(result)
        
        # Export collected data
        exported_files = collector.export_collected_data()
        
        # Get statistics
        stats = collector.get_collection_statistics()
        logger.info(f"Collection completed: {stats}")
        
        return list(exported_files.values())[0] if exported_files else ""
        
    finally:
        collector.stop_collection()


# Example usage and testing
if __name__ == "__main__":
    # Test data collection system
    logger.info("Testing HRI Data Collection System")
    
    # Create test configuration
    config = DataCollectionConfig(
        output_directory="test_data_collection",
        experiment_id="test_experiment",
        export_formats=[DataFormat.CSV, DataFormat.JSON],
        real_time_collection=False,
        validate_data=True,
        flush_interval=5.0
    )
    
    # Initialize collector
    collector = HRIDataCollector(config)
    
    # Start collection
    collector.start_collection()
    
    try:
        # Generate and collect sample data
        for i in range(20):
            # Sample trial data
            trial_data = type('MockTrial', (), {
                'trial_id': i,
                'method': 'Bayesian_RL_Full',
                'success': True,
                'task_completion_time': 5.0 + np.random.normal(0, 1),
                'safety_violations': np.random.poisson(0.1),
                'human_comfort_score': np.random.beta(8, 2),
                'step_count': np.random.randint(50, 200),
                'average_decision_time': np.random.uniform(0.02, 0.1),
                'max_decision_time': np.random.uniform(0.05, 0.2),
                'memory_usage': np.random.uniform(50, 200),
                'scenario_params': {'noise_level': 0.1},
                'additional_metrics': {'efficiency': np.random.random()}
            })()
            
            collector.collect_trial_data(trial_data)
            
            # Sample performance data
            perf_data = {
                'cpu_usage': np.random.uniform(10, 80),
                'memory_usage': np.random.uniform(100, 500),
                'processing_time': np.random.uniform(0.01, 0.1),
                'queue_size': np.random.randint(0, 10)
            }
            collector.collect_system_performance(perf_data)
            
            # Sample behavior data
            behavior_data = {
                'position': [0.8 + np.random.normal(0, 0.1), 
                           0.3 + np.random.normal(0, 0.1), 
                           0.8 + np.random.normal(0, 0.05)],
                'intent': 'handover_request',
                'comfort_level': np.random.beta(5, 2),
                'engagement_level': np.random.beta(6, 3)
            }
            collector.collect_human_behavior(behavior_data)
            
            time.sleep(0.1)  # Small delay
        
        # Export data
        exported_files = collector.export_collected_data()
        
        # Get final statistics
        stats = collector.get_collection_statistics()
        
        logger.info("Data collection test completed successfully!")
        logger.info(f"Exported files: {list(exported_files.values())}")
        logger.info(f"Collection statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Data collection test failed: {e}")
    finally:
        collector.stop_collection()
    
    print("Data collection system test completed!")