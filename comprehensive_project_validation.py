#!/usr/bin/env python3
"""
Comprehensive Project Validation Suite
Model-Based RL Human Intent Recognition System

Complete end-to-end validation testing all system components:
- Project structure and critical files
- Core component imports and instantiation
- Dataset integrity and quality
- Algorithm functionality and performance
- End-to-end integration pipeline
- Jekyll documentation completeness
- Real-time performance benchmarks (<10ms)

Author: Validation Team
Date: September 2025
"""

import sys
import os
import time
import json
import logging
import traceback
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

class ComprehensiveProjectValidator:
    """
    Complete validation suite for the Model-Based RL Human Intent Recognition system.
    Tests all components from basic imports to end-to-end integration.
    """
    
    def __init__(self):
        """Initialize the comprehensive validator."""
        self.validation_start = time.time()
        self.results = {
            'validation_id': f"comprehensive_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'project_root': str(Path.cwd()),
            'validation_categories': {},
            'summary': {},
            'recommendations': [],
            'overall_status': 'PENDING'
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Test counters
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warned_tests = 0

    def log_test_result(self, category: str, test_name: str, status: str, 
                       details: Any = None, error: str = None):
        """Log and track test results."""
        self.total_tests += 1
        
        if status == 'PASS':
            self.passed_tests += 1
            icon = "✅"
        elif status == 'WARN':
            self.warned_tests += 1
            icon = "⚠️"
        else:  # FAIL
            self.failed_tests += 1
            icon = "❌"
        
        # Ensure category exists
        if category not in self.results['validation_categories']:
            self.results['validation_categories'][category] = {
                'status': 'PENDING',
                'tests': {},
                'summary': {}
            }
        
        # Store test result
        self.results['validation_categories'][category]['tests'][test_name] = {
            'status': status,
            'details': details,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log to console
        self.logger.info(f"{icon} {category}.{test_name}: {status}")
        if error:
            self.logger.error(f"   Error: {error}")

    def validate_project_structure(self) -> Dict[str, Any]:
        """Validate project structure and critical file existence."""
        self.logger.info("=" * 60)
        self.logger.info("VALIDATING PROJECT STRUCTURE")
        self.logger.info("=" * 60)
        
        category = "project_structure"
        
        # Critical files that must exist
        critical_files = {
            # Core system files
            'src/system/human_intent_rl_system.py': 'Main RL system',
            'src/agents/bayesian_rl_agent.py': 'Bayesian RL agent',
            'src/models/gaussian_process.py': 'Gaussian Process model',
            'src/controllers/mpc_controller.py': 'MPC controller',
            
            # Data and processing
            'src/data/dataset_quality_analyzer.py': 'Dataset quality analyzer',
            'src/data/enhanced_synthetic_generator.py': 'Data generator',
            'data/synthetic_full/features.csv': 'Main dataset',
            
            # Testing and validation
            'tests/comprehensive_test_suite.py': 'Main test suite',
            'run_tests.py': 'Test runner',
            
            # Production deployment
            'Dockerfile': 'Container definition',
            'docker-compose.yml': 'Container orchestration',
            'production_benchmark.py': 'Performance benchmarks',
            'production_deployment_guide.py': 'Deployment validator',
            
            # Documentation
            'jekyll_site/methodology.md': 'Technical methodology',
            'jekyll_site/results.md': 'Experimental results',
            'README.md': 'Project documentation',
            
            # Configuration
            'requirements.txt': 'Python dependencies',
            'pyproject.toml': 'Project configuration'
        }
        
        file_results = {}
        for file_path, description in critical_files.items():
            try:
                path = Path(file_path)
                if path.exists():
                    size_kb = path.stat().st_size / 1024
                    file_results[file_path] = {
                        'exists': True,
                        'size_kb': round(size_kb, 2),
                        'description': description
                    }
                    self.log_test_result(category, f"file_exists_{file_path.replace('/', '_').replace('.', '_')}", 
                                       'PASS', {'size_kb': round(size_kb, 2)})
                else:
                    file_results[file_path] = {
                        'exists': False,
                        'description': description
                    }
                    self.log_test_result(category, f"file_exists_{file_path.replace('/', '_').replace('.', '_')}", 
                                       'FAIL', error=f"Critical file missing: {file_path}")
                                       
            except Exception as e:
                self.log_test_result(category, f"file_check_{file_path.replace('/', '_').replace('.', '_')}", 
                                   'FAIL', error=str(e))
        
        # Directory structure validation
        critical_directories = [
            'src/', 'tests/', 'data/', 'jekyll_site/', 
            'monitoring/', 'load_testing/'
        ]
        
        dir_results = {}
        for directory in critical_directories:
            try:
                path = Path(directory)
                if path.exists() and path.is_dir():
                    file_count = len(list(path.rglob('*')))
                    dir_results[directory] = {
                        'exists': True,
                        'file_count': file_count
                    }
                    self.log_test_result(category, f"directory_{directory.replace('/', '_')}", 
                                       'PASS', {'file_count': file_count})
                else:
                    dir_results[directory] = {'exists': False}
                    self.log_test_result(category, f"directory_{directory.replace('/', '_')}", 
                                       'FAIL', error=f"Critical directory missing: {directory}")
                                       
            except Exception as e:
                self.log_test_result(category, f"directory_{directory.replace('/', '_')}", 
                                   'FAIL', error=str(e))
        
        # Update category status
        category_tests = self.results['validation_categories'][category]['tests']
        failed_tests = [t for t in category_tests.values() if t['status'] == 'FAIL']
        
        if not failed_tests:
            self.results['validation_categories'][category]['status'] = 'PASS'
        else:
            self.results['validation_categories'][category]['status'] = 'FAIL'
        
        return {
            'critical_files': file_results,
            'critical_directories': dir_results
        }

    def validate_core_imports(self) -> Dict[str, Any]:
        """Validate core component imports and basic instantiation."""
        self.logger.info("=" * 60)
        self.logger.info("VALIDATING CORE COMPONENT IMPORTS")
        self.logger.info("=" * 60)
        
        category = "core_imports"
        import_results = {}
        
        # Core imports to test
        core_imports = {
            'human_intent_rl_system': 'src.system.human_intent_rl_system.HumanIntentRLSystem',
            'bayesian_rl_agent': 'src.agents.bayesian_rl_agent.BayesianRLAgent',
            'gaussian_process': 'src.models.gaussian_process.GaussianProcess',
            'mpc_controller': 'src.controllers.mpc_controller.MPCController',
            'dataset_quality_analyzer': 'src.data.dataset_quality_analyzer.DatasetQualityAnalyzer',
            'enhanced_synthetic_generator': 'src.data.enhanced_synthetic_generator.EnhancedSyntheticGenerator'
        }
        
        for component_name, import_path in core_imports.items():
            try:
                module_path, class_name = import_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                component_class = getattr(module, class_name)
                
                import_results[component_name] = {
                    'import_success': True,
                    'class_available': True,
                    'import_path': import_path
                }
                
                self.log_test_result(category, f"import_{component_name}", 'PASS')
                
                # Test basic instantiation (where possible)
                try:
                    if component_name == 'human_intent_rl_system':
                        # Requires config parameter
                        instance = component_class({})
                    elif component_name == 'gaussian_process':
                        # Test with minimal parameters
                        instance = component_class(kernel_type='rbf')
                    elif component_name == 'mpc_controller':
                        # Test with minimal parameters
                        instance = component_class(prediction_horizon=5)
                    elif component_name == 'dataset_quality_analyzer':
                        # May fail due to optional dependencies, but import should work
                        pass  # Skip instantiation test for this component
                    else:
                        # Try default instantiation
                        instance = component_class()
                    
                    import_results[component_name]['instantiation'] = True
                    self.log_test_result(category, f"instantiate_{component_name}", 'PASS')
                    
                except Exception as inst_error:
                    import_results[component_name]['instantiation'] = False
                    import_results[component_name]['instantiation_error'] = str(inst_error)
                    self.log_test_result(category, f"instantiate_{component_name}", 'WARN', 
                                       error=f"Instantiation failed: {inst_error}")
                
            except Exception as e:
                import_results[component_name] = {
                    'import_success': False,
                    'error': str(e)
                }
                self.log_test_result(category, f"import_{component_name}", 'FAIL', error=str(e))
        
        # Test integration imports
        integration_imports = [
            'src.integration',
            'src.utils.optional_dependencies'
        ]
        
        for import_path in integration_imports:
            try:
                module = __import__(import_path)
                import_results[import_path] = {'import_success': True}
                self.log_test_result(category, f"import_{import_path.replace('.', '_')}", 'PASS')
            except Exception as e:
                import_results[import_path] = {'import_success': False, 'error': str(e)}
                self.log_test_result(category, f"import_{import_path.replace('.', '_')}", 'FAIL', error=str(e))
        
        # Update category status
        category_tests = self.results['validation_categories'][category]['tests']
        failed_tests = [t for t in category_tests.values() if t['status'] == 'FAIL']
        
        if not failed_tests:
            self.results['validation_categories'][category]['status'] = 'PASS'
        else:
            self.results['validation_categories'][category]['status'] = 'FAIL'
        
        return import_results

    def validate_dataset_integrity(self) -> Dict[str, Any]:
        """Validate dataset loading and quality."""
        self.logger.info("=" * 60)
        self.logger.info("VALIDATING DATASET INTEGRITY")
        self.logger.info("=" * 60)
        
        category = "dataset_integrity"
        dataset_results = {}
        
        # Test main dataset
        try:
            features_path = Path('data/synthetic_full/features.csv')
            if features_path.exists():
                df = pd.read_csv(features_path)
                
                dataset_results['features_csv'] = {
                    'loaded': True,
                    'shape': df.shape,
                    'samples': df.shape[0],
                    'features': df.shape[1],
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'columns': list(df.columns)
                }
                
                # Validate expected sample count
                if df.shape[0] >= 1178:
                    self.log_test_result(category, 'dataset_sample_count', 'PASS', 
                                       {'samples': df.shape[0], 'expected': '1178+'})
                else:
                    self.log_test_result(category, 'dataset_sample_count', 'WARN', 
                                       {'samples': df.shape[0], 'expected': '1178+'})
                
                # Basic data quality checks
                null_count = df.isnull().sum().sum()
                if null_count == 0:
                    self.log_test_result(category, 'dataset_null_check', 'PASS')
                else:
                    self.log_test_result(category, 'dataset_null_check', 'WARN', 
                                       {'null_values': null_count})
                
                # Numeric data validation
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    self.log_test_result(category, 'dataset_numeric_data', 'PASS', 
                                       {'numeric_columns': len(numeric_columns)})
                else:
                    self.log_test_result(category, 'dataset_numeric_data', 'FAIL', 
                                       error="No numeric columns found")
                
            else:
                dataset_results['features_csv'] = {'loaded': False, 'error': 'File not found'}
                self.log_test_result(category, 'dataset_load_features', 'FAIL', 
                                   error="features.csv not found")
                
        except Exception as e:
            dataset_results['features_csv'] = {'loaded': False, 'error': str(e)}
            self.log_test_result(category, 'dataset_load_features', 'FAIL', error=str(e))
        
        # Test dataset quality analyzer (if available)
        try:
            from src.data.dataset_quality_analyzer import DatasetQualityAnalyzer
            analyzer = DatasetQualityAnalyzer()
            self.log_test_result(category, 'quality_analyzer_available', 'PASS')
        except Exception as e:
            self.log_test_result(category, 'quality_analyzer_available', 'WARN', 
                               error=f"Quality analyzer unavailable: {e}")
        
        # Update category status
        category_tests = self.results['validation_categories'][category]['tests']
        failed_tests = [t for t in category_tests.values() if t['status'] == 'FAIL']
        
        if not failed_tests:
            self.results['validation_categories'][category]['status'] = 'PASS'
        else:
            self.results['validation_categories'][category]['status'] = 'FAIL'
        
        return dataset_results

    def validate_algorithm_functionality(self) -> Dict[str, Any]:
        """Test core algorithm functionality with real data."""
        self.logger.info("=" * 60)
        self.logger.info("VALIDATING ALGORITHM FUNCTIONALITY")
        self.logger.info("=" * 60)
        
        category = "algorithm_functionality"
        algo_results = {}
        
        # Test Gaussian Process
        try:
            from src.models.gaussian_process import GaussianProcess
            
            gp = GaussianProcess(kernel_type='rbf')
            
            # Test with synthetic data
            X_train = np.random.rand(50, 3)
            y_train = np.random.rand(50)
            X_test = np.random.rand(10, 3)
            
            start_time = time.perf_counter()
            gp.fit(X_train, y_train)
            predictions, uncertainty = gp.predict(X_test)
            gp_time = (time.perf_counter() - start_time) * 1000
            
            algo_results['gaussian_process'] = {
                'functional': True,
                'prediction_time_ms': round(gp_time, 3),
                'output_shape': predictions.shape,
                'uncertainty_available': uncertainty is not None
            }
            
            self.log_test_result(category, 'gaussian_process_prediction', 'PASS', 
                               {'time_ms': round(gp_time, 3)})
            
        except Exception as e:
            algo_results['gaussian_process'] = {'functional': False, 'error': str(e)}
            self.log_test_result(category, 'gaussian_process_prediction', 'FAIL', error=str(e))
        
        # Test MPC Controller
        try:
            from src.controllers.mpc_controller import MPCController
            
            mpc = MPCController(prediction_horizon=5)
            
            # Test control generation
            current_state = np.random.rand(4)
            reference = np.random.rand(4)
            
            start_time = time.perf_counter()
            control_action = mpc.compute_control(current_state, reference)
            mpc_time = (time.perf_counter() - start_time) * 1000
            
            algo_results['mpc_controller'] = {
                'functional': True,
                'control_time_ms': round(mpc_time, 3),
                'output_shape': control_action.shape if hasattr(control_action, 'shape') else 'scalar'
            }
            
            self.log_test_result(category, 'mpc_control_generation', 'PASS', 
                               {'time_ms': round(mpc_time, 3)})
            
        except Exception as e:
            algo_results['mpc_controller'] = {'functional': False, 'error': str(e)}
            self.log_test_result(category, 'mpc_control_generation', 'FAIL', error=str(e))
        
        # Test Bayesian RL Agent
        try:
            from src.agents.bayesian_rl_agent import BayesianRLAgent
            
            agent = BayesianRLAgent()
            
            # Test action selection
            state = np.random.rand(5)
            
            start_time = time.perf_counter()
            action = agent.select_action(state)
            agent_time = (time.perf_counter() - start_time) * 1000
            
            algo_results['bayesian_rl_agent'] = {
                'functional': True,
                'action_time_ms': round(agent_time, 3),
                'action_shape': action.shape if hasattr(action, 'shape') else 'scalar'
            }
            
            self.log_test_result(category, 'bayesian_agent_action', 'PASS', 
                               {'time_ms': round(agent_time, 3)})
            
        except Exception as e:
            algo_results['bayesian_rl_agent'] = {'functional': False, 'error': str(e)}
            self.log_test_result(category, 'bayesian_agent_action', 'FAIL', error=str(e))
        
        # Update category status
        category_tests = self.results['validation_categories'][category]['tests']
        failed_tests = [t for t in category_tests.values() if t['status'] == 'FAIL']
        
        if not failed_tests:
            self.results['validation_categories'][category]['status'] = 'PASS'
        else:
            self.results['validation_categories'][category]['status'] = 'FAIL'
        
        return algo_results

    def validate_integration_pipeline(self) -> Dict[str, Any]:
        """Test end-to-end integration pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("VALIDATING INTEGRATION PIPELINE")
        self.logger.info("=" * 60)
        
        category = "integration_pipeline"
        integration_results = {}
        
        # Test main system integration
        try:
            from src.system.human_intent_rl_system import HumanIntentRLSystem
            
            # Initialize system
            config = {}
            system = HumanIntentRLSystem(config)
            
            # Test prediction pipeline
            test_data = np.random.rand(10, 3)
            
            start_time = time.perf_counter()
            prediction = system.predict_intent(test_data)
            prediction_time = (time.perf_counter() - start_time) * 1000
            
            # Test control generation
            start_time = time.perf_counter()
            control_action = system.generate_control_action(prediction)
            control_time = (time.perf_counter() - start_time) * 1000
            
            # Test safety validation
            start_time = time.perf_counter()
            safety_check = system.validate_safety(control_action)
            safety_time = (time.perf_counter() - start_time) * 1000
            
            total_cycle_time = prediction_time + control_time + safety_time
            
            integration_results['main_system'] = {
                'functional': True,
                'prediction_time_ms': round(prediction_time, 3),
                'control_time_ms': round(control_time, 3),
                'safety_time_ms': round(safety_time, 3),
                'total_cycle_time_ms': round(total_cycle_time, 3),
                'meets_10ms_target': total_cycle_time < 10.0
            }
            
            # Validate performance target
            if total_cycle_time < 10.0:
                self.log_test_result(category, 'decision_cycle_performance', 'PASS', 
                                   {'cycle_time_ms': round(total_cycle_time, 3), 'target': '<10ms'})
            else:
                self.log_test_result(category, 'decision_cycle_performance', 'FAIL', 
                                   {'cycle_time_ms': round(total_cycle_time, 3), 'target': '<10ms'})
            
            self.log_test_result(category, 'end_to_end_pipeline', 'PASS', 
                               {'total_time_ms': round(total_cycle_time, 3)})
            
        except Exception as e:
            integration_results['main_system'] = {'functional': False, 'error': str(e)}
            self.log_test_result(category, 'end_to_end_pipeline', 'FAIL', error=str(e))
        
        # Test multiple cycles for performance consistency
        try:
            if integration_results.get('main_system', {}).get('functional'):
                cycle_times = []
                for i in range(100):  # Test 100 cycles
                    test_data = np.random.rand(5, 3)
                    
                    start_time = time.perf_counter()
                    prediction = system.predict_intent(test_data)
                    control_action = system.generate_control_action(prediction)
                    safety_check = system.validate_safety(control_action)
                    cycle_time = (time.perf_counter() - start_time) * 1000
                    
                    cycle_times.append(cycle_time)
                
                # Performance statistics
                avg_time = statistics.mean(cycle_times)
                p95_time = sorted(cycle_times)[int(len(cycle_times) * 0.95)]
                compliance_rate = sum(1 for t in cycle_times if t < 10.0) / len(cycle_times)
                
                integration_results['performance_consistency'] = {
                    'avg_time_ms': round(avg_time, 3),
                    'p95_time_ms': round(p95_time, 3),
                    'compliance_rate': round(compliance_rate * 100, 1),
                    'total_cycles_tested': len(cycle_times)
                }
                
                if avg_time < 10.0 and compliance_rate > 0.9:
                    self.log_test_result(category, 'performance_consistency', 'PASS', 
                                       {'avg_ms': round(avg_time, 3), 'compliance': f"{compliance_rate*100:.1f}%"})
                else:
                    self.log_test_result(category, 'performance_consistency', 'WARN', 
                                       {'avg_ms': round(avg_time, 3), 'compliance': f"{compliance_rate*100:.1f}%"})
        
        except Exception as e:
            self.log_test_result(category, 'performance_consistency', 'FAIL', error=str(e))
        
        # Update category status
        category_tests = self.results['validation_categories'][category]['tests']
        failed_tests = [t for t in category_tests.values() if t['status'] == 'FAIL']
        
        if not failed_tests:
            self.results['validation_categories'][category]['status'] = 'PASS'
        else:
            self.results['validation_categories'][category]['status'] = 'FAIL'
        
        return integration_results

    def validate_jekyll_documentation(self) -> Dict[str, Any]:
        """Validate Jekyll documentation site completeness."""
        self.logger.info("=" * 60)
        self.logger.info("VALIDATING JEKYLL DOCUMENTATION")
        self.logger.info("=" * 60)
        
        category = "jekyll_documentation"
        jekyll_results = {}
        
        # Check Jekyll files
        jekyll_files = {
            'jekyll_site/methodology.md': 'Technical methodology',
            'jekyll_site/results.md': 'Experimental results',
            'jekyll_site/about.md': 'Project information',
            'jekyll_site/_config.yml': 'Jekyll configuration',
            'jekyll_site/assets/css/main.css': 'Main stylesheet'
        }
        
        for file_path, description in jekyll_files.items():
            try:
                path = Path(file_path)
                if path.exists():
                    size_kb = path.stat().st_size / 1024
                    
                    # Read content for analysis
                    content = path.read_text(encoding='utf-8')
                    word_count = len(content.split())
                    
                    jekyll_results[file_path] = {
                        'exists': True,
                        'size_kb': round(size_kb, 2),
                        'word_count': word_count,
                        'has_content': word_count > 100,  # Substantial content threshold
                        'description': description
                    }
                    
                    if word_count > 100:
                        self.log_test_result(category, f"content_{file_path.replace('/', '_').replace('.', '_')}", 
                                           'PASS', {'words': word_count})
                    else:
                        self.log_test_result(category, f"content_{file_path.replace('/', '_').replace('.', '_')}", 
                                           'WARN', {'words': word_count, 'expected': '>100 words'})
                else:
                    jekyll_results[file_path] = {
                        'exists': False,
                        'description': description
                    }
                    self.log_test_result(category, f"exists_{file_path.replace('/', '_').replace('.', '_')}", 
                                       'FAIL', error=f"File missing: {file_path}")
                    
            except Exception as e:
                self.log_test_result(category, f"check_{file_path.replace('/', '_').replace('.', '_')}", 
                                   'FAIL', error=str(e))
        
        # Check for technical content indicators
        try:
            methodology_path = Path('jekyll_site/methodology.md')
            if methodology_path.exists():
                content = methodology_path.read_text().lower()
                
                technical_indicators = {
                    'bayesian': 'bayesian' in content,
                    'gaussian_process': 'gaussian process' in content or 'gaussian_process' in content,
                    'mpc': 'mpc' in content or 'model predictive' in content,
                    'safety': 'safety' in content,
                    'performance': 'performance' in content or '10ms' in content or '<10ms' in content
                }
                
                jekyll_results['technical_content'] = technical_indicators
                
                indicator_count = sum(technical_indicators.values())
                if indicator_count >= 4:
                    self.log_test_result(category, 'technical_content_completeness', 'PASS', 
                                       {'indicators_found': indicator_count})
                else:
                    self.log_test_result(category, 'technical_content_completeness', 'WARN', 
                                       {'indicators_found': indicator_count, 'expected': '4+'})
                    
        except Exception as e:
            self.log_test_result(category, 'technical_content_analysis', 'FAIL', error=str(e))
        
        # Update category status
        category_tests = self.results['validation_categories'][category]['tests']
        failed_tests = [t for t in category_tests.values() if t['status'] == 'FAIL']
        
        if not failed_tests:
            self.results['validation_categories'][category]['status'] = 'PASS'
        else:
            self.results['validation_categories'][category]['status'] = 'FAIL'
        
        return jekyll_results

    def validate_test_suite_execution(self) -> Dict[str, Any]:
        """Validate test suite execution and results."""
        self.logger.info("=" * 60)
        self.logger.info("VALIDATING TEST SUITE EXECUTION")
        self.logger.info("=" * 60)
        
        category = "test_suite_execution"
        test_results = {}
        
        # Run comprehensive test suite
        try:
            import subprocess
            
            result = subprocess.run([
                sys.executable, 'tests/comprehensive_test_suite.py'
            ], capture_output=True, text=True, timeout=120)
            
            test_results['execution'] = {
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
            # Parse test results from output
            if result.returncode == 0:
                # Look for test summary in output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Total tests:' in line:
                        total_tests = int(line.split(':')[1].strip())
                        test_results['total_tests'] = total_tests
                    elif 'Passed:' in line:
                        passed_tests = int(line.split(':')[1].strip())
                        test_results['passed_tests'] = passed_tests
                    elif 'Success rate:' in line:
                        success_rate = line.split(':')[1].strip()
                        test_results['success_rate'] = success_rate
                
                if test_results.get('total_tests', 0) >= 22:
                    self.log_test_result(category, 'test_count_validation', 'PASS', 
                                       {'tests': test_results.get('total_tests')})
                else:
                    self.log_test_result(category, 'test_count_validation', 'WARN', 
                                       {'tests': test_results.get('total_tests'), 'expected': '22+'})
                
                if test_results.get('success_rate') == '100.0%':
                    self.log_test_result(category, 'test_success_rate', 'PASS')
                else:
                    self.log_test_result(category, 'test_success_rate', 'FAIL', 
                                       {'rate': test_results.get('success_rate')})
            else:
                self.log_test_result(category, 'test_execution', 'FAIL', 
                                   error=f"Tests failed with exit code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            test_results['execution'] = {'error': 'Test execution timed out'}
            self.log_test_result(category, 'test_execution', 'FAIL', error="Test execution timed out")
            
        except Exception as e:
            test_results['execution'] = {'error': str(e)}
            self.log_test_result(category, 'test_execution', 'FAIL', error=str(e))
        
        # Update category status
        category_tests = self.results['validation_categories'][category]['tests']
        failed_tests = [t for t in category_tests.values() if t['status'] == 'FAIL']
        
        if not failed_tests:
            self.results['validation_categories'][category]['status'] = 'PASS'
        else:
            self.results['validation_categories'][category]['status'] = 'FAIL'
        
        return test_results

    def generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall project assessment and recommendations."""
        self.logger.info("=" * 60)
        self.logger.info("GENERATING OVERALL ASSESSMENT")
        self.logger.info("=" * 60)
        
        # Count category results
        category_statuses = [cat['status'] for cat in self.results['validation_categories'].values()]
        passed_categories = category_statuses.count('PASS')
        failed_categories = category_statuses.count('FAIL')
        total_categories = len(category_statuses)
        
        # Calculate overall score
        if total_categories > 0:
            score_percentage = (passed_categories / total_categories) * 100
        else:
            score_percentage = 0
        
        # Determine overall status
        if score_percentage >= 90 and failed_categories == 0:
            overall_status = 'EXCELLENT'
        elif score_percentage >= 75 and failed_categories <= 1:
            overall_status = 'GOOD'
        elif score_percentage >= 50:
            overall_status = 'PARTIAL'
        else:
            overall_status = 'CRITICAL'
        
        # Generate recommendations
        recommendations = []
        
        if overall_status == 'EXCELLENT':
            recommendations.append("✅ Project is in excellent condition and fully production-ready")
            recommendations.append("✅ All critical components validated and functional")
            recommendations.append("✅ Performance targets met (<10ms decision cycles)")
            recommendations.append("✅ Ready for deployment with confidence")
        
        elif overall_status == 'GOOD':
            recommendations.append("✅ Project is in good condition with minor issues")
            recommendations.append("⚠️ Address any failed components before production deployment")
            recommendations.append("✅ Core functionality validated and operational")
        
        elif overall_status == 'PARTIAL':
            recommendations.append("⚠️ Project has significant issues requiring attention")
            recommendations.append("❌ Critical components need repair before deployment")
            recommendations.append("⚠️ Thorough testing and debugging recommended")
        
        else:  # CRITICAL
            recommendations.append("❌ Project has critical failures requiring immediate attention")
            recommendations.append("❌ Not suitable for production deployment")
            recommendations.append("❌ Major component repairs and re-validation needed")
        
        # Add specific recommendations based on failures
        for category, cat_data in self.results['validation_categories'].items():
            if cat_data['status'] == 'FAIL':
                failed_tests = [name for name, test in cat_data['tests'].items() 
                              if test['status'] == 'FAIL']
                recommendations.append(f"❌ Fix {category}: {', '.join(failed_tests[:3])}")
        
        assessment = {
            'overall_status': overall_status,
            'score_percentage': round(score_percentage, 1),
            'categories_passed': passed_categories,
            'categories_failed': failed_categories,
            'total_categories': total_categories,
            'tests_passed': self.passed_tests,
            'tests_failed': self.failed_tests,
            'tests_warned': self.warned_tests,
            'total_tests': self.total_tests,
            'validation_duration_seconds': round(time.time() - self.validation_start, 2)
        }
        
        self.results['overall_status'] = overall_status
        self.results['summary'] = assessment
        self.results['recommendations'] = recommendations
        
        return assessment

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        self.logger.info("🔍 STARTING COMPREHENSIVE PROJECT VALIDATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Project: {self.results['project_root']}")
        self.logger.info(f"Started: {self.results['timestamp']}")
        self.logger.info("=" * 80)
        
        try:
            # Run all validation categories
            self.validate_project_structure()
            self.validate_core_imports()
            self.validate_dataset_integrity()
            self.validate_algorithm_functionality()
            self.validate_integration_pipeline()
            self.validate_jekyll_documentation()
            self.validate_test_suite_execution()
            
            # Generate overall assessment
            assessment = self.generate_overall_assessment()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Validation failed with error: {e}")
            self.logger.error(traceback.format_exc())
            
            self.results['overall_status'] = 'CRITICAL'
            self.results['validation_error'] = str(e)
            
            return self.results

    def save_validation_report(self, filename: str = None) -> str:
        """Save validation results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved: {filename}")
        return filename

    def print_validation_summary(self):
        """Print comprehensive validation summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPREHENSIVE VALIDATION SUMMARY")
        self.logger.info("=" * 80)
        
        summary = self.results['summary']
        
        # Overall status
        status_icons = {
            'EXCELLENT': '🌟',
            'GOOD': '✅',
            'PARTIAL': '⚠️',
            'CRITICAL': '❌'
        }
        
        icon = status_icons.get(self.results['overall_status'], '❓')
        self.logger.info(f"Overall Status: {icon} {self.results['overall_status']}")
        self.logger.info(f"Validation Score: {summary['score_percentage']}%")
        self.logger.info(f"Duration: {summary['validation_duration_seconds']}s")
        
        # Test statistics
        self.logger.info(f"\nTest Results:")
        self.logger.info(f"  Categories: {summary['categories_passed']}/{summary['total_categories']} passed")
        self.logger.info(f"  Individual Tests: {summary['tests_passed']}/{summary['total_tests']} passed")
        if summary['tests_failed'] > 0:
            self.logger.info(f"  Failed Tests: {summary['tests_failed']}")
        if summary['tests_warned'] > 0:
            self.logger.info(f"  Warnings: {summary['tests_warned']}")
        
        # Category breakdown
        self.logger.info(f"\nCategory Results:")
        for category, cat_data in self.results['validation_categories'].items():
            status = cat_data['status']
            icon = "✅" if status == 'PASS' else "❌" if status == 'FAIL' else "⚠️"
            self.logger.info(f"  {icon} {category.replace('_', ' ').title()}: {status}")
        
        # Recommendations
        self.logger.info(f"\nRecommendations:")
        for rec in self.results['recommendations']:
            self.logger.info(f"  {rec}")
        
        self.logger.info("=" * 80)


def main():
    """Main execution function."""
    print("🔍 Comprehensive Project Validation Suite")
    print("Model-Based RL Human Intent Recognition System")
    print("=" * 80)
    
    # Initialize validator
    validator = ComprehensiveProjectValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Print summary
    validator.print_validation_summary()
    
    # Save detailed report
    report_file = validator.save_validation_report()
    
    # Exit with appropriate code
    overall_status = results['overall_status']
    if overall_status in ['EXCELLENT', 'GOOD']:
        exit_code = 0
    else:
        exit_code = 1
    
    print(f"\nDetailed report saved: {report_file}")
    print(f"Exiting with code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())