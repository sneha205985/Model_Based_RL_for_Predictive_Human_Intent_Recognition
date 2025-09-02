#!/usr/bin/env python3
"""
Production Deployment Guide & Validation
Model-Based RL Human Intent Recognition System

Complete guide for deploying and validating the system in production
environment with real-time <10ms performance validation.

Author: Production Team
Date: September 2025
"""

import subprocess
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class ProductionDeploymentValidator:
    """
    Comprehensive production deployment validator that ensures
    the system meets all performance and reliability requirements.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'deployment_validation': {},
            'performance_validation': {},
            'production_readiness': {}
        }

    def validate_docker_environment(self) -> Dict[str, Any]:
        """Validate Docker environment setup."""
        self.logger.info("Validating Docker environment...")
        
        validations = {}
        
        # Check Docker installation
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                validations['docker_installed'] = {
                    'status': 'PASS',
                    'version': result.stdout.strip()
                }
            else:
                validations['docker_installed'] = {
                    'status': 'FAIL',
                    'error': 'Docker not found'
                }
        except Exception as e:
            validations['docker_installed'] = {
                'status': 'FAIL',
                'error': str(e)
            }
        
        # Check Docker Compose
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                validations['docker_compose'] = {
                    'status': 'PASS',
                    'version': result.stdout.strip()
                }
            else:
                validations['docker_compose'] = {
                    'status': 'FAIL',
                    'error': 'Docker Compose not found'
                }
        except Exception as e:
            validations['docker_compose'] = {
                'status': 'FAIL',
                'error': str(e)
            }
        
        # Validate Dockerfile exists and is valid
        dockerfile_path = Path('Dockerfile')
        if dockerfile_path.exists():
            validations['dockerfile_exists'] = {'status': 'PASS'}
            
            # Basic Dockerfile validation
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
                
            required_elements = [
                'FROM python',
                'COPY requirements.txt',
                'RUN pip install',
                'WORKDIR /app',
                'HEALTHCHECK'
            ]
            
            missing_elements = [elem for elem in required_elements 
                              if elem not in dockerfile_content]
            
            if not missing_elements:
                validations['dockerfile_valid'] = {'status': 'PASS'}
            else:
                validations['dockerfile_valid'] = {
                    'status': 'WARN',
                    'missing_elements': missing_elements
                }
        else:
            validations['dockerfile_exists'] = {
                'status': 'FAIL',
                'error': 'Dockerfile not found'
            }
        
        return validations

    def validate_system_requirements(self) -> Dict[str, Any]:
        """Validate system meets production requirements."""
        self.logger.info("Validating system requirements...")
        
        validations = {}
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 8):
            validations['python_version'] = {
                'status': 'PASS',
                'version': python_version
            }
        else:
            validations['python_version'] = {
                'status': 'FAIL',
                'version': python_version,
                'requirement': 'Python 3.8+'
            }
        
        # Check core dependencies
        try:
            import numpy, pandas, scikit_learn, matplotlib, plotly
            validations['core_dependencies'] = {'status': 'PASS'}
        except ImportError as e:
            validations['core_dependencies'] = {
                'status': 'FAIL',
                'error': f'Missing core dependency: {e}'
            }
        
        # Check optional dependencies
        optional_deps = {
            'h5py': 'Advanced data storage',
            'dash': 'Interactive dashboards',
            'bokeh': 'Interactive plotting',
            'psutil': 'System monitoring'
        }
        
        optional_status = {}
        for dep, description in optional_deps.items():
            try:
                __import__(dep)
                optional_status[dep] = {'status': 'AVAILABLE', 'description': description}
            except ImportError:
                optional_status[dep] = {'status': 'MISSING', 'description': description}
        
        validations['optional_dependencies'] = optional_status
        
        return validations

    def run_production_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive production performance validation."""
        self.logger.info("Running production performance validation...")
        
        # Check if production_benchmark.py exists
        benchmark_script = Path('production_benchmark.py')
        if not benchmark_script.exists():
            return {
                'status': 'FAIL',
                'error': 'production_benchmark.py not found'
            }
        
        try:
            # Run production benchmark
            self.logger.info("Executing production benchmark...")
            result = subprocess.run([
                sys.executable, 'production_benchmark.py'
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                # Try to load results
                try:
                    # Look for most recent benchmark results
                    metrics_dir = Path('metrics')
                    if metrics_dir.exists():
                        benchmark_files = list(metrics_dir.glob('production_benchmark_*.json'))
                        if benchmark_files:
                            latest_file = max(benchmark_files, key=lambda p: p.stat().st_mtime)
                            with open(latest_file, 'r') as f:
                                benchmark_data = json.load(f)
                            
                            # Extract key metrics
                            decision_cycles = benchmark_data.get('benchmarks', {}).get('decision_cycles', {})
                            performance_stats = decision_cycles.get('statistics', {})
                            
                            return {
                                'status': 'PASS' if decision_cycles.get('status') == 'PASS' else 'FAIL',
                                'avg_performance_ms': performance_stats.get('mean_ms', 'unknown'),
                                'compliance_rate': performance_stats.get('compliance_rate', 'unknown'),
                                'p95_performance_ms': performance_stats.get('p95_ms', 'unknown'),
                                'benchmark_file': str(latest_file),
                                'full_results': benchmark_data
                            }
                    
                    return {
                        'status': 'PASS',
                        'message': 'Benchmark completed successfully',
                        'stdout': result.stdout
                    }
                    
                except Exception as e:
                    return {
                        'status': 'WARN',
                        'message': 'Benchmark completed but results parsing failed',
                        'error': str(e),
                        'stdout': result.stdout
                    }
            else:
                return {
                    'status': 'FAIL',
                    'error': 'Benchmark execution failed',
                    'return_code': result.returncode,
                    'stderr': result.stderr,
                    'stdout': result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'FAIL',
                'error': 'Benchmark timed out (>10 minutes)'
            }
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': f'Benchmark execution error: {str(e)}'
            }

    def validate_core_functionality(self) -> Dict[str, Any]:
        """Validate core system functionality."""
        self.logger.info("Validating core functionality...")
        
        validations = {}
        
        # Check if comprehensive test suite passes
        test_script = Path('tests/comprehensive_test_suite.py')
        if test_script.exists():
            try:
                result = subprocess.run([
                    sys.executable, str(test_script)
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    validations['test_suite'] = {
                        'status': 'PASS',
                        'output': result.stdout
                    }
                else:
                    validations['test_suite'] = {
                        'status': 'FAIL',
                        'error': result.stderr,
                        'output': result.stdout
                    }
            except Exception as e:
                validations['test_suite'] = {
                    'status': 'FAIL',
                    'error': str(e)
                }
        else:
            validations['test_suite'] = {
                'status': 'FAIL',
                'error': 'Test suite not found'
            }
        
        # Check project health
        health_script = Path('project_health_check.py')
        if health_script.exists():
            try:
                result = subprocess.run([
                    sys.executable, str(health_script)
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    validations['health_check'] = {
                        'status': 'PASS',
                        'output': result.stdout
                    }
                else:
                    validations['health_check'] = {
                        'status': 'WARN',
                        'output': result.stdout,
                        'error': result.stderr
                    }
            except Exception as e:
                validations['health_check'] = {
                    'status': 'FAIL',
                    'error': str(e)
                }
        
        return validations

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete production deployment validation."""
        self.logger.info("Starting comprehensive production deployment validation")
        
        validation_start = time.time()
        
        # Run all validations
        self.validation_results['deployment_validation']['docker_environment'] = \
            self.validate_docker_environment()
        
        self.validation_results['deployment_validation']['system_requirements'] = \
            self.validate_system_requirements()
        
        self.validation_results['deployment_validation']['core_functionality'] = \
            self.validate_core_functionality()
        
        self.validation_results['performance_validation'] = \
            self.run_production_performance_test()
        
        # Overall assessment
        validation_duration = time.time() - validation_start
        self.validation_results['validation_duration_seconds'] = round(validation_duration, 2)
        
        # Determine production readiness
        self.validation_results['production_readiness'] = self._assess_production_readiness()
        
        self.logger.info(f"Validation completed in {validation_duration:.1f} seconds")
        
        return self.validation_results

    def _assess_production_readiness(self) -> Dict[str, Any]:
        """Assess overall production readiness."""
        deployment_issues = []
        performance_issues = []
        warnings = []
        
        # Check deployment validations
        deploy_val = self.validation_results['deployment_validation']
        
        # Docker environment
        docker_env = deploy_val.get('docker_environment', {})
        if docker_env.get('docker_installed', {}).get('status') != 'PASS':
            deployment_issues.append('Docker not available')
        if docker_env.get('docker_compose', {}).get('status') != 'PASS':
            warnings.append('Docker Compose not available')
        
        # System requirements
        sys_req = deploy_val.get('system_requirements', {})
        if sys_req.get('python_version', {}).get('status') != 'PASS':
            deployment_issues.append('Python version incompatible')
        if sys_req.get('core_dependencies', {}).get('status') != 'PASS':
            deployment_issues.append('Missing core dependencies')
        
        # Core functionality
        core_func = deploy_val.get('core_functionality', {})
        if core_func.get('test_suite', {}).get('status') != 'PASS':
            deployment_issues.append('Test suite failing')
        
        # Performance validation
        perf_val = self.validation_results.get('performance_validation', {})
        if perf_val.get('status') != 'PASS':
            performance_issues.append('Performance benchmark failed')
        
        # Determine overall status
        if deployment_issues or performance_issues:
            if performance_issues:
                overall_status = 'NOT_PRODUCTION_READY'
            else:
                overall_status = 'DEPLOYMENT_ISSUES'
        else:
            overall_status = 'PRODUCTION_READY'
        
        return {
            'overall_status': overall_status,
            'deployment_issues': deployment_issues,
            'performance_issues': performance_issues,
            'warnings': warnings,
            'production_ready': overall_status == 'PRODUCTION_READY'
        }

    def generate_deployment_report(self) -> str:
        """Generate comprehensive deployment report."""
        report = []
        
        report.append("=" * 70)
        report.append("PRODUCTION DEPLOYMENT VALIDATION REPORT")
        report.append("Model-Based RL Human Intent Recognition System")
        report.append("=" * 70)
        
        # Overall status
        readiness = self.validation_results['production_readiness']
        report.append(f"Overall Status: {readiness['overall_status']}")
        report.append(f"Production Ready: {readiness['production_ready']}")
        report.append("")
        
        # Performance summary
        perf_val = self.validation_results.get('performance_validation', {})
        if perf_val.get('status') == 'PASS':
            avg_ms = perf_val.get('avg_performance_ms', 'unknown')
            compliance = perf_val.get('compliance_rate', 'unknown')
            report.append(f"✅ Performance: {avg_ms}ms average ({compliance} compliant)")
        else:
            report.append(f"❌ Performance: {perf_val.get('error', 'Failed')}")
        report.append("")
        
        # Deployment validation summary
        report.append("DEPLOYMENT VALIDATION:")
        deploy_val = self.validation_results['deployment_validation']
        
        for category, validations in deploy_val.items():
            report.append(f"  {category.replace('_', ' ').title()}:")
            
            if isinstance(validations, dict):
                for check, result in validations.items():
                    if isinstance(result, dict):
                        status = result.get('status', 'UNKNOWN')
                        icon = "✅" if status == 'PASS' else "⚠️" if status == 'WARN' else "❌"
                        report.append(f"    {icon} {check}: {status}")
                        
                        if status == 'FAIL' and 'error' in result:
                            report.append(f"      Error: {result['error']}")
        
        report.append("")
        
        # Issues and warnings
        if readiness['deployment_issues']:
            report.append("DEPLOYMENT ISSUES:")
            for issue in readiness['deployment_issues']:
                report.append(f"  ❌ {issue}")
            report.append("")
        
        if readiness['performance_issues']:
            report.append("PERFORMANCE ISSUES:")
            for issue in readiness['performance_issues']:
                report.append(f"  ❌ {issue}")
            report.append("")
        
        if readiness['warnings']:
            report.append("WARNINGS:")
            for warning in readiness['warnings']:
                report.append(f"  ⚠️ {warning}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if readiness['production_ready']:
            report.append("  ✅ System is ready for production deployment")
            report.append("  ✅ Use: docker-compose up -d to deploy")
            report.append("  ✅ Monitor performance with: monitoring/performance_monitor.py")
        else:
            report.append("  ❌ Address all deployment and performance issues before production")
            if readiness['deployment_issues']:
                report.append("  ❌ Fix deployment environment issues")
            if readiness['performance_issues']:
                report.append("  ❌ Resolve performance benchmark failures")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)

    def save_validation_results(self, filename: str = None):
        """Save validation results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"production_validation_{timestamp}.json"
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        self.logger.info(f"Validation results saved to {filename}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Production Deployment Validator")
    print("Model-Based RL Human Intent Recognition System")
    print("=" * 70)
    
    validator = ProductionDeploymentValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Generate and display report
    report = validator.generate_deployment_report()
    print(report)
    
    # Save results
    validator.save_validation_results()
    
    # Exit with appropriate code
    production_ready = results['production_readiness']['production_ready']
    exit_code = 0 if production_ready else 1
    print(f"Exiting with code: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()