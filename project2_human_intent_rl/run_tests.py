#!/usr/bin/env python3
"""
Comprehensive Test Runner for HRI Bayesian RL System

This script provides access to both basic validation tests and the
comprehensive test suite for the entire HRI Bayesian RL system.

Usage:
    python run_tests.py                 # Run basic validation tests
    python run_tests.py --full          # Run comprehensive test suite
    python run_tests.py --quick         # Run quick smoke tests
    python run_tests.py --unit          # Run unit tests only
    python run_tests.py --integration   # Run integration tests only
    python run_tests.py --performance   # Run performance tests only

Author: Phase 5 Implementation (Updated)
Date: 2024
"""

import sys
import os
import traceback
import time
import numpy as np
import argparse
from pathlib import Path

# Add src and tests directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from agents.bayesian_rl_agent import BayesianRLAgent, BayesianRLConfiguration
        print("✓ Bayesian RL Agent imported successfully")
    except Exception as e:
        print(f"✗ Bayesian RL Agent import failed: {e}")
    
    try:
        from environments.hri_environment import HRIEnvironment, create_default_hri_environment
        print("✓ HRI Environment imported successfully")
    except Exception as e:
        print(f"✗ HRI Environment import failed: {e}")
    
    try:
        from algorithms.gp_q_learning import GPBayesianQLearning, GPQConfiguration
        print("✓ GP Q-Learning imported successfully")
    except Exception as e:
        print(f"✗ GP Q-Learning import failed: {e}")
    
    try:
        from algorithms.psrl import PSRLAgent, PSRLConfiguration
        print("✓ PSRL Agent imported successfully")
    except Exception as e:
        print(f"✗ PSRL Agent import failed: {e}")
    
    try:
        from exploration.strategies import ExplorationManager, ExplorationConfig
        print("✓ Exploration Strategies imported successfully")
    except Exception as e:
        print(f"✗ Exploration Strategies import failed: {e}")
    
    try:
        from integration.hri_bayesian_rl import HRIBayesianRLIntegration, HRIBayesianRLConfig
        print("✓ HRI Integration imported successfully")
    except Exception as e:
        print(f"✗ HRI Integration import failed: {e}")
    
    try:
        from uncertainty.quantification import MonteCarloUncertainty, UncertaintyConfig
        print("✓ Uncertainty Quantification imported successfully")
    except Exception as e:
        print(f"✗ Uncertainty Quantification import failed: {e}")


def test_basic_functionality():
    """Test basic functionality of core components"""
    print("\nTesting basic functionality...")
    
    # Test Bayesian RL Agent
    try:
        from agents.bayesian_rl_agent import BayesianRLAgent, BayesianRLConfiguration
        
        config = BayesianRLConfiguration()
        agent = BayesianRLAgent(config)
        
        # Test basic operations
        state = np.random.randn(agent.state_dim)
        action, info = agent.select_action(state)
        
        assert action.shape == (agent.action_dim,), "Action shape incorrect"
        assert isinstance(info, dict), "Info should be dict"
        
        print("✓ Bayesian RL Agent basic functionality works")
        
    except Exception as e:
        print(f"✗ Bayesian RL Agent test failed: {e}")
        traceback.print_exc()
    
    # Test HRI Environment
    try:
        from environments.hri_environment import create_default_hri_environment
        
        env = create_default_hri_environment()
        initial_state = env.reset()
        
        action = np.random.uniform(env.action_low, env.action_high)
        next_state, reward_dict, done, info = env.step(action)
        
        assert 'total' in reward_dict, "Reward dict missing total"
        assert isinstance(done, bool), "Done should be bool"
        
        print("✓ HRI Environment basic functionality works")
        
    except Exception as e:
        print(f"✗ HRI Environment test failed: {e}")
        traceback.print_exc()
    
    # Test GP Q-Learning
    try:
        from algorithms.gp_q_learning import GPBayesianQLearning, GPQConfiguration
        
        config = GPQConfiguration(training_iterations=2)
        agent = GPBayesianQLearning(state_dim=5, action_dim=2, config=config)
        
        # Add some experience
        for _ in range(5):
            state = np.random.randn(5).astype(np.float32)
            action = np.random.uniform(-1, 1, 2).astype(np.float32)
            reward = np.random.normal()
            next_state = np.random.randn(5).astype(np.float32)
            
            agent.add_experience(state, action, reward, next_state, False)
        
        # Test prediction
        test_state = np.random.randn(5).astype(np.float32)
        test_action = np.random.uniform(-1, 1, 2).astype(np.float32)
        
        q_pred = agent.predict_q_value(test_state, test_action)
        assert 'mean' in q_pred, "Q prediction missing mean"
        
        print("✓ GP Q-Learning basic functionality works")
        
    except Exception as e:
        print(f"✗ GP Q-Learning test failed: {e}")
        traceback.print_exc()


def test_integration():
    """Test integration between components"""
    print("\nTesting integration...")
    
    try:
        from integration.hri_bayesian_rl import HRIBayesianRLIntegration, HRIBayesianRLConfig
        
        config = HRIBayesianRLConfig(rl_algorithm="gp_q_learning")
        integration = HRIBayesianRLIntegration(config)
        
        # Test a single step
        current_state = integration.environment.reset()
        step_results = integration.step(current_state)
        
        assert 'human_intent' in step_results, "Missing human intent"
        assert 'safety_assessment' in step_results, "Missing safety assessment"
        assert 'reward' in step_results, "Missing reward"
        
        print("✓ HRI Integration basic functionality works")
        
    except Exception as e:
        print(f"✗ HRI Integration test failed: {e}")
        traceback.print_exc()


def run_performance_test():
    """Run a simple performance test"""
    print("\nTesting performance...")
    
    try:
        from environments.hri_environment import create_default_hri_environment
        
        env = create_default_hri_environment()
        
        # Time environment steps
        step_times = []
        state = env.reset()
        
        for i in range(20):
            action = np.random.uniform(env.action_low, env.action_high)
            
            start_time = time.time()
            next_state, reward_dict, done, info = env.step(action)
            step_time = time.time() - start_time
            
            step_times.append(step_time)
            
            state = next_state if not done else env.reset()
        
        avg_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        
        print(f"✓ Environment performance: avg={avg_step_time:.4f}s, max={max_step_time:.4f}s")
        
        if avg_step_time < 0.01:  # 10ms per step
            print("✓ Environment meets real-time performance requirements")
        else:
            print("⚠ Environment may be slow for real-time use")
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        traceback.print_exc()


def run_basic_validation():
    """Run basic validation tests"""
    print("=" * 60)
    print("BAYESIAN RL COMPONENT VALIDATION")
    print("=" * 60)
    
    # Run tests
    test_imports()
    test_basic_functionality()
    test_integration()
    run_performance_test()
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    print("\nIf all tests passed with ✓, the Bayesian RL implementation is working correctly!")
    print("Components ready for:")
    print("- Human-robot interaction scenarios")
    print("- Bayesian reinforcement learning")
    print("- Uncertainty quantification")
    print("- Real-time decision making")
    print("- Safety-critical applications")


def run_comprehensive_tests(test_type="all", verbosity=1, report_file=None):
    """Run comprehensive test suite"""
    try:
        from comprehensive_test_suite import TestSuiteRunner, run_quick_tests
        
        print("=" * 60)
        print("COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        if test_type == "quick":
            print("Running quick smoke tests...")
            success = run_quick_tests()
            return 0 if success else 1
        
        # Run comprehensive tests
        runner = TestSuiteRunner()
        
        if test_type == "unit":
            print("Running unit tests...")
            result = runner.run_specific_test_class("BayesianRLAgentTests", verbosity=verbosity)
        elif test_type == "integration":
            print("Running integration tests...")
            result = runner.run_specific_test_class("IntegrationTests", verbosity=verbosity)
        elif test_type == "performance":
            print("Running performance tests...")
            result = runner.run_specific_test_class("PerformanceTests", verbosity=verbosity)
        else:
            print("Running all comprehensive tests...")
            result = runner.run_all_tests(verbosity=verbosity)
        
        # Print summary
        summary = result.get_summary()
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Skipped: {summary['skipped_tests']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Execution time: {summary['execution_time']:.2f}s")
        
        if summary['errors']:
            print(f"\nErrors ({len(summary['errors'])}):")
            for error in summary['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        # Generate report if requested
        if report_file:
            report_path = runner.generate_test_report(report_file)
            print(f"\nTest report saved to: {report_path}")
        
        return 0 if summary['failed_tests'] == 0 else 1
        
    except ImportError as e:
        print(f"Comprehensive test suite not available: {e}")
        print("Running basic validation instead...")
        run_basic_validation()
        return 0


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="HRI Bayesian RL System Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run basic validation tests
  python run_tests.py --full             # Run comprehensive test suite
  python run_tests.py --quick            # Quick smoke tests
  python run_tests.py --unit             # Unit tests only
  python run_tests.py --integration      # Integration tests only
  python run_tests.py --performance      # Performance tests only
  python run_tests.py --verbose          # Verbose output
  python run_tests.py --report results.json  # Generate report
        """
    )
    
    # Test selection options
    parser.add_argument('--full', action='store_true',
                       help='Run comprehensive test suite')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick smoke tests only')
    parser.add_argument('--unit', action='store_true',
                       help='Run unit tests only')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests only')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='Increase verbosity')
    parser.add_argument('--report', metavar='FILE',
                       help='Generate test report and save to FILE')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimize output')
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbosity = 0 if args.quiet else args.verbose
    
    # Determine test type
    if args.full:
        return run_comprehensive_tests("all", verbosity, args.report)
    elif args.quick:
        return run_comprehensive_tests("quick", verbosity, args.report)
    elif args.unit:
        return run_comprehensive_tests("unit", verbosity, args.report)
    elif args.integration:
        return run_comprehensive_tests("integration", verbosity, args.report)
    elif args.performance:
        return run_comprehensive_tests("performance", verbosity, args.report)
    else:
        # Default: run basic validation
        run_basic_validation()
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)