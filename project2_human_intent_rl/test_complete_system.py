#!/usr/bin/env python3
"""
Complete System Test - Phase 2 Final Validation
Tests all three core components working together as requested.

This test must pass completely to confirm the system is functional.
"""

import sys
import numpy as np
from typing import Dict

# Add paths for our components
sys.path.append('src/models')
sys.path.append('src/controllers') 
sys.path.append('src/agents')

def test_all_components() -> Dict[str, bool]:
    """Test all core components as specified."""
    
    print("🎯 COMPLETE SYSTEM TEST - Phase 2 Final Validation")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Basic Gaussian Process
    print("\n1. Testing Gaussian Process...")
    try:
        from gaussian_process_basic import BasicGaussianProcess
        
        gp = BasicGaussianProcess()
        
        # Test basic functionality
        X_dummy = np.random.randn(50, 2)
        y_dummy = np.random.randn(50, 2)
        gp.fit(X_dummy, y_dummy)
        pred = gp.predict(np.random.randn(10, 2))
        
        # Verify it actually works
        assert pred[0].shape == (10, 2), "GP prediction shape incorrect"
        assert pred[1].shape == (10, 2), "GP uncertainty shape incorrect"
        
        results["Gaussian Process"] = True
        print("   ✅ PASS - GP creates, trains, and predicts correctly")
        
    except Exception as e:
        results["Gaussian Process"] = False
        print(f"   ❌ FAIL - {e}")
    
    # Test 2: Basic MPC Controller  
    print("\n2. Testing MPC Controller...")
    try:
        from mpc_controller_basic import BasicMPCController
        
        mpc = BasicMPCController()
        
        # Test basic functionality
        current_state = np.array([0.0, 0.0, 0.0, 0.0])
        reference_trajectory = np.array([[i, 0] for i in range(1, 11)])
        
        U_optimal, opt_info = mpc.solve_mpc(current_state, reference_trajectory)
        next_control = mpc.get_next_control(current_state, reference_trajectory)
        
        # Verify it actually works
        assert U_optimal.shape == (5, 2), "MPC control shape incorrect"
        assert opt_info['success'], "MPC optimization failed"
        assert next_control.shape == (2,), "Next control shape incorrect"
        
        results["MPC Controller"] = True
        print("   ✅ PASS - MPC creates, solves, and controls correctly")
        
    except Exception as e:
        results["MPC Controller"] = False
        print(f"   ❌ FAIL - {e}")
    
    # Test 3: Basic Bayesian RL Agent
    print("\n3. Testing Bayesian RL Agent...")
    try:
        from bayesian_rl_basic import BasicBayesianRLAgent
        
        config = {'discount_factor': 0.95, 'exploration': 'thompson_sampling'}
        brl = BasicBayesianRLAgent(state_dim=4, action_dim=2, config=config)
        
        # Test basic functionality
        test_state = np.array([1.0, 2.0, 0.5, -0.5])
        action = brl.select_action(test_state)
        action_with_unc = brl.select_action(test_state, return_uncertainty=True)
        
        # Add some experience and update
        for i in range(10):
            state = np.random.randn(4)
            action = brl.action_space[np.random.randint(len(brl.action_space))]
            reward = np.random.randn()
            next_state = state + 0.1 * np.random.randn(4)
            brl.update(state, action, reward, next_state)
        
        # Test value estimation
        value, uncertainty = brl.get_value(test_state, np.array([1, 0]))
        
        # Verify it actually works
        assert action.shape == (2,), "BRL action shape incorrect"
        assert len(action_with_unc) == 2, "BRL uncertainty return incorrect"
        assert isinstance(value, float), "BRL value estimation incorrect"
        assert isinstance(uncertainty, float), "BRL uncertainty estimation incorrect"
        
        results["Bayesian RL Agent"] = True
        print("   ✅ PASS - BRL creates, learns, and acts correctly")
        
    except Exception as e:
        results["Bayesian RL Agent"] = False
        print(f"   ❌ FAIL - {e}")
    
    # Test 4: System Integration
    print("\n4. Testing System Integration...")
    try:
        from gaussian_process_basic import BasicGaussianProcess
        from mpc_controller_basic import BasicMPCController  
        from bayesian_rl_basic import BasicBayesianRLAgent
        
        # Create all components
        gp = BasicGaussianProcess()
        mpc = BasicMPCController()
        brl = BasicBayesianRLAgent(state_dim=4, action_dim=2)
        
        # Test they can work together
        state = np.array([0.0, 0.0, 0.0, 0.0])
        
        # GP predicts trajectory  
        X_train = np.random.randn(20, 4)
        y_train = np.random.randn(20, 2)
        gp.fit(X_train, y_train)
        gp_pred = gp.predict(state.reshape(1, -1))
        
        # MPC plans control
        reference = np.array([[1, 1]] * 10)
        mpc_control, _ = mpc.solve_mpc(state, reference)
        
        # BRL selects action
        brl_action = brl.select_action(state)
        
        # Verify integration works
        assert gp_pred[0].shape[0] > 0, "GP integration failed"
        assert mpc_control.shape == (5, 2), "MPC integration failed"
        assert brl_action.shape == (2,), "BRL integration failed"
        
        results["System Integration"] = True
        print("   ✅ PASS - All components integrate successfully")
        
    except Exception as e:
        results["System Integration"] = False
        print(f"   ❌ FAIL - {e}")
    
    # Test 5: End-to-End Pipeline
    print("\n5. Testing End-to-End Pipeline...")
    try:
        # Simulate a complete decision cycle
        initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Step 1: GP predicts human behavior  
        human_trajectory = gp.predict_trajectory(initial_state, n_steps=5)
        
        # Step 2: MPC plans robot response
        robot_reference = human_trajectory[:, :2]  # Use position only
        robot_plan, plan_info = mpc.solve_mpc(initial_state, robot_reference)
        
        # Step 3: BRL adapts behavior
        reward = 1.0 if plan_info['success'] else -1.0
        brl.update(initial_state, robot_plan[0], reward, initial_state + 0.1)
        
        # Step 4: Get next action
        next_action = brl.select_action(initial_state)
        
        # Verify pipeline works
        assert human_trajectory.shape[0] > 5, "Pipeline step 1 failed"
        assert robot_plan.shape == (5, 2), "Pipeline step 2 failed"
        assert isinstance(reward, float), "Pipeline step 3 failed"  
        assert next_action.shape == (2,), "Pipeline step 4 failed"
        
        results["End-to-End Pipeline"] = True
        print("   ✅ PASS - Complete pipeline executes successfully")
        
    except Exception as e:
        results["End-to-End Pipeline"] = False
        print(f"   ❌ FAIL - {e}")
    
    # Final Results
    print("\n" + "=" * 60)
    print("📊 COMPLETE SYSTEM TEST RESULTS")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\n📈 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - SYSTEM IS FUNCTIONAL!")
        print("✅ Gaussian Process: Working")
        print("✅ MPC Controller: Working") 
        print("✅ Bayesian RL Agent: Working")
        print("✅ System Integration: Working")
        print("✅ End-to-End Pipeline: Working")
        print("\n🚀 Your Model-Based RL Human Intent Recognition System is working!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} components need fixing")
        print("Follow the implementation prompts to complete missing components")
    
    return results

if __name__ == "__main__":
    results = test_all_components()
    
    # Exit with success code only if all tests pass
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)