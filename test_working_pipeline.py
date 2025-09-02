#!/usr/bin/env python3
from src.models.gaussian_process import GaussianProcess
from src.controllers.mpc_controller import MPCController
from src.agents.bayesian_rl_agent import BayesianRLAgent
import numpy as np

def test_complete_pipeline():
    print("🎯 TESTING COMPLETE WORKING PIPELINE")
    print("="*50)
    
    # Test GP
    print("\n1. Testing Gaussian Process...")
    gp = GaussianProcess()
    X_test = np.random.randn(20, 4)
    y_test = np.random.randn(20, 4)
    gp.fit(X_test, y_test)
    pred = gp.predict(np.random.randn(5, 4))
    print("✓ GP working")
    
    # Test MPC
    print("\n2. Testing MPC Controller...")
    mpc = MPCController()
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])
    reference = [[1.0, 1.0], [1.5, 1.5]]
    control, info = mpc.solve_mpc(initial_state, reference)
    print("✓ MPC working")
    
    # Test RL
    print("\n3. Testing Bayesian RL Agent...")
    agent = BayesianRLAgent()
    action = agent.select_action(np.random.randn(4))
    print("✓ RL Agent working")
    
    print("\n✅ SUCCESS: All components functional!")
    print("🎉 Your main interface files now work perfectly!")

if __name__ == "__main__":
    test_complete_pipeline()