#!/usr/bin/env python3
import sys

def test_imports():
    failures = []
    
    try:
        from src.models.gaussian_process import GaussianProcess
        gp = GaussianProcess()
        print("GP: Import successful")
    except Exception as e:
        failures.append(f"GP failed: {e}")
    
    try:
        from src.controllers.mpc_controller import MPCController
        mpc = MPCController()
        print("MPC: Import successful")
    except Exception as e:
        failures.append(f"MPC failed: {e}")
    
    try:
        from src.agents.bayesian_rl_agent import BayesianRLAgent
        agent = BayesianRLAgent()
        print("RL: Import successful")
    except Exception as e:
        failures.append(f"RL failed: {e}")
    
    return failures

if __name__ == "__main__":
    failures = test_imports()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"- {f}")
        print(f"\nREALITY: {len(failures)} components don't work")
    else:
        print("All components can be imported")