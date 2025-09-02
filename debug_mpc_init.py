#!/usr/bin/env python3
"""Debug MPC controller initialization issue"""

import numpy as np
import sys
sys.path.append('src/controllers')

print("Testing MPC controller initialization...")

try:
    from mpc_controller import MPCController
    
    print("Creating MPC controller...")
    mpc = MPCController(prediction_horizon=10, control_horizon=5, dt=0.1)
    print("✅ MPC controller created successfully")
    
except Exception as e:
    print(f"❌ MPC controller creation failed: {e}")
    import traceback
    traceback.print_exc()