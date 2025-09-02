#!/usr/bin/env python3
"""Debug LQR computation"""

import numpy as np
from scipy.linalg import solve_discrete_are

# System matrices
dt = 0.1
A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

B = np.array([
    [0, 0],
    [0, 0],
    [dt, 0],
    [0, dt]
])

Q = np.diag([10.0, 10.0, 1.0, 1.0])
R = np.diag([0.1, 0.1])

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"Q shape: {Q.shape}")
print(f"R shape: {R.shape}")

# Solve ARE
try:
    P = solve_discrete_are(A, B, Q, R)
    print(f"P shape: {P.shape}")
    
    # Compute LQR gain
    K_lqr = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    print(f"K_lqr shape: {K_lqr.shape}")
    
    # Check terminal invariant set constraints
    H = np.vstack([K_lqr, -K_lqr])
    print(f"H shape: {H.shape}")
    
    # Check closed-loop matrix
    A_cl = A + B @ K_lqr
    print(f"A_cl shape: {A_cl.shape}")
    
    # Test iteration 
    H_pred = H @ A_cl
    print(f"H_pred shape: {H_pred.shape}")
    
    # This should work
    H_new = np.vstack([H, H_pred])
    print(f"H_new shape: {H_new.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()