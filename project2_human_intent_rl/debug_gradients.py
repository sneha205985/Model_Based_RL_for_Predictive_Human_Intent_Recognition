#!/usr/bin/env python3
"""Debug gradient issues in production GP"""

import torch
import numpy as np
import sys
sys.path.append('src/models')

from gaussian_process import GaussianProcess, HumanMotionKernelOptimized

# Create small test case
X_train = torch.randn(10, 3, dtype=torch.float32)
y_train = torch.randn(10, 2, dtype=torch.float32)

print("Creating kernel directly...")
kernel = HumanMotionKernelOptimized(3)

print("Checking parameter gradients:")
for name, param in kernel.named_parameters():
    print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")

print("\nTesting kernel computation...")
K = kernel(X_train, X_train)
print(f"K shape: {K.shape}, requires_grad: {K.requires_grad}")

print("\nTesting backward...")
loss = K.sum()
try:
    loss.backward()
    print("✅ Backward successful")
    
    print("\nChecking gradients after backward:")
    for name, param in kernel.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")
        else:
            print(f"  {name}: grad=None")
            
except Exception as e:
    print(f"❌ Backward failed: {e}")