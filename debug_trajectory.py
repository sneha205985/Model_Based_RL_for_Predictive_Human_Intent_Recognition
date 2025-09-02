#!/usr/bin/env python3
"""Debug trajectory prediction issue"""

import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('src/models')

from gaussian_process import GaussianProcess

# Load test data
data = pd.read_csv("data/synthetic_full/features.csv")
X_data = data.iloc[:50, :6].values  
y_data = data.iloc[:50, 6:8].values  # Use 2D output to match working test  

print(f"Data shapes: X={X_data.shape}, y={y_data.shape}")

# Train GP
gp = GaussianProcess()
gp.fit(X_data, y_data)

print("Testing trajectory prediction with full method...")

try:
    result = gp.predict_trajectory(X_data[0], n_steps=3)
    print(f"✅ Trajectory prediction successful: {result.shape}")
except Exception as e:
    print(f"❌ Trajectory prediction failed: {e}")
    import traceback
    traceback.print_exc()