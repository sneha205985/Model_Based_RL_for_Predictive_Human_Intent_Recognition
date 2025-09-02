#!/usr/bin/env python3
"""Debug training issues in production GP"""

import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('src/models')

from gaussian_process import GaussianProcess

# Load real data for testing
data = pd.read_csv("data/synthetic_full/features.csv")
X_data = data.iloc[:50, :6].values  # Small subset
y_data = data.iloc[:50, 6:8].values  # Smaller output

print(f"Data shapes: X={X_data.shape}, y={y_data.shape}")

# Test basic training
print("\n🔧 Testing GP training...")
gp = GaussianProcess()

try:
    # Test fit method with detailed logging
    print("Starting fit...")
    result = gp.fit(X_data, y_data)
    print("✅ Training successful!")
    
    # Test prediction
    pred_mean = gp.predict(X_data[:5])
    print(f"✅ Prediction successful: {pred_mean.shape}")
    
    pred_mean, pred_std = gp.predict(X_data[:5], return_std=True)
    print(f"✅ Prediction with std successful: mean={pred_mean.shape}, std={pred_std.shape}")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()