#!/usr/bin/env python3
"""Debug uncertainty calibration tensor shapes"""

import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('src/models')

from gaussian_process import GaussianProcess

# Load data exactly as in the test
data = pd.read_csv("data/synthetic_full/features.csv")
X_data = data.iloc[:100, :6].values  
y_data = data.iloc[:100, 6:10].values

X_train, X_test = X_data[:80], X_data[80:]
y_train, y_test = y_data[:80], y_data[80:]

print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test shapes: X_test={X_test.shape}, y_test={y_test.shape}")

# Train GP
gp = GaussianProcess()
gp.fit(X_train, y_train)

# Get predictions
predictions, uncertainties = gp.predict(X_test, return_std=True)
print(f"Prediction shapes: pred={predictions.shape}, unc={uncertainties.shape}")

# Test the actual calibration method
print("Testing uncertainty calibration...")
try:
    metrics = gp.calibrate_uncertainties(X_test, y_test)
    print(f"✅ Calibration successful!")
    print(f"   ECE: {metrics['ece']:.4f}")
    print(f"   MCE: {metrics['mce']:.4f}")
except Exception as e:
    print(f"❌ Calibration failed: {e}")
    import traceback
    traceback.print_exc()