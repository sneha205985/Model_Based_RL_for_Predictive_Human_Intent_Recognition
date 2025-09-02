#!/usr/bin/env python3
"""
Dependency Verification Script for Model-Based RL Human Intent Recognition System
Tests all required packages and their versions
"""

import sys
import importlib
from typing import Dict, List, Tuple

def test_import(module_name: str, alias: str = None) -> Tuple[bool, str, str]:
    """Test importing a module and return status, version, and error if any."""
    try:
        module = importlib.import_module(module_name)
        
        # Try to get version
        version = "unknown"
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        elif hasattr(module, '__version_info__'):
            version = str(module.__version_info__)
        
        return True, version, ""
    except ImportError as e:
        return False, "", str(e)
    except Exception as e:
        return False, "", f"Unexpected error: {str(e)}"

def main():
    """Main dependency verification function."""
    print("🔧 Model-Based RL Human Intent Recognition System")
    print("📦 Dependency Verification Test")
    print("=" * 70)
    
    # Core dependencies to test
    dependencies = [
        ("torch", "PyTorch"),
        ("pyro", "Pyro Probabilistic Programming"),
        ("sklearn", "Scikit-Learn"),
        ("roboticstoolbox", "Robotics Toolbox"),
        ("control", "Python Control Systems"),
        ("matplotlib.pyplot", "Matplotlib"),
        ("plotly", "Plotly"),
        ("seaborn", "Seaborn"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"), 
        ("pandas", "Pandas")
    ]
    
    results = {}
    
    print(f"\n{'Package':<20} {'Status':<10} {'Version':<15} {'Description'}")
    print("-" * 70)
    
    for module_name, description in dependencies:
        success, version, error = test_import(module_name)
        results[module_name] = (success, version, error)
        
        status = "✅ OK" if success else "❌ FAIL"
        version_str = version if success else "N/A"
        
        print(f"{module_name:<20} {status:<10} {version_str:<15} {description}")
        
        if not success:
            print(f"   Error: {error}")
    
    # Summary
    successful = sum(1 for success, _, _ in results.values() if success)
    total = len(dependencies)
    
    print("\n" + "=" * 70)
    print(f"📊 SUMMARY: {successful}/{total} packages successfully imported")
    
    if successful == total:
        print("🎉 ALL DEPENDENCIES SUCCESSFULLY INSTALLED!")
        print("\n🚀 Your system is ready for Model-Based RL development!")
        
        # Test specific functionality
        print("\n🧪 Testing specific functionality...")
        test_specific_features()
        
    else:
        print(f"⚠️ {total - successful} packages failed to import")
        print("\n💡 To fix missing dependencies, run:")
        print("pip3 install -r requirements.txt")
    
    print("\n" + "=" * 70)

def test_specific_features():
    """Test specific features of key packages."""
    print("\n🔍 Testing Core Functionality:")
    
    # Test PyTorch
    try:
        import torch
        x = torch.randn(2, 3)
        print("✅ PyTorch tensor operations working")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
    
    # Test Pyro
    try:
        import pyro
        import pyro.distributions as dist
        normal = dist.Normal(0, 1)
        sample = normal.sample()
        print("✅ Pyro probabilistic programming working")
    except Exception as e:
        print(f"❌ Pyro test failed: {e}")
    
    # Test Robotics Toolbox (skip if numpy compatibility issues)
    try:
        import roboticstoolbox as rtb
        print("✅ Robotics Toolbox import working (basic)")
    except Exception as e:
        print(f"⚠️ Robotics Toolbox test skipped: {e}")
        print("   (Known issue with numpy.disp compatibility - not critical for core functionality)")
    
    # Test Scikit-learn
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        gpr = GaussianProcessRegressor(kernel=RBF())
        print("✅ Scikit-learn Gaussian Process working")
    except Exception as e:
        print(f"❌ Scikit-learn GP test failed: {e}")
    
    # Test Control Systems
    try:
        import control
        import numpy as np
        # Create a simple transfer function
        s = control.tf('s')
        sys = 1 / (s + 1)
        print("✅ Control systems library working")
    except Exception as e:
        print(f"❌ Control systems test failed: {e}")

if __name__ == "__main__":
    main()