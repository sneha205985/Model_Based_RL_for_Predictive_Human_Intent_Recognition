#!/usr/bin/env python3
"""
Core Functionality Test - Test individual algorithms with dependencies
"""

import torch
import pyro
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

def test_pytorch_functionality():
    """Test PyTorch deep learning functionality."""
    print("🧠 Testing PyTorch Deep Learning:")
    
    # Create a simple neural network
    class SimpleNet(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
            self.layer2 = torch.nn.Linear(hidden_dim, output_dim)
            self.relu = torch.nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.layer2(x)
            return x
    
    # Test network
    net = SimpleNet(10, 20, 3)
    x = torch.randn(5, 10)
    y = net(x)
    
    print(f"✅ Neural network created: {net}")
    print(f"   Input shape: {x.shape}, Output shape: {y.shape}")
    return True

def test_pyro_bayesian():
    """Test Pyro Bayesian modeling."""
    print("\n🎲 Testing Pyro Bayesian Modeling:")
    
    import pyro.distributions as dist
    
    # Simple Bayesian linear regression
    def model(x_data, y_data):
        # Priors
        weight = pyro.sample("weight", dist.Normal(0, 1))
        bias = pyro.sample("bias", dist.Normal(0, 1))
        sigma = pyro.sample("sigma", dist.Uniform(0, 10))
        
        # Likelihood
        mean = weight * x_data + bias
        with pyro.plate("data", len(x_data)):
            pyro.sample("y", dist.Normal(mean, sigma), obs=y_data)
    
    # Generate synthetic data
    x_data = torch.randn(10)
    y_data = 2 * x_data + 1 + 0.1 * torch.randn(10)
    
    # Test model
    try:
        trace = pyro.poutine.trace(model).get_trace(x_data, y_data)
        print("✅ Bayesian model executed successfully")
        print(f"   Model trace sites: {list(trace.nodes.keys())}")
        return True
    except Exception as e:
        print(f"❌ Pyro test failed: {e}")
        return False

def test_gaussian_process():
    """Test Gaussian Process regression."""
    print("\n📈 Testing Gaussian Process Regression:")
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.uniform(-3, 3, (10, 1))
    y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, X_train.shape[0])
    
    # Create and fit GP
    kernel = RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
    gpr.fit(X_train, y_train)
    
    # Make predictions
    X_test = np.linspace(-4, 4, 50).reshape(-1, 1)
    y_pred, y_std = gpr.predict(X_test, return_std=True)
    
    print("✅ Gaussian Process regression working")
    print(f"   Training points: {X_train.shape[0]}")
    print(f"   Test predictions: {X_test.shape[0]}")
    print(f"   Mean prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"   Uncertainty range: [{y_std.min():.2f}, {y_std.max():.2f}]")
    
    return True, gpr, X_train, y_train, X_test, y_pred, y_std

def test_control_systems():
    """Test control systems functionality."""
    print("\n🎛️ Testing Control Systems:")
    
    try:
        import control
        
        # Create a simple transfer function: 1/(s^2 + 2s + 1)
        num = [1]
        den = [1, 2, 1]
        sys = control.TransferFunction(num, den)
        
        # Simulate step response
        time_points = np.linspace(0, 10, 100)
        t, y = control.step_response(sys, time_points)
        
        print("✅ Control systems working")
        print(f"   Transfer function: {sys}")
        print(f"   Step response computed for {len(t)} time points")
        print(f"   Final value: {y[-1]:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Control systems test failed: {e}")
        return False

def test_visualization():
    """Test visualization capabilities."""
    print("\n📊 Testing Visualization:")
    
    # Create a simple plot
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, 'b-', label='sin(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Test Plot')
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Matplotlib visualization working")
    print("   Test plot saved as 'test_plot.png'")
    
    return True

def main():
    """Main test function."""
    print("🚀 Model-Based RL Human Intent Recognition System")
    print("🔧 Core Functionality Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    # Run all tests
    try:
        if test_pytorch_functionality():
            tests_passed += 1
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
    
    try:
        if test_pyro_bayesian():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Pyro test failed: {e}")
    
    try:
        result = test_gaussian_process()
        if result[0]:  # GP test returns tuple
            tests_passed += 1
    except Exception as e:
        print(f"❌ GP test failed: {e}")
    
    try:
        if test_control_systems():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Control test failed: {e}")
    
    try:
        if test_visualization():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY: {tests_passed}/{total_tests} core tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL CORE FUNCTIONALITY TESTS PASSED!")
        print("✅ Dependencies successfully configured")
        print("🚀 Ready to implement Model-Based RL algorithms!")
    else:
        print(f"⚠️ {total_tests - tests_passed} tests failed")
        print("💡 Check dependency installation and compatibility")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)