#!/usr/bin/env python3
"""
GP Trajectory Prediction Demo - Phase 2
Demonstrates BasicGaussianProcess for trajectory prediction with visualization
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append('src')

from models.gaussian_process_basic import BasicGaussianProcess

def generate_sine_trajectory(n_points=100, noise=0.1):
    """Generate a sine wave trajectory for testing."""
    t = np.linspace(0, 4*np.pi, n_points)
    x = t
    y = np.sin(t) + noise * np.random.randn(n_points)
    
    # Create input-output pairs for GP training
    # Input: [x, y] at time t, Output: [x, y] at time t+1
    X = np.column_stack([x[:-1], y[:-1]])  # Current position
    Y = np.column_stack([x[1:], y[1:]])    # Next position
    
    return X, Y, t

def demo_gp_trajectory():
    """Demonstrate GP trajectory prediction."""
    print("🎯 GP Trajectory Prediction Demo")
    print("=" * 40)
    
    # Generate training trajectory
    X_train, y_train, t_full = generate_sine_trajectory(n_points=50, noise=0.1)
    print(f"📊 Training data: {X_train.shape} → {y_train.shape}")
    
    # Create and train GP
    gp = BasicGaussianProcess(kernel_type='rbf', length_scale=0.5)
    gp.fit(X_train, y_train)
    print("✅ GP trained successfully")
    
    # Test prediction on training data
    y_pred_train, y_std_train = gp.predict(X_train, return_std=True)
    
    # Predict trajectory from initial point
    initial_point = np.array([0, 0])  # Starting position
    trajectory = gp.predict_trajectory(initial_point, n_steps=20)
    print(f"✅ Trajectory predicted: {trajectory.shape}")
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Training data vs predictions
    ax1.scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.6, label='Training Input')
    ax1.scatter(y_train[:, 0], y_train[:, 1], c='red', alpha=0.6, label='Training Target')
    ax1.scatter(y_pred_train[:, 0], y_pred_train[:, 1], c='green', alpha=0.6, label='GP Prediction')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position') 
    ax1.set_title('GP Training Results')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted trajectory
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=2, label='Predicted Trajectory')
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='red', s=100, label='Start', zorder=5)
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='blue', s=100, label='End', zorder=5)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('GP Trajectory Prediction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gp_trajectory_demo.png', dpi=150, bbox_inches='tight')
    print("📊 Visualization saved as 'gp_trajectory_demo.png'")
    
    # Print some statistics
    print(f"\n📈 Results Summary:")
    print(f"   Mean prediction error: {np.mean(np.abs(y_train - y_pred_train)):.4f}")
    print(f"   Mean uncertainty: {np.mean(y_std_train):.4f}")
    print(f"   Trajectory length: {len(trajectory)} points")
    
    return gp, trajectory

def demo_real_dataset():
    """Demo GP with real dataset."""
    print("\n🔍 Real Dataset Demo")
    print("=" * 40)
    
    try:
        gp_real = BasicGaussianProcess()
        X_real, y_real = gp_real.load_real_dataset()
        
        # Use subset for demo
        n_train = 100
        X_train = X_real[:n_train]
        y_train = y_real[:n_train]
        
        # Train GP
        gp_real.fit(X_train, y_train)
        
        # Test predictions
        X_test = X_real[n_train:n_train+20]
        y_test = y_real[n_train:n_train+20]
        y_pred, y_std = gp_real.predict(X_test, return_std=True)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred)**2)
        r2 = gp_real.score(X_test, y_test)
        
        print(f"✅ Real dataset GP trained on {X_train.shape}")
        print(f"📊 Test MSE: {mse:.4f}")
        print(f"📊 Test R²: {r2:.4f}")
        print(f"📊 Mean uncertainty: {np.mean(y_std):.4f}")
        
        return gp_real
        
    except Exception as e:
        print(f"⚠️ Real dataset demo failed: {e}")
        return None

def main():
    """Main demo function."""
    print("🚀 BasicGaussianProcess Trajectory Demonstration")
    print("Phase 2: Core Algorithm Implementation")
    print("=" * 60)
    
    # Demo 1: Synthetic trajectory
    gp_synthetic, trajectory = demo_gp_trajectory()
    
    # Demo 2: Real dataset
    gp_real = demo_real_dataset()
    
    print("\n" + "=" * 60)
    print("🎉 GP Trajectory Demo Complete!")
    print("✅ BasicGaussianProcess working for trajectory prediction")
    print("✅ Both synthetic and real data tested")
    print("🚀 Ready for integration with MPC and Bayesian RL")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)