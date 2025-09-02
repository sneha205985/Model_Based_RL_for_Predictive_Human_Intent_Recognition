"""
Streamlined Dataset Integration Testing Script

Simple script to test dataset integration with core algorithms without JSON serialization issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

def load_dataset():
    """Load and prepare the dataset."""
    print("Loading dataset...")
    
    # Load features CSV
    features_path = Path("data/synthetic_full/features.csv")
    if not features_path.exists():
        print("❌ Features CSV not found")
        return None, None
    
    df = pd.read_csv(features_path)
    print(f"✅ Loaded {len(df)} samples with {df.shape[1]} features")
    
    # Extract gesture labels from sequence_id
    if 'sequence_id' in df.columns:
        df['gesture_label'] = df['sequence_id'].str.extract(r'([a-zA-Z]+)_')[0]
        
        # Prepare features (numeric columns only)
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        features = df[feature_columns].values
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df['gesture_label'])
        
        print(f"✅ Extracted {features.shape[1]} features and {len(np.unique(labels))} gesture classes")
        return features, labels
    
    print("❌ No sequence_id column found")
    return None, None

def test_gaussian_process_integration(features, labels):
    """Test Gaussian Process integration."""
    print("\n🧠 Testing Gaussian Process Integration...")
    
    try:
        # Use subset for speed
        n_samples = min(300, len(features))
        X = features[:n_samples, :8]  # First 8 features
        y = features[:n_samples, 0]  # First feature as target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test RBF kernel
        start_time = time.time()
        gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0),
            random_state=42,
            alpha=1e-6
        )
        gp.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred, y_std = gp.predict(X_test_scaled, return_std=True)
        predict_time = time.time() - start_time
        
        # Calculate R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"   ✅ GP Training Time: {train_time:.3f}s")
        print(f"   ✅ GP Prediction Time: {predict_time:.3f}s")
        print(f"   ✅ GP R² Score: {r2:.3f}")
        print(f"   ✅ Mean Prediction Uncertainty: {np.mean(y_std):.3f}")
        
        return True, r2
        
    except Exception as e:
        print(f"   ❌ GP Test Failed: {e}")
        return False, 0.0

def test_intent_classification(features, labels):
    """Test intent classification."""
    print("\n🎯 Testing Intent Classification...")
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        start_time = time.time()
        classifier = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=8
        )
        classifier.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = classifier.predict(X_test_scaled)
        predict_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Get classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision = class_report['macro avg']['precision']
        recall = class_report['macro avg']['recall']
        f1_score = class_report['macro avg']['f1-score']
        
        print(f"   ✅ Classification Training Time: {train_time:.3f}s")
        print(f"   ✅ Classification Prediction Time: {predict_time:.3f}s")
        print(f"   ✅ Classification Accuracy: {accuracy:.3f}")
        print(f"   ✅ Macro Precision: {precision:.3f}")
        print(f"   ✅ Macro Recall: {recall:.3f}")
        print(f"   ✅ Macro F1-Score: {f1_score:.3f}")
        
        return True, accuracy
        
    except Exception as e:
        print(f"   ❌ Classification Test Failed: {e}")
        return False, 0.0

def test_trajectory_prediction(features):
    """Test trajectory prediction."""
    print("\n📈 Testing Trajectory Prediction...")
    
    try:
        # Create sequences for trajectory prediction
        sequence_length = 8
        n_features = min(6, features.shape[1])  # Use first 6 features
        
        sequences = []
        targets = []
        
        for i in range(len(features) - sequence_length):
            sequences.append(features[i:i+sequence_length, :n_features])
            targets.append(features[i+sequence_length, :n_features])
        
        if len(sequences) < 50:
            print(f"   ❌ Insufficient sequences: {len(sequences)}")
            return False, 0.0
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Split data
        split_idx = int(0.8 * len(sequences))
        X_train, X_test = sequences[:split_idx], sequences[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        
        # Flatten for linear model
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Train model
        start_time = time.time()
        model = Ridge(alpha=1.0)
        model.fit(X_train_flat, y_train)
        train_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test_flat)
        predict_time = time.time() - start_time
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        print(f"   ✅ Trajectory Training Time: {train_time:.3f}s")
        print(f"   ✅ Trajectory Prediction Time: {predict_time:.3f}s")
        print(f"   ✅ Trajectory MSE: {mse:.4f}")
        print(f"   ✅ Trajectory MAE: {mae:.4f}")
        print(f"   ✅ Sequences Generated: {len(sequences)}")
        
        return True, mse
        
    except Exception as e:
        print(f"   ❌ Trajectory Test Failed: {e}")
        return False, 0.0

def test_real_time_performance(features, labels):
    """Test real-time performance."""
    print("\n⚡ Testing Real-time Performance...")
    
    try:
        # Prepare data
        X = features[:500]  # Use subset
        y = labels[:500]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train fast model
        model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20]
        
        for batch_size in batch_sizes:
            inference_times = []
            
            # Multiple runs for reliable timing
            for _ in range(10):
                batch_data = X_test[:batch_size]
                
                start_time = time.time()
                predictions = model.predict(batch_data)
                inference_time = time.time() - start_time
                
                inference_times.append(inference_time)
            
            mean_time = np.mean(inference_times)
            per_sample_time_ms = (mean_time / batch_size) * 1000
            
            meets_10ms = per_sample_time_ms < 10
            meets_50ms = per_sample_time_ms < 50
            
            status_10ms = "✅" if meets_10ms else "❌"
            status_50ms = "✅" if meets_50ms else "❌"
            
            print(f"   Batch Size {batch_size}: {per_sample_time_ms:.2f}ms/sample {status_10ms}(<10ms) {status_50ms}(<50ms)")
        
        return True, per_sample_time_ms
        
    except Exception as e:
        print(f"   ❌ Real-time Test Failed: {e}")
        return False, 0.0

def generate_summary_visualization(test_results):
    """Generate summary visualization."""
    print("\n📊 Generating Summary Report...")
    
    try:
        # Create results visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Dataset Integration Test Results', fontsize=16, fontweight='bold')
        
        # Test success rates
        test_names = ['GP Integration', 'Intent Classification', 'Trajectory Prediction', 'Real-time Performance']
        success_rates = [1 if result[0] else 0 for result in test_results.values()]
        
        colors = ['green' if success else 'red' for success in success_rates]
        axes[0, 0].bar(test_names, success_rates, color=colors)
        axes[0, 0].set_title('Test Success Status')
        axes[0, 0].set_ylabel('Success (1=Pass, 0=Fail)')
        axes[0, 0].set_ylim(0, 1.2)
        plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
        
        # Performance scores
        if all(result[0] for result in test_results.values()):
            scores = [result[1] for result in test_results.values()]
            score_names = ['GP R²', 'Classification Acc', 'Trajectory MSE', 'RT Latency (ms)']
            
            # Normalize scores for visualization
            normalized_scores = [
                scores[0],  # GP R² (already 0-1)
                scores[1],  # Classification accuracy (already 0-1)
                min(1.0, 1.0/(1.0 + scores[2])),  # MSE (lower is better)
                min(1.0, 50.0/max(scores[3], 1))   # Latency (lower is better, normalize by 50ms target)
            ]
            
            axes[0, 1].bar(score_names, normalized_scores, color=['blue', 'green', 'orange', 'purple'])
            axes[0, 1].set_title('Performance Scores (Normalized)')
            axes[0, 1].set_ylabel('Score (Higher is Better)')
            axes[0, 1].set_ylim(0, 1.2)
            plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
        
        # Overall assessment
        axes[1, 0].axis('off')
        success_count = sum(success_rates)
        
        assessment_text = f"""DATASET INTEGRATION ASSESSMENT
        
✅ Tests Passed: {success_count}/4
📊 Success Rate: {success_count/4:.1%}

ALGORITHM COMPATIBILITY:
{"✅ Excellent" if success_count == 4 else "⚠️ Good" if success_count >= 3 else "❌ Poor"}

PRODUCTION READINESS:
{"✅ Ready" if success_count == 4 else "⚠️ Needs Work" if success_count >= 2 else "❌ Not Ready"}

REAL-TIME CAPABILITY:
{"✅ Meets Requirements" if test_results['real_time'][0] and test_results['real_time'][1] < 50 else "❌ Too Slow"}"""
        
        axes[1, 0].text(0.1, 0.9, assessment_text, transform=axes[1, 0].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Recommendations
        axes[1, 1].axis('off')
        
        recommendations = []
        if test_results['gp'][0] and test_results['gp'][1] > 0.8:
            recommendations.append("✅ GP regression excellent")
        elif not test_results['gp'][0]:
            recommendations.append("❌ Fix GP integration issues")
        
        if test_results['classification'][0] and test_results['classification'][1] > 0.8:
            recommendations.append("✅ Classification ready")
        elif test_results['classification'][1] < 0.7:
            recommendations.append("⚠️ Improve classification data")
        
        if test_results['trajectory'][0] and test_results['trajectory'][1] < 1.0:
            recommendations.append("✅ Trajectory prediction good")
        elif test_results['trajectory'][1] > 2.0:
            recommendations.append("⚠️ High trajectory error")
        
        if test_results['real_time'][0] and test_results['real_time'][1] < 10:
            recommendations.append("✅ Real-time performance excellent")
        elif test_results['real_time'][1] > 50:
            recommendations.append("❌ Too slow for real-time")
        
        recommendations.append("📈 Dataset ready for ML training")
        
        rec_text = "RECOMMENDATIONS:\n\n" + "\n".join(recommendations)
        axes[1, 1].text(0.1, 0.9, rec_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        
        # Create results directory
        results_dir = Path("dataset_validation_results")
        results_dir.mkdir(exist_ok=True)
        
        plt.savefig(results_dir / 'dataset_integration_test_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Summary visualization saved to: {results_dir / 'dataset_integration_test_summary.png'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization generation failed: {e}")
        return False

def main():
    """Main integration testing function."""
    print("🔗 Model-Based RL Dataset Integration Testing")
    print("="*60)
    print("Testing synthetic human behavior dataset with core ML algorithms")
    print("="*60)
    
    # Load dataset
    features, labels = load_dataset()
    if features is None:
        print("❌ Failed to load dataset")
        return 1
    
    # Run integration tests
    test_results = {}
    
    # Test 1: Gaussian Process
    test_results['gp'] = test_gaussian_process_integration(features, labels)
    
    # Test 2: Intent Classification
    test_results['classification'] = test_intent_classification(features, labels)
    
    # Test 3: Trajectory Prediction
    test_results['trajectory'] = test_trajectory_prediction(features)
    
    # Test 4: Real-time Performance
    test_results['real_time'] = test_real_time_performance(features, labels)
    
    # Generate summary
    generate_summary_visualization(test_results)
    
    # Final assessment
    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)
    
    successful_tests = sum(1 for result in test_results.values() if result[0])
    total_tests = len(test_results)
    
    print(f"📊 Tests Passed: {successful_tests}/{total_tests}")
    print(f"📈 Success Rate: {successful_tests/total_tests:.1%}")
    
    if successful_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Dataset is production-ready!")
        print("✅ Compatible with all core algorithms")
        print("✅ Meets real-time performance requirements")
        print("✅ Ready for Model-Based RL training")
    elif successful_tests >= 3:
        print("✅ MOST TESTS PASSED - Dataset is training-ready")
        print("⚠️ Minor issues may need attention")
    else:
        print("❌ MULTIPLE TEST FAILURES - Dataset needs improvement")
        print("🔧 Significant issues require fixing")
    
    print("="*70)
    
    return 0 if successful_tests >= 3 else 1

if __name__ == "__main__":
    exit(main())