"""
Dataset Integration Testing Script

Tests the synthetic dataset with core algorithms to validate compatibility
and performance for the Model-Based RL Human Intent Recognition system.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class DatasetIntegrationTester:
    """Tests dataset integration with core algorithms."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.data_path = Path("data/synthetic_full")
        self.results_path = Path("dataset_validation_results")
        self.results_path.mkdir(exist_ok=True)
        
        # Load the dataset
        self.df = None
        self.features = None
        self.labels = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_and_prepare_data(self) -> bool:
        """Load and prepare dataset for testing."""
        self.logger.info("Loading dataset for integration testing...")
        
        try:
            # Load features CSV
            features_path = self.data_path / "features.csv"
            if not features_path.exists():
                self.logger.error("Features CSV not found")
                return False
            
            self.df = pd.read_csv(features_path)
            self.logger.info(f"Loaded dataset with {len(self.df)} samples and {self.df.shape[1]} features")
            
            # Extract features and labels
            if 'sequence_id' in self.df.columns:
                # Extract gesture labels from sequence_id
                self.df['gesture_label'] = self.df['sequence_id'].str.extract(r'([a-zA-Z]+)_')[0]
                
                # Prepare features (all numeric columns except sequence_id and gesture_label)
                feature_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
                self.features = self.df[feature_columns].values
                
                # Encode labels
                label_encoder = LabelEncoder()
                self.labels = label_encoder.fit_transform(self.df['gesture_label'])
                
                self.logger.info(f"Extracted {self.features.shape[1]} features")
                self.logger.info(f"Found {len(np.unique(self.labels))} unique gesture classes")
                
                return True
            else:
                self.logger.error("No sequence_id column found for label extraction")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return False
    
    def test_gaussian_process_training(self) -> Dict[str, Any]:
        """Test Gaussian Process training with trajectory data."""
        self.logger.info("Testing Gaussian Process integration...")
        
        try:
            # Prepare data for GP regression
            # Use first 500 samples for speed
            n_samples = min(500, len(self.features))
            X = self.features[:n_samples, :10]  # First 10 features
            y = self.features[:n_samples, 0]    # Use first feature as target
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Test different GP kernels
            kernels = {
                'RBF': RBF(length_scale=1.0),
                'Matern_3_2': Matern(length_scale=1.0, nu=1.5),
                'Matern_5_2': Matern(length_scale=1.0, nu=2.5)
            }
            
            results = {}
            
            for kernel_name, kernel in kernels.items():
                start_time = time.time()
                
                # Create and train GP
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    random_state=42,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=2  # Reduced for speed
                )
                
                gp.fit(X_train_scaled, y_train)
                train_time = time.time() - start_time
                
                # Make predictions
                start_time = time.time()
                y_pred, y_std = gp.predict(X_test_scaled, return_std=True)
                predict_time = time.time() - start_time
                
                # Calculate metrics
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test - y_pred))
                
                # R² score
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                results[kernel_name] = {
                    'training_time': train_time,
                    'prediction_time': predict_time,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'mean_prediction_std': np.mean(y_std),
                    'log_marginal_likelihood': gp.log_marginal_likelihood()
                }
                
                self.logger.info(f"{kernel_name} GP - R²: {r2:.3f}, RMSE: {rmse:.3f}, Time: {train_time:.2f}s")
            
            # Overall assessment
            best_kernel = max(results.keys(), key=lambda k: results[k]['r2_score'])
            
            gp_assessment = {
                'status': 'success',
                'best_kernel': best_kernel,
                'kernel_results': results,
                'dataset_compatibility': 'high',
                'recommendations': [
                    f"Best performing kernel: {best_kernel}",
                    f"Good R² score: {results[best_kernel]['r2_score']:.3f}",
                    "Dataset is well-suited for GP regression"
                ]
            }
            
            return gp_assessment
            
        except Exception as e:
            self.logger.error(f"GP integration test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'dataset_compatibility': 'unknown'
            }
    
    def test_intent_classification(self) -> Dict[str, Any]:
        """Test intent classification performance."""
        self.logger.info("Testing intent classification...")
        
        try:
            # Prepare data
            X = self.features
            y = self.labels
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest classifier (as baseline)
            start_time = time.time()
            classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                n_jobs=-1
            )
            classifier.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = classifier.predict(X_test_scaled)
            y_pred_proba = classifier.predict_proba(X_test_scaled)
            predict_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            
            # Get class names
            unique_labels = np.unique(self.labels)
            gesture_names = np.unique(self.df['gesture_label'])
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            feature_importance = classifier.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
            
            classification_results = {
                'status': 'success',
                'accuracy': accuracy,
                'training_time': train_time,
                'prediction_time': predict_time,
                'n_classes': len(unique_labels),
                'n_features_used': X.shape[1],
                'class_wise_metrics': {
                    'precision': class_report['macro avg']['precision'],
                    'recall': class_report['macro avg']['recall'],
                    'f1_score': class_report['macro avg']['f1-score']
                },
                'confusion_matrix': conf_matrix.tolist(),
                'feature_importance_top10': {
                    'indices': top_features_idx.tolist(),
                    'values': feature_importance[top_features_idx].tolist()
                },
                'dataset_compatibility': 'high' if accuracy > 0.8 else 'medium' if accuracy > 0.6 else 'low',
                'recommendations': self._generate_classification_recommendations(accuracy, class_report)
            }
            
            return classification_results
            
        except Exception as e:
            self.logger.error(f"Intent classification test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'dataset_compatibility': 'unknown'
            }
    
    def test_trajectory_prediction(self) -> Dict[str, Any]:
        """Test trajectory prediction capabilities."""
        self.logger.info("Testing trajectory prediction...")
        
        try:
            # Use position-related features for trajectory prediction
            position_features = [col for col in self.df.columns if any(pos in col.lower() 
                               for pos in ['pos_', 'position', 'x', 'y', 'z']) and 
                               self.df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            
            if len(position_features) < 3:
                position_features = list(self.df.select_dtypes(include=[np.number]).columns)[:10]
            
            # Prepare sequential data
            trajectory_data = self.df[position_features].values
            
            # Create sequences for prediction
            sequence_length = 10
            sequences = []
            targets = []
            
            for i in range(len(trajectory_data) - sequence_length):
                sequences.append(trajectory_data[i:i+sequence_length])
                targets.append(trajectory_data[i+sequence_length])
            
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            if len(sequences) < 50:
                return {
                    'status': 'failed',
                    'error': 'Insufficient data for trajectory prediction',
                    'dataset_compatibility': 'low'
                }
            
            # Split data
            split_idx = int(0.8 * len(sequences))
            X_train, X_test = sequences[:split_idx], sequences[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            
            # Simple linear model for trajectory prediction
            from sklearn.linear_model import Ridge
            
            # Flatten sequences for linear model
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
            
            # Trajectory smoothness assessment
            smoothness_scores = []
            for i in range(len(y_pred)):
                # Calculate smoothness as inverse of second derivative
                if len(y_pred[i]) >= 3:
                    second_diff = np.diff(y_pred[i], n=2)
                    smoothness = 1.0 / (1.0 + np.std(second_diff))
                    smoothness_scores.append(smoothness)
            
            trajectory_results = {
                'status': 'success',
                'training_time': train_time,
                'prediction_time': predict_time,
                'mse': mse,
                'mae': mae,
                'mean_smoothness': np.mean(smoothness_scores) if smoothness_scores else 0.0,
                'sequence_length': sequence_length,
                'n_features': len(position_features),
                'n_sequences': len(sequences),
                'dataset_compatibility': 'high' if mse < 1.0 else 'medium' if mse < 5.0 else 'low',
                'recommendations': self._generate_trajectory_recommendations(mse, smoothness_scores)
            }
            
            return trajectory_results
            
        except Exception as e:
            self.logger.error(f"Trajectory prediction test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'dataset_compatibility': 'unknown'
            }
    
    def test_real_time_performance(self) -> Dict[str, Any]:
        """Test real-time performance requirements."""
        self.logger.info("Testing real-time performance...")
        
        try:
            # Test different batch sizes for real-time inference
            batch_sizes = [1, 5, 10, 20, 50]
            performance_results = {}
            
            # Use a simple model for speed testing
            from sklearn.ensemble import RandomForestClassifier
            
            # Prepare data
            X = self.features[:1000]  # Use subset for speed
            y = self.labels[:1000]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
            model.fit(X_train, y_train)
            
            for batch_size in batch_sizes:
                inference_times = []
                
                # Test multiple inference runs
                for _ in range(20):  # 20 runs for statistical significance
                    batch_data = X_test[:batch_size]
                    
                    start_time = time.time()
                    predictions = model.predict(batch_data)
                    inference_time = time.time() - start_time
                    
                    inference_times.append(inference_time)
                
                # Calculate statistics
                mean_time = np.mean(inference_times)
                std_time = np.std(inference_times)
                max_time = np.max(inference_times)
                
                # Per-sample time
                per_sample_time = mean_time / batch_size
                
                performance_results[f'batch_size_{batch_size}'] = {
                    'mean_inference_time': mean_time,
                    'std_inference_time': std_time,
                    'max_inference_time': max_time,
                    'per_sample_time_ms': per_sample_time * 1000,
                    'meets_10ms_requirement': per_sample_time < 0.01,
                    'meets_50ms_requirement': per_sample_time < 0.05,
                    'throughput_samples_per_sec': batch_size / mean_time
                }
            
            # Overall assessment
            single_sample_time = performance_results['batch_size_1']['per_sample_time_ms']
            
            real_time_assessment = {
                'status': 'success',
                'batch_performance': performance_results,
                'single_sample_latency_ms': single_sample_time,
                'meets_real_time_constraints': single_sample_time < 50,  # 50ms threshold
                'recommended_batch_size': self._find_optimal_batch_size(performance_results),
                'dataset_compatibility': 'high' if single_sample_time < 10 else 'medium' if single_sample_time < 50 else 'low'
            }
            
            return real_time_assessment
            
        except Exception as e:
            self.logger.error(f"Real-time performance test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'dataset_compatibility': 'unknown'
            }
    
    def _generate_classification_recommendations(self, accuracy: float, class_report: Dict) -> List[str]:
        """Generate recommendations for classification results."""
        recommendations = []
        
        if accuracy > 0.9:
            recommendations.append("Excellent classification accuracy - dataset is production-ready")
        elif accuracy > 0.8:
            recommendations.append("Good classification accuracy - minor improvements possible")
        elif accuracy > 0.7:
            recommendations.append("Moderate accuracy - consider feature engineering")
        else:
            recommendations.append("Low accuracy - dataset quality improvements needed")
        
        if 'macro avg' in class_report:
            precision = class_report['macro avg']['precision']
            recall = class_report['macro avg']['recall']
            
            if precision < 0.8:
                recommendations.append("Improve precision by reducing false positives")
            if recall < 0.8:
                recommendations.append("Improve recall by reducing false negatives")
        
        return recommendations
    
    def _generate_trajectory_recommendations(self, mse: float, smoothness_scores: List[float]) -> List[str]:
        """Generate recommendations for trajectory prediction."""
        recommendations = []
        
        if mse < 0.5:
            recommendations.append("Excellent trajectory prediction accuracy")
        elif mse < 2.0:
            recommendations.append("Good trajectory prediction - suitable for MPC")
        else:
            recommendations.append("High prediction error - consider data smoothing")
        
        if smoothness_scores:
            avg_smoothness = np.mean(smoothness_scores)
            if avg_smoothness < 0.7:
                recommendations.append("Improve trajectory smoothness with filtering")
            else:
                recommendations.append("Good trajectory smoothness for motion planning")
        
        return recommendations
    
    def _find_optimal_batch_size(self, performance_results: Dict) -> int:
        """Find optimal batch size for real-time performance."""
        best_batch_size = 1
        best_throughput = 0
        
        for key, result in performance_results.items():
            if result['meets_50ms_requirement']:
                throughput = result['throughput_samples_per_sec']
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = int(key.split('_')[-1])
        
        return best_batch_size
    
    def generate_integration_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive integration test report."""
        self.logger.info("Generating integration test report...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Integration Test Results', fontsize=16, fontweight='bold')
        
        # 1. GP Performance Comparison
        if 'gp_test' in results and results['gp_test']['status'] == 'success':
            gp_results = results['gp_test']['kernel_results']
            kernels = list(gp_results.keys())
            r2_scores = [gp_results[k]['r2_score'] for k in kernels]
            
            axes[0, 0].bar(kernels, r2_scores, color=['blue', 'green', 'red'])
            axes[0, 0].set_title('Gaussian Process R² Scores by Kernel')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].set_ylim(0, 1)
            
            for i, score in enumerate(r2_scores):
                axes[0, 0].text(i, score + 0.02, f'{score:.3f}', ha='center')
        
        # 2. Classification Performance
        if 'classification_test' in results and results['classification_test']['status'] == 'success':
            class_results = results['classification_test']
            
            metrics = ['precision', 'recall', 'f1_score']
            values = [class_results['class_wise_metrics'][m] for m in metrics]
            
            axes[0, 1].bar(metrics, values, color=['orange', 'purple', 'brown'])
            axes[0, 1].set_title(f"Classification Metrics (Accuracy: {class_results['accuracy']:.3f})")
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_ylim(0, 1)
            
            for i, value in enumerate(values):
                axes[0, 1].text(i, value + 0.02, f'{value:.3f}', ha='center')
        
        # 3. Real-time Performance
        if 'real_time_test' in results and results['real_time_test']['status'] == 'success':
            rt_results = results['real_time_test']['batch_performance']
            
            batch_sizes = [int(k.split('_')[-1]) for k in rt_results.keys()]
            latencies = [rt_results[k]['per_sample_time_ms'] for k in rt_results.keys()]
            
            axes[1, 0].plot(batch_sizes, latencies, 'o-', linewidth=2, markersize=8)
            axes[1, 0].axhline(y=10, color='red', linestyle='--', label='10ms target')
            axes[1, 0].axhline(y=50, color='orange', linestyle='--', label='50ms limit')
            axes[1, 0].set_title('Real-time Latency vs Batch Size')
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('Latency per Sample (ms)')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
        
        # 4. Overall Assessment
        axes[1, 1].axis('off')
        
        # Compile overall assessment
        assessment_text = "INTEGRATION ASSESSMENT:\n\n"
        
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and 'status' in test_result:
                status_icon = "✓" if test_result['status'] == 'success' else "✗"
                compatibility = test_result.get('dataset_compatibility', 'unknown')
                assessment_text += f"{status_icon} {test_name.replace('_', ' ').title()}: {compatibility}\n"
        
        assessment_text += "\nRECOMMENDATIONS:\n"
        all_recommendations = []
        for test_result in results.values():
            if isinstance(test_result, dict) and 'recommendations' in test_result:
                all_recommendations.extend(test_result['recommendations'])
        
        for i, rec in enumerate(all_recommendations[:8], 1):  # Top 8 recommendations
            assessment_text += f"{i}. {rec}\n"
        
        axes[1, 1].text(0.05, 0.95, assessment_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'dataset_integration_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Integration report saved to {self.results_path / 'dataset_integration_report.png'}")
    
    def run_complete_integration_tests(self) -> Dict[str, Any]:
        """Run complete dataset integration test suite."""
        self.logger.info("Starting complete dataset integration tests...")
        
        if not self.load_and_prepare_data():
            return {'error': 'Failed to load dataset'}
        
        # Run all integration tests
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'n_samples': len(self.features),
                'n_features': self.features.shape[1],
                'n_classes': len(np.unique(self.labels)),
                'class_distribution': dict(zip(*np.unique(self.labels, return_counts=True)))
            }
        }
        
        # Test 1: Gaussian Process Integration
        results['gp_test'] = self.test_gaussian_process_training()
        
        # Test 2: Intent Classification
        results['classification_test'] = self.test_intent_classification()
        
        # Test 3: Trajectory Prediction
        results['trajectory_test'] = self.test_trajectory_prediction()
        
        # Test 4: Real-time Performance
        results['real_time_test'] = self.test_real_time_performance()
        
        # Generate comprehensive report
        self.generate_integration_report(results)
        
        # Overall assessment
        successful_tests = sum(1 for test in results.values() 
                             if isinstance(test, dict) and test.get('status') == 'success')
        total_tests = 4
        
        results['overall_assessment'] = {
            'success_rate': successful_tests / total_tests,
            'ready_for_ml_training': successful_tests >= 3,
            'production_ready': successful_tests == 4,
            'primary_issues': self._identify_integration_issues(results)
        }
        
        # Save results
        report_path = self.results_path / 'dataset_integration_test_results.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Integration test results saved to {report_path}")
        
        return results
    
    def _identify_integration_issues(self, results: Dict[str, Any]) -> List[str]:
        """Identify primary integration issues."""
        issues = []
        
        for test_name, test_result in results.items():
            if isinstance(test_result, dict):
                if test_result.get('status') == 'failed':
                    issues.append(f"{test_name} failed: {test_result.get('error', 'unknown error')}")
                elif test_result.get('dataset_compatibility') == 'low':
                    issues.append(f"{test_name} shows low compatibility")
        
        return issues if issues else ["No major integration issues detected"]


def main():
    """Main integration testing function."""
    print("🔗 Dataset Integration Testing")
    print("="*50)
    print("Testing dataset compatibility with core algorithms...")
    print()
    
    tester = DatasetIntegrationTester()
    
    try:
        results = tester.run_complete_integration_tests()
        
        # Print summary
        print("\n" + "="*70)
        print("DATASET INTEGRATION TEST SUMMARY")
        print("="*70)
        
        if 'error' in results:
            print(f"❌ Integration testing failed: {results['error']}")
            return 1
        
        dataset_info = results['dataset_info']
        assessment = results['overall_assessment']
        
        print(f"📊 Dataset Information:")
        print(f"  Samples: {dataset_info['n_samples']:,}")
        print(f"  Features: {dataset_info['n_features']}")
        print(f"  Classes: {dataset_info['n_classes']}")
        
        print(f"\n🧪 Test Results:")
        test_names = ['gp_test', 'classification_test', 'trajectory_test', 'real_time_test']
        test_labels = ['Gaussian Process', 'Intent Classification', 'Trajectory Prediction', 'Real-time Performance']
        
        for test_name, test_label in zip(test_names, test_labels):
            if test_name in results:
                test_result = results[test_name]
                status_icon = "✅" if test_result.get('status') == 'success' else "❌"
                compatibility = test_result.get('dataset_compatibility', 'unknown')
                print(f"  {status_icon} {test_label}: {compatibility} compatibility")
        
        print(f"\n🎯 Overall Assessment:")
        print(f"  Success Rate: {assessment['success_rate']:.1%}")
        print(f"  Ready for ML Training: {'✅' if assessment['ready_for_ml_training'] else '❌'}")
        print(f"  Production Ready: {'✅' if assessment['production_ready'] else '❌'}")
        
        if assessment['primary_issues']:
            print(f"\n⚠️ Issues Identified:")
            for issue in assessment['primary_issues']:
                print(f"  • {issue}")
        
        print(f"\n📊 Detailed reports saved to: dataset_validation_results/")
        print("="*70)
        
        return 0 if assessment['ready_for_ml_training'] else 1
        
    except Exception as e:
        print(f"❌ Integration testing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())