#!/usr/bin/env python3
"""
Comprehensive Visualization Suite Demonstration.

This script demonstrates the full capabilities of the Phase 6 visualization
and results analysis system for the Model-Based RL Human Intent Recognition project.

Usage:
    python scripts/visualization_demo.py [--output-dir results/demo] [--format html,pdf]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import argparse
import logging
import time
from datetime import datetime
import json

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import visualization modules
from src.visualization import (
    # Core utilities
    create_publication_config, create_presentation_config,
    
    # Analyzers
    PerformanceAnalyzer, SafetyAnalyzer, BayesianAnalyzer, StatisticalFramework,
    
    # Interactive components
    InteractiveDashboard, RealTimeVisualizer, RealTimeDataGenerator,
    
    # Report generation
    AutomatedReportGenerator, ReportConfig, ReportType,
    
    # Data structures
    SafetyEvent, PosteriorDistribution, RealTimeData
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_experimental_data():
    """Generate comprehensive synthetic experimental data for demonstration."""
    
    logger.info("Generating synthetic experimental data...")
    
    np.random.seed(42)  # For reproducible results
    
    # Method names
    methods = ['Baseline MPC', 'Bayesian RL', 'Adaptive Hybrid', 'Deep Q-Learning']
    n_trials = 50
    
    # Performance data
    performance_data = {
        'success_rates': {},
        'completion_times': {},
        'learning_curves': {},
        'summary_stats': {}
    }
    
    base_success_rates = [0.72, 0.85, 0.88, 0.79]
    base_completion_times = [12.5, 9.8, 9.2, 11.3]
    
    for i, method in enumerate(methods):
        # Success rates with some variation
        success_rate = base_success_rates[i]
        performance_data['success_rates'][method] = np.random.beta(
            success_rate * 20, (1 - success_rate) * 20, n_trials
        )
        
        # Completion times
        completion_time = base_completion_times[i]
        performance_data['completion_times'][method] = np.random.gamma(
            4, completion_time / 4, n_trials
        )
        
        # Learning curves (50 epochs)
        n_epochs = 50
        if method == 'Baseline MPC':
            # Static performance
            loss_curve = np.ones(n_epochs) * 0.3 + np.random.normal(0, 0.02, n_epochs)
            acc_curve = np.ones(n_epochs) * success_rate + np.random.normal(0, 0.01, n_epochs)
        else:
            # Learning curves
            loss_curve = np.exp(-np.linspace(0, 3, n_epochs)) * 0.8 + 0.1
            loss_curve += np.random.normal(0, 0.02, n_epochs)
            
            acc_curve = 1 - np.exp(-np.linspace(0, 2.5, n_epochs)) * (1 - success_rate)
            acc_curve += np.random.normal(0, 0.01, n_epochs)
        
        performance_data['learning_curves'][method] = {
            'loss': np.clip(loss_curve, 0.05, 1.0).tolist(),
            'accuracy': np.clip(acc_curve, 0.0, 1.0).tolist(),
            'reward': (acc_curve * 100 - 50).tolist()
        }
    
    # Safety data
    n_timesteps = 200
    timestamps = np.linspace(0, 100, n_timesteps)  # 100 seconds
    
    # Generate human-robot distances
    base_distance = 1.5
    distances = []
    violations = []
    
    for t in timestamps:
        # Base distance with some interaction patterns
        distance = base_distance + 0.5 * np.sin(0.1 * t) + 0.3 * np.sin(0.05 * t)
        distance += np.random.normal(0, 0.1)  # Noise
        
        # Occasional close approaches
        if np.random.random() < 0.02:  # 2% chance
            distance *= 0.3  # Get very close
            
            if distance < 0.5:  # Safety threshold
                violations.append(SafetyEvent(
                    timestamp=t,
                    event_type="distance_violation",
                    severity="high" if distance < 0.3 else "moderate",
                    distance=distance,
                    position=np.array([2.0, 1.0, 0.8]),
                    human_position=np.array([2.0 + distance, 1.0, 0.8]),
                    robot_velocity=0.2
                ))
        
        distances.append(max(0.1, distance))
    
    safety_data = {
        'distances': distances,
        'timestamps': timestamps.tolist(),
        'safety_threshold': 0.5,
        'violations': violations,
        'risk_scores': [min(1.0, 0.8 / (d + 0.1)) for d in distances]
    }
    
    # Bayesian data
    bayesian_data = {
        'posterior_history': [],
        'uncertainty_data': {
            'epistemic': np.random.exponential(0.1, n_timesteps).tolist(),
            'aleatoric': np.random.exponential(0.05, n_timesteps).tolist()
        },
        'information_gain': np.random.exponential(0.3, n_timesteps).tolist(),
        'policy_entropy': 2.0 * np.exp(-np.linspace(0, 3, n_timesteps)) + 0.5
    }
    
    # Generate posterior evolution (5 time points)
    for i in range(5):
        # Two parameters evolving over time
        mean1 = 0.5 + i * 0.1 + np.random.normal(0, 0.02)
        mean2 = 1.0 - i * 0.05 + np.random.normal(0, 0.02)
        
        samples = np.random.multivariate_normal(
            [mean1, mean2],
            [[0.1, 0.02], [0.02, 0.08]],
            1000
        )
        
        posterior = PosteriorDistribution(
            samples=samples,
            parameter_names=['learning_rate', 'discount_factor']
        )
        bayesian_data['posterior_history'].append(posterior)
    
    # Uncertainty decomposition
    epistemic = np.array(bayesian_data['uncertainty_data']['epistemic'])
    aleatoric = np.array(bayesian_data['uncertainty_data']['aleatoric'])
    bayesian_data['uncertainty_data']['total'] = np.sqrt(epistemic**2 + aleatoric**2).tolist()
    
    # Experimental setup metadata
    experimental_setup = {
        'n_trials': n_trials,
        'n_methods': len(methods),
        'duration_hours': 8,
        'environment': 'High-fidelity Simulation',
        'robot_type': '6-DOF Manipulator',
        'human_subjects': 5,
        'scenarios': ['Handover', 'Collaborative Assembly', 'Shared Workspace']
    }
    
    return {
        'performance_data': performance_data,
        'safety_data': safety_data,
        'bayesian_data': bayesian_data,
        'experimental_setup': experimental_setup,
        'comparison_data': {
            method: performance_data['success_rates'][method] 
            for method in methods
        }
    }


def demonstrate_performance_analysis(experimental_data, output_dir):
    """Demonstrate performance analysis capabilities."""
    
    logger.info("Demonstrating performance analysis...")
    
    analyzer = PerformanceAnalyzer(create_publication_config())
    
    # Success rate comparison
    logger.info("Creating success rate comparison...")
    fig_success = analyzer.plot_success_rate_comparison(
        experimental_data['performance_data']['success_rates'],
        plot_type='bar',
        include_stats=True,
        save_path=str(output_dir / "success_rate_comparison.png")
    )
    plt.close(fig_success)
    
    # Learning curves
    logger.info("Creating learning curves...")
    fig_learning = analyzer.plot_learning_curves(
        experimental_data['performance_data']['learning_curves'],
        include_confidence=True,
        save_path=str(output_dir / "learning_curves.png")
    )
    plt.close(fig_learning)
    
    # Performance heatmap
    logger.info("Creating performance heatmap...")
    methods = list(experimental_data['performance_data']['success_rates'].keys())
    metrics_data = {
        'Success Rate': [np.mean(experimental_data['performance_data']['success_rates'][m]) for m in methods],
        'Completion Time': [np.mean(experimental_data['performance_data']['completion_times'][m]) for m in methods],
        'Final Accuracy': [experimental_data['performance_data']['learning_curves'][m]['accuracy'][-1] for m in methods],
        'Final Reward': [experimental_data['performance_data']['learning_curves'][m]['reward'][-1] for m in methods]
    }
    
    performance_matrix = pd.DataFrame(metrics_data, index=methods)
    fig_heatmap = analyzer.plot_performance_heatmap(
        performance_matrix,
        save_path=str(output_dir / "performance_heatmap.png")
    )
    plt.close(fig_heatmap)
    
    logger.info("Performance analysis demonstration completed")


def demonstrate_safety_analysis(experimental_data, output_dir):
    """Demonstrate safety analysis capabilities."""
    
    logger.info("Demonstrating safety analysis...")
    
    analyzer = SafetyAnalyzer(create_publication_config())
    
    # Distance over time
    logger.info("Creating distance monitoring plot...")
    fig_distance = analyzer.plot_distance_over_time(
        experimental_data['safety_data']['distances'],
        experimental_data['safety_data']['timestamps'],
        experimental_data['safety_data']['safety_threshold'],
        save_path=str(output_dir / "distance_monitoring.png"),
        include_violations=True,
        show_risk_zones=True
    )
    plt.close(fig_distance)
    
    # Safety violations analysis
    if experimental_data['safety_data']['violations']:
        logger.info("Creating safety violations analysis...")
        fig_violations = analyzer.plot_safety_violations_analysis(
            experimental_data['safety_data']['violations'],
            save_path=str(output_dir / "safety_violations.png")
        )
        plt.close(fig_violations)
    
    # Risk assessment
    logger.info("Creating risk assessment plot...")
    fig_risk = analyzer.plot_risk_assessment(
        experimental_data['safety_data']['risk_scores'],
        experimental_data['safety_data']['timestamps'],
        save_path=str(output_dir / "risk_assessment.png")
    )
    plt.close(fig_risk)
    
    logger.info("Safety analysis demonstration completed")


def demonstrate_bayesian_analysis(experimental_data, output_dir):
    """Demonstrate Bayesian analysis capabilities."""
    
    logger.info("Demonstrating Bayesian analysis...")
    
    analyzer = BayesianAnalyzer(create_publication_config())
    
    # Posterior evolution
    logger.info("Creating posterior evolution plot...")
    fig_posterior = analyzer.plot_posterior_evolution(
        experimental_data['bayesian_data']['posterior_history'],
        save_path=str(output_dir / "posterior_evolution.png")
    )
    plt.close(fig_posterior)
    
    # Uncertainty decomposition
    logger.info("Creating uncertainty decomposition plot...")
    fig_uncertainty = analyzer.plot_uncertainty_decomposition(
        experimental_data['bayesian_data']['uncertainty_data'],
        experimental_data['safety_data']['timestamps'],
        save_path=str(output_dir / "uncertainty_decomposition.png")
    )
    plt.close(fig_uncertainty)
    
    # Information gain
    logger.info("Creating information gain plot...")
    fig_info_gain = analyzer.plot_information_gain(
        experimental_data['bayesian_data']['information_gain'],
        experimental_data['safety_data']['timestamps'],
        save_path=str(output_dir / "information_gain.png")
    )
    plt.close(fig_info_gain)
    
    # Policy entropy
    logger.info("Creating policy entropy plot...")
    fig_entropy = analyzer.plot_policy_entropy(
        experimental_data['bayesian_data']['policy_entropy'].tolist(),
        experimental_data['safety_data']['timestamps'],
        save_path=str(output_dir / "policy_entropy.png"),
        include_target=1.0
    )
    plt.close(fig_entropy)
    
    logger.info("Bayesian analysis demonstration completed")


def demonstrate_statistical_analysis(experimental_data, output_dir):
    """Demonstrate statistical analysis capabilities."""
    
    logger.info("Demonstrating statistical analysis...")
    
    framework = StatisticalFramework(create_publication_config())
    
    # Comprehensive statistical analysis
    logger.info("Performing comprehensive statistical analysis...")
    analysis_results = framework.perform_comprehensive_analysis(
        experimental_data['comparison_data'],
        alpha=0.05,
        correction_method='fdr_bh'
    )
    
    # Statistical summary
    fig_stats = framework.plot_statistical_summary(
        analysis_results,
        save_path=str(output_dir / "statistical_summary.png")
    )
    plt.close(fig_stats)
    
    # Effect sizes analysis
    logger.info("Creating effect sizes analysis...")
    fig_effect = framework.plot_effect_sizes_analysis(
        analysis_results['effect_sizes'],
        save_path=str(output_dir / "effect_sizes.png")
    )
    plt.close(fig_effect)
    
    # Bootstrap analysis for best method
    best_method = max(experimental_data['comparison_data'].keys(),
                     key=lambda x: np.mean(experimental_data['comparison_data'][x]))
    
    logger.info(f"Creating bootstrap analysis for {best_method}...")
    fig_bootstrap = framework.plot_bootstrap_analysis(
        experimental_data['comparison_data'][best_method],
        statistic=np.mean,
        n_bootstrap=1000,
        save_path=str(output_dir / "bootstrap_analysis.png")
    )
    plt.close(fig_bootstrap)
    
    # Save statistical results
    results_file = output_dir / "statistical_results.json"
    
    # Convert results to serializable format
    serializable_results = {
        'statistical_tests': [
            {
                'test_name': test.test_name,
                'statistic': float(test.statistic),
                'p_value': float(test.p_value),
                'effect_size': float(test.effect_size) if test.effect_size else None,
                'description': test.description
            }
            for test in analysis_results['statistical_tests']
        ],
        'effect_sizes': {k: float(v) for k, v in analysis_results['effect_sizes'].items()},
        'recommendations': analysis_results['recommendations']
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info("Statistical analysis demonstration completed")


def demonstrate_interactive_dashboard(experimental_data, output_dir):
    """Demonstrate interactive dashboard capabilities."""
    
    logger.info("Demonstrating interactive dashboard...")
    
    dashboard = InteractiveDashboard()
    
    # Create comprehensive dashboard
    layout_config = {
        'layout': 'grid',
        'components': [
            {
                'name': 'performance_metrics',
                'type': 'time_series',
                'title': 'Performance Metrics Over Time',
                'data': {
                    'x': list(range(50)),
                    'y': {
                        'Success Rate': experimental_data['performance_data']['learning_curves']['Bayesian RL']['accuracy'],
                        'Reward': experimental_data['performance_data']['learning_curves']['Bayesian RL']['reward']
                    }
                }
            },
            {
                'name': 'safety_distance',
                'type': 'time_series',
                'title': 'Human-Robot Distance',
                'data': {
                    'x': experimental_data['safety_data']['timestamps'][:100],  # First 100 points
                    'y': {
                        'Distance': experimental_data['safety_data']['distances'][:100]
                    }
                }
            },
            {
                'name': 'method_comparison',
                'type': 'bar_chart',
                'title': 'Method Comparison',
                'data': {
                    'x': list(experimental_data['performance_data']['success_rates'].keys()),
                    'y': [np.mean(rates) for rates in experimental_data['performance_data']['success_rates'].values()]
                }
            },
            {
                'name': 'uncertainty_gauge',
                'type': 'gauge',
                'title': 'Current Uncertainty',
                'data': {
                    'value': np.mean(experimental_data['bayesian_data']['uncertainty_data']['total'][:10]),
                    'min': 0,
                    'max': 1,
                    'threshold': 0.8
                }
            }
        ]
    }
    
    fig_dashboard = dashboard.create_comprehensive_dashboard(
        layout_config,
        save_path=str(output_dir / "interactive_dashboard.html")
    )
    
    # Parameter sensitivity dashboard
    logger.info("Creating parameter sensitivity dashboard...")
    
    def mock_sensitivity_function(params):
        # Mock function that computes performance based on parameters
        learning_rate = params.get('learning_rate', 0.01)
        batch_size = params.get('batch_size', 32)
        hidden_units = params.get('hidden_units', 128)
        
        # Simple heuristic combining parameters
        performance = (
            0.8 + 0.1 * np.log10(learning_rate / 0.001) +
            0.05 * (1 - abs(batch_size - 64) / 64) +
            0.05 * min(1, hidden_units / 256)
        )
        
        return max(0.5, min(0.95, performance + np.random.normal(0, 0.02)))
    
    parameter_ranges = {
        'learning_rate': (0.001, 0.1),
        'batch_size': (16, 128),
        'hidden_units': (64, 512)
    }
    
    fig_sensitivity = dashboard.create_parameter_sensitivity_dashboard(
        parameter_ranges,
        mock_sensitivity_function,
        save_path=str(output_dir / "parameter_sensitivity.html")
    )
    
    logger.info("Interactive dashboard demonstration completed")


def demonstrate_real_time_system(experimental_data, output_dir):
    """Demonstrate real-time visualization system."""
    
    logger.info("Demonstrating real-time visualization system...")
    
    # Create real-time data generator
    data_generator = RealTimeDataGenerator(seed=123)
    
    # Generate and collect real-time data
    rt_visualizer = RealTimeVisualizer()
    
    logger.info("Generating real-time data stream...")
    for i in range(100):
        data = data_generator.generate_sample_data()
        rt_visualizer.add_data(data)
        
        if i % 20 == 0:
            logger.info(f"Generated {i+1}/100 data points...")
    
    # Export real-time data
    logger.info("Exporting real-time data...")
    rt_visualizer.export_data(
        str(output_dir / "realtime_data.json"),
        format='json'
    )
    
    rt_visualizer.export_data(
        str(output_dir / "realtime_data.csv"),
        format='csv'
    )
    
    # Create streaming plot
    def mock_data_source():
        return {
            'cpu_usage': np.random.uniform(20, 80),
            'memory_usage': np.random.uniform(30, 70),
            'mpc_solve_time': np.random.uniform(0.005, 0.02),
            'success_rate': np.random.uniform(0.75, 0.95)
        }
    
    metrics = ['cpu_usage', 'memory_usage', 'mpc_solve_time', 'success_rate']
    
    fig_streaming = rt_visualizer.create_streaming_plot(
        metrics,
        mock_data_source,
        save_path=str(output_dir / "streaming_plot.png")
    )
    plt.close(fig_streaming)
    
    logger.info("Real-time system demonstration completed")


def demonstrate_report_generation(experimental_data, output_dir, formats):
    """Demonstrate automated report generation."""
    
    logger.info("Demonstrating automated report generation...")
    
    # Configure report generator
    config = ReportConfig(
        title="Model-Based RL Human Intent Recognition - Analysis Report",
        author="Claude Code Automated Analysis System",
        organization="Research Demonstration",
        output_formats=formats,
        include_executive_summary=True,
        include_methodology=True,
        include_statistical_tests=True,
        include_visualizations=True,
        include_recommendations=True
    )
    
    generator = AutomatedReportGenerator(config)
    
    # Generate comprehensive report
    logger.info("Generating comprehensive analysis report...")
    report_path = generator.generate_comprehensive_report(
        experimental_data,
        str(output_dir / "comprehensive_report"),
        ReportType.COMPREHENSIVE
    )
    
    # Generate publication figures
    logger.info("Creating publication-ready figures...")
    figure_paths = generator.create_publication_figures(
        experimental_data,
        str(output_dir / "publication_figures")
    )
    
    logger.info(f"Generated report: {report_path}")
    logger.info(f"Generated {len(figure_paths)} publication figures")
    
    logger.info("Report generation demonstration completed")


def main():
    """Main demonstration function."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Visualization Suite Demonstration"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/visualization_demo',
        help='Output directory for demonstration results'
    )
    parser.add_argument(
        '--formats',
        type=str,
        default='html',
        help='Report formats to generate (comma-separated: html,pdf)'
    )
    parser.add_argument(
        '--components',
        type=str,
        default='all',
        help='Components to demonstrate (all, performance, safety, bayesian, stats, dashboard, realtime, reports)'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting comprehensive visualization demonstration...")
    logger.info(f"Output directory: {output_dir}")
    
    # Generate experimental data
    experimental_data = generate_synthetic_experimental_data()
    
    # Parse components to demonstrate
    if args.components.lower() == 'all':
        components = ['performance', 'safety', 'bayesian', 'stats', 'dashboard', 'realtime', 'reports']
    else:
        components = [c.strip().lower() for c in args.components.split(',')]
    
    # Parse formats
    formats = [f.strip().lower() for f in args.formats.split(',')]
    
    start_time = time.time()
    
    try:
        # Demonstrate each component
        if 'performance' in components:
            demonstrate_performance_analysis(experimental_data, output_dir)
        
        if 'safety' in components:
            demonstrate_safety_analysis(experimental_data, output_dir)
        
        if 'bayesian' in components:
            demonstrate_bayesian_analysis(experimental_data, output_dir)
        
        if 'stats' in components:
            demonstrate_statistical_analysis(experimental_data, output_dir)
        
        if 'dashboard' in components:
            demonstrate_interactive_dashboard(experimental_data, output_dir)
        
        if 'realtime' in components:
            demonstrate_real_time_system(experimental_data, output_dir)
        
        if 'reports' in components:
            demonstrate_report_generation(experimental_data, output_dir, formats)
        
        # Save experimental data for reference
        logger.info("Saving experimental data for reference...")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif hasattr(obj, '__dict__'):
                return {k: convert_for_json(v) for k, v in obj.__dict__.items() 
                       if not k.startswith('_')}
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            else:
                return obj
        
        # Save subset of experimental data (excluding complex objects)
        data_to_save = {
            'performance_data': {
                'success_rates': {k: convert_for_json(v) for k, v in experimental_data['performance_data']['success_rates'].items()},
                'completion_times': {k: convert_for_json(v) for k, v in experimental_data['performance_data']['completion_times'].items()},
                'learning_curves': experimental_data['performance_data']['learning_curves']
            },
            'safety_data': {
                'distances': experimental_data['safety_data']['distances'],
                'timestamps': experimental_data['safety_data']['timestamps'],
                'safety_threshold': experimental_data['safety_data']['safety_threshold'],
                'violation_count': len(experimental_data['safety_data']['violations'])
            },
            'experimental_setup': experimental_data['experimental_setup']
        }
        
        with open(output_dir / 'experimental_data.json', 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        # Create summary report
        elapsed_time = time.time() - start_time
        
        summary = {
            'demonstration_completed': datetime.now().isoformat(),
            'elapsed_time_seconds': elapsed_time,
            'components_demonstrated': components,
            'output_formats': formats,
            'total_files_generated': len(list(output_dir.glob('*'))),
            'key_findings': [
                f"Best performing method: {max(experimental_data['comparison_data'].keys(), key=lambda x: np.mean(experimental_data['comparison_data'][x]))}",
                f"Safety violations detected: {len(experimental_data['safety_data']['violations'])}",
                f"Statistical tests performed: Multiple comparison analysis with FDR correction",
                "Comprehensive visualization suite successfully demonstrated"
            ]
        }
        
        with open(output_dir / 'demonstration_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Demonstration completed successfully in {elapsed_time:.1f} seconds")
        logger.info(f"Generated {len(list(output_dir.glob('*')))} output files")
        logger.info(f"Results saved to: {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("VISUALIZATION DEMONSTRATION COMPLETED")
        print("="*60)
        print(f"Output Directory: {output_dir}")
        print(f"Components: {', '.join(components)}")
        print(f"Formats: {', '.join(formats)}")
        print(f"Execution Time: {elapsed_time:.1f} seconds")
        print(f"Files Generated: {len(list(output_dir.glob('*')))}")
        print("\nKey Outputs:")
        
        for component in components:
            if component == 'performance':
                print("  • Performance Analysis: success_rate_comparison.png, learning_curves.png")
            elif component == 'safety':
                print("  • Safety Analysis: distance_monitoring.png, risk_assessment.png")
            elif component == 'bayesian':
                print("  • Bayesian Analysis: posterior_evolution.png, uncertainty_decomposition.png")
            elif component == 'stats':
                print("  • Statistical Analysis: statistical_summary.png, effect_sizes.png")
            elif component == 'dashboard':
                print("  • Interactive Dashboards: interactive_dashboard.html, parameter_sensitivity.html")
            elif component == 'realtime':
                print("  • Real-time System: realtime_data.json, streaming_plot.png")
            elif component == 'reports':
                print("  • Automated Reports: comprehensive_report.html, publication_figures/")
        
        print("\nAll visualization components successfully demonstrated!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()