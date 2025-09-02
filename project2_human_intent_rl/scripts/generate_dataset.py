"""
Script to generate and validate synthetic human behavior dataset.

This script generates a comprehensive dataset of 1000+ synthetic human behavior
sequences for training and testing the human intent recognition system.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.synthetic_generator import (
    SyntheticHumanBehaviorGenerator,
    GestureType
)
from src.data.validation import DataValidator, ValidationLevel
from src.data.feature_extraction import FeatureExtractor
from src.visualization.behavior_plots import BehaviorVisualizer
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic human behavior dataset")
    
    parser.add_argument(
        '--n_sequences', 
        type=int, 
        default=1200,
        help='Number of sequences to generate (default: 1200)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/synthetic',
        help='Output directory for dataset (default: data/synthetic)'
    )
    
    parser.add_argument(
        '--workspace_bounds',
        nargs=6,
        type=float,
        default=[-1.0, 1.0, -1.0, 1.0, 0.0, 2.0],
        help='Workspace bounds: x_min x_max y_min y_max z_min z_max'
    )
    
    parser.add_argument(
        '--sampling_rate',
        type=float,
        default=30.0,
        help='Sampling frequency in Hz (default: 30.0)'
    )
    
    parser.add_argument(
        '--noise_range',
        nargs=2,
        type=float,
        default=[0.005, 0.03],
        help='Noise range: min_noise max_noise (default: 0.005 0.03)'
    )
    
    parser.add_argument(
        '--validation_level',
        type=str,
        choices=['basic', 'standard', 'strict'],
        default='standard',
        help='Data validation level (default: standard)'
    )
    
    parser.add_argument(
        '--generate_plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--extract_features',
        action='store_true',
        help='Extract features from generated sequences'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducible generation (default: 42)'
    )
    
    return parser.parse_args()


def create_gesture_distribution() -> Dict[GestureType, float]:
    """Create realistic distribution of gesture types."""
    return {
        GestureType.REACH: 0.30,      # Most common - reaching for objects
        GestureType.GRAB: 0.25,       # Grasping objects
        GestureType.HANDOVER: 0.20,   # Human-robot handovers
        GestureType.POINT: 0.15,      # Pointing gestures
        GestureType.WAVE: 0.08,       # Communication gestures
        GestureType.IDLE: 0.02        # Idle/waiting states
    }


def generate_synthetic_dataset(args: argparse.Namespace) -> List:
    """
    Generate synthetic dataset with specified parameters.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of synthetic sequences
    """
    logger.info("Initializing synthetic behavior generator...")
    
    # Create workspace bounds
    workspace_bounds = np.array(args.workspace_bounds)
    
    # Initialize generator
    generator = SyntheticHumanBehaviorGenerator(
        workspace_bounds=workspace_bounds,
        sampling_frequency=args.sampling_rate,
        random_seed=args.random_seed
    )
    
    # Create gesture distribution
    gesture_distribution = create_gesture_distribution()
    
    logger.info(f"Generating {args.n_sequences} sequences...")
    logger.info(f"Workspace bounds: {workspace_bounds}")
    logger.info(f"Sampling rate: {args.sampling_rate} Hz")
    logger.info(f"Noise range: {args.noise_range}")
    logger.info(f"Gesture distribution: {gesture_distribution}")
    
    # Generate dataset
    sequences = generator.generate_dataset(
        n_sequences=args.n_sequences,
        gesture_distribution=gesture_distribution,
        noise_range=tuple(args.noise_range)
    )
    
    logger.info(f"Successfully generated {len(sequences)} sequences")
    
    return sequences


def validate_dataset(sequences: List, validation_level: str) -> tuple:
    """
    Validate generated dataset.
    
    Args:
        sequences: List of synthetic sequences
        validation_level: Level of validation to perform
        
    Returns:
        Tuple of (validation_results, batch_statistics)
    """
    logger.info(f"Validating dataset with {validation_level} validation...")
    
    # Map validation level
    level_map = {
        'basic': ValidationLevel.BASIC,
        'standard': ValidationLevel.STANDARD,
        'strict': ValidationLevel.STRICT
    }
    
    # Create validator
    validator = DataValidator(
        validation_level=level_map[validation_level],
        workspace_bounds=sequences[0].context_info['workspace_bounds'],
        expected_sampling_rate=30.0
    )
    
    # Validate batch
    results, batch_stats = validator.validate_batch(sequences)
    
    # Log validation summary
    logger.info(f"Validation complete:")
    logger.info(f"  Total sequences: {batch_stats['total_sequences']}")
    logger.info(f"  Valid sequences: {batch_stats['valid_sequences']}")
    logger.info(f"  Invalid sequences: {batch_stats['invalid_sequences']}")
    logger.info(f"  Average quality: {batch_stats['average_quality']:.3f}")
    
    # Log common issues
    if batch_stats['common_issues']:
        logger.info("Common issues found:")
        for issue, count in batch_stats['common_issues'].items():
            logger.info(f"  {issue}: {count}")
    
    return results, batch_stats


def extract_dataset_features(sequences: List, output_dir: Path) -> Optional[tuple]:
    """
    Extract features from dataset sequences.
    
    Args:
        sequences: List of synthetic sequences
        output_dir: Output directory for features
        
    Returns:
        Tuple of (feature_matrix, feature_names, sequence_ids) or None
    """
    logger.info("Extracting features from dataset...")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    try:
        # Extract features
        feature_matrix, feature_names, sequence_ids = extractor.extract_batch_features(
            sequences, normalize=True
        )
        
        # Save features
        features_path = output_dir / "features.csv"
        extractor.save_features(
            feature_matrix, feature_names, sequence_ids, str(features_path)
        )
        
        logger.info(f"Extracted {feature_matrix.shape[1]} features from {len(sequences)} sequences")
        logger.info(f"Features saved to {features_path}")
        
        return feature_matrix, feature_names, sequence_ids
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return None


def generate_visualization_plots(sequences: List, output_dir: Path, max_plots: int = 10) -> None:
    """
    Generate visualization plots for sample sequences.
    
    Args:
        sequences: List of synthetic sequences
        output_dir: Output directory for plots
        max_plots: Maximum number of plots to generate
    """
    logger.info(f"Generating visualization plots for {min(max_plots, len(sequences))} sequences...")
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = BehaviorVisualizer()
    
    # Select diverse sequences for plotting
    gesture_counts = {}
    selected_sequences = []
    
    for sequence in sequences:
        gesture = sequence.gesture_type
        if gesture not in gesture_counts:
            gesture_counts[gesture] = 0
        
        if gesture_counts[gesture] < max_plots // len(GestureType) + 1:
            selected_sequences.append(sequence)
            gesture_counts[gesture] += 1
        
        if len(selected_sequences) >= max_plots:
            break
    
    # Generate plots for selected sequences
    for i, sequence in enumerate(selected_sequences):
        try:
            logger.info(f"Generating plots for sequence {i+1}/{len(selected_sequences)}: {sequence.sequence_id}")
            
            # 3D trajectory plot
            fig = visualizer.plot_trajectory_3d(
                sequence, 
                save_path=str(plots_dir / f"{sequence.sequence_id}_trajectory_3d.png")
            )
            
            # Time series plot
            fig_ts = visualizer.plot_trajectory_time_series(
                sequence,
                save_path=str(plots_dir / f"{sequence.sequence_id}_timeseries.png")
            )
            
            # Close figures to save memory
            import matplotlib.pyplot as plt
            plt.close(fig)
            plt.close(fig_ts)
            
        except Exception as e:
            logger.warning(f"Failed to generate plots for {sequence.sequence_id}: {e}")
    
    logger.info(f"Plots saved to {plots_dir}")


def save_dataset_summary(sequences: List, validation_results: List, batch_stats: Dict, 
                        output_dir: Path) -> None:
    """
    Save dataset summary and statistics.
    
    Args:
        sequences: Generated sequences
        validation_results: Validation results
        batch_stats: Batch validation statistics
        output_dir: Output directory
    """
    # Collect dataset statistics
    gesture_counts = {}
    duration_stats = []
    quality_scores = [result.quality_score for result in validation_results]
    
    for sequence in sequences:
        gesture = sequence.gesture_type
        gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        duration = sequence.timestamps[-1] - sequence.timestamps[0]
        duration_stats.append(duration)
    
    # Create summary
    summary = {
        'generation_timestamp': datetime.now().isoformat(),
        'total_sequences': len(sequences),
        'valid_sequences': batch_stats['valid_sequences'],
        'invalid_sequences': batch_stats['invalid_sequences'],
        'validation_success_rate': batch_stats['valid_sequences'] / len(sequences),
        'average_quality_score': batch_stats['average_quality'],
        'quality_score_std': batch_stats.get('quality_std', 0),
        'gesture_distribution': {
            gesture.value: count for gesture, count in gesture_counts.items()
        },
        'duration_statistics': {
            'mean': float(np.mean(duration_stats)),
            'std': float(np.std(duration_stats)),
            'min': float(np.min(duration_stats)),
            'max': float(np.max(duration_stats))
        },
        'common_issues': batch_stats['common_issues'],
        'workspace_bounds': sequences[0].context_info['workspace_bounds'] if sequences else None,
        'sampling_frequency': 30.0
    }
    
    # Save summary
    import json
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Dataset summary saved to {summary_path}")
    
    # Print summary to console
    logger.info("Dataset Generation Summary:")
    logger.info(f"  Total sequences: {summary['total_sequences']}")
    logger.info(f"  Valid sequences: {summary['valid_sequences']} ({summary['validation_success_rate']:.1%})")
    logger.info(f"  Average quality: {summary['average_quality_score']:.3f}")
    logger.info(f"  Average duration: {summary['duration_statistics']['mean']:.2f}s")
    logger.info("  Gesture distribution:")
    for gesture, count in summary['gesture_distribution'].items():
        percentage = count / summary['total_sequences'] * 100
        logger.info(f"    {gesture}: {count} ({percentage:.1f}%)")


def main():
    """Main execution function."""
    args = parse_args()
    
    logger.info("Starting synthetic human behavior dataset generation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate synthetic dataset
        sequences = generate_synthetic_dataset(args)
        
        if not sequences:
            logger.error("No sequences were generated successfully")
            return 1
        
        # Save raw dataset
        dataset_path = output_dir / "synthetic_dataset"
        from src.data.synthetic_generator import SyntheticHumanBehaviorGenerator
        generator = SyntheticHumanBehaviorGenerator(
            workspace_bounds=np.array(args.workspace_bounds),
            sampling_frequency=args.sampling_rate
        )
        generator.save_dataset(sequences, str(dataset_path))
        
        # Validate dataset
        validation_results, batch_stats = validate_dataset(sequences, args.validation_level)
        
        # Extract features if requested
        features_data = None
        if args.extract_features:
            features_data = extract_dataset_features(sequences, output_dir)
        
        # Generate plots if requested
        if args.generate_plots:
            generate_visualization_plots(sequences, output_dir)
        
        # Save dataset summary
        save_dataset_summary(sequences, validation_results, batch_stats, output_dir)
        
        logger.info("Dataset generation completed successfully!")
        logger.info(f"Dataset saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)