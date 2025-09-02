"""
Dataset Quality Analyzer for Human-Robot Interaction Data

This module provides comprehensive analysis and validation tools for HRI datasets,
ensuring publication-quality realism and suitable training characteristics.

Features:
- Statistical analysis against human movement literature
- Intent pattern validation with temporal consistency
- Noise model verification and enhancement
- Dataset balance and coverage analysis
- Ground truth quality assessment
- Domain expert validation metrics

Author: Claude Code Research Team  
Date: 2024
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import time

try:
    from .enhanced_synthetic_generator import EnhancedSyntheticGenerator, BiomechanicalConstraints
    from ..utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class DatasetCoverageMetrics:
    """Metrics for dataset coverage analysis"""
    gesture_distribution: Dict[str, float]
    intent_coverage: Dict[str, float] 
    temporal_coverage: Dict[str, float]
    spatial_coverage: Dict[str, float]
    demographic_coverage: Dict[str, float]
    scenario_coverage: Dict[str, float]
    edge_case_coverage: float
    rare_event_coverage: float


@dataclass
class LiteratureComparisonData:
    """Reference data from human movement literature"""
    # Peak velocities from reaching studies (Jeannerod 1984, Flash & Hogan 1985)
    peak_velocity_mean: float = 1.2  # m/s
    peak_velocity_std: float = 0.4
    
    # Movement durations (Fitts 1954, MacKenzie 1992)  
    duration_fitts_a: float = 0.1  # seconds
    duration_fitts_b: float = 0.15  # seconds/bit
    
    # Path efficiency (Morasso 1981, Uno et al. 1989)
    path_efficiency_mean: float = 0.85
    path_efficiency_std: float = 0.08
    
    # Velocity profile characteristics (Gottlieb et al. 1989)
    time_to_peak_velocity_mean: float = 0.45  # 45% of movement time
    time_to_peak_velocity_std: float = 0.1
    
    # Jerk smoothness (Teulings et al. 1997, Balasubramanian et al. 2015)
    normalized_jerk_mean: float = 20.0
    normalized_jerk_std: float = 15.0
    
    # Intent transition patterns (Ansuini et al. 2015)
    intent_stability_probability: float = 0.92
    preparation_phase_duration: float = 0.15  # 15% of movement
    execution_phase_duration: float = 0.70   # 70% of movement


class DatasetQualityAnalyzer:
    """Comprehensive dataset quality analysis tool"""
    
    def __init__(
        self,
        literature_data: Optional[LiteratureComparisonData] = None,
        analysis_config: Optional[Dict] = None
    ):
        """
        Initialize dataset quality analyzer.
        
        Args:
            literature_data: Reference data from literature
            analysis_config: Analysis configuration parameters
        """
        self.literature_data = literature_data or LiteratureComparisonData()
        self.config = analysis_config or {
            'significance_threshold': 0.05,
            'effect_size_threshold': 0.5,
            'min_samples_per_class': 50,
            'coverage_threshold': 0.8,
            'quality_score_weights': {
                'realism': 0.3,
                'balance': 0.2,
                'coverage': 0.2,
                'consistency': 0.15,
                'noise_quality': 0.15
            }
        }
        
        # Initialize analysis results
        self.analysis_results = {}
        self.quality_scores = {}
        self.recommendations = []
        
        logger.info("Dataset quality analyzer initialized")
    
    def analyze_complete_dataset(
        self,
        dataset: List[Dict],
        save_report: bool = True,
        output_path: str = "dataset_quality_report"
    ) -> Dict[str, Any]:
        """
        Run complete dataset quality analysis.
        
        Args:
            dataset: List of data sequences
            save_report: Whether to save analysis report
            output_path: Path for saving report
            
        Returns:
            Comprehensive analysis report
        """
        logger.info(f"Starting comprehensive analysis of {len(dataset)} sequences")
        
        # Initialize report
        report = {
            'dataset_info': self._extract_dataset_info(dataset),
            'movement_realism_analysis': {},
            'intent_pattern_analysis': {},
            'noise_model_analysis': {},
            'balance_coverage_analysis': {},
            'ground_truth_analysis': {},
            'literature_comparison': {},
            'quality_assessment': {},
            'recommendations': [],
            'timestamp': time.time()
        }
        
        # Run analysis modules
        try:
            report['movement_realism_analysis'] = self.analyze_movement_realism(dataset)
            report['intent_pattern_analysis'] = self.analyze_intent_patterns(dataset)  
            report['noise_model_analysis'] = self.analyze_noise_models(dataset)
            report['balance_coverage_analysis'] = self.analyze_dataset_balance_coverage(dataset)
            report['ground_truth_analysis'] = self.analyze_ground_truth_quality(dataset)
            report['literature_comparison'] = self.compare_with_literature(dataset)
            
            # Compute overall quality assessment
            report['quality_assessment'] = self.compute_overall_quality(report)
            
            # Generate recommendations
            report['recommendations'] = self.generate_improvement_recommendations(report)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            report['error'] = str(e)
            
        # Save report
        if save_report:
            self.save_analysis_report(report, output_path)
        
        logger.info("Dataset analysis complete")
        return report
    
    def analyze_movement_realism(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Analyze movement trajectory realism"""
        logger.info("Analyzing movement realism...")
        
        # Extract movement data
        trajectories = []
        metrics = []
        
        for seq in dataset:
            if 'hand_trajectory' in seq and 'movement_metrics' in seq:
                trajectories.append(seq['hand_trajectory'])
                metrics.append(seq['movement_metrics'])
        
        if not metrics:
            return {'error': 'No movement metrics found in dataset'}
        
        # Analyze biomechanical realism
        biomech_analysis = self._analyze_biomechanical_constraints(metrics)
        
        # Analyze trajectory smoothness
        smoothness_analysis = self._analyze_trajectory_smoothness(trajectories, metrics)
        
        # Analyze velocity profiles
        velocity_analysis = self._analyze_velocity_profiles(trajectories, metrics)
        
        # Analyze acceleration patterns
        acceleration_analysis = self._analyze_acceleration_patterns(trajectories, metrics)
        
        # Analyze path characteristics  
        path_analysis = self._analyze_path_characteristics(metrics)
        
        return {
            'biomechanical_constraints': biomech_analysis,
            'trajectory_smoothness': smoothness_analysis,
            'velocity_profiles': velocity_analysis,
            'acceleration_patterns': acceleration_analysis,
            'path_characteristics': path_analysis,
            'realism_score': self._compute_realism_score(
                biomech_analysis, smoothness_analysis, velocity_analysis, 
                acceleration_analysis, path_analysis
            )
        }
    
    def _analyze_biomechanical_constraints(self, metrics: List[Dict]) -> Dict:
        """Analyze adherence to biomechanical constraints"""
        # Extract biomechanical parameters
        peak_velocities = [m.get('peak_velocity', 0) for m in metrics]
        peak_accelerations = [m.get('peak_acceleration', 0) for m in metrics]
        movement_durations = [m.get('movement_duration', 0) for m in metrics]
        
        # Human biomechanical limits
        constraints = BiomechanicalConstraints()
        
        # Check constraint violations
        velocity_violations = sum(1 for v in peak_velocities if v > constraints.max_hand_velocity)
        acceleration_violations = sum(1 for a in peak_accelerations if a > constraints.max_hand_acceleration)
        
        # Analyze duration realism (should follow power law for different distances)
        distances = [m.get('straight_line_distance', 0) for m in metrics]
        valid_pairs = [(d, dur) for d, dur in zip(distances, movement_durations) if d > 0 and dur > 0]
        
        if len(valid_pairs) > 10:
            distances_valid, durations_valid = zip(*valid_pairs)
            
            # Fit power law: duration = a * distance^b
            try:
                log_dist = np.log(distances_valid)
                log_dur = np.log(durations_valid)
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_dist, log_dur)
                
                power_law_fit_quality = r_value**2
                realistic_scaling = 0.3 <= slope <= 0.7  # Realistic power law exponent
            except:
                power_law_fit_quality = 0.0
                realistic_scaling = False
        else:
            power_law_fit_quality = 0.0
            realistic_scaling = False
        
        return {
            'velocity_constraint_adherence': 1 - (velocity_violations / len(peak_velocities)),
            'acceleration_constraint_adherence': 1 - (acceleration_violations / len(peak_accelerations)),
            'duration_scaling_realism': power_law_fit_quality,
            'realistic_scaling_exponent': realistic_scaling,
            'velocity_distribution': {
                'mean': np.mean(peak_velocities),
                'std': np.std(peak_velocities),
                'range': [np.min(peak_velocities), np.max(peak_velocities)]
            },
            'constraint_violations': {
                'velocity': velocity_violations,
                'acceleration': acceleration_violations
            }
        }
    
    def _analyze_trajectory_smoothness(self, trajectories: List[np.ndarray], metrics: List[Dict]) -> Dict:
        """Analyze trajectory smoothness using multiple metrics"""
        smoothness_scores = []
        spectral_characteristics = []
        
        for i, traj in enumerate(trajectories):
            if len(traj) < 10:
                continue
                
            # Compute jerk-based smoothness
            dt = 1.0 / 100.0  # Assume 100Hz sampling
            velocity = np.diff(traj, axis=0) / dt
            acceleration = np.diff(velocity, axis=0) / dt
            jerk = np.diff(acceleration, axis=0) / dt
            
            # SPARC (Spectral Arc Length) - Balasubramanian et al. 2015
            sparc_score = self._compute_sparc(velocity)
            smoothness_scores.append(sparc_score)
            
            # Spectral analysis
            for dim in range(traj.shape[1]):
                freqs, psd = signal.welch(traj[:, dim], fs=100, nperseg=min(256, len(traj)//4))
                # Power concentration in low frequencies indicates smoothness
                low_freq_power = np.sum(psd[freqs <= 5]) / np.sum(psd)
                spectral_characteristics.append(low_freq_power)
        
        return {
            'sparc_scores': {
                'mean': np.mean(smoothness_scores) if smoothness_scores else 0,
                'std': np.std(smoothness_scores) if smoothness_scores else 0,
                'distribution': smoothness_scores
            },
            'spectral_smoothness': {
                'mean_low_freq_power': np.mean(spectral_characteristics) if spectral_characteristics else 0,
                'smoothness_consistency': 1 - np.std(spectral_characteristics) if spectral_characteristics else 0
            },
            'movement_units': self._analyze_movement_submovements(trajectories),
            'smoothness_quality': self._rate_smoothness_quality(smoothness_scores, spectral_characteristics)
        }
    
    def _compute_sparc(self, velocity: np.ndarray) -> float:
        """Compute SPARC (Spectral Arc Length) smoothness metric"""
        if len(velocity) < 4:
            return 0.0
            
        # Compute magnitude of velocity
        vel_mag = np.linalg.norm(velocity, axis=1)
        
        # Frequency domain analysis
        n = len(vel_mag)
        freqs = np.fft.fftfreq(n, d=0.01)[:n//2]  # Positive frequencies
        vel_fft = np.fft.fft(vel_mag)[:n//2]
        
        # Normalize spectrum
        vel_spectrum = np.abs(vel_fft)
        vel_spectrum = vel_spectrum / np.max(vel_spectrum) if np.max(vel_spectrum) > 0 else vel_spectrum
        
        # Compute spectral arc length
        if len(vel_spectrum) > 1:
            arc_length = np.sum(np.sqrt(np.diff(freqs)**2 + np.diff(vel_spectrum)**2))
            # Normalize (negative because higher arc length = less smooth)
            sparc = -arc_length / len(vel_spectrum)
        else:
            sparc = 0.0
        
        return sparc
    
    def _analyze_movement_submovements(self, trajectories: List[np.ndarray]) -> Dict:
        """Analyze movement submovements (corrective movements)"""
        submovement_counts = []
        
        for traj in trajectories:
            if len(traj) < 10:
                continue
                
            # Compute velocity profile
            dt = 1.0 / 100.0
            velocity = np.diff(traj, axis=0) / dt
            vel_mag = np.linalg.norm(velocity, axis=1)
            
            # Find velocity peaks (submovements)
            peaks, properties = signal.find_peaks(vel_mag, 
                                                 prominence=0.1 * np.max(vel_mag),
                                                 distance=5)  # Minimum 50ms between peaks
            
            submovement_counts.append(len(peaks))
        
        return {
            'mean_submovements': np.mean(submovement_counts) if submovement_counts else 0,
            'submovement_distribution': Counter(submovement_counts),
            'single_movement_ratio': submovement_counts.count(1) / len(submovement_counts) if submovement_counts else 0,
            'complex_movement_ratio': sum(1 for c in submovement_counts if c > 2) / len(submovement_counts) if submovement_counts else 0
        }
    
    def _rate_smoothness_quality(self, sparc_scores: List[float], spectral_chars: List[float]) -> float:
        """Rate overall smoothness quality (0-1 scale)"""
        quality_factors = []
        
        if sparc_scores:
            # SPARC scores should be > -10 for smooth movements
            sparc_quality = np.mean([min(1.0, max(0.0, (s + 10) / 5)) for s in sparc_scores])
            quality_factors.append(sparc_quality)
        
        if spectral_chars:
            # Higher low-frequency power indicates smoother movement
            spectral_quality = np.mean(spectral_chars)
            quality_factors.append(spectral_quality)
        
        return np.mean(quality_factors) if quality_factors else 0.5
    
    def analyze_intent_patterns(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal intent patterns"""
        logger.info("Analyzing intent patterns...")
        
        # Extract intent sequences
        intent_sequences = []
        confidence_sequences = []
        temporal_patterns = []
        
        for seq in dataset:
            if 'intent_labels' in seq:
                intent_sequences.append(seq['intent_labels'])
                if 'confidence_scores' in seq:
                    confidence_sequences.append(seq['confidence_scores'])
                
                # Analyze temporal structure
                labels = seq['intent_labels']
                if len(labels) > 0:
                    temporal_patterns.append(self._analyze_sequence_temporal_pattern(labels))
        
        # Analyze phase structure
        phase_analysis = self._analyze_intent_phases(intent_sequences)
        
        # Analyze intent transitions
        transition_analysis = self._analyze_intent_transitions(intent_sequences)
        
        # Analyze confidence patterns
        confidence_analysis = self._analyze_confidence_patterns(confidence_sequences)
        
        # Validate against literature
        literature_validation = self._validate_intent_patterns_against_literature(
            temporal_patterns, transition_analysis
        )
        
        return {
            'phase_structure': phase_analysis,
            'intent_transitions': transition_analysis,
            'confidence_patterns': confidence_analysis,
            'temporal_consistency': self._compute_temporal_consistency(intent_sequences),
            'literature_validation': literature_validation,
            'intent_pattern_quality': self._compute_intent_quality_score(
                phase_analysis, transition_analysis, confidence_analysis
            )
        }
    
    def _analyze_sequence_temporal_pattern(self, intent_labels: List[str]) -> Dict:
        """Analyze temporal pattern of a single intent sequence"""
        if len(intent_labels) == 0:
            return {}
        
        # Find intent changes
        changes = []
        for i in range(1, len(intent_labels)):
            if intent_labels[i] != intent_labels[i-1]:
                changes.append(i / len(intent_labels))  # Normalized time
        
        # Identify phases
        unique_intents = []
        intent_durations = []
        current_intent = intent_labels[0]
        current_start = 0
        
        for i, intent in enumerate(intent_labels + [None]):  # Add None to handle last segment
            if intent != current_intent:
                unique_intents.append(current_intent)
                intent_durations.append((i - current_start) / len(intent_labels))
                if intent is not None:
                    current_intent = intent
                    current_start = i
        
        return {
            'num_phases': len(unique_intents),
            'phase_durations': intent_durations,
            'transition_times': changes,
            'dominant_intent': max(set(intent_labels), key=intent_labels.count),
            'intent_stability': intent_labels.count(max(set(intent_labels), key=intent_labels.count)) / len(intent_labels)
        }
    
    def _analyze_intent_phases(self, intent_sequences: List[List[str]]) -> Dict:
        """Analyze phase structure across all sequences with preparation-execution-completion validation"""
        phase_counts = []
        phase_durations = []
        temporal_phase_analysis = []
        
        for seq in intent_sequences:
            if len(seq) == 0:
                continue
                
            pattern = self._analyze_sequence_temporal_pattern(seq)
            if 'num_phases' in pattern:
                phase_counts.append(pattern['num_phases'])
                phase_durations.extend(pattern['phase_durations'])
                
                # Analyze temporal phase structure (preparation → execution → completion)
                temporal_phases = self._identify_temporal_phases(seq)
                temporal_phase_analysis.append(temporal_phases)
        
        # Validate preparation-execution-completion structure
        phase_structure_validation = self._validate_temporal_phase_structure(temporal_phase_analysis)
        
        return {
            'mean_phases_per_sequence': np.mean(phase_counts) if phase_counts else 0,
            'phase_count_distribution': Counter(phase_counts),
            'mean_phase_duration': np.mean(phase_durations) if phase_durations else 0,
            'phase_duration_std': np.std(phase_durations) if phase_durations else 0,
            'realistic_phase_structure': (
                2 <= np.mean(phase_counts) <= 4 if phase_counts else False
            ),
            'temporal_phase_structure': phase_structure_validation,
            'preparation_phase_characteristics': self._analyze_preparation_phases(temporal_phase_analysis),
            'execution_phase_characteristics': self._analyze_execution_phases(temporal_phase_analysis),
            'completion_phase_characteristics': self._analyze_completion_phases(temporal_phase_analysis)
        }
    
    def _analyze_intent_transitions(self, intent_sequences: List[List[str]]) -> Dict:
        """Analyze intent transition patterns"""
        all_transitions = []
        transition_counts = defaultdict(int)
        
        for seq in intent_sequences:
            seq_transitions = 0
            for i in range(1, len(seq)):
                if seq[i] != seq[i-1]:
                    transition = f"{seq[i-1]} -> {seq[i]}"
                    transition_counts[transition] += 1
                    all_transitions.append(transition)
                    seq_transitions += 1
        
        # Compute transition statistics
        transitions_per_sequence = []
        for seq in intent_sequences:
            if len(seq) > 0:
                seq_transitions = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
                transitions_per_sequence.append(seq_transitions / len(seq))
        
        return {
            'transition_frequency': np.mean(transitions_per_sequence) if transitions_per_sequence else 0,
            'common_transitions': dict(Counter(all_transitions).most_common(10)),
            'transition_diversity': len(set(all_transitions)),
            'stability_score': 1 - np.mean(transitions_per_sequence) if transitions_per_sequence else 0
        }
    
    def _analyze_confidence_patterns(self, confidence_sequences: List[np.ndarray]) -> Dict:
        """Analyze confidence score patterns"""
        if not confidence_sequences:
            return {'error': 'No confidence data available'}
        
        all_confidences = np.concatenate(confidence_sequences)
        
        # Analyze confidence evolution within sequences
        confidence_trends = []
        for conf_seq in confidence_sequences:
            if len(conf_seq) > 2:
                # Linear trend (confidence should generally increase)
                x = np.arange(len(conf_seq))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, conf_seq)
                confidence_trends.append({
                    'slope': slope,
                    'r_squared': r_value**2,
                    'initial_confidence': conf_seq[0],
                    'final_confidence': conf_seq[-1]
                })
        
        return {
            'confidence_distribution': {
                'mean': np.mean(all_confidences),
                'std': np.std(all_confidences),
                'range': [np.min(all_confidences), np.max(all_confidences)]
            },
            'confidence_evolution': {
                'mean_slope': np.mean([t['slope'] for t in confidence_trends]) if confidence_trends else 0,
                'positive_trends': sum(1 for t in confidence_trends if t['slope'] > 0) / len(confidence_trends) if confidence_trends else 0,
                'mean_r_squared': np.mean([t['r_squared'] for t in confidence_trends]) if confidence_trends else 0
            },
            'confidence_calibration': self._analyze_confidence_calibration(confidence_sequences)
        }
    
    def _analyze_confidence_calibration(self, confidence_sequences: List[np.ndarray]) -> Dict:
        """Analyze whether confidence scores are well-calibrated"""
        # For synthetic data, we can't easily validate calibration without ground truth
        # Instead, we check for reasonable patterns
        
        all_confidences = np.concatenate(confidence_sequences) if confidence_sequences else np.array([])
        
        if len(all_confidences) == 0:
            return {'error': 'No confidence data'}
        
        # Check for reasonable distribution (should not be too concentrated)
        confidence_entropy = stats.entropy(np.histogram(all_confidences, bins=10)[0] + 1e-8)
        max_entropy = np.log(10)  # Maximum entropy for 10 bins
        normalized_entropy = confidence_entropy / max_entropy
        
        # Check for temporal smoothness (confidence shouldn't jump around too much)
        temporal_smoothness = []
        for conf_seq in confidence_sequences:
            if len(conf_seq) > 1:
                smoothness = 1 - np.mean(np.abs(np.diff(conf_seq))) / np.std(conf_seq) if np.std(conf_seq) > 0 else 1
                temporal_smoothness.append(max(0, smoothness))
        
        return {
            'distribution_entropy': normalized_entropy,
            'temporal_smoothness': np.mean(temporal_smoothness) if temporal_smoothness else 0,
            'reasonable_range': 0.2 <= np.mean(all_confidences) <= 0.9,
            'calibration_quality': (normalized_entropy + np.mean(temporal_smoothness)) / 2 if temporal_smoothness else normalized_entropy / 2
        }
    
    def _identify_temporal_phases(self, intent_sequence: List[str]) -> Dict:
        """Identify preparation, execution, and completion phases in intent sequence"""
        if len(intent_sequence) < 3:
            return {'phases': [], 'phase_structure': 'insufficient_data'}
        
        # Analyze intent progression to identify phases
        unique_intents = []
        phase_boundaries = []
        
        current_intent = intent_sequence[0]
        current_start = 0
        
        for i, intent in enumerate(intent_sequence + [None]):
            if intent != current_intent:
                unique_intents.append({
                    'intent': current_intent,
                    'start_time': current_start / len(intent_sequence),
                    'end_time': i / len(intent_sequence),
                    'duration': (i - current_start) / len(intent_sequence)
                })
                phase_boundaries.append(i / len(intent_sequence))
                
                if intent is not None:
                    current_intent = intent
                    current_start = i
        
        # Classify phases based on intent progression patterns
        phases = self._classify_intent_phases(unique_intents, intent_sequence)
        
        return {
            'phases': phases,
            'phase_boundaries': phase_boundaries,
            'phase_structure': self._determine_phase_structure(phases),
            'preparation_duration': self._get_phase_duration(phases, 'preparation'),
            'execution_duration': self._get_phase_duration(phases, 'execution'),
            'completion_duration': self._get_phase_duration(phases, 'completion')
        }
    
    def _classify_intent_phases(self, unique_intents: List[Dict], intent_sequence: List[str]) -> List[Dict]:
        """Classify intent segments into preparation, execution, and completion phases"""
        phases = []
        
        # Identify likely phase patterns
        for i, intent_segment in enumerate(unique_intents):
            intent = intent_segment['intent']
            duration = intent_segment['duration']
            position_in_sequence = i / len(unique_intents) if len(unique_intents) > 1 else 0.5
            
            # Classification logic based on intent type and temporal position
            phase_type = self._determine_phase_type(intent, position_in_sequence, duration, i, len(unique_intents))
            
            phases.append({
                'phase_type': phase_type,
                'intent': intent,
                'start_time': intent_segment['start_time'],
                'end_time': intent_segment['end_time'],
                'duration': duration,
                'characteristics': self._analyze_phase_characteristics(intent, phase_type, intent_sequence)
            })
        
        return phases
    
    def _determine_phase_type(self, intent: str, position: float, duration: float, index: int, total_phases: int) -> str:
        """Determine phase type based on intent and temporal characteristics"""
        # Preparation phase indicators
        if (index == 0 or position < 0.3) and any(keyword in intent.lower() for keyword in 
            ['idle', 'rest', 'prepare', 'wait', 'approach', 'orient']):
            return 'preparation'
        
        # Completion phase indicators  
        if (index == total_phases - 1 or position > 0.7) and any(keyword in intent.lower() for keyword in
            ['complete', 'finish', 'rest', 'return', 'release']):
            return 'completion'
        
        # Execution phase indicators
        if any(keyword in intent.lower() for keyword in 
            ['reach', 'grasp', 'move', 'point', 'select', 'interact', 'execute']):
            return 'execution'
        
        # Default classification based on position
        if position < 0.33:
            return 'preparation'
        elif position > 0.67:
            return 'completion'
        else:
            return 'execution'
    
    def _analyze_phase_characteristics(self, intent: str, phase_type: str, sequence: List[str]) -> Dict:
        """Analyze characteristics specific to each phase type"""
        characteristics = {
            'intent_stability': sequence.count(intent) / len(sequence),
            'phase_consistency': 1.0  # Default for synthetic data
        }
        
        if phase_type == 'preparation':
            characteristics.update({
                'has_hesitation': 'wait' in intent.lower() or 'idle' in intent.lower(),
                'orientation_behavior': 'orient' in intent.lower() or 'approach' in intent.lower(),
                'planning_indicators': any(word in intent.lower() for word in ['plan', 'consider', 'decide'])
            })
        elif phase_type == 'execution':
            characteristics.update({
                'action_directness': not any(word in intent.lower() for word in ['hesitate', 'pause', 'wait']),
                'goal_oriented': any(word in intent.lower() for word in ['reach', 'grasp', 'select', 'move']),
                'correction_behavior': 'correct' in intent.lower() or 'adjust' in intent.lower()
            })
        elif phase_type == 'completion':
            characteristics.update({
                'task_finalization': any(word in intent.lower() for word in ['complete', 'finish', 'done']),
                'return_behavior': 'return' in intent.lower() or 'rest' in intent.lower(),
                'verification_behavior': 'check' in intent.lower() or 'verify' in intent.lower()
            })
        
        return characteristics
    
    def _determine_phase_structure(self, phases: List[Dict]) -> str:
        """Determine overall phase structure pattern"""
        if len(phases) < 2:
            return 'insufficient_phases'
        
        phase_types = [phase['phase_type'] for phase in phases]
        
        # Check for classic preparation → execution → completion pattern
        if len(phase_types) >= 3:
            if (phase_types[0] == 'preparation' and 
                'execution' in phase_types[1:-1] and 
                phase_types[-1] == 'completion'):
                return 'classic_three_phase'
        
        # Check for simplified patterns
        if len(phase_types) == 2:
            if phase_types == ['preparation', 'execution']:
                return 'preparation_execution'
            elif phase_types == ['execution', 'completion']:
                return 'execution_completion'
        
        # Check for execution-dominant pattern
        if phase_types.count('execution') > len(phase_types) / 2:
            return 'execution_dominant'
        
        return 'atypical_structure'
    
    def _get_phase_duration(self, phases: List[Dict], phase_type: str) -> float:
        """Get total duration of specific phase type"""
        total_duration = sum(phase['duration'] for phase in phases if phase['phase_type'] == phase_type)
        return total_duration
    
    def _validate_temporal_phase_structure(self, temporal_phase_analyses: List[Dict]) -> Dict:
        """Validate temporal phase structures against human behavior literature"""
        if not temporal_phase_analyses:
            return {'validation_status': 'no_data'}
        
        structure_counts = Counter()
        phase_duration_analysis = {'preparation': [], 'execution': [], 'completion': []}
        
        for analysis in temporal_phase_analyses:
            if 'phase_structure' in analysis:
                structure_counts[analysis['phase_structure']] += 1
            
            # Collect phase durations for analysis
            for phase_type in ['preparation', 'execution', 'completion']:
                duration_key = f'{phase_type}_duration'
                if duration_key in analysis:
                    phase_duration_analysis[phase_type].append(analysis[duration_key])
        
        # Literature-based validation
        validation_results = {
            'structure_distribution': dict(structure_counts),
            'classic_pattern_prevalence': structure_counts['classic_three_phase'] / len(temporal_phase_analyses),
            'phase_duration_realism': self._validate_phase_durations(phase_duration_analysis),
            'temporal_consistency': self._analyze_temporal_consistency(temporal_phase_analyses)
        }
        
        # Overall validation score
        classic_score = min(1.0, validation_results['classic_pattern_prevalence'] * 2)  # Target 50% classic patterns
        duration_score = validation_results['phase_duration_realism']['overall_realism']
        consistency_score = validation_results['temporal_consistency']['consistency_score']
        
        validation_results['overall_validation_score'] = (classic_score + duration_score + consistency_score) / 3
        validation_results['validation_status'] = 'pass' if validation_results['overall_validation_score'] > 0.7 else 'fail'
        
        return validation_results
    
    def _validate_phase_durations(self, phase_durations: Dict[str, List[float]]) -> Dict:
        """Validate phase durations against human behavior literature"""
        validation_results = {}
        
        # Literature-based expected durations (normalized)
        expected_durations = {
            'preparation': {'mean': 0.25, 'std': 0.1, 'range': (0.1, 0.4)},  # ~25% of gesture
            'execution': {'mean': 0.50, 'std': 0.15, 'range': (0.3, 0.7)},   # ~50% of gesture  
            'completion': {'mean': 0.25, 'std': 0.1, 'range': (0.1, 0.4)}    # ~25% of gesture
        }
        
        overall_scores = []
        
        for phase_type, durations in phase_durations.items():
            if not durations:
                validation_results[f'{phase_type}_validation'] = {'status': 'no_data', 'score': 0.0}
                continue
            
            expected = expected_durations[phase_type]
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            
            # Score based on how close to expected values
            mean_score = 1.0 - min(1.0, abs(mean_duration - expected['mean']) / expected['mean'])
            range_score = np.mean([(expected['range'][0] <= d <= expected['range'][1]) for d in durations])
            variability_score = 1.0 - min(1.0, abs(std_duration - expected['std']) / expected['std'])
            
            phase_score = (mean_score + range_score + variability_score) / 3
            overall_scores.append(phase_score)
            
            validation_results[f'{phase_type}_validation'] = {
                'mean_duration': mean_duration,
                'std_duration': std_duration,
                'expected_mean': expected['mean'],
                'within_expected_range': range_score,
                'score': phase_score,
                'status': 'pass' if phase_score > 0.6 else 'fail'
            }
        
        validation_results['overall_realism'] = np.mean(overall_scores) if overall_scores else 0.0
        return validation_results
    
    def _analyze_temporal_consistency(self, temporal_analyses: List[Dict]) -> Dict:
        """Analyze consistency of temporal patterns across sequences"""
        if not temporal_analyses:
            return {'consistency_score': 0.0, 'status': 'no_data'}
        
        # Analyze consistency of phase structures
        structures = [analysis.get('phase_structure', 'unknown') for analysis in temporal_analyses]
        structure_consistency = 1.0 - (len(set(structures)) - 1) / max(1, len(structures))
        
        # Analyze consistency of phase durations
        phase_duration_consistency = {}
        for phase_type in ['preparation', 'execution', 'completion']:
            durations = []
            for analysis in temporal_analyses:
                duration_key = f'{phase_type}_duration'
                if duration_key in analysis and analysis[duration_key] > 0:
                    durations.append(analysis[duration_key])
            
            if len(durations) > 1:
                cv = np.std(durations) / np.mean(durations)  # Coefficient of variation
                phase_duration_consistency[phase_type] = max(0.0, 1.0 - cv)  # Lower CV = higher consistency
            else:
                phase_duration_consistency[phase_type] = 0.0
        
        overall_duration_consistency = np.mean(list(phase_duration_consistency.values())) if phase_duration_consistency else 0.0
        consistency_score = (structure_consistency + overall_duration_consistency) / 2
        
        return {
            'structure_consistency': structure_consistency,
            'phase_duration_consistency': phase_duration_consistency,
            'overall_duration_consistency': overall_duration_consistency,
            'consistency_score': consistency_score,
            'status': 'consistent' if consistency_score > 0.7 else 'inconsistent'
        }
    
    def _analyze_preparation_phases(self, temporal_analyses: List[Dict]) -> Dict:
        """Analyze characteristics of preparation phases"""
        prep_characteristics = []
        prep_durations = []
        
        for analysis in temporal_analyses:
            if 'phases' in analysis:
                for phase in analysis['phases']:
                    if phase['phase_type'] == 'preparation':
                        prep_characteristics.append(phase['characteristics'])
                        prep_durations.append(phase['duration'])
        
        if not prep_characteristics:
            return {'status': 'no_preparation_phases_found'}
        
        # Analyze common characteristics
        hesitation_rate = np.mean([char.get('has_hesitation', False) for char in prep_characteristics])
        orientation_rate = np.mean([char.get('orientation_behavior', False) for char in prep_characteristics])
        planning_rate = np.mean([char.get('planning_indicators', False) for char in prep_characteristics])
        
        return {
            'average_duration': np.mean(prep_durations),
            'duration_std': np.std(prep_durations),
            'hesitation_behavior_rate': hesitation_rate,
            'orientation_behavior_rate': orientation_rate,
            'planning_behavior_rate': planning_rate,
            'realism_score': (hesitation_rate + orientation_rate + planning_rate) / 3,
            'count': len(prep_characteristics)
        }
    
    def _analyze_execution_phases(self, temporal_analyses: List[Dict]) -> Dict:
        """Analyze characteristics of execution phases"""
        exec_characteristics = []
        exec_durations = []
        
        for analysis in temporal_analyses:
            if 'phases' in analysis:
                for phase in analysis['phases']:
                    if phase['phase_type'] == 'execution':
                        exec_characteristics.append(phase['characteristics'])
                        exec_durations.append(phase['duration'])
        
        if not exec_characteristics:
            return {'status': 'no_execution_phases_found'}
        
        # Analyze execution characteristics
        directness_rate = np.mean([char.get('action_directness', False) for char in exec_characteristics])
        goal_orientation_rate = np.mean([char.get('goal_oriented', False) for char in exec_characteristics])
        correction_rate = np.mean([char.get('correction_behavior', False) for char in exec_characteristics])
        
        return {
            'average_duration': np.mean(exec_durations),
            'duration_std': np.std(exec_durations),
            'action_directness_rate': directness_rate,
            'goal_orientation_rate': goal_orientation_rate,
            'correction_behavior_rate': correction_rate,
            'execution_efficiency': (directness_rate + goal_orientation_rate) / 2,
            'realism_score': (directness_rate + goal_orientation_rate + (1 - correction_rate * 0.5)) / 3,
            'count': len(exec_characteristics)
        }
    
    def _analyze_completion_phases(self, temporal_analyses: List[Dict]) -> Dict:
        """Analyze characteristics of completion phases"""
        comp_characteristics = []
        comp_durations = []
        
        for analysis in temporal_analyses:
            if 'phases' in analysis:
                for phase in analysis['phases']:
                    if phase['phase_type'] == 'completion':
                        comp_characteristics.append(phase['characteristics'])
                        comp_durations.append(phase['duration'])
        
        if not comp_characteristics:
            return {'status': 'no_completion_phases_found'}
        
        # Analyze completion characteristics
        finalization_rate = np.mean([char.get('task_finalization', False) for char in comp_characteristics])
        return_behavior_rate = np.mean([char.get('return_behavior', False) for char in comp_characteristics])
        verification_rate = np.mean([char.get('verification_behavior', False) for char in comp_characteristics])
        
        return {
            'average_duration': np.mean(comp_durations),
            'duration_std': np.std(comp_durations),
            'task_finalization_rate': finalization_rate,
            'return_behavior_rate': return_behavior_rate,
            'verification_behavior_rate': verification_rate,
            'completion_quality': (finalization_rate + return_behavior_rate + verification_rate) / 3,
            'realism_score': (finalization_rate + return_behavior_rate + verification_rate * 0.5) / 2.5,
            'count': len(comp_characteristics)
        }
    
    def analyze_ground_truth_quality(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Comprehensive ground truth quality and consistency analysis"""
        logger.info("Analyzing ground truth quality and consistency...")
        
        # Initialize analysis containers
        label_consistency_analysis = self._analyze_label_consistency(dataset)
        confidence_quality_analysis = self._analyze_confidence_scores_quality(dataset)
        temporal_evolution_analysis = self._analyze_temporal_intent_evolution(dataset)
        multi_modal_validation = self._perform_multi_modal_ground_truth_validation(dataset)
        ambiguity_handling_analysis = self._analyze_ambiguity_handling(dataset)
        
        # Cross-validation consistency checks
        cross_validation_results = self._perform_ground_truth_cross_validation(dataset)
        
        # Compute overall ground truth quality score
        gt_quality_components = {
            'label_consistency': label_consistency_analysis.get('consistency_score', 0),
            'confidence_calibration': confidence_quality_analysis.get('calibration_score', 0),
            'temporal_coherence': temporal_evolution_analysis.get('coherence_score', 0),
            'multi_modal_agreement': multi_modal_validation.get('agreement_score', 0),
            'ambiguity_handling': ambiguity_handling_analysis.get('handling_quality', 0)
        }
        
        overall_gt_quality = np.mean(list(gt_quality_components.values()))
        
        return {
            'label_consistency': label_consistency_analysis,
            'confidence_quality': confidence_quality_analysis,
            'temporal_evolution': temporal_evolution_analysis,
            'multi_modal_validation': multi_modal_validation,
            'ambiguity_handling': ambiguity_handling_analysis,
            'cross_validation': cross_validation_results,
            'quality_components': gt_quality_components,
            'overall_quality_score': overall_gt_quality,
            'quality_grade': self._grade_ground_truth_quality(overall_gt_quality),
            'recommendations': self._generate_ground_truth_recommendations(
                label_consistency_analysis, confidence_quality_analysis, 
                temporal_evolution_analysis, ambiguity_handling_analysis
            )
        }
    
    def analyze_dataset_balance_coverage(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Analyze dataset balance and coverage"""
        logger.info("Analyzing dataset balance and coverage...")
        
        # Extract relevant features
        gesture_types = []
        intent_types = []
        durations = []
        spatial_positions = []
        noise_levels = []
        
        for seq in dataset:
            if 'gesture_type' in seq:
                gesture_types.append(seq['gesture_type'])
            
            if 'intent_labels' in seq:
                intent_types.extend(seq['intent_labels'])
            
            if 'movement_metrics' in seq and 'movement_duration' in seq['movement_metrics']:
                durations.append(seq['movement_metrics']['movement_duration'])
            
            if 'hand_trajectory' in seq:
                # Sample spatial positions from trajectory
                traj = seq['hand_trajectory']
                if len(traj) > 0:
                    spatial_positions.extend(traj[::max(1, len(traj)//10)])  # Sample 10 points
            
            if 'noise_level' in seq:
                noise_levels.append(seq['noise_level'])
        
        # Analyze balance
        balance_analysis = {
            'gesture_balance': self._analyze_class_balance(gesture_types, 'gesture'),
            'intent_balance': self._analyze_class_balance(intent_types, 'intent'),
            'duration_coverage': self._analyze_continuous_coverage(durations, 'duration'),
            'spatial_coverage': self._analyze_spatial_coverage(spatial_positions),
            'noise_diversity': self._analyze_continuous_coverage(noise_levels, 'noise')
        }
        
        # Compute coverage metrics
        coverage_metrics = DatasetCoverageMetrics(
            gesture_distribution=dict(Counter(gesture_types)),
            intent_coverage=dict(Counter(intent_types)),
            temporal_coverage={'duration_range': [min(durations), max(durations)] if durations else [0, 0]},
            spatial_coverage=self._compute_spatial_coverage_metrics(spatial_positions),
            demographic_coverage={'diversity_score': 0.7},  # Placeholder - would analyze user demographics
            scenario_coverage={'scenario_diversity': len(set(gesture_types))},
            edge_case_coverage=self._compute_edge_case_coverage(dataset),
            rare_event_coverage=self._compute_rare_event_coverage(dataset)
        )
        
        return {
            'balance_analysis': balance_analysis,
            'coverage_metrics': coverage_metrics,
            'overall_balance_score': self._compute_overall_balance_score(balance_analysis),
            'coverage_completeness': self._compute_coverage_completeness(coverage_metrics),
            'recommendations': self._generate_balance_recommendations(balance_analysis, coverage_metrics)
        }
    
    def _analyze_class_balance(self, class_labels: List[str], class_type: str) -> Dict:
        """Analyze balance of categorical classes"""
        if not class_labels:
            return {'error': f'No {class_type} data available'}
        
        class_counts = Counter(class_labels)
        total_samples = len(class_labels)
        
        # Compute balance metrics
        class_proportions = {k: v/total_samples for k, v in class_counts.items()}
        
        # Entropy-based balance measure
        proportions = np.array(list(class_proportions.values()))
        entropy = -np.sum(proportions * np.log2(proportions + 1e-8))
        max_entropy = np.log2(len(class_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Imbalance ratio (max class / min class)
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values()) if class_counts else 1
        
        # Check for adequate representation
        min_samples_threshold = self.config['min_samples_per_class']
        underrepresented_classes = [k for k, v in class_counts.items() if v < min_samples_threshold]
        
        return {
            'class_counts': dict(class_counts),
            'class_proportions': class_proportions,
            'balance_entropy': normalized_entropy,
            'imbalance_ratio': imbalance_ratio,
            'underrepresented_classes': underrepresented_classes,
            'balance_quality': min(1.0, normalized_entropy * (2.0 / imbalance_ratio))
        }
    
    def _analyze_continuous_coverage(self, values: List[float], value_type: str) -> Dict:
        """Analyze coverage of continuous variables"""
        if not values:
            return {'error': f'No {value_type} data available'}
        
        values = np.array(values)
        
        # Coverage statistics
        coverage_stats = {
            'range': [float(np.min(values)), float(np.max(values))],
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'distribution_shape': self._assess_distribution_shape(values)
        }
        
        # Coverage uniformity (using histogram)
        hist, bin_edges = np.histogram(values, bins=10)
        coverage_uniformity = 1 - (np.std(hist) / np.mean(hist)) if np.mean(hist) > 0 else 0
        
        return {
            'coverage_statistics': coverage_stats,
            'coverage_uniformity': max(0, coverage_uniformity),
            'coverage_completeness': len(set(np.round(values, 2))) / len(values),  # Diversity measure
            'coverage_quality': (coverage_uniformity + (len(set(np.round(values, 2))) / len(values))) / 2
        }
    
    def _assess_distribution_shape(self, values: np.ndarray) -> str:
        """Assess the shape of a distribution"""
        if len(values) < 10:
            return 'insufficient_data'
        
        # Test for normality
        _, p_normal = stats.shapiro(values[:5000])  # Limit for computational efficiency
        
        if p_normal > 0.05:
            return 'normal'
        
        # Check skewness
        skewness = stats.skew(values)
        if skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        
        # Check for uniform distribution
        _, p_uniform = stats.kstest(values, 'uniform')
        if p_uniform > 0.05:
            return 'uniform'
        
        return 'other'
    
    def _analyze_spatial_coverage(self, positions: List[np.ndarray]) -> Dict:
        """Analyze spatial coverage of movements"""
        if not positions:
            return {'error': 'No spatial position data available'}
        
        positions = np.array(positions)
        if len(positions.shape) != 2 or positions.shape[1] != 3:
            return {'error': 'Invalid spatial position data format'}
        
        # Compute workspace coverage
        workspace_bounds = [
            [np.min(positions[:, i]), np.max(positions[:, i])] for i in range(3)
        ]
        
        workspace_volume = np.prod([bounds[1] - bounds[0] for bounds in workspace_bounds])
        
        # Analyze coverage uniformity using clustering
        if len(positions) > 10:
            # Use K-means to find spatial clusters
            n_clusters = min(8, len(positions) // 5)  # Adaptive number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(positions)
            
            # Measure cluster balance
            cluster_counts = Counter(cluster_labels)
            cluster_balance = len(cluster_counts) / n_clusters  # How many clusters are used
            
            # Silhouette score for cluster quality
            silhouette_avg = silhouette_score(positions, cluster_labels)
        else:
            cluster_balance = 1.0
            silhouette_avg = 0.5
        
        return {
            'workspace_bounds': workspace_bounds,
            'workspace_volume': workspace_volume,
            'spatial_distribution_quality': cluster_balance,
            'clustering_quality': silhouette_avg,
            'coverage_score': (cluster_balance + max(0, silhouette_avg)) / 2
        }
    
    def compare_with_literature(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Compare dataset characteristics with human movement literature"""
        logger.info("Comparing with literature...")
        
        # Extract movement metrics
        metrics = [seq.get('movement_metrics', {}) for seq in dataset if 'movement_metrics' in seq]
        
        if not metrics:
            return {'error': 'No movement metrics available for literature comparison'}
        
        # Compare key parameters
        comparisons = {}
        
        # Peak velocity comparison
        peak_vels = [m.get('peak_velocity', 0) for m in metrics if 'peak_velocity' in m]
        if peak_vels:
            comparisons['peak_velocity'] = self._compare_parameter_to_literature(
                peak_vels, 
                self.literature_data.peak_velocity_mean,
                self.literature_data.peak_velocity_std,
                'Peak Velocity (m/s)'
            )
        
        # Movement duration comparison (via Fitts' law)
        fitts_ratios = [m.get('fitts_law_ratio', 1) for m in metrics if 'fitts_law_ratio' in m]
        if fitts_ratios:
            comparisons['fitts_law_adherence'] = self._compare_parameter_to_literature(
                fitts_ratios, 1.0, 0.2, 'Fitts Law Ratio'
            )
        
        # Path efficiency comparison
        path_effs = [m.get('path_efficiency', 0) for m in metrics if 'path_efficiency' in m]
        if path_effs:
            comparisons['path_efficiency'] = self._compare_parameter_to_literature(
                path_effs,
                self.literature_data.path_efficiency_mean,
                self.literature_data.path_efficiency_std,
                'Path Efficiency'
            )
        
        # Velocity profile timing
        time_to_peaks = [m.get('time_to_peak_velocity', 0.5) for m in metrics if 'time_to_peak_velocity' in m]
        if time_to_peaks:
            comparisons['velocity_profile_timing'] = self._compare_parameter_to_literature(
                time_to_peaks,
                self.literature_data.time_to_peak_velocity_mean,
                self.literature_data.time_to_peak_velocity_std,
                'Time to Peak Velocity'
            )
        
        # Movement smoothness
        jerk_scores = [m.get('normalized_jerk', 0) for m in metrics if 'normalized_jerk' in m]
        if jerk_scores:
            comparisons['movement_smoothness'] = self._compare_parameter_to_literature(
                jerk_scores,
                self.literature_data.normalized_jerk_mean,
                self.literature_data.normalized_jerk_std,
                'Normalized Jerk'
            )
        
        # Overall literature compliance
        compliance_scores = [comp['compliance_score'] for comp in comparisons.values() if 'compliance_score' in comp]
        overall_compliance = np.mean(compliance_scores) if compliance_scores else 0
        
        return {
            'parameter_comparisons': comparisons,
            'overall_literature_compliance': overall_compliance,
            'significant_deviations': [k for k, v in comparisons.items() 
                                     if v.get('significant_deviation', False)],
            'compliance_grade': self._grade_literature_compliance(overall_compliance)
        }
    
    def _compare_parameter_to_literature(
        self, 
        observed_values: List[float],
        literature_mean: float,
        literature_std: float,
        parameter_name: str
    ) -> Dict:
        """Compare observed parameter values to literature values"""
        
        observed_mean = np.mean(observed_values)
        observed_std = np.std(observed_values)
        
        # Statistical test (t-test assuming normal distribution)
        # Compare observed mean to literature mean
        t_stat = (observed_mean - literature_mean) / (observed_std / np.sqrt(len(observed_values)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(observed_values) - 1))
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((observed_std**2 + literature_std**2) / 2)
        cohens_d = abs(observed_mean - literature_mean) / pooled_std if pooled_std > 0 else 0
        
        # Determine significance
        significant_deviation = (p_value < self.config['significance_threshold'] and 
                               cohens_d > self.config['effect_size_threshold'])
        
        # Compliance score (1 = perfect match, 0 = very different)
        z_score = abs(observed_mean - literature_mean) / literature_std if literature_std > 0 else 0
        compliance_score = max(0, 1 - z_score / 3)  # Within 3 standard deviations = good
        
        return {
            'observed_mean': observed_mean,
            'observed_std': observed_std,
            'literature_mean': literature_mean,
            'literature_std': literature_std,
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d
            },
            'significant_deviation': significant_deviation,
            'compliance_score': compliance_score,
            'parameter_name': parameter_name
        }
    
    def _grade_literature_compliance(self, compliance_score: float) -> str:
        """Assign letter grade to literature compliance"""
        if compliance_score >= 0.9:
            return 'A'
        elif compliance_score >= 0.8:
            return 'B'
        elif compliance_score >= 0.7:
            return 'C'
        elif compliance_score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def compute_overall_quality(self, analysis_report: Dict) -> Dict[str, Any]:
        """Compute overall dataset quality score"""
        
        quality_components = {}
        weights = self.config['quality_score_weights']
        
        # Realism score
        movement_analysis = analysis_report.get('movement_realism_analysis', {})
        quality_components['realism'] = movement_analysis.get('realism_score', 0.5)
        
        # Balance score  
        balance_analysis = analysis_report.get('balance_coverage_analysis', {})
        quality_components['balance'] = balance_analysis.get('overall_balance_score', 0.5)
        
        # Coverage score
        quality_components['coverage'] = balance_analysis.get('coverage_completeness', 0.5)
        
        # Consistency score (from intent patterns)
        intent_analysis = analysis_report.get('intent_pattern_analysis', {})
        quality_components['consistency'] = intent_analysis.get('intent_pattern_quality', 0.5)
        
        # Noise quality score
        noise_analysis = analysis_report.get('noise_model_analysis', {})
        quality_components['noise_quality'] = noise_analysis.get('noise_realism_score', 0.5)
        
        # Weighted overall score
        overall_score = sum(weights[k] * v for k, v in quality_components.items() 
                          if k in weights)
        
        # Literature compliance bonus
        literature_compliance = analysis_report.get('literature_comparison', {}).get('overall_literature_compliance', 0)
        overall_score = min(1.0, overall_score + 0.1 * literature_compliance)
        
        return {
            'component_scores': quality_components,
            'weighted_overall_score': overall_score,
            'literature_compliance_bonus': 0.1 * literature_compliance,
            'final_quality_score': overall_score,
            'quality_grade': self._assign_quality_grade(overall_score),
            'publication_ready': overall_score >= 0.8
        }
    
    def _assign_quality_grade(self, score: float) -> str:
        """Assign quality grade based on overall score"""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Fair'
        elif score >= 0.6:
            return 'Poor'
        else:
            return 'Unacceptable'
    
    def _compute_edge_case_coverage(self, dataset: List[Dict]) -> float:
        """Compute coverage of edge cases and boundary conditions"""
        edge_case_indicators = {
            'workspace_boundary_reaches': 0,
            'minimum_duration_movements': 0,
            'maximum_velocity_movements': 0,
            'high_jerk_movements': 0,
            'occlusion_events': 0,
            'sensor_dropout_events': 0,
            'multi_intent_transitions': 0,
            'hesitation_corrections': 0
        }
        
        total_sequences = len(dataset)
        if total_sequences == 0:
            return 0.0
        
        for seq in dataset:
            # Check for workspace boundary reaches
            if 'hand_trajectory' in seq:
                trajectory = np.array(seq['hand_trajectory'])
                if len(trajectory) > 0:
                    # Check if trajectory reaches near workspace boundaries
                    distances_from_origin = np.linalg.norm(trajectory, axis=1)
                    if np.max(distances_from_origin) > 0.7:  # Near maximum reach
                        edge_case_indicators['workspace_boundary_reaches'] += 1
            
            # Check movement duration extremes
            if 'movement_metrics' in seq:
                duration = seq['movement_metrics'].get('movement_duration', 0)
                if duration < 0.3:  # Very fast movements
                    edge_case_indicators['minimum_duration_movements'] += 1
                    
                # Check velocity extremes
                peak_velocity = seq['movement_metrics'].get('peak_velocity', 0)
                if peak_velocity > 2.0:  # High velocity movements
                    edge_case_indicators['maximum_velocity_movements'] += 1
                
                # Check jerk levels
                jerk = seq['movement_metrics'].get('normalized_jerk', 0)
                if jerk > 100:  # High jerk (less smooth) movements
                    edge_case_indicators['high_jerk_movements'] += 1
            
            # Check for sensor issues
            if 'sensor_quality' in seq:
                if seq['sensor_quality'].get('occlusion_events', 0) > 0:
                    edge_case_indicators['occlusion_events'] += 1
                if seq['sensor_quality'].get('dropout_rate', 0) > 0.05:
                    edge_case_indicators['sensor_dropout_events'] += 1
            
            # Check intent complexity
            if 'intent_labels' in seq:
                unique_intents = len(set(seq['intent_labels']))
                if unique_intents > 2:  # Multiple intent transitions
                    edge_case_indicators['multi_intent_transitions'] += 1
            
            # Check for hesitations and corrections
            if 'hesitation_events' in seq and seq['hesitation_events'] > 0:
                edge_case_indicators['hesitation_corrections'] += 1
        
        # Compute coverage score
        expected_edge_case_coverage = {
            'workspace_boundary_reaches': 0.1,  # 10% should reach boundaries
            'minimum_duration_movements': 0.05,  # 5% fast movements
            'maximum_velocity_movements': 0.05,  # 5% high velocity
            'high_jerk_movements': 0.1,  # 10% less smooth movements
            'occlusion_events': 0.08,  # 8% with occlusions
            'sensor_dropout_events': 0.03,  # 3% with significant dropout
            'multi_intent_transitions': 0.15,  # 15% complex intent patterns
            'hesitation_corrections': 0.12   # 12% with hesitations
        }
        
        coverage_scores = []
        for edge_case, expected_rate in expected_edge_case_coverage.items():
            actual_rate = edge_case_indicators[edge_case] / total_sequences
            # Score based on how close to expected rate (penalty for too much or too little)
            if expected_rate > 0:
                score = 1.0 - abs(actual_rate - expected_rate) / expected_rate
                coverage_scores.append(max(0, score))
        
        return np.mean(coverage_scores) if coverage_scores else 0.0
    
    def _compute_rare_event_coverage(self, dataset: List[Dict]) -> float:
        """Compute coverage of rare but important events"""
        rare_event_indicators = {
            'trajectory_reversals': 0,
            'double_corrections': 0,
            'extreme_curvature_paths': 0,
            'velocity_plateaus': 0,
            'intent_ambiguity_periods': 0,
            'sensor_recovery_events': 0,
            'biomechanical_limit_approaches': 0,
            'environmental_disturbances': 0
        }
        
        total_sequences = len(dataset)
        if total_sequences == 0:
            return 0.0
        
        for seq in dataset:
            # Analyze trajectory characteristics for rare patterns
            if 'hand_trajectory' in seq and len(seq['hand_trajectory']) > 3:
                trajectory = np.array(seq['hand_trajectory'])
                
                # Check for trajectory reversals (change of direction)
                if len(trajectory) > 2:
                    velocity = np.diff(trajectory, axis=0)
                    speed = np.linalg.norm(velocity, axis=1)
                    # Look for direction reversals
                    for i in range(1, len(velocity)):
                        if len(velocity) > i:
                            dot_product = np.dot(velocity[i-1], velocity[i])
                            if dot_product < -0.5 * speed[i-1] * speed[i]:  # Significant reversal
                                rare_event_indicators['trajectory_reversals'] += 1
                                break
                
                # Check for extreme curvature
                if 'movement_metrics' in seq:
                    path_curvature = seq['movement_metrics'].get('path_curvature', 0)
                    if path_curvature > 2.0:  # High curvature path
                        rare_event_indicators['extreme_curvature_paths'] += 1
            
            # Check for double corrections (multiple hesitations)
            if 'hesitation_events' in seq and seq['hesitation_events'] > 1:
                rare_event_indicators['double_corrections'] += 1
            
            # Check for velocity plateaus (constant speed periods)
            if 'velocity_profile_characteristics' in seq:
                if seq['velocity_profile_characteristics'].get('plateau_detected', False):
                    rare_event_indicators['velocity_plateaus'] += 1
            
            # Check for intent ambiguity
            if 'confidence_scores' in seq:
                confidence = np.array(seq['confidence_scores'])
                low_confidence_periods = np.sum(confidence < 0.6) / len(confidence)
                if low_confidence_periods > 0.3:  # More than 30% low confidence
                    rare_event_indicators['intent_ambiguity_periods'] += 1
            
            # Check for sensor recovery (from dropouts)
            if 'sensor_quality' in seq:
                recovery_events = seq['sensor_quality'].get('recovery_events', 0)
                if recovery_events > 0:
                    rare_event_indicators['sensor_recovery_events'] += 1
            
            # Check biomechanical limit approaches
            if 'biomechanical_stress' in seq:
                max_stress = seq['biomechanical_stress'].get('max_joint_stress', 0)
                if max_stress > 0.8:  # Approaching limits
                    rare_event_indicators['biomechanical_limit_approaches'] += 1
        
        # Expected rare event rates (these should be infrequent but present)
        expected_rare_event_rates = {
            'trajectory_reversals': 0.02,  # 2% with reversals
            'double_corrections': 0.03,   # 3% with multiple corrections
            'extreme_curvature_paths': 0.05,  # 5% with high curvature
            'velocity_plateaus': 0.01,    # 1% with plateaus
            'intent_ambiguity_periods': 0.08,  # 8% with ambiguity
            'sensor_recovery_events': 0.02,    # 2% with sensor recovery
            'biomechanical_limit_approaches': 0.03,  # 3% near limits
            'environmental_disturbances': 0.01     # 1% with disturbances
        }
        
        coverage_scores = []
        for rare_event, expected_rate in expected_rare_event_rates.items():
            actual_rate = rare_event_indicators[rare_event] / total_sequences
            # For rare events, we want some presence but not too much
            if expected_rate > 0:
                if actual_rate == 0:
                    score = 0.0  # Missing rare events is bad
                elif actual_rate <= expected_rate * 2:
                    score = min(1.0, actual_rate / expected_rate)  # Good coverage
                else:
                    score = max(0.0, 1.0 - (actual_rate - expected_rate * 2) / expected_rate)  # Too many
                coverage_scores.append(score)
        
        return np.mean(coverage_scores) if coverage_scores else 0.0
    
    def _compute_spatial_coverage_metrics(self, positions: List[np.ndarray]) -> Dict:
        """Compute detailed spatial coverage metrics"""
        if not positions:
            return {'coverage_score': 0.0, 'workspace_utilization': 0.0}
        
        positions = np.array(positions)
        if len(positions.shape) != 2 or positions.shape[1] != 3:
            return {'coverage_score': 0.0, 'workspace_utilization': 0.0}
        
        # Compute workspace bounds
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        workspace_volume = np.prod(max_bounds - min_bounds)
        
        # Expected human workspace (sphere with ~80cm radius)
        expected_workspace_volume = (4/3) * np.pi * (0.8)**3
        workspace_utilization = min(1.0, workspace_volume / expected_workspace_volume)
        
        # Compute coverage uniformity using voxel analysis
        n_voxels = 10  # 10x10x10 voxel grid
        coverage_grid = np.zeros((n_voxels, n_voxels, n_voxels))
        
        # Normalize positions to [0,1] range for each dimension
        normalized_positions = (positions - min_bounds) / (max_bounds - min_bounds + 1e-8)
        
        # Map to voxel indices
        voxel_indices = (normalized_positions * (n_voxels - 1)).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, n_voxels - 1)
        
        # Fill coverage grid
        for idx in voxel_indices:
            coverage_grid[tuple(idx)] = 1
        
        # Compute coverage metrics
        total_voxels = n_voxels ** 3
        covered_voxels = np.sum(coverage_grid > 0)
        coverage_uniformity = covered_voxels / total_voxels
        
        return {
            'coverage_score': (workspace_utilization + coverage_uniformity) / 2,
            'workspace_utilization': workspace_utilization,
            'coverage_uniformity': coverage_uniformity,
            'covered_workspace_fraction': covered_voxels / total_voxels,
            'workspace_bounds': [min_bounds.tolist(), max_bounds.tolist()],
            'effective_workspace_volume': workspace_volume
        }
    
    def _compute_overall_balance_score(self, balance_analysis: Dict) -> float:
        """Compute overall balance score from individual balance analyses"""
        balance_components = []
        
        for analysis_name, analysis_result in balance_analysis.items():
            if isinstance(analysis_result, dict) and 'balance_quality' in analysis_result:
                balance_components.append(analysis_result['balance_quality'])
            elif isinstance(analysis_result, dict) and 'coverage_score' in analysis_result:
                balance_components.append(analysis_result['coverage_score'])
        
        return np.mean(balance_components) if balance_components else 0.0
    
    def _compute_coverage_completeness(self, coverage_metrics) -> float:
        """Compute overall coverage completeness score"""
        coverage_scores = []
        
        # Extract coverage scores from different metrics
        if hasattr(coverage_metrics, 'spatial_coverage') and isinstance(coverage_metrics.spatial_coverage, dict):
            coverage_scores.append(coverage_metrics.spatial_coverage.get('coverage_score', 0))
        
        coverage_scores.append(coverage_metrics.edge_case_coverage if hasattr(coverage_metrics, 'edge_case_coverage') else 0)
        coverage_scores.append(coverage_metrics.rare_event_coverage if hasattr(coverage_metrics, 'rare_event_coverage') else 0)
        
        # Add demographic and scenario coverage
        if hasattr(coverage_metrics, 'demographic_coverage'):
            coverage_scores.append(coverage_metrics.demographic_coverage.get('diversity_score', 0))
        
        if hasattr(coverage_metrics, 'scenario_coverage'):
            scenario_diversity = coverage_metrics.scenario_coverage.get('scenario_diversity', 0)
            # Normalize scenario diversity (assume 8 different scenario types as good coverage)
            normalized_scenario = min(1.0, scenario_diversity / 8.0)
            coverage_scores.append(normalized_scenario)
        
        return np.mean(coverage_scores) if coverage_scores else 0.0
    
    def _generate_balance_recommendations(self, balance_analysis: Dict, coverage_metrics) -> List[str]:
        """Generate recommendations for improving dataset balance and coverage"""
        recommendations = []
        
        # Check gesture balance
        gesture_balance = balance_analysis.get('gesture_balance', {})
        if isinstance(gesture_balance, dict):
            balance_quality = gesture_balance.get('balance_quality', 1.0)
            if balance_quality < 0.7:
                underrepresented = gesture_balance.get('underrepresented_classes', [])
                if underrepresented:
                    recommendations.append(f"Increase samples for underrepresented gesture types: {', '.join(underrepresented)}")
                else:
                    recommendations.append("Improve gesture type balance across dataset")
        
        # Check intent balance
        intent_balance = balance_analysis.get('intent_balance', {})
        if isinstance(intent_balance, dict):
            balance_quality = intent_balance.get('balance_quality', 1.0)
            if balance_quality < 0.7:
                recommendations.append("Increase diversity and balance of intent patterns")
        
        # Check spatial coverage
        spatial_coverage = balance_analysis.get('spatial_coverage', {})
        if isinstance(spatial_coverage, dict):
            coverage_score = spatial_coverage.get('coverage_score', 1.0)
            if coverage_score < 0.6:
                recommendations.append("Expand spatial coverage to include more workspace regions")
        
        # Check edge case coverage
        if hasattr(coverage_metrics, 'edge_case_coverage'):
            if coverage_metrics.edge_case_coverage < 0.5:
                recommendations.append("Add more edge cases including boundary conditions and sensor failures")
        
        # Check rare event coverage
        if hasattr(coverage_metrics, 'rare_event_coverage'):
            if coverage_metrics.rare_event_coverage < 0.4:
                recommendations.append("Include more rare but realistic movement patterns and corrections")
        
        # Check duration coverage
        duration_coverage = balance_analysis.get('duration_coverage', {})
        if isinstance(duration_coverage, dict):
            coverage_quality = duration_coverage.get('coverage_quality', 1.0)
            if coverage_quality < 0.6:
                recommendations.append("Expand movement duration range to include both fast and slow movements")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _analyze_label_consistency(self, dataset: List[Dict]) -> Dict:
        """Analyze consistency of intent labels across similar contexts"""
        if not dataset:
            return {'consistency_score': 0.0, 'status': 'no_data'}
        
        consistency_metrics = {
            'intra_sequence_consistency': [],
            'inter_sequence_consistency': [],
            'context_dependent_consistency': [],
            'label_stability_periods': []
        }
        
        # Analyze intra-sequence consistency
        for seq in dataset:
            if 'intent_labels' in seq and len(seq['intent_labels']) > 1:
                labels = seq['intent_labels']
                
                # Measure label stability (consecutive identical labels)
                stability_periods = self._compute_label_stability_periods(labels)
                consistency_metrics['label_stability_periods'].extend(stability_periods)
                
                # Measure reasonable transition patterns
                transition_reasonableness = self._evaluate_transition_reasonableness(labels)
                consistency_metrics['intra_sequence_consistency'].append(transition_reasonableness)
        
        # Analyze inter-sequence consistency for similar contexts
        similarity_groups = self._group_sequences_by_similarity(dataset)
        for group in similarity_groups:
            if len(group) > 1:
                group_consistency = self._compute_group_label_consistency(group)
                consistency_metrics['inter_sequence_consistency'].append(group_consistency)
        
        # Compute overall consistency score
        intra_consistency = np.mean(consistency_metrics['intra_sequence_consistency']) if consistency_metrics['intra_sequence_consistency'] else 0
        inter_consistency = np.mean(consistency_metrics['inter_sequence_consistency']) if consistency_metrics['inter_sequence_consistency'] else 0
        
        overall_consistency = (intra_consistency + inter_consistency) / 2
        
        return {
            'intra_sequence_consistency': intra_consistency,
            'inter_sequence_consistency': inter_consistency,
            'label_stability_metrics': {
                'mean_stability_period': np.mean(consistency_metrics['label_stability_periods']) if consistency_metrics['label_stability_periods'] else 0,
                'stability_period_std': np.std(consistency_metrics['label_stability_periods']) if consistency_metrics['label_stability_periods'] else 0
            },
            'consistency_score': overall_consistency,
            'consistency_grade': 'Excellent' if overall_consistency > 0.9 else 'Good' if overall_consistency > 0.8 else 'Fair' if overall_consistency > 0.7 else 'Poor'
        }
    
    def _analyze_confidence_scores_quality(self, dataset: List[Dict]) -> Dict:
        """Analyze quality and calibration of confidence scores"""
        confidence_data = []
        label_changes = []
        
        for seq in dataset:
            if 'confidence_scores' in seq and 'intent_labels' in seq:
                confidence_scores = seq['confidence_scores']
                intent_labels = seq['intent_labels']
                
                if len(confidence_scores) == len(intent_labels):
                    confidence_data.extend(confidence_scores)
                    
                    # Analyze confidence around label changes
                    for i in range(1, len(intent_labels)):
                        if intent_labels[i] != intent_labels[i-1]:
                            # Confidence before and after label change
                            conf_before = confidence_scores[i-1] if i > 0 else 0.5
                            conf_after = confidence_scores[i]
                            label_changes.append({
                                'confidence_before': conf_before,
                                'confidence_after': conf_after,
                                'confidence_change': conf_after - conf_before
                            })
        
        if not confidence_data:
            return {'calibration_score': 0.0, 'status': 'no_confidence_data'}
        
        # Analyze confidence distribution
        confidence_array = np.array(confidence_data)
        distribution_analysis = {
            'mean_confidence': np.mean(confidence_array),
            'std_confidence': np.std(confidence_array),
            'confidence_range': [np.min(confidence_array), np.max(confidence_array)],
            'low_confidence_rate': np.mean(confidence_array < 0.5),
            'high_confidence_rate': np.mean(confidence_array > 0.8)
        }
        
        # Expected behavior: confidence should drop before uncertain transitions
        expected_behavior_score = 0.7  # Default if no transitions
        if label_changes:
            # Good confidence behavior: lower before changes, higher after
            before_change_appropriate = np.mean([c['confidence_before'] for c in label_changes]) < 0.7
            after_change_appropriate = np.mean([c['confidence_after'] for c in label_changes]) > 0.6
            expected_behavior_score = (before_change_appropriate + after_change_appropriate) / 2
        
        # Compute calibration score based on distribution and behavior
        distribution_score = 1.0 - abs(distribution_analysis['mean_confidence'] - 0.7)  # Target ~70% confidence
        range_score = min(1.0, (distribution_analysis['confidence_range'][1] - distribution_analysis['confidence_range'][0]) / 0.8)  # Good range utilization
        
        calibration_score = (distribution_score + range_score + expected_behavior_score) / 3
        
        return {
            'distribution_analysis': distribution_analysis,
            'expected_behavior_score': expected_behavior_score,
            'calibration_score': calibration_score,
            'calibration_quality': 'Well-calibrated' if calibration_score > 0.8 else 'Moderately calibrated' if calibration_score > 0.6 else 'Poorly calibrated'
        }
    
    def _analyze_temporal_intent_evolution(self, dataset: List[Dict]) -> Dict:
        """Analyze temporal evolution and coherence of intent labels"""
        evolution_metrics = {
            'smooth_transitions': 0,
            'abrupt_transitions': 0,
            'coherent_sequences': 0,
            'incoherent_sequences': 0,
            'temporal_consistency_scores': []
        }
        
        for seq in dataset:
            if 'intent_labels' in seq and len(seq['intent_labels']) > 2:
                labels = seq['intent_labels']
                
                # Analyze transition patterns
                transition_smoothness = self._compute_transition_smoothness(labels)
                evolution_metrics['temporal_consistency_scores'].append(transition_smoothness)
                
                # Count transition types
                abrupt_transitions = sum(1 for i in range(1, len(labels)) 
                                       if labels[i] != labels[i-1] and 
                                       not self._is_reasonable_transition(labels[i-1], labels[i]))
                
                smooth_transitions = sum(1 for i in range(1, len(labels)) 
                                       if labels[i] != labels[i-1] and 
                                       self._is_reasonable_transition(labels[i-1], labels[i]))
                
                evolution_metrics['smooth_transitions'] += smooth_transitions
                evolution_metrics['abrupt_transitions'] += abrupt_transitions
                
                # Evaluate sequence coherence
                sequence_coherence = self._evaluate_sequence_coherence(labels)
                if sequence_coherence > 0.7:
                    evolution_metrics['coherent_sequences'] += 1
                else:
                    evolution_metrics['incoherent_sequences'] += 1
        
        # Compute coherence score
        total_transitions = evolution_metrics['smooth_transitions'] + evolution_metrics['abrupt_transitions']
        transition_quality = (evolution_metrics['smooth_transitions'] / max(1, total_transitions))
        
        total_sequences = evolution_metrics['coherent_sequences'] + evolution_metrics['incoherent_sequences']
        sequence_quality = (evolution_metrics['coherent_sequences'] / max(1, total_sequences))
        
        temporal_consistency = np.mean(evolution_metrics['temporal_consistency_scores']) if evolution_metrics['temporal_consistency_scores'] else 0
        
        coherence_score = (transition_quality + sequence_quality + temporal_consistency) / 3
        
        return {
            'transition_analysis': {
                'smooth_transitions': evolution_metrics['smooth_transitions'],
                'abrupt_transitions': evolution_metrics['abrupt_transitions'],
                'transition_quality_score': transition_quality
            },
            'sequence_coherence': {
                'coherent_sequences': evolution_metrics['coherent_sequences'],
                'incoherent_sequences': evolution_metrics['incoherent_sequences'],
                'sequence_quality_score': sequence_quality
            },
            'temporal_consistency_score': temporal_consistency,
            'coherence_score': coherence_score,
            'coherence_grade': 'Excellent' if coherence_score > 0.9 else 'Good' if coherence_score > 0.8 else 'Fair' if coherence_score > 0.6 else 'Poor'
        }
    
    def _perform_multi_modal_ground_truth_validation(self, dataset: List[Dict]) -> Dict:
        """Validate ground truth against multiple modalities and contexts"""
        validation_results = {
            'trajectory_intent_agreement': [],
            'velocity_intent_consistency': [],
            'spatial_context_alignment': [],
            'gesture_intent_coherence': []
        }
        
        for seq in dataset:
            if 'intent_labels' not in seq:
                continue
            
            intent_labels = seq['intent_labels']
            
            # Validate against trajectory patterns
            if 'hand_trajectory' in seq:
                trajectory_agreement = self._validate_intent_trajectory_agreement(intent_labels, seq['hand_trajectory'])
                validation_results['trajectory_intent_agreement'].append(trajectory_agreement)
            
            # Validate against gesture type
            if 'gesture_type' in seq:
                gesture_coherence = self._validate_gesture_intent_coherence(intent_labels, seq['gesture_type'])
                validation_results['gesture_intent_coherence'].append(gesture_coherence)
        
        # Compute agreement scores
        agreement_scores = {}
        for modality, agreements in validation_results.items():
            if agreements:
                agreement_scores[modality] = np.mean(agreements)
            else:
                agreement_scores[modality] = 0.5  # Neutral if no data
        
        overall_agreement = np.mean(list(agreement_scores.values()))
        
        return {
            'modality_agreements': agreement_scores,
            'agreement_score': overall_agreement,
            'validation_quality': 'High' if overall_agreement > 0.8 else 'Medium' if overall_agreement > 0.6 else 'Low',
            'cross_modal_consistency': overall_agreement > 0.7
        }
    
    def _analyze_ambiguity_handling(self, dataset: List[Dict]) -> Dict:
        """Analyze how well the ground truth handles ambiguous cases"""
        ambiguity_metrics = {
            'identified_ambiguous_cases': 0,
            'confidence_modulated_cases': 0,
            'multiple_valid_interpretations': 0
        }
        
        total_sequences = len(dataset)
        
        for seq in dataset:
            if 'intent_labels' not in seq:
                continue
            
            labels = seq['intent_labels']
            confidences = seq.get('confidence_scores', [])
            
            # Identify potentially ambiguous periods
            if confidences and len(confidences) == len(labels):
                low_confidence_periods = [i for i, conf in enumerate(confidences) if conf < 0.6]
                
                if low_confidence_periods:
                    ambiguity_metrics['identified_ambiguous_cases'] += 1
                    
                    # Check if confidence appropriately modulates during ambiguity
                    confidence_variation = np.std([confidences[i] for i in low_confidence_periods])
                    if confidence_variation > 0.1:  # Some variation in confidence during ambiguity
                        ambiguity_metrics['confidence_modulated_cases'] += 1
            
            # Look for sequences with multiple reasonable interpretations
            unique_intents = len(set(labels))
            if unique_intents > 2:  # Multiple intents suggest possible ambiguity
                ambiguity_metrics['multiple_valid_interpretations'] += 1
        
        # Compute handling quality
        if ambiguity_metrics['identified_ambiguous_cases'] > 0:
            modulation_rate = ambiguity_metrics['confidence_modulated_cases'] / ambiguity_metrics['identified_ambiguous_cases']
            handling_quality = modulation_rate
        else:
            handling_quality = 0.8  # Good default if no ambiguous cases detected
        
        ambiguity_coverage = ambiguity_metrics['identified_ambiguous_cases'] / max(1, total_sequences)
        
        return {
            'ambiguous_cases_identified': ambiguity_metrics['identified_ambiguous_cases'],
            'confidence_modulated_cases': ambiguity_metrics['confidence_modulated_cases'],
            'ambiguity_coverage': ambiguity_coverage,
            'handling_quality': handling_quality,
            'expected_ambiguity_rate': 0.1,  # Expect ~10% ambiguous cases
            'ambiguity_handling_adequacy': abs(ambiguity_coverage - 0.1) < 0.05  # Within reasonable range
        }
    
    def _perform_ground_truth_cross_validation(self, dataset: List[Dict]) -> Dict:
        """Perform cross-validation consistency checks on ground truth"""
        if len(dataset) < 10:
            return {'cross_validation_score': 0.5, 'status': 'insufficient_data'}
        
        # Split dataset for cross-validation
        n_folds = 5
        fold_size = len(dataset) // n_folds
        consistency_scores = []
        
        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else len(dataset)
            
            test_fold = dataset[start_idx:end_idx]
            train_folds = dataset[:start_idx] + dataset[end_idx:]
            
            # Compare patterns between train and test
            fold_consistency = self._compute_fold_consistency(train_folds, test_fold)
            consistency_scores.append(fold_consistency)
        
        cross_validation_score = np.mean(consistency_scores)
        cross_validation_std = np.std(consistency_scores)
        
        return {
            'cross_validation_score': cross_validation_score,
            'cross_validation_std': cross_validation_std,
            'fold_consistency_scores': consistency_scores,
            'validation_quality': 'High' if cross_validation_score > 0.8 and cross_validation_std < 0.1 else 'Medium' if cross_validation_score > 0.6 else 'Low'
        }
    
    # Helper methods for ground truth validation
    def _compute_label_stability_periods(self, labels: List[str]) -> List[int]:
        """Compute lengths of stable label periods"""
        if not labels:
            return []
        
        stability_periods = []
        current_period = 1
        
        for i in range(1, len(labels)):
            if labels[i] == labels[i-1]:
                current_period += 1
            else:
                stability_periods.append(current_period)
                current_period = 1
        
        stability_periods.append(current_period)  # Add final period
        return stability_periods
    
    def _evaluate_transition_reasonableness(self, labels: List[str]) -> float:
        """Evaluate reasonableness of intent transitions"""
        if len(labels) < 2:
            return 1.0
        
        reasonable_transitions = 0
        total_transitions = 0
        
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                total_transitions += 1
                if self._is_reasonable_transition(labels[i-1], labels[i]):
                    reasonable_transitions += 1
        
        return reasonable_transitions / max(1, total_transitions)
    
    def _is_reasonable_transition(self, from_intent: str, to_intent: str) -> bool:
        """Check if intent transition is reasonable"""
        # Define reasonable transition patterns
        reasonable_patterns = {
            'idle': ['approach', 'orient', 'reach'],
            'approach': ['orient', 'reach', 'idle'],
            'orient': ['reach', 'approach', 'idle'],
            'reach': ['grasp', 'point', 'select', 'orient'],
            'grasp': ['move', 'release', 'hold'],
            'point': ['idle', 'reach', 'select'],
            'select': ['idle', 'reach', 'confirm'],
            'move': ['release', 'place', 'hold'],
            'release': ['idle', 'return'],
            'return': ['idle', 'rest']
        }
        
        from_clean = from_intent.lower()
        to_clean = to_intent.lower()
        
        return to_clean in reasonable_patterns.get(from_clean, [to_clean])  # Allow if not in patterns
    
    def _group_sequences_by_similarity(self, dataset: List[Dict]) -> List[List[Dict]]:
        """Group sequences by similarity for consistency analysis"""
        # Simple grouping by gesture type for now
        groups = {}
        
        for seq in dataset:
            gesture_type = seq.get('gesture_type', 'unknown')
            if gesture_type not in groups:
                groups[gesture_type] = []
            groups[gesture_type].append(seq)
        
        return [group for group in groups.values() if len(group) > 1]
    
    def _compute_group_label_consistency(self, group: List[Dict]) -> float:
        """Compute label consistency within a group of similar sequences"""
        all_labels = []
        for seq in group:
            if 'intent_labels' in seq:
                all_labels.extend(seq['intent_labels'])
        
        if not all_labels:
            return 0.0
        
        # Compute entropy of label distribution as consistency metric
        label_counts = Counter(all_labels)
        total_labels = len(all_labels)
        
        entropy = -sum((count/total_labels) * np.log2(count/total_labels) 
                      for count in label_counts.values())
        max_entropy = np.log2(len(label_counts))
        
        # Lower entropy = higher consistency
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
    
    def _compute_transition_smoothness(self, labels: List[str]) -> float:
        """Compute smoothness of transitions between labels"""
        if len(labels) < 2:
            return 1.0
        
        # Count abrupt vs smooth transitions
        abrupt_transitions = 0
        total_transitions = 0
        
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                total_transitions += 1
                if not self._is_reasonable_transition(labels[i-1], labels[i]):
                    abrupt_transitions += 1
        
        return 1.0 - (abrupt_transitions / max(1, total_transitions))
    
    def _evaluate_sequence_coherence(self, labels: List[str]) -> float:
        """Evaluate overall coherence of a label sequence"""
        if len(labels) < 3:
            return 1.0
        
        # Check for logical progression
        coherence_score = 0.0
        
        # Penalty for too many rapid changes
        changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
        change_rate = changes / len(labels)
        if change_rate < 0.5:  # Not too many changes
            coherence_score += 0.5
        
        # Bonus for logical progression patterns
        has_progression = any(
            self._has_logical_progression(labels[i:i+3]) 
            for i in range(len(labels)-2)
        )
        if has_progression:
            coherence_score += 0.5
        
        return coherence_score
    
    def _has_logical_progression(self, sequence: List[str]) -> bool:
        """Check if a 3-element sequence has logical progression"""
        if len(sequence) != 3:
            return False
        
        # Define some logical progressions
        progressions = [
            ['idle', 'approach', 'reach'],
            ['approach', 'reach', 'grasp'],
            ['reach', 'grasp', 'move'],
            ['grasp', 'move', 'release'],
            ['move', 'release', 'return'],
            ['release', 'return', 'idle']
        ]
        
        sequence_clean = [s.lower() for s in sequence]
        return sequence_clean in progressions
    
    def _validate_intent_trajectory_agreement(self, intent_labels: List[str], trajectory: List) -> float:
        """Validate intent labels against trajectory patterns"""
        if not intent_labels or not trajectory:
            return 0.5
        
        # Simple validation: moving intents should have non-zero velocity
        trajectory = np.array(trajectory)
        if len(trajectory) < 2:
            return 0.5
        
        velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        
        agreement_score = 0.0
        
        for i, intent in enumerate(intent_labels):
            if i < len(velocities):
                velocity = velocities[i]
                
                # Moving intents should have higher velocity
                if intent.lower() in ['reach', 'move', 'approach']:
                    if velocity > 0.01:  # Some movement
                        agreement_score += 1
                # Static intents should have lower velocity
                elif intent.lower() in ['idle', 'hold', 'grasp']:
                    if velocity < 0.05:  # Minimal movement
                        agreement_score += 1
        
        return agreement_score / len(intent_labels) if intent_labels else 0.5
    
    def _validate_gesture_intent_coherence(self, intent_labels: List[str], gesture_type: str) -> float:
        """Validate intent labels against gesture type"""
        if not intent_labels:
            return 0.5
        
        # Expected intents for different gesture types
        expected_intents = {
            'reach': ['idle', 'approach', 'reach', 'return'],
            'grasp': ['approach', 'reach', 'grasp', 'hold', 'release'],
            'point': ['approach', 'orient', 'point', 'hold'],
            'select': ['approach', 'orient', 'select', 'confirm'],
            'move': ['grasp', 'move', 'place', 'release']
        }
        
        gesture_clean = gesture_type.lower()
        expected = expected_intents.get(gesture_clean, [])
        
        if not expected:
            return 0.7  # Neutral if unknown gesture type
        
        # Count how many intent labels match expected intents
        matches = sum(1 for intent in intent_labels 
                     if intent.lower() in expected)
        
        return matches / len(intent_labels)
    
    def _compute_fold_consistency(self, train_fold: List[Dict], test_fold: List[Dict]) -> float:
        """Compute consistency between train and test folds"""
        # Extract intent patterns from both folds
        train_patterns = self._extract_intent_patterns(train_fold)
        test_patterns = self._extract_intent_patterns(test_fold)
        
        # Compute pattern similarity
        common_patterns = set(train_patterns.keys()) & set(test_patterns.keys())
        
        if not common_patterns:
            return 0.5
        
        consistency_scores = []
        for pattern in common_patterns:
            train_freq = train_patterns[pattern]
            test_freq = test_patterns[pattern]
            
            # Consistency based on frequency similarity
            consistency = 1.0 - abs(train_freq - test_freq) / max(train_freq, test_freq)
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores)
    
    def _extract_intent_patterns(self, sequences: List[Dict]) -> Dict[str, float]:
        """Extract intent patterns and their frequencies"""
        pattern_counts = Counter()
        total_patterns = 0
        
        for seq in sequences:
            if 'intent_labels' in seq:
                labels = seq['intent_labels']
                
                # Extract bigram patterns
                for i in range(len(labels) - 1):
                    pattern = f"{labels[i]}->{labels[i+1]}"
                    pattern_counts[pattern] += 1
                    total_patterns += 1
        
        # Return normalized frequencies
        return {pattern: count/max(1, total_patterns) 
               for pattern, count in pattern_counts.items()}
    
    def _grade_ground_truth_quality(self, quality_score: float) -> str:
        """Grade ground truth quality"""
        if quality_score >= 0.9:
            return 'Excellent'
        elif quality_score >= 0.8:
            return 'Good'
        elif quality_score >= 0.7:
            return 'Fair'
        elif quality_score >= 0.6:
            return 'Poor'
        else:
            return 'Unacceptable'
    
    def _generate_ground_truth_recommendations(self, label_consistency: Dict, confidence_quality: Dict, 
                                             temporal_evolution: Dict, ambiguity_handling: Dict) -> List[str]:
        """Generate recommendations for improving ground truth quality"""
        recommendations = []
        
        # Label consistency recommendations
        if label_consistency.get('consistency_score', 1.0) < 0.7:
            recommendations.append("Improve label consistency by reviewing annotation guidelines and inter-annotator agreement")
        
        # Confidence calibration recommendations
        if confidence_quality.get('calibration_score', 1.0) < 0.6:
            recommendations.append("Recalibrate confidence scores to better reflect actual uncertainty levels")
        
        # Temporal coherence recommendations
        if temporal_evolution.get('coherence_score', 1.0) < 0.7:
            recommendations.append("Review temporal transitions between intents for logical progression")
        
        # Ambiguity handling recommendations
        if ambiguity_handling.get('handling_quality', 1.0) < 0.6:
            recommendations.append("Improve handling of ambiguous cases with appropriate confidence modulation")
        
        # Add general recommendations
        if len(recommendations) == 0:
            recommendations.append("Ground truth quality is good - consider adding more edge cases for robustness")
        
        return recommendations[:5]
    
    def save_analysis_report(self, report: Dict, output_path: str):
        """Save comprehensive analysis report"""
        
        # Save JSON report
        json_path = f"{output_path}.json"
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_report = self._make_json_serializable(report)
            json.dump(json_report, f, indent=2)
        
        # Save summary report as text
        txt_path = f"{output_path}_summary.txt"
        self._save_summary_report(report, txt_path)
        
        logger.info(f"Analysis report saved to {json_path} and {txt_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-JSON types to serializable formats"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _save_summary_report(self, report: Dict, txt_path: str):
        """Save human-readable summary report"""
        with open(txt_path, 'w') as f:
            f.write("DATASET QUALITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset info
            dataset_info = report.get('dataset_info', {})
            f.write(f"Dataset Size: {dataset_info.get('total_sequences', 'N/A')} sequences\n")
            f.write(f"Analysis Date: {time.ctime(report.get('timestamp', time.time()))}\n\n")
            
            # Overall quality
            quality = report.get('quality_assessment', {})
            f.write(f"OVERALL QUALITY GRADE: {quality.get('quality_grade', 'N/A')}\n")
            f.write(f"Final Quality Score: {quality.get('final_quality_score', 0):.3f}\n")
            f.write(f"Publication Ready: {'YES' if quality.get('publication_ready', False) else 'NO'}\n\n")
            
            # Component scores
            component_scores = quality.get('component_scores', {})
            f.write("COMPONENT SCORES:\n")
            f.write("-" * 20 + "\n")
            for component, score in component_scores.items():
                f.write(f"{component.capitalize()}: {score:.3f}\n")
            f.write("\n")
            
            # Literature comparison
            literature = report.get('literature_comparison', {})
            f.write(f"LITERATURE COMPLIANCE: {literature.get('compliance_grade', 'N/A')}\n")
            f.write(f"Overall Compliance Score: {literature.get('overall_literature_compliance', 0):.3f}\n\n")
            
            # Key findings
            f.write("KEY FINDINGS:\n")
            f.write("-" * 15 + "\n")
            
            # Movement realism
            movement = report.get('movement_realism_analysis', {})
            biomech = movement.get('biomechanical_constraints', {})
            f.write(f"• Velocity constraint adherence: {biomech.get('velocity_constraint_adherence', 0):.2%}\n")
            f.write(f"• Path efficiency realistic: {biomech.get('realistic_scaling_exponent', False)}\n")
            
            # Intent patterns  
            intent = report.get('intent_pattern_analysis', {})
            f.write(f"• Intent pattern quality: {intent.get('intent_pattern_quality', 0):.3f}\n")
            
            # Balance and coverage
            balance = report.get('balance_coverage_analysis', {})
            f.write(f"• Dataset balance score: {balance.get('overall_balance_score', 0):.3f}\n")
            f.write(f"• Coverage completeness: {balance.get('coverage_completeness', 0):.3f}\n\n")
            
            # Recommendations
            recommendations = report.get('recommendations', [])
            if recommendations:
                f.write("IMPROVEMENT RECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")


# Helper functions for missing implementations
def _compute_realism_score(self, *analyses) -> float:
    """Compute overall realism score from component analyses"""
    scores = []
    for analysis in analyses:
        if isinstance(analysis, dict):
            # Extract numeric scores from analysis
            for key, value in analysis.items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    scores.append(value)
    return np.mean(scores) if scores else 0.5


def run_dataset_quality_analysis_example():
    """Run comprehensive dataset quality analysis example"""
    print("📊 Running Comprehensive Dataset Quality Analysis")
    print("=" * 60)
    
    # Load or generate sample dataset (using the enhanced generator)
    from .enhanced_synthetic_generator import EnhancedSyntheticGenerator
    
    workspace = np.array([-1, 1, -1, 1, 0, 2])
    generator = EnhancedSyntheticGenerator(workspace, sampling_frequency=100, random_seed=42)
    
    print("Generating comprehensive test dataset...")
    
    # Generate diverse dataset with different characteristics
    test_dataset = []
    gesture_types = ['reach', 'grab', 'point', 'handover', 'wave', 'idle']
    
    for i in range(200):  # Larger dataset for comprehensive analysis
        gesture = np.random.choice(gesture_types, p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1])
        
        # Varied parameters
        start_pos = np.random.uniform([-0.2, -0.2, 0.8], [0.2, 0.2, 1.2])
        end_pos = np.random.uniform([-0.6, -0.6, 0.4], [0.6, 0.6, 1.6])
        duration = np.random.uniform(0.8, 3.5)
        noise_level = np.random.uniform(0.005, 0.04)
        
        # User demographics variation
        age = np.random.randint(20, 80)
        skill = np.random.uniform(0.7, 1.3)
        
        user_params = {
            'age': age,
            'skill_level': skill,
            'fatigue_level': np.random.uniform(0, 0.3)
        }
        
        # Generate trajectory
        t, pos, vel, acc, metrics = generator.generate_biomechanically_realistic_trajectory(
            start_pos, end_pos, duration, user_params
        )
        
        # Generate intent sequence
        intent_t, intent_labels, confidence = generator.generate_realistic_intent_sequence(
            gesture, duration, {'target_object': {'position': end_pos}}
        )
        
        # Add sensor noise
        clean_data = {'positions': pos, 'gaze': np.random.randn(len(pos), 3)}
        noisy_data = generator.add_realistic_sensor_noise(
            clean_data, 'mocap', {'lighting': np.random.choice(['bright', 'normal', 'dim'])}
        )
        
        # Create comprehensive sequence
        sequence = {
            'sequence_id': f"seq_{i:04d}",
            'gesture_type': gesture,
            'hand_trajectory': noisy_data['positions'],
            'velocities': vel,
            'accelerations': acc,
            'timestamps': t,
            'intent_labels': intent_labels,
            'confidence_scores': confidence,
            'movement_metrics': metrics,
            'noise_level': noise_level,
            'duration': duration,
            'user_demographics': user_params,
            'environmental_conditions': {
                'lighting': np.random.choice(['bright', 'normal', 'dim']),
                'workspace_clutter': np.random.uniform(0, 0.5)
            }
        }
        
        test_dataset.append(sequence)
    
    print(f"Generated {len(test_dataset)} sequences for analysis")
    
    # Initialize analyzer
    analyzer = DatasetQualityAnalyzer()
    
    # Run comprehensive analysis
    print("Running comprehensive quality analysis...")
    analysis_report = analyzer.analyze_complete_dataset(
        test_dataset, 
        save_report=True,
        output_path="comprehensive_dataset_analysis"
    )
    
    # Print summary results
    print("\n🎯 DATASET QUALITY ANALYSIS RESULTS")
    print("=" * 45)
    
    quality = analysis_report.get('quality_assessment', {})
    print(f"Overall Quality Grade: {quality.get('quality_grade', 'N/A')}")
    print(f"Final Quality Score: {quality.get('final_quality_score', 0):.3f}/1.0")
    print(f"Publication Ready: {'✅ YES' if quality.get('publication_ready', False) else '❌ NO'}")
    
    print(f"\nComponent Breakdown:")
    component_scores = quality.get('component_scores', {})
    for component, score in component_scores.items():
        print(f"  • {component.title()}: {score:.3f}")
    
    literature = analysis_report.get('literature_comparison', {})
    print(f"\nLiterature Compliance: {literature.get('compliance_grade', 'N/A')} ({literature.get('overall_literature_compliance', 0):.3f})")
    
    # Key insights
    print(f"\n📋 Key Quality Insights:")
    
    movement = analysis_report.get('movement_realism_analysis', {})
    biomech = movement.get('biomechanical_constraints', {})
    print(f"  • Biomechanical realism: {biomech.get('velocity_constraint_adherence', 0):.1%} constraint adherence")
    
    balance = analysis_report.get('balance_coverage_analysis', {})
    print(f"  • Dataset balance: {balance.get('overall_balance_score', 0):.3f}")
    print(f"  • Coverage completeness: {balance.get('coverage_completeness', 0):.1%}")
    
    intent = analysis_report.get('intent_pattern_analysis', {})
    print(f"  • Intent pattern realism: {intent.get('intent_pattern_quality', 0):.3f}")
    
    # Recommendations
    recommendations = analysis_report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Top Improvement Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n📄 Detailed reports saved to:")
    print(f"  • comprehensive_dataset_analysis.json")
    print(f"  • comprehensive_dataset_analysis_summary.txt")
    
    return analysis_report


if __name__ == "__main__":
    # Run comprehensive analysis
    report = run_dataset_quality_analysis_example()
    print("\n✅ Comprehensive dataset quality analysis complete!")