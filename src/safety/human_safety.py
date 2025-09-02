#!/usr/bin/env python3
"""
Human Safety Modeling System
============================

This module implements comprehensive human safety modeling including dynamic
human safety zone computation, biomechanical injury risk assessment, comfort
zone modeling and enforcement, stress detection and adaptive response, and
cultural/individual safety preference adaptation.

Features:
- Dynamic human safety zone computation based on motion prediction
- Biomechanical injury risk assessment using established models
- Comfort zone modeling and enforcement for psychological safety
- Real-time stress detection and adaptive system responses
- Cultural and individual safety preference adaptation
- Physiological monitoring integration
- Personalized safety profiles

Mathematical Models:
===================

Injury Risk Assessment:
    HIC = (1/T) ∫[a(t)]^2.5 dt  (Head Injury Criterion)
    
Dynamic Safety Zones:
    R(t) = R_base + k₁·||v_human|| + k₂·σ_prediction + k₃·comfort_factor
    
Stress Detection:
    Stress = f(HR, GSR, Facial_Features, Motion_Patterns)
    
Cultural Adaptation:
    Preference(culture, individual) = α·Cultural_Model + β·Individual_History

Author: Claude Code (Anthropic)
Date: 2025-01-15
Version: 1.0
"""

import time
import threading
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import queue
import json
from pathlib import Path
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

logger = logging.getLogger(__name__)


class InjuryType(Enum):
    """Types of potential injuries"""
    HEAD_INJURY = "head_injury"
    NECK_INJURY = "neck_injury"  
    CHEST_INJURY = "chest_injury"
    ARM_INJURY = "arm_injury"
    LEG_INJURY = "leg_injury"
    BACK_INJURY = "back_injury"
    PSYCHOLOGICAL_TRAUMA = "psychological_trauma"


class ComfortLevel(IntEnum):
    """Human comfort levels"""
    VERY_UNCOMFORTABLE = 1
    UNCOMFORTABLE = 2
    NEUTRAL = 3
    COMFORTABLE = 4
    VERY_COMFORTABLE = 5


class StressLevel(IntEnum):
    """Human stress levels"""
    NO_STRESS = 0
    LOW_STRESS = 1
    MODERATE_STRESS = 2
    HIGH_STRESS = 3
    EXTREME_STRESS = 4


class CulturalContext(Enum):
    """Cultural contexts affecting safety preferences"""
    WESTERN_INDIVIDUALISTIC = "western_individualistic"
    EASTERN_COLLECTIVIST = "eastern_collectivist" 
    LATIN_AMERICAN = "latin_american"
    NORTHERN_EUROPEAN = "northern_european"
    MIDDLE_EASTERN = "middle_eastern"
    AFRICAN = "african"
    UNKNOWN = "unknown"


@dataclass
class BiomechanicalLimits:
    """Human biomechanical safety limits"""
    # Force limits (N)
    max_contact_force: float = 150.0  # ISO/TS 15066
    max_impact_force: float = 65.0
    
    # Pressure limits (N/cm²)
    max_body_pressure: float = 10.0
    max_head_pressure: float = 13.0
    
    # Acceleration limits (m/s²)
    max_head_acceleration: float = 80.0  # HIC threshold
    max_chest_acceleration: float = 60.0
    
    # Velocity limits (m/s)
    max_impact_velocity: float = 0.5
    
    # Duration limits (ms)
    max_contact_duration: float = 100.0


@dataclass
class PhysiologicalState:
    """Human physiological state measurements"""
    timestamp: float
    heart_rate: float = 70.0  # bpm
    skin_conductance: float = 5.0  # μS (galvanic skin response)
    body_temperature: float = 37.0  # °C
    blood_pressure_systolic: float = 120.0  # mmHg
    blood_pressure_diastolic: float = 80.0  # mmHg
    
    # Derived metrics
    stress_indicator: float = 0.0  # 0-1 scale
    fatigue_level: float = 0.0    # 0-1 scale
    alertness_level: float = 1.0  # 0-1 scale


@dataclass
class HumanProfile:
    """Individual human safety profile"""
    person_id: str
    name: str
    age: int
    height: float  # cm
    weight: float  # kg
    
    # Medical information
    medical_conditions: List[str] = field(default_factory=list)
    mobility_limitations: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    
    # Cultural context
    cultural_background: CulturalContext = CulturalContext.UNKNOWN
    language_preference: str = "english"
    
    # Safety preferences (learned over time)
    preferred_robot_distance: float = 1.0  # m
    preferred_robot_speed: float = 0.3     # m/s
    comfort_zone_size: float = 0.8         # m radius
    noise_sensitivity: float = 0.5         # 0-1 scale
    
    # Historical data
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    stress_patterns: Dict[str, float] = field(default_factory=dict)
    injury_history: List[str] = field(default_factory=list)


@dataclass
class SafetyZoneConfig:
    """Dynamic safety zone configuration"""
    base_radius: float = 0.6  # Base safety zone radius (m)
    velocity_factor: float = 0.3  # Expansion factor for human velocity
    uncertainty_factor: float = 0.2  # Expansion for prediction uncertainty
    comfort_factor: float = 0.1  # Additional comfort margin
    
    # Dynamic adjustment parameters
    min_radius: float = 0.3
    max_radius: float = 2.0
    adaptation_rate: float = 0.1
    
    # Stress-based adjustments
    stress_expansion_factor: float = 0.4
    fatigue_expansion_factor: float = 0.2


class InjuryRiskAssessment:
    """Biomechanical injury risk assessment"""
    
    def __init__(self):
        """Initialize injury risk assessment"""
        self.biomech_limits = BiomechanicalLimits()
        self.injury_models = {}
        self._initialize_injury_models()
        
        logger.info("Injury risk assessment initialized")
    
    def _initialize_injury_models(self) -> None:
        """Initialize biomechanical injury models"""
        
        # Head Injury Criterion (HIC) model
        def hic_model(acceleration_profile: np.ndarray, duration: float) -> float:
            """Calculate Head Injury Criterion"""
            if len(acceleration_profile) == 0:
                return 0.0
            
            # HIC = (1/t) * (∫a²·⁵dt)
            # Simplified discrete version
            dt = duration / len(acceleration_profile)
            integral = np.sum(np.power(acceleration_profile, 2.5)) * dt
            hic = integral / duration if duration > 0 else 0.0
            
            # HIC threshold: 1000 for serious injury, 700 for moderate
            injury_probability = min(1.0, hic / 1000.0)
            return injury_probability
        
        self.injury_models[InjuryType.HEAD_INJURY] = hic_model
        
        # Chest injury model (Viscous Criterion)
        def chest_injury_model(velocity: float, compression: float) -> float:
            """Calculate chest injury risk using Viscous Criterion"""
            # VC = V * C (velocity * compression)
            vc = velocity * compression
            # VC threshold: 1.0 m/s for serious injury
            injury_probability = min(1.0, vc / 1.0)
            return injury_probability
        
        self.injury_models[InjuryType.CHEST_INJURY] = chest_injury_model
        
        # Force-based injury model
        def force_injury_model(force: float, contact_area: float, body_part: str) -> float:
            """Calculate injury risk based on contact force"""
            pressure = force / max(contact_area, 1e-6)  # N/cm²
            
            # ISO/TS 15066 pressure limits
            if body_part in ['head', 'face']:
                max_pressure = 13.0
            elif body_part in ['neck']:
                max_pressure = 35.0
            elif body_part in ['chest', 'back']:
                max_pressure = 10.0
            else:
                max_pressure = 16.0  # limbs
            
            injury_probability = min(1.0, pressure / max_pressure)
            return injury_probability
        
        self.injury_models['force_based'] = force_injury_model
    
    def assess_collision_risk(self,
                            robot_trajectory: np.ndarray,
                            human_trajectory: np.ndarray,
                            robot_mass: float = 50.0,
                            human_profile: Optional[HumanProfile] = None) -> Dict[str, Any]:
        """Assess injury risk for potential collision"""
        
        if len(robot_trajectory) == 0 or len(human_trajectory) == 0:
            return {'total_risk': 0.0, 'injury_risks': {}}
        
        # Calculate relative motion
        relative_positions = robot_trajectory[:, :3] - human_trajectory[:, :3]
        distances = np.linalg.norm(relative_positions, axis=1)
        
        # Find minimum distance point
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # Check if collision is likely (distance < threshold)
        collision_threshold = 0.1  # 10cm
        if min_distance > collision_threshold:
            return {'total_risk': 0.0, 'injury_risks': {}, 'collision_probability': 0.0}
        
        # Calculate collision parameters
        dt = 0.1  # Assume 0.1s time steps
        
        # Robot velocity at collision point
        if min_distance_idx > 0:
            robot_velocity = (robot_trajectory[min_distance_idx] - robot_trajectory[min_distance_idx-1]) / dt
        else:
            robot_velocity = np.zeros(3)
        
        robot_speed = np.linalg.norm(robot_velocity)
        
        # Estimate impact force (simplified)
        # F = ma, assuming deceleration over 0.01s (10ms contact time)
        contact_time = 0.01
        impact_acceleration = robot_speed / contact_time
        impact_force = robot_mass * impact_acceleration
        
        # Assess different injury types
        injury_risks = {}
        
        # Head injury (if robot height in head region)
        if human_profile:
            head_height = human_profile.height * 0.9  # 90% of height
            robot_height = robot_trajectory[min_distance_idx, 2]  # Z coordinate
            
            if abs(robot_height - head_height) < 0.3:  # Within 30cm of head
                head_accel_profile = np.array([impact_acceleration] * 10)  # 10ms profile
                head_risk = self.injury_models[InjuryType.HEAD_INJURY](head_accel_profile, contact_time)
                injury_risks[InjuryType.HEAD_INJURY.value] = head_risk
        
        # Force-based injury assessment
        contact_area = 10.0  # Assume 10cm² contact area
        body_parts = ['chest', 'arm', 'leg']  # Most likely contact points
        
        for body_part in body_parts:
            force_risk = self.injury_models['force_based'](impact_force, contact_area, body_part)
            injury_risks[f"{body_part}_injury"] = force_risk
        
        # Overall risk (max of individual risks)
        total_risk = max(injury_risks.values()) if injury_risks else 0.0
        
        # Collision probability based on distance and relative velocity
        collision_probability = max(0.0, 1.0 - min_distance / collision_threshold)
        
        return {
            'total_risk': total_risk,
            'injury_risks': injury_risks,
            'collision_probability': collision_probability,
            'impact_force': impact_force,
            'impact_velocity': robot_speed,
            'minimum_distance': min_distance,
            'collision_details': {
                'robot_velocity': robot_velocity.tolist(),
                'contact_time': contact_time,
                'impact_acceleration': impact_acceleration
            }
        }
    
    def assess_chronic_risk(self, 
                          interaction_history: List[Dict[str, Any]],
                          human_profile: HumanProfile) -> Dict[str, float]:
        """Assess chronic injury risk from repeated interactions"""
        
        if not interaction_history:
            return {}
        
        # Analyze repetitive stress patterns
        chronic_risks = {}
        
        # Calculate cumulative exposure metrics
        total_exposure_time = sum(interaction.get('duration', 0) for interaction in interaction_history)
        avg_stress_level = np.mean([interaction.get('stress_level', 0) for interaction in interaction_history])
        
        # Repetitive motion assessment
        motion_patterns = [interaction.get('motion_pattern', 'normal') for interaction in interaction_history]
        repetitive_count = sum(1 for pattern in motion_patterns if pattern == 'repetitive')
        repetitive_ratio = repetitive_count / len(motion_patterns) if motion_patterns else 0.0
        
        # Psychological stress accumulation
        if avg_stress_level > 2.0:  # Moderate to high stress
            chronic_risks['psychological_strain'] = min(1.0, avg_stress_level / 4.0)
        
        # Physical strain from prolonged interaction
        if total_exposure_time > 3600:  # More than 1 hour total
            chronic_risks['physical_fatigue'] = min(1.0, total_exposure_time / 14400)  # 4 hour reference
        
        # Repetitive strain injury risk
        if repetitive_ratio > 0.3:  # More than 30% repetitive interactions
            chronic_risks['repetitive_strain'] = min(1.0, repetitive_ratio * 2.0)
        
        return chronic_risks


class StressDetectionSystem:
    """Real-time human stress detection and monitoring"""
    
    def __init__(self):
        """Initialize stress detection system"""
        self.stress_model = None
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Stress indicators
        self.stress_history = deque(maxlen=1000)
        self.baseline_metrics = {}
        
        # Feature weights for stress detection
        self.feature_weights = {
            'heart_rate_change': 0.3,
            'skin_conductance': 0.25,
            'facial_tension': 0.2,
            'motion_agitation': 0.15,
            'voice_stress': 0.1
        }
        
        self._initialize_stress_model()
        
        logger.info("Stress detection system initialized")
    
    def _initialize_stress_model(self) -> None:
        """Initialize stress detection model"""
        self.stress_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def update_baseline(self, physiological_state: PhysiologicalState) -> None:
        """Update baseline physiological metrics"""
        if not self.baseline_metrics:
            self.baseline_metrics = {
                'heart_rate': physiological_state.heart_rate,
                'skin_conductance': physiological_state.skin_conductance,
                'body_temperature': physiological_state.body_temperature
            }
        else:
            # Exponential moving average
            alpha = 0.1
            self.baseline_metrics['heart_rate'] = (
                alpha * physiological_state.heart_rate + 
                (1 - alpha) * self.baseline_metrics['heart_rate']
            )
            self.baseline_metrics['skin_conductance'] = (
                alpha * physiological_state.skin_conductance +
                (1 - alpha) * self.baseline_metrics['skin_conductance']
            )
    
    def detect_stress(self,
                     physiological_state: PhysiologicalState,
                     facial_features: Optional[Dict[str, float]] = None,
                     motion_data: Optional[np.ndarray] = None,
                     voice_features: Optional[Dict[str, float]] = None) -> Tuple[StressLevel, float, Dict[str, float]]:
        """Detect stress level from multiple indicators"""
        
        # Extract features for stress detection
        features = self._extract_stress_features(
            physiological_state, facial_features, motion_data, voice_features
        )
        
        if not features:
            return StressLevel.NO_STRESS, 0.0, {}
        
        # Calculate stress score
        if self.model_trained:
            # Use trained model
            feature_array = np.array(list(features.values())).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            stress_probability = self.stress_model.predict_proba(scaled_features)[0]
            stress_score = np.max(stress_probability)
        else:
            # Use heuristic approach
            stress_score = self._heuristic_stress_detection(features)
        
        # Determine stress level
        stress_level = self._classify_stress_level(stress_score)
        
        # Record stress measurement
        stress_record = {
            'timestamp': time.time(),
            'stress_level': stress_level,
            'stress_score': stress_score,
            'features': features.copy()
        }
        self.stress_history.append(stress_record)
        
        return stress_level, stress_score, features
    
    def _extract_stress_features(self,
                               physiological_state: PhysiologicalState,
                               facial_features: Optional[Dict[str, float]],
                               motion_data: Optional[np.ndarray],
                               voice_features: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Extract features for stress detection"""
        features = {}
        
        # Physiological features
        if self.baseline_metrics:
            hr_baseline = self.baseline_metrics.get('heart_rate', 70)
            gsr_baseline = self.baseline_metrics.get('skin_conductance', 5)
            
            features['heart_rate_change'] = (physiological_state.heart_rate - hr_baseline) / hr_baseline
            features['skin_conductance_ratio'] = physiological_state.skin_conductance / max(gsr_baseline, 1e-6)
            features['temperature_change'] = physiological_state.body_temperature - 37.0
        else:
            features['heart_rate_raw'] = physiological_state.heart_rate
            features['skin_conductance_raw'] = physiological_state.skin_conductance
        
        # Facial features (if available)
        if facial_features:
            features.update({
                f"facial_{k}": v for k, v in facial_features.items()
                if k in ['brow_tension', 'eye_strain', 'jaw_clench']
            })
        
        # Motion features
        if motion_data is not None and len(motion_data) > 1:
            # Calculate motion agitation (variance in movement)
            motion_variance = np.var(motion_data, axis=0)
            features['motion_agitation'] = np.mean(motion_variance)
            
            # Acceleration patterns
            if len(motion_data) > 2:
                acceleration = np.diff(motion_data, axis=0)
                features['acceleration_variance'] = np.var(acceleration)
        
        # Voice features (if available)
        if voice_features:
            features.update({
                f"voice_{k}": v for k, v in voice_features.items()
                if k in ['pitch_variance', 'speaking_rate', 'voice_tremor']
            })
        
        return features
    
    def _heuristic_stress_detection(self, features: Dict[str, float]) -> float:
        """Heuristic stress detection when model is not trained"""
        stress_indicators = []
        
        # Heart rate indicator
        if 'heart_rate_change' in features:
            hr_change = abs(features['heart_rate_change'])
            if hr_change > 0.2:  # 20% increase
                stress_indicators.append(min(1.0, hr_change))
        
        # Skin conductance indicator
        if 'skin_conductance_ratio' in features:
            gsr_ratio = features['skin_conductance_ratio']
            if gsr_ratio > 1.5:  # 50% increase
                stress_indicators.append(min(1.0, (gsr_ratio - 1.0) * 2))
        
        # Motion agitation
        if 'motion_agitation' in features:
            agitation = features['motion_agitation']
            if agitation > 0.1:  # Threshold for agitated movement
                stress_indicators.append(min(1.0, agitation * 5))
        
        # Combine indicators
        if stress_indicators:
            return np.mean(stress_indicators)
        else:
            return 0.0
    
    def _classify_stress_level(self, stress_score: float) -> StressLevel:
        """Classify stress level from score"""
        if stress_score < 0.2:
            return StressLevel.NO_STRESS
        elif stress_score < 0.4:
            return StressLevel.LOW_STRESS
        elif stress_score < 0.6:
            return StressLevel.MODERATE_STRESS
        elif stress_score < 0.8:
            return StressLevel.HIGH_STRESS
        else:
            return StressLevel.EXTREME_STRESS
    
    def train_personalized_model(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train personalized stress detection model"""
        if len(training_data) < 20:  # Need minimum data
            logger.warning("Insufficient training data for personalized stress model")
            return False
        
        try:
            # Prepare training data
            features_list = []
            labels = []
            
            for sample in training_data:
                if 'features' in sample and 'stress_level' in sample:
                    feature_vector = list(sample['features'].values())
                    features_list.append(feature_vector)
                    labels.append(sample['stress_level'])
            
            if not features_list:
                return False
            
            # Scale features
            X = np.array(features_list)
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.stress_model.fit(X_scaled, labels)
            self.model_trained = True
            
            logger.info("Personalized stress model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training stress model: {e}")
            return False
    
    def get_stress_trends(self, time_window: float = 300.0) -> Dict[str, Any]:
        """Get stress trends over time window"""
        current_time = time.time()
        recent_records = [
            record for record in self.stress_history
            if current_time - record['timestamp'] <= time_window
        ]
        
        if not recent_records:
            return {'trend': 'stable', 'average_stress': 0.0}
        
        stress_scores = [record['stress_score'] for record in recent_records]
        timestamps = [record['timestamp'] for record in recent_records]
        
        # Calculate trend (linear regression slope)
        if len(stress_scores) > 1:
            time_deltas = [(t - timestamps[0]) for t in timestamps]
            trend_slope = np.polyfit(time_deltas, stress_scores, 1)[0]
            
            if trend_slope > 0.01:
                trend = 'increasing'
            elif trend_slope < -0.01:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            trend_slope = 0.0
        
        return {
            'trend': trend,
            'trend_slope': trend_slope,
            'average_stress': np.mean(stress_scores),
            'max_stress': np.max(stress_scores),
            'stress_variability': np.std(stress_scores),
            'sample_count': len(recent_records)
        }


class DynamicSafetyZoneManager:
    """Manages dynamic safety zones around humans"""
    
    def __init__(self, config: SafetyZoneConfig):
        """Initialize dynamic safety zone manager"""
        self.config = config
        self.safety_zones: Dict[str, Dict[str, Any]] = {}
        self.zone_history = deque(maxlen=1000)
        
        logger.info("Dynamic safety zone manager initialized")
    
    def update_safety_zone(self,
                          person_id: str,
                          human_position: np.ndarray,
                          human_velocity: np.ndarray,
                          prediction_uncertainty: float,
                          human_profile: HumanProfile,
                          physiological_state: PhysiologicalState,
                          stress_level: StressLevel) -> Dict[str, Any]:
        """Update dynamic safety zone for a person"""
        
        current_time = time.time()
        
        # Calculate base radius adjustments
        velocity_magnitude = np.linalg.norm(human_velocity)
        velocity_adjustment = self.config.velocity_factor * velocity_magnitude
        
        uncertainty_adjustment = self.config.uncertainty_factor * prediction_uncertainty
        
        # Stress-based adjustments
        stress_adjustment = 0.0
        if stress_level >= StressLevel.MODERATE_STRESS:
            stress_factor = (stress_level.value - 1) / 4.0  # Normalize to 0-1
            stress_adjustment = self.config.stress_expansion_factor * stress_factor
        
        # Fatigue adjustment
        fatigue_adjustment = self.config.fatigue_expansion_factor * physiological_state.fatigue_level
        
        # Individual preference adjustment
        preference_adjustment = (human_profile.preferred_robot_distance - 1.0)  # Relative to 1m default
        
        # Cultural adjustment
        cultural_adjustment = self._get_cultural_distance_preference(human_profile.cultural_background)
        
        # Calculate dynamic radius
        dynamic_radius = (
            self.config.base_radius +
            velocity_adjustment +
            uncertainty_adjustment +
            stress_adjustment +
            fatigue_adjustment +
            preference_adjustment +
            cultural_adjustment
        )
        
        # Apply bounds
        dynamic_radius = np.clip(dynamic_radius, self.config.min_radius, self.config.max_radius)
        
        # Create safety zone
        safety_zone = {
            'person_id': person_id,
            'timestamp': current_time,
            'center': human_position.copy(),
            'radius': dynamic_radius,
            'velocity': human_velocity.copy(),
            'components': {
                'base_radius': self.config.base_radius,
                'velocity_adjustment': velocity_adjustment,
                'uncertainty_adjustment': uncertainty_adjustment,
                'stress_adjustment': stress_adjustment,
                'fatigue_adjustment': fatigue_adjustment,
                'preference_adjustment': preference_adjustment,
                'cultural_adjustment': cultural_adjustment
            },
            'context': {
                'stress_level': stress_level.value,
                'prediction_uncertainty': prediction_uncertainty,
                'velocity_magnitude': velocity_magnitude,
                'fatigue_level': physiological_state.fatigue_level
            }
        }
        
        # Store and adapt
        self._adapt_zone_parameters(person_id, safety_zone, human_profile)
        self.safety_zones[person_id] = safety_zone
        self.zone_history.append(safety_zone.copy())
        
        return safety_zone
    
    def _get_cultural_distance_preference(self, cultural_context: CulturalContext) -> float:
        """Get cultural preference for personal space"""
        cultural_factors = {
            CulturalContext.WESTERN_INDIVIDUALISTIC: 0.1,   # Larger personal space
            CulturalContext.EASTERN_COLLECTIVIST: -0.05,    # Smaller personal space
            CulturalContext.LATIN_AMERICAN: 0.0,            # Moderate personal space
            CulturalContext.NORTHERN_EUROPEAN: 0.15,        # Large personal space
            CulturalContext.MIDDLE_EASTERN: 0.05,           # Moderate-large personal space
            CulturalContext.AFRICAN: -0.02,                 # Variable, slightly smaller
            CulturalContext.UNKNOWN: 0.0                    # No adjustment
        }
        
        return cultural_factors.get(cultural_context, 0.0)
    
    def _adapt_zone_parameters(self,
                             person_id: str,
                             current_zone: Dict[str, Any],
                             human_profile: HumanProfile) -> None:
        """Adapt zone parameters based on historical interactions"""
        
        # Get historical zones for this person
        person_zones = [
            zone for zone in self.zone_history
            if zone['person_id'] == person_id and 
            time.time() - zone['timestamp'] < 3600  # Last hour
        ]
        
        if len(person_zones) < 5:  # Need sufficient history
            return
        
        # Analyze patterns
        avg_stress = np.mean([zone['context']['stress_level'] for zone in person_zones])
        avg_velocity = np.mean([zone['context']['velocity_magnitude'] for zone in person_zones])
        
        # Adapt preferences based on observed patterns
        if avg_stress > 2.5:  # Consistently high stress
            # Increase preferred distance slightly
            adjustment = min(0.2, 0.05 * (avg_stress - 2.0))
            human_profile.preferred_robot_distance += adjustment * self.config.adaptation_rate
        
        # Adapt comfort zone based on velocity patterns
        if avg_velocity > 0.8:  # Fast-moving person
            human_profile.comfort_zone_size += 0.1 * self.config.adaptation_rate
        
        # Ensure preferences stay within reasonable bounds
        human_profile.preferred_robot_distance = np.clip(
            human_profile.preferred_robot_distance, 0.5, 3.0
        )
        human_profile.comfort_zone_size = np.clip(
            human_profile.comfort_zone_size, 0.3, 2.0
        )
    
    def check_zone_violations(self,
                            robot_position: np.ndarray,
                            robot_velocity: np.ndarray) -> List[Dict[str, Any]]:
        """Check for safety zone violations"""
        violations = []
        current_time = time.time()
        
        for person_id, zone in self.safety_zones.items():
            # Check if zone is recent
            if current_time - zone['timestamp'] > 5.0:  # Zone older than 5 seconds
                continue
            
            # Calculate distance to zone center
            distance = np.linalg.norm(robot_position - zone['center'])
            
            # Check violation
            if distance < zone['radius']:
                violation_severity = 1.0 - (distance / zone['radius'])
                
                # Predict future violation
                robot_speed = np.linalg.norm(robot_velocity)
                if robot_speed > 0.01:  # Robot is moving
                    # Project robot position 1 second ahead
                    future_robot_pos = robot_position + robot_velocity * 1.0
                    future_distance = np.linalg.norm(future_robot_pos - zone['center'])
                    
                    time_to_violation = distance / robot_speed if robot_speed > 0 else float('inf')
                else:
                    future_distance = distance
                    time_to_violation = 0.0
                
                violation = {
                    'person_id': person_id,
                    'violation_severity': violation_severity,
                    'current_distance': distance,
                    'zone_radius': zone['radius'],
                    'time_to_violation': time_to_violation,
                    'future_distance': future_distance,
                    'zone_context': zone['context'].copy()
                }
                
                violations.append(violation)
        
        return violations
    
    def get_zone_status(self, person_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of safety zones"""
        current_time = time.time()
        
        if person_id:
            if person_id in self.safety_zones:
                zone = self.safety_zones[person_id]
                zone_age = current_time - zone['timestamp']
                return {
                    'person_id': person_id,
                    'zone': zone,
                    'zone_age': zone_age,
                    'is_active': zone_age < 5.0
                }
            else:
                return {'error': f'No zone found for person {person_id}'}
        else:
            # All zones status
            active_zones = 0
            total_zones = len(self.safety_zones)
            
            for zone in self.safety_zones.values():
                if current_time - zone['timestamp'] < 5.0:
                    active_zones += 1
            
            return {
                'total_zones': total_zones,
                'active_zones': active_zones,
                'zone_details': {
                    pid: {
                        'radius': zone['radius'],
                        'age': current_time - zone['timestamp'],
                        'stress_level': zone['context']['stress_level']
                    }
                    for pid, zone in self.safety_zones.items()
                }
            }


class HumanSafetySystem:
    """Comprehensive human safety modeling system"""
    
    def __init__(self):
        """Initialize human safety system"""
        self.injury_assessment = InjuryRiskAssessment()
        self.stress_detector = StressDetectionSystem()
        self.zone_manager = DynamicSafetyZoneManager(SafetyZoneConfig())
        
        # Human profiles and states
        self.human_profiles: Dict[str, HumanProfile] = {}
        self.physiological_states: Dict[str, PhysiologicalState] = {}
        
        # System state
        self.active_interactions: Dict[str, Dict[str, Any]] = {}
        self.safety_alerts = deque(maxlen=1000)
        
        # Monitoring
        self.monitoring_enabled = True
        self.monitoring_thread = None
        
        logger.info("Human safety system initialized")
    
    def register_human(self, profile: HumanProfile) -> None:
        """Register human profile"""
        self.human_profiles[profile.person_id] = profile
        self.physiological_states[profile.person_id] = PhysiologicalState(timestamp=time.time())
        logger.info(f"Registered human profile: {profile.person_id}")
    
    def update_human_state(self,
                          person_id: str,
                          position: np.ndarray,
                          velocity: np.ndarray,
                          physiological_state: PhysiologicalState,
                          prediction_uncertainty: float = 0.1,
                          facial_features: Optional[Dict[str, float]] = None,
                          motion_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Update human state and assess safety"""
        
        if person_id not in self.human_profiles:
            logger.warning(f"Unknown person: {person_id}")
            return {'error': 'Unknown person'}
        
        profile = self.human_profiles[person_id]
        
        # Update physiological state
        self.physiological_states[person_id] = physiological_state
        
        # Detect stress
        stress_level, stress_score, stress_features = self.stress_detector.detect_stress(
            physiological_state, facial_features, motion_data
        )
        
        # Update safety zone
        safety_zone = self.zone_manager.update_safety_zone(
            person_id, position, velocity, prediction_uncertainty,
            profile, physiological_state, stress_level
        )
        
        # Update interaction record
        self.active_interactions[person_id] = {
            'timestamp': time.time(),
            'position': position.copy(),
            'velocity': velocity.copy(),
            'stress_level': stress_level,
            'stress_score': stress_score,
            'safety_zone': safety_zone,
            'physiological_state': physiological_state
        }
        
        # Check for safety alerts
        self._check_safety_conditions(person_id)
        
        return {
            'person_id': person_id,
            'stress_level': stress_level.value,
            'stress_score': stress_score,
            'safety_zone_radius': safety_zone['radius'],
            'zone_components': safety_zone['components'],
            'alerts': len([alert for alert in self.safety_alerts 
                          if alert.get('person_id') == person_id and 
                          time.time() - alert['timestamp'] < 60])
        }
    
    def assess_interaction_safety(self,
                                robot_trajectory: np.ndarray,
                                human_trajectories: Dict[str, np.ndarray],
                                robot_mass: float = 50.0) -> Dict[str, Any]:
        """Assess safety of robot-human interaction"""
        
        safety_assessment = {
            'timestamp': time.time(),
            'overall_safety_score': 1.0,
            'injury_risks': {},
            'zone_violations': [],
            'stress_levels': {},
            'recommendations': []
        }
        
        # Check each human interaction
        for person_id, human_traj in human_trajectories.items():
            if person_id not in self.human_profiles:
                continue
            
            profile = self.human_profiles[person_id]
            
            # Assess collision/injury risk
            injury_risk = self.injury_assessment.assess_collision_risk(
                robot_trajectory, human_traj, robot_mass, profile
            )
            
            safety_assessment['injury_risks'][person_id] = injury_risk
            
            # Check zone violations
            if len(robot_trajectory) > 0:
                robot_pos = robot_trajectory[-1, :3]  # Current robot position
                robot_vel = (robot_trajectory[-1] - robot_trajectory[-2]) if len(robot_trajectory) > 1 else np.zeros(3)
                
                violations = self.zone_manager.check_zone_violations(robot_pos, robot_vel)
                person_violations = [v for v in violations if v['person_id'] == person_id]
                
                if person_violations:
                    safety_assessment['zone_violations'].extend(person_violations)
            
            # Current stress level
            if person_id in self.active_interactions:
                stress_level = self.active_interactions[person_id]['stress_level']
                safety_assessment['stress_levels'][person_id] = stress_level.value
        
        # Calculate overall safety score
        risk_factors = []
        
        # Injury risk factor
        max_injury_risk = 0.0
        for person_risks in safety_assessment['injury_risks'].values():
            max_injury_risk = max(max_injury_risk, person_risks.get('total_risk', 0.0))
        
        risk_factors.append(max_injury_risk)
        
        # Zone violation factor
        max_violation_severity = 0.0
        for violation in safety_assessment['zone_violations']:
            max_violation_severity = max(max_violation_severity, violation['violation_severity'])
        
        risk_factors.append(max_violation_severity)
        
        # Stress factor
        max_stress = 0.0
        for stress_level in safety_assessment['stress_levels'].values():
            max_stress = max(max_stress, stress_level / 4.0)  # Normalize to 0-1
        
        risk_factors.append(max_stress)
        
        # Overall safety score (inverse of maximum risk)
        if risk_factors:
            safety_assessment['overall_safety_score'] = 1.0 - max(risk_factors)
        
        # Generate recommendations
        recommendations = self._generate_safety_recommendations(safety_assessment)
        safety_assessment['recommendations'] = recommendations
        
        return safety_assessment
    
    def _check_safety_conditions(self, person_id: str) -> None:
        """Check for safety alert conditions"""
        
        if person_id not in self.active_interactions:
            return
        
        interaction = self.active_interactions[person_id]
        current_time = time.time()
        
        # High stress alert
        if interaction['stress_level'] >= StressLevel.HIGH_STRESS:
            alert = {
                'timestamp': current_time,
                'person_id': person_id,
                'type': 'high_stress',
                'severity': 'warning',
                'message': f'High stress detected for {person_id}',
                'stress_score': interaction['stress_score']
            }
            self.safety_alerts.append(alert)
            logger.warning(f"High stress alert for {person_id}: {interaction['stress_score']:.3f}")
        
        # Physiological anomaly alert
        physio_state = interaction['physiological_state']
        if physio_state.heart_rate > 100 or physio_state.skin_conductance > 10:
            alert = {
                'timestamp': current_time,
                'person_id': person_id,
                'type': 'physiological_anomaly',
                'severity': 'caution',
                'message': f'Physiological anomaly for {person_id}',
                'heart_rate': physio_state.heart_rate,
                'skin_conductance': physio_state.skin_conductance
            }
            self.safety_alerts.append(alert)
    
    def _generate_safety_recommendations(self, safety_assessment: Dict[str, Any]) -> List[str]:
        """Generate safety recommendations based on assessment"""
        recommendations = []
        
        # High injury risk recommendations
        for person_id, injury_risks in safety_assessment['injury_risks'].items():
            if injury_risks.get('total_risk', 0.0) > 0.3:
                recommendations.append(f"Reduce robot speed near {person_id} due to collision risk")
                
            if injury_risks.get('impact_velocity', 0.0) > 0.3:
                recommendations.append(f"Implement predictive emergency stop for {person_id}")
        
        # Zone violation recommendations
        if safety_assessment['zone_violations']:
            recommendations.append("Increase safety zone margins")
            recommendations.append("Implement more conservative trajectory planning")
        
        # High stress recommendations
        for person_id, stress_level in safety_assessment['stress_levels'].items():
            if stress_level >= 3:  # High stress
                recommendations.append(f"Implement calming behaviors for {person_id}")
                recommendations.append(f"Consider breaking interaction with {person_id}")
        
        # Overall safety recommendations
        if safety_assessment['overall_safety_score'] < 0.7:
            recommendations.append("Consider switching to safe mode operation")
            recommendations.append("Increase monitoring frequency")
        
        return recommendations
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Get comprehensive safety report"""
        current_time = time.time()
        
        # Recent alerts (last hour)
        recent_alerts = [
            alert for alert in self.safety_alerts
            if current_time - alert['timestamp'] < 3600
        ]
        
        # Active interactions summary
        interaction_summary = {}
        for person_id, interaction in self.active_interactions.items():
            if current_time - interaction['timestamp'] < 60:  # Active in last minute
                interaction_summary[person_id] = {
                    'stress_level': interaction['stress_level'].name,
                    'stress_score': interaction['stress_score'],
                    'safety_zone_radius': interaction['safety_zone']['radius'],
                    'last_update': current_time - interaction['timestamp']
                }
        
        # Stress trends
        stress_trends = {}
        for person_id in self.human_profiles:
            trends = self.stress_detector.get_stress_trends()
            stress_trends[person_id] = trends
        
        return {
            'timestamp': current_time,
            'registered_humans': len(self.human_profiles),
            'active_interactions': len(interaction_summary),
            'recent_alerts': len(recent_alerts),
            'alert_summary': {
                'high_stress': sum(1 for a in recent_alerts if a['type'] == 'high_stress'),
                'physiological_anomaly': sum(1 for a in recent_alerts if a['type'] == 'physiological_anomaly'),
                'zone_violations': sum(1 for a in recent_alerts if a['type'] == 'zone_violation')
            },
            'interaction_summary': interaction_summary,
            'stress_trends': stress_trends,
            'zone_status': self.zone_manager.get_zone_status()
        }


# Example usage and testing
if __name__ == "__main__":
    # Create human safety system
    safety_system = HumanSafetySystem()
    
    # Create example human profile
    human_profile = HumanProfile(
        person_id="user_001",
        name="Test User",
        age=35,
        height=170.0,
        weight=70.0,
        cultural_background=CulturalContext.WESTERN_INDIVIDUALISTIC,
        preferred_robot_distance=1.2,
        medical_conditions=[]
    )
    
    # Register human
    safety_system.register_human(human_profile)
    
    print("Testing human safety system...")
    
    # Simulate interaction sequence
    for i in range(20):
        # Simulate human position and movement
        human_pos = np.array([1.0 + 0.1*i, 0.0, 0.0])
        human_vel = np.array([0.1, 0.0, 0.0])
        
        # Simulate physiological state
        base_hr = 70 + 5 * np.sin(i * 0.1)  # Varying heart rate
        stress_factor = min(1.0, i / 10.0)  # Increasing stress over time
        
        physio_state = PhysiologicalState(
            timestamp=time.time(),
            heart_rate=base_hr + 20 * stress_factor,
            skin_conductance=5.0 + 3.0 * stress_factor
        )
        
        # Update human state
        result = safety_system.update_human_state(
            "user_001", human_pos, human_vel, physio_state,
            prediction_uncertainty=0.1 + 0.05 * stress_factor
        )
        
        print(f"Step {i}: Stress level={result.get('stress_level', 0)}, "
              f"Zone radius={result.get('safety_zone_radius', 0):.2f}m")
        
        # Test interaction safety assessment
        if i % 5 == 0:  # Every 5 steps
            robot_traj = np.array([
                [0.5 + 0.05*j, 0.0, 0.0] for j in range(10)
            ])
            human_trajs = {"user_001": np.array([
                [1.0 + 0.1*(i+j), 0.0, 0.0] for j in range(10)
            ])}
            
            assessment = safety_system.assess_interaction_safety(robot_traj, human_trajs)
            print(f"  Safety assessment: Score={assessment['overall_safety_score']:.3f}")
            
            if assessment['recommendations']:
                print(f"  Recommendations: {assessment['recommendations'][0]}")
        
        time.sleep(0.1)
    
    # Get final safety report
    report = safety_system.get_safety_report()
    print(f"\nFinal Safety Report:")
    print(f"Active interactions: {report['active_interactions']}")
    print(f"Recent alerts: {report['recent_alerts']}")
    print(f"Alert summary: {report['alert_summary']}")
    
    print("\nHuman safety system test completed")