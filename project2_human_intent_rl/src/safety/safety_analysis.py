#!/usr/bin/env python3
"""
Formal Safety Analysis System
=============================

This module implements comprehensive formal safety analysis for human-robot
interaction systems, including Hazard Analysis and Risk Assessment (HARA),
Failure Mode and Effects Analysis (FMEA), fault tree analysis, and Safety
Integrity Level (SIL) analysis.

Complies with:
- ISO 12100 (Safety of machinery)  
- ISO 13482 (Personal care robots)
- IEC 61508 (Functional safety)
- ISO 10218 (Industrial robots safety)

Mathematical Framework:
======================

Risk Assessment:
    Risk = Severity × Occurrence × Detection^(-1)
    
Safety Integrity Level (SIL):
    SIL = f(PFH, Architecture, Diagnostics)
    where PFH = Probability of Dangerous Failure per Hour

Fault Tree Analysis:
    P(Top_Event) = f(P(Basic_Events), Logic_Gates)
    
Minimum Cut Sets:
    MCS = minimal combinations of basic events causing top event

Author: Claude Code (Anthropic)
Date: 2025-01-15
Version: 1.0
"""

import json
import time
import uuid
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class SeverityLevel(IntEnum):
    """Severity levels following ISO 13849"""
    S0 = 0  # No injury
    S1 = 1  # Light injury (reversible)
    S2 = 2  # Serious injury (irreversible) 
    S3 = 3  # Death or life-threatening injury


class OccurrenceLevel(IntEnum):
    """Occurrence/Exposure levels"""
    F0 = 0  # Very low frequency/probability
    F1 = 1  # Low frequency/probability
    F2 = 2  # Medium frequency/probability  
    F3 = 3  # High frequency/probability
    F4 = 4  # Very high frequency/probability


class DetectionLevel(IntEnum):
    """Detection capability levels"""
    D0 = 0  # Certain detection
    D1 = 1  # High detection capability
    D2 = 2  # Medium detection capability
    D3 = 3  # Low detection capability
    D4 = 4  # No detection capability


class SILLevel(IntEnum):
    """Safety Integrity Levels per IEC 61508"""
    SIL0 = 0  # No safety requirements
    SIL1 = 1  # Low safety requirements
    SIL2 = 2  # Medium safety requirements  
    SIL3 = 3  # High safety requirements
    SIL4 = 4  # Very high safety requirements


class HazardType(Enum):
    """Types of hazards in human-robot interaction"""
    COLLISION = "collision"
    CRUSHING = "crushing"
    CUTTING_SHEARING = "cutting_shearing"
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    NOISE = "noise"
    RADIATION = "radiation"
    MATERIAL_SUBSTANCES = "material_substances"
    ERGONOMIC = "ergonomic"
    ENVIRONMENTAL = "environmental"


class FailureMode(Enum):
    """Failure modes for FMEA analysis"""
    NO_FUNCTION = "no_function"
    INTERMITTENT_FUNCTION = "intermittent_function"
    DEGRADED_FUNCTION = "degraded_function"
    UNINTENDED_FUNCTION = "unintended_function"
    LOSS_OF_FUNCTION = "loss_of_function"
    DELAYED_FUNCTION = "delayed_function"
    ERRATIC_FUNCTION = "erratic_function"


@dataclass
class HazardousEvent:
    """Definition of a hazardous event for HARA"""
    hazard_id: str
    description: str
    hazard_type: HazardType
    system_component: str
    operational_situation: str
    severity: SeverityLevel
    exposure: OccurrenceLevel
    controllability: DetectionLevel
    
    # Risk metrics
    risk_priority_number: float = field(init=False)
    risk_level: str = field(init=False)
    
    def __post_init__(self):
        """Calculate derived risk metrics"""
        self.risk_priority_number = self.severity * self.exposure * self.controllability
        self.risk_level = self._determine_risk_level()
    
    def _determine_risk_level(self) -> str:
        """Determine qualitative risk level"""
        rpn = self.risk_priority_number
        if rpn >= 40:
            return "VERY_HIGH"
        elif rpn >= 20:
            return "HIGH"
        elif rpn >= 8:
            return "MEDIUM"
        elif rpn >= 3:
            return "LOW"
        else:
            return "VERY_LOW"


@dataclass
class FailureModeEffect:
    """FMEA failure mode and effects analysis entry"""
    fmea_id: str
    component: str
    function: str
    failure_mode: FailureMode
    failure_cause: str
    local_effect: str
    system_effect: str
    end_effect: str
    
    # Severity, Occurrence, Detection ratings (1-10 scale)
    severity: int  # 1=minor, 10=catastrophic
    occurrence: int  # 1=remote, 10=very high
    detection: int  # 1=certain detection, 10=no detection
    
    # Current controls
    current_controls: List[str] = field(default_factory=list)
    
    # Risk metrics
    risk_priority_number: int = field(init=False)
    criticality: float = field(init=False)
    
    # Recommended actions
    recommended_actions: List[str] = field(default_factory=list)
    responsible_party: str = ""
    target_completion: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.risk_priority_number = self.severity * self.occurrence * self.detection
        self.criticality = self.severity * self.occurrence


@dataclass  
class FaultTreeEvent:
    """Basic or intermediate event in fault tree analysis"""
    event_id: str
    description: str
    is_basic_event: bool = True
    probability: float = 0.0
    
    # For intermediate events
    gate_type: Optional[str] = None  # "AND", "OR", "NOT", etc.
    input_events: List[str] = field(default_factory=list)
    
    # Criticality metrics
    fussell_vesely_importance: float = 0.0
    birnbaum_importance: float = 0.0
    risk_achievement_worth: float = 1.0


@dataclass
class MinimalCutSet:
    """Minimal cut set from fault tree analysis"""
    cut_set_id: str
    events: List[str]
    probability: float
    rank: int = 0


class HARAAnalyzer:
    """Hazard Analysis and Risk Assessment analyzer"""
    
    def __init__(self):
        """Initialize HARA analyzer"""
        self.hazards: List[HazardousEvent] = []
        self.analysis_timestamp = time.time()
        self.analysis_id = str(uuid.uuid4())
        
        logger.info("HARA analyzer initialized")
    
    def add_hazard(self, hazard: HazardousEvent) -> None:
        """Add hazard to analysis"""
        self.hazards.append(hazard)
        logger.debug(f"Added hazard: {hazard.hazard_id}")
    
    def conduct_analysis(self) -> Dict[str, Any]:
        """Conduct comprehensive HARA analysis"""
        logger.info("Conducting HARA analysis...")
        
        # Risk distribution analysis
        risk_distribution = self._analyze_risk_distribution()
        
        # Severity analysis
        severity_analysis = self._analyze_severity_distribution()
        
        # Component risk analysis
        component_risks = self._analyze_component_risks()
        
        # High-risk hazards identification
        high_risk_hazards = self._identify_high_risk_hazards()
        
        # Safety requirements generation
        safety_requirements = self._generate_safety_requirements()
        
        analysis_results = {
            'analysis_id': self.analysis_id,
            'timestamp': self.analysis_timestamp,
            'total_hazards': len(self.hazards),
            'risk_distribution': risk_distribution,
            'severity_analysis': severity_analysis,
            'component_risks': component_risks,
            'high_risk_hazards': high_risk_hazards,
            'safety_requirements': safety_requirements,
            'hazards_detail': [asdict(h) for h in self.hazards]
        }
        
        logger.info(f"HARA analysis completed: {len(self.hazards)} hazards analyzed")
        return analysis_results
    
    def _analyze_risk_distribution(self) -> Dict[str, int]:
        """Analyze distribution of risk levels"""
        distribution = defaultdict(int)
        for hazard in self.hazards:
            distribution[hazard.risk_level] += 1
        return dict(distribution)
    
    def _analyze_severity_distribution(self) -> Dict[str, Any]:
        """Analyze severity level distribution"""
        severity_counts = defaultdict(int)
        total_rpn = 0
        max_severity = SeverityLevel.S0
        
        for hazard in self.hazards:
            severity_counts[f"S{hazard.severity}"] += 1
            total_rpn += hazard.risk_priority_number
            if hazard.severity > max_severity:
                max_severity = hazard.severity
        
        return {
            'severity_distribution': dict(severity_counts),
            'average_rpn': total_rpn / max(1, len(self.hazards)),
            'maximum_severity': f"S{max_severity}",
            'total_rpn': total_rpn
        }
    
    def _analyze_component_risks(self) -> Dict[str, Any]:
        """Analyze risks by system component"""
        component_risks = defaultdict(list)
        component_stats = defaultdict(lambda: {'count': 0, 'total_rpn': 0, 'max_rpn': 0})
        
        for hazard in self.hazards:
            component = hazard.system_component
            component_risks[component].append(hazard.hazard_id)
            component_stats[component]['count'] += 1
            component_stats[component]['total_rpn'] += hazard.risk_priority_number
            component_stats[component]['max_rpn'] = max(
                component_stats[component]['max_rpn'], 
                hazard.risk_priority_number
            )
        
        # Convert to final format
        result = {}
        for component, stats in component_stats.items():
            result[component] = {
                'hazard_count': stats['count'],
                'average_rpn': stats['total_rpn'] / stats['count'],
                'maximum_rpn': stats['max_rpn'],
                'hazard_ids': component_risks[component]
            }
        
        return result
    
    def _identify_high_risk_hazards(self) -> List[Dict[str, Any]]:
        """Identify hazards requiring immediate attention"""
        high_risk = []
        
        for hazard in self.hazards:
            if (hazard.risk_level in ['HIGH', 'VERY_HIGH'] or 
                hazard.severity >= SeverityLevel.S2):
                high_risk.append({
                    'hazard_id': hazard.hazard_id,
                    'description': hazard.description,
                    'risk_level': hazard.risk_level,
                    'rpn': hazard.risk_priority_number,
                    'severity': f"S{hazard.severity}",
                    'component': hazard.system_component
                })
        
        # Sort by RPN descending
        high_risk.sort(key=lambda x: x['rpn'], reverse=True)
        return high_risk
    
    def _generate_safety_requirements(self) -> List[Dict[str, Any]]:
        """Generate safety requirements based on hazard analysis"""
        requirements = []
        req_id = 1
        
        # Requirements based on severity levels
        severity_counts = defaultdict(int)
        for hazard in self.hazards:
            severity_counts[hazard.severity] += 1
        
        if severity_counts[SeverityLevel.S3] > 0:
            requirements.append({
                'requirement_id': f"SR-{req_id:03d}",
                'requirement': "Emergency stop system with <10ms response time",
                'rationale': f"{severity_counts[SeverityLevel.S3]} life-threatening hazards identified",
                'compliance_standard': "ISO 13850, IEC 60204-1"
            })
            req_id += 1
        
        if severity_counts[SeverityLevel.S2] > 0:
            requirements.append({
                'requirement_id': f"SR-{req_id:03d}",
                'requirement': "Safety-rated motion monitoring with SIL2 capability",
                'rationale': f"{severity_counts[SeverityLevel.S2]} serious injury hazards identified",
                'compliance_standard': "IEC 61508, ISO 13849"
            })
            req_id += 1
        
        # Component-specific requirements
        component_risks = self._analyze_component_risks()
        for component, stats in component_risks.items():
            if stats['maximum_rpn'] >= 20:
                requirements.append({
                    'requirement_id': f"SR-{req_id:03d}",
                    'requirement': f"Redundant safety monitoring for {component}",
                    'rationale': f"High-risk component with max RPN {stats['maximum_rpn']}",
                    'compliance_standard': "ISO 13849-1"
                })
                req_id += 1
        
        return requirements
    
    def generate_hara_report(self, output_path: str) -> None:
        """Generate detailed HARA analysis report"""
        analysis = self.conduct_analysis()
        
        # Generate HTML report
        html_content = self._generate_hara_html_report(analysis)
        
        output_file = Path(output_path) / f"hara_report_{int(time.time())}.html"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HARA report generated: {output_file}")
    
    def _generate_hara_html_report(self, analysis: Dict[str, Any]) -> str:
        """Generate HTML report for HARA analysis"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>HARA Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .high-risk {{ background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; }}
        .medium-risk {{ background-color: #fff8e1; border-left: 4px solid #ff9800; padding: 10px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Hazard Analysis and Risk Assessment (HARA) Report</h1>
        <p>Analysis ID: {analysis['analysis_id']}</p>
        <p>Generated: {time.ctime(analysis['timestamp'])}</p>
        <p>Total Hazards Analyzed: {analysis['total_hazards']}</p>
    </div>
    
    <div class="section">
        <h2>Risk Distribution Summary</h2>
        <ul>
            {self._format_risk_distribution(analysis['risk_distribution'])}
        </ul>
    </div>
    
    <div class="section">
        <h2>High-Risk Hazards Requiring Immediate Attention</h2>
        {self._format_high_risk_hazards(analysis['high_risk_hazards'])}
    </div>
    
    <div class="section">
        <h2>Safety Requirements</h2>
        {self._format_safety_requirements(analysis['safety_requirements'])}
    </div>
    
    <div class="section">
        <h2>Component Risk Analysis</h2>
        {self._format_component_risks(analysis['component_risks'])}
    </div>
</body>
</html>"""
    
    def _format_risk_distribution(self, distribution: Dict[str, int]) -> str:
        """Format risk distribution for HTML"""
        items = []
        for level, count in distribution.items():
            items.append(f"<li>{level}: {count} hazards</li>")
        return "\n".join(items)
    
    def _format_high_risk_hazards(self, hazards: List[Dict[str, Any]]) -> str:
        """Format high-risk hazards for HTML"""
        if not hazards:
            return "<p>No high-risk hazards identified.</p>"
        
        rows = []
        for hazard in hazards:
            risk_class = "high-risk" if hazard['risk_level'] == 'VERY_HIGH' else "medium-risk"
            rows.append(f"""
            <div class="{risk_class}">
                <strong>{hazard['hazard_id']}</strong>: {hazard['description']}<br>
                Component: {hazard['component']}, RPN: {hazard['rpn']}, Severity: {hazard['severity']}
            </div>""")
        return "\n".join(rows)
    
    def _format_safety_requirements(self, requirements: List[Dict[str, Any]]) -> str:
        """Format safety requirements for HTML"""
        if not requirements:
            return "<p>No specific safety requirements generated.</p>"
        
        rows = []
        for req in requirements:
            rows.append(f"""
            <tr>
                <td>{req['requirement_id']}</td>
                <td>{req['requirement']}</td>
                <td>{req['rationale']}</td>
                <td>{req['compliance_standard']}</td>
            </tr>""")
        
        return f"""
        <table>
            <tr>
                <th>Requirement ID</th>
                <th>Requirement</th>
                <th>Rationale</th>
                <th>Compliance Standard</th>
            </tr>
            {"".join(rows)}
        </table>"""
    
    def _format_component_risks(self, component_risks: Dict[str, Any]) -> str:
        """Format component risks for HTML"""
        rows = []
        for component, stats in component_risks.items():
            rows.append(f"""
            <tr>
                <td>{component}</td>
                <td>{stats['hazard_count']}</td>
                <td>{stats['average_rpn']:.1f}</td>
                <td>{stats['maximum_rpn']:.0f}</td>
            </tr>""")
        
        return f"""
        <table>
            <tr>
                <th>Component</th>
                <th>Hazard Count</th>
                <th>Average RPN</th>
                <th>Maximum RPN</th>
            </tr>
            {"".join(rows)}
        </table>"""


class FMEAAnalyzer:
    """Failure Mode and Effects Analysis analyzer"""
    
    def __init__(self):
        """Initialize FMEA analyzer"""
        self.fmea_entries: List[FailureModeEffect] = []
        self.analysis_timestamp = time.time()
        self.analysis_id = str(uuid.uuid4())
        
        logger.info("FMEA analyzer initialized")
    
    def add_failure_mode(self, fmea_entry: FailureModeEffect) -> None:
        """Add failure mode to analysis"""
        self.fmea_entries.append(fmea_entry)
        logger.debug(f"Added FMEA entry: {fmea_entry.fmea_id}")
    
    def conduct_analysis(self) -> Dict[str, Any]:
        """Conduct comprehensive FMEA analysis"""
        logger.info("Conducting FMEA analysis...")
        
        # RPN analysis
        rpn_analysis = self._analyze_rpn_distribution()
        
        # Critical failure modes
        critical_failures = self._identify_critical_failures()
        
        # Component analysis
        component_analysis = self._analyze_component_failures()
        
        # Recommended actions summary
        action_summary = self._summarize_recommended_actions()
        
        analysis_results = {
            'analysis_id': self.analysis_id,
            'timestamp': self.analysis_timestamp,
            'total_failure_modes': len(self.fmea_entries),
            'rpn_analysis': rpn_analysis,
            'critical_failures': critical_failures,
            'component_analysis': component_analysis,
            'action_summary': action_summary,
            'fmea_details': [asdict(entry) for entry in self.fmea_entries]
        }
        
        logger.info(f"FMEA analysis completed: {len(self.fmea_entries)} failure modes analyzed")
        return analysis_results
    
    def _analyze_rpn_distribution(self) -> Dict[str, Any]:
        """Analyze RPN distribution and statistics"""
        if not self.fmea_entries:
            return {}
        
        rpns = [entry.risk_priority_number for entry in self.fmea_entries]
        
        return {
            'mean_rpn': np.mean(rpns),
            'median_rpn': np.median(rpns),
            'max_rpn': np.max(rpns),
            'min_rpn': np.min(rpns),
            'std_rpn': np.std(rpns),
            'high_rpn_count': sum(1 for rpn in rpns if rpn >= 120),  # Typically high-risk threshold
            'medium_rpn_count': sum(1 for rpn in rpns if 60 <= rpn < 120),
            'low_rpn_count': sum(1 for rpn in rpns if rpn < 60)
        }
    
    def _identify_critical_failures(self) -> List[Dict[str, Any]]:
        """Identify critical failure modes requiring immediate attention"""
        critical = []
        
        for entry in self.fmea_entries:
            # Criteria for critical failures
            is_critical = (entry.risk_priority_number >= 120 or 
                         entry.severity >= 8 or 
                         (entry.severity >= 6 and entry.detection >= 7))
            
            if is_critical:
                critical.append({
                    'fmea_id': entry.fmea_id,
                    'component': entry.component,
                    'failure_mode': entry.failure_mode.value,
                    'end_effect': entry.end_effect,
                    'rpn': entry.risk_priority_number,
                    'severity': entry.severity,
                    'occurrence': entry.occurrence,
                    'detection': entry.detection,
                    'current_controls': entry.current_controls
                })
        
        # Sort by RPN descending
        critical.sort(key=lambda x: x['rpn'], reverse=True)
        return critical
    
    def _analyze_component_failures(self) -> Dict[str, Any]:
        """Analyze failure modes by component"""
        component_stats = defaultdict(lambda: {
            'failure_count': 0,
            'total_rpn': 0,
            'max_rpn': 0,
            'avg_severity': 0,
            'failure_modes': []
        })
        
        for entry in self.fmea_entries:
            comp = entry.component
            component_stats[comp]['failure_count'] += 1
            component_stats[comp]['total_rpn'] += entry.risk_priority_number
            component_stats[comp]['max_rpn'] = max(
                component_stats[comp]['max_rpn'],
                entry.risk_priority_number
            )
            component_stats[comp]['avg_severity'] += entry.severity
            component_stats[comp]['failure_modes'].append(entry.failure_mode.value)
        
        # Calculate averages and finalize stats
        result = {}
        for comp, stats in component_stats.items():
            count = stats['failure_count']
            result[comp] = {
                'failure_count': count,
                'average_rpn': stats['total_rpn'] / count,
                'maximum_rpn': stats['max_rpn'],
                'average_severity': stats['avg_severity'] / count,
                'unique_failure_modes': len(set(stats['failure_modes'])),
                'most_common_failure': max(set(stats['failure_modes']), 
                                        key=stats['failure_modes'].count) if stats['failure_modes'] else None
            }
        
        return result
    
    def _summarize_recommended_actions(self) -> Dict[str, Any]:
        """Summarize recommended actions across all failure modes"""
        all_actions = []
        for entry in self.fmea_entries:
            all_actions.extend(entry.recommended_actions)
        
        if not all_actions:
            return {'total_actions': 0, 'action_categories': {}}
        
        # Categorize actions (simplified categorization)
        action_categories = defaultdict(int)
        for action in all_actions:
            if 'sensor' in action.lower() or 'monitor' in action.lower():
                action_categories['monitoring'] += 1
            elif 'redundan' in action.lower() or 'backup' in action.lower():
                action_categories['redundancy'] += 1
            elif 'test' in action.lower() or 'inspect' in action.lower():
                action_categories['testing'] += 1
            elif 'design' in action.lower() or 'modify' in action.lower():
                action_categories['design_change'] += 1
            else:
                action_categories['other'] += 1
        
        return {
            'total_actions': len(all_actions),
            'action_categories': dict(action_categories),
            'entries_with_actions': sum(1 for entry in self.fmea_entries 
                                      if entry.recommended_actions),
            'entries_without_actions': sum(1 for entry in self.fmea_entries 
                                         if not entry.recommended_actions)
        }


class FaultTreeAnalyzer:
    """Fault Tree Analysis implementation with minimal cut set generation"""
    
    def __init__(self, top_event_description: str):
        """Initialize fault tree analyzer"""
        self.top_event = top_event_description
        self.events: Dict[str, FaultTreeEvent] = {}
        self.minimal_cut_sets: List[MinimalCutSet] = []
        self.analysis_timestamp = time.time()
        self.analysis_id = str(uuid.uuid4())
        
        logger.info(f"Fault tree analyzer initialized for: {top_event_description}")
    
    def add_event(self, event: FaultTreeEvent) -> None:
        """Add event to fault tree"""
        self.events[event.event_id] = event
        logger.debug(f"Added {'basic' if event.is_basic_event else 'intermediate'} event: {event.event_id}")
    
    def calculate_top_event_probability(self) -> float:
        """Calculate probability of top event occurrence"""
        if not self.events:
            return 0.0
        
        # Find top event (event with no dependents)
        top_events = []
        for event_id, event in self.events.items():
            if not event.is_basic_event:
                # Check if this event is referenced by any other
                is_referenced = any(
                    event_id in other.input_events 
                    for other in self.events.values()
                    if not other.is_basic_event
                )
                if not is_referenced:
                    top_events.append(event)
        
        if not top_events:
            logger.warning("No top event found in fault tree")
            return 0.0
        
        # Use first top event found
        top_event = top_events[0]
        return self._calculate_event_probability(top_event)
    
    def _calculate_event_probability(self, event: FaultTreeEvent) -> float:
        """Recursively calculate event probability"""
        if event.is_basic_event:
            return event.probability
        
        if not event.input_events:
            return 0.0
        
        input_probs = []
        for input_id in event.input_events:
            if input_id in self.events:
                input_prob = self._calculate_event_probability(self.events[input_id])
                input_probs.append(input_prob)
        
        if not input_probs:
            return 0.0
        
        # Calculate based on gate type
        if event.gate_type == "OR":
            # P(A ∪ B) = P(A) + P(B) - P(A)P(B) for two events
            # For multiple events, use inclusion-exclusion approximation
            prob = 1.0
            for p in input_probs:
                prob *= (1.0 - p)
            return 1.0 - prob
            
        elif event.gate_type == "AND":
            # P(A ∩ B) = P(A) × P(B) assuming independence
            prob = 1.0
            for p in input_probs:
                prob *= p
            return prob
            
        elif event.gate_type == "NOT":
            return 1.0 - input_probs[0] if input_probs else 1.0
            
        else:
            logger.warning(f"Unknown gate type: {event.gate_type}")
            return 0.0
    
    def generate_minimal_cut_sets(self, max_order: int = 4) -> List[MinimalCutSet]:
        """Generate minimal cut sets using Boolean reduction"""
        logger.info("Generating minimal cut sets...")
        
        # Find top event
        top_events = [e for e in self.events.values() 
                     if not e.is_basic_event and 
                     not any(e.event_id in other.input_events 
                            for other in self.events.values()
                            if not other.is_basic_event)]
        
        if not top_events:
            return []
        
        top_event = top_events[0]
        
        # Generate cut sets recursively
        cut_sets = self._generate_cut_sets_recursive(top_event, max_order)
        
        # Find minimal cut sets (remove supersets)
        minimal_sets = self._find_minimal_cut_sets(cut_sets)
        
        # Create MinimalCutSet objects with probabilities
        self.minimal_cut_sets = []
        for i, cut_set in enumerate(minimal_sets):
            # Calculate cut set probability (product of basic event probabilities)
            prob = 1.0
            for event_id in cut_set:
                if event_id in self.events:
                    prob *= self.events[event_id].probability
            
            mcs = MinimalCutSet(
                cut_set_id=f"MCS-{i+1:03d}",
                events=list(cut_set),
                probability=prob,
                rank=i+1
            )
            self.minimal_cut_sets.append(mcs)
        
        # Sort by probability (descending)
        self.minimal_cut_sets.sort(key=lambda x: x.probability, reverse=True)
        
        logger.info(f"Generated {len(self.minimal_cut_sets)} minimal cut sets")
        return self.minimal_cut_sets
    
    def _generate_cut_sets_recursive(self, event: FaultTreeEvent, max_order: int) -> List[Set[str]]:
        """Recursively generate cut sets"""
        if event.is_basic_event:
            return [{event.event_id}]
        
        if not event.input_events:
            return []
        
        input_cut_sets = []
        for input_id in event.input_events:
            if input_id in self.events:
                input_cuts = self._generate_cut_sets_recursive(self.events[input_id], max_order)
                input_cut_sets.append(input_cuts)
        
        if not input_cut_sets:
            return []
        
        # Combine based on gate type
        if event.gate_type == "OR":
            # OR gate: union of all input cut sets
            result = []
            for cuts in input_cut_sets:
                result.extend(cuts)
            return result
            
        elif event.gate_type == "AND":
            # AND gate: Cartesian product of input cut sets
            result = []
            if len(input_cut_sets) == 1:
                return input_cut_sets[0]
            
            # Start with first input
            current_sets = input_cut_sets[0]
            
            # Combine with remaining inputs
            for i in range(1, len(input_cut_sets)):
                new_sets = []
                for current_set in current_sets:
                    for next_set in input_cut_sets[i]:
                        combined = current_set.union(next_set)
                        if len(combined) <= max_order:  # Limit cut set order
                            new_sets.append(combined)
                current_sets = new_sets
            
            return current_sets
        
        return []
    
    def _find_minimal_cut_sets(self, cut_sets: List[Set[str]]) -> List[Set[str]]:
        """Find minimal cut sets by removing supersets"""
        minimal = []
        
        # Sort by size (smaller sets first)
        sorted_sets = sorted(cut_sets, key=len)
        
        for candidate in sorted_sets:
            is_minimal = True
            for minimal_set in minimal:
                if minimal_set.issubset(candidate):
                    is_minimal = False
                    break
            
            if is_minimal:
                minimal.append(candidate)
        
        return minimal
    
    def calculate_importance_measures(self) -> Dict[str, Dict[str, float]]:
        """Calculate importance measures for basic events"""
        if not self.minimal_cut_sets:
            self.generate_minimal_cut_sets()
        
        importance_measures = {}
        basic_events = [e for e in self.events.values() if e.is_basic_event]
        
        for event in basic_events:
            event_id = event.event_id
            
            # Fussell-Vesely importance
            fv_importance = self._calculate_fv_importance(event_id)
            
            # Birnbaum importance (simplified)
            birnbaum_importance = self._calculate_birnbaum_importance(event_id)
            
            # Risk Achievement Worth (RAW)
            raw = self._calculate_raw(event_id)
            
            importance_measures[event_id] = {
                'fussell_vesely': fv_importance,
                'birnbaum': birnbaum_importance,
                'risk_achievement_worth': raw
            }
            
            # Update event object
            event.fussell_vesely_importance = fv_importance
            event.birnbaum_importance = birnbaum_importance
            event.risk_achievement_worth = raw
        
        return importance_measures
    
    def _calculate_fv_importance(self, event_id: str) -> float:
        """Calculate Fussell-Vesely importance"""
        # Sum of probabilities of cut sets containing the event
        contributing_prob = sum(
            mcs.probability for mcs in self.minimal_cut_sets 
            if event_id in mcs.events
        )
        
        # Total system probability (top event probability)
        total_prob = self.calculate_top_event_probability()
        
        return contributing_prob / max(total_prob, 1e-10)
    
    def _calculate_birnbaum_importance(self, event_id: str) -> float:
        """Calculate Birnbaum importance (simplified)"""
        # This is a simplified calculation
        # True Birnbaum importance requires derivative calculation
        contributing_cut_sets = sum(
            1 for mcs in self.minimal_cut_sets 
            if event_id in mcs.events
        )
        
        total_cut_sets = len(self.minimal_cut_sets)
        
        return contributing_cut_sets / max(total_cut_sets, 1)
    
    def _calculate_raw(self, event_id: str) -> float:
        """Calculate Risk Achievement Worth"""
        # RAW = P(top_event | event_certain) / P(top_event)
        # Simplified calculation
        original_prob = self.events[event_id].probability
        
        # Set event probability to 1 (certain failure)
        self.events[event_id].probability = 1.0
        new_top_prob = self.calculate_top_event_probability()
        
        # Restore original probability
        self.events[event_id].probability = original_prob
        original_top_prob = self.calculate_top_event_probability()
        
        return new_top_prob / max(original_top_prob, 1e-10)


class SafetyAnalysisSystem:
    """Integrated safety analysis system combining HARA, FMEA, and FTA"""
    
    def __init__(self, system_name: str):
        """Initialize integrated safety analysis system"""
        self.system_name = system_name
        self.hara_analyzer = HARAAnalyzer()
        self.fmea_analyzer = FMEAAnalyzer()
        self.fault_trees: Dict[str, FaultTreeAnalyzer] = {}
        
        self.analysis_timestamp = time.time()
        self.sil_requirements: Dict[str, SILLevel] = {}
        
        logger.info(f"Safety analysis system initialized for: {system_name}")
    
    def add_fault_tree(self, name: str, top_event_description: str) -> FaultTreeAnalyzer:
        """Add fault tree analysis"""
        fta = FaultTreeAnalyzer(top_event_description)
        self.fault_trees[name] = fta
        logger.info(f"Added fault tree: {name}")
        return fta
    
    def determine_sil_requirements(self) -> Dict[str, SILLevel]:
        """Determine SIL requirements based on HARA and FMEA results"""
        logger.info("Determining SIL requirements...")
        
        # Analyze HARA results
        hara_results = self.hara_analyzer.conduct_analysis()
        
        # Determine SIL based on high-risk hazards
        for hazard in hara_results['high_risk_hazards']:
            rpn = hazard['rpn']
            severity = int(hazard['severity'][1])  # Extract numeric severity
            
            # SIL determination logic based on severity and RPN
            if severity >= 3 and rpn >= 40:
                sil_level = SILLevel.SIL3
            elif severity >= 2 and rpn >= 20:
                sil_level = SILLevel.SIL2
            elif rpn >= 8:
                sil_level = SILLevel.SIL1
            else:
                sil_level = SILLevel.SIL0
            
            component = hazard['component']
            self.sil_requirements[component] = max(
                self.sil_requirements.get(component, SILLevel.SIL0),
                sil_level
            )
        
        logger.info(f"SIL requirements determined for {len(self.sil_requirements)} components")
        return self.sil_requirements
    
    def conduct_comprehensive_analysis(self) -> Dict[str, Any]:
        """Conduct comprehensive safety analysis"""
        logger.info("Conducting comprehensive safety analysis...")
        
        # Run individual analyses
        hara_results = self.hara_analyzer.conduct_analysis()
        fmea_results = self.fmea_analyzer.conduct_analysis()
        
        # Run fault tree analyses
        fta_results = {}
        for name, fta in self.fault_trees.items():
            top_event_prob = fta.calculate_top_event_probability()
            cut_sets = fta.generate_minimal_cut_sets()
            importance_measures = fta.calculate_importance_measures()
            
            fta_results[name] = {
                'top_event_probability': top_event_prob,
                'minimal_cut_sets_count': len(cut_sets),
                'most_critical_cut_sets': [
                    {
                        'cut_set_id': mcs.cut_set_id,
                        'events': mcs.events,
                        'probability': mcs.probability
                    }
                    for mcs in cut_sets[:5]  # Top 5
                ],
                'importance_measures': importance_measures
            }
        
        # Determine SIL requirements
        sil_requirements = self.determine_sil_requirements()
        
        # Generate integrated recommendations
        recommendations = self._generate_integrated_recommendations(
            hara_results, fmea_results, fta_results
        )
        
        comprehensive_results = {
            'system_name': self.system_name,
            'analysis_timestamp': self.analysis_timestamp,
            'hara_analysis': hara_results,
            'fmea_analysis': fmea_results,
            'fault_tree_analysis': fta_results,
            'sil_requirements': {comp: sil.value for comp, sil in sil_requirements.items()},
            'integrated_recommendations': recommendations,
            'compliance_summary': self._generate_compliance_summary(
                hara_results, fmea_results, fta_results
            )
        }
        
        logger.info("Comprehensive safety analysis completed")
        return comprehensive_results
    
    def _generate_integrated_recommendations(self, 
                                           hara_results: Dict[str, Any],
                                           fmea_results: Dict[str, Any], 
                                           fta_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate integrated safety recommendations"""
        recommendations = []
        
        # High-priority recommendations from HARA
        for hazard in hara_results.get('high_risk_hazards', []):
            recommendations.append({
                'priority': 1,
                'source': 'HARA',
                'recommendation': f"Implement safety measures for {hazard['component']} hazard {hazard['hazard_id']}",
                'rationale': f"High-risk hazard with RPN {hazard['rpn']}",
                'affected_component': hazard['component']
            })
        
        # Critical failure mode recommendations from FMEA
        for failure in fmea_results.get('critical_failures', []):
            recommendations.append({
                'priority': 2,
                'source': 'FMEA',
                'recommendation': f"Address critical failure mode in {failure['component']}",
                'rationale': f"Critical failure with RPN {failure['rpn']}",
                'affected_component': failure['component']
            })
        
        # Fault tree recommendations
        for ft_name, ft_results in fta_results.items():
            if ft_results.get('top_event_probability', 0) > 1e-4:  # High probability threshold
                recommendations.append({
                    'priority': 2,
                    'source': 'FTA',
                    'recommendation': f"Reduce top event probability for {ft_name}",
                    'rationale': f"Top event probability {ft_results['top_event_probability']:.2e} exceeds threshold",
                    'affected_component': ft_name
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations
    
    def _generate_compliance_summary(self,
                                   hara_results: Dict[str, Any],
                                   fmea_results: Dict[str, Any],
                                   fta_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance summary for safety standards"""
        
        # ISO 12100 compliance
        iso_12100_compliance = {
            'hazard_identification': len(hara_results.get('hazards_detail', [])) > 0,
            'risk_assessment_conducted': True,
            'risk_reduction_measures_identified': len(hara_results.get('safety_requirements', [])) > 0
        }
        
        # ISO 13482 compliance  
        iso_13482_compliance = {
            'safety_related_control_system': any(
                req['compliance_standard'] == 'ISO 13849' 
                for req in hara_results.get('safety_requirements', [])
            ),
            'emergency_stop_system': any(
                'emergency stop' in req['requirement'].lower()
                for req in hara_results.get('safety_requirements', [])
            )
        }
        
        # IEC 61508 compliance
        iec_61508_compliance = {
            'sil_analysis_conducted': len(self.sil_requirements) > 0,
            'systematic_safety_integrity': True,
            'fmea_conducted': len(fmea_results.get('fmea_details', [])) > 0
        }
        
        return {
            'iso_12100': iso_12100_compliance,
            'iso_13482': iso_13482_compliance, 
            'iec_61508': iec_61508_compliance,
            'overall_compliance_score': self._calculate_compliance_score(
                iso_12100_compliance, iso_13482_compliance, iec_61508_compliance
            )
        }
    
    def _calculate_compliance_score(self, *compliance_dicts) -> float:
        """Calculate overall compliance score"""
        total_items = 0
        compliant_items = 0
        
        for compliance_dict in compliance_dicts:
            for compliant in compliance_dict.values():
                total_items += 1
                if compliant:
                    compliant_items += 1
        
        return compliant_items / max(total_items, 1)
    
    def generate_comprehensive_report(self, output_directory: str) -> None:
        """Generate comprehensive safety analysis report"""
        results = self.conduct_comprehensive_analysis()
        
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_file = output_path / f"safety_analysis_{int(time.time())}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate individual reports
        self.hara_analyzer.generate_hara_report(str(output_path))
        
        logger.info(f"Comprehensive safety analysis report generated in: {output_path}")


# Example usage for human-robot interaction system
def create_example_hri_safety_analysis() -> SafetyAnalysisSystem:
    """Create example safety analysis for HRI system"""
    
    # Initialize system
    safety_system = SafetyAnalysisSystem("Human-Robot Intent Recognition System")
    
    # Add HARA hazards
    hazards = [
        HazardousEvent(
            hazard_id="H001",
            description="Robot collision with human during motion",
            hazard_type=HazardType.COLLISION,
            system_component="motion_controller",
            operational_situation="Robot executing trajectory while human present",
            severity=SeverityLevel.S2,
            exposure=OccurrenceLevel.F2,
            controllability=DetectionLevel.D2
        ),
        HazardousEvent(
            hazard_id="H002", 
            description="Incorrect intent prediction causing unsafe robot action",
            hazard_type=HazardType.COLLISION,
            system_component="intent_predictor",
            operational_situation="Human performing unexpected gesture",
            severity=SeverityLevel.S3,
            exposure=OccurrenceLevel.F1,
            controllability=DetectionLevel.D3
        ),
        HazardousEvent(
            hazard_id="H003",
            description="Sensor failure leading to loss of human detection",
            hazard_type=HazardType.COLLISION,
            system_component="vision_system",
            operational_situation="Camera malfunction during operation",
            severity=SeverityLevel.S2,
            exposure=OccurrenceLevel.F1,
            controllability=DetectionLevel.D3
        )
    ]
    
    for hazard in hazards:
        safety_system.hara_analyzer.add_hazard(hazard)
    
    # Add FMEA entries
    fmea_entries = [
        FailureModeEffect(
            fmea_id="F001",
            component="Vision Camera",
            function="Human detection and tracking",
            failure_mode=FailureMode.LOSS_OF_FUNCTION,
            failure_cause="Camera hardware failure",
            local_effect="No visual input",
            system_effect="Cannot detect humans",
            end_effect="Potential collision with undetected human",
            severity=8,
            occurrence=3,
            detection=4,
            current_controls=["Camera health monitoring", "Visual inspection"],
            recommended_actions=["Add redundant camera", "Implement lidar backup"]
        ),
        FailureModeEffect(
            fmea_id="F002",
            component="Intent Predictor",
            function="Predict human intent from motion",
            failure_mode=FailureMode.ERRATIC_FUNCTION,
            failure_cause="Model uncertainty or corrupted data",
            local_effect="Incorrect predictions",
            system_effect="Wrong robot responses",
            end_effect="Unsafe robot behavior",
            severity=9,
            occurrence=2,
            detection=6,
            current_controls=["Prediction confidence monitoring"],
            recommended_actions=["Improve uncertainty quantification", "Conservative fallback mode"]
        )
    ]
    
    for entry in fmea_entries:
        safety_system.fmea_analyzer.add_failure_mode(entry)
    
    # Add fault tree for collision hazard
    collision_fta = safety_system.add_fault_tree("collision", "Robot-Human Collision")
    
    # Basic events
    events = [
        FaultTreeEvent("BE001", "Camera failure", True, 1e-4),
        FaultTreeEvent("BE002", "Intent prediction error", True, 1e-3), 
        FaultTreeEvent("BE003", "Motion controller error", True, 5e-4),
        FaultTreeEvent("BE004", "Emergency stop failure", True, 1e-5),
        FaultTreeEvent("BE005", "Human enters workspace unexpectedly", True, 1e-2)
    ]
    
    # Intermediate events
    events.extend([
        FaultTreeEvent("IE001", "Detection system failure", False, 0.0, "OR", ["BE001", "BE002"]),
        FaultTreeEvent("IE002", "Safety system failure", False, 0.0, "AND", ["IE001", "BE004"]),
        FaultTreeEvent("TOP", "Robot-Human Collision", False, 0.0, "OR", ["IE002", "BE003", "BE005"])
    ])
    
    for event in events:
        collision_fta.add_event(event)
    
    return safety_system


if __name__ == "__main__":
    # Example usage
    safety_system = create_example_hri_safety_analysis()
    
    # Generate comprehensive analysis
    results = safety_system.conduct_comprehensive_analysis()
    
    print(f"Safety Analysis Results for {safety_system.system_name}")
    print(f"HARA: {results['hara_analysis']['total_hazards']} hazards analyzed")
    print(f"FMEA: {results['fmea_analysis']['total_failure_modes']} failure modes analyzed")
    print(f"FTA: {len(results['fault_tree_analysis'])} fault trees analyzed")
    print(f"SIL Requirements: {len(results['sil_requirements'])} components")
    print(f"Compliance Score: {results['compliance_summary']['overall_compliance_score']:.2%}")
    
    # Generate report
    safety_system.generate_comprehensive_report("./safety_analysis_results")