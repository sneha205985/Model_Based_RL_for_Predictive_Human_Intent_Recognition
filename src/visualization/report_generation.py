"""
Automated Report Generation System.

This module provides comprehensive automated report generation capabilities
including HTML reports, PDF exports, executive summaries, and publication-ready
figure compilation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
from jinja2 import Template, Environment, FileSystemLoader
import base64
from io import BytesIO
import warnings

# PDF generation (optional dependencies)
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    warnings.warn("WeasyPrint not available. PDF generation will be disabled.")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    warnings.warn("ReportLab not available. Advanced PDF features will be disabled.")

from .core_utils import BaseVisualizer, PlotConfig
from .performance_analysis import PerformanceAnalyzer
from .safety_analysis import SafetyAnalyzer
from .bayesian_analysis import BayesianAnalyzer
from .statistical_framework import StatisticalFramework
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReportSection:
    """Container for report section data."""
    
    title: str
    content: str = ""
    figures: List[str] = field(default_factory=list)  # Base64 encoded images
    tables: List[pd.DataFrame] = field(default_factory=list)
    subsections: List['ReportSection'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    title: str = "Model-Based RL Analysis Report"
    author: str = "Claude Code System"
    organization: str = ""
    template: str = "default"
    
    # Content settings
    include_executive_summary: bool = True
    include_methodology: bool = True
    include_statistical_tests: bool = True
    include_visualizations: bool = True
    include_recommendations: bool = True
    
    # Format settings
    output_formats: List[str] = field(default_factory=lambda: ["html", "pdf"])
    figure_format: str = "png"
    figure_dpi: int = 300
    
    # Styling
    color_scheme: str = "default"
    font_family: str = "Arial"
    font_size: int = 12


class ReportType(Enum):
    """Types of reports that can be generated."""
    COMPREHENSIVE = "comprehensive"
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL = "technical"
    COMPARISON = "comparison"
    SAFETY_ANALYSIS = "safety_analysis"
    PERFORMANCE = "performance"
    PUBLICATION = "publication"


class AutomatedReportGenerator:
    """
    Comprehensive automated report generation system.
    
    Provides:
    - Multi-format report generation (HTML, PDF, LaTeX)
    - Template-based customization
    - Automatic figure inclusion
    - Statistical summary generation
    - Executive summary creation
    - Publication-ready outputs
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize automated report generator.
        
        Args:
            config: Report generation configuration
        """
        self.config = config or ReportConfig()
        
        # Initialize analyzers
        self.performance_analyzer = PerformanceAnalyzer()
        self.safety_analyzer = SafetyAnalyzer()
        self.bayesian_analyzer = BayesianAnalyzer()
        self.statistical_framework = StatisticalFramework()
        
        # Template setup
        self.template_env = self._setup_templates()
        
        logger.info("Initialized automated report generator")
    
    def _setup_templates(self) -> Environment:
        """Setup Jinja2 template environment."""
        
        # Create templates directory if it doesn't exist
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        
        # Create default template if it doesn't exist
        default_template_path = template_dir / "default_report.html"
        if not default_template_path.exists():
            self._create_default_template(default_template_path)
        
        # Setup environment
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        return env
    
    def _create_default_template(self, path: Path) -> None:
        """Create default HTML report template."""
        
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <style>
        body {
            font-family: {{ config.font_family }}, sans-serif;
            font-size: {{ config.font_size }}px;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .metadata {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #2196F3;
            border-bottom: 1px solid #2196F3;
            padding-bottom: 5px;
        }
        .section h3 {
            color: #FF9800;
            margin-top: 25px;
        }
        .figure {
            text-align: center;
            margin: 20px 0;
        }
        .figure img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .figure-caption {
            font-style: italic;
            color: #666;
            margin-top: 5px;
        }
        .table-container {
            overflow-x: auto;
            margin: 20px 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .summary-box {
            background-color: #e8f5e8;
            border: 1px solid #4CAF50;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .footer {
            border-top: 1px solid #ddd;
            padding-top: 20px;
            margin-top: 40px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ config.title }}</h1>
        <div class="metadata">
            <p><strong>Generated:</strong> {{ timestamp }}</p>
            <p><strong>Author:</strong> {{ config.author }}</p>
            {% if config.organization %}
            <p><strong>Organization:</strong> {{ config.organization }}</p>
            {% endif %}
        </div>
    </div>

    {% for section in sections %}
    <div class="section">
        <h2>{{ section.title }}</h2>
        {% if section.content %}
        <div>{{ section.content | safe }}</div>
        {% endif %}
        
        {% for figure in section.figures %}
        <div class="figure">
            <img src="data:image/{{ figure.format }};base64,{{ figure.data }}" 
                 alt="{{ figure.caption }}">
            {% if figure.caption %}
            <div class="figure-caption">{{ figure.caption }}</div>
            {% endif %}
        </div>
        {% endfor %}
        
        {% for table in section.tables %}
        <div class="table-container">
            {{ table.html | safe }}
        </div>
        {% endfor %}
        
        {% for subsection in section.subsections %}
        <div class="subsection">
            <h3>{{ subsection.title }}</h3>
            {% if subsection.content %}
            <div>{{ subsection.content | safe }}</div>
            {% endif %}
        </div>
        {% endfor %}
    </div>
    {% endfor %}

    <div class="footer">
        <p>Generated by Claude Code Automated Report System</p>
        <p>{{ timestamp }}</p>
    </div>
</body>
</html>
        """
        
        with open(path, 'w') as f:
            f.write(template_content.strip())
    
    def generate_comprehensive_report(self,
                                    experimental_data: Dict[str, Any],
                                    save_path: str,
                                    report_type: ReportType = ReportType.COMPREHENSIVE) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            experimental_data: Dictionary containing all experimental results
            save_path: Path to save the report
            report_type: Type of report to generate
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating {report_type.value} report...")
        
        # Create report sections
        sections = []
        
        # Executive Summary
        if self.config.include_executive_summary:
            sections.append(self._create_executive_summary(experimental_data))
        
        # Methodology
        if self.config.include_methodology:
            sections.append(self._create_methodology_section(experimental_data))
        
        # Performance Analysis
        if 'performance_data' in experimental_data:
            sections.append(self._create_performance_section(experimental_data['performance_data']))
        
        # Safety Analysis
        if 'safety_data' in experimental_data:
            sections.append(self._create_safety_section(experimental_data['safety_data']))
        
        # Bayesian Analysis
        if 'bayesian_data' in experimental_data:
            sections.append(self._create_bayesian_section(experimental_data['bayesian_data']))
        
        # Statistical Analysis
        if self.config.include_statistical_tests:
            sections.append(self._create_statistical_section(experimental_data))
        
        # Recommendations
        if self.config.include_recommendations:
            sections.append(self._create_recommendations_section(experimental_data))
        
        # Generate report in requested formats
        output_paths = []
        for format_type in self.config.output_formats:
            if format_type == 'html':
                output_path = self._generate_html_report(sections, save_path)
                output_paths.append(output_path)
            elif format_type == 'pdf' and WEASYPRINT_AVAILABLE:
                output_path = self._generate_pdf_report(sections, save_path)
                output_paths.append(output_path)
            elif format_type == 'latex':
                output_path = self._generate_latex_report(sections, save_path)
                output_paths.append(output_path)
        
        logger.info(f"Report generation completed. Generated {len(output_paths)} files.")
        return output_paths[0] if output_paths else ""
    
    def _create_executive_summary(self, experimental_data: Dict[str, Any]) -> ReportSection:
        """Create executive summary section."""
        
        summary_points = []
        
        # Performance summary
        if 'performance_data' in experimental_data:
            perf_data = experimental_data['performance_data']
            if 'success_rates' in perf_data:
                best_method = max(perf_data['success_rates'].keys(),
                                key=lambda x: np.mean(perf_data['success_rates'][x]))
                best_rate = np.mean(perf_data['success_rates'][best_method])
                summary_points.append(
                    f"Best performing method: <strong>{best_method}</strong> "
                    f"with {best_rate:.1%} success rate"
                )
        
        # Safety summary
        if 'safety_data' in experimental_data:
            safety_data = experimental_data['safety_data']
            if 'violations' in safety_data:
                total_violations = len(safety_data['violations'])
                summary_points.append(
                    f"Safety analysis: <strong>{total_violations}</strong> violations detected"
                )
        
        # Statistical significance
        if 'statistical_results' in experimental_data:
            stat_results = experimental_data['statistical_results']
            significant_comparisons = sum(1 for result in stat_results.get('tests', [])
                                        if result.p_value < 0.05)
            summary_points.append(
                f"Statistical analysis: <strong>{significant_comparisons}</strong> "
                f"significant differences found"
            )
        
        content = "<div class='summary-box'>"
        content += "<h3>Key Findings:</h3><ul>"
        for point in summary_points:
            content += f"<li>{point}</li>"
        content += "</ul></div>"
        
        return ReportSection(
            title="Executive Summary",
            content=content
        )
    
    def _create_methodology_section(self, experimental_data: Dict[str, Any]) -> ReportSection:
        """Create methodology section."""
        
        content = """
        <p>This report presents a comprehensive analysis of the Model-Based Reinforcement Learning
        system for Predictive Human Intent Recognition. The analysis includes:</p>
        
        <ul>
            <li><strong>Performance Evaluation:</strong> Success rates, completion times, and learning curves</li>
            <li><strong>Safety Analysis:</strong> Distance monitoring, violation detection, and risk assessment</li>
            <li><strong>Bayesian Analysis:</strong> Uncertainty quantification and posterior distributions</li>
            <li><strong>Statistical Testing:</strong> Hypothesis testing with multiple comparison corrections</li>
        </ul>
        
        <h3>Experimental Setup</h3>
        """
        
        # Add experimental parameters if available
        if 'experimental_setup' in experimental_data:
            setup = experimental_data['experimental_setup']
            content += "<div class='metadata'>"
            for key, value in setup.items():
                content += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
            content += "</div>"
        
        return ReportSection(
            title="Methodology",
            content=content
        )
    
    def _create_performance_section(self, performance_data: Dict[str, Any]) -> ReportSection:
        """Create performance analysis section."""
        
        figures = []
        
        # Generate performance visualizations
        if 'success_rates' in performance_data:
            # Success rate comparison
            fig = self.performance_analyzer.plot_success_rate_comparison(
                performance_data['success_rates'],
                plot_type='bar'
            )
            figures.append(self._figure_to_base64(fig, "Success Rate Comparison"))
            plt.close(fig)
        
        if 'learning_curves' in performance_data:
            # Learning curves
            fig = self.performance_analyzer.plot_learning_curves(
                performance_data['learning_curves']
            )
            figures.append(self._figure_to_base64(fig, "Learning Curves"))
            plt.close(fig)
        
        # Performance summary table
        tables = []
        if 'summary_stats' in performance_data:
            tables.append(pd.DataFrame(performance_data['summary_stats']))
        
        content = """
        <p>Performance analysis evaluates the effectiveness of different approaches
        across multiple metrics including success rates, completion times, and learning efficiency.</p>
        """
        
        return ReportSection(
            title="Performance Analysis",
            content=content,
            figures=figures,
            tables=tables
        )
    
    def _create_safety_section(self, safety_data: Dict[str, Any]) -> ReportSection:
        """Create safety analysis section."""
        
        figures = []
        
        # Generate safety visualizations
        if 'distances' in safety_data and 'timestamps' in safety_data:
            fig = self.safety_analyzer.plot_distance_over_time(
                safety_data['distances'],
                safety_data['timestamps'],
                safety_data.get('safety_threshold', 0.5)
            )
            figures.append(self._figure_to_base64(fig, "Distance to Human Over Time"))
            plt.close(fig)
        
        if 'violations' in safety_data:
            fig = self.safety_analyzer.plot_safety_violations_analysis(
                safety_data['violations']
            )
            figures.append(self._figure_to_base64(fig, "Safety Violations Analysis"))
            plt.close(fig)
        
        content = """
        <p>Safety analysis monitors human-robot interactions to ensure safe operation.
        Key metrics include distance monitoring, violation detection, and risk assessment.</p>
        """
        
        # Safety summary
        if 'violations' in safety_data:
            violation_count = len(safety_data['violations'])
            if violation_count > 0:
                content += f"""
                <div class='warning-box'>
                    <strong>Warning:</strong> {violation_count} safety violations detected.
                    Immediate review recommended.
                </div>
                """
            else:
                content += """
                <div class='summary-box'>
                    <strong>Good:</strong> No safety violations detected during operation.
                </div>
                """
        
        return ReportSection(
            title="Safety Analysis",
            content=content,
            figures=figures
        )
    
    def _create_bayesian_section(self, bayesian_data: Dict[str, Any]) -> ReportSection:
        """Create Bayesian analysis section."""
        
        figures = []
        
        # Generate Bayesian visualizations
        if 'posterior_history' in bayesian_data:
            fig = self.bayesian_analyzer.plot_posterior_evolution(
                bayesian_data['posterior_history']
            )
            figures.append(self._figure_to_base64(fig, "Posterior Evolution"))
            plt.close(fig)
        
        if 'uncertainty_data' in bayesian_data:
            fig = self.bayesian_analyzer.plot_uncertainty_decomposition(
                bayesian_data['uncertainty_data']
            )
            figures.append(self._figure_to_base64(fig, "Uncertainty Decomposition"))
            plt.close(fig)
        
        content = """
        <p>Bayesian analysis provides insights into model uncertainty, parameter estimation,
        and the evolution of beliefs over time.</p>
        """
        
        return ReportSection(
            title="Bayesian Analysis",
            content=content,
            figures=figures
        )
    
    def _create_statistical_section(self, experimental_data: Dict[str, Any]) -> ReportSection:
        """Create statistical analysis section."""
        
        # Perform comprehensive statistical analysis
        data_groups = experimental_data.get('comparison_data', {})
        
        if data_groups:
            analysis_results = self.statistical_framework.perform_comprehensive_analysis(
                data_groups
            )
            
            # Generate statistical visualization
            fig = self.statistical_framework.plot_statistical_summary(analysis_results)
            figures = [self._figure_to_base64(fig, "Statistical Analysis Summary")]
            plt.close(fig)
            
            # Create results summary
            content = "<h3>Statistical Test Results</h3>"
            
            for test_result in analysis_results['statistical_tests']:
                significance = "significant" if test_result.p_value < 0.05 else "not significant"
                content += f"""
                <p><strong>{test_result.test_name}:</strong> 
                p-value = {test_result.p_value:.4f} ({significance})</p>
                """
            
            # Effect sizes
            if analysis_results['effect_sizes']:
                content += "<h3>Effect Sizes</h3>"
                for comparison, effect_size in analysis_results['effect_sizes'].items():
                    magnitude = self._interpret_effect_size(abs(effect_size))
                    content += f"""
                    <p><strong>{comparison}:</strong> d = {effect_size:.3f} ({magnitude})</p>
                    """
            
            # Recommendations
            if analysis_results['recommendations']:
                content += "<h3>Statistical Recommendations</h3><ul>"
                for rec in analysis_results['recommendations']:
                    content += f"<li>{rec}</li>"
                content += "</ul>"
        else:
            figures = []
            content = "<p>No comparison data available for statistical analysis.</p>"
        
        return ReportSection(
            title="Statistical Analysis",
            content=content,
            figures=figures
        )
    
    def _create_recommendations_section(self, experimental_data: Dict[str, Any]) -> ReportSection:
        """Create recommendations section."""
        
        recommendations = []
        
        # Performance-based recommendations
        if 'performance_data' in experimental_data:
            perf_data = experimental_data['performance_data']
            if 'success_rates' in perf_data:
                rates = perf_data['success_rates']
                best_method = max(rates.keys(), key=lambda x: np.mean(rates[x]))
                recommendations.append(
                    f"<strong>Performance:</strong> Consider adopting {best_method} "
                    f"as the primary approach based on superior success rates."
                )
        
        # Safety-based recommendations
        if 'safety_data' in experimental_data:
            safety_data = experimental_data['safety_data']
            if 'violations' in safety_data and len(safety_data['violations']) > 0:
                recommendations.append(
                    "<strong>Safety:</strong> Review and strengthen safety constraints "
                    "to reduce violation frequency."
                )
        
        # General recommendations
        recommendations.extend([
            "<strong>Monitoring:</strong> Implement continuous performance monitoring "
            "to detect degradation early.",
            "<strong>Validation:</strong> Regular validation with new data is recommended "
            "to ensure model reliability.",
            "<strong>Documentation:</strong> Maintain detailed logs of system performance "
            "and safety metrics."
        ])
        
        content = "<ul>"
        for rec in recommendations:
            content += f"<li>{rec}</li>"
        content += "</ul>"
        
        return ReportSection(
            title="Recommendations",
            content=content
        )
    
    def _figure_to_base64(self, fig: plt.Figure, caption: str = "") -> Dict[str, str]:
        """Convert matplotlib figure to base64 string."""
        
        buffer = BytesIO()
        fig.savefig(buffer, format=self.config.figure_format, 
                   dpi=self.config.figure_dpi, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            'data': image_base64,
            'format': self.config.figure_format,
            'caption': caption
        }
    
    def _generate_html_report(self, sections: List[ReportSection], base_path: str) -> str:
        """Generate HTML report."""
        
        # Load template
        template = self.template_env.get_template('default_report.html')
        
        # Prepare template data
        template_data = {
            'config': self.config,
            'sections': sections,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Render HTML
        html_content = template.render(**template_data)
        
        # Save HTML file
        html_path = Path(base_path).with_suffix('.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_path}")
        return str(html_path)
    
    def _generate_pdf_report(self, sections: List[ReportSection], base_path: str) -> str:
        """Generate PDF report using WeasyPrint."""
        
        if not WEASYPRINT_AVAILABLE:
            logger.warning("WeasyPrint not available. Skipping PDF generation.")
            return ""
        
        # First generate HTML
        html_path = self._generate_html_report(sections, base_path)
        
        # Convert to PDF
        pdf_path = Path(base_path).with_suffix('.pdf')
        
        try:
            HTML(filename=html_path).write_pdf(pdf_path)
            logger.info(f"PDF report saved to {pdf_path}")
            return str(pdf_path)
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return ""
    
    def _generate_latex_report(self, sections: List[ReportSection], base_path: str) -> str:
        """Generate LaTeX report."""
        
        latex_content = self._create_latex_document(sections)
        
        latex_path = Path(base_path).with_suffix('.tex')
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX report saved to {latex_path}")
        return str(latex_path)
    
    def _create_latex_document(self, sections: List[ReportSection]) -> str:
        """Create LaTeX document content."""
        
        latex_content = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{margin=1in}

\title{""" + self.config.title + r"""}
\author{""" + self.config.author + r"""}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

"""
        
        for section in sections:
            latex_content += f"\\section{{{section.title}}}\n"
            
            # Convert HTML content to LaTeX (simplified)
            content = section.content
            content = content.replace('<p>', '').replace('</p>', '\n\n')
            content = content.replace('<strong>', '\\textbf{').replace('</strong>', '}')
            content = content.replace('<ul>', '\\begin{itemize}').replace('</ul>', '\\end{itemize}')
            content = content.replace('<li>', '\\item ').replace('</li>', '')
            
            latex_content += content + "\n\n"
        
        latex_content += r"\end{document}"
        
        return latex_content
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def create_publication_figures(self,
                                 experimental_data: Dict[str, Any],
                                 output_dir: str) -> List[str]:
        """
        Create publication-ready figures.
        
        Args:
            experimental_data: Experimental data dictionary
            output_dir: Output directory for figures
            
        Returns:
            List of generated figure paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        figure_paths = []
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 300
        })
        
        # Generate key figures
        if 'performance_data' in experimental_data:
            # Performance comparison figure
            fig = self.performance_analyzer.plot_success_rate_comparison(
                experimental_data['performance_data']['success_rates'],
                plot_type='bar'
            )
            fig_path = output_path / "performance_comparison.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight', format='png')
            figure_paths.append(str(fig_path))
            plt.close(fig)
        
        if 'safety_data' in experimental_data:
            # Safety analysis figure
            fig = self.safety_analyzer.plot_distance_over_time(
                experimental_data['safety_data']['distances'],
                experimental_data['safety_data']['timestamps'],
                experimental_data['safety_data'].get('safety_threshold', 0.5)
            )
            fig_path = output_path / "safety_analysis.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight', format='png')
            figure_paths.append(str(fig_path))
            plt.close(fig)
        
        logger.info(f"Generated {len(figure_paths)} publication figures")
        return figure_paths


logger.info("Automated report generation system loaded successfully")