"""
Setup script for Model-Based RL Human Intent Recognition System

This package provides a research-grade system for human-robot interaction
with formal mathematical validation and production-ready monitoring.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Model-Based RL Human Intent Recognition System"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="model-based-rl-human-intent",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Model-Based RL Human Intent Recognition System with Research-Grade Validation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/research-team/model-based-rl-human-intent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
        ],
        "docker": [
            "docker>=5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "run-ablation-studies=run_ablation_studies:main",
            "run-baseline-comparisons=run_baseline_comparisons:main", 
            "run-performance-benchmarks=run_performance_benchmarks:main",
            "run-system-validation=run_final_system_validation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="reinforcement-learning robotics human-intent gaussian-process model-predictive-control bayesian",
    project_urls={
        "Bug Reports": "https://github.com/research-team/model-based-rl-human-intent/issues",
        "Source": "https://github.com/research-team/model-based-rl-human-intent",
        "Documentation": "https://model-based-rl-human-intent.readthedocs.io/",
    },
)