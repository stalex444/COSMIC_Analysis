#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main report generator for CMB analysis results.

This module provides comprehensive reporting functionality that converts raw test results
into detailed HTML, Markdown and PDF reports with explanations and visualizations.
"""

import os
import json
import datetime
import logging
import markdown
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Any
import pickle
import numpy as np
from jinja2 import Environment, FileSystemLoader
import weasyprint

# Import report generators
from .info_architecture_report import generate_info_architecture_report
from .scale_transition_report import generate_scale_transition_report
from .transfer_entropy_report import generate_transfer_entropy_report
from .meta_coherence_report import generate_meta_coherence_report
from .generic_report import generate_generic_report
from .spectral_report import generate_spectral_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CMB-Reporter")

@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str
    author: str = "Stephanie Alexander"
    include_html: bool = True
    include_markdown: bool = True
    include_pdf: bool = True
    template_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.template_dir is None:
            # Use default templates in the same directory as this script
            self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
            
            # Create templates directory if it doesn't exist
            if not os.path.exists(self.template_dir):
                os.makedirs(self.template_dir)


def load_results(results_file: str) -> Dict:
    """
    Load test results from a file.
    
    Parameters:
    - results_file: Path to the results file (pickle or JSON)
    
    Returns:
    - Dictionary containing test results
    """
    logger.info(f"Loading results from {results_file}")
    
    if results_file.endswith('.pkl'):
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    elif results_file.endswith('.json'):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {results_file}")
    
    return results


def detect_test_type(results: Dict) -> str:
    """
    Automatically detect the type of test from the results dictionary.
    
    Parameters:
    - results: Dictionary containing test results
    
    Returns:
    - Test type string ('info_architecture', 'scale_transition', 'transfer_entropy', etc.)
    """
    # Look for key indicators in the results dictionary
    if 'constants' in results or 'phi_optimality' in results:
        return 'info_architecture'
    elif 'transition_points' in results or 'golden_ratio_score' in results:
        return 'scale_transition'
    elif 'transfer_entropy' in results or 'bootstrap_ci' in results:
        return 'transfer_entropy'
    elif 'meta_coherence' in results:
        return 'meta_coherence'
    elif 'actual_hurst' in results or 'hurst_exponent' in results or 'sim_hursts' in results:
        return 'spectral'
    else:
        logger.warning("Could not automatically detect test type. Defaulting to generic report.")
        return 'generic'


def generate_report(results_file: str, output_dir: str, config: ReportConfig = None) -> Dict[str, str]:
    """
    Generate comprehensive reports from test results.
    
    Parameters:
    - results_file: Path to the results file (pickle or JSON)
    - output_dir: Directory to save the reports
    - config: Optional configuration for report generation
    
    Returns:
    - Dictionary with paths to generated reports
    """
    if config is None:
        config = ReportConfig(title="CMB Analysis Report")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load results
    results = load_results(results_file)
    
    # Detect test type
    test_type = detect_test_type(results)
    logger.info(f"Detected test type: {test_type}")
    
    # Import specific report generator based on test type
    try:
        if test_type == 'info_architecture':
            from .info_architecture_report import generate_info_architecture_report
            content = generate_info_architecture_report(results)
        elif test_type == 'scale_transition':
            from .scale_transition_report import generate_scale_transition_report
            content = generate_scale_transition_report(results)
        elif test_type == 'transfer_entropy':
            from .transfer_entropy_report import generate_transfer_entropy_report
            content = generate_transfer_entropy_report(results)
        elif test_type == 'meta_coherence':
            from .meta_coherence_report import generate_meta_coherence_report
            content = generate_meta_coherence_report(results)
        elif test_type == 'spectral':
            from .spectral_report import generate_spectral_report
            content = generate_spectral_report(results)
        else:
            from .generic_report import generate_generic_report
            content = generate_generic_report(results)
    except ImportError as e:
        logger.error(f"Could not import report generator for {test_type}: {e}")
        # Fall back to generic report
        from .generic_report import generate_generic_report
        content = generate_generic_report(results)
    
    # Get timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = f"{test_type}_report_{timestamp}"
    
    # Generate reports in different formats
    output_files = {}
    
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(config.template_dir))
    
    # Create base template if it doesn't exist
    base_template_path = os.path.join(config.template_dir, 'base.html')
    if not os.path.exists(base_template_path):
        create_default_templates(config.template_dir)
    
    # Render HTML with template
    if config.include_html:
        html_template = env.get_template('base.html')
        html_content = html_template.render(
            title=config.title,
            author=config.author,
            date=datetime.datetime.now().strftime("%Y-%m-%d"),
            content=content,
            test_type=test_type.replace('_', ' ').title()
        )
        
        html_file = os.path.join(output_dir, f"{basename}.html")
        with open(html_file, 'w') as f:
            f.write(html_content)
        output_files['html'] = html_file
        logger.info(f"Generated HTML report: {html_file}")
    
    # Generate Markdown
    if config.include_markdown:
        md_content = content['markdown']
        md_file = os.path.join(output_dir, f"{basename}.md")
        with open(md_file, 'w') as f:
            f.write(md_content)
        output_files['markdown'] = md_file
        logger.info(f"Generated Markdown report: {md_file}")
    
    # Generate PDF from HTML
    if config.include_pdf and config.include_html:
        pdf_file = os.path.join(output_dir, f"{basename}.pdf")
        weasyprint.HTML(string=html_content).write_pdf(pdf_file)
        output_files['pdf'] = pdf_file
        logger.info(f"Generated PDF report: {pdf_file}")
    
    return output_files


def create_default_templates(template_dir: str) -> None:
    """
    Create default templates for report generation.
    
    Parameters:
    - template_dir: Directory to save the templates
    """
    os.makedirs(template_dir, exist_ok=True)
    
    # Create base HTML template
    base_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .header {
            border-bottom: 1px solid #ddd;
            margin-bottom: 30px;
            padding-bottom: 10px;
        }
        .footer {
            margin-top: 50px;
            border-top: 1px solid #ddd;
            padding-top: 10px;
            font-size: 0.9em;
            color: #777;
        }
        .visualization {
            text-align: center;
            margin: 20px 0;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
        }
        .interpretation {
            background-color: #f8f9fa;
            border-left: 4px solid #5bc0de;
            padding: 10px 15px;
            margin: 15px 0;
        }
        .significant {
            color: #28a745;
            font-weight: bold;
        }
        .not-significant {
            color: #dc3545;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Author: {{ author }} | Date: {{ date }}</p>
        <p>Test Type: {{ test_type }}</p>
    </div>
    
    <div class="content">
        {{ content.html|safe }}
    </div>
    
    <div class="footer">
        <p>Generated by CMB Analysis Reporting Module</p>
        <p> {{ date.split('-')[0] }} Stephanie Alexander</p>
    </div>
</body>
</html>"""
    
    with open(os.path.join(template_dir, 'base.html'), 'w') as f:
        f.write(base_html)
    
    logger.info(f"Created default templates in {template_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive reports from CMB analysis results")
    parser.add_argument("results_file", help="Path to the results file (pickle or JSON)")
    parser.add_argument("output_dir", help="Directory to save the reports")
    parser.add_argument("--title", default="CMB Analysis Report", help="Report title")
    parser.add_argument("--author", default="Stephanie Alexander", help="Report author")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--no-markdown", action="store_true", help="Skip Markdown report generation")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF report generation")
    
    args = parser.parse_args()
    
    config = ReportConfig(
        title=args.title,
        author=args.author,
        include_html=not args.no_html,
        include_markdown=not args.no_markdown,
        include_pdf=not args.no_pdf
    )
    
    generate_report(args.results_file, args.output_dir, config)
