#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Information Architecture Test report generator.

This module provides functions to convert Information Architecture Test results
into comprehensive reports with detailed explanations and visualizations.
"""

import os
import json
import datetime
import logging
from typing import Dict, List, Any
import numpy as np
import markdown
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CMB-Reporter-InfoArch")

# Mathematical constants for reference
PHI = 1.618033988749895  # Golden ratio
PI = np.pi
E = np.e
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
LN2 = np.log(2)

CONSTANTS_MAP = {
    "PHI": "φ (Golden Ratio)",
    "PI": "π (Pi)",
    "E": "e (Euler's Number)",
    "SQRT2": "√2 (Square Root of 2)",
    "SQRT3": "√3 (Square Root of 3)",
    "LN2": "ln(2) (Natural Log of 2)"
}

def generate_info_architecture_report(results: Dict) -> Dict[str, str]:
    """
    Generate a comprehensive report for Information Architecture Test results.
    
    Parameters:
    - results: Dictionary containing test results
    
    Returns:
    - Dictionary with 'html' and 'markdown' versions of the report
    """
    logger.info("Generating Information Architecture Test report")
    
    # Extract key components from results
    constants_results = results.get('constants_results', {})
    layer_results = results.get('layer_results', {})
    interlayer_results = results.get('interlayer_results', {})
    dataset_name = results.get('dataset_name', 'Unknown Dataset')
    
    # Generate Markdown report
    md_content = f"# Information Architecture Test Report: {dataset_name}\n\n"
    md_content += f"_Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
    
    # Executive Summary
    md_content += "## Executive Summary\n\n"
    
    # Find significant constants
    significant_constants = []
    for const, data in constants_results.items():
        if data.get('significant', False):
            significant_constants.append(const)
    
    if significant_constants:
        significant_list = ', '.join([CONSTANTS_MAP.get(c, c) for c in significant_constants])
        md_content += f"The analysis found statistically significant organization for the following mathematical constants: **{significant_list}**.\n\n"
    else:
        md_content += "The analysis did not find statistically significant organization for any of the tested mathematical constants.\n\n"
    
    # Check for dominant constants in layers
    dominant_layer_constants = {}
    for layer, data in layer_results.items():
        dominant = data.get('dominant_constant')
        if dominant and dominant.lower() != 'none':
            if dominant not in dominant_layer_constants:
                dominant_layer_constants[dominant] = []
            dominant_layer_constants[dominant].append(layer)
    
    if dominant_layer_constants:
        md_content += "### Key Scale Findings:\n\n"
        for const, layers in dominant_layer_constants.items():
            layers_str = ', '.join(layers)
            md_content += f"- **{CONSTANTS_MAP.get(const.upper(), const)}** is the dominant organizational principle in scales: {layers_str}\n"
        md_content += "\n"
    
    # Mathematical Constants Analysis
    md_content += "## Mathematical Constants Analysis\n\n"
    
    # Create a table for constants results
    md_content += "| Constant | Architecture Score | Mean Surrogate | Std Dev | Z-score | P-value | Significant | Phi-optimality |\n"
    md_content += "|----------|-------------------|---------------|---------|---------|---------|-------------|----------------|\n"
    
    for const, data in sorted(constants_results.items(), key=lambda x: x[1].get('p_value', 1.0)):
        # Extract values with fallbacks to handle missing data
        architecture_score = data.get('architecture_score', 0)
        mean_surrogate = data.get('mean_surrogate', 0)
        std_dev = data.get('std_dev', 0)
        z_score = data.get('z_score', 0)
        p_value = data.get('p_value', 1.0)
        significant = data.get('significant', False)
        phi_optimality = data.get('phi_optimality', 0)
        
        # Format values
        sig_str = "Yes" if significant else "No"
        
        # Add row to table
        md_content += f"| {CONSTANTS_MAP.get(const, const)} | {architecture_score:.6f} | {mean_surrogate:.6f} | {std_dev:.6f} | {z_score:.4f} | {p_value:.6f} | {sig_str} | {phi_optimality:.6f} |\n"
    
    md_content += "\n"
    
    # Add interpretation
    md_content += "### Interpretation\n\n"
    
    if significant_constants:
        md_content += f"The {dataset_name} data shows mathematical organization with "
        if len(significant_constants) == 1:
            md_content += f"{CONSTANTS_MAP.get(significant_constants[0], significant_constants[0])} showing statistical significance"
        else:
            constants_list = [CONSTANTS_MAP.get(c, c) for c in significant_constants[:-1]]
            constants_str = ", ".join(constants_list)
            last_constant = CONSTANTS_MAP.get(significant_constants[-1], significant_constants[-1])
            md_content += f"{constants_str}, and {last_constant} all showing statistical significance"
        
        # Find the constant with the highest Z-score
        highest_z = 0
        highest_const = None
        for const, data in constants_results.items():
            z_score = data.get('z_score', 0)
            if z_score > highest_z:
                highest_z = z_score
                highest_const = const
        
        if highest_const:
            md_content += f", with {CONSTANTS_MAP.get(highest_const, highest_const)} showing the strongest signal (Z-score: {highest_z:.4f}).\n\n"
        else:
            md_content += ".\n\n"
            
        # Commentary on PHI if it's significant
        if "PHI" in significant_constants:
            phi_data = constants_results.get("PHI", {})
            md_content += f"The Golden Ratio (φ) shows significance with a p-value of {phi_data.get('p_value', 'N/A')}, supporting the hypothesis that this mathematical constant plays a role in the organization of cosmic microwave background radiation.\n\n"
    else:
        md_content += f"The {dataset_name} data does not show statistically significant organization for any of the tested mathematical constants. This may indicate that the structure in this dataset follows different organizational principles or that the signal is too subtle to detect with the current methodology.\n\n"
    
    # Layer Specialization Analysis
    md_content += "## Layer Specialization Analysis\n\n"
    
    # Create a table for layer results
    md_content += "| Scale | Range | Dominant Constant | Specialization Ratio |\n"
    md_content += "|-------|------------|-------------------|----------------------|\n"
    
    for layer, data in sorted(layer_results.items(), key=lambda x: x[0]):
        # Extract values with fallbacks
        scale_range = data.get('range', 'Unknown')
        dominant_constant = data.get('dominant_constant', 'None')
        specialization_ratio = data.get('specialization_ratio', 1.0)
        
        # Format values
        if dominant_constant.lower() == 'none':
            dominant_constant_formatted = "None"
        else:
            dominant_constant_formatted = dominant_constant.lower()
        
        # Add row to table
        md_content += f"| {layer} | {scale_range} | {dominant_constant_formatted} | {specialization_ratio:.4f} |\n"
    
    md_content += "\n"
    
    # Add interpretation
    md_content += "### Interpretation\n\n"
    
    # Find dominant constants and their specialization ratios
    dominant_scales = {}
    for layer, data in layer_results.items():
        dominant = data.get('dominant_constant', '').lower()
        if dominant and dominant != 'none':
            ratio = data.get('specialization_ratio', 1.0)
            scale_range = data.get('range', 'Unknown')
            if dominant not in dominant_scales:
                dominant_scales[dominant] = []
            dominant_scales[dominant].append((layer, scale_range, ratio))
    
    if dominant_scales:
        for const, scales in dominant_scales.items():
            scales_info = []
            for layer, scale_range, ratio in scales:
                scales_info.append(f"{layer} ({scale_range}, ratio: {ratio:.4f})")
            
            scales_str = ", ".join(scales_info)
            md_content += f"- **{const}** is the dominant organizational principle in scales: {scales_str}\n"
        
        md_content += "\n"
        
        # Special commentary for sqrt2 if it's a dominant constant
        if 'sqrt2' in dominant_scales:
            sqrt2_scales = dominant_scales['sqrt2']
            # Check if scale containing multipole 55 is dominated by sqrt2
            scale_55_dominated = False
            for layer, scale_range, ratio in sqrt2_scales:
                range_parts = scale_range.strip('()').split(', ')
                if len(range_parts) == 2:
                    try:
                        lower = int(range_parts[0])
                        upper = int(range_parts[1])
                        if lower <= 55 <= upper:
                            scale_55_dominated = True
                            break
                    except ValueError:
                        pass
            
            if scale_55_dominated:
                md_content += "Of particular interest is the strong √2 organization at scales that include multipole 55, confirming previous findings about the special significance of this scale in cosmic microwave background radiation.\n\n"
    else:
        md_content += "None of the scale ranges show a clear dominant mathematical constant organization, suggesting more complex or different organizational principles may be at work.\n\n"
    
    # Interlayer Connection Analysis
    md_content += "## Interlayer Connection Analysis\n\n"
    
    # Create a table for interlayer results
    md_content += "| Connection | Scale Ranges | Dominant Constant | Specialization Ratio |\n"
    md_content += "|------------|--------------|-------------------|----------------------|\n"
    
    for connection, data in sorted(interlayer_results.items(), key=lambda x: x[0]):
        # Extract values with fallbacks
        scale_ranges = data.get('ranges', 'Unknown')
        dominant_constant = data.get('dominant_constant', 'None')
        specialization_ratio = data.get('specialization_ratio', 1.0)
        
        # Format values
        if dominant_constant.lower() == 'none':
            dominant_constant_formatted = "None"
        else:
            dominant_constant_formatted = dominant_constant.lower()
        
        # Add row to table
        md_content += f"| {connection} | {scale_ranges} | {dominant_constant_formatted} | {specialization_ratio:.4f} |\n"
    
    md_content += "\n"
    
    # Add interpretation
    md_content += "### Interpretation\n\n"
    
    # Find dominant constants in interlayer connections
    dominant_connections = {}
    for connection, data in interlayer_results.items():
        dominant = data.get('dominant_constant', '').lower()
        if dominant and dominant != 'none':
            ratio = data.get('specialization_ratio', 1.0)
            scale_ranges = data.get('ranges', 'Unknown')
            if dominant not in dominant_connections:
                dominant_connections[dominant] = []
            dominant_connections[dominant].append((connection, scale_ranges, ratio))
    
    if dominant_connections:
        md_content += "The analysis of connections between different scale ranges reveals interesting organizational patterns:\n\n"
        
        for const, connections in dominant_connections.items():
            # Sort connections by specialization ratio (descending)
            connections.sort(key=lambda x: x[2], reverse=True)
            
            md_content += f"- **{const}** dominates the following connections:\n"
            for connection, scale_ranges, ratio in connections:
                md_content += f"  - {connection} ({scale_ranges}) with a specialization ratio of {ratio:.4f}\n"
            md_content += "\n"
        
        # Special commentary for pi if it's a dominant connection constant
        if 'pi' in dominant_connections:
            md_content += "The prevalence of π as a dominant organizational principle between different scale ranges suggests circular/periodic organizational principles in the transitions between scales, which may indicate wave-like behavior in the cosmic microwave background fluctuations.\n\n"
    else:
        md_content += "None of the interlayer connections show a clear dominant mathematical constant organization, suggesting the transitions between scales may follow more complex or different organizational principles.\n\n"
    
    # Cross-Dataset Comparison (if available)
    if 'comparison' in results:
        md_content += "## Cross-Dataset Comparison\n\n"
        comparison = results['comparison']
        
        for dataset, findings in comparison.items():
            md_content += f"### {dataset}\n\n"
            md_content += findings + "\n\n"
    
    # Conclusions
    md_content += "## Conclusions\n\n"
    
    if significant_constants:
        md_content += f"The Information Architecture Test on {dataset_name} data reveals significant mathematical organization, particularly with "
        
        if len(significant_constants) == 1:
            md_content += f"{CONSTANTS_MAP.get(significant_constants[0], significant_constants[0])}."
        else:
            constants_list = [CONSTANTS_MAP.get(c, c) for c in significant_constants[:-1]]
            constants_str = ", ".join(constants_list)
            last_constant = CONSTANTS_MAP.get(significant_constants[-1], significant_constants[-1])
            md_content += f"{constants_str}, and {last_constant}."
        
        md_content += " This suggests that these mathematical principles may play important roles in the organization of cosmic structure.\n\n"
        
        # Add specific conclusions about scale organization
        if dominant_scales:
            md_content += "The analysis of scale-specific organization reveals clear patterns, with "
            
            if len(dominant_scales) == 1:
                const = list(dominant_scales.keys())[0]
                md_content += f"{const} dominating specific scale ranges."
            else:
                const_list = list(dominant_scales.keys())[:-1]
                const_str = ", ".join(const_list)
                last_const = list(dominant_scales.keys())[-1]
                md_content += f"{const_str}, and {last_const} dominating specific scale ranges."
            
            md_content += " This differentiation across scales indicates a complex hierarchical structure in the cosmic microwave background radiation.\n\n"
    else:
        md_content += f"The Information Architecture Test on {dataset_name} data does not reveal statistically significant mathematical organization at the global level, though specific scales may still show interesting patterns. Further investigation with alternative methodologies or more sensitive tests may be warranted.\n\n"
    
    md_content += "These findings contribute to our understanding of the organizational principles underlying cosmic structure and may have implications for theoretical models of the early universe.\n\n"
    
    # Convert markdown to HTML for the HTML version
    html_content = markdown.markdown(md_content, extensions=['tables'])
    
    return {'html': html_content, 'markdown': md_content}


if __name__ == "__main__":
    # Example usage
    import pickle
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python info_architecture_report.py <results_file> <output_dir>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    report = generate_info_architecture_report(results)
    
    # Save markdown report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_file = os.path.join(output_dir, f"info_architecture_report_{timestamp}.md")
    with open(md_file, 'w') as f:
        f.write(report['markdown'])
    
    print(f"Generated report: {md_file}")
