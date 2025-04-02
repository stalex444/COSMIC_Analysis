#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scale Transition Test report generator.

This module provides functions to convert Scale Transition Test results
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
logger = logging.getLogger("CMB-Reporter-ScaleTransition")

def generate_scale_transition_report(results: Dict) -> Dict[str, str]:
    """
    Generate a comprehensive report for Scale Transition Test results.
    
    Parameters:
    - results: Dictionary containing test results
    
    Returns:
    - Dictionary with 'html' and 'markdown' versions of the report
    """
    logger.info("Generating Scale Transition Test report")
    
    # Extract key components from results
    transition_points = results.get('transition_points', [])
    golden_ratio_score = results.get('golden_ratio_score', 0)
    p_value = results.get('p_value', 1.0)
    significant = results.get('significant', False)
    best_n_clusters = results.get('best_n_clusters', 0)
    dataset_name = results.get('dataset_name', 'Unknown Dataset')
    window_size = results.get('window_size', 10)
    complexity_values = results.get('complexity_values', [])
    n_simulations = results.get('n_simulations', 0)
    
    # Generate Markdown report
    md_content = f"# Scale Transition Test Report: {dataset_name}\n\n"
    md_content += f"_Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
    
    # Executive Summary
    md_content += "## Executive Summary\n\n"
    
    if transition_points:
        md_content += f"The Scale Transition Test identified **{len(transition_points)} transition points** in the {dataset_name} data where the organizational principles of the cosmic microwave background radiation change significantly.\n\n"
        
        if significant:
            md_content += f"The alignment of these transition points with the Golden Ratio is **statistically significant** (p-value: {p_value:.6f}), suggesting that the Golden Ratio may play a role in organizing scale transitions in the cosmic microwave background.\n\n"
        else:
            md_content += f"The alignment of these transition points with the Golden Ratio is not statistically significant (p-value: {p_value:.6f}).\n\n"
    else:
        md_content += f"The Scale Transition Test did not identify any clear transition points in the {dataset_name} data.\n\n"
    
    # Test Parameters
    md_content += "## Test Parameters\n\n"
    md_content += f"- **Dataset**: {dataset_name}\n"
    md_content += f"- **Window Size**: {window_size}\n"
    md_content += f"- **Number of Simulations**: {n_simulations}\n"
    md_content += f"- **Best Number of Clusters**: {best_n_clusters}\n\n"
    
    # Transition Points Analysis
    md_content += "## Transition Points Analysis\n\n"
    
    if transition_points:
        md_content += "### Identified Transition Points\n\n"
        md_content += "| Index | Multipole (ℓ) |\n"
        md_content += "|-------|-------------|\n"
        
        for i, point in enumerate(sorted(transition_points)):
            md_content += f"| {i+1} | {point} |\n"
        
        md_content += "\n"
        
        # Group transition points by scale ranges
        ranges = [(2, 10), (11, 50), (51, 200), (201, 500), (501, 1000), (1001, 2500)]
        range_counts = {f"{r[0]}-{r[1]}": 0 for r in ranges}
        
        for point in transition_points:
            for r in ranges:
                if r[0] <= point <= r[1]:
                    range_counts[f"{r[0]}-{r[1]}"] += 1
                    break
        
        md_content += "### Distribution by Scale Range\n\n"
        md_content += "| Scale Range | Number of Transitions | Percentage |\n"
        md_content += "|-------------|------------------------|------------|\n"
        
        for r, count in range_counts.items():
            percentage = (count / len(transition_points)) * 100 if transition_points else 0
            md_content += f"| {r} | {count} | {percentage:.2f}% |\n"
        
        md_content += "\n"
    else:
        md_content += "No transition points were identified in the data.\n\n"
    
    # Golden Ratio Alignment
    md_content += "## Golden Ratio Alignment\n\n"
    md_content += f"- **Golden Ratio Alignment Score**: {golden_ratio_score:.6f}\n"
    md_content += f"- **P-value**: {p_value:.6f}\n"
    md_content += f"- **Statistically Significant**: {'Yes' if significant else 'No'}\n\n"
    
    # Add interpretation
    md_content += "### Interpretation\n\n"
    
    if significant:
        md_content += f"The Golden Ratio alignment score of {golden_ratio_score:.6f} with a p-value of {p_value:.6f} indicates that the transition points in the {dataset_name} data are organized in a way that significantly aligns with the Golden Ratio (φ = 1.618...). This suggests that the Golden Ratio may play a fundamental role in organizing the scale structure of cosmic microwave background radiation.\n\n"
        
        md_content += "This finding supports the hypothesis that there are specific mathematical principles governing the transitions between different organizational regimes in the cosmic microwave background, potentially reflecting fundamental properties of the early universe.\n\n"
    else:
        md_content += f"The Golden Ratio alignment score of {golden_ratio_score:.6f} with a p-value of {p_value:.6f} indicates that while there may be some alignment with the Golden Ratio, it does not reach statistical significance based on the Monte Carlo simulations.\n\n"
        
        md_content += "This suggests that the transitions between different organizational regimes in the cosmic microwave background may follow different mathematical principles or more complex patterns that cannot be fully captured by simple Golden Ratio alignment.\n\n"
    
    # Complexity Analysis
    if complexity_values:
        md_content += "## Local Complexity Analysis\n\n"
        md_content += "The Scale Transition Test measures local complexity across different scales to identify points where the organizational principles change significantly. Below is a summary of the complexity distribution:\n\n"
        
        # Calculate complexity statistics
        mean_complexity = np.mean(complexity_values)
        std_complexity = np.std(complexity_values)
        min_complexity = np.min(complexity_values)
        max_complexity = np.max(complexity_values)
        
        md_content += f"- **Mean Complexity**: {mean_complexity:.6f}\n"
        md_content += f"- **Standard Deviation**: {std_complexity:.6f}\n"
        md_content += f"- **Minimum Complexity**: {min_complexity:.6f}\n"
        md_content += f"- **Maximum Complexity**: {max_complexity:.6f}\n\n"
        
        # Add interpretation about complexity
        md_content += "### Interpretation\n\n"
        md_content += "The complexity values measure how organized or random the power spectrum is at each scale. Higher complexity values indicate regions of the power spectrum that show more intricate patterns, potentially indicating interesting physical phenomena.\n\n"
        
        md_content += "Transition points occur where there are significant changes in complexity, suggesting a shift from one organizational regime to another. These transitions may correspond to important physical scales in the early universe, such as the horizon scale at recombination or the scale of baryon acoustic oscillations.\n\n"
    
    # Cluster Analysis
    md_content += "## Cluster Analysis\n\n"
    md_content += f"The optimal number of clusters identified in the data is **{best_n_clusters}**. This suggests that the cosmic microwave background has {best_n_clusters} distinct organizational regimes across the measured scales.\n\n"
    
    md_content += "### Interpretation\n\n"
    md_content += f"The identification of {best_n_clusters} distinct clusters suggests that the cosmic microwave background radiation exhibits different organizational principles at different scales. These clusters may correspond to different physical processes that dominated at different scales in the early universe.\n\n"
    
    md_content += "The transition points between these clusters represent scales where the organizational principles change, potentially indicating fundamental scales in cosmological models.\n\n"
    
    # Monte Carlo Simulation Results
    md_content += "## Monte Carlo Simulation Results\n\n"
    md_content += f"The statistical significance of the Golden Ratio alignment was assessed using {n_simulations} Monte Carlo simulations, which generated randomized data with similar statistical properties to the original data.\n\n"
    
    if significant:
        md_content += f"The p-value of {p_value:.6f} indicates that only approximately {p_value*100:.4f}% of random simulations showed Golden Ratio alignment as strong as or stronger than the observed data. This suggests that the alignment is unlikely to be due to chance alone.\n\n"
    else:
        md_content += f"The p-value of {p_value:.6f} indicates that approximately {p_value*100:.4f}% of random simulations showed Golden Ratio alignment as strong as or stronger than the observed data. This suggests that the observed alignment could be explained by chance.\n\n"
    
    # Conclusions
    md_content += "## Conclusions\n\n"
    
    if transition_points:
        md_content += f"The Scale Transition Test identified {len(transition_points)} transition points in the {dataset_name} data where the organizational principles change significantly. "
        
        if significant:
            md_content += "These transition points show statistically significant alignment with the Golden Ratio, suggesting that this mathematical constant may play a fundamental role in organizing the scale structure of the cosmic microwave background radiation.\n\n"
        else:
            md_content += "While these transition points do not show statistically significant alignment with the Golden Ratio, they still represent important scales where the organizational principles of the cosmic microwave background change.\n\n"
        
        md_content += f"The identification of {best_n_clusters} distinct organizational regimes suggests a complex hierarchical structure in the cosmic microwave background, potentially reflecting different physical processes that dominated at different scales in the early universe.\n\n"
    else:
        md_content += f"The Scale Transition Test did not identify clear transition points in the {dataset_name} data, suggesting that the organizational principles may be more continuous across scales or that the methodology needs further refinement to detect subtle transitions.\n\n"
    
    md_content += "These findings contribute to our understanding of the scale structure of the cosmic microwave background radiation and may have implications for cosmological models of the early universe.\n\n"
    
    # Convert markdown to HTML for the HTML version
    html_content = markdown.markdown(md_content, extensions=['tables'])
    
    return {'html': html_content, 'markdown': md_content}


if __name__ == "__main__":
    # Example usage
    import pickle
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python scale_transition_report.py <results_file> <output_dir>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    report = generate_scale_transition_report(results)
    
    # Save markdown report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_file = os.path.join(output_dir, f"scale_transition_report_{timestamp}.md")
    with open(md_file, 'w') as f:
        f.write(report['markdown'])
    
    print(f"Generated report: {md_file}")
