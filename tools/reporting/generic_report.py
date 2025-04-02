#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic report generator for any CMB analysis test results.

This module provides a fallback for generating reports from test types
that don't have specialized report generators.
"""

import os
import json
import datetime
import logging
from typing import Dict, Any
import markdown

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CMB-Reporter-Generic")

def generate_generic_report(results: Dict) -> Dict[str, str]:
    """
    Generate a generic report for any CMB analysis test results.
    
    Parameters:
    - results: Dictionary containing test results
    
    Returns:
    - Dictionary with 'html' and 'markdown' versions of the report
    """
    logger.info("Generating generic report for unidentified test type")
    
    # Try to extract dataset name or use default
    dataset_name = results.get('dataset_name', 'Unknown Dataset')
    test_name = results.get('test_name', 'CMB Analysis Test')
    
    # Generate Markdown report
    md_content = f"# {test_name} Report: {dataset_name}\n\n"
    md_content += f"_Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
    
    # Executive Summary
    md_content += "## Executive Summary\n\n"
    md_content += "This report presents the results of a Cosmic Microwave Background (CMB) analysis test. The specialized report generator for this test type was not available, so this is a generic representation of the results.\n\n"
    
    # Result Summary
    md_content += "## Result Summary\n\n"
    
    # Try to find key metrics and add them to the report
    key_metrics = {}
    for key, value in results.items():
        # Skip complex nested structures and focus on simple metrics
        if isinstance(value, (int, float, str, bool)) and key not in ('dataset_name', 'test_name'):
            key_metrics[key] = value
    
    if key_metrics:
        md_content += "### Key Metrics\n\n"
        for key, value in sorted(key_metrics.items()):
            # Format the key for better readability
            formatted_key = key.replace('_', ' ').title()
            
            # Format the value based on its type
            if isinstance(value, float):
                # Format floats with appropriate precision
                if abs(value) < 0.001 or abs(value) >= 1000:
                    formatted_value = f"{value:.6e}"
                else:
                    formatted_value = f"{value:.6f}"
            elif isinstance(value, bool):
                # Format booleans as Yes/No
                formatted_value = "Yes" if value else "No"
            else:
                # Use the value as is for other types
                formatted_value = str(value)
            
            md_content += f"- **{formatted_key}**: {formatted_value}\n"
        
        md_content += "\n"
    else:
        md_content += "No simple metrics were found in the results data.\n\n"
    
    # Try to detect nested structure parts to report
    nested_parts = []
    for key, value in results.items():
        if isinstance(value, dict) and key not in ('dataset_name', 'test_name'):
            nested_parts.append(key)
        elif isinstance(value, list) and value and not isinstance(value[0], (int, float, str, bool)):
            nested_parts.append(key)
    
    if nested_parts:
        md_content += "### Data Structure\n\n"
        md_content += "The results contain the following complex data structures:\n\n"
        
        for part in nested_parts:
            # Format the part name for better readability
            formatted_part = part.replace('_', ' ').title()
            md_content += f"- **{formatted_part}**\n"
        
        md_content += "\n"
        md_content += "For a more detailed analysis, please implement a specialized report generator for this test type.\n\n"
    
    # Raw Data
    md_content += "## Raw Results Data\n\n"
    md_content += "Below is a simplified representation of the raw results data:\n\n"
    
    # Try to create a simplified representation of the results
    md_content += "```json\n"
    simplified_results = {}
    
    for key, value in results.items():
        if isinstance(value, (int, float, str, bool)):
            simplified_results[key] = value
        elif isinstance(value, list):
            if value and isinstance(value[0], (int, float, str, bool)):
                # For simple lists, include first few and last few elements
                if len(value) > 10:
                    simplified_results[key] = value[:3] + ["..."] + value[-3:]
                else:
                    simplified_results[key] = value
            else:
                # For complex lists, just indicate the length
                simplified_results[key] = f"[{len(value)} complex items]"
        elif isinstance(value, dict):
            # For dictionaries, include the keys
            simplified_results[key] = f"{{Dict with keys: {', '.join(list(value.keys())[:5])}}}"
            if len(value) > 5:
                simplified_results[key] += ", ..."
        else:
            # For other types, just indicate the type
            simplified_results[key] = f"[{type(value).__name__}]"
    
    md_content += json.dumps(simplified_results, indent=2)
    md_content += "\n```\n\n"
    
    # Recommendations
    md_content += "## Recommendations\n\n"
    md_content += "For a more detailed and specialized report, consider implementing a custom report generator for this specific test type. The implementation would be similar to the existing report generators in the `reporting` module, but tailored to the structure and significance of this test's results.\n\n"
    
    # Convert markdown to HTML for the HTML version
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    return {'html': html_content, 'markdown': md_content}


if __name__ == "__main__":
    # Example usage
    import pickle
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python generic_report.py <results_file> <output_dir>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    report = generate_generic_report(results)
    
    # Save markdown report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_file = os.path.join(output_dir, f"generic_report_{timestamp}.md")
    with open(md_file, 'w') as f:
        f.write(report['markdown'])
    
    print(f"Generated report: {md_file}")
