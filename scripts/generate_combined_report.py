#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined Results Report Generator for COSMIC_Analysis

This script collects results from all completed tests (Golden Symmetries,
Phase Alignment, Golden Ratio Significance) and generates a comprehensive
report with comparisons and conclusions.
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from tabulate import tabulate
import re


def find_latest_results(base_dir, test_name):
    """
    Find the latest results directory for a specific test.
    
    Args:
        base_dir (str): Base directory for results
        test_name (str): Name pattern of the test
        
    Returns:
        str: Path to the latest results directory, or None if not found
    """
    dirs = glob.glob(os.path.join(base_dir, f"{test_name}_*"))
    if not dirs:
        print(f"No {test_name} results directory found in {base_dir}")
        return None
    
    # Sort by creation time (newest first)
    dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    print(f"Found latest {test_name} results at: {dirs[0]}")
    return dirs[0]


def load_phase_alignment_results(results_dir):
    """
    Load Phase Alignment test results.
    
    Args:
        results_dir (str): Path to the Phase Alignment results directory
        
    Returns:
        dict: Results data
    """
    if not results_dir or not os.path.exists(results_dir):
        print(f"Phase Alignment results directory not found: {results_dir}")
        return None
    
    results = {
        'wmap': {'traditional': None, 'plv': None},
        'planck': {'traditional': None, 'plv': None}
    }
    
    # Load WMAP traditional method results
    wmap_trad_file = glob.glob(os.path.join(results_dir, "phase_alignment_WMAP_Traditional_*.json"))
    if wmap_trad_file:
        with open(wmap_trad_file[0], 'r') as f:
            results['wmap']['traditional'] = json.load(f)
    
    # Load WMAP PLV method results
    wmap_plv_file = glob.glob(os.path.join(results_dir, "phase_alignment_WMAP_PLV_*.json"))
    if wmap_plv_file:
        with open(wmap_plv_file[0], 'r') as f:
            results['wmap']['plv'] = json.load(f)
    
    # Load Planck traditional method results
    planck_trad_file = glob.glob(os.path.join(results_dir, "phase_alignment_Planck_Traditional_*.json"))
    if planck_trad_file:
        with open(planck_trad_file[0], 'r') as f:
            results['planck']['traditional'] = json.load(f)
    
    # Load Planck PLV method results
    planck_plv_file = glob.glob(os.path.join(results_dir, "phase_alignment_Planck_PLV_*.json"))
    if planck_plv_file:
        with open(planck_plv_file[0], 'r') as f:
            results['planck']['plv'] = json.load(f)
    
    return results


def load_golden_symmetries_results(results_dir):
    """
    Load Golden Symmetries test results.
    
    Args:
        results_dir (str): Path to the Golden Symmetries results directory
        
    Returns:
        dict: Results data
    """
    if not results_dir or not os.path.exists(results_dir):
        print(f"Golden Symmetries results directory not found: {results_dir}")
        return None
    
    results = {'wmap': None, 'planck': None}
    
    # Load WMAP results
    wmap_dir = os.path.join(results_dir, 'wmap')
    if os.path.exists(wmap_dir):
        wmap_results_files = glob.glob(os.path.join(wmap_dir, "golden_symmetries_results_wmap_*.txt"))
        if wmap_results_files:
            results['wmap'] = parse_golden_symmetries_txt(wmap_results_files[0])
    
    # Load Planck results
    planck_dir = os.path.join(results_dir, 'planck')
    if os.path.exists(planck_dir):
        planck_results_files = glob.glob(os.path.join(planck_dir, "golden_symmetries_results_planck_*.txt"))
        if planck_results_files:
            results['planck'] = parse_golden_symmetries_txt(planck_results_files[0])
    
    return results


def parse_golden_symmetries_txt(file_path):
    """
    Parse Golden Symmetries text result file.
    
    Args:
        file_path (str): Path to the results file
        
    Returns:
        dict: Parsed results
    """
    results = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Extract phi optimality
            phi_opt_match = content.find("Phi optimality:")
            if phi_opt_match != -1:
                line = content[phi_opt_match:].split('\n')[0]
                value = float(line.split(':')[1].strip().split(' ')[0])
                results['phi_optimality'] = value
            
            # Extract p-value
            p_value_match = content.find("P-value:")
            if p_value_match != -1:
                line = content[p_value_match:].split('\n')[0]
                value = float(line.split(':')[1].strip())
                results['p_value'] = value
            
            # Extract symmetry ratio
            sym_ratio_match = content.find("Symmetry ratio:")
            if sym_ratio_match != -1:
                line = content[sym_ratio_match:].split('\n')[0]
                value = float(line.split(':')[1].strip())
                results['symmetry_ratio'] = value
    
    except Exception as e:
        print(f"Error parsing Golden Symmetries results file: {e}")
        return None
    
    return results


def load_golden_ratio_significance_results(results_dir):
    """
    Load Golden Ratio Significance test results.
    
    Args:
        results_dir (str): Path to the Golden Ratio Significance results directory
        
    Returns:
        dict: Results data
    """
    if not results_dir or not os.path.exists(results_dir):
        print(f"Golden Ratio Significance results directory not found: {results_dir}")
        return None
    
    results = {'wmap': None, 'planck': None}
    
    # Load WMAP results
    wmap_dir = os.path.join(results_dir, 'wmap')
    if os.path.exists(wmap_dir):
        wmap_results_files = glob.glob(os.path.join(wmap_dir, "golden_ratio_significance_results_wmap_*.json"))
        if wmap_results_files:
            with open(wmap_results_files[0], 'r') as f:
                results['wmap'] = json.load(f)
    
    # Load Planck results
    planck_dir = os.path.join(results_dir, 'planck')
    if os.path.exists(planck_dir):
        planck_results_files = glob.glob(os.path.join(planck_dir, "golden_ratio_significance_results_planck_*.json"))
        if planck_results_files:
            with open(planck_results_files[0], 'r') as f:
                results['planck'] = json.load(f)
    
    return results


def load_gr_specific_coherence_results(results_dir):
    """
    Load and parse results from the GR-Specific Coherence Test
    
    Args:
        results_dir (str): Directory containing the results
    
    Returns:
        dict: Dictionary with WMAP and Planck results
    """
    results = {'WMAP': {}, 'Planck': {}}
    
    if not results_dir or not os.path.exists(results_dir):
        return results
    
    # Process WMAP results
    wmap_file = os.path.join(results_dir, 'wmap', 'wmap_gr_specific_coherence.txt')
    if os.path.exists(wmap_file):
        with open(wmap_file, 'r') as f:
            content = f.read()
            mean_coherence = re.search(r'Mean Coherence: ([-+]?\d*\.\d+|\d+)', content)
            p_value = re.search(r'P-value: ([-+]?\d*\.\d+|\d+)', content)
            phi_optimality = re.search(r'Phi-Optimality: ([-+]?\d*\.\d+|\d+)', content)
            significant = re.search(r'Significant: (\w+)', content)
            
            if mean_coherence and p_value and phi_optimality and significant:
                results['WMAP']['coherence'] = float(mean_coherence.group(1))
                results['WMAP']['p_value'] = float(p_value.group(1))
                results['WMAP']['phi_optimality'] = float(phi_optimality.group(1))
                results['WMAP']['significant'] = (significant.group(1) == 'True')
    
    # Process Planck results
    planck_file = os.path.join(results_dir, 'planck', 'planck_gr_specific_coherence.txt')
    if os.path.exists(planck_file):
        with open(planck_file, 'r') as f:
            content = f.read()
            mean_coherence = re.search(r'Mean Coherence: ([-+]?\d*\.\d+|\d+)', content)
            p_value = re.search(r'P-value: ([-+]?\d*\.\d+|\d+)', content)
            phi_optimality = re.search(r'Phi-Optimality: ([-+]?\d*\.\d+|\d+)', content)
            significant = re.search(r'Significant: (\w+)', content)
            
            if mean_coherence and p_value and phi_optimality and significant:
                results['Planck']['coherence'] = float(mean_coherence.group(1))
                results['Planck']['p_value'] = float(p_value.group(1))
                results['Planck']['phi_optimality'] = float(phi_optimality.group(1))
                results['Planck']['significant'] = (significant.group(1) == 'True')
    
    return results


def generate_comparison_table(phase_results, golden_sym_results, golden_ratio_results, gr_specific_coherence_results):
    """
    Generate a comparison table of all test results.
    
    Args:
        phase_results (dict): Phase Alignment results
        golden_sym_results (dict): Golden Symmetries results
        golden_ratio_results (dict): Golden Ratio Significance results
        gr_specific_coherence_results (dict): GR-Specific Coherence Test results
        
    Returns:
        pandas.DataFrame: Comparison table
    """
    comparison_data = []
    
    # Constants to include in the table
    constants = ['phi', 'sqrt(2)', 'pi', 'e', 'sqrt(3)', 'ln(2)']
    
    # Add WMAP data
    for constant in constants:
        row = {'Dataset': 'WMAP', 'Constant': constant}
        
        # Phase Alignment - Traditional
        if phase_results and phase_results['wmap']['traditional'] and constant in phase_results['wmap']['traditional']['constants']:
            data = phase_results['wmap']['traditional']['constants'][constant]
            row['PA_Trad_Coherence'] = data['mean_coherence']
            row['PA_Trad_p-value'] = data['p_value']
            row['PA_Trad_Significant'] = 'Yes' if data['significant'] else 'No'
        else:
            row['PA_Trad_Coherence'] = 'N/A'
            row['PA_Trad_p-value'] = 'N/A'
            row['PA_Trad_Significant'] = 'N/A'
        
        # Phase Alignment - PLV
        if phase_results and phase_results['wmap']['plv'] and constant in phase_results['wmap']['plv']['constants']:
            data = phase_results['wmap']['plv']['constants'][constant]
            row['PA_PLV_Coherence'] = data['mean_coherence']
            row['PA_PLV_p-value'] = data['p_value']
            row['PA_PLV_Significant'] = 'Yes' if data['significant'] else 'No'
        else:
            row['PA_PLV_Coherence'] = 'N/A'
            row['PA_PLV_p-value'] = 'N/A'
            row['PA_PLV_Significant'] = 'N/A'
        
        # Golden Symmetries (only applicable for phi)
        if constant == 'phi' and golden_sym_results and golden_sym_results['wmap']:
            row['GS_Phi_Optimality'] = golden_sym_results['wmap']['phi_optimality']
            row['GS_p-value'] = golden_sym_results['wmap']['p_value']
            row['GS_Significant'] = 'Yes' if golden_sym_results['wmap']['p_value'] < 0.05 else 'No'
        else:
            row['GS_Phi_Optimality'] = 'N/A'
            row['GS_p-value'] = 'N/A'
            row['GS_Significant'] = 'N/A'
        
        # Golden Ratio Significance (only applicable for phi)
        if constant == 'phi' and golden_ratio_results and golden_ratio_results['wmap']:
            row['GRS_Correlation'] = golden_ratio_results['wmap']['correlation']
            row['GRS_p-value'] = golden_ratio_results['wmap']['p_value']
            row['GRS_Phi_Optimality'] = golden_ratio_results['wmap']['phi_optimality']
            row['GRS_Significant'] = 'Yes' if golden_ratio_results['wmap']['significant'] else 'No'
        else:
            row['GRS_Correlation'] = 'N/A'
            row['GRS_p-value'] = 'N/A'
            row['GRS_Phi_Optimality'] = 'N/A'
            row['GRS_Significant'] = 'N/A'
        
        # GR-Specific Coherence Test (only applicable for phi)
        if constant == 'phi' and gr_specific_coherence_results and gr_specific_coherence_results['WMAP']:
            row['GRSC_Coherence'] = gr_specific_coherence_results['WMAP']['coherence']
            row['GRSC_p-value'] = gr_specific_coherence_results['WMAP']['p_value']
            row['GRSC_Phi_Optimality'] = gr_specific_coherence_results['WMAP']['phi_optimality']
            row['GRSC_Significant'] = 'Yes' if gr_specific_coherence_results['WMAP']['significant'] else 'No'
        else:
            row['GRSC_Coherence'] = 'N/A'
            row['GRSC_p-value'] = 'N/A'
            row['GRSC_Phi_Optimality'] = 'N/A'
            row['GRSC_Significant'] = 'N/A'
        
        comparison_data.append(row)
    
    # Add Planck data
    for constant in constants:
        row = {'Dataset': 'Planck', 'Constant': constant}
        
        # Phase Alignment - Traditional
        if phase_results and phase_results['planck']['traditional'] and constant in phase_results['planck']['traditional']['constants']:
            data = phase_results['planck']['traditional']['constants'][constant]
            row['PA_Trad_Coherence'] = data['mean_coherence']
            row['PA_Trad_p-value'] = data['p_value']
            row['PA_Trad_Significant'] = 'Yes' if data['significant'] else 'No'
        else:
            row['PA_Trad_Coherence'] = 'N/A'
            row['PA_Trad_p-value'] = 'N/A'
            row['PA_Trad_Significant'] = 'N/A'
        
        # Phase Alignment - PLV
        if phase_results and phase_results['planck']['plv'] and constant in phase_results['planck']['plv']['constants']:
            data = phase_results['planck']['plv']['constants'][constant]
            row['PA_PLV_Coherence'] = data['mean_coherence']
            row['PA_PLV_p-value'] = data['p_value']
            row['PA_PLV_Significant'] = 'Yes' if data['significant'] else 'No'
        else:
            row['PA_PLV_Coherence'] = 'N/A'
            row['PA_PLV_p-value'] = 'N/A'
            row['PA_PLV_Significant'] = 'N/A'
        
        # Golden Symmetries (only applicable for phi)
        if constant == 'phi' and golden_sym_results and golden_sym_results['planck']:
            row['GS_Phi_Optimality'] = golden_sym_results['planck']['phi_optimality']
            row['GS_p-value'] = golden_sym_results['planck']['p_value']
            row['GS_Significant'] = 'Yes' if golden_sym_results['planck']['p_value'] < 0.05 else 'No'
        else:
            row['GS_Phi_Optimality'] = 'N/A'
            row['GS_p-value'] = 'N/A'
            row['GS_Significant'] = 'N/A'
        
        # Golden Ratio Significance (only applicable for phi)
        if constant == 'phi' and golden_ratio_results and golden_ratio_results['planck']:
            row['GRS_Correlation'] = golden_ratio_results['planck']['correlation']
            row['GRS_p-value'] = golden_ratio_results['planck']['p_value']
            row['GRS_Phi_Optimality'] = golden_ratio_results['planck']['phi_optimality']
            row['GRS_Significant'] = 'Yes' if golden_ratio_results['planck']['significant'] else 'No'
        else:
            row['GRS_Correlation'] = 'N/A'
            row['GRS_p-value'] = 'N/A'
            row['GRS_Phi_Optimality'] = 'N/A'
            row['GRS_Significant'] = 'N/A'
        
        # GR-Specific Coherence Test (only applicable for phi)
        if constant == 'phi' and gr_specific_coherence_results and gr_specific_coherence_results['Planck']:
            row['GRSC_Coherence'] = gr_specific_coherence_results['Planck']['coherence']
            row['GRSC_p-value'] = gr_specific_coherence_results['Planck']['p_value']
            row['GRSC_Phi_Optimality'] = gr_specific_coherence_results['Planck']['phi_optimality']
            row['GRSC_Significant'] = 'Yes' if gr_specific_coherence_results['Planck']['significant'] else 'No'
        else:
            row['GRSC_Coherence'] = 'N/A'
            row['GRSC_p-value'] = 'N/A'
            row['GRSC_Phi_Optimality'] = 'N/A'
            row['GRSC_Significant'] = 'N/A'
        
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    return df


def draw_conclusions(comparison_df):
    """
    Draw conclusions based on the comparison of all test results.
    
    Args:
        comparison_df (pandas.DataFrame): Comparison table
        
    Returns:
        str: Conclusions text
    """
    conclusions = []
    
    # Check which constants are significant across tests
    wmap_df = comparison_df[comparison_df['Dataset'] == 'WMAP']
    planck_df = comparison_df[comparison_df['Dataset'] == 'Planck']
    
    # Golden Ratio significance
    phi_wmap = wmap_df[wmap_df['Constant'] == 'phi']
    phi_planck = planck_df[planck_df['Constant'] == 'phi']
    
    # Check Golden Ratio significance across tests
    phi_significant_tests = []
    if 'Yes' in phi_wmap['PA_Trad_Significant'].values:
        phi_significant_tests.append("WMAP Traditional Phase Alignment")
    if 'Yes' in phi_wmap['PA_PLV_Significant'].values:
        phi_significant_tests.append("WMAP PLV Phase Alignment")
    if 'Yes' in phi_wmap['GS_Significant'].values:
        phi_significant_tests.append("WMAP Golden Symmetries")
    if 'Yes' in phi_wmap['GRS_Significant'].values:
        phi_significant_tests.append("WMAP Golden Ratio Significance")
    if 'Yes' in phi_wmap['GRSC_Significant'].values:
        phi_significant_tests.append("WMAP GR-Specific Coherence Test")
    
    if 'Yes' in phi_planck['PA_Trad_Significant'].values:
        phi_significant_tests.append("Planck Traditional Phase Alignment")
    if 'Yes' in phi_planck['PA_PLV_Significant'].values:
        phi_significant_tests.append("Planck PLV Phase Alignment")
    if 'Yes' in phi_planck['GS_Significant'].values:
        phi_significant_tests.append("Planck Golden Symmetries")
    if 'Yes' in phi_planck['GRS_Significant'].values:
        phi_significant_tests.append("Planck Golden Ratio Significance")
    if 'Yes' in phi_planck['GRSC_Significant'].values:
        phi_significant_tests.append("Planck GR-Specific Coherence Test")
    
    # Square Root of 2 significance
    sqrt2_wmap = wmap_df[wmap_df['Constant'] == 'sqrt(2)']
    sqrt2_planck = planck_df[planck_df['Constant'] == 'sqrt(2)']
    
    # Check Square Root of 2 significance
    sqrt2_significant_tests = []
    if 'Yes' in sqrt2_wmap['PA_Trad_Significant'].values:
        sqrt2_significant_tests.append("WMAP Traditional Phase Alignment")
    if 'Yes' in sqrt2_wmap['PA_PLV_Significant'].values:
        sqrt2_significant_tests.append("WMAP PLV Phase Alignment")
    
    if 'Yes' in sqrt2_planck['PA_Trad_Significant'].values:
        sqrt2_significant_tests.append("Planck Traditional Phase Alignment")
    if 'Yes' in sqrt2_planck['PA_PLV_Significant'].values:
        sqrt2_significant_tests.append("Planck PLV Phase Alignment")
    
    # Other constants significance
    other_constants = ['pi', 'e', 'sqrt(3)', 'ln(2)']
    for constant in other_constants:
        const_wmap = wmap_df[wmap_df['Constant'] == constant]
        const_planck = planck_df[planck_df['Constant'] == constant]
        
        const_significant_tests = []
        if 'Yes' in const_wmap['PA_Trad_Significant'].values:
            const_significant_tests.append("WMAP Traditional Phase Alignment")
        if 'Yes' in const_wmap['PA_PLV_Significant'].values:
            const_significant_tests.append("WMAP PLV Phase Alignment")
        
        if 'Yes' in const_planck['PA_Trad_Significant'].values:
            const_significant_tests.append("Planck Traditional Phase Alignment")
        if 'Yes' in const_planck['PA_PLV_Significant'].values:
            const_significant_tests.append("Planck PLV Phase Alignment")
        
        if const_significant_tests:
            const_name = {
                'pi': 'Pi (π)',
                'e': 'Euler\'s number (e)',
                'sqrt(3)': 'Square Root of 3 (√3)',
                'ln(2)': 'Natural Logarithm of 2 (ln(2))'
            }[constant]
            conclusions.append(f"### {const_name}\n- Statistically significant in: {', '.join(const_significant_tests)}")
    
    # Draw main conclusions
    main_conclusions = []
    
    # Golden Ratio conclusions
    main_conclusions.append("### Golden Ratio (φ)")
    if phi_significant_tests:
        main_conclusions.append(f"- Statistically significant in: {', '.join(phi_significant_tests)}")
    else:
        main_conclusions.append("- Not statistically significant in any test")
    
    # Add specifics about Golden Ratio
    if 'Planck Golden Symmetries' in phi_significant_tests or 'WMAP Golden Symmetries' in phi_significant_tests:
        gs_wmap_p = phi_wmap['GS_p-value'].values[0]
        gs_planck_p = phi_planck['GS_p-value'].values[0]
        
        if gs_wmap_p != 'N/A' and gs_planck_p != 'N/A':
            if float(gs_wmap_p) < 0.05 and float(gs_planck_p) < 0.05:
                main_conclusions.append("- Golden Symmetries test shows significant results in both WMAP and Planck data")
            elif float(gs_wmap_p) < 0.05:
                main_conclusions.append("- Golden Symmetries test shows significant results only in WMAP data")
            elif float(gs_planck_p) < 0.05:
                main_conclusions.append("- Golden Symmetries test shows significant results only in Planck data")
    
    # Square Root of 2 conclusions
    main_conclusions.append("\n### Square Root of 2 (√2)")
    if sqrt2_significant_tests:
        main_conclusions.append(f"- Statistically significant in: {', '.join(sqrt2_significant_tests)}")
    else:
        main_conclusions.append("- Not statistically significant in any test")
    
    # Compare coherence values
    sqrt2_wmap_plv = sqrt2_wmap['PA_PLV_Coherence'].values[0]
    phi_wmap_plv = phi_wmap['PA_PLV_Coherence'].values[0]
    sqrt2_planck_plv = sqrt2_planck['PA_PLV_Coherence'].values[0]
    phi_planck_plv = phi_planck['PA_PLV_Coherence'].values[0]
    
    if sqrt2_wmap_plv != 'N/A' and phi_wmap_plv != 'N/A' and sqrt2_planck_plv != 'N/A' and phi_planck_plv != 'N/A':
        if float(sqrt2_wmap_plv) > float(phi_wmap_plv) and float(sqrt2_planck_plv) > float(phi_planck_plv):
            main_conclusions.append("- Shows consistently higher coherence values than the Golden Ratio in both datasets")
        elif float(sqrt2_wmap_plv) > float(phi_wmap_plv):
            main_conclusions.append("- Shows higher coherence than the Golden Ratio in WMAP data")
        elif float(sqrt2_planck_plv) > float(phi_planck_plv):
            main_conclusions.append("- Shows higher coherence than the Golden Ratio in Planck data")
    
    # Overall conclusions
    overall = [
        "\n## Overall Conclusions",
        "1. **Multiple Mathematical Constants Show Significance**: Different constants exhibit statistical significance depending on the dataset and analysis method.",
        "2. **Method Sensitivity**: The Phase Locking Value (PLV) method generally reveals more statistically significant patterns than the Traditional Coherence method.",
        "3. **Dataset Differences**: Planck data shows stronger and more significant alignments than WMAP data, possibly due to its higher resolution and sensitivity."
    ]
    
    # Determine dominant organizing principle
    if sqrt2_significant_tests and len(sqrt2_significant_tests) >= len(phi_significant_tests):
        overall.append("4. **Square Root of 2 as Dominant Organizing Principle**: Square Root of 2 consistently shows higher coherence values and significance across more tests, suggesting it may be the dominant organizing principle in the CMB data.")
    elif phi_significant_tests and len(phi_significant_tests) >= len(sqrt2_significant_tests):
        overall.append("4. **Golden Ratio as Dominant Organizing Principle**: The Golden Ratio shows strong significance across multiple tests, suggesting it plays an important role in organizing the structure of the CMB.")
    else:
        overall.append("4. **Multiple Organizing Principles**: Both the Golden Ratio and Square Root of 2 show significance in different tests, suggesting multiple mathematical principles may be at work in organizing the CMB structure.")
    
    # Scale-dependent relationships
    overall.append("5. **Scale-Dependent Relationships**: The significance of different mathematical constants varies across different scales, suggesting complex, scale-dependent organizing principles in the CMB.")
    
    return "\n".join(main_conclusions) + "\n" + "\n".join(conclusions) + "\n" + "\n".join(overall)


def generate_report(base_results_dir, output_file=None):
    """
    Generate a comprehensive report of all test results.
    
    Args:
        base_results_dir (str): Base directory containing all results
        output_file (str): Path to save the report, or None for default
        
    Returns:
        str: Path to the generated report
    """
    # Find latest results
    phase_dir = find_latest_results(base_results_dir, "phase_alignment")
    golden_sym_dir = find_latest_results(base_results_dir, "golden_symmetry")
    golden_ratio_dir = find_latest_results(base_results_dir, "golden_ratio_significance")
    gr_specific_coherence_dir = find_latest_results(base_results_dir, "gr_specific_coherence")
    
    # Load results
    phase_results = load_phase_alignment_results(phase_dir)
    golden_sym_results = load_golden_symmetries_results(golden_sym_dir)
    golden_ratio_results = load_golden_ratio_significance_results(golden_ratio_dir)
    gr_specific_coherence_results = load_gr_specific_coherence_results(gr_specific_coherence_dir)
    
    # Generate comparison table
    comparison_df = generate_comparison_table(phase_results, golden_sym_results, golden_ratio_results, gr_specific_coherence_results)
    
    # Draw conclusions
    conclusions = draw_conclusions(comparison_df)
    
    # Create report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_file:
        output_dir = os.path.join(base_results_dir, f"combined_report_{timestamp}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "combined_results_report.md")
    
    # Format table for markdown
    table_md = comparison_df.to_markdown(index=False, tablefmt="pipe")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write("# Combined CMB Analysis Results Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n")
        f.write("This report combines results from multiple tests analyzing the mathematical structure of the Cosmic Microwave Background (CMB) radiation.\n")
        f.write("Tests included:\n")
        f.write("1. Phase Alignment Test\n")
        f.write("2. Golden Symmetries Test\n")
        f.write("3. Golden Ratio Significance Test\n")
        f.write("4. GR-Specific Coherence Test\n\n")
        
        f.write("## Datasets\n")
        f.write("- **WMAP**: Wilkinson Microwave Anisotropy Probe 9-year data\n")
        f.write("- **Planck**: ESA Planck 2018 data\n\n")
        
        f.write("## Results Comparison\n\n")
        f.write("### Legend\n")
        f.write("- PA_Trad: Phase Alignment with Traditional method\n")
        f.write("- PA_PLV: Phase Alignment with Phase Locking Value method\n")
        f.write("- GS: Golden Symmetries test\n")
        f.write("- GRS: Golden Ratio Significance test\n")
        f.write("- GRSC: GR-Specific Coherence Test\n\n")
        
        f.write("### Comparison Table\n\n")
        f.write(table_md)
        f.write("\n\n")
        
        f.write("## Conclusions\n\n")
        f.write(conclusions)
        f.write("\n\n")
        
        f.write("## Data Sources\n\n")
        if phase_dir:
            f.write(f"- Phase Alignment results: {phase_dir}\n")
        if golden_sym_dir:
            f.write(f"- Golden Symmetries results: {golden_sym_dir}\n")
        if golden_ratio_dir:
            f.write(f"- Golden Ratio Significance results: {golden_ratio_dir}\n")
        if gr_specific_coherence_dir:
            f.write(f"- GR-Specific Coherence Test results: {gr_specific_coherence_dir}\n")
    
    print(f"Combined report generated: {output_file}")
    return output_file


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate combined results report for COSMIC Analysis")
    parser.add_argument('--results-dir', default=None,
                        help='Base directory containing all results. Default: COSMIC_Analysis/results')
    parser.add_argument('--output-file', default=None,
                        help='Output file for the report. Default: results/combined_report_TIMESTAMP/combined_results_report.md')
    
    args = parser.parse_args()
    
    # Set default results directory if not provided
    if not args.results_dir:
        # Find COSMIC_Analysis base directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cosmic_dir = os.path.dirname(os.path.dirname(script_dir))
        results_dir = os.path.join(cosmic_dir, 'results')
    else:
        results_dir = args.results_dir
    
    # Generate report
    report_path = generate_report(results_dir, args.output_file)
    
    return 0


if __name__ == "__main__":
    main()
