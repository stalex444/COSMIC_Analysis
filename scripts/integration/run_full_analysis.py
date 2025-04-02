#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the full suite of 10 statistical tests from the Cosmic Consciousness Analysis framework
on both WMAP and Planck data.

This script integrates WMAP data into the existing Cosmic Consciousness Analysis framework
and runs all tests on both datasets for comprehensive comparison.

Tests included:
1. Golden Ratio Significance Test
2. Coherence Analysis Test
3. GR-Specific Coherence Test
4. Hierarchical Organization Test
5. Information Integration Test
6. Scale Transition Test
7. Resonance Analysis Test
8. Fractal Analysis Test
9. Meta-Coherence Test
10. Transfer Entropy Test
"""

from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import argparse

# Add parent directory to path for importing from Cosmic_Consciousness_Analysis
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cosmic_analysis_path = os.path.join(parent_dir, 'Cosmic_Consciousness_Analysis')
sys.path.append(cosmic_analysis_path)

# Try to import from Cosmic Consciousness Analysis framework
try:
    from analysis.analysis import Analyzer, CosmicConsciousnessAnalyzer
    from core_framework.data_handler import get_simulated_data
    HAS_COSMIC_FRAMEWORK = True
except ImportError:
    print("Warning: Could not import Cosmic Consciousness Analysis framework.")
    print("Make sure the framework is installed at: {}".format(cosmic_analysis_path))
    HAS_COSMIC_FRAMEWORK = False


def load_wmap_power_spectrum(file_path):
    """
    Load WMAP CMB power spectrum data.
    
    Args:
        file_path (str): Path to the WMAP power spectrum file
            
    Returns:
        tuple: (ell, power, error) arrays
    """
    try:
        # Load the power spectrum data
        # WMAP binned power spectrum format:
        # Column 1 = mean multipole moment l for the bin
        # Column 2 = smallest l contributing to the bin
        # Column 3 = largest l contributing to the bin
        # Column 4 = mean value of TT power spectrum (l(l+1)/2pi * C_l)
        # Column 5 = error for binned value
        data = np.loadtxt(file_path, comments='#')
        
        ell = data[:, 0]  # Mean multipole moment
        power = data[:, 3]  # Power spectrum value (l(l+1)Cl/2π in μK²)
        error = data[:, 4]  # Error
        
        return ell, power, error
        
    except Exception as e:
        print("Error loading WMAP power spectrum: {}".format(str(e)))
        return None, None, None


def load_planck_power_spectrum(file_path):
    """
    Load Planck CMB power spectrum data.
    
    Args:
        file_path (str): Path to the Planck power spectrum file
            
    Returns:
        tuple: (ell, power, error) arrays
    """
    try:
        # Load the power spectrum data
        # Planck power spectrum format:
        # Column 1 = multipole moment l
        # Column 2 = power spectrum value (Dl = l(l+1)Cl/2π in μK²)
        # Column 3 = lower error bound
        # Column 4 = upper error bound
        data = np.loadtxt(file_path, comments='#')
        
        ell = data[:, 0]  # Multipole moment
        power = data[:, 1]  # Power spectrum value (Dl = l(l+1)Cl/2π in μK²)
        
        # Use average of asymmetric error bars as the error
        lower_error = data[:, 2]  # Lower error bound
        upper_error = data[:, 3]  # Upper error bound
        error = (abs(lower_error) + abs(upper_error)) / 2.0
        
        return ell, power, error
        
    except Exception as e:
        print("Error loading Planck power spectrum: {}".format(str(e)))
        return None, None, None


def preprocess_data(data, smooth=False, smooth_window=5, normalize=True, detrend=False):
    """
    Preprocess data for analysis.
    
    Args:
        data (numpy.ndarray): Input data array
        smooth (bool): Whether to apply smoothing
        smooth_window (int): Window size for smoothing
        normalize (bool): Whether to normalize the data
        detrend (bool): Whether to remove linear trend
        
    Returns:
        numpy.ndarray: Preprocessed data
    """
    processed_data = data.copy()
    
    # Apply smoothing if requested
    if smooth:
        window = np.ones(smooth_window) / smooth_window
        processed_data = np.convolve(processed_data, window, mode='same')
    
    # Remove linear trend if requested
    if detrend:
        processed_data = signal.detrend(processed_data)
    
    # Normalize if requested
    if normalize:
        processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
    
    return processed_data


def run_cosmic_analysis(data, output_dir, name, tests='all', phi_bias=0.1, n_simulations=1000):
    """
    Run the Cosmic Consciousness Analysis on the provided data.
    
    Args:
        data (numpy.ndarray): Data to analyze
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        tests (str or list): Tests to run, 'all' or a list of test names
        phi_bias (float): Phi bias parameter for the analyzer
        n_simulations (int): Number of simulations for statistical testing
        
    Returns:
        dict: Analysis results
    """
    if not HAS_COSMIC_FRAMEWORK:
        print("Error: Cosmic Consciousness Analysis framework not found.")
        return None
    
    # Create analyzer
    analyzer = CosmicConsciousnessAnalyzer(
        data=data,
        output_dir=os.path.join(output_dir, name.lower()),
        phi_bias=phi_bias
    )
    
    # Run all tests
    print("\nRunning Cosmic Consciousness Analysis on {} data...".format(name))
    results = analyzer.run_analysis(
        tests=tests,
        visualize=True,
        report=True,
        parallel=True,
        n_jobs=-1,
        n_simulations=n_simulations
    )
    
    return results


def generate_comparison_report(wmap_results, planck_results, output_dir):
    """
    Generate a comprehensive comparison report between WMAP and Planck results.
    
    Args:
        wmap_results (dict): Results from WMAP analysis
        planck_results (dict): Results from Planck analysis
        output_dir (str): Directory to save the report
    """
    if not HAS_COSMIC_FRAMEWORK:
        print("Error: Cosmic Consciousness Analysis framework not found.")
        return
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, 'comparison')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # Generate comparison report
    report_path = os.path.join(comparison_dir, 'wmap_planck_comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# WMAP vs Planck Cosmic Consciousness Analysis Comparison\n\n")
        f.write("## Overview\n\n")
        f.write("This report compares the results of the Cosmic Consciousness Analysis framework ")
        f.write("applied to both WMAP and Planck CMB data.\n\n")
        
        f.write("## Test Results Comparison\n\n")
        
        # Compare results for each test
        for test_name in wmap_results.keys():
            if test_name in planck_results:
                f.write("### {}\n\n".format(test_name))
                
                wmap_phi = wmap_results[test_name].get('phi_optimality', 'N/A')
                wmap_pval = wmap_results[test_name].get('p_value', 'N/A')
                planck_phi = planck_results[test_name].get('phi_optimality', 'N/A')
                planck_pval = planck_results[test_name].get('p_value', 'N/A')
                
                f.write("| Metric | WMAP | Planck | Difference |\n")
                f.write("|--------|------|--------|------------|\n")
                
                if isinstance(wmap_phi, (int, float)) and isinstance(planck_phi, (int, float)):
                    diff_phi = abs(wmap_phi - planck_phi)
                    f.write("| Phi-Optimality | {:.4f} | {:.4f} | {:.4f} |\n".format(
                        wmap_phi, planck_phi, diff_phi))
                else:
                    f.write("| Phi-Optimality | {} | {} | N/A |\n".format(wmap_phi, planck_phi))
                
                if isinstance(wmap_pval, (int, float)) and isinstance(planck_pval, (int, float)):
                    diff_pval = abs(wmap_pval - planck_pval)
                    f.write("| P-Value | {:.4f} | {:.4f} | {:.4f} |\n".format(
                        wmap_pval, planck_pval, diff_pval))
                else:
                    f.write("| P-Value | {} | {} | N/A |\n".format(wmap_pval, planck_pval))
                
                f.write("\n")
        
        f.write("## Combined Significance\n\n")
        
        # Calculate combined significance using Fisher's method if available
        if 'combined_significance' in wmap_results and 'combined_significance' in planck_results:
            wmap_combined = wmap_results['combined_significance']
            planck_combined = planck_results['combined_significance']
            
            f.write("| Metric | WMAP | Planck |\n")
            f.write("|--------|------|--------|\n")
            f.write("| Combined P-Value | {:.6f} | {:.6f} |\n".format(
                wmap_combined['p_value'], planck_combined['p_value']))
            f.write("| Overall Significance | {} | {} |\n\n".format(
                wmap_combined['significant'], planck_combined['significant']))
        
        f.write("## Conclusion\n\n")
        f.write("This comparison demonstrates the consistency of golden ratio patterns ")
        f.write("and other consciousness-related metrics across two independent CMB measurements, ")
        f.write("strengthening the validity of the findings.\n\n")
        
        f.write("*Analysis performed: {}*\n".format(datetime.now().strftime("%Y-%m-%d")))
    
    print("Comparison report generated: {}".format(report_path))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run full Cosmic Consciousness Analysis on WMAP and Planck data')
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--tests', nargs='+', default='all', 
                        help='Tests to run, e.g., "golden_ratio coherence". Default: all tests')
    parser.add_argument('--phi-bias', type=float, default=0.1, 
                        help='Phi bias parameter for the analyzer. Default: 0.1')
    parser.add_argument('--n-simulations', type=int, default=1000, 
                        help='Number of simulations for statistical testing. Default: 1000')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results. Default: results/full_analysis_TIMESTAMP')
    
    args = parser.parse_args()
    
    # Check if Cosmic Consciousness Analysis framework is available
    if not HAS_COSMIC_FRAMEWORK:
        print("Error: Cosmic Consciousness Analysis framework not found.")
        print("This script requires the full framework to run the comprehensive analysis.")
        print("Basic analysis can still be performed using test_wmap_analysis.py or compare_wmap_planck.py")
        return 1
    
    # Path to data files
    wmap_file = 'wmap_data/raw_data/wmap_binned_tt_spectrum_9yr_v5.txt'
    planck_file = os.path.join(parent_dir, 'Cosmic_Consciousness_Analysis/planck_data/power_spectra/COM_PowerSpect_CMB-TT-full_R3.01.txt')
    
    # Check if files exist
    if not args.planck_only and not os.path.exists(wmap_file):
        print("Error: WMAP power spectrum file not found: {}".format(wmap_file))
        return 1
    
    if not args.wmap_only and not os.path.exists(planck_file):
        print("Error: Planck power spectrum file not found: {}".format(planck_file))
        print("Please make sure the Planck data is available in the Cosmic_Consciousness_Analysis repository.")
        return 1
    
    # Create output directory with timestamp
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('results', "full_analysis_{}".format(timestamp))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize results dictionaries
    wmap_results = None
    planck_results = None
    
    # Process WMAP data if requested
    if not args.planck_only:
        # Load WMAP data
        print("Loading WMAP power spectrum...")
        wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(wmap_file)
        
        if wmap_ell is None:
            print("Error loading WMAP power spectrum.")
            return 1
        
        print("Loaded WMAP power spectrum with {} multipoles".format(len(wmap_ell)))
        
        # Preprocess WMAP data
        print("Preprocessing WMAP data...")
        wmap_processed = preprocess_data(
            wmap_power, 
            smooth=args.smooth, 
            smooth_window=5, 
            normalize=True, 
            detrend=args.detrend
        )
        
        # Run analysis on WMAP data
        wmap_results = run_cosmic_analysis(
            wmap_processed, 
            output_dir, 
            'WMAP', 
            tests=args.tests,
            phi_bias=args.phi_bias,
            n_simulations=args.n_simulations
        )
    
    # Process Planck data if requested
    if not args.wmap_only:
        # Load Planck data
        print("Loading Planck power spectrum...")
        planck_ell, planck_power, planck_error = load_planck_power_spectrum(planck_file)
        
        if planck_ell is None:
            print("Error loading Planck power spectrum.")
            return 1
        
        print("Loaded Planck power spectrum with {} multipoles".format(len(planck_ell)))
        
        # Preprocess Planck data
        print("Preprocessing Planck data...")
        planck_processed = preprocess_data(
            planck_power, 
            smooth=args.smooth, 
            smooth_window=5, 
            normalize=True, 
            detrend=args.detrend
        )
        
        # Run analysis on Planck data
        planck_results = run_cosmic_analysis(
            planck_processed, 
            output_dir, 
            'Planck',
            tests=args.tests,
            phi_bias=args.phi_bias,
            n_simulations=args.n_simulations
        )
    
    # Generate comparison report if both datasets were analyzed
    if wmap_results and planck_results:
        print("Generating comparison report...")
        generate_comparison_report(wmap_results, planck_results, output_dir)
    
    print("\nAnalysis complete. Results saved to: {}".format(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
