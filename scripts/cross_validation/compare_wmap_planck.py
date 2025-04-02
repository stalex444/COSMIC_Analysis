#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare WMAP and Planck data analysis results for the Cosmic Consciousness Analysis framework.
This script loads both WMAP and Planck power spectrum data, performs golden ratio analysis,
and compares the results between the two datasets.
"""

from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime

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


def analyze_golden_ratio(data, n_shuffles=1000):
    """
    Simple test to analyze golden ratio patterns in the data.
    
    Args:
        data (numpy.ndarray): Input data array
        n_shuffles (int): Number of shuffles for significance testing
        
    Returns:
        dict: Analysis results
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Calculate correlations between data points separated by golden ratio
    correlations = []
    for i in range(len(data) - int(phi)):
        j = i + int(phi)
        correlations.append(data[i] * data[j])
    
    # Calculate mean correlation
    mean_correlation = np.mean(correlations)
    
    # Perform significance testing
    shuffled_correlations = []
    for _ in range(n_shuffles):
        shuffled_data = np.random.permutation(data)
        shuffle_corrs = []
        for i in range(len(shuffled_data) - int(phi)):
            j = i + int(phi)
            shuffle_corrs.append(shuffled_data[i] * shuffled_data[j])
        shuffled_correlations.append(np.mean(shuffle_corrs))
    
    # Calculate p-value
    p_value = np.mean([1 if abs(sc) >= abs(mean_correlation) else 0 for sc in shuffled_correlations])
    
    return {
        'correlation': mean_correlation,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def run_cosmic_analysis(data, output_dir, name):
    """
    Run the Cosmic Consciousness Analysis on the provided data.
    
    Args:
        data (numpy.ndarray): Data to analyze
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        
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
        phi_bias=0.1
    )
    
    # Run all tests
    print("\nRunning Cosmic Consciousness Analysis on {} data...".format(name))
    results = analyzer.run_analysis(
        tests='all',
        visualize=True,
        report=True,
        parallel=True,
        n_jobs=-1
    )
    
    return results


def compare_datasets(wmap_data, planck_data, output_dir):
    """
    Compare WMAP and Planck datasets.
    
    Args:
        wmap_data (dict): WMAP data dictionary with ell, power, and processed_power
        planck_data (dict): Planck data dictionary with ell, power, and processed_power
        output_dir (str): Directory to save comparison results
    """
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, 'comparison')
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # Compare golden ratio correlations
    wmap_gr = analyze_golden_ratio(wmap_data['processed_power'], n_shuffles=1000)
    planck_gr = analyze_golden_ratio(planck_data['processed_power'], n_shuffles=1000)
    
    # Save comparison results
    with open(os.path.join(comparison_dir, 'golden_ratio_comparison.txt'), 'w') as f:
        f.write("Golden Ratio Correlation Comparison: WMAP vs Planck\n")
        f.write("=================================================\n\n")
        f.write("WMAP Golden Ratio Correlation: {:.4f}\n".format(wmap_gr['correlation']))
        f.write("WMAP P-value: {:.4f}\n".format(wmap_gr['p_value']))
        f.write("WMAP Significant: {}\n\n".format(wmap_gr['significant']))
        f.write("Planck Golden Ratio Correlation: {:.4f}\n".format(planck_gr['correlation']))
        f.write("Planck P-value: {:.4f}\n".format(planck_gr['p_value']))
        f.write("Planck Significant: {}\n\n".format(planck_gr['significant']))
        f.write("Difference in correlation: {:.4f}\n".format(
            abs(wmap_gr['correlation'] - planck_gr['correlation'])))
    
    # Plot comparison of power spectra
    plt.figure(figsize=(12, 8))
    
    # Original power spectra
    plt.subplot(2, 1, 1)
    plt.errorbar(wmap_data['ell'], wmap_data['power'], yerr=wmap_data['error'], 
                fmt='o', markersize=3, alpha=0.5, label='WMAP')
    plt.errorbar(planck_data['ell'], planck_data['power'], yerr=planck_data['error'], 
                fmt='o', markersize=3, alpha=0.5, label='Planck')
    plt.xlabel('Multipole (l)')
    plt.ylabel('Power (uK^2)')
    plt.title('WMAP vs Planck Power Spectrum Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Processed power spectra
    plt.subplot(2, 1, 2)
    plt.plot(wmap_data['ell'], wmap_data['processed_power'], label='WMAP')
    plt.plot(planck_data['ell'], planck_data['processed_power'], label='Planck')
    plt.xlabel('Multipole (l)')
    plt.ylabel('Normalized Power')
    plt.title('Processed Power Spectrum Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(comparison_dir, 'wmap_planck_comparison.png'), dpi=150)
    print("Comparison plot saved to: {}".format(
        os.path.join(comparison_dir, 'wmap_planck_comparison.png')))
    
    return {
        'wmap_gr': wmap_gr,
        'planck_gr': planck_gr
    }


def main():
    # Path to data files
    wmap_file = 'wmap_data/raw_data/wmap_binned_tt_spectrum_9yr_v5.txt'
    planck_file = os.path.join(parent_dir, 'Cosmic_Consciousness_Analysis/planck_data/power_spectra/COM_PowerSpect_CMB-TT-full_R3.01.txt')
    
    # Check if files exist
    if not os.path.exists(wmap_file):
        print("Error: WMAP power spectrum file not found: {}".format(wmap_file))
        return 1
    
    if not os.path.exists(planck_file):
        print("Error: Planck power spectrum file not found: {}".format(planck_file))
        print("Please make sure the Planck data is available in the Cosmic_Consciousness_Analysis repository.")
        return 1
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('results', "comparison_{}".format(timestamp))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load WMAP data
    print("Loading WMAP power spectrum...")
    wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(wmap_file)
    
    if wmap_ell is None:
        print("Error loading WMAP power spectrum.")
        return 1
    
    print("Loaded WMAP power spectrum with {} multipoles".format(len(wmap_ell)))
    
    # Load Planck data
    print("Loading Planck power spectrum...")
    planck_ell, planck_power, planck_error = load_planck_power_spectrum(planck_file)
    
    if planck_ell is None:
        print("Error loading Planck power spectrum.")
        return 1
    
    print("Loaded Planck power spectrum with {} multipoles".format(len(planck_ell)))
    
    # Preprocess data
    print("Preprocessing WMAP data...")
    wmap_processed = preprocess_data(
        wmap_power, 
        smooth=True, 
        smooth_window=5, 
        normalize=True, 
        detrend=True
    )
    
    print("Preprocessing Planck data...")
    planck_processed = preprocess_data(
        planck_power, 
        smooth=True, 
        smooth_window=5, 
        normalize=True, 
        detrend=True
    )
    
    # Create data dictionaries
    wmap_data = {
        'ell': wmap_ell,
        'power': wmap_power,
        'error': wmap_error,
        'processed_power': wmap_processed
    }
    
    planck_data = {
        'ell': planck_ell,
        'power': planck_power,
        'error': planck_error,
        'processed_power': planck_processed
    }
    
    # Compare datasets
    print("Comparing WMAP and Planck datasets...")
    comparison_results = compare_datasets(wmap_data, planck_data, output_dir)
    
    # Print comparison results
    print("\nComparison Results:")
    print("  WMAP Golden Ratio Correlation: {:.4f}".format(comparison_results['wmap_gr']['correlation']))
    print("  WMAP P-value: {:.4f}".format(comparison_results['wmap_gr']['p_value']))
    print("  WMAP Significant: {}".format(comparison_results['wmap_gr']['significant']))
    print("  Planck Golden Ratio Correlation: {:.4f}".format(comparison_results['planck_gr']['correlation']))
    print("  Planck P-value: {:.4f}".format(comparison_results['planck_gr']['p_value']))
    print("  Planck Significant: {}".format(comparison_results['planck_gr']['significant']))
    print("  Difference in correlation: {:.4f}".format(
        abs(comparison_results['wmap_gr']['correlation'] - comparison_results['planck_gr']['correlation'])))
    
    # Run full Cosmic Consciousness Analysis if framework is available
    if HAS_COSMIC_FRAMEWORK:
        # Run analysis on WMAP data
        wmap_cosmic_results = run_cosmic_analysis(
            wmap_processed, 
            os.path.join(output_dir, 'cosmic_analysis'),
            'WMAP'
        )
        
        # Run analysis on Planck data
        planck_cosmic_results = run_cosmic_analysis(
            planck_processed, 
            os.path.join(output_dir, 'cosmic_analysis'),
            'Planck'
        )
        
        if wmap_cosmic_results and planck_cosmic_results:
            print("\nCosmic Consciousness Analysis completed successfully for both datasets.")
            print("Results saved to: {}".format(os.path.join(output_dir, 'cosmic_analysis')))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
