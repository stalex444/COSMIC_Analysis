#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coherence Analysis Test for WMAP and Planck CMB data.

This script implements the Coherence Analysis Test, which evaluates if the CMB spectrum 
shows more coherence than random chance. The test is applied to both WMAP and Planck data
for comparison.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import argparse
from scipy.stats import pearsonr, spearmanr
from scipy.signal import coherence

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def compute_coherence(data, fs=1.0, nperseg=None):
    """
    Compute the coherence of a signal with itself at different scales.
    
    Args:
        data (numpy.ndarray): Input data array
        fs (float): Sampling frequency
        nperseg (int): Length of each segment
        
    Returns:
        tuple: (frequencies, coherence values)
    """
    if nperseg is None:
        nperseg = min(256, len(data) // 4)
    
    # Create a shifted version of the data
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    shift = int(len(data) / phi) % len(data)
    data_shifted = np.roll(data, shift)
    
    # Compute coherence between original and shifted data
    f, Cxy = coherence(data, data_shifted, fs=fs, nperseg=nperseg)
    
    return f, Cxy


def run_monte_carlo(data, n_simulations=10000, fs=1.0, nperseg=None):
    """
    Run Monte Carlo simulations to assess the significance of coherence values.
    
    Args:
        data (numpy.ndarray): Input data array
        n_simulations (int): Number of simulations
        fs (float): Sampling frequency
        nperseg (int): Length of each segment
        
    Returns:
        tuple: (p_value, phi_optimality)
    """
    # Compute actual coherence
    f, Cxy = compute_coherence(data, fs=fs, nperseg=nperseg)
    actual_coherence = np.mean(Cxy)
    
    # Run simulations
    sim_coherences = []
    for _ in range(n_simulations):
        # Create random permutation of the data
        sim_data = np.random.permutation(data)
        _, sim_Cxy = compute_coherence(sim_data, fs=fs, nperseg=nperseg)
        sim_coherences.append(np.mean(sim_Cxy))
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_coherence else 0 for sim in sim_coherences])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    sim_mean = np.mean(sim_coherences)
    sim_std = np.std(sim_coherences)
    if sim_std == 0:
        phi_optimality = 0
    else:
        phi_optimality = (actual_coherence - sim_mean) / sim_std
        # Scale to -1 to 1 range
        phi_optimality = np.tanh(phi_optimality)
    
    return p_value, phi_optimality, actual_coherence, sim_coherences


def plot_coherence_results(f, Cxy, p_value, phi_optimality, sim_coherences, actual_coherence, 
                          title, output_path):
    """
    Plot coherence analysis results.
    
    Args:
        f (numpy.ndarray): Frequencies
        Cxy (numpy.ndarray): Coherence values
        p_value (float): P-value from Monte Carlo simulations
        phi_optimality (float): Phi-optimality metric
        sim_coherences (list): Coherence values from simulations
        actual_coherence (float): Actual mean coherence
        title (str): Plot title
        output_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot coherence
    ax1.plot(f, Cxy, 'b-', linewidth=2)
    ax1.set_title('Coherence Analysis')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Coherence')
    ax1.grid(True)
    
    # Plot simulation results
    ax2.hist(sim_coherences, bins=30, alpha=0.7, color='gray', label='Random Simulations')
    ax2.axvline(actual_coherence, color='r', linestyle='--', linewidth=2, 
               label='Actual Coherence: {:.4f}'.format(actual_coherence))
    ax2.set_title('Monte Carlo Simulations')
    ax2.set_xlabel('Mean Coherence')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True)
    
    # Add text with results
    plt.figtext(0.5, 0.01, 'P-value: {:.4f} | Phi-Optimality: {:.4f}'.format(p_value, phi_optimality), 
               ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()


def run_coherence_analysis(data, output_dir, name, n_simulations=10000, fs=1.0, nperseg=None):
    """
    Run coherence analysis on the provided data.
    
    Args:
        data (numpy.ndarray): Data to analyze
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        n_simulations (int): Number of simulations for Monte Carlo
        fs (float): Sampling frequency
        nperseg (int): Length of each segment
        
    Returns:
        dict: Analysis results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Compute coherence
    f, Cxy = compute_coherence(data, fs=fs, nperseg=nperseg)
    
    # Run Monte Carlo simulations
    p_value, phi_optimality, actual_coherence, sim_coherences = run_monte_carlo(
        data, n_simulations=n_simulations, fs=fs, nperseg=nperseg)
    
    # Plot results
    plot_path = os.path.join(output_dir, '{}_coherence_analysis.png'.format(name.lower()))
    plot_coherence_results(
        f, Cxy, p_value, phi_optimality, sim_coherences, actual_coherence,
        'Coherence Analysis: {} CMB Data'.format(name), plot_path
    )
    
    # Save results to file
    results_path = os.path.join(output_dir, '{}_coherence_analysis.txt'.format(name.lower()))
    with open(results_path, 'w') as f:
        f.write('Coherence Analysis Results: {} CMB Data\n'.format(name))
        f.write('=' * 50 + '\n\n')
        f.write('Mean Coherence: {:.6f}\n'.format(actual_coherence))
        f.write('P-value: {:.6f}\n'.format(p_value))
        f.write('Phi-Optimality: {:.6f}\n'.format(phi_optimality))
        f.write('Significant: {}\n\n'.format(p_value < 0.05))
        f.write('Analysis performed on: {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write('Number of simulations: {}\n'.format(n_simulations))
    
    print('{} Coherence Analysis Results:'.format(name))
    print('  Mean Coherence: {:.6f}'.format(actual_coherence))
    print('  P-value: {:.6f}'.format(p_value))
    print('  Phi-Optimality: {:.6f}'.format(phi_optimality))
    print('  Significant: {}'.format(p_value < 0.05))
    
    return {
        'mean_coherence': actual_coherence,
        'p_value': p_value,
        'phi_optimality': phi_optimality,
        'significant': p_value < 0.05
    }


def compare_results(wmap_results, planck_results, output_dir):
    """
    Compare coherence analysis results between WMAP and Planck data.
    
    Args:
        wmap_results (dict): Results from WMAP analysis
        planck_results (dict): Results from Planck analysis
        output_dir (str): Directory to save comparison results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate differences
    coherence_diff = abs(wmap_results['mean_coherence'] - planck_results['mean_coherence'])
    phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
    
    # Save comparison to file
    comparison_path = os.path.join(output_dir, 'coherence_comparison.txt')
    with open(comparison_path, 'w') as f:
        f.write('Coherence Analysis Comparison: WMAP vs Planck\n')
        f.write('=' * 50 + '\n\n')
        
        f.write('WMAP Mean Coherence: {:.6f}\n'.format(wmap_results['mean_coherence']))
        f.write('WMAP P-value: {:.6f}\n'.format(wmap_results['p_value']))
        f.write('WMAP Phi-Optimality: {:.6f}\n'.format(wmap_results['phi_optimality']))
        f.write('WMAP Significant: {}\n\n'.format(wmap_results['significant']))
        
        f.write('Planck Mean Coherence: {:.6f}\n'.format(planck_results['mean_coherence']))
        f.write('Planck P-value: {:.6f}\n'.format(planck_results['p_value']))
        f.write('Planck Phi-Optimality: {:.6f}\n'.format(planck_results['phi_optimality']))
        f.write('Planck Significant: {}\n\n'.format(planck_results['significant']))
        
        f.write('Difference in Mean Coherence: {:.6f}\n'.format(coherence_diff))
        f.write('Difference in Phi-Optimality: {:.6f}\n'.format(phi_diff))
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Mean Coherence', 'Phi-Optimality']
    wmap_values = [wmap_results['mean_coherence'], wmap_results['phi_optimality']]
    planck_values = [planck_results['mean_coherence'], planck_results['phi_optimality']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, wmap_values, width, label='WMAP', color='blue', alpha=0.7)
    ax.bar(x + width/2, planck_values, width, label='Planck', color='red', alpha=0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('Coherence Analysis: WMAP vs Planck')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add text with p-values
    for i, metric in enumerate(metrics):
        plt.text(i - width/2, wmap_values[i] + 0.02, 
                'p={:.4f}'.format(wmap_results["p_value"]), 
                ha='center', va='bottom', color='blue', fontweight='bold')
        plt.text(i + width/2, planck_values[i] + 0.02, 
                'p={:.4f}'.format(planck_results["p_value"]), 
                ha='center', va='bottom', color='red', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    comparison_plot_path = os.path.join(output_dir, 'coherence_comparison.png')
    plt.savefig(comparison_plot_path)
    plt.close()
    
    print("\nComparison Results:")
    print("  Difference in Mean Coherence: {:.6f}".format(coherence_diff))
    print("  Difference in Phi-Optimality: {:.6f}".format(phi_diff))
    print("  Comparison saved to: {}".format(comparison_path))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Coherence Analysis Test on WMAP and Planck data')
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--n-simulations', type=int, default=10000, 
                        help='Number of simulations for Monte Carlo. Default: 10000')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results. Default: results/coherence_analysis_TIMESTAMP')
    
    args = parser.parse_args()
    
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
        output_dir = os.path.join('results', "coherence_analysis_{}".format(timestamp))
    
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
        
        # Run coherence analysis on WMAP data
        print("Running coherence analysis on WMAP data...")
        wmap_results = run_coherence_analysis(
            wmap_processed, 
            os.path.join(output_dir, 'wmap'), 
            'WMAP', 
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
        
        # Run coherence analysis on Planck data
        print("Running coherence analysis on Planck data...")
        planck_results = run_coherence_analysis(
            planck_processed, 
            os.path.join(output_dir, 'planck'), 
            'Planck',
            n_simulations=args.n_simulations
        )
    
    # Compare results if both datasets were analyzed
    if wmap_results and planck_results:
        print("Comparing WMAP and Planck coherence analysis results...")
        compare_results(wmap_results, planck_results, os.path.join(output_dir, 'comparison'))
    
    print("\nCoherence analysis complete. Results saved to: {}".format(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
