#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hierarchical Organization Test for WMAP and Planck CMB data.

This script implements the Hierarchical Organization Test, which checks for hierarchical 
patterns based on the golden ratio in the CMB power spectrum. The test is applied to both 
WMAP and Planck data for comparison.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import argparse
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
import multiprocessing
from functools import partial
import time

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


def create_hierarchical_levels(data, n_levels=5):
    """
    Create hierarchical levels based on the golden ratio.
    
    Args:
        data (numpy.ndarray): Input data array
        n_levels (int): Number of hierarchical levels to create
        
    Returns:
        list: List of data arrays at different hierarchical levels
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    levels = []
    
    for i in range(n_levels):
        # Calculate the scale factor for this level
        scale = phi ** i
        
        # Create a downsampled version of the data
        step = max(1, int(scale))
        level_data = data[::step]
        
        levels.append(level_data)
    
    return levels


def compute_hierarchical_correlation(levels):
    """
    Compute correlation between adjacent hierarchical levels.
    
    Args:
        levels (list): List of data arrays at different hierarchical levels
        
    Returns:
        tuple: (correlations, mean_correlation)
    """
    correlations = []
    
    for i in range(len(levels) - 1):
        # Get the current and next level
        current_level = levels[i]
        next_level = levels[i + 1]
        
        # Ensure the arrays have the same length for correlation
        min_len = min(len(current_level), len(next_level))
        current_level = current_level[:min_len]
        next_level = next_level[:min_len]
        
        # Compute correlation
        if min_len > 1:
            corr, _ = pearsonr(current_level, next_level)
            correlations.append(corr)
    
    # Calculate mean correlation
    mean_correlation = np.mean(correlations) if correlations else 0
    
    return correlations, mean_correlation


def run_monte_carlo_simulation(data, n_levels):
    """
    Run a single Monte Carlo simulation.
    
    Args:
        data (numpy.ndarray): Input data array
        n_levels (int): Number of hierarchical levels
        
    Returns:
        float: Mean correlation of the simulation
    """
    # Create random permutation of the data
    sim_data = np.random.permutation(data)
    sim_levels = create_hierarchical_levels(sim_data, n_levels)
    _, sim_mean_correlation = compute_hierarchical_correlation(sim_levels)
    return sim_mean_correlation


def _run_simulation_wrapper(args):
    """
    Wrapper function for multiprocessing to handle simulation runs.
    
    Args:
        args (tuple): Tuple containing (data, n_levels)
        
    Returns:
        float: Mean correlation of the simulation
    """
    data, n_levels = args
    return run_monte_carlo_simulation(data, n_levels)


def run_monte_carlo_parallel(data, n_simulations=10000, n_levels=5, num_processes=None):
    """
    Run Monte Carlo simulations in parallel to assess the significance of hierarchical organization.
    
    Args:
        data (numpy.ndarray): Input data array
        n_simulations (int): Number of simulations
        n_levels (int): Number of hierarchical levels
        num_processes (int): Number of processes to use for parallelization. Default: None (uses all available cores)
        
    Returns:
        tuple: (p_value, phi_optimality, actual_correlation, sim_correlations)
    """
    # Compute actual hierarchical correlation
    actual_levels = create_hierarchical_levels(data, n_levels)
    actual_correlations, actual_mean_correlation = compute_hierarchical_correlation(actual_levels)
    
    # Run simulations in parallel
    # Python 2 compatible version (no context manager)
    pool = multiprocessing.Pool(processes=num_processes)
    try:
        # Create a list of arguments for each simulation
        args_list = [(data, n_levels)] * n_simulations
        sim_mean_correlations = pool.map(_run_simulation_wrapper, args_list)
    finally:
        pool.close()
        pool.join()
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_mean_correlation else 0 for sim in sim_mean_correlations])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    sim_mean = np.mean(sim_mean_correlations)
    sim_std = np.std(sim_mean_correlations)
    if sim_std == 0:
        phi_optimality = 0
    else:
        phi_optimality = (actual_mean_correlation - sim_mean) / sim_std
        # Scale to -1 to 1 range
        phi_optimality = np.tanh(phi_optimality)
    
    return p_value, phi_optimality, actual_mean_correlation, sim_mean_correlations, actual_correlations


def run_monte_carlo(data, n_simulations=10000, n_levels=5):
    """
    Run Monte Carlo simulations to assess the significance of hierarchical organization.
    
    Args:
        data (numpy.ndarray): Input data array
        n_simulations (int): Number of simulations
        n_levels (int): Number of hierarchical levels
        
    Returns:
        tuple: (p_value, phi_optimality, actual_correlation, sim_correlations)
    """
    # Compute actual hierarchical correlation
    actual_levels = create_hierarchical_levels(data, n_levels)
    actual_correlations, actual_mean_correlation = compute_hierarchical_correlation(actual_levels)
    
    # Run simulations
    sim_mean_correlations = []
    for _ in range(n_simulations):
        # Create random permutation of the data
        sim_data = np.random.permutation(data)
        sim_levels = create_hierarchical_levels(sim_data, n_levels)
        _, sim_mean_correlation = compute_hierarchical_correlation(sim_levels)
        sim_mean_correlations.append(sim_mean_correlation)
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_mean_correlation else 0 for sim in sim_mean_correlations])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    sim_mean = np.mean(sim_mean_correlations)
    sim_std = np.std(sim_mean_correlations)
    if sim_std == 0:
        phi_optimality = 0
    else:
        phi_optimality = (actual_mean_correlation - sim_mean) / sim_std
        # Scale to -1 to 1 range
        phi_optimality = np.tanh(phi_optimality)
    
    return p_value, phi_optimality, actual_mean_correlation, sim_mean_correlations, actual_correlations


def plot_hierarchical_results(actual_correlations, p_value, phi_optimality, 
                             sim_correlations, actual_mean_correlation, 
                             title, output_path):
    """
    Plot hierarchical organization results.
    
    Args:
        actual_correlations (list): Correlations between adjacent hierarchical levels
        p_value (float): P-value from Monte Carlo simulations
        phi_optimality (float): Phi-optimality metric
        sim_correlations (list): Mean correlations from simulations
        actual_mean_correlation (float): Actual mean correlation
        title (str): Plot title
        output_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot correlations between hierarchical levels
    levels = list(range(1, len(actual_correlations) + 1))
    ax1.plot(levels, actual_correlations, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Correlations Between Hierarchical Levels')
    ax1.set_xlabel('Level')
    ax1.set_ylabel('Correlation')
    ax1.grid(True)
    
    # Plot simulation results
    ax2.hist(sim_correlations, bins=30, alpha=0.7, color='gray', label='Random Simulations')
    ax2.axvline(actual_mean_correlation, color='r', linestyle='--', linewidth=2, 
               label='Actual Mean Correlation: {:.4f}'.format(actual_mean_correlation))
    ax2.set_title('Monte Carlo Simulations')
    ax2.set_xlabel('Mean Correlation')
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


def run_hierarchical_organization_test(data, output_dir, name, n_simulations=10000, n_levels=5, parallel=False, num_processes=None):
    """
    Run hierarchical organization test on the provided data.
    
    Args:
        data (numpy.ndarray): Data to analyze
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        n_simulations (int): Number of simulations for Monte Carlo
        n_levels (int): Number of hierarchical levels
        parallel (bool): Whether to run Monte Carlo simulations in parallel. Default: False
        num_processes (int): Number of processes to use for parallel execution. Default: None (uses all available cores)
        
    Returns:
        dict: Analysis results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Running Hierarchical Organization Test on {} data...".format(name))
    print("Number of simulations: {}".format(n_simulations))
    
    if parallel:
        num_processes = num_processes or multiprocessing.cpu_count()
        print("Using parallel processing with {} processes".format(num_processes))
    
    # Run Monte Carlo simulations
    start_time = time.time()
    if parallel:
        p_value, phi_optimality, actual_mean_correlation, sim_correlations, actual_correlations = run_monte_carlo_parallel(
            data, n_simulations=n_simulations, n_levels=n_levels, num_processes=num_processes)
    else:
        p_value, phi_optimality, actual_mean_correlation, sim_correlations, actual_correlations = run_monte_carlo(
            data, n_simulations=n_simulations, n_levels=n_levels)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Completed {} simulations in {:.2f} seconds".format(n_simulations, execution_time))
    print("p-value: {:.4f}, phi-optimality: {:.4f}".format(p_value, phi_optimality))
    
    # Plot results
    plot_path = os.path.join(output_dir, '{}_hierarchical_organization.png'.format(name.lower()))
    plot_hierarchical_results(
        actual_correlations, 
        p_value, 
        phi_optimality, 
        sim_correlations, 
        actual_mean_correlation, 
        "{} Hierarchical Organization".format(name), 
        plot_path
    )
    
    # Save results to file
    results_path = os.path.join(output_dir, '{}_hierarchical_organization.txt'.format(name.lower()))
    with open(results_path, 'w') as f:
        f.write('Hierarchical Organization Test Results: {} CMB Data\n'.format(name))
        f.write('=' * 60 + '\n\n')
        f.write('Mean Correlation: {:.6f}\n'.format(actual_mean_correlation))
        f.write('P-value: {:.6f}\n'.format(p_value))
        f.write('Phi-Optimality: {:.6f}\n'.format(phi_optimality))
        f.write('Significant: {}\n\n'.format(p_value < 0.05))
        
        f.write('Correlations Between Hierarchical Levels:\n')
        for i, corr in enumerate(actual_correlations):
            f.write('  Level {} to {}: {:.6f}\n'.format(i+1, i+2, corr))
        
        f.write('\nAnalysis performed on: {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write('Number of simulations: {}\n'.format(n_simulations))
        f.write('Number of hierarchical levels: {}\n'.format(n_levels))
    
    print("{} Hierarchical Organization Test Results:".format(name))
    print("  Mean Correlation: {:.6f}".format(actual_mean_correlation))
    print("  P-value: {:.6f}".format(p_value))
    print("  Phi-Optimality: {:.6f}".format(phi_optimality))
    print("  Significant: {}".format(p_value < 0.05))
    
    return {
        'mean_correlation': actual_mean_correlation,
        'p_value': p_value,
        'phi_optimality': phi_optimality,
        'significant': p_value < 0.05,
        'level_correlations': actual_correlations
    }


def compare_results(wmap_results, planck_results, output_dir):
    """
    Compare hierarchical organization test results between WMAP and Planck data.
    
    Args:
        wmap_results (dict): Results from WMAP analysis
        planck_results (dict): Results from Planck analysis
        output_dir (str): Directory to save comparison results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate differences
    correlation_diff = abs(wmap_results['mean_correlation'] - planck_results['mean_correlation'])
    phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
    
    # Save comparison to file
    comparison_path = os.path.join(output_dir, 'hierarchical_organization_comparison.txt')
    with open(comparison_path, 'w') as f:
        f.write('Hierarchical Organization Test Comparison: WMAP vs Planck\n')
        f.write('=' * 60 + '\n\n')
        
        f.write('WMAP Mean Correlation: {:.6f}\n'.format(wmap_results['mean_correlation']))
        f.write('WMAP P-value: {:.6f}\n'.format(wmap_results['p_value']))
        f.write('WMAP Phi-Optimality: {:.6f}\n'.format(wmap_results['phi_optimality']))
        f.write('WMAP Significant: {}\n\n'.format(wmap_results['significant']))
        
        f.write('Planck Mean Correlation: {:.6f}\n'.format(planck_results['mean_correlation']))
        f.write('Planck P-value: {:.6f}\n'.format(planck_results['p_value']))
        f.write('Planck Phi-Optimality: {:.6f}\n'.format(planck_results['phi_optimality']))
        f.write('Planck Significant: {}\n\n'.format(planck_results['significant']))
        
        f.write('Difference in Mean Correlation: {:.6f}\n'.format(correlation_diff))
        f.write('Difference in Phi-Optimality: {:.6f}\n'.format(phi_diff))
        
        # Compare level correlations if available
        if ('level_correlations' in wmap_results and 'level_correlations' in planck_results and
            len(wmap_results['level_correlations']) == len(planck_results['level_correlations'])):
            
            f.write('\nLevel-by-Level Correlation Comparison:\n')
            for i in range(len(wmap_results['level_correlations'])):
                wmap_corr = wmap_results['level_correlations'][i]
                planck_corr = planck_results['level_correlations'][i]
                diff = abs(wmap_corr - planck_corr)
                
                f.write('  Level {} to {}:\n'.format(i+1, i+2))
                f.write('    WMAP: {:.6f}\n'.format(wmap_corr))
                f.write('    Planck: {:.6f}\n'.format(planck_corr))
                f.write('    Difference: {:.6f}\n'.format(diff))
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar chart of mean correlation and phi-optimality
    metrics = ['Mean Correlation', 'Phi-Optimality']
    wmap_values = [wmap_results['mean_correlation'], wmap_results['phi_optimality']]
    planck_values = [planck_results['mean_correlation'], planck_results['phi_optimality']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, wmap_values, width, label='WMAP', color='blue', alpha=0.7)
    ax1.bar(x + width/2, planck_values, width, label='Planck', color='red', alpha=0.7)
    
    ax1.set_ylabel('Value')
    ax1.set_title('Hierarchical Organization: WMAP vs Planck')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    
    # Add text with p-values
    for i, metric in enumerate(metrics):
        ax1.text(i - width/2, wmap_values[i] + 0.02, 
                'p={:.4f}'.format(wmap_results["p_value"]), 
                ha='center', va='bottom', color='blue', fontweight='bold')
        ax1.text(i + width/2, planck_values[i] + 0.02, 
                'p={:.4f}'.format(planck_results["p_value"]), 
                ha='center', va='bottom', color='red', fontweight='bold')
    
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Level correlations comparison
    if ('level_correlations' in wmap_results and 'level_correlations' in planck_results and
        len(wmap_results['level_correlations']) > 0):
        
        levels = list(range(1, len(wmap_results['level_correlations']) + 1))
        
        ax2.plot(levels, wmap_results['level_correlations'], 'bo-', linewidth=2, 
                label='WMAP', markersize=8)
        ax2.plot(levels, planck_results['level_correlations'], 'ro-', linewidth=2, 
                label='Planck', markersize=8)
        
        ax2.set_title('Level-by-Level Correlation Comparison')
        ax2.set_xlabel('Level')
        ax2.set_ylabel('Correlation')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    comparison_plot_path = os.path.join(output_dir, 'hierarchical_organization_comparison.png')
    plt.savefig(comparison_plot_path)
    plt.close()
    
    print("\nComparison Results:")
    print("  Difference in Mean Correlation: {:.6f}".format(correlation_diff))
    print("  Difference in Phi-Optimality: {:.6f}".format(phi_diff))
    print("  Comparison saved to: {}".format(comparison_path))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Hierarchical Organization Test on WMAP and Planck data')
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--n-simulations', type=int, default=10000, 
                        help='Number of simulations for Monte Carlo. Default: 10000')
    parser.add_argument('--n-levels', type=int, default=5,
                        help='Number of hierarchical levels. Default: 5')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results. Default: results/hierarchical_organization_TIMESTAMP')
    parser.add_argument('--parallel', action='store_true', help='Run Monte Carlo simulations in parallel')
    parser.add_argument('--num-processes', type=int, default=None, 
                        help='Number of processes to use for parallel execution. Default: use all available cores')
    
    args = parser.parse_args()
    
    # Start timing the entire test
    overall_start_time = time.time()
    
    # Print test header
    print("=" * 50)
    print("Starting Hierarchical Organization Test at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 50)
    print("Parameters:")
    print("  - Number of simulations: {}".format(args.n_simulations))
    print("  - Number of hierarchical levels: {}".format(args.n_levels))
    print("  - Smooth: {}".format(args.smooth))
    print("  - Detrend: {}".format(args.detrend))
    print("  - Parallel: {}".format(args.parallel))
    if args.parallel:
        num_processes = args.num_processes or multiprocessing.cpu_count()
        print("  - Number of processes: {}".format(num_processes if args.num_processes else "all available"))
    print("=" * 50)
    
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
        output_dir = os.path.join('results', "hierarchical_organization_{}".format(timestamp))
    
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
        
        # Run hierarchical organization test on WMAP data
        print("Running hierarchical organization test on WMAP data...")
        wmap_results = run_hierarchical_organization_test(
            wmap_processed, 
            os.path.join(output_dir, 'wmap'), 
            'WMAP', 
            n_simulations=args.n_simulations,
            n_levels=args.n_levels,
            parallel=args.parallel,
            num_processes=args.num_processes
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
        
        # Run hierarchical organization test on Planck data
        print("Running hierarchical organization test on Planck data...")
        planck_results = run_hierarchical_organization_test(
            planck_processed, 
            os.path.join(output_dir, 'planck'), 
            'Planck',
            n_simulations=args.n_simulations,
            n_levels=args.n_levels,
            parallel=args.parallel,
            num_processes=args.num_processes
        )
    
    # Compare results if both datasets were analyzed
    if wmap_results and planck_results:
        print("Comparing WMAP and Planck hierarchical organization test results...")
        compare_results(wmap_results, planck_results, os.path.join(output_dir, 'comparison'))
    
    # Print overall execution time
    overall_end_time = time.time()
    overall_execution_time = overall_end_time - overall_start_time
    print("\nHierarchical organization test complete. Results saved to: {}".format(output_dir))
    print("Total execution time: {:.2f} seconds".format(overall_execution_time))
    return 0


if __name__ == "__main__":
    sys.exit(main())
