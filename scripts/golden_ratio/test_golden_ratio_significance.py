#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Golden Ratio Significance Test for WMAP and Planck CMB data.

This script implements the Golden Ratio Significance Test, which tests if multipoles related 
by the golden ratio have statistically significant power in the CMB power spectrum.
The test is applied to both WMAP and Planck data for comparison.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import argparse
from scipy.stats import pearsonr, spearmanr
import multiprocessing
from functools import partial
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def find_golden_ratio_pairs(ell, max_l=2500):
    """
    Find pairs of multipoles related by the golden ratio.
    
    Args:
        ell (numpy.ndarray): Array of multipole moments
        max_l (int): Maximum multipole moment to consider
        
    Returns:
        list: List of (l1, l2) pairs where l2/l1 ≈ φ
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
    tolerance = 0.05  # Tolerance for considering a ratio as close to phi
    
    # Filter multipoles within range
    valid_ell = ell[ell <= max_l]
    
    pairs = []
    for i, l1 in enumerate(valid_ell):
        for j, l2 in enumerate(valid_ell[i+1:], i+1):
            ratio = l2 / l1
            if abs(ratio - phi) < tolerance:
                pairs.append((l1, l2))
    
    return pairs


def calculate_golden_ratio_correlation(ell, power, max_l=2500):
    """
    Calculate correlation between powers at multipoles related by the golden ratio.
    
    Args:
        ell (numpy.ndarray): Array of multipole moments
        power (numpy.ndarray): Array of power spectrum values
        max_l (int): Maximum multipole moment to consider
        
    Returns:
        tuple: (correlation, p_value, gr_pairs, powers_l1, powers_l2)
    """
    # Find golden ratio pairs
    gr_pairs = find_golden_ratio_pairs(ell, max_l)
    
    if len(gr_pairs) < 2:
        return 0, 1, gr_pairs, [], []
    
    # Extract powers for each pair
    powers_l1 = []
    powers_l2 = []
    
    for l1, l2 in gr_pairs:
        idx1 = np.where(ell == l1)[0]
        idx2 = np.where(ell == l2)[0]
        
        # Skip if indices not found
        if len(idx1) == 0 or len(idx2) == 0:
            continue
            
        idx1 = idx1[0]
        idx2 = idx2[0]
        
        # Skip negative or zero power values
        if idx1 >= len(power) or idx2 >= len(power) or power[idx1] <= 0 or power[idx2] <= 0:
            continue
        
        powers_l1.append(power[idx1])
        powers_l2.append(power[idx2])
    
    # Calculate correlation
    if len(powers_l1) >= 2:  # Need at least 2 points for correlation
        correlation, p_value = pearsonr(powers_l1, powers_l2)
    else:
        correlation, p_value = 0, 1
    
    return correlation, p_value, gr_pairs, powers_l1, powers_l2


def run_monte_carlo(ell, power, n_simulations=100, max_l=2500):
    """
    Run Monte Carlo simulations to assess the significance of golden ratio correlations.
    
    Args:
        ell (numpy.ndarray): Array of multipole moments
        power (numpy.ndarray): Array of power spectrum values
        n_simulations (int): Number of simulations
        max_l (int): Maximum multipole moment to consider
        
    Returns:
        tuple: (p_value, phi_optimality, actual_corr, sim_corrs, gr_pairs, powers_l1, powers_l2)
    """
    # Calculate actual correlation
    actual_corr, _, gr_pairs, powers_l1, powers_l2 = calculate_golden_ratio_correlation(ell, power, max_l)
    
    # Run simulations
    sim_corrs = []
    for i in range(n_simulations):
        if i % 10 == 0:
            print("  Simulation {}/{}".format(i, n_simulations))
        # Create random permutation of the power spectrum
        sim_power = np.random.permutation(power)
        sim_corr, _, _, _, _ = calculate_golden_ratio_correlation(ell, sim_power, max_l)
        sim_corrs.append(sim_corr)
    
    # Calculate p-value
    p_value = np.mean([1 if abs(sim) >= abs(actual_corr) else 0 for sim in sim_corrs])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    sim_mean = np.mean(sim_corrs)
    sim_std = np.std(sim_corrs)
    if sim_std == 0:
        phi_optimality = 0
    else:
        phi_optimality = (actual_corr - sim_mean) / sim_std
        # Scale to -1 to 1 range
        phi_optimality = np.tanh(phi_optimality)
    
    return p_value, phi_optimality, actual_corr, sim_corrs, gr_pairs, powers_l1, powers_l2


# Helper function for parallel processing - moved outside to make it picklable
def _simulate_correlation(args):
    """
    Simulate golden ratio correlation on randomized data.
    
    Args:
        args: Tuple containing (seed, ell, power)
        
    Returns:
        float: Correlation coefficient for simulated data
    """
    seed, ell, power = args
    np.random.seed(seed)
    
    # Create a randomized version of the power spectrum
    random_power = np.copy(power)
    np.random.shuffle(random_power)
    
    # Skip if all power values are invalid
    if np.all(random_power <= 0):
        return np.nan
        
    # Calculate correlation for this random data
    try:
        corr, _, _, _, _ = calculate_golden_ratio_correlation(ell, random_power)
        return corr if not np.isnan(corr) else 0.0
    except Exception:
        # Return NaN if correlation calculation fails
        return np.nan

def run_monte_carlo_parallel(ell, power, n_simulations=100, max_l=2500, num_processes=4):
    """
    Run Monte Carlo simulations in parallel to assess the significance of golden ratio correlations.
    
    Args:
        ell (numpy.ndarray): Array of multipole moments
        power (numpy.ndarray): Array of power spectrum values
        n_simulations (int): Number of simulations
        max_l (int): Maximum multipole moment to consider
        num_processes (int): Number of processes to use for parallelization
        
    Returns:
        tuple: (p_value, phi_optimality, actual_corr, sim_corrs, gr_pairs, powers_l1, powers_l2)
    """
    print("Running {} Monte Carlo simulations with {} processes...".format(n_simulations, num_processes))
    
    # Calculate actual correlation for real data
    actual_corr, _, gr_pairs, powers_l1, powers_l2 = calculate_golden_ratio_correlation(ell, power, max_l)
    
    # Prepare the arguments for the parallel map
    args_list = [(i, ell, power) for i in range(n_simulations)]
    
    # Run simulations in parallel
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        sim_corrs = pool.map(_simulate_correlation, args_list)
    
    end_time = time.time()
    print("Simulations completed in {:.2f} seconds".format(end_time - start_time))
    
    # Remove any NaN values that might have occurred during simulation
    sim_corrs = np.array([corr for corr in sim_corrs if not np.isnan(corr)])
    
    # Handle case where all simulations might have failed
    if len(sim_corrs) == 0:
        return 1.0, 0.0, actual_corr, [], gr_pairs, powers_l1, powers_l2
    
    # Calculate p-value
    p_value = (np.sum(np.array(sim_corrs) >= actual_corr) + 1) / (n_simulations + 1)
    
    # Calculate phi optimality as the ratio between actual and mean simulated correlation
    mean_sim_corr = np.mean(sim_corrs)
    if np.isnan(mean_sim_corr) or mean_sim_corr == 0:
        phi_optimality = 0
    else:
        raw_optimality = (actual_corr - mean_sim_corr) / abs(mean_sim_corr)
        phi_optimality = min(1.0, raw_optimality / 3.0) if raw_optimality > 0 else max(-1.0, raw_optimality)
    
    return p_value, phi_optimality, actual_corr, sim_corrs, gr_pairs, powers_l1, powers_l2


def plot_golden_ratio_results(gr_pairs, powers_l1, powers_l2, p_value, phi_optimality, 
                             sim_corrs, actual_corr, title, output_path):
    """
    Plot golden ratio significance test results.
    
    Args:
        gr_pairs (list): List of golden ratio pairs
        powers_l1 (list): Powers at first multipoles in pairs
        powers_l2 (list): Powers at second multipoles in pairs
        p_value (float): P-value from Monte Carlo simulations
        phi_optimality (float): Phi-optimality metric
        sim_corrs (list): Correlations from simulations
        actual_corr (float): Actual correlation
        title (str): Plot title
        output_path (str): Path to save the plot
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot correlation between powers at golden ratio-related multipoles
        if len(powers_l1) > 0 and len(powers_l2) > 0:
            ax1.scatter(powers_l1, powers_l2, c='blue', alpha=0.7)
            
            # Add regression line if there are enough points
            if len(powers_l1) >= 2:
                m, b = np.polyfit(powers_l1, powers_l2, 1)
                x_range = np.linspace(min(powers_l1), max(powers_l1), 100)
                ax1.plot(x_range, m * x_range + b, 'r-', linewidth=2)
        
        ax1.set_title('Correlation Between Golden Ratio Multipoles')
        ax1.set_xlabel('Power at l')
        ax1.set_ylabel('Power at l * GR')
        ax1.grid(True)
        
        # Add correlation coefficient
        ax1.text(0.05, 0.95, 'r = {:.4f}'.format(actual_corr), 
                transform=ax1.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot simulation results
        if len(sim_corrs) > 0:
            ax2.hist(sim_corrs, bins=min(30, len(sim_corrs)//3), alpha=0.7, color='gray', label='Random Simulations')
            ax2.axvline(actual_corr, color='r', linestyle='--', linewidth=2, 
                       label='Actual Correlation: {:.4f}'.format(actual_corr))
        
        ax2.set_title('Monte Carlo Simulations')
        ax2.set_xlabel('Correlation Coefficient')
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
    except Exception as e:
        print("Warning: Error in plotting golden ratio results: {}".format(str(e)))
        print("Continuing with analysis...")


def run_golden_ratio_test(ell, power, output_dir, name, n_simulations=100, max_l=2500, num_processes=4):
    """
    Run golden ratio significance test on the provided data.
    
    Args:
        ell (numpy.ndarray): Array of multipole moments
        power (numpy.ndarray): Array of power spectrum values
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        n_simulations (int): Number of simulations for Monte Carlo
        max_l (int): Maximum multipole moment to consider
        num_processes (int): Number of processes to use for parallelization
        
    Returns:
        dict: Analysis results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run Monte Carlo simulations
    p_value, phi_optimality, actual_corr, sim_corrs, gr_pairs, powers_l1, powers_l2 = run_monte_carlo_parallel(
        ell, power, n_simulations=n_simulations, max_l=max_l, num_processes=num_processes)
    
    # Plot results
    plot_path = os.path.join(output_dir, '{}_golden_ratio_significance.png'.format(name.lower()))
    plot_golden_ratio_results(
        gr_pairs, powers_l1, powers_l2, p_value, phi_optimality, sim_corrs, actual_corr,
        'Golden Ratio Significance Test: {} CMB Data'.format(name), plot_path
    )
    
    # Save results to file
    results_path = os.path.join(output_dir, '{}_golden_ratio_significance.txt'.format(name.lower()))
    with open(results_path, 'w') as f:
        f.write('Golden Ratio Significance Test Results: {} CMB Data\n'.format(name))
        f.write('=' * 60 + '\n\n')
        f.write('Correlation Coefficient: {:.6f}\n'.format(actual_corr))
        f.write('P-value: {:.6f}\n'.format(p_value))
        f.write('Phi-Optimality: {:.6f}\n'.format(phi_optimality))
        f.write('Significant: {}\n\n'.format(p_value < 0.05))
        
        f.write('Golden Ratio Pairs Found: {}\n\n'.format(len(gr_pairs)))
        
        if len(gr_pairs) > 0:
            f.write('Multipole Pairs (l, l*GR):\n')
            for i, (l1, l2) in enumerate(gr_pairs):
                f.write('  Pair {}: l1 = {}, l2 = {} (ratio = {:.4f})\n'.format(
                    i+1, l1, l2, l2/l1))
        
        f.write('\nAnalysis performed on: {}\n'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write('Number of simulations: {}\n'.format(n_simulations))
        f.write('Maximum multipole considered: {}\n'.format(max_l))
    
    print('{} Golden Ratio Significance Test Results:'.format(name))
    print('  Correlation Coefficient: {:.6f}'.format(actual_corr))
    print('  P-value: {:.6f}'.format(p_value))
    print('  Phi-Optimality: {:.6f}'.format(phi_optimality))
    print('  Significant: {}'.format(p_value < 0.05))
    print('  Golden Ratio Pairs Found: {}'.format(len(gr_pairs)))
    
    return {
        'correlation': actual_corr,
        'p_value': p_value,
        'phi_optimality': phi_optimality,
        'significant': p_value < 0.05,
        'gr_pairs': gr_pairs,
        'powers_l1': powers_l1,
        'powers_l2': powers_l2
    }


def compare_results(wmap_results, planck_results, output_dir):
    """
    Compare golden ratio significance test results between WMAP and Planck data.
    
    Args:
        wmap_results (dict): Results from WMAP analysis
        planck_results (dict): Results from Planck analysis
        output_dir (str): Directory to save comparison results
    """
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate differences
        corr_diff = abs(wmap_results['correlation'] - planck_results['correlation'])
        phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
        
        # Save comparison to file
        comparison_path = os.path.join(output_dir, 'golden_ratio_significance_comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write('Golden Ratio Significance Test Comparison: WMAP vs Planck\n')
            f.write('=' * 60 + '\n\n')
            
            f.write('WMAP Correlation: {:.6f}\n'.format(wmap_results['correlation']))
            f.write('WMAP P-value: {:.6f}\n'.format(wmap_results['p_value']))
            f.write('WMAP Phi-Optimality: {:.6f}\n'.format(wmap_results['phi_optimality']))
            f.write('WMAP Significant: {}\n'.format(wmap_results['significant']))
            f.write('WMAP Golden Ratio Pairs: {}\n\n'.format(len(wmap_results['gr_pairs'])))
            
            f.write('Planck Correlation: {:.6f}\n'.format(planck_results['correlation']))
            f.write('Planck P-value: {:.6f}\n'.format(planck_results['p_value']))
            f.write('Planck Phi-Optimality: {:.6f}\n'.format(planck_results['phi_optimality']))
            f.write('Planck Significant: {}\n'.format(planck_results['significant']))
            f.write('Planck Golden Ratio Pairs: {}\n\n'.format(len(planck_results['gr_pairs'])))
            
            f.write('Difference in Correlation: {:.6f}\n'.format(corr_diff))
            f.write('Difference in Phi-Optimality: {:.6f}\n'.format(phi_diff))
        
        # Create comparison plot
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Bar chart of correlation and phi-optimality
            metrics = ['Correlation', 'Phi-Optimality']
            wmap_values = [wmap_results['correlation'], wmap_results['phi_optimality']]
            planck_values = [planck_results['correlation'], planck_results['phi_optimality']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, wmap_values, width, label='WMAP', color='blue', alpha=0.7)
            ax1.bar(x + width/2, planck_values, width, label='Planck', color='red', alpha=0.7)
            
            ax1.set_ylabel('Value')
            ax1.set_title('Golden Ratio Significance: WMAP vs Planck')
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
            
            # Plot 2: Scatter plot of golden ratio pairs
            if len(wmap_results['powers_l1']) > 0:
                ax2.scatter(wmap_results['powers_l1'], wmap_results['powers_l2'], 
                           c='blue', alpha=0.7, label='WMAP')
            
            if len(planck_results['powers_l1']) > 0:
                ax2.scatter(planck_results['powers_l1'], planck_results['powers_l2'], 
                           c='red', alpha=0.7, label='Planck')
            
            ax2.set_title('Golden Ratio Multipole Correlations')
            ax2.set_xlabel('Power at l')
            ax2.set_ylabel('Power at l * GR')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            comparison_plot_path = os.path.join(output_dir, 'golden_ratio_significance_comparison.png')
            plt.savefig(comparison_plot_path)
            plt.close()
        except Exception as e:
            print("Warning: Error in creating comparison plot: {}".format(str(e)))
            print("Continuing with analysis...")
        
        print("\nComparison Results:")
        print("  Difference in Correlation: {:.6f}".format(corr_diff))
        print("  Difference in Phi-Optimality: {:.6f}".format(phi_diff))
        print("  Comparison saved to: {}".format(comparison_path))
    except Exception as e:
        print("Error in comparing results: {}".format(str(e)))
        print("Continuing with analysis...")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Golden Ratio Significance Test on WMAP and Planck data')
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--n-simulations', type=int, default=10000, 
                        help='Number of simulations for Monte Carlo. Default: 10000')
    parser.add_argument('--max-l', type=int, default=2500,
                        help='Maximum multipole moment to consider. Default: 2500')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results. Default: results/golden_ratio_significance_TIMESTAMP')
    parser.add_argument('--num-processes', type=int, default=multiprocessing.cpu_count(),
                        help='Number of processes to use for parallel computation (default: all available cores)')
    
    args = parser.parse_args()
    
    # Define base directory for the COSMIC_Analysis project
    cosmic_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Path to data files - updated to use new directory structure
    wmap_file = os.path.join(cosmic_dir, 'data', 'wmap', 'wmap_tt_spectrum_9yr_v5.txt')
    planck_file = os.path.join(cosmic_dir, 'data', 'planck', 'planck_tt_spectrum_2018.txt')
    
    # Check if files exist
    if not args.planck_only and not os.path.exists(wmap_file):
        print("Error: WMAP power spectrum file not found: {}".format(wmap_file))
        return 1
    
    if not args.wmap_only and not os.path.exists(planck_file):
        print("Error: Planck power spectrum file not found: {}".format(planck_file))
        return 1
    
    # Create output directory with timestamp
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(cosmic_dir, 'results', "golden_ratio_significance_{}".format(timestamp))
    
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
        
        # Run golden ratio significance test on WMAP data
        print("Running golden ratio significance test on WMAP data...")
        wmap_results = run_golden_ratio_test(
            wmap_ell, 
            wmap_processed, 
            os.path.join(output_dir, 'wmap'), 
            'WMAP', 
            n_simulations=args.n_simulations,
            max_l=args.max_l,
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
        
        # Run golden ratio significance test on Planck data
        print("Running golden ratio significance test on Planck data...")
        planck_results = run_golden_ratio_test(
            planck_ell, 
            planck_processed, 
            os.path.join(output_dir, 'planck'), 
            'Planck',
            n_simulations=args.n_simulations,
            max_l=args.max_l,
            num_processes=args.num_processes
        )
    
    # Compare results if both datasets were analyzed
    if wmap_results and planck_results:
        print("Comparing WMAP and Planck golden ratio significance test results...")
        compare_results(wmap_results, planck_results, os.path.join(output_dir, 'comparison'))
    
    print("\nGolden ratio significance test complete. Results saved to: {}".format(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
