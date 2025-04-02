#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Add Python 2.7 compatibility
from __future__ import division, print_function

"""
Transfer Entropy Test for WMAP and Planck CMB data.

This script implements the Transfer Entropy Test, which measures information flow
between different scales in the CMB power spectrum.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import argparse
import time
import traceback
import multiprocessing
from functools import partial

# Check if running on Python 2
PY2 = sys.version_info[0] == 2

if PY2:
    # Python 2 compatibility
    range = xrange

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_wmap_power_spectrum(file_path):
    """Load WMAP CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Mean multipole moment
        power = data[:, 3]  # Power spectrum value
        error = data[:, 4]  # Error
        return ell, power, error
    except Exception as e:
        print("Error loading WMAP power spectrum: {}".format(str(e)))
        return None, None, None


def load_planck_power_spectrum(file_path):
    """Load Planck CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Multipole moment
        power = data[:, 1]  # Power spectrum value
        # Use average of asymmetric error bars as the error
        lower_error = data[:, 2]  # Lower error bound
        upper_error = data[:, 3]  # Upper error bound
        error = (abs(lower_error) + abs(upper_error)) / 2.0
        return ell, power, error
    except Exception as e:
        print("Error loading Planck power spectrum: {}".format(str(e)))
        return None, None, None


def preprocess_data(data, smooth=False, smooth_window=5, normalize=True, detrend=False):
    """Preprocess data for analysis."""
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


def calculate_transfer_entropy(source, target, bins=10, delay=1, max_points=500):
    """
    Calculate transfer entropy from source to target time series.
    
    Transfer entropy measures the amount of information transfer from source to target.
    
    Args:
        source (numpy.ndarray): Source time series
        target (numpy.ndarray): Target time series
        bins (int): Number of bins for discretization
        delay (int): Time delay for information transfer
        max_points (int): Maximum number of points to use for calculation
        
    Returns:
        float: Transfer entropy value
    """
    # Ensure arrays are the same length and limit size for performance
    length = min(len(source), len(target), max_points)
    source = source[:length]
    target = target[:length]
    
    # Create delayed versions
    target_past = target[:-delay]
    target_future = target[delay:]
    source_past = source[:-delay]
    
    # Use fewer bins if we have limited data points
    actual_bins = min(bins, max(3, length // 10))
    
    # Discretize the data using binning
    s_bins = np.linspace(min(source_past), max(source_past), actual_bins+1)
    t_bins = np.linspace(min(target_past), max(target_past), actual_bins+1)
    tf_bins = np.linspace(min(target_future), max(target_future), actual_bins+1)
    
    # Ensure discretized values are within bounds (0 to bins-1)
    s_disc = np.clip(np.digitize(source_past, s_bins) - 1, 0, actual_bins-1)
    t_disc = np.clip(np.digitize(target_past, t_bins) - 1, 0, actual_bins-1)
    tf_disc = np.clip(np.digitize(target_future, tf_bins) - 1, 0, actual_bins-1)
    
    # Use numpy's histogram2d and histogramdd for faster joint probability calculation
    st_joint_counts, _, _ = np.histogram2d(s_disc, t_disc, bins=[actual_bins, actual_bins])
    tf_joint_counts, _, _ = np.histogram2d(t_disc, tf_disc, bins=[actual_bins, actual_bins])
    
    # For 3D histogram, we need to reshape the data
    stf_data = np.vstack([s_disc, t_disc, tf_disc]).T
    stf_joint_counts, _ = np.histogramdd(stf_data, bins=[actual_bins, actual_bins, actual_bins])
    
    # Normalize to get probabilities (add small epsilon to avoid division by zero)
    epsilon = 1e-10
    st_joint_prob = st_joint_counts / (np.sum(st_joint_counts) + epsilon)
    stf_joint_prob = stf_joint_counts / (np.sum(stf_joint_counts) + epsilon)
    tf_joint_prob = tf_joint_counts / (np.sum(tf_joint_counts) + epsilon)
    
    # Calculate transfer entropy using vectorized operations where possible
    te = 0
    for i in range(actual_bins):
        for j in range(actual_bins):
            for k in range(actual_bins):
                if stf_joint_prob[i, j, k] > epsilon and st_joint_prob[i, j] > epsilon and tf_joint_prob[j, k] > epsilon:
                    te += stf_joint_prob[i, j, k] * np.log2(stf_joint_prob[i, j, k] * tf_joint_prob[j, k] / 
                                                      (st_joint_prob[i, j] * tf_joint_prob[j, k]))
    
    return te


def _run_simulation(sim_index, data, scales, bins, delay):
    """
    Run a single simulation for transfer entropy calculation.
    
    Args:
        sim_index (int): Simulation index
        data (numpy.ndarray): Input data array
        scales (int): Number of scales to analyze
        bins (int): Number of bins for discretization
        delay (int): Time delay for information transfer
        
    Returns:
        float: Average transfer entropy for this simulation
    """
    # Create random permutation of the data
    sim_data = np.random.permutation(data)
    
    # Split into scales
    scale_size = len(data) // scales
    sim_scale_data = [sim_data[i*scale_size:(i+1)*scale_size] for i in range(scales)]
    
    # Calculate transfer entropy
    sim_te_values = []
    for s1 in range(scales):
        for s2 in range(scales):
            if s1 != s2:
                te = calculate_transfer_entropy(sim_scale_data[s1], sim_scale_data[s2], bins=bins, delay=delay)
                sim_te_values.append(te)
    
    return np.mean(sim_te_values)


def run_monte_carlo_parallel(data, scales=5, n_simulations=100, bins=10, delay=1, timeout=3600, num_processes=None):
    """
    Run Monte Carlo simulations in parallel to assess the significance of transfer entropy.
    
    Args:
        data (numpy.ndarray): Input data array
        scales (int): Number of scales to analyze
        n_simulations (int): Number of simulations
        bins (int): Number of bins for discretization
        delay (int): Time delay for information transfer
        timeout (int): Maximum execution time in seconds
        num_processes (int): Number of processes to use for parallelization
        
    Returns:
        tuple: (p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values)
    """
    start_time = time.time()
    
    # Split data into scales
    scale_size = len(data) // scales
    scale_data = [data[i*scale_size:(i+1)*scale_size] for i in range(scales)]
    
    # Calculate transfer entropy between all pairs of scales
    scale_pairs = []
    te_values = []
    
    print("Calculating transfer entropy for actual data...")
    for i in range(scales):
        for j in range(scales):
            if i != j:
                scale_pairs.append((i, j))
                te = calculate_transfer_entropy(scale_data[i], scale_data[j], bins=bins, delay=delay)
                te_values.append(te)
                
                # Check timeout
                if timeout > 0 and time.time() - start_time > timeout:
                    print("Timeout exceeded during actual data calculation.")
                    return 1.0, 0.0, 0.0, [], scale_pairs, te_values
    
    # Calculate average transfer entropy
    actual_te = np.mean(te_values)
    
    # Set up parallel processing
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print("Running %d Monte Carlo simulations in parallel using %d processes..." % (n_simulations, num_processes))
    
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Run simulations in parallel
    try:
        sim_tes = pool.map(
            partial(_run_simulation, data=data, scales=scales, bins=bins, delay=delay),
            range(n_simulations)
        )
    except Exception as e:
        print("Error during parallel simulation: %s" % str(e))
        traceback.print_exc()
        return 1.0, 0.0, actual_te, [], scale_pairs, te_values
    finally:
        pool.close()
        pool.join()
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_te else 0 for sim in sim_tes])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    sim_mean = np.mean(sim_tes)
    sim_std = np.std(sim_tes) if len(sim_tes) > 1 else 0
    if sim_std == 0:
        phi_optimality = 0
    else:
        z_score = (actual_te - sim_mean) / sim_std
        # Convert z-score to a value between -1 and 1 using a sigmoid-like function
        phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
    
    print("Completed %d simulations in %.2f seconds" % (len(sim_tes), time.time() - start_time))
    print("p-value: %.4f, phi-optimality: %.4f" % (p_value, phi_optimality))
    
    return p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values


def run_transfer_entropy_test(data, output_dir, name, n_simulations=30, scales=5, bins=10, delay=1, timeout=3600, parallel=False, num_processes=None):
    """Run transfer entropy test on the provided data."""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run Monte Carlo simulations
    print("Running Transfer Entropy Test on %s data..." % name)
    print("Number of simulations: %d" % n_simulations)
    
    if parallel:
        print("Using parallel processing with %s processes" % ("all available" if num_processes is None else str(num_processes)))
        p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values = run_monte_carlo_parallel(
            data, scales=scales, n_simulations=n_simulations, bins=bins, delay=delay, 
            timeout=timeout, num_processes=num_processes
        )
    else:
        print("Using sequential processing")
        p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values = run_monte_carlo_parallel(
            data, scales=scales, n_simulations=n_simulations, bins=bins, delay=delay, timeout=timeout
        )
    
    # Plot results
    plot_path = os.path.join(output_dir, '%s_transfer_entropy.png' % name.lower())
    plot_transfer_entropy_results(
        scale_pairs, te_values, p_value, phi_optimality, sim_tes, actual_te,
        "%s Transfer Entropy Analysis" % name, plot_path
    )
    
    # Save results
    results = {
        'p_value': p_value,
        'phi_optimality': phi_optimality,
        'actual_te': actual_te,
        'sim_tes': sim_tes,
        'scale_pairs': scale_pairs,
        'te_values': te_values,
        'significant': p_value < 0.05
    }
    
    return results


def plot_transfer_entropy_results(scale_pairs, te_values, p_value, phi_optimality, 
                                 sim_tes, actual_te, title, output_path):
    """Plot transfer entropy analysis results."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot transfer entropy between scales
        if len(scale_pairs) > 0 and len(te_values) > 0:
            # Create a matrix representation
            scales = max(max(pair) for pair in scale_pairs) + 1
            te_matrix = np.zeros((scales, scales))
            
            for (i, j), te in zip(scale_pairs, te_values):
                te_matrix[i, j] = te
            
            im = ax1.imshow(te_matrix, cmap='viridis')
            plt.colorbar(im, ax=ax1, label='Transfer Entropy')
            
            # Add labels
            ax1.set_xticks(range(scales))
            ax1.set_yticks(range(scales))
            ax1.set_xticklabels(['Scale %d' % (i+1) for i in range(scales)])
            ax1.set_yticklabels(['Scale %d' % (i+1) for i in range(scales)])
            ax1.set_xlabel('Target Scale')
            ax1.set_ylabel('Source Scale')
        
        ax1.set_title('Transfer Entropy Between Scales')
        
        # Add average transfer entropy
        ax1.text(0.05, 0.95, 'Avg. Transfer Entropy = %.4f' % actual_te, 
                transform=ax1.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot simulation results
        if len(sim_tes) > 0:
            ax2.hist(sim_tes, bins=min(30, len(sim_tes)//3), 
                    alpha=0.7, color='gray', label='Random Simulations')
            ax2.axvline(actual_te, color='r', linestyle='--', linewidth=2, 
                       label='Actual TE: %.4f' % actual_te)
        
        ax2.set_title('Monte Carlo Simulations')
        ax2.set_xlabel('Average Transfer Entropy')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True)
        
        # Add text with results
        plt.figtext(0.5, 0.01, 'P-value: %.4f | Phi-Optimality: %.4f' % (p_value, phi_optimality), 
                   ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print("Warning: Error in plotting transfer entropy results: %s" % str(e))
        print("Continuing with analysis...")


def compare_results(wmap_results, planck_results, output_dir):
    """Compare transfer entropy test results between WMAP and Planck data."""
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate differences
        te_diff = abs(wmap_results['actual_te'] - planck_results['actual_te'])
        phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
        
        # Save comparison to file
        comparison_path = os.path.join(output_dir, 'transfer_entropy_comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write('Transfer Entropy Test Comparison: WMAP vs Planck\n')
            f.write('=' * 50 + '\n\n')
            
            f.write('WMAP Average Transfer Entropy: %.6f\n' % wmap_results['actual_te'])
            f.write('WMAP P-value: %.6f\n' % wmap_results['p_value'])
            f.write('WMAP Phi-Optimality: %.6f\n' % wmap_results['phi_optimality'])
            f.write('WMAP Significant: %s\n\n' % wmap_results['significant'])
            
            f.write('Planck Average Transfer Entropy: %.6f\n' % planck_results['actual_te'])
            f.write('Planck P-value: %.6f\n' % planck_results['p_value'])
            f.write('Planck Phi-Optimality: %.6f\n' % planck_results['phi_optimality'])
            f.write('Planck Significant: %s\n\n' % planck_results['significant'])
            
            f.write('Difference in Transfer Entropy: %.6f\n' % te_diff)
            f.write('Difference in Phi-Optimality: %.6f\n' % phi_diff)
        
        # Create comparison plot
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Bar chart of transfer entropy and phi-optimality
            metrics = ['Transfer Entropy', 'Phi-Optimality']
            wmap_values = [wmap_results['actual_te'], wmap_results['phi_optimality']]
            planck_values = [planck_results['actual_te'], planck_results['phi_optimality']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, wmap_values, width, label='WMAP', color='blue', alpha=0.7)
            ax1.bar(x + width/2, planck_values, width, label='Planck', color='red', alpha=0.7)
            
            ax1.set_ylabel('Value')
            ax1.set_title('Transfer Entropy: WMAP vs Planck')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            
            # Add text with p-values
            for i, metric in enumerate(metrics):
                ax1.text(i - width/2, wmap_values[i] + 0.02, 
                        'p=%.4f' % wmap_results["p_value"], 
                        ha='center', va='bottom', color='blue', fontweight='bold')
                ax1.text(i + width/2, planck_values[i] + 0.02, 
                        'p=%.4f' % planck_results["p_value"], 
                        ha='center', va='bottom', color='red', fontweight='bold')
            
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot 2: Heatmap comparison
            # Create a combined heatmap showing the difference between WMAP and Planck
            if len(wmap_results['scale_pairs']) > 0 and len(planck_results['scale_pairs']) > 0:
                scales = max(max(pair) for pair in wmap_results['scale_pairs']) + 1
                wmap_matrix = np.zeros((scales, scales))
                planck_matrix = np.zeros((scales, scales))
                
                for (i, j), te in zip(wmap_results['scale_pairs'], wmap_results['te_values']):
                    wmap_matrix[i, j] = te
                
                for (i, j), te in zip(planck_results['scale_pairs'], planck_results['te_values']):
                    planck_matrix[i, j] = te
                
                diff_matrix = wmap_matrix - planck_matrix
                
                im = ax2.imshow(diff_matrix, cmap='coolwarm')
                plt.colorbar(im, ax=ax2, label='WMAP - Planck TE')
                
                # Add labels
                ax2.set_xticks(range(scales))
                ax2.set_yticks(range(scales))
                ax2.set_xticklabels(['Scale %d' % (i+1) for i in range(scales)])
                ax2.set_yticklabels(['Scale %d' % (i+1) for i in range(scales)])
                ax2.set_xlabel('Target Scale')
                ax2.set_ylabel('Source Scale')
            
            ax2.set_title('Transfer Entropy Difference')
            
            plt.tight_layout()
            
            # Save the plot
            comparison_plot_path = os.path.join(output_dir, 'transfer_entropy_comparison.png')
            plt.savefig(comparison_plot_path)
            plt.close()
        except Exception as e:
            print("Warning: Error in creating comparison plot: %s" % str(e))
            print("Continuing with analysis...")
        
        print("\nComparison Results:")
        print("  Difference in Transfer Entropy: %.6f" % te_diff)
        print("  Difference in Phi-Optimality: %.6f" % phi_diff)
        print("  Comparison saved to: %s" % comparison_path)
    except Exception as e:
        print("Error in comparing results: %s" % str(e))
        print("Continuing with analysis...")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Transfer Entropy Test on WMAP and Planck data')
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--n-simulations', type=int, default=30, 
                        help='Number of simulations for Monte Carlo. Default: 30')
    parser.add_argument('--scales', type=int, default=5,
                        help='Number of scales to analyze. Default: 5')
    parser.add_argument('--bins', type=int, default=10,
                        help='Number of bins for discretization. Default: 10')
    parser.add_argument('--delay', type=int, default=1,
                        help='Time delay for information transfer. Default: 1')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results. Default: results/transfer_entropy_TIMESTAMP')
    parser.add_argument('--timeout', type=int, default=3600, 
                        help='Timeout in seconds for the analysis. Default: 3600 (1 hour)')
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate visualizations of the results')
    parser.add_argument('--parallel', action='store_true', help='Run Monte Carlo simulations in parallel')
    parser.add_argument('--num-processes', type=int, default=None, 
                        help='Number of processes to use for parallel computation (default: all available cores)')

    args = parser.parse_args()
    
    # Print start time and parameters
    start_time = time.time()
    print("Starting Transfer Entropy Test at %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Parameters:")
    print("  - Number of simulations: %d" % args.n_simulations)
    print("  - Number of scales: %d" % args.scales)
    print("  - Number of bins: %d" % args.bins)
    print("  - Delay: %d" % args.delay)
    print("  - Smooth: %s" % args.smooth)
    print("  - Detrend: %s" % args.detrend)
    print("  - Timeout: %d seconds" % args.timeout)
    print("  - Visualize: %s" % args.visualize)
    print("  - Parallel: %s" % args.parallel)
    print("  - Number of processes: %s" % ("all available" if args.num_processes is None else str(args.num_processes)))
    
    # Path to data files
    wmap_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/wmap/wmap_tt_spectrum_9yr_v5.txt')
    planck_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/planck/planck_tt_spectrum_2018.txt')
    
    # Check if files exist
    if not args.planck_only and not os.path.exists(wmap_file):
        print("Error: WMAP power spectrum file not found: %s" % wmap_file)
        return 1
    
    if not args.wmap_only and not os.path.exists(planck_file):
        print("Error: Planck power spectrum file not found: %s" % planck_file)
        print("Please make sure the Planck data is available in the Cosmic_Consciousness_Analysis repository.")
        return 1
    
    # Create output directory with timestamp
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('results', "transfer_entropy_%s" % timestamp)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory: %s" % output_dir)
    
    # Initialize results dictionaries
    wmap_results = None
    planck_results = None
    
    # Process WMAP data if requested
    if not args.planck_only:
        print("\n==================================================")
        print("Processing WMAP data")
        print("==================================================")
        print("Time remaining: %.1f seconds" % (args.timeout - (time.time() - start_time)))
        
        # Load WMAP data
        print("Loading WMAP power spectrum...")
        wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(wmap_file)
        
        if wmap_ell is None:
            print("Error loading WMAP power spectrum.")
            return 1
        
        print("Loaded WMAP power spectrum with %d multipoles" % len(wmap_ell))
        
        # Preprocess WMAP data
        print("Preprocessing WMAP data...")
        wmap_processed = preprocess_data(
            wmap_power, 
            smooth=args.smooth, 
            smooth_window=5, 
            normalize=True, 
            detrend=args.detrend
        )
        
        # Run transfer entropy test on WMAP data
        print("Running transfer entropy test on WMAP data...")
        start_time_wmap = time.time()
        try:
            wmap_results = run_transfer_entropy_test(
                wmap_processed, 
                os.path.join(output_dir, 'wmap'), 
                'WMAP', 
                n_simulations=args.n_simulations,
                scales=args.scales,
                bins=args.bins,
                delay=args.delay,
                timeout=args.timeout,
                parallel=args.parallel,
                num_processes=args.num_processes
            )
            
            # Generate visualizations if requested
            if args.visualize and wmap_results:
                print("Generating WMAP visualizations...")
                plot_transfer_entropy_results(
                    wmap_results["scale_pairs"], 
                    wmap_results["te_values"], 
                    wmap_results["p_value"], 
                    wmap_results["phi_optimality"], 
                    wmap_results["sim_tes"], 
                    wmap_results["actual_te"], 
                    'WMAP Transfer Entropy Analysis', 
                    os.path.join(output_dir, 'wmap_transfer_entropy.png')
                )
                
        except Exception as e:
            print("Error running transfer entropy test on WMAP data: %s" % str(e))
            traceback.print_exc()
            if args.timeout > 0 and time.time() - start_time > args.timeout:
                print("Timeout exceeded. Stopping analysis.")
                return 1
    
    # Process Planck data if requested
    if not args.wmap_only:
        print("\n==================================================")
        print("Processing Planck data")
        print("==================================================")
        print("Time remaining: %.1f seconds" % (args.timeout - (time.time() - start_time)))
        
        # Load Planck data
        print("Loading Planck power spectrum...")
        planck_ell, planck_power, planck_error = load_planck_power_spectrum(planck_file)
        
        if planck_ell is None:
            print("Error loading Planck power spectrum.")
            return 1
        
        print("Loaded Planck power spectrum with %d multipoles" % len(planck_ell))
        
        # Preprocess Planck data
        print("Preprocessing Planck data...")
        planck_processed = preprocess_data(
            planck_power, 
            smooth=args.smooth, 
            smooth_window=5, 
            normalize=True, 
            detrend=args.detrend
        )
        
        # Run transfer entropy test on Planck data
        print("Running transfer entropy test on Planck data...")
        start_time_planck = time.time()
        try:
            planck_results = run_transfer_entropy_test(
                planck_processed, 
                os.path.join(output_dir, 'planck'), 
                'Planck',
                n_simulations=args.n_simulations,
                scales=args.scales,
                bins=args.bins,
                delay=args.delay,
                timeout=args.timeout,
                parallel=args.parallel,
                num_processes=args.num_processes
            )
            
            # Generate visualizations if requested
            if args.visualize and planck_results:
                print("Generating Planck visualizations...")
                plot_transfer_entropy_results(
                    planck_results["scale_pairs"], 
                    planck_results["te_values"], 
                    planck_results["p_value"], 
                    planck_results["phi_optimality"], 
                    planck_results["sim_tes"], 
                    planck_results["actual_te"], 
                    'Planck Transfer Entropy Analysis', 
                    os.path.join(output_dir, 'planck_transfer_entropy.png')
                )
                
        except Exception as e:
            print("Error running transfer entropy test on Planck data: %s" % str(e))
            traceback.print_exc()
            if args.timeout > 0 and time.time() - start_time_planck > args.timeout:
                print("Timeout exceeded. Stopping analysis.")
                return 1
    
    # Compare results if both datasets were analyzed
    if wmap_results and planck_results:
        print("\n==================================================")
        print("Comparing WMAP and Planck results")
        print("==================================================")
        try:
            compare_results(wmap_results, planck_results, os.path.join(output_dir, 'comparison'))
        except Exception as e:
            print("Error comparing results: %s" % str(e))
            traceback.print_exc()
    
    # Write results to file
    print("\n==================================================")
    print("Writing results to file")
    print("==================================================")
    results_file = os.path.join(output_dir, 'transfer_entropy_results.txt')
    with open(results_file, 'w') as f:
        f.write("Transfer Entropy Test Results\n")
        f.write("===========================\n\n")
        
        if wmap_results:
            f.write("WMAP Results:\n")
            f.write("  - p-value: %.4f\n" % wmap_results["p_value"])
            f.write("  - phi-optimality: %.4f\n" % wmap_results["phi_optimality"])
            f.write("  - actual transfer entropy: %.4f\n" % wmap_results["actual_te"])
            f.write("\n")
        
        if planck_results:
            f.write("Planck Results:\n")
            f.write("  - p-value: %.4f\n" % planck_results["p_value"])
            f.write("  - phi-optimality: %.4f\n" % planck_results["phi_optimality"])
            f.write("  - actual transfer entropy: %.4f\n" % planck_results["actual_te"])
            f.write("\n")
    
    print("Results written to: %s" % results_file)
    print("\nTotal execution time: %.2f seconds" % (time.time() - start_time))
    print("Results saved to: %s" % output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
