#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Information Integration Test for WMAP and Planck CMB data.

This script implements the Information Integration Test, which measures mutual information
between adjacent spectrum regions in the CMB power spectrum. The test is applied to both
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
from sklearn.feature_selection import mutual_info_regression
import multiprocessing
from functools import partial

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


def create_segments(data, n_segments=10):
    """
    Divide data into segments for mutual information analysis.
    
    Args:
        data (numpy.ndarray): Input data array
        n_segments (int): Number of segments to create
        
    Returns:
        list: List of data segments
    """
    segment_size = len(data) // n_segments
    segments = []
    
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(data)
        segments.append(data[start:end])
    
    return segments


def compute_mutual_information(segments):
    """
    Compute mutual information between adjacent segments.
    
    Args:
        segments (list): List of data segments
        
    Returns:
        tuple: (mutual_information_values, mean_mutual_information)
    """
    mi_values = []
    
    for i in range(len(segments) - 1):
        # Get the current and next segment
        current_segment = segments[i]
        next_segment = segments[i + 1]
        
        # Reshape for mutual_info_regression
        X = current_segment.reshape(-1, 1)
        y = next_segment
        
        # Ensure the arrays have the same length
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        # Compute mutual information
        if min_len > 1:
            mi = mutual_info_regression(X, y, random_state=42)[0]
            mi_values.append(mi)
    
    # Calculate mean mutual information
    mean_mi = np.mean(mi_values) if mi_values else 0
    
    return mi_values, mean_mi


def run_monte_carlo(data, n_simulations=1000, n_segments=10):
    """
    Run Monte Carlo simulations to assess the significance of mutual information.
    
    Args:
        data (numpy.ndarray): Input data array
        n_simulations (int): Number of simulations
        n_segments (int): Number of segments
        
    Returns:
        tuple: (p_value, phi_optimality, actual_mi, sim_mis, mi_values)
    """
    # Compute actual mutual information
    actual_segments = create_segments(data, n_segments)
    mi_values, actual_mi = compute_mutual_information(actual_segments)
    
    # Run simulations
    sim_mis = []
    for _ in range(n_simulations):
        # Create random permutation of the data
        sim_data = np.random.permutation(data)
        sim_segments = create_segments(sim_data, n_segments)
        _, sim_mi = compute_mutual_information(sim_segments)
        sim_mis.append(sim_mi)
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_mi else 0 for sim in sim_mis])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    sim_mean = np.mean(sim_mis)
    sim_std = np.std(sim_mis)
    if sim_std == 0:
        phi_optimality = 0
    else:
        phi_optimality = (actual_mi - sim_mean) / sim_std
        # Scale to -1 to 1 range
        phi_optimality = np.tanh(phi_optimality)
    
    return p_value, phi_optimality, actual_mi, sim_mis, mi_values


def run_monte_carlo_parallel(data, n_simulations=1000, n_segments=10, num_processes=None):
    """
    Run Monte Carlo simulations in parallel to assess the significance of mutual information.
    
    Args:
        data (numpy.ndarray): Input data array
        n_simulations (int): Number of simulations
        n_segments (int): Number of segments
        num_processes (int): Number of processes to use for parallelization
        
    Returns:
        tuple: (p_value, phi_optimality, actual_mi, sim_mis, mi_values)
    """
    # If num_processes is not specified, use all available cores
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Compute actual mutual information
    actual_segments = create_segments(data, n_segments)
    mi_values, actual_mi = compute_mutual_information(actual_segments)
    
    # For Python 2 compatibility, we'll use a simpler approach without Pool
    sim_mis = []
    print("Running {} Monte Carlo simulations...".format(n_simulations))
    
    # Create chunks of simulations for each process
    chunk_size = max(1, min(1000, n_simulations // num_processes))  # Limit chunk size for better load balancing
    chunks = [range(i, min(i + chunk_size, n_simulations)) 
              for i in range(0, n_simulations, chunk_size)]
    
    # Define a worker function that processes a chunk of simulations
    def process_chunk(chunk_indices):
        chunk_results = []
        for i in chunk_indices:
            # Create random permutation of the data
            sim_data = np.random.permutation(data)
            sim_segments = create_segments(sim_data, n_segments)
            _, sim_mi = compute_mutual_information(sim_segments)
            chunk_results.append(sim_mi)
            
            # Report progress every 100 simulations
            if i % 100 == 0 and i > 0:
                print("  Process completed {} simulations".format(i))
        return chunk_results
    
    # Create and start processes
    processes = []
    result_queue = multiprocessing.Queue()
    
    for i, chunk in enumerate(chunks):
        process = multiprocessing.Process(
            target=lambda q, chunk_idx: q.put((chunk_idx, process_chunk(chunk))),
            args=(result_queue, i)
        )
        processes.append(process)
        process.start()
    
    # Collect results
    results = [None] * len(chunks)
    completed_chunks = 0
    
    while completed_chunks < len(chunks):
        chunk_idx, chunk_results = result_queue.get()
        results[chunk_idx] = chunk_results
        completed_chunks += 1
        
        # Print progress
        if completed_chunks % max(1, len(chunks) // 10) == 0:
            print("Completed {}/{} chunks ({:.1f}%)".format(
                completed_chunks, len(chunks), 100.0 * completed_chunks / len(chunks)))
    
    # Wait for all processes to finish
    for process in processes:
        process.join()
    
    # Flatten results
    sim_mis = [mi for chunk in results if chunk is not None for mi in chunk]
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_mi else 0 for sim in sim_mis])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    sim_mean = np.mean(sim_mis)
    sim_std = np.std(sim_mis)
    if sim_std == 0:
        phi_optimality = 0
    else:
        phi_optimality = (actual_mi - sim_mean) / sim_std
        # Scale to -1 to 1 range
        phi_optimality = np.tanh(phi_optimality)
    
    print("Final results after {} simulations:".format(len(sim_mis)))
    print("  p-value: {:.6f}".format(p_value))
    print("  phi-optimality: {:.6f}".format(phi_optimality))
    
    return p_value, phi_optimality, actual_mi, sim_mis, mi_values


def plot_information_integration_results(mi_values, p_value, phi_optimality, 
                                        sim_mis, actual_mi, 
                                        title, output_path):
    """
    Plot information integration results.
    
    Args:
        mi_values (list): Mutual information values between adjacent segments
        p_value (float): P-value from Monte Carlo simulations
        phi_optimality (float): Phi-optimality metric
        sim_mis (list): Mutual information values from simulations
        actual_mi (float): Actual mean mutual information
        title (str): Plot title
        output_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot mutual information between adjacent segments
    segments = list(range(1, len(mi_values) + 1))
    ax1.plot(segments, mi_values, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Mutual Information Between Adjacent Segments')
    ax1.set_xlabel('Segment Pair')
    ax1.set_ylabel('Mutual Information')
    ax1.grid(True)
    
    # Plot simulation results
    ax2.hist(sim_mis, bins=30, alpha=0.7, color='gray', label='Random Simulations')
    ax2.axvline(actual_mi, color='r', linestyle='--', linewidth=2, 
               label='Actual Mean MI: {:.4f}'.format(actual_mi))
    ax2.set_title('Monte Carlo Simulations')
    ax2.set_xlabel('Mean Mutual Information')
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


def run_information_integration_test(data, output_dir, name, n_simulations=1000, n_segments=10, parallel=True, num_processes=None):
    """
    Run information integration test on the provided data.
    
    Args:
        data (numpy.ndarray): Data to analyze
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        n_simulations (int): Number of simulations for Monte Carlo
        n_segments (int): Number of segments
        parallel (bool): Whether to use parallel processing
        num_processes (int): Number of processes to use for parallelization
        
    Returns:
        dict: Analysis results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Start timing
    start_time = datetime.now()
    
    # Run Monte Carlo simulations
    print("Running {} Monte Carlo simulations for {} data...".format(n_simulations, name))
    
    if parallel:
        p_value, phi_optimality, actual_mi, sim_mis, mi_values = run_monte_carlo_parallel(
            data, n_simulations=n_simulations, n_segments=n_segments, num_processes=num_processes)
    else:
        p_value, phi_optimality, actual_mi, sim_mis, mi_values = run_monte_carlo(
            data, n_simulations=n_simulations, n_segments=n_segments)
    
    # Calculate execution time
    execution_time = (datetime.now() - start_time).total_seconds()
    print("Execution time: {:.2f} seconds".format(execution_time))
    
    # Plot results
    plot_path = os.path.join(output_dir, '{}_information_integration.png'.format(name.lower()))
    plot_information_integration_results(
        mi_values, p_value, phi_optimality, sim_mis, actual_mi,
        'Information Integration Test: {} CMB Data'.format(name), plot_path
    )
    
    # Save results to file
    results_path = os.path.join(output_dir, '{}_information_integration.txt'.format(name.lower()))
    with open(results_path, 'w') as f:
        f.write('Information Integration Test Results: {} CMB Data\n'.format(name))
        f.write('=' * 60 + '\n\n')
        f.write('Number of segments: {}\n'.format(n_segments))
        f.write('Number of simulations: {}\n'.format(n_simulations))
        f.write('Mean Mutual Information: {:.6f}\n'.format(actual_mi))
        f.write('P-value: {:.6f}\n'.format(p_value))
        f.write('Phi-Optimality: {:.6f}\n'.format(phi_optimality))
        f.write('Execution time: {:.2f} seconds\n'.format(execution_time))
        
        # Determine significance
        significant = p_value < 0.05
        f.write('\nStatistical Significance: {}\n'.format('Significant' if significant else 'Not Significant'))
        
        if significant:
            f.write('The mutual information between adjacent segments is significantly higher than expected by chance.\n')
            f.write('This suggests structured information integration across scales in the CMB power spectrum.\n')
        else:
            f.write('The mutual information between adjacent segments is not significantly different from random chance.\n')
            f.write('This suggests no structured information integration across scales in the CMB power spectrum.\n')
    
    # Return results dictionary
    results = {
        'name': name,
        'n_segments': n_segments,
        'n_simulations': n_simulations,
        'mutual_information': mi_values,
        'mean_mutual_information': actual_mi,
        'p_value': p_value,
        'phi_optimality': phi_optimality,
        'significant': p_value < 0.05,
        'execution_time': execution_time
    }
    
    return results


def compare_results(wmap_results, planck_results, output_dir):
    """
    Compare information integration test results between WMAP and Planck data.
    
    Args:
        wmap_results (dict): Results from WMAP analysis
        planck_results (dict): Results from Planck analysis
        output_dir (str): Directory to save comparison results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate differences
    mi_diff = abs(wmap_results['mean_mutual_information'] - planck_results['mean_mutual_information'])
    phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
    
    # Save comparison to file
    comparison_path = os.path.join(output_dir, 'information_integration_comparison.txt')
    with open(comparison_path, 'w') as f:
        f.write('Information Integration Test Comparison: WMAP vs Planck\n')
        f.write('=' * 60 + '\n\n')
        
        f.write('WMAP Mean Mutual Information: {:.6f}\n'.format(wmap_results['mean_mutual_information']))
        f.write('WMAP P-value: {:.6f}\n'.format(wmap_results['p_value']))
        f.write('WMAP Phi-Optimality: {:.6f}\n'.format(wmap_results['phi_optimality']))
        f.write('WMAP Significant: {}\n\n'.format(wmap_results['significant']))
        
        f.write('Planck Mean Mutual Information: {:.6f}\n'.format(planck_results['mean_mutual_information']))
        f.write('Planck P-value: {:.6f}\n'.format(planck_results['p_value']))
        f.write('Planck Phi-Optimality: {:.6f}\n'.format(planck_results['phi_optimality']))
        f.write('Planck Significant: {}\n\n'.format(planck_results['significant']))
        
        f.write('Difference in Mean Mutual Information: {:.6f}\n'.format(mi_diff))
        f.write('Difference in Phi-Optimality: {:.6f}\n'.format(phi_diff))
        
        # Compare segment MI values if available
        if ('segment_mi_values' in wmap_results and 'segment_mi_values' in planck_results and
            len(wmap_results['segment_mi_values']) == len(planck_results['segment_mi_values'])):
            
            f.write('\nSegment-by-Segment Mutual Information Comparison:\n')
            for i in range(len(wmap_results['segment_mi_values'])):
                wmap_mi = wmap_results['segment_mi_values'][i]
                planck_mi = planck_results['segment_mi_values'][i]
                diff = abs(wmap_mi - planck_mi)
                
                f.write('  Segments {}-{}:\n'.format(i+1, i+2))
                f.write('    WMAP: {:.6f}\n'.format(wmap_mi))
                f.write('    Planck: {:.6f}\n'.format(planck_mi))
                f.write('    Difference: {:.6f}\n'.format(diff))
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar chart of mean MI and phi-optimality
    metrics = ['Mean Mutual Information', 'Phi-Optimality']
    wmap_values = [wmap_results['mean_mutual_information'], wmap_results['phi_optimality']]
    planck_values = [planck_results['mean_mutual_information'], planck_results['phi_optimality']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, wmap_values, width, label='WMAP', color='blue', alpha=0.7)
    ax1.bar(x + width/2, planck_values, width, label='Planck', color='red', alpha=0.7)
    
    ax1.set_ylabel('Value')
    ax1.set_title('Information Integration: WMAP vs Planck')
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
    
    # Plot 2: Segment MI values comparison
    if ('segment_mi_values' in wmap_results and 'segment_mi_values' in planck_results and
        len(wmap_results['segment_mi_values']) > 0):
        
        segments = list(range(1, len(wmap_results['segment_mi_values']) + 1))
        
        ax2.plot(segments, wmap_results['segment_mi_values'], 'bo-', linewidth=2, 
                label='WMAP', markersize=8)
        ax2.plot(segments, planck_results['segment_mi_values'], 'ro-', linewidth=2, 
                label='Planck', markersize=8)
        
        ax2.set_title('Segment-by-Segment Mutual Information Comparison')
        ax2.set_xlabel('Segment Pair')
        ax2.set_ylabel('Mutual Information')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    comparison_plot_path = os.path.join(output_dir, 'information_integration_comparison.png')
    plt.savefig(comparison_plot_path)
    plt.close()
    
    print("\nComparison Results:")
    print("  Difference in Mean Mutual Information: {:.6f}".format(mi_diff))
    print("  Difference in Phi-Optimality: {:.6f}".format(phi_diff))
    print("  Comparison saved to: {}".format(comparison_path))


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Information Integration Test on WMAP and Planck data')
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--n-simulations', type=int, default=1000, 
                        help='Number of simulations for Monte Carlo. Default: 1000')
    parser.add_argument('--n-segments', type=int, default=10,
                        help='Number of segments for analysis. Default: 10')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results. Default: results/information_integration_TIMESTAMP')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--num-processes', type=int, default=multiprocessing.cpu_count(),
                        help='Number of processes to use for parallel computation (default: all available cores)')
    
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
        output_dir = os.path.join('results', "information_integration_{}".format(timestamp))
    
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
        
        # Run information integration test on WMAP data
        print("Running information integration test on WMAP data...")
        wmap_results = run_information_integration_test(
            wmap_processed, 
            os.path.join(output_dir, 'wmap'), 
            'WMAP', 
            n_simulations=args.n_simulations,
            n_segments=args.n_segments,
            parallel=not args.no_parallel,
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
        
        # Run information integration test on Planck data
        print("Running information integration test on Planck data...")
        planck_results = run_information_integration_test(
            planck_processed, 
            os.path.join(output_dir, 'planck'), 
            'Planck', 
            n_simulations=args.n_simulations,
            n_segments=args.n_segments,
            parallel=not args.no_parallel,
            num_processes=args.num_processes
        )
    
    # Compare results if both datasets were analyzed
    if wmap_results and planck_results:
        print("Comparing WMAP and Planck information integration test results...")
        compare_results(wmap_results, planck_results, os.path.join(output_dir, 'comparison'))
    
    print("\nInformation integration test complete. Results saved to: {}".format(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
