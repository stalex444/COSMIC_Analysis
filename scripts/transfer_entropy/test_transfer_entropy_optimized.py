#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Add Python 2.7 compatibility
from __future__ import division, print_function

"""
Optimized Transfer Entropy Test for WMAP and Planck CMB data.

This script implements the Transfer Entropy Test, which measures information flow
between different scales in the CMB power spectrum. Optimized for handling 10,000+
Monte Carlo simulations efficiently.
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
import gc  # Garbage collection
import psutil  # For memory monitoring (install with: pip install psutil)
import json  # For data configuration

# Check if running on Python 2
PY2 = sys.version_info[0] == 2

if PY2:
    # Python 2 compatibility
    range = xrange

# Data configuration management
def load_data_config(config_file='data_config.json'):
    """Load data configuration from JSON file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Data configuration file not found at {config_path}")
            return None
    except Exception as e:
        print(f"Error loading data configuration: {str(e)}")
        return None


def find_file_in_directories(filename, directories):
    """Search for a file in multiple directories."""
    for directory in directories:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return filepath
            
    # If not found in specified directories, try looking in the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    if os.path.exists(filepath):
        return filepath
        
    return None


def get_data_path(data_key, config=None):
    """Get data file path using configuration."""
    # If configuration provided, use it
    if config is not None:
        if data_key in config:
            path = config[data_key]['power_spectrum']
            if os.path.exists(path):
                return path
    
    # Default fallbacks for WMAP and Planck data
    if data_key == 'wmap_data':
        filenames = ['wmap_tt_spectrum_9yr_v5.txt', "../data/wmap_tt_spectrum_9yr_v5.txt"]
    elif data_key == 'planck_data':
        filenames = ['planck_tt_spectrum_2018.txt', "../data/planck_tt_spectrum_2018.txt"]
    else:
        return None
    
    # Search for the file in possible locations
    directories = []
    if config is not None and 'data_directories' in config:
        directories.extend(config['data_directories'])
    
    # Add script directory and common data directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directories.extend([
        script_dir,
        os.path.join(script_dir, 'data'),
        os.path.join(script_dir, 'wmap_data'),
        os.path.join(script_dir, 'planck_data')
    ])
    
    # Try each filename in the directories
    for filename in filenames:
        filepath = find_file_in_directories(filename, directories)
        if filepath is not None:
            return filepath
    
    return None


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
    Optimized version for faster calculations.
    
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
    
    # Calculate transfer entropy using a more efficient approach
    # Create mask for non-zero probabilities to avoid unnecessary calculations
    mask = (stf_joint_prob > epsilon) & (np.take(st_joint_prob.flatten(), np.ravel_multi_index(np.mgrid[:actual_bins, :actual_bins], (actual_bins, actual_bins))).reshape(actual_bins, actual_bins, 1) > epsilon)
    
    # Only use relevant slices of tf_joint_prob where needed
    tf_joint_prob_expanded = np.zeros((actual_bins, actual_bins, actual_bins)) + epsilon
    for j in range(actual_bins):
        for k in range(actual_bins):
            tf_joint_prob_expanded[:, j, k] = tf_joint_prob[j, k]
    
    # Use vectorized operations where possible
    log_terms = np.zeros_like(stf_joint_prob)
    valid_indices = np.where(mask & (tf_joint_prob_expanded > epsilon))
    if len(valid_indices[0]) > 0:
        i, j, k = valid_indices
        st_probs = st_joint_prob[i, j]
        tf_probs = tf_joint_prob[j, k]
        stf_probs = stf_joint_prob[i, j, k]
        log_terms[i, j, k] = stf_probs * np.log2(stf_probs * tf_probs / (st_probs * tf_probs))
    
    # Sum up the values
    te = np.sum(log_terms)
    
    return te


def _run_simulation(sim_index, data, scales, bins, delay, chunk_size=100, tracking_freq=10):
    """
    Run a single simulation for transfer entropy calculation.
    
    Args:
        sim_index (int): Simulation index
        data (numpy.ndarray): Input data array
        scales (int): Number of scales to analyze
        bins (int): Number of bins for discretization
        delay (int): Time delay for information transfer
        chunk_size (int): Process in chunks for memory efficiency 
        tracking_freq (int): How often to report progress
        
    Returns:
        float: Average transfer entropy for this simulation
    """
    # Progress tracking for large batches
    if sim_index % tracking_freq == 0:
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / 1024 / 1024  # RAM usage in MB
        sys.stdout.write(f"\rSimulation {sim_index} - RAM: {mem_usage:.1f} MB")
        sys.stdout.flush()
    
    # Create random permutation of the data
    np.random.seed(42 + sim_index)  # Ensure reproducibility but different for each sim
    sim_data = np.random.permutation(data)
    
    # Split into scales
    scale_size = len(data) // scales
    sim_scale_data = [sim_data[i*scale_size:(i+1)*scale_size] for i in range(scales)]
    
    # Calculate transfer entropy
    sim_te_values = []
    
    # Process in smaller batches of scale pairs to reduce memory pressure
    scale_pairs = [(s1, s2) for s1 in range(scales) for s2 in range(scales) if s1 != s2]
    
    for chunk_start in range(0, len(scale_pairs), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(scale_pairs))
        chunk_pairs = scale_pairs[chunk_start:chunk_end]
        
        for s1, s2 in chunk_pairs:
            try:
                te = calculate_transfer_entropy(sim_scale_data[s1], sim_scale_data[s2], bins=bins, delay=delay)
                sim_te_values.append(te)
            except Exception as e:
                # Gracefully handle errors in individual calculations
                print(f"\nError in calculation for scales {s1},{s2} in simulation {sim_index}: {str(e)}")
                sim_te_values.append(0.0)  # Use neutral value to allow process to continue
    
    # Clean up to reduce memory usage
    del sim_data, sim_scale_data
    gc.collect()
    
    return np.mean(sim_te_values) if sim_te_values else 0.0


def run_monte_carlo_parallel(data, scales=5, n_simulations=100, bins=10, delay=1, timeout=3600, 
                           parallel=True, num_processes=None, batch_size=10):
    """
    Run Monte Carlo simulations in parallel to assess the significance of transfer entropy.
    
    Args:
        data (numpy.ndarray): Input data array
        scales (int): Number of scales to analyze
        n_simulations (int): Number of simulations
        bins (int): Number of bins for discretization
        delay (int): Time delay for information transfer
        timeout (int): Maximum execution time in seconds (0 means no timeout)
        parallel (bool): Whether to run simulations in parallel
        num_processes (int): Number of processes to use for parallelization
        batch_size (int): Number of simulations to run in each batch
        
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
                
                # Check timeout - only if it's not 0 (no timeout)
                if timeout > 0 and time.time() - start_time > timeout:
                    print("Timeout exceeded during actual data calculation.")
                    return 1.0, 0.0, 0.0, [], scale_pairs, te_values
    
    # Calculate average transfer entropy
    actual_te = np.mean(te_values)
    
    # Set up parallel processing
    if num_processes is None:
        # Use slightly fewer processes than available cores to avoid system overload
        available_cores = multiprocessing.cpu_count()
        num_processes = max(1, int(available_cores * 0.8))
    
    print(f"Running {n_simulations} Monte Carlo simulations in parallel using {num_processes} processes...")
    
    # Process in batches to manage memory usage
    sim_tes = []
    
    # Track statistics as we go to avoid storing all results
    exceeds_actual_count = 0
    sum_values = 0
    sum_squares = 0
    
    # Process in smaller batches for memory efficiency
    for batch_start in range(0, n_simulations, batch_size):
        batch_end = min(batch_start + batch_size, n_simulations)
        batch_size_actual = batch_end - batch_start
        
        # Display progress information
        elapsed = time.time() - start_time
        progress = batch_start / n_simulations
        if progress > 0:
            est_total = elapsed / progress
            est_remaining = est_total - elapsed
            print(f"\nBatch {batch_start//batch_size + 1}/{(n_simulations+batch_size-1)//batch_size}: Simulations {batch_start}-{batch_end-1}")
            print(f"Progress: {progress*100:.1f}% - Elapsed: {elapsed/60:.1f} min - Est. Remaining: {est_remaining/60:.1f} min")
            
            # Show memory usage
            process = psutil.Process(os.getpid())
            print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        # Check for timeout (if not disabled with 0)
        if timeout > 0 and time.time() - start_time > timeout:
            print("Timeout exceeded during simulation. Using results collected so far.")
            break
        
        # Create a fresh pool for each batch to avoid memory leaks
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                batch_indices = range(batch_start, batch_end)
                
                # Request simulations with a timeout to avoid hanging
                batch_results = pool.map(
                    partial(_run_simulation, data=data, scales=scales, bins=bins, delay=delay),
                    batch_indices
                )
                
                # Process batch results
                for result in batch_results:
                    sim_tes.append(result)  # Keep all values for histogram plot
                    
                    # Update running statistics
                    if result >= actual_te:
                        exceeds_actual_count += 1
                    sum_values += result
                    sum_squares += result * result
                
                # Force cleanup after batch
                del batch_results
                gc.collect()
        
        except Exception as e:
            print(f"\nError in batch {batch_start//batch_size + 1}: {str(e)}")
            traceback.print_exc()
            # Continue with next batch
    
    # Calculate final statistics using running totals
    n_completed = len(sim_tes)
    if n_completed == 0:
        print("No simulations completed successfully.")
        return 1.0, 0.0, 0.0, [], scale_pairs, te_values
    
    p_value = exceeds_actual_count / n_completed
    sim_mean = sum_values / n_completed
    
    # Calculate standard deviation using the computational formula
    sim_std = np.sqrt((sum_squares / n_completed) - (sim_mean * sim_mean)) if n_completed > 1 else 0
    
    # Calculate phi-optimality (scaled between -1 and 1)
    if sim_std == 0:
        phi_optimality = 0
    else:
        z_score = (actual_te - sim_mean) / sim_std
        # Convert z-score to a value between -1 and 1 using a sigmoid-like function
        phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
    
    print(f"\nCompleted {n_completed} simulations in {(time.time() - start_time)/60:.2f} minutes")
    print(f"p-value: {p_value:.6f}, phi-optimality: {phi_optimality:.6f}")
    
    return p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values


def run_transfer_entropy_test(data, output_dir, name, n_simulations=30, scales=5, bins=10, delay=1, timeout=3600, 
                           parallel=True, num_processes=None, batch_size=10):
    """Run transfer entropy test on the provided data."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Monte Carlo simulations
    print(f"Running Transfer Entropy Test on {name} data...")
    print(f"Number of simulations: {n_simulations}")
    
    # Save the actual data parameters to allow picking up where we left off if needed
    params_file = os.path.join(output_dir, f"{name.lower()}_params.npz")
    np.savez(params_file, data=data, n_simulations=n_simulations, scales=scales, bins=bins, delay=delay)
    
    if parallel:
        print(f"Using parallel processing with {'all available' if num_processes is None else str(num_processes)} processes")
        p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values = run_monte_carlo_parallel(
            data, scales=scales, n_simulations=n_simulations, bins=bins, delay=delay, 
            timeout=timeout, num_processes=num_processes, batch_size=batch_size
        )
    else:
        # For sequential processing, we'll just run the parallel version with 1 process
        print("Using sequential processing")
        p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values = run_monte_carlo_parallel(
            data, scales=scales, n_simulations=n_simulations, bins=bins, delay=delay, 
            timeout=timeout, num_processes=1, batch_size=batch_size
        )
    
    # Save detailed simulation results (conserve memory by using numpy's efficient format)
    sim_results_file = os.path.join(output_dir, f"{name.lower()}_simulation_results.npz")
    np.savez_compressed(
        sim_results_file,
        actual_te=actual_te,
        sim_tes=np.array(sim_tes),
        p_value=p_value,
        phi_optimality=phi_optimality,
        scale_pairs=np.array(scale_pairs),
        te_values=np.array(te_values)
    )
    
    # Plot results
    plot_path = os.path.join(output_dir, f'{name.lower()}_transfer_entropy.png')
    try:
        plot_transfer_entropy_results(
            scale_pairs, te_values, p_value, phi_optimality, sim_tes, actual_te,
            f"{name} Transfer Entropy Analysis", plot_path
        )
    except Exception as e:
        print(f"Warning: Error in plotting results: {str(e)}")
        print("Continuing with analysis...")
    
    # Save results as a dictionary for further use
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
        # Don't attempt to plot if no display is available
        import matplotlib
        matplotlib.use('Agg')  # Force non-interactive backend
        
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
            ax1.set_xticklabels([f'Scale {i+1}' for i in range(scales)])
            ax1.set_yticklabels([f'Scale {i+1}' for i in range(scales)])
            ax1.set_xlabel('Target Scale')
            ax1.set_ylabel('Source Scale')
        
        ax1.set_title('Transfer Entropy Between Scales')
        
        # Add average transfer entropy
        ax1.text(0.05, 0.95, f'Avg. Transfer Entropy = {actual_te:.4f}', 
                transform=ax1.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot simulation results
        if len(sim_tes) > 0:
            # For very large numbers of simulations, use smaller bin sizes
            num_bins = min(30, max(10, len(sim_tes)//200))
            ax2.hist(sim_tes, bins=num_bins, 
                    alpha=0.7, color='gray', label='Random Simulations')
            ax2.axvline(actual_te, color='r', linestyle='--', linewidth=2, 
                       label=f'Actual TE: {actual_te:.4f}')
        
        ax2.set_title('Monte Carlo Simulations')
        ax2.set_xlabel('Average Transfer Entropy')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add text with results
        plt.figtext(0.5, 0.01, f'P-value: {p_value:.6f} | Phi-Optimality: {phi_optimality:.6f}', 
                   ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plt.savefig(output_path)
        plt.close(fig)  # Explicitly close the figure to free memory
    except Exception as e:
        print(f"Warning: Error in plotting transfer entropy results: {str(e)}")
        traceback.print_exc()
        print("Continuing with analysis...")


def compare_results(wmap_results, planck_results, output_dir):
    """Compare transfer entropy test results between WMAP and Planck data."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate differences
        te_diff = abs(wmap_results['actual_te'] - planck_results['actual_te'])
        phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
        
        # Save comparison to file
        comparison_path = os.path.join(output_dir, 'transfer_entropy_comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write('Transfer Entropy Test Comparison: WMAP vs Planck\n')
            f.write('=' * 50 + '\n\n')
            
            f.write(f"WMAP Average Transfer Entropy: {wmap_results['actual_te']:.6f}\n")
            f.write(f"WMAP P-value: {wmap_results['p_value']:.6f}\n")
            f.write(f"WMAP Phi-Optimality: {wmap_results['phi_optimality']:.6f}\n")
            f.write(f"WMAP Significant: {wmap_results['significant']}\n\n")
            
            f.write(f"Planck Average Transfer Entropy: {planck_results['actual_te']:.6f}\n")
            f.write(f"Planck P-value: {planck_results['p_value']:.6f}\n")
            f.write(f"Planck Phi-Optimality: {planck_results['phi_optimality']:.6f}\n")
            f.write(f"Planck Significant: {planck_results['significant']}\n\n")
            
            f.write(f"Difference in Transfer Entropy: {te_diff:.6f}\n")
            f.write(f"Difference in Phi-Optimality: {phi_diff:.6f}\n")
        
        # Create comparison plot
        try:
            # Don't attempt to plot if no display is available
            import matplotlib
            matplotlib.use('Agg')  # Force non-interactive backend
            
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
                        f"p={wmap_results['p_value']:.6f}", 
                        ha='center', va='bottom', color='blue', fontweight='bold')
                ax1.text(i + width/2, planck_values[i] + 0.02, 
                        f"p={planck_results['p_value']:.6f}", 
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
                ax2.set_xticklabels([f'Scale {i+1}' for i in range(scales)])
                ax2.set_yticklabels([f'Scale {i+1}' for i in range(scales)])
                ax2.set_xlabel('Target Scale')
                ax2.set_ylabel('Source Scale')
            
            ax2.set_title('Transfer Entropy Difference')
            
            plt.tight_layout()
            
            # Save the plot
            comparison_plot_path = os.path.join(output_dir, 'transfer_entropy_comparison.png')
            plt.savefig(comparison_plot_path)
            plt.close(fig)  # Explicitly close to free memory
        except Exception as e:
            print(f"Warning: Error in creating comparison plot: {str(e)}")
            print("Continuing with analysis...")
        
        print("\nComparison Results:")
        print(f"  Difference in Transfer Entropy: {te_diff:.6f}")
        print(f"  Difference in Phi-Optimality: {phi_diff:.6f}")
        print(f"  Comparison saved to: {comparison_path}")
    except Exception as e:
        print(f"Error in comparing results: {str(e)}")
        traceback.print_exc()
        print("Continuing with analysis...")


def memory_monitor(interval=30):
    """
    Print memory usage statistics at regular intervals.
    
    Args:
        interval (int): Interval in seconds between memory checks
    """
    while True:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"\nMemory Usage: {mem_info.rss / 1024 / 1024:.1f} MB at {datetime.now().strftime('%H:%M:%S')}")
        sys.stdout.flush()
        time.sleep(interval)


def resume_simulations(results_file, data, scales, bins, delay, n_simulations, timeout=3600, 
                      num_processes=None, batch_size=10):
    """
    Resume monte carlo simulations from previously saved results.
    
    Args:
        results_file (str): Path to saved results file
        data (numpy.ndarray): Input data array
        scales (int): Number of scales to analyze
        bins (int): Number of bins for discretization
        delay (int): Time delay for transfer entropy
        n_simulations (int): Target number of simulations (total)
        timeout (int): Maximum execution time in seconds
        num_processes (int): Number of processes to use
        batch_size (int): Number of simulations to process in each batch
        
    Returns:
        tuple: Updated results
    """
    try:
        # Load previous results
        prev_results = np.load(results_file)
        actual_te = float(prev_results['actual_te'])
        sim_tes = list(prev_results['sim_tes'])
        p_value = float(prev_results['p_value'])
        phi_optimality = float(prev_results['phi_optimality'])
        scale_pairs = list(prev_results['scale_pairs'])
        te_values = list(prev_results['te_values'])
        
        # Calculate how many simulations we still need
        sims_done = len(sim_tes)
        sims_needed = max(0, n_simulations - sims_done)
        
        if sims_needed == 0:
            print(f"Already completed {sims_done} simulations, no need to resume.")
            return p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values
        
        print(f"Resuming simulations: {sims_done} already done, {sims_needed} more needed.")
        
        # Setup parallel processing
        if num_processes is None:
            available_cores = multiprocessing.cpu_count()
            num_processes = max(1, int(available_cores * 0.8))
        
        start_time = time.time()
        
        # Set up tracking variables
        exceeds_actual_count = sum(1 for te in sim_tes if te >= actual_te)
        sum_values = sum(sim_tes)
        sum_squares = sum(te * te for te in sim_tes)
        
        # Process in smaller batches for memory efficiency
        for batch_start in range(0, sims_needed, batch_size):
            batch_end = min(batch_start + batch_size, sims_needed)
            batch_size_actual = batch_end - batch_start
            
            # Display progress information
            elapsed = time.time() - start_time
            progress = batch_start / sims_needed
            if progress > 0:
                est_total = elapsed / progress
                est_remaining = est_total - elapsed
                print(f"\nBatch {batch_start//batch_size + 1}/{(sims_needed+batch_size-1)//batch_size}: Simulations {batch_start}-{batch_end-1}")
                print(f"Progress: {progress*100:.1f}% - Elapsed: {elapsed/60:.1f} min - Est. Remaining: {est_remaining/60:.1f} min")
                
                # Show memory usage
                process = psutil.Process(os.getpid())
                print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
            
            # Check for timeout
            if timeout > 0 and time.time() - start_time > timeout:
                print("Timeout exceeded during simulation. Using results collected so far.")
                break
            
            # Create a fresh pool for each batch to avoid memory leaks
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    # We need to offset the simulation indices by sims_done
                    batch_indices = range(sims_done + batch_start, sims_done + batch_end)
                    
                    batch_results = pool.map(
                        partial(_run_simulation, data=data, scales=scales, bins=bins, delay=delay),
                        batch_indices
                    )
                    
                    # Process batch results
                    for result in batch_results:
                        sim_tes.append(result)
                        
                        # Update statistics
                        if result >= actual_te:
                            exceeds_actual_count += 1
                        sum_values += result
                        sum_squares += result * result
                    
                    # Force cleanup
                    del batch_results
                    gc.collect()
            
            except Exception as e:
                print(f"\nError in batch {batch_start//batch_size + 1}: {str(e)}")
                traceback.print_exc()
                # Continue with next batch
        
        # Recalculate statistics
        n_completed = len(sim_tes)
        if n_completed == 0:
            return p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values
        
        p_value = exceeds_actual_count / n_completed
        sim_mean = sum_values / n_completed
        
        # Calculate standard deviation
        sim_std = np.sqrt((sum_squares / n_completed) - (sim_mean * sim_mean)) if n_completed > 1 else 0
        
        # Recalculate phi-optimality
        if sim_std == 0:
            phi_optimality = 0
        else:
            z_score = (actual_te - sim_mean) / sim_std
            phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
        
        print(f"\nCompleted {n_completed} total simulations (including {sims_done} from previous run)")
        print(f"p-value: {p_value:.6f}, phi-optimality: {phi_optimality:.6f}")
        
        return p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values
    
    except Exception as e:
        print(f"Error resuming simulations: {str(e)}")
        traceback.print_exc()
        return None


def main():
    """Main function to run transfer entropy test on CMB data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimized Transfer Entropy Test for WMAP and Planck CMB data')
    
    # Data selection parameters
    parser.add_argument('--wmap', action='store_true', help='Run test on WMAP data')
    parser.add_argument('--planck', action='store_true', help='Run test on Planck data')
    parser.add_argument('--wmap-file', type=str, default="../data/wmap_tt_spectrum_9yr_v5.txt", 
                      help='Path to WMAP power spectrum file')
    parser.add_argument('--planck-file', type=str, default="../data/planck_tt_spectrum_2018.txt", 
                      help='Path to Planck power spectrum file')
    
    # Analysis parameters
    parser.add_argument('--scales', type=int, default=5, help='Number of scales to divide the data into')
    parser.add_argument('--simulations', type=int, default=1000, help='Number of Monte Carlo simulations')
    parser.add_argument('--bins', type=int, default=10, help='Number of bins for discretization')
    parser.add_argument('--delay', type=int, default=1, help='Time delay for transfer entropy')
    
    # Processing parameters
    parser.add_argument('--timeout', type=int, default=3600, 
                      help='Maximum execution time in seconds (0 for no timeout)')
    parser.add_argument('--processes', type=int, default=None, 
                      help='Number of parallel processes (default: 80% of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=10, 
                      help='Number of simulations to process in each batch')
    parser.add_argument('--sequential', action='store_true', 
                      help='Run simulations sequentially (not in parallel)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='results', 
                      help='Directory to save results')
    parser.add_argument('--resume', action='store_true', 
                      help='Resume from previous results if available')
    parser.add_argument('--monitor-memory', action='store_true',
                      help='Start a separate thread to monitor memory usage')
    
    # Preprocessing parameters
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--smooth-window', type=int, default=5, help='Smoothing window size')
    parser.add_argument('--no-normalize', action='store_true', help='Do not normalize the data')
    parser.add_argument('--detrend', action='store_true', help='Remove linear trend from data')
    
    args = parser.parse_args()
    
    # If neither WMAP nor Planck is specified, run both
    if not args.wmap and not args.planck:
        args.wmap = True
        args.planck = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data configuration
    config = load_data_config()
    
    # Log configuration
    log_file = os.path.join(args.output_dir, 'transfer_entropy_config.txt')
    with open(log_file, 'w') as f:
        f.write('Transfer Entropy Test Configuration\n')
        f.write('=' * 40 + '\n\n')
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of scales: {args.scales}\n")
        f.write(f"Number of simulations: {args.simulations}\n")
        f.write(f"Bins: {args.bins}\n")
        f.write(f"Delay: {args.delay}\n")
        f.write(f"Timeout: {args.timeout} seconds\n")
        f.write(f"Parallel processing: {not args.sequential}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Smoothing: {args.smooth}\n")
        if args.smooth:
            f.write(f"Smoothing window: {args.smooth_window}\n")
        f.write(f"Normalization: {not args.no_normalize}\n")
        f.write(f"Detrending: {args.detrend}\n")
    
    # Start memory monitor if requested
    if args.monitor_memory:
        import threading
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
    
    # Process WMAP data
    wmap_results = None
    if args.wmap:
        try:
            # Get WMAP data path using configuration
            wmap_data_path = get_data_path('wmap_data', config)
            if wmap_data_path is None:
                print("Error: WMAP data file not found.")
                return
            
            print(f"Looking for WMAP data at: {wmap_data_path}")
            ell, power, error = load_wmap_power_spectrum(wmap_data_path)
            
            if ell is not None and power is not None:
                # Preprocess WMAP data
                processed_power = preprocess_data(
                    power, 
                    smooth=args.smooth, 
                    smooth_window=args.smooth_window,
                    normalize=not args.no_normalize,
                    detrend=args.detrend
                )
                
                # Check if we should resume from previous results
                wmap_results_file = os.path.join(args.output_dir, 'wmap_simulation_results.npz')
                if args.resume and os.path.exists(wmap_results_file):
                    print("Resuming WMAP simulations from previous results...")
                    wmap_results = resume_simulations(
                        wmap_results_file, processed_power, args.scales, args.bins, args.delay,
                        args.simulations, args.timeout, None if args.sequential else args.processes,
                        args.batch_size
                    )
                    
                    if wmap_results is not None:
                        p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values = wmap_results
                        # Save updated results
                        np.savez_compressed(
                            wmap_results_file,
                            actual_te=actual_te,
                            sim_tes=np.array(sim_tes),
                            p_value=p_value,
                            phi_optimality=phi_optimality,
                            scale_pairs=np.array(scale_pairs),
                            te_values=np.array(te_values)
                        )
                        
                        # Update plot
                        plot_path = os.path.join(args.output_dir, 'wmap_transfer_entropy.png')
                        plot_transfer_entropy_results(
                            scale_pairs, te_values, p_value, phi_optimality, sim_tes, actual_te,
                            'WMAP Transfer Entropy Analysis', plot_path
                        )
                        
                        # Create results dictionary
                        wmap_results = {
                            'p_value': p_value,
                            'phi_optimality': phi_optimality,
                            'actual_te': actual_te,
                            'sim_tes': sim_tes,
                            'scale_pairs': scale_pairs,
                            'te_values': te_values,
                            'significant': p_value < 0.05
                        }
                    else:
                        # If resume failed, run from scratch
                        print("Resume failed, running new WMAP analysis...")
                        wmap_results = run_transfer_entropy_test(
                            processed_power, args.output_dir, 'WMAP',
                            n_simulations=args.simulations,
                            scales=args.scales,
                            bins=args.bins,
                            delay=args.delay,
                            timeout=args.timeout,
                            parallel=not args.sequential,
                            num_processes=args.processes,
                            batch_size=args.batch_size
                        )
                else:
                    # Run fresh analysis
                    wmap_results = run_transfer_entropy_test(
                        processed_power, args.output_dir, 'WMAP',
                        n_simulations=args.simulations,
                        scales=args.scales,
                        bins=args.bins,
                        delay=args.delay,
                        timeout=args.timeout,
                        parallel=not args.sequential,
                        num_processes=args.processes,
                        batch_size=args.batch_size
                    )
            else:
                print("Error loading WMAP data. Skipping WMAP analysis.")
        except Exception as e:
            print(f"Error processing WMAP data: {str(e)}")
            traceback.print_exc()
    
    # Process Planck data
    planck_results = None
    if args.planck:
        try:
            # Get Planck data path using configuration
            planck_data_path = get_data_path('planck_data', config)
            if planck_data_path is None:
                print("Error: Planck data file not found.")
                return
            
            print(f"Looking for Planck data at: {planck_data_path}")
            ell, power, error = load_planck_power_spectrum(planck_data_path)
            
            if ell is not None and power is not None:
                # Preprocess Planck data
                processed_power = preprocess_data(
                    power, 
                    smooth=args.smooth, 
                    smooth_window=args.smooth_window,
                    normalize=not args.no_normalize,
                    detrend=args.detrend
                )
                
                # Check if we should resume from previous results
                planck_results_file = os.path.join(args.output_dir, 'planck_simulation_results.npz')
                if args.resume and os.path.exists(planck_results_file):
                    print("Resuming Planck simulations from previous results...")
                    planck_results = resume_simulations(
                        planck_results_file, processed_power, args.scales, args.bins, args.delay,
                        args.simulations, args.timeout, None if args.sequential else args.processes,
                        args.batch_size
                    )
                    
                    if planck_results is not None:
                        p_value, phi_optimality, actual_te, sim_tes, scale_pairs, te_values = planck_results
                        # Save updated results
                        np.savez_compressed(
                            planck_results_file,
                            actual_te=actual_te,
                            sim_tes=np.array(sim_tes),
                            p_value=p_value,
                            phi_optimality=phi_optimality,
                            scale_pairs=np.array(scale_pairs),
                            te_values=np.array(te_values)
                        )
                        
                        # Update plot
                        plot_path = os.path.join(args.output_dir, 'planck_transfer_entropy.png')
                        plot_transfer_entropy_results(
                            scale_pairs, te_values, p_value, phi_optimality, sim_tes, actual_te,
                            'Planck Transfer Entropy Analysis', plot_path
                        )
                        
                        # Create results dictionary
                        planck_results = {
                            'p_value': p_value,
                            'phi_optimality': phi_optimality,
                            'actual_te': actual_te,
                            'sim_tes': sim_tes,
                            'scale_pairs': scale_pairs,
                            'te_values': te_values,
                            'significant': p_value < 0.05
                        }
                    else:
                        # If resume failed, run from scratch
                        print("Resume failed, running new Planck analysis...")
                        planck_results = run_transfer_entropy_test(
                            processed_power, args.output_dir, 'Planck',
                            n_simulations=args.simulations,
                            scales=args.scales,
                            bins=args.bins,
                            delay=args.delay,
                            timeout=args.timeout,
                            parallel=not args.sequential,
                            num_processes=args.processes,
                            batch_size=args.batch_size
                        )
                else:
                    # Run fresh analysis
                    planck_results = run_transfer_entropy_test(
                        processed_power, args.output_dir, 'Planck',
                        n_simulations=args.simulations,
                        scales=args.scales,
                        bins=args.bins,
                        delay=args.delay,
                        timeout=args.timeout,
                        parallel=not args.sequential,
                        num_processes=args.processes,
                        batch_size=args.batch_size
                    )
            else:
                print("Error loading Planck data. Skipping Planck analysis.")
        except Exception as e:
            print(f"Error processing Planck data: {str(e)}")
            traceback.print_exc()
    
    # Compare results if both analyses were performed
    if wmap_results is not None and planck_results is not None:
        try:
            compare_results(wmap_results, planck_results, args.output_dir)
        except Exception as e:
            print(f"Error comparing results: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
