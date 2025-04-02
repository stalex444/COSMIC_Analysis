#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meta-Coherence Test for WMAP and Planck CMB data.

This script implements the Meta-Coherence Test, which analyzes the coherence
of local coherence measures across different scales in the CMB power spectrum.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from datetime import datetime
import argparse
import multiprocessing
from functools import partial
import queue
import traceback
import time

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


def calculate_local_coherence(data, window_size=10, step_size=5):
    """
    Calculate local coherence measures across the data using sliding windows.
    
    Args:
        data (numpy.ndarray): Input data array
        window_size (int): Size of the sliding window
        step_size (int): Step size for sliding the window
        
    Returns:
        tuple: (window_centers, local_coherence_values)
    """
    if len(data) < window_size:
        return [], []
    
    window_centers = []
    local_coherence_values = []
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i+window_size]
        window_center = i + window_size // 2
        window_centers.append(window_center)
        
        # Calculate local coherence as the inverse of the standard deviation
        # of the first differences (higher values indicate more coherence)
        diffs = np.diff(window)
        local_coherence = 1.0 / (np.std(diffs) + 1e-10)  # Add small value to avoid division by zero
        local_coherence_values.append(local_coherence)
    
    return np.array(window_centers), np.array(local_coherence_values)


def calculate_meta_coherence(local_coherence_values):
    """
    Calculate meta-coherence as a measure of how coherent the local coherence values are.
    
    Args:
        local_coherence_values (numpy.ndarray): Array of local coherence values
        
    Returns:
        float: Meta-coherence value
    """
    if len(local_coherence_values) < 2:
        return 0.0
    
    # Calculate meta-coherence as the inverse of the coefficient of variation
    # of the local coherence values (higher values indicate more meta-coherence)
    mean = np.mean(local_coherence_values)
    std = np.std(local_coherence_values)
    
    if mean == 0:
        return 0.0
    
    # Coefficient of variation (CV) = std / mean
    # Meta-coherence = 1 / CV = mean / std
    meta_coherence = mean / (std + 1e-10)  # Add small value to avoid division by zero
    
    # Normalize to [0, 1] range using a sigmoid-like function
    normalized_meta_coherence = 2.0 / (1.0 + np.exp(-0.1 * meta_coherence)) - 1.0
    
    return normalized_meta_coherence


def run_monte_carlo(data, n_simulations=10000, window_size=10, step_size=5):
    """
    Run Monte Carlo simulations to assess the significance of meta-coherence.
    
    Args:
        data (numpy.ndarray): Input data array
        n_simulations (int): Number of simulations
        window_size (int): Size of the sliding window
        step_size (int): Step size for sliding the window
        
    Returns:
        tuple: (p_value, phi_optimality, actual_meta_coherence, sim_meta_coherences, 
                window_centers, local_coherence_values)
    """
    # Calculate actual meta-coherence
    window_centers, local_coherence_values = calculate_local_coherence(
        data, window_size=window_size, step_size=step_size)
    actual_meta_coherence = calculate_meta_coherence(local_coherence_values)
    
    # Run simulations
    sim_meta_coherences = []
    for i in range(n_simulations):
        if i % 10 == 0:
            print("  Simulation {}/{}".format(i, n_simulations))
        # Create random permutation of the data
        sim_data = np.random.permutation(data)
        _, sim_local_coherence = calculate_local_coherence(
            sim_data, window_size=window_size, step_size=step_size)
        sim_meta_coherence = calculate_meta_coherence(sim_local_coherence)
        sim_meta_coherences.append(sim_meta_coherence)
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_meta_coherence else 0 for sim in sim_meta_coherences])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    sim_mean = np.mean(sim_meta_coherences)
    sim_std = np.std(sim_meta_coherences)
    if sim_std == 0:
        phi_optimality = 0
    else:
        z_score = (actual_meta_coherence - sim_mean) / sim_std
        # Convert z-score to a value between -1 and 1 using a sigmoid-like function
        phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
    
    return (p_value, phi_optimality, actual_meta_coherence, sim_meta_coherences, 
            window_centers, local_coherence_values)


def run_monte_carlo_chunk(result_queue, chunk_idx, data, n_simulations, window_size, step_size):
    """
    Run a chunk of Monte Carlo simulations for parallel processing.
    
    Args:
        result_queue: Queue to store results
        chunk_idx: Index of the current chunk
        data: Input data array
        n_simulations: Number of simulations in this chunk
        window_size: Size of the sliding window
        step_size: Step size for sliding the window
    """
    try:
        # Set a unique seed for this process
        # Use a combination of chunk index and current time to ensure uniqueness
        # Python 2.7 compatible timestamp creation
        current_time = int(time.mktime(datetime.now().timetuple()))
        np.random.seed(chunk_idx * 1000 + current_time)
        
        # Calculate actual meta-coherence (same for all simulations)
        window_centers, local_coherence_values = calculate_local_coherence(
            data, window_size=window_size, step_size=step_size)
        actual_meta_coherence = calculate_meta_coherence(local_coherence_values)
        
        # Run simulations for this chunk
        sim_meta_coherences = []
        for i in range(n_simulations):
            # Create random permutation of the data
            sim_data = np.random.permutation(data)
            _, sim_local_coherence = calculate_local_coherence(
                sim_data, window_size=window_size, step_size=step_size)
            sim_meta_coherence = calculate_meta_coherence(sim_local_coherence)
            sim_meta_coherences.append(sim_meta_coherence)
        
        # Put results in the queue
        result_queue.put({
            'chunk_idx': chunk_idx,
            'sim_meta_coherences': sim_meta_coherences,
            'actual_meta_coherence': actual_meta_coherence,
            'window_centers': window_centers,
            'local_coherence_values': local_coherence_values
        })
        
        return True
    except Exception as e:
        print("Error in process {}: {}".format(chunk_idx, str(e)))
        print(traceback.format_exc())
        result_queue.put({
            'chunk_idx': chunk_idx,
            'error': str(e)
        })
        return False


def run_monte_carlo_parallel(data, n_simulations=10000, window_size=10, step_size=5, num_processes=None):
    """
    Run Monte Carlo simulations in parallel to assess the significance of meta-coherence.
    
    Args:
        data (numpy.ndarray): Input data array
        n_simulations (int): Number of simulations
        window_size (int): Size of the sliding window
        step_size (int): Step size for sliding the window
        num_processes (int): Number of processes to use
        
    Returns:
        tuple: (p_value, phi_optimality, actual_meta_coherence, sim_meta_coherences, 
                window_centers, local_coherence_values)
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print("Running {} Monte Carlo simulations using {} processes...".format(
        n_simulations, num_processes))
    
    # Calculate actual meta-coherence
    window_centers, local_coherence_values = calculate_local_coherence(
        data, window_size=window_size, step_size=step_size)
    actual_meta_coherence = calculate_meta_coherence(local_coherence_values)
    
    # Split simulations into chunks
    chunk_size = max(1, n_simulations // num_processes)
    chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = min(start + chunk_size, n_simulations)
        if end > start:
            chunks.append((i, end - start))
    
    # Create a queue for results
    result_queue = multiprocessing.Queue()
    
    # Create and start processes
    processes = []
    for chunk_idx, chunk_n_simulations in chunks:
        process = multiprocessing.Process(
            target=run_monte_carlo_chunk,
            args=(result_queue, chunk_idx, data, chunk_n_simulations, window_size, step_size)
        )
        processes.append(process)
        process.start()
    
    # Collect results
    results = []
    completed_chunks = 0
    total_chunks = len(chunks)
    
    while completed_chunks < total_chunks:
        try:
            result = result_queue.get(timeout=1.0)
            if 'error' in result:
                print("Error in chunk {}: {}".format(result['chunk_idx'], result['error']))
            else:
                results.append(result)
                completed_chunks += 1
                print("  Completed chunk {}/{} ({:.1f}%)".format(
                    completed_chunks, total_chunks, 100.0 * completed_chunks / total_chunks))
        except queue.Empty:
            # Check if any process is still alive
            if not any(p.is_alive() for p in processes):
                print("Warning: All processes finished but not all results collected.")
                break
    
    # Wait for all processes to finish
    for process in processes:
        process.join()
    
    # Flatten results
    sim_meta_coherences = []
    for result in results:
        if 'sim_meta_coherences' in result:
            sim_meta_coherences.extend(result['sim_meta_coherences'])
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_meta_coherence else 0 for sim in sim_meta_coherences])
    
    # Calculate phi-optimality
    sim_mean = np.mean(sim_meta_coherences)
    sim_std = np.std(sim_meta_coherences)
    if sim_std == 0:
        phi_optimality = 0
    else:
        z_score = (actual_meta_coherence - sim_mean) / sim_std
        # Convert z-score to a value between -1 and 1 using a sigmoid-like function
        phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
    
    return (p_value, phi_optimality, actual_meta_coherence, sim_meta_coherences, 
            window_centers, local_coherence_values)


def run_meta_coherence_test(data, output_dir, name, n_simulations=10000, window_size=10, step_size=5, parallel=True, num_processes=None):
    """
    Run meta-coherence test on the provided data.
    
    Args:
        data (numpy.ndarray): Data to analyze
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        n_simulations (int): Number of simulations for Monte Carlo
        window_size (int): Size of the sliding window
        step_size (int): Step size for sliding the window
        parallel (bool): Whether to use parallel processing
        num_processes (int): Number of processes to use for parallelization
        
    Returns:
        dict: Analysis results
    """
    try:
        print("Running meta-coherence test on {} data...".format(name))
        print("Running {} Monte Carlo simulations...".format(n_simulations))
        
        start_time = datetime.now()
        
        # Create output directory for this dataset
        dataset_dir = os.path.join(output_dir, name.lower())
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        # Run Monte Carlo simulations
        if parallel and n_simulations > 10:
            print("Using parallel processing with {} processes".format(
                num_processes if num_processes else multiprocessing.cpu_count()))
            p_value, phi_optimality, actual_meta_coherence, sim_meta_coherences, window_centers, local_coherence = \
                run_monte_carlo_parallel(data, n_simulations=n_simulations, window_size=window_size, 
                                       step_size=step_size, num_processes=num_processes)
        else:
            p_value, phi_optimality, actual_meta_coherence, sim_meta_coherences, window_centers, local_coherence = \
                run_monte_carlo(data, n_simulations=n_simulations, window_size=window_size, step_size=step_size)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Plot results
        plot_meta_coherence_results(
            window_centers, local_coherence, p_value, phi_optimality, 
            sim_meta_coherences, actual_meta_coherence, 
            "{} Meta-Coherence Analysis".format(name), 
            os.path.join(dataset_dir, "{}_meta_coherence.png".format(name.lower())))
        
        # Save results to file
        result_file = os.path.join(dataset_dir, "{}_meta_coherence.txt".format(name.lower()))
        with open(result_file, 'w') as f:
            f.write("Meta-Coherence Test Results: {} CMB Data\n".format(name))
            f.write("============================================================\n\n")
            f.write("Window size: {}\n".format(window_size))
            f.write("Step size: {}\n".format(step_size))
            f.write("Number of simulations: {}\n".format(n_simulations))
            f.write("Meta-Coherence: {:.6f}\n".format(actual_meta_coherence))
            f.write("P-value: {:.6f}\n".format(p_value))
            f.write("Phi-Optimality: {:.6f}\n".format(phi_optimality))
            f.write("Execution time: {:.2f} seconds\n\n".format(execution_time))
            
            f.write("Statistical Significance: {}\n".format("Significant" if p_value < 0.05 else "Not Significant"))
            if p_value < 0.05:
                f.write("The meta-coherence is significantly higher than expected by random chance.\n")
                f.write("This suggests structured coherence patterns across scales in the CMB power spectrum.\n")
            else:
                f.write("The meta-coherence is not significantly different from random chance.\n")
                f.write("This suggests no structured coherence patterns across scales in the CMB power spectrum.\n")
        
        print("Meta-coherence test completed in {:.2f} seconds".format(execution_time))
        print("Results saved to {}".format(result_file))
        
        # Return results
        return {
            'meta_coherence': actual_meta_coherence,
            'p_value': p_value,
            'phi_optimality': phi_optimality,
            'window_centers': window_centers,
            'local_coherence': local_coherence,
            'sim_meta_coherences': sim_meta_coherences,
            'execution_time': execution_time
        }
        
    except Exception as e:
        print("Error in meta-coherence test for {} data: {}".format(name, str(e)))
        print(traceback.format_exc())
        return None


def plot_meta_coherence_results(window_centers, local_coherence, p_value, phi_optimality, 
                               sim_meta_coherences, actual_meta_coherence, title, output_path):
    """Plot meta-coherence analysis results."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot local coherence values
        if len(window_centers) > 0 and len(local_coherence) > 0:
            ax1.plot(window_centers, local_coherence, 'b-', linewidth=2)
            ax1.scatter(window_centers, local_coherence, c='blue', alpha=0.7)
        
        ax1.set_title('Local Coherence Measures')
        ax1.set_xlabel('Window Center Position')
        ax1.set_ylabel('Local Coherence')
        ax1.grid(True)
        
        # Add meta-coherence value
        ax1.text(0.05, 0.95, 'Meta-Coherence = {:.4f}'.format(actual_meta_coherence), 
                transform=ax1.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot simulation results
        if len(sim_meta_coherences) > 0:
            ax2.hist(sim_meta_coherences, bins=min(30, len(sim_meta_coherences)//3), 
                    alpha=0.7, color='gray', label='Random Simulations')
            ax2.axvline(actual_meta_coherence, color='r', linestyle='--', linewidth=2, 
                       label='Actual Meta-Coherence: {:.4f}'.format(actual_meta_coherence))
        
        ax2.set_title('Monte Carlo Simulations')
        ax2.set_xlabel('Meta-Coherence')
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
        print("Warning: Error in plotting meta-coherence results: {}".format(str(e)))
        print("Continuing with analysis...")


def compare_results(wmap_results, planck_results, output_dir):
    """Compare meta-coherence test results between WMAP and Planck data."""
    try:
        # Create comparison directory
        comparison_dir = os.path.join(output_dir, 'comparison')
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)
        
        # Calculate differences
        meta_coherence_diff = abs(wmap_results['meta_coherence'] - planck_results['meta_coherence'])
        phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
        
        # Save comparison to file
        comparison_path = os.path.join(comparison_dir, 'meta_coherence_comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write('Meta-Coherence Test Comparison: WMAP vs Planck\n')
            f.write('=' * 50 + '\n\n')
            
            f.write('WMAP Meta-Coherence: {:.6f}\n'.format(wmap_results['meta_coherence']))
            f.write('WMAP P-value: {:.6f}\n'.format(wmap_results['p_value']))
            f.write('WMAP Phi-Optimality: {:.6f}\n'.format(wmap_results['phi_optimality']))
            f.write('WMAP Significant: {}\n\n'.format(wmap_results['p_value'] < 0.05))
            
            f.write('Planck Meta-Coherence: {:.6f}\n'.format(planck_results['meta_coherence']))
            f.write('Planck P-value: {:.6f}\n'.format(planck_results['p_value']))
            f.write('Planck Phi-Optimality: {:.6f}\n'.format(planck_results['phi_optimality']))
            f.write('Planck Significant: {}\n\n'.format(planck_results['p_value'] < 0.05))
            
            f.write('Difference in Meta-Coherence: {:.6f}\n'.format(meta_coherence_diff))
            f.write('Difference in Phi-Optimality: {:.6f}\n'.format(phi_diff))
            
            f.write('\nExecution Information\n')
            f.write('-' * 50 + '\n')
            f.write('WMAP Execution Time: {:.2f} seconds\n'.format(wmap_results.get('execution_time', 0)))
            f.write('Planck Execution Time: {:.2f} seconds\n'.format(planck_results.get('execution_time', 0)))
            f.write('Total Execution Time: {:.2f} seconds\n'.format(
                wmap_results.get('execution_time', 0) + planck_results.get('execution_time', 0)))
        
        # Create comparison plot
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Bar chart of meta-coherence and phi-optimality
            metrics = ['Meta-Coherence', 'Phi-Optimality']
            wmap_values = [wmap_results['meta_coherence'], wmap_results['phi_optimality']]
            planck_values = [planck_results['meta_coherence'], planck_results['phi_optimality']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, wmap_values, width, label='WMAP', color='blue', alpha=0.7)
            ax1.bar(x + width/2, planck_values, width, label='Planck', color='red', alpha=0.7)
            
            ax1.set_ylabel('Value')
            ax1.set_title('Meta-Coherence: WMAP vs Planck')
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
            
            # Plot 2: Local coherence for both datasets
            # Normalize window centers to compare datasets of different lengths
            if len(wmap_results['window_centers']) > 0:
                wmap_x = wmap_results['window_centers'] / max(wmap_results['window_centers'])
                ax2.plot(wmap_x, wmap_results['local_coherence'], 'b-', alpha=0.7, label='WMAP')
            
            if len(planck_results['window_centers']) > 0:
                planck_x = planck_results['window_centers'] / max(planck_results['window_centers'])
                ax2.plot(planck_x, planck_results['local_coherence'], 'r-', alpha=0.7, label='Planck')
            
            ax2.set_title('Local Coherence Comparison')
            ax2.set_xlabel('Normalized Position')
            ax2.set_ylabel('Local Coherence')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            comparison_plot_path = os.path.join(comparison_dir, 'meta_coherence_comparison.png')
            plt.savefig(comparison_plot_path)
            plt.close()
        except Exception as e:
            print("Warning: Error in creating comparison plot: {}".format(str(e)))
            print("Continuing with analysis...")
        
        print("\nComparison Results:")
        print("  Difference in Meta-Coherence: {:.6f}".format(meta_coherence_diff))
        print("  Difference in Phi-Optimality: {:.6f}".format(phi_diff))
        print("  Comparison saved to: {}".format(comparison_path))
    except Exception as e:
        print("Error in comparing results: {}".format(str(e)))
        print(traceback.format_exc())


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Meta-Coherence Test on WMAP and Planck data')
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--n-simulations', type=int, default=10000, 
                        help='Number of simulations for Monte Carlo. Default: 10000')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Window size for coherence calculation. Default: 10')
    parser.add_argument('--step-size', type=int, default=5,
                        help='Step size for sliding window. Default: 5')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results. Default: results/meta_coherence_TIMESTAMP')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--num-processes', type=int, default=multiprocessing.cpu_count(),
                        help='Number of processes to use for parallel computation (default: all available cores)')
    
    args = parser.parse_args()
    
    # Start timing the execution
    start_time = datetime.now()
    print("Starting Meta-Coherence Test at {}".format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    
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
        output_dir = os.path.join('results', "meta_coherence_{}".format(timestamp))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory: {}".format(output_dir))
    
    # Initialize results dictionaries
    wmap_results = None
    planck_results = None
    
    # Process WMAP data
    if not args.planck_only:
        print("\n" + "="*50)
        print("Processing WMAP data")
        print("="*50)
        
        # Load and preprocess WMAP data
        wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(wmap_file)
        
        if wmap_ell is None:
            print("Error: Failed to load WMAP data")
        else:
            print("Loaded WMAP power spectrum with {} multipoles".format(len(wmap_ell)))
            
            # Preprocess WMAP data
            print("Preprocessing WMAP data...")
            processed_wmap_power = preprocess_data(
                wmap_power, 
                smooth=args.smooth, 
                detrend=args.detrend
            )
            
            # Run meta-coherence test on WMAP data
            wmap_results = run_meta_coherence_test(
                processed_wmap_power, 
                output_dir, 
                "WMAP", 
                n_simulations=args.n_simulations,
                window_size=args.window_size,
                step_size=args.step_size,
                parallel=not args.no_parallel,
                num_processes=args.num_processes
            )
    
    # Process Planck data
    if not args.wmap_only:
        print("\n" + "="*50)
        print("Processing Planck data")
        print("="*50)
        
        # Load and preprocess Planck data
        planck_ell, planck_power, planck_error = load_planck_power_spectrum(planck_file)
        
        if planck_ell is None:
            print("Error: Failed to load Planck data")
        else:
            print("Loaded Planck power spectrum with {} multipoles".format(len(planck_ell)))
            
            # Preprocess Planck data
            print("Preprocessing Planck data...")
            processed_planck_power = preprocess_data(
                planck_power, 
                smooth=args.smooth, 
                detrend=args.detrend
            )
            
            # Run meta-coherence test on Planck data
            planck_results = run_meta_coherence_test(
                processed_planck_power, 
                output_dir, 
                "Planck", 
                n_simulations=args.n_simulations,
                window_size=args.window_size,
                step_size=args.step_size,
                parallel=not args.no_parallel,
                num_processes=args.num_processes
            )
    
    # Compare results if both datasets were analyzed
    if wmap_results is not None and planck_results is not None:
        print("\n" + "="*50)
        print("Comparing WMAP and Planck results")
        print("="*50)
        compare_results(wmap_results, planck_results, output_dir)
    
    # Calculate total execution time
    execution_time = (datetime.now() - start_time).total_seconds()
    print("\nMeta-coherence test completed successfully.")
    print("Total execution time: {:.2f} seconds".format(execution_time))
    print("Results saved to {}".format(output_dir))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
