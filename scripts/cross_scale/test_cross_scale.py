#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-Scale Correlation Test Module.

This test examines correlations between phi-related scales in the CMB power spectrum
and compares them to correlations between random scales to assess statistical significance.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
try:
    import Queue  # Python 2
except ImportError:
    import queue as Queue  # Python 3
import pickle
import argparse
import re
from scipy import stats  # Add missing import for stats module

# Constants related to the golden ratio and other mathematical constants
CONSTANTS = {
    'phi': (1 + np.sqrt(5)) / 2,  # Golden ratio
    'e': np.e,
    'pi': np.pi,
    'sqrt2': np.sqrt(2),
    'sqrt3': np.sqrt(3),
}

def load_power_spectrum(filename):
    """
    Load power spectrum data from file.
    
    Args:
        filename (str): Path to the power spectrum file
        
    Returns:
        tuple: (ell, power) arrays
    """
    try:
        data = np.loadtxt(filename)
        ell = data[:, 0]
        power = data[:, 1]
        return ell, power
    except Exception as e:
        print("Error loading power spectrum: {}".format(str(e)))
        sys.exit(1)

def preprocess_data(power, log_transform=True, normalize=True):
    """
    Preprocess the power spectrum data.
    
    Args:
        power (numpy.ndarray): Power spectrum values
        log_transform (bool): Whether to apply log transform
        normalize (bool): Whether to normalize the data
        
    Returns:
        numpy.ndarray: Preprocessed power spectrum
    """
    # Make a copy to avoid modifying the original
    processed_power = np.copy(power)
    
    # Replace any negative or zero values with a small positive value
    if log_transform:
        min_positive = np.min(processed_power[processed_power > 0]) if np.any(processed_power > 0) else 1e-10
        processed_power[processed_power <= 0] = min_positive / 10.0
    
    # Apply log transform if requested
    if log_transform:
        processed_power = np.log(processed_power)
    
    # Normalize if requested
    if normalize:
        processed_power = (processed_power - np.mean(processed_power)) / (np.std(processed_power) or 1.0)
    
    return processed_power

def calculate_phi_scales(ell_max, min_scale=2):
    """
    Calculate phi-related scales up to ell_max.
    
    Args:
        ell_max (float): Maximum ell value to consider
        min_scale (int): Minimum scale to start from
        
    Returns:
        list: List of phi-related scales
    """
    phi = CONSTANTS['phi']
    phi_scales = []
    n = 0
    
    while True:
        scale = int(phi**n)
        if scale >= min_scale and scale <= ell_max:
            phi_scales.append(scale)
        elif scale > ell_max:
            break
        n += 1
    
    return phi_scales

def calculate_cross_scale_correlations(ell, power, scales, window_size=5):
    """
    Calculate correlations between power spectrum at different scales.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        scales (list): List of scales to analyze
        window_size (int): Size of window around each scale
        
    Returns:
        list: Correlation values between all pairs of scales
    """
    correlations = []
    
    for i in range(len(scales)):
        for j in range(i+1, len(scales)):
            # Get indices closest to these scales
            idx_i = np.abs(ell - scales[i]).argmin()
            idx_j = np.abs(ell - scales[j]).argmin()
            
            # Define windows
            i_start = max(0, idx_i - window_size // 2)
            i_end = min(len(power), idx_i + window_size // 2 + 1)
            j_start = max(0, idx_j - window_size // 2)
            j_end = min(len(power), idx_j + window_size // 2 + 1)
            
            # Extract power spectrum windows
            window_i = power[i_start:i_end]
            window_j = power[j_start:j_end]
            
            # Ensure both windows have the same length
            min_len = min(len(window_i), len(window_j))
            if min_len > 1:  # Need at least 2 points for correlation
                window_i = window_i[:min_len]
                window_j = window_j[:min_len]
                
                # Calculate correlation
                corr = np.corrcoef(window_i, window_j)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))  # Use absolute correlation
    
    return correlations

def process_chunk(ell, power, chunk_indices, phi_scales, window_size=5):
    """
    Process a chunk of simulations for cross-scale correlations test.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        chunk_indices (list): Indices of simulations to process
        phi_scales (list): Phi-related scales to analyze
        window_size (int): Size of window around each scale
        
    Returns:
        tuple: (phi_corrs, random_corrs) for this chunk
    """
    # Use process ID and time to seed
    np.random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    
    phi_corrs = []
    random_corrs = []
    
    n_phi_scales = len(phi_scales)
    
    for sim_idx in chunk_indices:
        # Generate surrogate data for each simulation except the first (actual data)
        if sim_idx > 0:
            # Shuffle the power spectrum to destroy correlations while preserving marginal distribution
            shuffled_power = np.random.permutation(power)
        else:
            # Use the original power spectrum for the first simulation
            shuffled_power = power
        
        # Calculate phi-related correlations
        if sim_idx == 0:
            # For actual data, calculate correlations between phi scales
            phi_correlations = calculate_cross_scale_correlations(ell, shuffled_power, phi_scales, window_size)
            phi_corrs.append(np.mean(phi_correlations) if phi_correlations else 0)
        
        # Calculate random scale correlations
        # Select random scales matching the number of phi scales
        random_scales = np.random.choice(ell, size=n_phi_scales, replace=False)
        random_scales.sort()  # Sort for consistency
        
        # Calculate correlations between randomly selected scales
        random_correlations = calculate_cross_scale_correlations(ell, shuffled_power, random_scales, window_size)
        random_corrs.append(np.mean(random_correlations) if random_correlations else 0)
        
        # Report progress periodically
        if sim_idx % 50 == 0 and sim_idx > 0:
            print("  Completed {} simulations in chunk".format(sim_idx))
    
    return phi_corrs, random_corrs

def calculate_statistics(mean_phi_corr, random_corrs):
    """
    Calculate statistical significance and phi-optimality.
    
    Args:
        mean_phi_corr (float): Mean correlation for phi-related scales
        random_corrs (list): Correlations for random scales across simulations
        
    Returns:
        dict: Dictionary with statistical results
    """
    # Convert to numpy arrays
    random_corrs = np.array(random_corrs)
    
    # Calculate mean and standard deviation for random correlations
    mean_random_corr = np.mean(random_corrs)
    std_random_corr = np.std(random_corrs)
    
    # Calculate z-score and p-value
    z_score = (mean_phi_corr - mean_random_corr) / (std_random_corr if std_random_corr > 0 else 1.0)
    p_value = 1 - stats.norm.cdf(z_score) if z_score > 0 else stats.norm.cdf(z_score)
    
    # Calculate ratio of phi to random correlations
    corr_ratio = mean_phi_corr / mean_random_corr if mean_random_corr > 0 else float('inf')
    
    # Calculate phi optimality using sigmoid function
    phi_optimality = 1.0 / (1.0 + np.exp(-10 * (corr_ratio - 1.0)))
    
    return {
        'mean_phi_corr': mean_phi_corr,
        'mean_random_corr': mean_random_corr,
        'z_score': z_score,
        'p_value': p_value,
        'corr_ratio': corr_ratio,
        'phi_optimality': phi_optimality
    }

def run_monte_carlo_parallel(ell, power, n_simulations=10000, window_size=5, num_processes=None, timeout_seconds=3600):
    """
    Run cross-scale correlations test using Monte Carlo simulations in parallel.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        n_simulations (int): Number of simulations to run
        window_size (int): Size of window around each scale
        num_processes (int): Number of processes to use (default: number of CPU cores)
        timeout_seconds (int): Timeout in seconds
        
    Returns:
        dict: Dictionary with test results
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print("Running Monte Carlo simulations with {} processes...".format(num_processes))
    
    # Preprocessed power for analysis
    processed_power = preprocess_data(power)
    
    # Calculate phi-related scales
    phi_scales = calculate_phi_scales(max(ell))
    print("Phi-related scales: {}".format(phi_scales))
    
    # Create a list of all simulation indices (0 = actual data)
    all_indices = list(range(n_simulations + 1))
    
    # Split indices into chunks for each process
    chunk_size = len(all_indices) // num_processes
    if chunk_size == 0:
        chunk_size = 1
    
    chunks = [all_indices[i:i+chunk_size] for i in range(0, len(all_indices), chunk_size)]
    
    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Submit tasks to the pool
    results = []
    for chunk in chunks:
        result = pool.apply_async(process_chunk, (ell, processed_power, chunk, phi_scales, window_size))
        results.append(result)
    
    # Close the pool (no more tasks can be submitted)
    pool.close()
    
    # Wait for all tasks to complete or timeout
    start_time = time.time()
    completed_chunks = 0
    
    phi_corrs_list = []
    random_corrs_list = []
    
    while completed_chunks < len(chunks):
        if time.time() - start_time > timeout_seconds:
            print("Timeout reached! Terminating...")
            pool.terminate()
            break
        
        # Check if any tasks have completed
        for i, result in enumerate(results):
            if result is not None and result.ready() and result.successful():
                phi_corrs, random_corrs = result.get()
                phi_corrs_list.extend(phi_corrs)
                random_corrs_list.extend(random_corrs)
                results[i] = None
                completed_chunks += 1
                print("Completed chunk {} of {} ({:.1f}%)".format(
                    completed_chunks, len(chunks), 100 * completed_chunks / len(chunks)))
        
        # Sleep briefly to avoid busy waiting
        time.sleep(0.1)
    
    # Terminate the pool if not already done
    if not pool._state:  # Check if pool is still running
        pool.terminate()
    pool.join()
    
    # Extract actual data results vs. simulation results
    try:
        # Find index 0 (actual data) in the results
        actual_indices = [i for i, idx in enumerate(all_indices) if idx == 0]
        if actual_indices:
            actual_idx = actual_indices[0]
            mean_phi_corr = phi_corrs_list[actual_idx]
            
            # Random correlations from all simulations
            random_corrs_array = np.array(random_corrs_list)
            
            # Calculate statistics
            stats_results = calculate_statistics(mean_phi_corr, random_corrs_array)
            
            print("Monte Carlo simulations completed in {:.1f} seconds".format(time.time() - start_time))
            return stats_results
        else:
            print("Error: Could not find actual data results (index 0)")
            return None
    except Exception as e:
        print("Error processing results: {}".format(str(e)))
        return None

def plot_results(results, output_dir):
    """
    Create visualizations of cross-scale correlation results.
    
    Args:
        results (dict): Dictionary with test results
        output_dir (str): Directory to save plots
        
    Returns:
        str: Path to the saved plot
    """
    plt.figure(figsize=(12, 10))
    
    # Create a 2x2 grid
    plt.subplot(2, 2, 1)
    bars = plt.bar(['Phi-Related', 'Random'], 
                  [results['mean_phi_corr'], results['mean_random_corr']],
                  color=['gold', 'gray'], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                '{:.4f}'.format(height), ha='center', va='bottom')
    plt.ylabel('Mean Correlation')
    plt.title('Cross-Scale Correlation Comparison')
    
    plt.subplot(2, 2, 2)
    plt.bar(['Correlation Ratio'], [results['corr_ratio']], color='gold', alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5)
    plt.text(0, results['corr_ratio'] + 0.1, "{:.4f}x".format(results['corr_ratio']), 
            ha='center', va='bottom')
    plt.ylabel('Ratio')
    plt.title('Phi-Related vs Random Correlation Ratio\n(Higher = Stronger Phi Pattern)')
    
    plt.subplot(2, 2, 3)
    plt.bar(['Phi Optimality'], [results['phi_optimality']], color='gold', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.5)
    plt.text(0, results['phi_optimality'] + 0.05, "{:.4f}".format(results['phi_optimality']), 
            ha='center', va='bottom')
    plt.ylabel('Optimality')
    plt.title('Phi Optimality\n(Higher = More Optimal)')
    
    plt.subplot(2, 2, 4)
    plt.bar(['p-value'], [results['p_value']], color='blue', alpha=0.7)
    plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
    plt.text(0, results['p_value'] + 0.01, "{:.4f}".format(results['p_value']), 
            ha='center', va='bottom')
    plt.ylabel('p-value')
    plt.title('Statistical Significance\n(p < 0.05 is significant)')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'cross_scale_correlations.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def run_cross_scale_test(data_path, output_base_dir, n_simulations=10000, window_size=5):
    """
    Run cross-scale correlation test on power spectrum data.
    
    Args:
        data_path (str): Path to the power spectrum data
        output_base_dir (str): Base directory for output
        n_simulations (int): Number of simulations to run
        window_size (int): Size of window around each scale
        
    Returns:
        dict: Test results
    """
    start_time = time.time()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, "cross_scale_{}_{}".format(n_simulations, timestamp))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n=== CROSS-SCALE CORRELATION TEST ===")
    print("Data: {}".format(data_path))
    print("Output: {}".format(output_dir))
    print("Simulations: {}".format(n_simulations))
    print("Window size: {}".format(window_size))
    
    # Load data
    print("\nLoading power spectrum data...")
    ell, power = load_power_spectrum(data_path)
    print("Loaded {} data points".format(len(ell)))
    
    # Run Monte Carlo simulations
    print("\nRunning cross-scale correlation test...")
    results = run_monte_carlo_parallel(ell, power, n_simulations=n_simulations, 
                                      window_size=window_size)
    
    if results:
        # Plot results
        print("\nCreating visualizations...")
        plot_path = plot_results(results, output_dir)
        
        # Save results
        results_path = os.path.join(output_dir, "results.txt")
        with open(results_path, "w") as f:
            f.write("=== CROSS-SCALE CORRELATION TEST RESULTS ===\n\n")
            f.write("Data: {}\n".format(data_path))
            f.write("Simulations: {}\n".format(n_simulations))
            f.write("Window size: {}\n\n".format(window_size))
            f.write("Phi-Related Mean Correlation: {:.6f}\n".format(results["mean_phi_corr"]))
            f.write("Random Mean Correlation: {:.6f}\n".format(results["mean_random_corr"]))
            f.write("Correlation Ratio: {:.6f}x\n".format(results["corr_ratio"]))
            f.write("Phi Optimality: {:.6f}\n".format(results["phi_optimality"]))
            f.write("p-value: {:.6f}\n".format(results["p_value"]))
            f.write("z-score: {:.6f}\n\n".format(results["z_score"]))
            
            # Add interpretation
            f.write("=== INTERPRETATION ===\n\n")
            
            # Correlation significance
            if results["p_value"] < 0.01:
                significance = "highly significant"
            elif results["p_value"] < 0.05:
                significance = "significant"
            elif results["p_value"] < 0.1:
                significance = "marginally significant"
            else:
                significance = "not significant"
            
            # Effect size
            if results["corr_ratio"] > 5:
                effect = "very strong"
            elif results["corr_ratio"] > 2:
                effect = "strong"
            elif results["corr_ratio"] > 1.5:
                effect = "moderate"
            elif results["corr_ratio"] > 1.1:
                effect = "weak"
            else:
                effect = "negligible"
            
            f.write("The cross-scale correlations between phi-related scales are {} times stronger ".format(round(results["corr_ratio"], 2)))
            f.write("than correlations between random scales, which is {} (p = {:.4f}).\n".format(significance, results["p_value"]))
            f.write("This represents a {} effect size.\n".format(effect))
            
            if results["phi_optimality"] > 0.9:
                f.write("The golden ratio appears to be highly optimal for organizing cross-scale correlations in this data.\n")
            elif results["phi_optimality"] > 0.7:
                f.write("The golden ratio appears to be moderately optimal for organizing cross-scale correlations in this data.\n")
            elif results["phi_optimality"] > 0.5:
                f.write("The golden ratio shows some optimality for organizing cross-scale correlations in this data.\n")
            else:
                f.write("The golden ratio does not appear to be optimal for organizing cross-scale correlations in this data.\n")
        
        # Save results as pickle for later analysis
        pickle_path = os.path.join(output_dir, "results.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)
        
        print("\nCross-scale correlation test completed in {:.2f} seconds".format(time.time() - start_time))
        print("Results saved to {}".format(output_dir))
        
        return {
            "results": results,
            "output_dir": output_dir,
            "plot_path": plot_path,
            "results_path": results_path
        }
    else:
        print("\nError: Cross-scale correlation test failed!")
        return None

def main():
    """
    Main function to run the cross-scale correlation test.
    """
    parser = argparse.ArgumentParser(description="Run cross-scale correlation test on CMB power spectrum data")
    parser.add_argument("--wmap", action="store_true", help="Run test on WMAP data")
    parser.add_argument("--planck", action="store_true", help="Run test on Planck data")
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations to run")
    parser.add_argument("--window", type=int, default=5, help="Window size around each scale")
    parser.add_argument("--output", type=str, default="results/cross_scale", 
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Set default if no specific dataset is selected
    if not (args.wmap or args.planck):
        args.wmap = True
        args.planck = True
    
    results = {}
    
    # Create base output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Run test on WMAP data
    if args.wmap:
        wmap_file = "data/wmap/wmap_tt_spectrum_9yr_v5.txt"
        if not os.path.exists(wmap_file):
            print("Error: WMAP power spectrum file not found at {}".format(wmap_file))
            wmap_file = "data/wmap/wmap_binned_tt_spectrum_9yr_v5.txt"
            if os.path.exists(wmap_file):
                print("Using alternative WMAP file: {}".format(wmap_file))
            else:
                print("No WMAP file found. Skipping WMAP analysis.")
                args.wmap = False
        
        if args.wmap:
            print("\n========== ANALYZING WMAP DATA ==========")
            wmap_output_dir = os.path.join(args.output, "wmap")
            results["wmap"] = run_cross_scale_test(
                wmap_file, wmap_output_dir, n_simulations=args.sims, window_size=args.window)
    
    # Run test on Planck data
    if args.planck:
        planck_file = "data/planck/planck_tt_spectrum_2018.txt"
        if not os.path.exists(planck_file):
            print("Error: Planck power spectrum file not found at {}".format(planck_file))
            planck_file = "data/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt"
            if os.path.exists(planck_file):
                print("Using alternative Planck file: {}".format(planck_file))
            else:
                print("No Planck file found. Skipping Planck analysis.")
                args.planck = False
        
        if args.planck:
            print("\n========== ANALYZING PLANCK DATA ==========")
            planck_output_dir = os.path.join(args.output, "planck")
            results["planck"] = run_cross_scale_test(
                planck_file, planck_output_dir, n_simulations=args.sims, window_size=args.window)
    
    # Compare results if both datasets were analyzed
    if args.wmap and args.planck and "wmap" in results and "planck" in results:
        print("\nComparing WMAP and Planck results...")
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        metrics = ["corr_ratio", "phi_optimality", "mean_phi_corr", "p_value"]
        titles = ["Correlation Ratio", "Phi Optimality", "Mean Phi Correlation", "p-value"]
        colors = ['blue', 'red']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(2, 2, i+1)
            values = [results["wmap"]["results"][metric], results["planck"]["results"][metric]]
            bars = plt.bar(["WMAP", "Planck"], values, color=colors, alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        '{:.4f}'.format(height), ha='center', va='bottom')
            
            if metric == "p_value":
                plt.axhline(y=0.05, color='black', linestyle='--', alpha=0.5)
            elif metric in ["corr_ratio", "phi_optimality"]:
                plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            plt.ylabel(title)
            plt.title(title)
        
        plt.tight_layout()
        
        # Save comparison
        comparison_dir = os.path.join(args.output, "comparison")
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)
        
        comparison_path = os.path.join(comparison_dir, "wmap_vs_planck.png")
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        
        print("Comparison saved to {}".format(comparison_path))
    
    print("\nCross-scale correlation test completed.")

if __name__ == "__main__":
    main()
