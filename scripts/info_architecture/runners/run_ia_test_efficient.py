#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Efficient Information Architecture Test Runner
This script runs the Information Architecture test with 10,000 simulations
with optimized performance and proper progress tracking for Python 2.7 compatibility.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
import multiprocessing
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import the Information Architecture Test
from scripts.info_architecture.archive.information_architecture_test import InformationArchitectureTest, load_wmap_power_spectrum, load_planck_power_spectrum

def ensure_dir_exists(dir_path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Standalone functions for parallel processing
def _run_single_simulation(data, constant, seed, config):
    """Run a single simulation for parallel processing."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create a temporary test instance
    test = InformationArchitectureTest(config)
    
    # Generate surrogate data using Fourier shuffling to preserve power spectrum
    try:
        # Get data length
        N = len(data)
        
        # Convert to frequency domain
        fft_data = np.fft.rfft(data)
        
        # Get amplitudes and phases
        amplitudes = np.abs(fft_data)
        phases = np.angle(fft_data)
        
        # Randomize phases while preserving amplitudes
        random_phases = np.random.uniform(0, 2 * np.pi, len(phases))
        
        # Reconstruct with random phases
        fft_random = amplitudes * np.exp(1j * random_phases)
        
        # Convert back to time domain
        surrogate_data = np.fft.irfft(fft_random, n=N)
        
        # Normalize to match original data statistics
        surrogate_data = (surrogate_data - np.mean(surrogate_data)) / np.std(surrogate_data)
        surrogate_data = surrogate_data * np.std(data) + np.mean(data)
    except:
        # Fallback to simple permutation if Fourier shuffling fails
        surrogate_data = np.random.permutation(data)
    
    # Calculate architecture score for surrogate data
    score = test.calculate_architecture_score(surrogate_data, constant)
    
    return score

def _run_single_simulation_star(args):
    """Wrapper for _run_single_simulation to unpack arguments."""
    return _run_single_simulation(*args)

def run_monte_carlo_simulation(data, constant, num_simulations, output_dir, config):
    """
    Run Monte Carlo simulation efficiently with proper progress reporting.
    
    Args:
        data: Input data
        constant: Mathematical constant to test
        num_simulations: Number of simulations to run
        output_dir: Directory to save progress and results
        config: Configuration dictionary
        
    Returns:
        dict: Results including p-value and significance
    """
    # Create test instance for actual score calculation
    test = InformationArchitectureTest(config)
    
    # Get actual score
    actual_score = test.calculate_architecture_score(data, constant)
    
    # Initialize progress tracking
    constant_dir = os.path.join(output_dir, str(constant).replace('.', '_'))
    ensure_dir_exists(constant_dir)
    progress_file = os.path.join(constant_dir, "progress.txt")
    
    # Write header to progress file
    with open(progress_file, 'w') as f:
        f.write("# Information Architecture Test - Monte Carlo Simulation\n")
        f.write("# Constant: %s\n" % constant)
        f.write("# Actual Score: %s\n" % actual_score)
        f.write("# Simulation Progress:\n")
        f.write("# Simulation,Score,p-value\n")
    
    # Count how many random scores are >= actual score
    count_greater_equal = 0
    all_scores = []
    
    # Use parallel processing
    start_time = time.time()
    last_update_time = start_time
    
    # Create a pool of workers
    num_workers = min(cpu_count(), config.get('max_workers', cpu_count()))
    pool = Pool(processes=num_workers)
    
    # Prepare arguments for parallel execution
    args = [(data, constant, i, config) for i in range(num_simulations)]
    
    # Execute simulations in parallel with a chunksize that balances overhead and distribution
    chunksize = max(1, num_simulations // (num_workers * 4))
    
    # Track progress
    completed = 0
    
    try:
        # Process results as they come in
        for i, score in enumerate(pool.imap_unordered(_run_single_simulation_star, args, chunksize=chunksize)):
            all_scores.append(score)
            if score >= actual_score:
                count_greater_equal += 1
            
            # Calculate p-value
            p_value = float(count_greater_equal) / (i + 1)
            
            # Update progress file periodically to avoid excessive I/O
            if i % 50 == 0 or i == num_simulations - 1:
                with open(progress_file, 'a') as f:
                    f.write("%d,%f,%f\n" % (i, score, p_value))
            
            # Print progress
            completed += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Update progress more frequently (every 50 simulations or every 10 seconds)
            if completed % 50 == 0 or (current_time - last_update_time) >= 10 or completed == num_simulations:
                last_update_time = current_time
                
                # Calculate progress percentage and estimated time remaining
                progress_pct = 100.0 * completed / num_simulations
                if completed > 0:
                    avg_time_per_sim = elapsed_time / completed
                    remaining_sims = num_simulations - completed
                    est_remaining_time = avg_time_per_sim * remaining_sims
                else:
                    est_remaining_time = 0
                
                # Create a text-based progress bar
                bar_length = 30
                filled_length = int(bar_length * completed // num_simulations)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                # Print progress with progress bar
                sys.stdout.write("\rProgress: [%s] %5.1f%% (%d/%d) - p-value: %.6f - Elapsed: %.1fs - Est. remaining: %.1fs" % 
                      (bar, progress_pct, completed, num_simulations, p_value, 
                       elapsed_time, est_remaining_time))
                sys.stdout.flush()
    
    finally:
        # Close the pool
        pool.close()
        pool.join()
    
    # Print final progress
    print("\nCompleted %d/%d simulations in %.1f seconds" % (completed, num_simulations, elapsed_time))
    
    # Calculate final p-value
    p_value = float(count_greater_equal) / len(all_scores) if all_scores else 1.0
    
    # Calculate z-score
    if len(all_scores) > 1:
        mean_surrogate = np.mean(all_scores)
        std_surrogate = np.std(all_scores)
        if std_surrogate > 0:
            z_score = (actual_score - mean_surrogate) / std_surrogate
        else:
            z_score = 0
    else:
        z_score = 0
    
    # Determine significance
    significance_level = config.get('significance_level', 0.05)
    significant = p_value < significance_level
    
    # Save surrogate distribution
    surrogate_file = os.path.join(constant_dir, "surrogate_scores.txt")
    np.savetxt(surrogate_file, all_scores)
    
    # Create histogram of surrogate scores
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=30, alpha=0.7, color='skyblue')
    plt.axvline(actual_score, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Architecture Score')
    plt.ylabel('Frequency')
    plt.title('Surrogate Distribution for %s (p=%.4f)' % (constant, p_value))
    plt.legend(['Actual Score', 'Surrogate Scores'])
    plt.tight_layout()
    plt.savefig(os.path.join(constant_dir, "surrogate_distribution.png"), dpi=300)
    plt.close()
    
    # Return results
    return {
        'constant': constant,
        'actual_score': actual_score,
        'surrogate_scores': all_scores,
        'p_value': p_value,
        'z_score': z_score,
        'significant': significant,
        'num_simulations': len(all_scores)
    }

def run_test_for_dataset(data, constants, output_dir, config):
    """
    Run the Information Architecture Test for a single dataset.
    
    Args:
        data: Input data (power spectrum)
        constants: Dictionary of constants to test
        output_dir: Directory to save results
        config: Configuration dictionary
        
    Returns:
        dict: Results for each constant
    """
    # Ensure output directory exists
    ensure_dir_exists(output_dir)
    
    # Run test for each constant
    results = {}
    for name, value in constants.items():
        print("\n" + "="*50)
        print("Testing constant: %s = %s" % (name, value))
        print("="*50)
        
        # Create output directory for this constant
        constant_output_dir = os.path.join(output_dir, name)
        ensure_dir_exists(constant_output_dir)
        
        # Run Monte Carlo simulation
        constant_results = run_monte_carlo_simulation(
            data, 
            value, 
            config['num_simulations'],
            constant_output_dir,
            config
        )
        
        # Store results
        results[name] = constant_results
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    visualize_results(results, output_dir)
    
    return results

def generate_summary_report(results, output_dir):
    """
    Generate a summary report of the test results.
    
    Args:
        results: Dictionary of test results
        output_dir: Directory to save report
    """
    # Ensure output directory exists
    ensure_dir_exists(output_dir)
    
    # Create summary report
    report_file = os.path.join(output_dir, "summary_report.txt")
    with open(report_file, 'w') as f:
        f.write("# Information Architecture Test - Summary Report\n")
        f.write("# Date: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write("# Number of Simulations: %s\n" % results[list(results.keys())[0]]['num_simulations'])
        f.write("\n")
        
        # Write table header
        f.write("Constant,Value,Architecture Score,p-value,z-score,Significant\n")
        
        # Write results for each constant
        for name, constant_results in results.items():
            f.write("%s,%s,%.6f,%.6f,%.2f,%s\n" % (
                name, 
                constant_results['constant'], 
                constant_results['actual_score'], 
                constant_results['p_value'], 
                constant_results['z_score'], 
                constant_results['significant']
            ))

def visualize_results(results, output_dir):
    """
    Generate visualizations of the test results.
    
    Args:
        results: Dictionary of test results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    ensure_dir_exists(output_dir)
    
    # Create bar chart of architecture scores
    plt.figure(figsize=(12, 6))
    
    # Extract data for plotting
    constant_names = list(results.keys())
    architecture_scores = [results[name]['actual_score'] for name in constant_names]
    p_values = [results[name]['p_value'] for name in constant_names]
    
    # Sort by architecture score
    sorted_indices = np.argsort(architecture_scores)[::-1]
    constant_names = [constant_names[i] for i in sorted_indices]
    architecture_scores = [architecture_scores[i] for i in sorted_indices]
    p_values = [p_values[i] for i in sorted_indices]
    
    # Create bar chart
    bars = plt.bar(constant_names, architecture_scores, color='skyblue')
    
    # Highlight significant results
    for i, p in enumerate(p_values):
        if p < 0.05:  # Using standard significance level
            bars[i].set_color('green')
    
    # Add labels and title
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Architecture Score')
    plt.title('Information Architecture Test Results')
    
    # Add p-values as text
    for i, (score, p) in enumerate(zip(architecture_scores, p_values)):
        plt.text(i, score + 0.02, "p=%.4f" % p, ha='center')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "architecture_scores.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    # Configuration
    config = {
        'num_simulations': 10000,
        'constants': {
            'phi': 1.61803398875,
            'sqrt2': 1.41421356237,
            'sqrt3': 1.73205080757,
            'ln2': 0.693147180559945,
            'e': 2.71828182846,
            'pi': 3.14159265359
        },
        'parallel_processing': True,
        'max_workers': 8,  # Adjust based on available CPU cores
        'output_dir': "../results/information_architecture_10k_efficient"
    }
    
    # Create output directory
    ensure_dir_exists(config['output_dir'])
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run test on WMAP data
    print("\n" + "="*50)
    print("Running Information Architecture Test on WMAP data (%d simulations)" % config['num_simulations'])
    print("="*50)
    
    # Load WMAP data
    wmap_data = load_wmap_power_spectrum()
    
    # Run test for WMAP data
    wmap_output_dir = os.path.join(config['output_dir'], 'wmap')
    ensure_dir_exists(wmap_output_dir)
    
    wmap_results = run_test_for_dataset(
        wmap_data, 
        config['constants'],
        wmap_output_dir,
        config
    )
    
    # Run test on Planck data
    print("\n" + "="*50)
    print("Running Information Architecture Test on Planck data (%d simulations)" % config['num_simulations'])
    print("="*50)
    
    # Load Planck data
    planck_data = load_planck_power_spectrum()
    
    # Run test for Planck data
    planck_output_dir = os.path.join(config['output_dir'], 'planck')
    ensure_dir_exists(planck_output_dir)
    
    planck_results = run_test_for_dataset(
        planck_data, 
        config['constants'],
        planck_output_dir,
        config
    )
    
    # Print overall summary
    print("\n" + "="*50)
    print("Information Architecture Test Complete")
    print("="*50)
    print("Results saved to: %s" % config['output_dir'])
