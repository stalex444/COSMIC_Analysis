#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved Information Architecture Test Runner with Enhanced Progress Reporting
This script runs the Information Architecture test with 10,000 simulations
with optimized performance and reliable progress tracking for Python 2.7 compatibility.
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
    
    # Generate surrogate data (shuffle the original data)
    surrogate_data = np.random.permutation(data)
    
    # Calculate architecture score for surrogate data
    score = test.calculate_architecture_score(surrogate_data, constant)
    
    return score

def _run_single_simulation_star(args):
    """Wrapper for _run_single_simulation to unpack arguments."""
    return _run_single_simulation(*args)

def run_monte_carlo_simulation(data, constant, num_simulations, output_dir, config):
    """
    Run Monte Carlo simulation efficiently with improved progress reporting.
    
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
    
    # Create a separate file for real-time status updates
    status_file = os.path.join(constant_dir, "status.txt")
    with open(status_file, 'w') as f:
        f.write("Starting simulations...\n")
    
    # Count how many random scores are >= actual score
    count_greater_equal = 0
    all_scores = []
    
    # Use parallel processing
    start_time = time.time()
    last_update_time = start_time
    
    # Create a pool of workers
    num_workers = min(cpu_count(), config.get('max_workers', cpu_count()))
    
    # Update status file with initialization info
    with open(status_file, 'a') as f:
        f.write("Initialized %d worker processes\n" % num_workers)
        f.write("Starting time: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Process simulations in smaller batches for better progress reporting
    batch_size = 100  # Process 100 simulations at a time
    completed = 0
    
    try:
        while completed < num_simulations:
            # Determine how many simulations to run in this batch
            current_batch_size = min(batch_size, num_simulations - completed)
            
            # Update status file
            with open(status_file, 'a') as f:
                f.write("Processing batch of %d simulations (total completed: %d/%d)\n" % 
                        (current_batch_size, completed, num_simulations))
            
            # Create a pool for this batch
            pool = Pool(processes=num_workers)
            
            # Prepare arguments for parallel execution
            args = [(data, constant, completed + i, config) for i in range(current_batch_size)]
            
            # Execute simulations in parallel
            batch_scores = pool.map(_run_single_simulation_star, args)
            
            # Close and join the pool
            pool.close()
            pool.join()
            
            # Process the results from this batch
            for i, score in enumerate(batch_scores):
                all_scores.append(score)
                if score >= actual_score:
                    count_greater_equal += 1
                
                # Calculate p-value
                p_value = float(count_greater_equal) / len(all_scores)
                
                # Update progress file for each simulation
                with open(progress_file, 'a') as f:
                    f.write("%d,%f,%f\n" % (completed + i, score, p_value))
            
            # Update completed count
            completed += current_batch_size
            
            # Calculate and display progress
            current_time = time.time()
            elapsed_time = current_time - start_time
            
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
            progress_str = "\rProgress: [%s] %5.1f%% (%d/%d) - p-value: %.6f - Elapsed: %.1fs - Est. remaining: %.1fs" % (
                bar, progress_pct, completed, num_simulations, p_value, elapsed_time, est_remaining_time)
            
            sys.stdout.write(progress_str)
            sys.stdout.flush()
            
            # Update status file with current progress
            with open(status_file, 'a') as f:
                f.write("%s\n" % progress_str.strip())
    
    except Exception as e:
        # Log any errors
        with open(status_file, 'a') as f:
            f.write("ERROR: %s\n" % str(e))
        raise
    
    # Print final progress
    print("\nCompleted %d/%d simulations in %.1f seconds" % (completed, num_simulations, elapsed_time))
    
    # Update status file with completion info
    with open(status_file, 'a') as f:
        f.write("Completed %d/%d simulations in %.1f seconds\n" % (completed, num_simulations, elapsed_time))
        f.write("End time: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
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
    
    # Create a status file for the dataset
    status_file = os.path.join(output_dir, "dataset_status.txt")
    with open(status_file, 'w') as f:
        f.write("# Information Architecture Test - Dataset Status\n")
        f.write("# Start Time: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write("# Constants to test: %s\n" % ", ".join(constants.keys()))
    
    # Run test for each constant
    results = {}
    for name, value in constants.items():
        print("\n" + "="*50)
        print("Testing constant: %s = %s" % (name, value))
        print("="*50)
        
        # Update dataset status file
        with open(status_file, 'a') as f:
            f.write("\nStarting test for constant: %s = %s at %s\n" % 
                    (name, value, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
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
        
        # Update dataset status file
        with open(status_file, 'a') as f:
            f.write("Completed test for constant: %s = %s at %s\n" % 
                    (name, value, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            f.write("Results: p-value = %.6f, significant = %s\n" % 
                    (constant_results['p_value'], constant_results['significant']))
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    visualize_results(results, output_dir)
    
    # Update dataset status file
    with open(status_file, 'a') as f:
        f.write("\nAll tests completed at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
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
        'output_dir': "../results/information_architecture_10k_improved"
    }
    
    # Create output directory
    ensure_dir_exists(config['output_dir'])
    
    # Create a master status file
    master_status_file = os.path.join(config['output_dir'], "master_status.txt")
    with open(master_status_file, 'w') as f:
        f.write("# Information Architecture Test - Master Status\n")
        f.write("# Start Time: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write("# Configuration:\n")
        f.write("#   Number of Simulations: %d\n" % config['num_simulations'])
        f.write("#   Constants: %s\n" % ", ".join(config['constants'].keys()))
        f.write("#   Max Workers: %d\n" % config['max_workers'])
    
    try:
        # Run test on WMAP data
        print("\n" + "="*50)
        print("Running Information Architecture Test on WMAP data (%d simulations)" % config['num_simulations'])
        print("="*50)
        
        # Update master status file
        with open(master_status_file, 'a') as f:
            f.write("\nStarting WMAP data analysis at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
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
        
        # Update master status file
        with open(master_status_file, 'a') as f:
            f.write("Completed WMAP data analysis at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Run test on Planck data
        print("\n" + "="*50)
        print("Running Information Architecture Test on Planck data (%d simulations)" % config['num_simulations'])
        print("="*50)
        
        # Update master status file
        with open(master_status_file, 'a') as f:
            f.write("\nStarting Planck data analysis at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
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
        
        # Update master status file
        with open(master_status_file, 'a') as f:
            f.write("Completed Planck data analysis at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Print overall summary
        print("\n" + "="*50)
        print("Information Architecture Test Complete")
        print("="*50)
        print("Results saved to: %s" % config['output_dir'])
        
        # Update master status file
        with open(master_status_file, 'a') as f:
            f.write("\nAll tests completed successfully at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    except Exception as e:
        # Log any errors to the master status file
        with open(master_status_file, 'a') as f:
            f.write("\nERROR: %s\n" % str(e))
            f.write("Test failed at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        raise
