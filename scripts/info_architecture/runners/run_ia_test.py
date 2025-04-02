#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved Information Architecture Test Runner
This script runs the Information Architecture test with 10,000 simulations
and proper progress tracking for Python 2.7 compatibility.
Includes multiple testing correction.
"""

import numpy as np
import os
import sys
import time
import argparse
from datetime import datetime

# Add multiple testing correction
def apply_multiple_testing_correction(p_values, method='fdr_bh'):
    """
    Apply correction for multiple testing.
    
    Parameters:
    - p_values: Dictionary mapping test names to p-values
    - method: Correction method ('bonferroni' or 'fdr_bh')
    
    Returns:
    - Dictionary with adjusted p-values
    """
    test_names = list(p_values.keys())
    p_vals = [p_values[name] for name in test_names]
    
    if method == 'bonferroni':
        # Bonferroni correction
        adjusted_p = [min(p * len(p_vals), 1.0) for p in p_vals]
    else:
        # Benjamini-Hochberg FDR correction
        # Sort p-values
        p_vals_sorted = sorted([(p, i) for i, p in enumerate(p_vals)])
        # Calculate adjusted p-values
        m = len(p_vals)
        adjusted_p = [0] * m
        
        # Implement the Benjamini-Hochberg procedure
        for rank, (p, i) in enumerate(p_vals_sorted):
            adjusted_p[i] = min(p * m / (rank + 1), 1.0)
        
        # Ensure monotonicity
        for i in range(m-2, -1, -1):
            adjusted_p[p_vals_sorted[i][1]] = min(
                adjusted_p[p_vals_sorted[i][1]],
                adjusted_p[p_vals_sorted[i+1][1]]
            )
    
    return {name: adj_p for name, adj_p in zip(test_names, adjusted_p)}

def load_power_spectrum(filename):
    """Load power spectrum data from file."""
    try:
        data = np.loadtxt(filename)
        if data.ndim > 1:  # If data has multiple columns
            # Use the second column (index 1) which typically contains power values
            return data[:, 1]
        return data
    except Exception as e:
        print("Error loading data from {}: {}".format(filename, e))
        return None

def calculate_architecture_score(data, constant):
    """
    Calculate the architecture score for a given dataset and constant.
    Higher score indicates stronger alignment with the constant.
    """
    # Simple implementation - can be replaced with more sophisticated metrics
    data_normalized = data / np.max(data)
    
    # Calculate ratios between consecutive elements
    ratios = data_normalized[1:] / data_normalized[:-1]
    
    # Calculate how close these ratios are to the constant
    closeness = 1.0 / (1.0 + np.abs(ratios - constant))
    
    # Return the mean closeness as the architecture score
    return np.mean(closeness)

def shuffle_data(data, seed=None):
    """Shuffle data while preserving its statistical properties."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create a copy of the data
    shuffled_data = data.copy()
    
    # Shuffle the data
    np.random.shuffle(shuffled_data)
    
    return shuffled_data

def run_monte_carlo_simulation(data, constant, num_simulations, output_dir):
    """
    Run Monte Carlo simulations to assess statistical significance.
    """
    # Get the constant name for display purposes
    constants = {
        'phi': (1 + 5**0.5) / 2,
        'sqrt2': 2**0.5,
        'e': np.e,
        'pi': np.pi,
        'sqrt3': 3**0.5,
        'ln2': np.log(2)
    }
    
    constant_name = "unknown"
    for name, value in constants.items():
        if abs(value - constant) < 1e-10:
            constant_name = name
            break
    
    # Calculate actual architecture score
    actual_score = calculate_architecture_score(data, constant)
    
    # Initialize progress file
    progress_file = os.path.join(output_dir, "{}_progress.txt".format(constant_name.lower()))
    
    # Write initial information to progress file
    with open(progress_file, 'w') as f:
        f.write("Starting {} simulations for {}\n".format(num_simulations, constant_name))
        f.write("Actual score: {}\n".format(actual_score))
        f.write("Progress:\n")
    
    # Initialize list to store simulation scores
    simulation_scores = []
    
    # Run simulations sequentially
    for i in range(num_simulations):
        # Run a single simulation
        seed = i  # Use iteration as seed for reproducibility
        shuffled_data = shuffle_data(data, seed)
        score = calculate_architecture_score(shuffled_data, constant)
        simulation_scores.append(score)
        
        # Update progress file periodically
        if (i + 1) % 100 == 0 or i == 0 or i == num_simulations - 1:
            # Count how many simulations exceed the actual score
            exceeds_count = sum(1 for s in simulation_scores if s >= actual_score)
            p_value = float(exceeds_count) / (i + 1)
            
            # Write progress to file
            with open(progress_file, 'w') as f:
                f.write("Starting {} simulations for {}\n".format(num_simulations, constant_name))
                f.write("Actual score: {}\n".format(actual_score))
                f.write("Progress:\n")
                f.write("Completed {}/{} simulations ({:.1f}%)\n".format(
                    i+1, num_simulations, (i+1)*100.0/num_simulations))
                f.write("Current exceeds count: {}/{}\n".format(exceeds_count, i+1))
                f.write("Current p-value: {:.6f}\n".format(p_value))
            
            # Print progress
            print("Completed {}/{} simulations for {} ({:.1f}%)".format(
                i+1, num_simulations, constant_name, (i+1)*100.0/num_simulations))
        
        # Check for early stopping - ONLY if we've run at least 1000 simulations
        if (i + 1) >= 1000:
            # Count how many simulations exceed the actual score
            exceeds_count = sum(1 for s in simulation_scores if s >= actual_score)
            p_value = float(exceeds_count) / (i + 1)
            
            # If 5% or more exceed, we can stop early (not significant)
            if p_value >= 0.05:
                with open(progress_file, 'a') as f:
                    f.write("\nEarly stopping triggered after {} simulations\n".format(i+1))
                    f.write("Exceeds count: {}, p-value: {:.6f}\n".format(exceeds_count, p_value))
                break
    
    # Calculate final p-value
    exceeds_count = sum(1 for s in simulation_scores if s >= actual_score)
    p_value = float(exceeds_count) / len(simulation_scores)
    
    # Write final results to file
    results_file = os.path.join(output_dir, "{}_results.txt".format(constant_name.lower()))
    with open(results_file, 'w') as f:
        f.write("Final Results for {}:\n".format(constant_name))
        f.write("Actual score: {}\n".format(actual_score))
        f.write("Total simulations: {}\n".format(len(simulation_scores)))
        f.write("Exceeds count: {}\n".format(exceeds_count))
        f.write("Raw p-value: {:.6f}\n".format(p_value))
        f.write("Significant at alpha=0.05 (before correction): {}\n".format("Yes" if p_value < 0.05 else "No"))
        f.write("\nNote: Multiple testing correction will be applied after all tests are complete.\n")
    
    return {
        'constant': constant_name,
        'actual_score': actual_score,
        'p_value': p_value,
        'num_simulations': len(simulation_scores),
        'significant': p_value < 0.05
    }

def run_full_test(data, constants, output_dir, num_simulations):
    """
    Run the full Information Architecture Test on the data.
    Tests multiple mathematical constants and generates a summary report.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize results
    results = {}
    p_values = {}
    
    # Test each constant
    for name, value in constants.items():
        print("\nTesting {}...".format(name))
        result = run_monte_carlo_simulation(data, value, num_simulations, output_dir)
        results[name] = result
        p_values[name] = result['p_value']
    
    # Apply multiple testing correction
    print("\nApplying multiple testing correction...")
    
    # Benjamini-Hochberg FDR correction
    adjusted_p_fdr = apply_multiple_testing_correction(p_values, method='fdr_bh')
    
    # Bonferroni correction
    adjusted_p_bonferroni = apply_multiple_testing_correction(p_values, method='bonferroni')
    
    # Update results with adjusted p-values
    for name in results:
        results[name]['adjusted_p_fdr'] = adjusted_p_fdr[name]
        results[name]['adjusted_p_bonferroni'] = adjusted_p_bonferroni[name]
        results[name]['significant_after_fdr'] = adjusted_p_fdr[name] < 0.05
        results[name]['significant_after_bonferroni'] = adjusted_p_bonferroni[name] < 0.05
    
    # Generate summary report
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Information Architecture Test Summary\n")
        f.write("===================================\n\n")
        f.write("Date: {}\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        f.write("Results:\n")
        f.write("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}\n".format(
            "Constant", "Actual Score", "Raw p-value", "FDR p-value", "Bonferroni p", "Significant", "Simulations"))
        f.write("-" * 100 + "\n")
        
        for name, result in results.items():
            f.write("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15} {:<15}\n".format(
                name, 
                result['actual_score'], 
                result['p_value'],
                result['adjusted_p_fdr'],
                result['adjusted_p_bonferroni'],
                "Yes" if result['significant_after_fdr'] else "No", 
                result['num_simulations']))
        
        f.write("\nNote: Significance is based on FDR-corrected p-values with alpha=0.05\n")
        f.write("Multiple testing corrections applied: Benjamini-Hochberg FDR and Bonferroni\n")
    
    print("\nTest completed. Results saved to {}".format(summary_file))
    return results

def main():
    """Main function to run the Information Architecture Test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Information Architecture Test for CMB Data')
    
    parser.add_argument('--wmap-file', type=str, required=True,
                        help='Path to WMAP power spectrum data file')
    
    parser.add_argument('--planck-file', type=str, required=True,
                        help='Path to Planck power spectrum data file')
    
    parser.add_argument('--output-dir', type=str, default="../results/information_architecture",
                        help='Directory to save results')
    
    parser.add_argument('--num-simulations', type=int, default=10000,
                        help='Number of Monte Carlo simulations to run')
    
    parser.add_argument('--early-stopping', action='store_true', default=True,
                        help='Enable early stopping')
    
    args = parser.parse_args()
    
    # Define constants to test
    constants = {
        'phi': (1 + 5**0.5) / 2,  # Golden ratio
        'sqrt2': 2**0.5,          # Square root of 2
        'e': np.e,                # Euler's number
        'pi': np.pi,              # Pi
        'sqrt3': 3**0.5,          # Square root of 3
        'ln2': np.log(2)          # Natural log of 2
    }
    
    # Run test on WMAP data
    print("\n" + "="*50)
    print("Running Information Architecture Test on WMAP data")
    print("="*50)
    
    # Load WMAP data
    wmap_data = load_power_spectrum(args.wmap_file)
    if wmap_data is not None:
        # Create output directory for WMAP results
        wmap_output_dir = os.path.join(args.output_dir, 'wmap')
        if not os.path.exists(wmap_output_dir):
            os.makedirs(wmap_output_dir)
        
        # Run test on WMAP data
        wmap_results = run_full_test(wmap_data, constants, wmap_output_dir, args.num_simulations)
    else:
        print("Error: Failed to load WMAP data.")
    
    # Run test on Planck data
    print("\n" + "="*50)
    print("Running Information Architecture Test on Planck data")
    print("="*50)
    
    # Load Planck data
    planck_data = load_power_spectrum(args.planck_file)
    if planck_data is not None:
        # Create output directory for Planck results
        planck_output_dir = os.path.join(args.output_dir, 'planck')
        if not os.path.exists(planck_output_dir):
            os.makedirs(planck_output_dir)
        
        # Run test on Planck data
        planck_results = run_full_test(planck_data, constants, planck_output_dir, args.num_simulations)
    else:
        print("Error: Failed to load Planck data.")

if __name__ == "__main__":
    main()
