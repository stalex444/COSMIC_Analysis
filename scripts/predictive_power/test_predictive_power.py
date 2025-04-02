#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predictive Power Test Module.

This test examines how well the golden ratio can predict actual peaks in the CMB power spectrum
by calculating the match rate between golden ratio-based predictions and actual peaks,
then comparing this to random predictions.
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
from scipy import stats
from scipy.signal import find_peaks

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

def preprocess_data(power, log_transform=False, normalize=True, smooth=False, window_size=5):
    """
    Preprocess the power spectrum data for peak detection.
    
    Args:
        power (numpy.ndarray): Power spectrum values
        log_transform (bool): Whether to apply log transform
        normalize (bool): Whether to normalize the data
        smooth (bool): Whether to apply smoothing
        window_size (int): Window size for smoothing
        
    Returns:
        numpy.ndarray: Preprocessed power spectrum
    """
    # Make a copy to avoid modifying the original
    processed_power = np.copy(power)
    
    # Replace any negative or zero values with a small positive value if log transform
    if log_transform:
        min_positive = np.min(processed_power[processed_power > 0]) if np.any(processed_power > 0) else 1e-10
        processed_power[processed_power <= 0] = min_positive / 10.0
    
    # Apply log transform if requested
    if log_transform:
        processed_power = np.log(processed_power)
    
    # Apply smoothing if requested
    if smooth and window_size > 1:
        kernel = np.ones(window_size) / window_size
        # Use convolution for smoothing
        padded = np.pad(processed_power, (window_size//2, window_size//2), mode='edge')
        processed_power = np.convolve(padded, kernel, mode='valid')
    
    # Normalize if requested
    if normalize:
        processed_power = (processed_power - np.mean(processed_power)) / (np.std(processed_power) or 1.0)
    
    return processed_power

def find_peaks_in_spectrum(ell, power, height=0, distance=10, prominence=0.5, width=None):
    """
    Find peaks in the CMB power spectrum using scipy's find_peaks function.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        height (float): Minimum height of peaks
        distance (int): Minimum distance between peaks
        prominence (float): Minimum prominence of peaks
        width (float): Minimum width of peaks
        
    Returns:
        list: Multipole values (ell) where peaks are located
    """
    # Process power spectrum for better peak detection
    processed_power = preprocess_data(power, smooth=True, window_size=5)
    
    # Use scipy's find_peaks function to identify peaks
    peak_indices, _ = find_peaks(
        processed_power, 
        height=height, 
        distance=distance, 
        prominence=prominence,
        width=width
    )
    
    # Convert peak indices to ell values
    peak_ells = [ell[i] for i in peak_indices]
    
    print("Identified {} peaks in the CMB power spectrum".format(len(peak_ells)))
    print("Peak locations (ell values): {}".format(", ".join(map(str, [int(p) for p in peak_ells]))))
    
    return peak_ells

def find_first_peak(ell, power):
    """
    Find the first significant peak in the power spectrum.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        
    Returns:
        float: Multipole value of the first peak
    """
    processed_power = preprocess_data(power, smooth=True, window_size=5)
    
    # Use more conservative parameters for first peak detection
    peak_indices, properties = find_peaks(
        processed_power, 
        height=0.5,  # Higher threshold for first peak 
        distance=10, 
        prominence=1.0,  # Higher prominence for better certainty
        width=None
    )
    
    if len(peak_indices) > 0:
        return ell[peak_indices[0]]
    else:
        # Fall back to default if no peaks found (unlikely)
        print("Warning: No first peak detected, using default value")
        return 220.0

def generate_gr_predictions(ell, power, n_forward=6, n_backward=6):
    """
    Generate predictions based on golden ratio relationships from the first detected peak.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        n_forward (int): Number of forward predictions
        n_backward (int): Number of backward predictions
        
    Returns:
        list: List of predicted peak positions
    """
    phi = CONSTANTS['phi']
    ell_max = max(ell)
    
    # Find first peak automatically
    first_peak = find_first_peak(ell, power)
    print("Detected first peak at ell ≈ {:.1f}".format(first_peak))
    
    predictions = []
    
    # Forward predictions (multiplying by powers of phi)
    for i in range(1, n_forward + 1):
        prediction = int(first_peak * phi**i)
        if prediction <= ell_max:
            predictions.append(prediction)
    
    # Add the first peak itself
    predictions.append(int(first_peak))
    
    # Backward predictions (dividing by powers of phi)
    for i in range(1, n_backward + 1):
        prediction = int(first_peak / phi**i)
        if prediction > 0:  # Ensure we don't go below zero
            predictions.append(prediction)
    
    # Sort predictions
    predictions.sort()
    
    print("Generated {} golden ratio-based predictions".format(len(predictions)))
    print("Predicted peak locations: {}".format(", ".join(map(str, predictions))))
    
    return predictions

def calculate_match_rate(actual_peaks, predictions, base_tolerance=0.08):
    """
    Calculate the match rate between predicted and actual peaks with scale-dependent tolerance.
    
    Args:
        actual_peaks (list): List of actual peak positions
        predictions (list): List of predicted peak positions
        base_tolerance (float): Base tolerance for matching
        
    Returns:
        tuple: (match_count, match_rate, matched_peaks, matched_predictions)
    """
    matches = 0
    matched_peaks = []
    matched_predictions = []
    
    for pred in predictions:
        # Scale-dependent tolerance - lower tolerance for higher multipoles
        scale_factor = max(0.5, min(1.0, 500 / pred))
        tolerance = base_tolerance * scale_factor
        
        # Calculate absolute tolerance based on the prediction value
        abs_tolerance = pred * tolerance
        
        # Check if any actual peak is within tolerance of the prediction
        for peak in actual_peaks:
            if abs(peak - pred) <= abs_tolerance:
                matches += 1
                matched_peaks.append(peak)
                matched_predictions.append(pred)
                break
    
    match_rate = float(matches) / len(predictions) if predictions else 0
    
    print("Match count: {}/{}".format(matches, len(predictions)))
    print("Match rate: {:.4f}".format(match_rate))
    
    return matches, match_rate, matched_peaks, matched_predictions

def generate_random_predictions(actual_peaks, n_predictions, ell_max):
    """
    Generate random predictions that better match the distribution of actual peaks.
    
    Args:
        actual_peaks (list): List of actual peak positions
        n_predictions (int): Number of predictions to generate
        ell_max (float): Maximum ell value
        
    Returns:
        list: List of random predictions
    """
    # Calculate mean spacing between actual peaks
    if len(actual_peaks) > 1:
        spacings = np.diff(sorted(actual_peaks))
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)
    else:
        # Default values if insufficient peaks
        mean_spacing = ell_max / 10
        std_spacing = mean_spacing / 2
    
    # Generate random starting point
    start = np.random.uniform(10, ell_max / 3)
    
    # Generate predictions with similar spacing distribution to real peaks
    random_predictions = []
    current = start
    
    while len(random_predictions) < n_predictions and current < ell_max:
        # Add some noise to spacing to make it random
        spacing = np.random.normal(mean_spacing, std_spacing)
        current += max(5, spacing)  # Ensure minimum spacing
        
        if current < ell_max:
            random_predictions.append(int(current))
    
    # If we still need more predictions, add some completely random ones
    while len(random_predictions) < n_predictions:
        random_predictions.append(np.random.randint(10, int(ell_max)))
    
    return random_predictions

def process_chunk(ell, power, chunk_indices, actual_peaks, ell_max, n_predictions, base_tolerance):
    """
    Process a chunk of simulations for predictive power test.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        chunk_indices (list): Indices of simulations to process
        actual_peaks (list): List of actual peak positions
        ell_max (float): Maximum ell value
        n_predictions (int): Number of predictions to generate
        base_tolerance (float): Base tolerance for matching
        
    Returns:
        list: Random match rates for this chunk
    """
    # Use process ID and time to seed
    np.random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    
    random_match_rates = []
    
    for sim_idx in chunk_indices:
        # Generate better random predictions
        random_predictions = generate_random_predictions(actual_peaks, n_predictions, ell_max)
        
        # Calculate match rate
        _, match_rate, _, _ = calculate_match_rate(actual_peaks, random_predictions, base_tolerance=base_tolerance)
        random_match_rates.append(match_rate)
        
        # Report progress periodically
        if (sim_idx + 1) % 50 == 0:
            print("  Completed {} simulations in chunk".format(sim_idx + 1))
    
    return random_match_rates

def run_monte_carlo_parallel(ell, power, actual_peaks, gr_predictions, n_simulations=10000, 
                          tolerance=0.10, num_processes=None, timeout_seconds=3600):
    """
    Run predictive power test using Monte Carlo simulations in parallel.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        actual_peaks (list): List of actual peak positions
        gr_predictions (list): List of golden ratio predictions
        n_simulations (int): Number of simulations to run
        tolerance (float): Tolerance for matching
        num_processes (int): Number of processes to use
        timeout_seconds (int): Timeout in seconds
        
    Returns:
        dict: Dictionary with test results
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print("Running Monte Carlo simulations with {} processes...".format(num_processes))
    
    # Calculate match rate for golden ratio predictions
    gr_match_count, gr_match_rate, matched_peaks, matched_predictions = calculate_match_rate(
        actual_peaks, gr_predictions, tolerance=tolerance
    )
    
    # Create a list of all simulation indices
    all_indices = list(range(n_simulations))
    
    # Split indices into chunks for each process
    chunk_size = len(all_indices) // num_processes
    if chunk_size == 0:
        chunk_size = 1
    
    chunks = [all_indices[i:i+chunk_size] for i in range(0, len(all_indices), chunk_size)]
    
    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Submit tasks to the pool
    ell_max = max(ell)
    n_predictions = len(gr_predictions)
    
    results = []
    for chunk in chunks:
        result = pool.apply_async(
            process_chunk, (ell, power, chunk, actual_peaks, ell_max, n_predictions, tolerance)
        )
        results.append(result)
    
    # Close the pool (no more tasks can be submitted)
    pool.close()
    
    # Wait for all tasks to complete or timeout
    start_time = time.time()
    completed_chunks = 0
    
    random_match_rates = []
    
    while completed_chunks < len(chunks):
        if time.time() - start_time > timeout_seconds:
            print("Timeout reached! Terminating...")
            pool.terminate()
            break
        
        # Check if any tasks have completed
        for i, result in enumerate(results):
            if result is not None and result.ready() and result.successful():
                chunk_rates = result.get()
                random_match_rates.extend(chunk_rates)
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
    
    # Calculate statistics
    random_match_rates = np.array(random_match_rates)
    mean_random_rate = np.mean(random_match_rates)
    std_random_rate = np.std(random_match_rates)
    
    # Calculate z-score and p-value
    z_score = (gr_match_rate - mean_random_rate) / (std_random_rate if std_random_rate > 0 else 1.0)
    p_value = 1 - stats.norm.cdf(z_score)
    
    # Calculate ratio of GR to random match rates
    match_ratio = gr_match_rate / mean_random_rate if mean_random_rate > 0 else float('inf')
    
    # Calculate phi optimality
    phi_optimality = 1.0 / (1.0 + np.exp(-10 * (match_ratio - 1.0)))
    
    print("Monte Carlo simulations completed in {:.1f} seconds".format(time.time() - start_time))
    
    return {
        'gr_match_rate': gr_match_rate,
        'mean_random_rate': mean_random_rate,
        'std_random_rate': std_random_rate,
        'z_score': z_score,
        'p_value': p_value,
        'match_ratio': match_ratio,
        'phi_optimality': phi_optimality,
        'matched_peaks': matched_peaks,
        'matched_predictions': matched_predictions
    }

def run_predictive_power_test(ell, power, n_simulations=1000, n_forward=6, n_backward=6, 
                             peak_height=0, peak_distance=10, peak_prominence=0.5, peak_width=None,
                             base_tolerance=0.08, smoothing=True, window_size=5, 
                             num_processes=None, output_dir=None):
    """
    Run the predictive power test.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        n_simulations (int): Number of Monte Carlo simulations
        n_forward (int): Number of forward predictions
        n_backward (int): Number of backward predictions
        peak_height (float): Minimum height of peaks
        peak_distance (int): Minimum distance between peaks
        peak_prominence (float): Minimum prominence of peaks
        peak_width (float): Minimum width of peaks
        base_tolerance (float): Base tolerance for matching
        smoothing (bool): Whether to smooth the data
        window_size (int): Window size for smoothing
        num_processes (int): Number of processes to use
        output_dir (str): Directory to save results
        
    Returns:
        dict: Results of the test
    """
    # Process the data
    processed_power = preprocess_data(power, smooth=smoothing, window_size=window_size)
    
    # Find peaks in the actual data
    print("\nFinding peaks in CMB power spectrum...")
    actual_peaks = find_peaks_in_spectrum(
        ell, processed_power,
        height=peak_height,
        distance=peak_distance,
        prominence=peak_prominence,
        width=peak_width
    )
    print("Found {} peaks in the power spectrum".format(len(actual_peaks)))
    print("Peak locations: {}".format(", ".join(map(str, actual_peaks))))
    
    # Generate golden ratio predictions
    print("\nGenerating golden ratio predictions...")
    gr_predictions = generate_gr_predictions(
        ell, power, n_forward=n_forward, n_backward=n_backward
    )
    
    # Run Monte Carlo simulations
    print("\nRunning {} Monte Carlo simulations...".format(n_simulations))
    
    # Calculate match rate for golden ratio predictions
    print("\nCalculating match rate for golden ratio predictions...")
    gr_matches, gr_match_rate, matched_peaks, matched_predictions = calculate_match_rate(
        actual_peaks, gr_predictions, base_tolerance=base_tolerance
    )
    
    # Run simulations using multiple processes
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print("Using {} processes for simulations".format(num_processes))
    
    # Calculate the number of simulations per process
    chunk_size = n_simulations // num_processes
    extra = n_simulations % num_processes
    
    # Create chunks of indices
    chunks = []
    start = 0
    for i in range(num_processes):
        size = chunk_size + (1 if i < extra else 0)
        end = start + size
        chunks.append(list(range(start, end)))
        start = end
    
    # Create a pool of processes
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Define arguments for each chunk
    chunk_args = [(ell, power, chunk, actual_peaks, max(ell), len(gr_predictions), base_tolerance) 
                  for chunk in chunks]
    
    # Run simulations in parallel - Python 2.7 compatible version
    random_match_rates = []
    try:
        # In Python 2.7, we need to use apply_async or map instead of starmap
        results = []
        for args in chunk_args:
            results.append(pool.apply_async(process_chunk, args))
        
        # Wait for all processes to complete and collect results
        for result in results:
            random_match_rates.extend(result.get())
    finally:
        pool.close()
        pool.join()
    
    # Calculate p-value
    p_value = calculate_p_value(gr_match_rate, random_match_rates)
    
    # Create results dictionary
    results = {
        'ell': ell,
        'power': power,
        'processed_power': processed_power,
        'actual_peaks': actual_peaks,
        'gr_predictions': gr_predictions,
        'gr_match_rate': gr_match_rate,
        'random_match_rates': random_match_rates,
        'p_value': p_value,
        'matched_peaks': matched_peaks,
        'matched_predictions': matched_predictions
    }
    
    # Save results if output directory is provided
    if output_dir:
        save_results(results, output_dir)
    
    # Print summary
    print_summary(results)
    
    return results

def plot_results(ell, power, results, output_dir):
    """
    Create visualizations of predictive power results.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        results (dict): Dictionary with test results
        output_dir (str): Directory to save plots
        
    Returns:
        str: Path to the saved plot
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Power spectrum with matched peaks
    plt.subplot(2, 2, 1)
    plt.plot(ell, power, 'b-', alpha=0.5, label='CMB Power Spectrum')
    
    # Highlight matched peaks
    for peak in results['matched_peaks']:
        idx = np.abs(ell - peak).argmin()
        plt.plot(peak, power[idx], 'ro', markersize=8, label='_nolegend_')
    
    # Add vertical lines for predictions
    for pred in results['matched_predictions']:
        plt.axvline(x=pred, color='g', linestyle='--', alpha=0.5, label='_nolegend_')
    
    plt.xscale('log')
    plt.xlabel('Multipole (l)')
    plt.ylabel('Power')
    plt.title('CMB Power Spectrum with Matched Peaks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Match rate comparison
    plt.subplot(2, 2, 2)
    bars = plt.bar(['Golden Ratio', 'Random'], 
                  [results['gr_match_rate'], results['random_match_rates'].mean()],
                  color=['gold', 'gray'], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                '{:.4f}'.format(height), ha='center', va='bottom')
    plt.ylabel('Match Rate')
    plt.title('Prediction Match Rate Comparison')
    plt.ylim(0, max(results['gr_match_rate'], results['random_match_rates'].mean()) * 1.2)
    
    # Plot 3: Match ratio
    plt.subplot(2, 2, 3)
    plt.bar(['Match Ratio'], [results['gr_match_rate'] / results['random_match_rates'].mean()], color='gold', alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5)
    plt.text(0, results['gr_match_rate'] / results['random_match_rates'].mean() + 0.1, "{:.2f}x".format(results['gr_match_rate'] / results['random_match_rates'].mean()), 
            ha='center', va='bottom')
    plt.ylabel('Ratio')
    plt.title('Golden Ratio vs Random Match Ratio\n(Higher = Better Prediction)')
    
    # Plot 4: P-value and phi optimality
    plt.subplot(2, 2, 4)
    bars = plt.bar(['p-value', 'Phi Optimality'], 
                  [results['p_value'], 1.0 / (1.0 + np.exp(-10 * (results['gr_match_rate'] / results['random_match_rates'].mean() - 1.0)))],
                  color=['blue', 'gold'], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                '{:.4f}'.format(height), ha='center', va='bottom')
    plt.axhline(y=0.05, color='black', linestyle='--', alpha=0.5)
    plt.ylabel('Value')
    plt.title('Statistical Significance and Phi Optimality')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'predictive_power.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def save_results(results, output_dir):
    """
    Save results to files.
    
    Args:
        results (dict): Results dictionary
        output_dir (str): Output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results as text file
    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write("=== PREDICTIVE POWER TEST RESULTS ===\n\n")
        
        # Basic statistics
        f.write("Golden Ratio Match Rate: {:.6f}\n".format(results["gr_match_rate"]))
        f.write("Mean Random Match Rate: {:.6f}\n".format(np.mean(results["random_match_rates"])))
        f.write("Match Ratio: {:.6f}x\n".format(results["gr_match_rate"] / np.mean(results["random_match_rates"])))
        f.write("p-value: {:.6f}\n\n".format(results["p_value"]))
        
        # Detailed peak information
        f.write("Actual Peaks: {}\n".format(", ".join(map(str, results["actual_peaks"]))))
        f.write("GR Predictions: {}\n".format(", ".join(map(str, results["gr_predictions"]))))
        f.write("Matched Peaks: {}\n".format(", ".join(map(str, results["matched_peaks"]))))
        
        # Add interpretation
        f.write("\n=== INTERPRETATION ===\n\n")
        match_ratio = results["gr_match_rate"] / np.mean(results["random_match_rates"])
        
        # Match significance
        if results["p_value"] < 0.01:
            significance = "highly significant"
        elif results["p_value"] < 0.05:
            significance = "significant"
        elif results["p_value"] < 0.1:
            significance = "marginally significant"
        else:
            significance = "not significant"
        
        # Effect size
        if match_ratio > 5:
            effect = "very strong"
        elif match_ratio > 2:
            effect = "strong"
        elif match_ratio > 1.5:
            effect = "moderate"
        elif match_ratio > 1.1:
            effect = "weak"
        else:
            effect = "negligible"
        
        f.write("The golden ratio-based predictions match actual peaks {:.2f}x better ".format(match_ratio))
        f.write("than random predictions, which is {} (p = {:.4f}).\n".format(significance, results["p_value"]))
        f.write("This represents a {} effect size.\n".format(effect))
    
    # Save results as pickle for later analysis
    pickle_path = os.path.join(output_dir, "results.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    
    return results_path, pickle_path

def print_summary(results):
    """
    Print a summary of the results.
    
    Args:
        results (dict): Results dictionary
    """
    mean_random_rate = np.mean(results["random_match_rates"])
    match_ratio = results["gr_match_rate"] / mean_random_rate
    
    print("\n=== SUMMARY ===")
    print("Golden Ratio Match Rate: {:.4f}".format(results["gr_match_rate"]))
    print("Mean Random Match Rate: {:.4f}".format(mean_random_rate))
    print("Match Ratio: {:.4f}x".format(match_ratio))
    print("p-value: {:.4f}".format(results["p_value"]))
    
    if results["p_value"] < 0.05:
        print("\nThe golden ratio predictions perform SIGNIFICANTLY better than random (p < 0.05)")
    else:
        print("\nThe golden ratio predictions do NOT perform significantly better than random (p >= 0.05)")

def calculate_p_value(gr_match_rate, random_match_rates):
    """
    Calculate p-value for the golden ratio match rate.
    
    Args:
        gr_match_rate (float): Golden ratio match rate
        random_match_rates (list): Random match rates
        
    Returns:
        float: p-value
    """
    # Count number of random match rates greater than or equal to golden ratio match rate
    count = sum(1 for rate in random_match_rates if rate >= gr_match_rate)
    
    # Calculate p-value
    p_value = float(count) / len(random_match_rates) if random_match_rates else 1.0
    
    print("\nP-value: {:.6f}".format(p_value))
    
    return p_value

def main():
    """
    Main function to run the predictive power test.
    """
    parser = argparse.ArgumentParser(description="Run predictive power test on CMB power spectrum data")
    parser.add_argument("--wmap", action="store_true", help="Run test on WMAP data")
    parser.add_argument("--planck", action="store_true", help="Run test on Planck data")
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations to run")
    parser.add_argument("--tolerance", type=float, default=0.10, 
                        help="Fractional tolerance for matching (default: 0.10)")
    parser.add_argument("--first-peak", type=float, default=220.0,
                        help="Approximation of the first acoustic peak (default: 220)")
    parser.add_argument("--forward", type=int, default=6,
                        help="Number of forward predictions (default: 6)")
    parser.add_argument("--backward", type=int, default=6,
                        help="Number of backward predictions (default: 6)")
    parser.add_argument("--output", type=str, default="results/predictive_power", 
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
    
    print("\n=== PREDICTIVE POWER TEST ===")
    print("Data: {}".format(args.output))
    print("Simulations: {}".format(args.sims))
    print("Tolerance: {:.2f}".format(args.tolerance))
    print("First peak approximation: {}".format(args.first_peak))
    
    # Load data
    print("\nLoading power spectrum data...")
    if args.wmap:
        wmap_file = "data/wmap/wmap_tt_spectrum_9yr_v5.txt"
        if not os.path.exists(wmap_file):
            print("Error: WMAP power spectrum file not found at {}".format(wmap_file))
            # Try alternative locations
            alternative_wmap_files = [
                "data/wmap_tt_spectrum_9yr_v5.txt",
                "../data/wmap/wmap_tt_spectrum_9yr_v5.txt",
                "../data/wmap_tt_spectrum_9yr_v5.txt"
            ]
            
            for alt_file in alternative_wmap_files:
                if os.path.exists(alt_file):
                    wmap_file = alt_file
                    print("Using alternative WMAP file: {}".format(wmap_file))
                    break
            else:
                print("No WMAP file found. Skipping WMAP analysis.")
                args.wmap = False
        
        if args.wmap:
            print("\n========== ANALYZING WMAP DATA ==========")
            wmap_output_dir = os.path.join(args.output, "wmap")
            ell, power = load_power_spectrum(wmap_file)
            results["wmap"] = run_predictive_power_test(
                ell, power, n_simulations=args.sims, n_forward=args.forward, n_backward=args.backward,
                base_tolerance=args.tolerance, output_dir=wmap_output_dir
            )
    
    # Run test on Planck data
    if args.planck:
        planck_file = "data/planck/planck_tt_spectrum_2018.txt"
        if not os.path.exists(planck_file):
            print("Error: Planck power spectrum file not found at {}".format(planck_file))
            # Try alternative locations
            alternative_planck_files = [
                "data/planck_tt_spectrum_2018.txt",
                "../data/planck/planck_tt_spectrum_2018.txt",
                "../data/planck_tt_spectrum_2018.txt",
                "data/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt",
                "../data/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt"
            ]
            
            for alt_file in alternative_planck_files:
                if os.path.exists(alt_file):
                    planck_file = alt_file
                    print("Using alternative Planck file: {}".format(planck_file))
                    break
            else:
                print("No Planck file found. Skipping Planck analysis.")
                args.planck = False
        
        if args.planck:
            print("\n========== ANALYZING PLANCK DATA ==========")
            planck_output_dir = os.path.join(args.output, "planck")
            ell, power = load_power_spectrum(planck_file)
            results["planck"] = run_predictive_power_test(
                ell, power, n_simulations=args.sims, n_forward=args.forward, n_backward=args.backward,
                base_tolerance=args.tolerance, output_dir=planck_output_dir
            )
    
    # Compare results if both datasets were analyzed
    if args.wmap and args.planck and "wmap" in results and "planck" in results:
        print("\nComparing WMAP and Planck results...")
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        metrics = ["gr_match_rate", "p_value"]
        titles = ["GR Match Rate", "p-value"]
        colors = ['blue', 'red']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(2, 2, i+1)
            values = [results["wmap"][metric], results["planck"][metric]]
            bars = plt.bar(["WMAP", "Planck"], values, color=colors, alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        '{:.4f}'.format(height), ha='center', va='bottom')
            
            if metric == "p_value":
                plt.axhline(y=0.05, color='black', linestyle='--', alpha=0.5)
            
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
    
    print("\nPredictive power test completed.")

if __name__ == "__main__":
    main()
