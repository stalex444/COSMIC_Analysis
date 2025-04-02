#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Laminarity Test for WMAP Cosmic Analysis

This test analyzes the laminarity (smoothness vs. turbulence) of the CMB power spectrum
across different scales. Laminarity is a key metric for identifying structural transitions
in the cosmic microwave background radiation, particularly in relation to the Planck scale.

The test implements parallel processing for Monte Carlo simulations to assess statistical
significance of observed laminarity patterns.
"""

from __future__ import division, print_function
import os
import sys
import argparse
import numpy as np
from datetime import datetime
import multiprocessing
try:
    import Queue  # Python 2
except ImportError:
    import queue as Queue  # Python 3
import time
import pickle
import matplotlib.pyplot as plt

# Constants related to the golden ratio and other mathematical constants
CONSTANTS = {
    'phi': (1 + np.sqrt(5)) / 2,  # Golden ratio
    'e': np.e,                    # Euler's number
    'pi': np.pi,                  # Pi
    'sqrt2': np.sqrt(2),          # Square root of 2
    'sqrt3': np.sqrt(3)           # Square root of 3
}

def load_power_spectrum(file_path):
    """
    Load power spectrum data from a file.
    
    Args:
        file_path (str): Path to the power spectrum file
        
    Returns:
        tuple: (ell, power) arrays
    """
    try:
        data = np.loadtxt(file_path)
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

def calculate_true_phi_optimality(actual_values, surrogate_values, constants=CONSTANTS):
    """
    Calculate phi-optimality for a given set of values against mathematical constants
    by comparing the correlation patterns and comparing to surrogate data.
    
    Args:
        actual_values (numpy.ndarray): Actual values to test
        surrogate_values (list): List of surrogate datasets
        constants (dict): Dictionary of constants to test against
        
    Returns:
        float: Phi-optimality score (1.0 means phi is optimal)
    """
    # For each constant, measure how well it organizes the data compared to surrogates
    optimality = {}
    for name, const in constants.items():
        # Compare pattern of actual values to pattern predicted by this constant
        pattern = np.array([const**i for i in range(len(actual_values))])
        # Calculate correlation between patterns
        correlation = np.corrcoef(actual_values, pattern)[0,1]
        # Compare to surrogate correlations
        surrogate_correlations = []
        for surr in surrogate_values:
            if len(surr) >= len(pattern):  # Ensure surrogate is at least as long as pattern
                surr_corr = np.corrcoef(surr[:len(pattern)], pattern)[0,1]
                if not np.isnan(surr_corr):  # Skip NaN correlations
                    surrogate_correlations.append(surr_corr)
        
        # Calculate optimality as z-score if we have valid surrogate correlations
        if surrogate_correlations:
            surr_mean = np.mean(surrogate_correlations)
            surr_std = np.std(surrogate_correlations)
            if surr_std > 0:  # Avoid division by zero
                optimality[name] = (correlation - surr_mean) / surr_std
            else:
                optimality[name] = 0.0
        else:
            optimality[name] = 0.0
    
    # Find which constant has highest optimality
    best_item = max(optimality.items(), key=lambda x: x[1])
    best_constant = best_item[0]
    best_value = best_item[1]
    
    # Get phi value
    phi_value = optimality.get('phi', 0.0)
    
    # Calculate phi-optimality as how close phi is to the best constant
    if best_constant == 'phi' or best_value <= 0:
        return 1.0  # Phi is optimal (or all values are non-positive)
    else:
        # How close is phi to the best value (normalized)
        return max(0.0, min(1.0, phi_value / best_value))

def calculate_phi_optimality(value, constants=CONSTANTS):
    """
    LEGACY: Calculate phi-optimality for a given value against mathematical constants.
    Kept for backwards compatibility.
    
    Args:
        value (float): Value to test
        constants (dict): Dictionary of constants to test against
        
    Returns:
        dict: Optimality values for each constant
    """
    optimality = {}
    
    for name, const in constants.items():
        # Calculate ratio (value / constant)
        ratio = value / const
        
        # Calculate how close the ratio is to an integer or its reciprocal
        closest_int = round(ratio)
        optimality1 = 1.0 / (1.0 + abs(ratio - closest_int))
        
        # Check reciprocal
        recip_ratio = 1.0 / ratio if ratio != 0 else float('inf')
        closest_recip_int = round(recip_ratio)
        optimality2 = 1.0 / (1.0 + abs(recip_ratio - closest_recip_int))
        
        # Use the better of the two optimalities
        optimality[name] = max(optimality1, optimality2)
    
    return optimality

def segment_data(data, scale, overlap=0.5):
    """
    Segment data into windows of specified scale with overlap.
    
    Args:
        data (numpy.ndarray): Input data array
        scale (int): Window size
        overlap (float): Overlap fraction between windows
        
    Returns:
        numpy.ndarray: Segmented data
    """
    step = int(scale * (1 - overlap))
    n_segments = max(1, (len(data) - scale) // step + 1)
    segments = np.zeros((n_segments, scale))
    
    for i in range(n_segments):
        start = i * step
        end = start + scale
        if end <= len(data):
            segments[i] = data[start:end]
        else:
            # Pad with zeros if needed
            segment = np.zeros(scale)
            segment[:len(data) - start] = data[start:]
            segments[i] = segment
    
    return segments

def calculate_laminarity(segments):
    """
    Calculate laminarity at the given scale.
    
    Laminarity measures the smoothness or turbulence of the data.
    Higher values indicate more laminar (smooth) behavior.
    
    Args:
        segments (numpy.ndarray): Segmented data
        
    Returns:
        float: Laminarity value
    """
    # Calculate variance for each segment
    variances = np.var(segments, axis=1)
    
    # Calculate mean variance
    mean_variance = np.mean(variances)
    
    # Calculate laminarity (inverse of variance)
    laminarity = 1.0 / (1.0 + mean_variance)
    
    return laminarity

def generate_phase_randomized_surrogate(data):
    """
    Generate surrogate with same power spectrum but randomized phases.
    This preserves the spectral properties while randomizing the temporal structure.
    
    Args:
        data (numpy.ndarray): Original data
        
    Returns:
        numpy.ndarray: Phase-randomized surrogate
    """
    # Get FFT
    fft_data = np.fft.rfft(data)
    # Extract amplitude and phase
    amplitude = np.abs(fft_data)
    # Create random phases
    random_phases = np.random.uniform(0, 2*np.pi, len(amplitude))
    # Combine amplitude with random phases
    surrogate_fft = amplitude * np.exp(1j * random_phases)
    # Inverse FFT
    surrogate = np.fft.irfft(surrogate_fft, n=len(data))
    return surrogate

def generate_surrogate_data(data, n_surrogates=100, seed=None):
    """
    Generate surrogate data by phase randomization (preserving power spectrum).
    
    Args:
        data (numpy.ndarray): Original data
        n_surrogates (int): Number of surrogate datasets to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of surrogate datasets
    """
    # Set random seed if provided for reproducibility
    if seed is not None:
        np.random.seed(seed)
    else:
        # Use process ID and time to seed
        np.random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    
    surrogates = []
    for i in range(n_surrogates):
        surrogate = generate_phase_randomized_surrogate(data)
        surrogates.append(surrogate)
    
    return surrogates

def process_chunk(data, chunk_indices, scales):
    """
    Process a chunk of simulations.
    
    Args:
        data (numpy.ndarray): Original data
        chunk_indices (list): Indices of simulations to process
        scales (list): Scales to analyze
        
    Returns:
        tuple: (Results for each simulation, Optimality for each scale)
    """
    # Use process ID and time to seed
    np.random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    
    # Process each simulation
    results = []
    optimality_scores = {scale: [] for scale in scales}
    
    for sim_idx in chunk_indices:
        # Generate surrogate data for this simulation
        surrogates = generate_surrogate_data(data, n_surrogates=1)
        surrogate = surrogates[0]
        
        # Calculate laminarity for each scale
        for scale in scales:
            segments = segment_data(surrogate, scale, overlap=0.5)
            lam = calculate_laminarity(segments)
            optimality_scores[scale].append(lam)
        
        # Report progress periodically
        if sim_idx % 50 == 0 and sim_idx > 0:
            print("  Completed {} simulations in chunk".format(sim_idx))
            
    return results, optimality_scores

def run_monte_carlo_parallel(data, n_simulations=10000, scales=None, num_processes=None, timeout_seconds=3600):
    """
    Run Monte Carlo simulations in parallel to assess the significance of laminarity patterns.
    
    Args:
        data (numpy.ndarray): Original data
        n_simulations (int): Number of simulations to run
        scales (list): List of scales to analyze
        num_processes (int): Number of processes to use (default: number of CPU cores)
        timeout_seconds (int): Timeout in seconds
        
    Returns:
        tuple: (p-values for each scale, phi-optimality for each scale, actual laminarity for each scale)
    """
    start_time = time.time()
    
    if scales is None:
        scales = list(range(10, 101, 10))
    
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print("Running {} Monte Carlo simulations with {} processes".format(n_simulations, num_processes))
    
    # Calculate actual laminarity for each scale
    actual_laminarity = {}
    actual_lams_by_scale = {}
    for scale in scales:
        segments = segment_data(data, scale, overlap=0.5)
        lam = calculate_laminarity(segments)
        actual_laminarity[scale] = lam
        # Store the actual laminarity value for the scale
        actual_lams_by_scale[scale] = lam
    
    # Generate surrogate datasets for true phi-optimality calculation
    print("Generating surrogate datasets for phi-optimality calculation...")
    surrogate_datasets = generate_surrogate_data(data, n_surrogates=100)
    
    # Calculate true phi-optimality
    phi_optimality = {}
    for scale in scales:
        surrogate_values = []
        for surrogate in surrogate_datasets:
            segments = segment_data(surrogate, scale, overlap=0.5)
            lam = calculate_laminarity(segments)
            surrogate_values.append(lam)
        
        # Convert to numpy arrays of consistent shape
        actual_value = np.array([actual_lams_by_scale[scale]])
        phi_opt = calculate_true_phi_optimality(actual_value, [surrogate_values], constants=CONSTANTS)
        phi_optimality[scale] = phi_opt
    
    # Divide the simulations among the processes
    chunk_size = n_simulations // num_processes
    if chunk_size < 1:
        chunk_size = 1
    
    chunks = []
    for i in range(0, n_simulations, chunk_size):
        end = min(i + chunk_size, n_simulations)
        chunks.append(list(range(i, end)))
    
    # Create a queue for results
    result_queue = multiprocessing.Queue()
    
    # Create and start processes
    processes = []
    
    for i, chunk in enumerate(chunks):
        # Use a wrapper function to avoid issues with pickle
        def worker_func(q, chunk_idx, data_copy, chunk_copy, scales_copy):
            try:
                chunk_results, optimality_scores = process_chunk(data_copy, chunk_copy, scales_copy)
                q.put((chunk_idx, chunk_results, optimality_scores))
            except Exception as e:
                print("Worker process error: {}".format(str(e)))
                q.put((chunk_idx, [], {}))
        
        process = multiprocessing.Process(
            target=worker_func,
            args=(result_queue, i, data.copy(), chunk, scales)
        )
        processes.append(process)
        process.start()
    
    # Collect results
    results = [None] * len(chunks)
    completed_chunks = 0
    start_time = time.time()
    
    # Initialize counters for each scale
    sim_values = {scale: [] for scale in scales}
    
    print("Waiting for processes to complete...")
    while completed_chunks < len(chunks):
        if time.time() - start_time > timeout_seconds:
            print("Timeout after {} seconds".format(timeout_seconds))
            for process in processes:
                if process.is_alive():
                    process.terminate()
            break
        
        # Get next completed chunk
        try:
            # Use a timeout to avoid blocking indefinitely
            chunk_idx, chunk_results, optimality_scores = result_queue.get(timeout=10)
            
            # Add optimality scores to our collected values
            for scale in scales:
                if scale in optimality_scores:
                    sim_values[scale].extend(optimality_scores[scale])
            
            results[chunk_idx] = chunk_results
            completed_chunks += 1
            
            print("Completed chunk {} of {} ({:.1f}%)".format(
                completed_chunks, len(chunks), 100.0 * completed_chunks / len(chunks)))
        except Queue.Empty:
            # No results yet, just continue waiting
            continue
        except Exception as e:
            print("Error collecting results: {}".format(str(e)))
    
    # Wait for processes to finish
    for process in processes:
        if process.is_alive():
            process.join(1)
    
    # Calculate p-values for each scale
    p_values = {}
    for scale in scales:
        # Count how many simulations had higher laminarity than actual data
        count = sum(1 for sim_val in sim_values[scale] if sim_val >= actual_laminarity[scale])
        # Calculate p-value
        p_values[scale] = count / float(len(sim_values[scale])) if sim_values[scale] else 1.0
    
    print("Monte Carlo simulations completed in {:.1f} seconds".format(time.time() - start_time))
    
    return p_values, phi_optimality, actual_laminarity

def plot_laminarity_results(scales, p_values, phi_optimality, actual_laminarity, output_dir, dataset_name):
    """
    Plot the results of the laminarity test.
    
    Args:
        scales (list): Scales analyzed
        p_values (dict): p-values for each scale
        phi_optimality (dict): Phi-optimality for each scale
        actual_laminarity (dict): Actual laminarity for each scale
        output_dir (str): Output directory for plots
        dataset_name (str): Name of the dataset
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot p-values
    plt.figure(figsize=(10, 6))
    plt.plot(scales, [p_values[scale] for scale in scales], 'o-')
    plt.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
    plt.xlabel('Scale')
    plt.ylabel('p-value')
    plt.title('Laminarity p-values for {}'.format(dataset_name))
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'p_values.png'))
    plt.close()
    
    # Plot phi-optimality
    plt.figure(figsize=(10, 6))
    plt.plot(scales, [phi_optimality[scale] for scale in scales], 'o-')
    plt.xlabel('Scale')
    plt.ylabel('Phi-optimality')
    plt.title('Phi-optimality for {}'.format(dataset_name))
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'phi_optimality.png'))
    plt.close()
    
    # Plot actual laminarity
    plt.figure(figsize=(10, 6))
    plt.plot(scales, [actual_laminarity[scale] for scale in scales], 'o-')
    plt.xlabel('Scale')
    plt.ylabel('Laminarity')
    plt.title('Actual laminarity for {}'.format(dataset_name))
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'actual_laminarity.png'))
    plt.close()
    
    # Write results to CSV
    with open(os.path.join(output_dir, 'results.csv'), 'w') as f:
        f.write('Scale,p-value,phi-optimality,actual-laminarity\n')
        for scale in scales:
            f.write('{},{},{},{}\n'.format(
                scale, 
                p_values[scale], 
                phi_optimality[scale], 
                actual_laminarity[scale]
            ))
            
    # Save summary to text file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write('Laminarity Test Results for {}\n'.format(dataset_name))
        f.write('=' * 50 + '\n\n')
        
        f.write('Scales analyzed: {}\n\n'.format(scales))
        
        # Find significant scales
        significant_scales = [scale for scale in scales if p_values[scale] < 0.05]
        if significant_scales:
            f.write('Significant scales (p < 0.05): {}\n\n'.format(significant_scales))
        else:
            f.write('No significant scales found.\n\n')
        
        # Find phi-optimal scales
        optimal_threshold = 0.9
        optimal_scales = [scale for scale in scales if phi_optimality[scale] > optimal_threshold]
        if optimal_scales:
            f.write('Phi-optimal scales (optimality > {}): {}\n\n'.format(
                optimal_threshold, optimal_scales))
        else:
            f.write('No phi-optimal scales found.\n\n')
        
        # Display detailed results
        f.write('Detailed results:\n')
        f.write('{:<10} {:<15} {:<15} {:<15}\n'.format(
            'Scale', 'p-value', 'phi-optimality', 'actual-laminarity'))
        f.write('-' * 55 + '\n')
        
        for scale in scales:
            f.write('{:<10} {:<15.6f} {:<15.6f} {:<15.6f}\n'.format(
                scale, 
                p_values[scale], 
                phi_optimality[scale], 
                actual_laminarity[scale]
            ))

def run_laminarity_test(ell, power, output_dir, dataset_name, n_simulations=10000, scales=None, timeout_seconds=3600, parallel=True, num_processes=None):
    """
    Run the laminarity test on power spectrum data.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        output_dir (str): Output directory for results
        dataset_name (str): Name of the dataset
        n_simulations (int): Number of simulations to run
        scales (list): List of scales to analyze
        timeout_seconds (int): Maximum time in seconds
        parallel (bool): Whether to use parallel processing
        num_processes (int): Number of processes to use
        
    Returns:
        dict: Results of the laminarity test
    """
    start_time = time.time()
    
    print("Running laminarity test for {} with {} simulations...".format(dataset_name, n_simulations))
    
    # Preprocess data
    processed_power = preprocess_data(power, log_transform=True, normalize=True)
    
    # Generate scales if not provided
    if scales is None:
        scales = generate_scales(min_scale=2, max_scale=min(2048, len(processed_power) // 2), n_scales=10)
    
    print("Analyzing scales: {}".format(scales))
    
    # Run Monte Carlo simulations
    if parallel:
        p_values, phi_optimality, actual_laminarity = run_monte_carlo_parallel(
            processed_power, 
            n_simulations=n_simulations,
            scales=scales,
            num_processes=num_processes,
            timeout_seconds=timeout_seconds
        )
    else:
        print("Using serial processing (not recommended for large simulations)")
        # Implement serial version if needed
        raise NotImplementedError("Serial processing not implemented")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot results
    plot_laminarity_results(scales, p_values, phi_optimality, actual_laminarity, output_dir, dataset_name)
    
    # Save results for later use
    results = {
        'dataset_name': dataset_name,
        'scales': scales,
        'p_values': p_values,
        'phi_optimality': phi_optimality,
        'actual_laminarity': actual_laminarity,
        'n_simulations': n_simulations,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("Laminarity test completed in {:.2f} seconds".format(time.time() - start_time))
    print("Results saved to {}".format(output_dir))
    
    return results

def generate_scales(min_scale=2, max_scale=2048, n_scales=10):
    """
    Generate scales for analysis based on Fibonacci sequence.
    
    Args:
        min_scale (int): Minimum scale
        max_scale (int): Maximum scale
        n_scales (int): Number of scales
        
    Returns:
        list: List of scales
    """
    # Generate Fibonacci sequence
    fib_seq = [0, 1]
    while len(fib_seq) < n_scales + 5:
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    
    # Filter and scale to desired range
    fib_seq = [f for f in fib_seq if f >= 1]  # Remove 0
    
    # Scale to desired range
    min_fib = min(fib_seq)
    max_fib = max(fib_seq)
    
    scales = [int(min_scale + (max_scale - min_scale) * (f - min_fib) / (max_fib - min_fib)) for f in fib_seq]
    
    # Remove duplicates and sort
    scales = sorted(list(set(scales)))
    
    # Ensure min and max scales are included
    if scales[0] > min_scale:
        scales = [min_scale] + scales
    if scales[-1] < max_scale:
        scales = scales + [max_scale]
    
    # Limit to n_scales
    if len(scales) > n_scales:
        indices = np.linspace(0, len(scales) - 1, n_scales).astype(int)
        scales = [scales[i] for i in indices]
    
    return scales

def main():
    """Main function to run the laminarity test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run laminarity test on cosmic microwave background data.')
    parser.add_argument('--wmap-file', type=str, default="../data/wmap_tt_spectrum_9yr_v5.txt",
                        help='Path to WMAP power spectrum file')
    parser.add_argument('--planck-file', type=str, default="../data/planck_tt_spectrum_2018.txt",
                        help='Path to Planck power spectrum file')
    parser.add_argument('--output-dir', type=str, default="../results/laminarity_test",
                        help='Output directory for results')
    parser.add_argument('--n-simulations', type=int, default=10000,
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Maximum time in seconds to spend on simulations')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--num-processes', type=int, default=None,
                        help='Number of processes to use for parallelization')
    parser.add_argument('--min-scale', type=int, default=2,
                        help='Minimum scale for analysis')
    parser.add_argument('--max-scale', type=int, default=2048,
                        help='Maximum scale for analysis')
    parser.add_argument('--n-scales', type=int, default=10,
                        help='Number of scales for analysis')
    parser.add_argument('--wmap-only', action='store_true',
                        help='Run test only on WMAP data, ignoring Planck data')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load WMAP power spectrum
    print("Loading WMAP power spectrum...")
    try:
        wmap_ell, wmap_power = load_power_spectrum(args.wmap_file)
        print("Loaded WMAP power spectrum with {} multipoles".format(len(wmap_ell)))
        
        # Generate scales
        scales = generate_scales(min_scale=args.min_scale, 
                               max_scale=min(args.max_scale, len(wmap_power) // 2),
                               n_scales=args.n_scales)
        
        # Run laminarity test on WMAP data
        print("Running laminarity test on WMAP data...")
        wmap_results = run_laminarity_test(
            wmap_ell, 
            wmap_power, 
            os.path.join(args.output_dir, 'wmap'), 
            'WMAP', 
            n_simulations=args.n_simulations,
            scales=scales,
            timeout_seconds=args.timeout,
            parallel=not args.no_parallel,
            num_processes=args.num_processes
        )
    except Exception as e:
        print("Error processing WMAP data: {}".format(str(e)))
        print("Cannot continue without WMAP data.")
        return 1
    
    # Skip Planck data if --wmap-only is specified
    if args.wmap_only:
        print("\nSkipping Planck data analysis as requested.")
        print("\nLaminarity test completed successfully.")
        print("Results saved to {}".format(args.output_dir))
        return 0
    
    # Load Planck power spectrum
    print("\nLoading Planck power spectrum...")
    planck_results = None
    try:
        if not os.path.exists(args.planck_file):
            print("Warning: Planck data file not found at {}".format(args.planck_file))
            print("Checking for alternative locations...")
            
            # Try alternative locations
            alt_locations = [
                "../data/planck_power_spectrum.txt",
                "../data/planck_tt_spectrum_2018.txt",
                "../data/planck_power_spectrum.txt"
            ]
            
            for alt_file in alt_locations:
                if os.path.exists(alt_file):
                    print("Found Planck data at {}".format(alt_file))
                    args.planck_file = alt_file
                    break
            
            if not os.path.exists(args.planck_file):
                print("No Planck data file found. Continuing with WMAP results only.")
                print("\nLaminarity test completed successfully with WMAP data only.")
                print("Results saved to {}".format(args.output_dir))
                return 0
        
        planck_ell, planck_power = load_power_spectrum(args.planck_file)
        print("Loaded Planck power spectrum with {} multipoles".format(len(planck_ell)))
        
        # Run laminarity test on Planck data
        print("Running laminarity test on Planck data...")
        planck_results = run_laminarity_test(
            planck_ell, 
            planck_power, 
            os.path.join(args.output_dir, 'planck'), 
            'Planck', 
            n_simulations=args.n_simulations,
            scales=scales,
            timeout_seconds=args.timeout,
            parallel=not args.no_parallel,
            num_processes=args.num_processes
        )
    except Exception as e:
        print("Error processing Planck data: {}".format(str(e)))
        print("Continuing with WMAP results only.")
    
    # Compare WMAP and Planck results if both are available
    if planck_results:
        print("\nComparing WMAP and Planck results...")
        
        # Create output directory for comparison
        comparison_dir = os.path.join(args.output_dir, 'comparison')
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)
        
        # Plot comparison of laminarity
        plt.figure(figsize=(10, 6))
        plt.plot(scales, [wmap_results['actual_laminarity'][scale] for scale in scales], 
                'o-', color='blue', label='WMAP')
        plt.plot(scales, [planck_results['actual_laminarity'][scale] for scale in scales], 
                'o-', color='red', label='Planck')
        plt.xlabel('Scale')
        plt.ylabel('Laminarity')
        plt.title('Laminarity Comparison: WMAP vs Planck')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(comparison_dir, 'laminarity_comparison.png'))
        plt.close()
        
        # Plot comparison of p-values
        plt.figure(figsize=(10, 6))
        plt.plot(scales, [wmap_results['p_values'][scale] for scale in scales], 
                'o-', color='blue', label='WMAP')
        plt.plot(scales, [planck_results['p_values'][scale] for scale in scales], 
                'o-', color='red', label='Planck')
        plt.axhline(y=0.05, color='gray', linestyle='--', label='p=0.05')
        plt.xlabel('Scale')
        plt.ylabel('p-value')
        plt.title('p-value Comparison: WMAP vs Planck')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(comparison_dir, 'p_value_comparison.png'))
        plt.close()
        
        # Plot comparison of phi-optimality
        plt.figure(figsize=(10, 6))
        plt.plot(scales, [wmap_results['phi_optimality'][scale] for scale in scales], 
                'o-', color='blue', label='WMAP')
        plt.plot(scales, [planck_results['phi_optimality'][scale] for scale in scales], 
                'o-', color='red', label='Planck')
        plt.xlabel('Scale')
        plt.ylabel('Phi-optimality')
        plt.title('Phi-optimality Comparison: WMAP vs Planck')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(comparison_dir, 'phi_optimality_comparison.png'))
        plt.close()
    
    print("\nLaminarity test completed successfully.")
    print("Results saved to {}".format(args.output_dir))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
