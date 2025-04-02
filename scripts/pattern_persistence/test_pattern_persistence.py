#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pattern Persistence Test Module.

This test examines how consistent golden ratio patterns are across different
subsets of the CMB power spectrum data, compared to random patterns.
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

def calculate_pattern_strength(ell, power, pattern_ratio, tolerance=0.1):
    """
    Calculate the strength of a specific pattern ratio in the data.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        pattern_ratio (float): The ratio to test for (e.g., phi)
        tolerance (float): Tolerance for considering a ratio match
        
    Returns:
        float: Pattern strength (0-1)
    """
    pattern_strengths = []
    
    for i in range(len(ell) - 1):
        for j in range(i + 1, len(ell)):
            # Check if the ratio between multipoles is close to the pattern ratio
            ell_ratio = ell[j] / ell[i]
            if (1-tolerance) * pattern_ratio <= ell_ratio <= (1+tolerance) * pattern_ratio:
                # Calculate power ratio, avoiding division by zero
                if power[i] != 0:
                    power_ratio = power[j] / power[i]
                    # Measure how close the power ratio is to 1 (perfect correlation)
                    # or to pattern_ratio (perfect scaling relationship)
                    strength1 = 1.0 / (1.0 + abs(power_ratio - 1.0))
                    strength2 = 1.0 / (1.0 + abs(power_ratio - pattern_ratio))
                    # Take the maximum of these two measures
                    pattern_strengths.append(max(strength1, strength2))
    
    # Return the mean strength if any patterns were found
    return np.mean(pattern_strengths) if pattern_strengths else 0.0

def process_chunk(ell, power, chunk_indices, num_patterns=20, subset_fraction=0.8, phi=CONSTANTS['phi']):
    """
    Process a chunk of simulations for pattern persistence test.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        chunk_indices (list): Indices of simulations to process
        num_patterns (int): Number of random patterns to test per subset
        subset_fraction (float): Fraction of data to use in each subset
        phi (float): Golden ratio value
        
    Returns:
        tuple: (GR strengths, random strengths) for this chunk
    """
    # Use process ID and time to seed
    np.random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    
    # Number of subsets to test
    num_subsets = 10
    
    gr_strengths = []
    random_strengths = []
    
    for sim_idx in chunk_indices:
        sim_gr_strengths = []
        sim_random_strengths = []
        
        # Generate surrogate data for this simulation if not the first simulation (actual data)
        if sim_idx > 0:
            power = generate_phase_randomized_surrogate(power)
        
        for i in range(num_subsets):
            # Create a random subset of the data
            subset_size = int(subset_fraction * len(ell))
            subset_indices = np.random.choice(len(ell), size=subset_size, replace=False)
            subset_indices.sort()  # Keep indices in order
            
            subset_ell = ell[subset_indices]
            subset_power = power[subset_indices]
            
            # Calculate golden ratio pattern strength
            gr_strength = calculate_pattern_strength(subset_ell, subset_power, phi)
            sim_gr_strengths.append(gr_strength)
            
            # Calculate random pattern strengths for comparison
            subset_random_strengths = []
            for _ in range(num_patterns):
                # Use a random ratio between 1.1 and 2.5
                random_ratio = 1.1 + 1.4 * np.random.random()
                random_strength = calculate_pattern_strength(subset_ell, subset_power, random_ratio)
                subset_random_strengths.append(random_strength)
            
            # Average the random pattern strengths for this subset
            sim_random_strengths.append(np.mean(subset_random_strengths))
        
        # Average across subsets
        gr_strengths.append(np.mean(sim_gr_strengths))
        random_strengths.append(np.mean(sim_random_strengths))
        
        # Report progress periodically
        if sim_idx % 50 == 0 and sim_idx > 0:
            print("  Completed {} simulations in chunk".format(sim_idx))
    
    return gr_strengths, random_strengths

def calculate_statistics(actual_gr_strength, sim_gr_strengths, sim_random_strengths):
    """
    Calculate statistical significance and persistence metrics.
    
    Args:
        actual_gr_strength (float): Golden ratio pattern strength for actual data
        sim_gr_strengths (list): Golden ratio pattern strengths for simulations
        sim_random_strengths (list): Random pattern strengths for simulations
        
    Returns:
        dict: Dictionary with statistical results
    """
    # Convert to numpy arrays
    sim_gr_strengths = np.array(sim_gr_strengths)
    sim_random_strengths = np.array(sim_random_strengths)
    
    # Calculate means
    mean_gr_strength = actual_gr_strength
    mean_random_strength = np.mean(sim_random_strengths[0])  # For actual data
    
    # Calculate strength ratio
    strength_ratio = mean_gr_strength / mean_random_strength if mean_random_strength > 0 else float('inf')
    
    # Calculate statistical significance (p-value)
    # The p-value is the fraction of simulations with GR strength >= actual
    p_value = np.mean(sim_gr_strengths[1:] >= actual_gr_strength)
    
    # Calculate Z-score 
    z_score = (actual_gr_strength - np.mean(sim_gr_strengths[1:])) / (np.std(sim_gr_strengths[1:]) or 1.0)
    
    # Calculate persistence (inverse of coefficient of variation)
    gr_variance = np.var(sim_gr_strengths)
    random_variance = np.var(sim_random_strengths.mean(axis=1))
    
    # Persistence ratio (lower GR variance means higher persistence)
    persistence_ratio = random_variance / gr_variance if gr_variance > 0 else float('inf')
    
    # Calculate phi optimality (normalized measure of how optimal phi is)
    phi_optimality = 1.0 / (1.0 + np.exp(-10 * (strength_ratio - 1.0)))
    
    return {
        'actual_gr_strength': actual_gr_strength,
        'mean_random_strength': mean_random_strength,
        'strength_ratio': strength_ratio,
        'p_value': p_value,
        'z_score': z_score, 
        'persistence_ratio': persistence_ratio,
        'phi_optimality': phi_optimality
    }

def run_monte_carlo_parallel(ell, power, n_simulations=10000, num_processes=None, timeout_seconds=3600):
    """
    Run pattern persistence test using Monte Carlo simulations in parallel.
    
    Args:
        ell (numpy.ndarray): Multipole values
        power (numpy.ndarray): Power spectrum values
        n_simulations (int): Number of simulations to run
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
        result = pool.apply_async(process_chunk, (ell, processed_power, chunk))
        results.append(result)
    
    # Close the pool (no more tasks can be submitted)
    pool.close()
    
    # Wait for all tasks to complete or timeout
    start_time = time.time()
    completed_chunks = 0
    
    gr_strengths_list = []
    random_strengths_list = []
    
    while completed_chunks < len(chunks):
        if time.time() - start_time > timeout_seconds:
            print("Timeout reached! Terminating...")
            pool.terminate()
            break
        
        # Check if any tasks have completed
        for i, result in enumerate(results):
            if result is not None and result.ready() and result.successful():
                gr_strengths, random_strengths = result.get()
                gr_strengths_list.extend(gr_strengths)
                random_strengths_list.extend(random_strengths)
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
    # Index 0 is for the actual data
    try:
        actual_indices = [i for i, idx in enumerate(all_indices) if idx == 0]
        if actual_indices:
            actual_idx = actual_indices[0]
            actual_gr_strength = gr_strengths_list[actual_idx]
            
            # Reshape into (n_simulations+1, values per simulation)
            gr_strengths_array = np.array(gr_strengths_list).reshape(-1, 1)
            random_strengths_array = np.array(random_strengths_list).reshape(-1, 1)
            
            # Calculate statistics
            stats = calculate_statistics(actual_gr_strength, gr_strengths_array, random_strengths_array)
            
            print("Monte Carlo simulations completed in {:.1f} seconds".format(time.time() - start_time))
            return stats
        else:
            print("Error: Could not find actual data results (index 0)")
            return None
    except Exception as e:
        print("Error processing results: {}".format(str(e)))
        return None

def plot_results(results, output_dir):
    """
    Create visualizations of pattern persistence results.
    
    Args:
        results (dict): Dictionary with test results
        output_dir (str): Directory to save plots
        
    Returns:
        str: Path to the saved plot
    """
    plt.figure(figsize=(12, 10))
    
    # Create a 2x2 grid
    plt.subplot(2, 2, 1)
    bars = plt.bar(['Golden Ratio', 'Random Patterns'], 
                  [results['actual_gr_strength'], results['mean_random_strength']],
                  color=['gold', 'gray'], alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                '{:.4f}'.format(height), ha='center', va='bottom')
    plt.ylabel('Pattern Strength')
    plt.title('Pattern Strength Comparison')
    
    plt.subplot(2, 2, 2)
    plt.bar(['Pattern Persistence'], [results['persistence_ratio']], color='gold', alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5)
    plt.text(0, results['persistence_ratio'] + 0.1, "{:.4f}x".format(results['persistence_ratio']), 
            ha='center', va='bottom')
    plt.ylabel('Persistence Ratio')
    plt.title('Pattern Persistence Ratio\n(Higher = More Consistent)')
    
    plt.subplot(2, 2, 3)
    plt.bar(['Strength Ratio'], [results['strength_ratio']], color='gold', alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5)
    plt.text(0, results['strength_ratio'] + 0.1, "{:.4f}x".format(results['strength_ratio']), 
            ha='center', va='bottom')
    plt.ylabel('Strength Ratio')
    plt.title('Golden Ratio vs Random Patterns\n(Higher = Stronger GR Pattern)')
    
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
    
    output_path = os.path.join(output_dir, 'pattern_persistence.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def run_pattern_persistence_test(data_path, output_base_dir, n_simulations=10000):
    """
    Run pattern persistence test on power spectrum data.
    
    Args:
        data_path (str): Path to the power spectrum data
        output_base_dir (str): Base directory for output
        n_simulations (int): Number of simulations to run
        
    Returns:
        dict: Test results
    """
    start_time = time.time()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, "pattern_persistence_{}_{}".format(n_simulations, timestamp))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n=== PATTERN PERSISTENCE TEST ===")
    print("Data: {}".format(data_path))
    print("Output: {}".format(output_dir))
    print("Simulations: {}".format(n_simulations))
    
    # Load data
    print("\nLoading power spectrum data...")
    ell, power = load_power_spectrum(data_path)
    print("Loaded {} data points".format(len(ell)))
    
    # Run Monte Carlo simulations
    print("\nRunning pattern persistence test...")
    results = run_monte_carlo_parallel(ell, power, n_simulations=n_simulations)
    
    if results:
        # Plot results
        print("\nCreating visualizations...")
        plot_path = plot_results(results, output_dir)
        
        # Save results
        results_path = os.path.join(output_dir, "results.txt")
        with open(results_path, "w") as f:
            f.write("=== PATTERN PERSISTENCE TEST RESULTS ===\n\n")
            f.write("Data: {}\n".format(data_path))
            f.write("Simulations: {}\n\n".format(n_simulations))
            f.write("Golden Ratio Pattern Strength: {:.6f}\n".format(results["actual_gr_strength"]))
            f.write("Random Patterns Strength: {:.6f}\n".format(results["mean_random_strength"]))
            f.write("Strength Ratio: {:.6f}x\n".format(results["strength_ratio"]))
            f.write("Pattern Persistence Ratio: {:.6f}x\n".format(results["persistence_ratio"]))
            f.write("Phi Optimality: {:.6f}\n".format(results["phi_optimality"]))
            f.write("p-value: {:.6f}\n".format(results["p_value"]))
            f.write("z-score: {:.6f}\n\n".format(results["z_score"]))
            
            # Add interpretation
            f.write("=== INTERPRETATION ===\n\n")
            
            # Strength significance
            if results["p_value"] < 0.01:
                significance = "highly significant"
            elif results["p_value"] < 0.05:
                significance = "significant"
            elif results["p_value"] < 0.1:
                significance = "marginally significant"
            else:
                significance = "not significant"
            
            # Effect size
            if results["strength_ratio"] > 2:
                effect = "strong"
            elif results["strength_ratio"] > 1.5:
                effect = "moderate"
            elif results["strength_ratio"] > 1.1:
                effect = "weak"
            else:
                effect = "negligible"
            
            # Persistence
            if results["persistence_ratio"] > 2:
                persistence = "highly persistent"
            elif results["persistence_ratio"] > 1.5:
                persistence = "moderately persistent"
            elif results["persistence_ratio"] > 1.1:
                persistence = "slightly persistent"
            else:
                persistence = "not persistent"
            
            f.write("The golden ratio pattern is {} times stronger than random patterns, ".format(round(results["strength_ratio"], 2)))
            f.write("which is {} (p = {:.4f}).\n".format(significance, results["p_value"]))
            f.write("The pattern shows a {} effect size and is {}.\n".format(effect, persistence))
            
            if results["phi_optimality"] > 0.9:
                f.write("The golden ratio appears to be highly optimal for organizing patterns in this data.\n")
            elif results["phi_optimality"] > 0.7:
                f.write("The golden ratio appears to be moderately optimal for organizing patterns in this data.\n")
            elif results["phi_optimality"] > 0.5:
                f.write("The golden ratio shows some optimality for organizing patterns in this data.\n")
            else:
                f.write("The golden ratio does not appear to be optimal for organizing patterns in this data.\n")
        
        # Save results as pickle for later analysis
        pickle_path = os.path.join(output_dir, "results.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)
        
        print("\nPattern persistence test completed in {:.2f} seconds".format(time.time() - start_time))
        print("Results saved to {}".format(output_dir))
        
        return {
            "results": results,
            "output_dir": output_dir,
            "plot_path": plot_path,
            "results_path": results_path
        }
    else:
        print("\nError: Pattern persistence test failed!")
        return None

def main():
    """
    Main function to run the pattern persistence test.
    """
    parser = argparse.ArgumentParser(description="Run pattern persistence test on CMB power spectrum data")
    parser.add_argument("--wmap", action="store_true", help="Run test on WMAP data")
    parser.add_argument("--planck", action="store_true", help="Run test on Planck data")
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations to run")
    parser.add_argument("--output", type=str, default="results/pattern_persistence", 
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
            results["wmap"] = run_pattern_persistence_test(
                wmap_file, wmap_output_dir, n_simulations=args.sims)
    
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
            results["planck"] = run_pattern_persistence_test(
                planck_file, planck_output_dir, n_simulations=args.sims)
    
    # Compare results if both datasets were analyzed
    if args.wmap and args.planck and "wmap" in results and "planck" in results:
        print("\nComparing WMAP and Planck results...")
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        metrics = ["strength_ratio", "persistence_ratio", "phi_optimality", "p_value"]
        titles = ["Strength Ratio", "Persistence Ratio", "Phi Optimality", "p-value"]
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
            else:
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
    
    print("\nPattern persistence test completed.")

if __name__ == "__main__":
    main()
