#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Information Architecture Test Runner (1,000 Simulations)
This script runs the Information Architecture test with 1,000 simulations
with optimized performance and enhanced progress tracking.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import the Information Architecture Test
from scripts.info_architecture.archive.information_architecture_test import InformationArchitectureTest, load_wmap_power_spectrum, load_planck_power_spectrum

def ensure_dir_exists(dir_path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Standalone functions for parallel processing
def _run_single_simulation(data, constant, seed, config, result_queue, sim_index):
    """Run a single simulation for parallel processing with progress tracking."""
    try:
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
        start_time = time.time()
        score = test.calculate_architecture_score(surrogate_data, constant)
        elapsed_time = time.time() - start_time
        
        # Put result in queue
        result_queue.put((sim_index, score, elapsed_time))
        
        return score
    except Exception as e:
        # Log error and return a default score
        sys.stderr.write("Error in simulation %d: %s\n" % (sim_index, str(e)))
        result_queue.put((sim_index, 0.0, 0.0))
        return 0.0

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
    
    print("Calculating actual architecture score...")
    start_time = time.time()
    actual_score = test.calculate_architecture_score(data, constant)
    score_time = time.time() - start_time
    print("Actual score: %.6f (calculated in %.1f seconds)" % (actual_score, score_time))
    
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
        f.write("# Simulation,Score,p-value,time\n")
    
    # Count how many random scores are >= actual score
    count_greater_equal = 0
    all_scores = []
    
    # Use parallel processing
    overall_start_time = time.time()
    last_update_time = overall_start_time
    
    # Create a pool of workers
    num_workers = min(cpu_count(), config.get('max_workers', cpu_count()))
    
    # Create a manager for shared queue
    manager = Manager()
    result_queue = manager.Queue()
    
    # Create and start worker processes
    pool = Pool(processes=num_workers)
    
    # Track progress
    completed = 0
    batch_size = min(100, num_simulations)  # Process in smaller batches
    
    try:
        # Process in batches to provide more frequent updates
        for batch_start in range(0, num_simulations, batch_size):
            batch_end = min(batch_start + batch_size, num_simulations)
            batch_size_actual = batch_end - batch_start
            
            print("Processing batch %d-%d of %d simulations..." % 
                  (batch_start, batch_end-1, num_simulations))
            
            # Prepare arguments for this batch
            args = [(data, constant, i, config, result_queue, i) 
                   for i in range(batch_start, batch_end)]
            
            # Start the batch of simulations asynchronously
            pool.starmap_async(_run_single_simulation, args)
            
            # Process results as they come in
            batch_completed = 0
            while batch_completed < batch_size_actual:
                try:
                    # Get result with timeout to allow for keyboard interrupts
                    sim_index, score, sim_time = result_queue.get(timeout=1.0)
                    
                    # Process the result
                    all_scores.append(score)
                    if score >= actual_score:
                        count_greater_equal += 1
                    
                    # Calculate p-value
                    p_value = float(count_greater_equal) / (completed + batch_completed + 1)
                    
                    # Update progress file
                    with open(progress_file, 'a') as f:
                        f.write("%d,%f,%f,%f\n" % (sim_index, score, p_value, sim_time))
                    
                    # Print progress for this simulation
                    batch_completed += 1
                    current_time = time.time()
                    elapsed_time = current_time - overall_start_time
                    
                    # Update progress every 5 simulations or every 10 seconds
                    if batch_completed % 5 == 0 or (current_time - last_update_time) >= 10:
                        last_update_time = current_time
                        
                        # Calculate overall progress
                        total_completed = completed + batch_completed
                        progress_pct = 100.0 * total_completed / num_simulations
                        
                        # Calculate estimated time remaining
                        if total_completed > 0:
                            avg_time_per_sim = elapsed_time / total_completed
                            remaining_sims = num_simulations - total_completed
                            est_remaining_time = avg_time_per_sim * remaining_sims
                        else:
                            est_remaining_time = 0
                        
                        # Create a text-based progress bar
                        bar_length = 30
                        filled_length = int(bar_length * total_completed // num_simulations)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        
                        # Print progress with progress bar
                        sys.stdout.write("\rProgress: [%s] %5.1f%% (%d/%d) - p-value: %.6f - Elapsed: %.1fs - Est. remaining: %.1fs" % 
                              (bar, progress_pct, total_completed, num_simulations, p_value, 
                               elapsed_time, est_remaining_time))
                        sys.stdout.flush()
                    
                    # Early stopping if p-value is already significant or definitely not significant
                    if config.get('early_stopping', True) and (completed + batch_completed) >= config.get('min_simulations', 100):
                        # If p-value is very low, we can stop early
                        if p_value < config.get('significance_threshold', 0.01) / 10:
                            print("\nEarly stopping: p-value is significantly low (%.6f)" % p_value)
                            raise StopIteration()
                        
                        # If p-value is very high, we can also stop early
                        if p_value > 0.5 and (completed + batch_completed) >= num_simulations / 2:
                            print("\nEarly stopping: p-value is high (%.6f) after %d simulations" % 
                                  (p_value, completed + batch_completed))
                            raise StopIteration()
                
                except queue.Empty:
                    # No result available yet, just continue waiting
                    continue
                except StopIteration:
                    # Early stopping triggered
                    break
            
            # Update completed count
            completed += batch_completed
            
            # Break if we've completed all simulations or triggered early stopping
            if completed >= num_simulations or batch_completed < batch_size_actual:
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user after %d simulations" % completed)
    finally:
        # Close the pool
        pool.terminate()
        pool.join()
    
    # Calculate final p-value
    p_value = float(count_greater_equal) / len(all_scores) if all_scores else 1.0
    
    # Calculate z-score
    if len(all_scores) > 1:
        mean_surrogate = np.mean(all_scores)
        std_surrogate = np.std(all_scores)
        z_score = (actual_score - mean_surrogate) / std_surrogate if std_surrogate > 0 else 0
    else:
        mean_surrogate = 0
        std_surrogate = 0
        z_score = 0
    
    # Print final results
    print("\nResults for constant %.8f:" % constant)
    print("Actual score: %.6f" % actual_score)
    print("Mean surrogate score: %.6f" % mean_surrogate)
    print("P-value: %.6f" % p_value)
    print("Z-score: %.4f" % z_score)
    print("Significant: %s" % ("Yes" if p_value < config.get('significance_level', 0.05) else "No"))
    print("Total time: %.1fs" % (time.time() - overall_start_time))
    
    # Save final results to progress file
    with open(progress_file, 'a') as f:
        f.write("\n# Final Results:\n")
        f.write("# Actual score: %.6f\n" % actual_score)
        f.write("# Mean surrogate score: %.6f\n" % mean_surrogate)
        f.write("# P-value: %.6f\n" % p_value)
        f.write("# Z-score: %.4f\n" % z_score)
        f.write("# Significant: %s\n" % ("Yes" if p_value < config.get('significance_level', 0.05) else "No"))
        f.write("# Total time: %.1fs\n" % (time.time() - overall_start_time))
    
    # Return results
    return {
        'constant': constant,
        'actual_score': actual_score,
        'surrogate_scores': all_scores,
        'num_simulations': len(all_scores),
        'p_value': p_value,
        'z_score': z_score,
        'significant': p_value < config.get('significance_level', 0.05),
        'mean_surrogate': mean_surrogate,
        'std_surrogate': std_surrogate
    }

if __name__ == "__main__":
    # Configuration
    config = {
        'num_simulations': 1000,
        'constants': {
            'phi': 1.61803398875,
            'sqrt2': 1.41421356237,
            'sqrt3': 1.73205080757,
            'ln2': 0.693147180559945,
            'e': 2.71828182846,
            'pi': 3.14159265359
        },
        'parallel_processing': True,
        'early_stopping': True,
        'min_simulations': 100,
        'significance_threshold': 0.01,
        'significance_level': 0.05,
        'max_workers': 8,  # Adjust based on available CPU cores
        'output_dir': "../results/information_architecture_1k"
    }
    
    # Create output directory
    ensure_dir_exists(config['output_dir'])
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create test instance
    ia_test = InformationArchitectureTest(config)
    
    # Run test on WMAP data
    print("\n" + "="*50)
    print("Running Information Architecture Test on WMAP data (%d simulations)" % config['num_simulations'])
    print("="*50)
    
    # Load WMAP data
    wmap_data = load_wmap_power_spectrum()
    print("WMAP data loaded: %d data points" % len(wmap_data))
    
    # Run test for WMAP data
    wmap_output_dir = os.path.join(config['output_dir'], 'wmap')
    ensure_dir_exists(wmap_output_dir)
    
    wmap_results = {}
    for name, value in config['constants'].items():
        print("\n" + "="*50)
        print("Testing constant: %s = %s" % (name, value))
        print("="*50)
        
        # Create output directory for this constant
        constant_dir = os.path.join(wmap_output_dir, name)
        ensure_dir_exists(constant_dir)
        
        # Run Monte Carlo simulation
        constant_results = run_monte_carlo_simulation(
            wmap_data, value, config['num_simulations'], constant_dir, config
        )
        
        # Store results
        wmap_results[name] = constant_results
    
    # Save WMAP results summary
    ia_test.generate_summary_report(wmap_results, wmap_output_dir)
    ia_test.visualize_results(wmap_results, wmap_output_dir)
    
    # Run test on Planck data
    print("\n" + "="*50)
    print("Running Information Architecture Test on Planck data (%d simulations)" % config['num_simulations'])
    print("="*50)
    
    # Load Planck data
    planck_data = load_planck_power_spectrum()
    print("Planck data loaded: %d data points" % len(planck_data))
    
    # Run test for Planck data
    planck_output_dir = os.path.join(config['output_dir'], 'planck')
    ensure_dir_exists(planck_output_dir)
    
    planck_results = {}
    for name, value in config['constants'].items():
        print("\n" + "="*50)
        print("Testing constant: %s = %s" % (name, value))
        print("="*50)
        
        # Create output directory for this constant
        constant_dir = os.path.join(planck_output_dir, name)
        ensure_dir_exists(constant_dir)
        
        # Run Monte Carlo simulation
        constant_results = run_monte_carlo_simulation(
            planck_data, value, config['num_simulations'], constant_dir, config
        )
        
        # Store results
        planck_results[name] = constant_results
    
    # Save Planck results summary
    ia_test.generate_summary_report(planck_results, planck_output_dir)
    ia_test.visualize_results(planck_results, planck_output_dir)
    
    # Print overall summary
    print("\n" + "="*50)
    print("INFORMATION ARCHITECTURE TEST RESULTS SUMMARY")
    print("="*50)
    
    print("\nWMAP Results:")
    for constant_name, result in wmap_results.items():
        significance = "Significant" if result['significant'] else "Not Significant"
        print("- %s: Score %.6f, p-value %.6f (%s)" % 
              (constant_name, result['actual_score'], result['p_value'], significance))
    
    print("\nPlanck Results:")
    for constant_name, result in planck_results.items():
        significance = "Significant" if result['significant'] else "Not Significant"
        print("- %s: Score %.6f, p-value %.6f (%s)" % 
              (constant_name, result['actual_score'], result['p_value'], significance))
    
    print("\nTest completed successfully. Results saved to: %s" % config['output_dir'])
    
    # Compare with previous results
    print("\nComparison with Previous Results:")
    print("\nPrevious WMAP Results:")
    print("- Golden Ratio (φ): Score 0.574394, p-value 0.000000 (Significant)")
    print("- Square Root of 2: Score 0.640046, p-value 0.000000 (Significant)")
    print("- Square Root of 3: Score 0.541628, p-value 0.000000 (Significant)")
    print("- Natural Log of 2: Score 0.688917, p-value 0.000000 (Significant)")
    print("- Euler's Number (e): Score 0.360607, p-value 1.000000 (Not Significant)")
    print("- Pi (π): Score 0.314280, p-value 1.000000 (Not Significant)")
    
    print("\nPrevious Planck Results:")
    print("- Golden Ratio (φ): Score 0.610188, p-value 0.000000 (Significant)")
    print("- Square Root of 2: Score 0.690893, p-value 0.000000 (Significant)")
    print("- Square Root of 3: Score 0.571989, p-value 0.000000 (Significant)")
    print("- Natural Log of 2: Score 0.743526, p-value 0.000000 (Significant)")
    print("- Euler's Number (e): Score 0.369580, p-value 0.000000 (Significant)")
    print("- Pi (π): Score 0.321007, p-value 0.068000 (Not Significant)")
