#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug Version of Information Architecture Test Runner
This script runs the Information Architecture test with extensive debugging
to identify issues with progress reporting.
"""

import os
import sys
import time
import traceback
import numpy as np
from datetime import datetime
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import the Information Architecture Test
from scripts.info_architecture.archive.information_architecture_test import InformationArchitectureTest, load_wmap_power_spectrum, load_planck_power_spectrum

# Set up logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('IA_Test')

def debug_log(message):
    """Log a debug message and print to console"""
    logger.debug(message)
    print(message)

def ensure_dir_exists(dir_path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        debug_log("Created directory: %s" % dir_path)

# Standalone functions for parallel processing
def _run_single_simulation(data, constant, seed, config, result_queue, sim_index):
    """Run a single simulation for parallel processing with detailed logging."""
    try:
        debug_log("Starting simulation %d with seed %d" % (sim_index, seed))
        start_time = time.time()
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create a temporary test instance
        test = InformationArchitectureTest(config)
        
        # Generate surrogate data (shuffle the original data)
        debug_log("Simulation %d: Generating surrogate data" % sim_index)
        surrogate_data = np.random.permutation(data)
        
        # Calculate architecture score for surrogate data
        debug_log("Simulation %d: Calculating architecture score" % sim_index)
        score = test.calculate_architecture_score(surrogate_data, constant)
        
        # Put result in queue
        elapsed = time.time() - start_time
        debug_log("Simulation %d completed in %.2fs with score %s" % (sim_index, elapsed, score))
        result_queue.put((sim_index, score))
        
        return score
    except Exception as e:
        error_msg = "ERROR in simulation %d: %s\n%s" % (sim_index, str(e), traceback.format_exc())
        debug_log(error_msg)
        result_queue.put((sim_index, None, error_msg))
        return None

def run_monte_carlo_simulation(data, constant, num_simulations, output_dir, config):
    """
    Run Monte Carlo simulation with extensive debugging.
    """
    debug_log("Starting Monte Carlo simulation for constant %s" % constant)
    debug_log("Number of simulations: %d" % num_simulations)
    
    # Create test instance for actual score calculation
    test = InformationArchitectureTest(config)
    
    # Get actual score
    debug_log("Calculating actual architecture score")
    start_time = time.time()
    actual_score = test.calculate_architecture_score(data, constant)
    elapsed = time.time() - start_time
    debug_log("Actual score: %s (calculated in %.2fs)" % (actual_score, elapsed))
    
    # Initialize progress tracking
    constant_dir = os.path.join(output_dir, str(constant).replace('.', '_'))
    ensure_dir_exists(constant_dir)
    progress_file = os.path.join(constant_dir, "progress.txt")
    debug_file = os.path.join(constant_dir, "debug.log")
    
    # Write header to progress file
    debug_log("Writing header to progress file: %s" % progress_file)
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
    
    # Use parallel processing with a shared queue for results
    debug_log("Setting up parallel processing")
    manager = Manager()
    result_queue = manager.Queue()
    
    # Create a pool of workers
    num_workers = min(cpu_count(), config.get('max_workers', cpu_count()))
    debug_log("Using %d worker processes" % num_workers)
    
    # Update status file with initialization info
    with open(status_file, 'a') as f:
        f.write("Initialized %d worker processes\n" % num_workers)
        f.write("Starting time: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Process simulations in smaller batches for better progress reporting
    batch_size = 10  # Process 10 simulations at a time for debugging
    completed = 0
    
    try:
        while completed < num_simulations:
            # Determine how many simulations to run in this batch
            current_batch_size = min(batch_size, num_simulations - completed)
            debug_log("Processing batch of %d simulations (total completed: %d/%d)" % 
                     (current_batch_size, completed, num_simulations))
            
            # Update status file
            with open(status_file, 'a') as f:
                f.write("Processing batch of %d simulations (total completed: %d/%d)\n" % 
                       (current_batch_size, completed, num_simulations))
            
            # Create processes for this batch
            processes = []
            for i in range(current_batch_size):
                sim_index = completed + i
                seed = sim_index  # Use simulation index as seed
                p = multiprocessing.Process(
                    target=_run_single_simulation,
                    args=(data, constant, seed, config, result_queue, sim_index)
                )
                processes.append(p)
                p.start()
                debug_log("Started process for simulation %d" % sim_index)
            
            # Wait for all processes to complete
            debug_log("Waiting for %d processes to complete" % len(processes))
            for p in processes:
                p.join()
            
            # Process results from the queue
            debug_log("Processing results from queue")
            results_processed = 0
            while not result_queue.empty() and results_processed < current_batch_size:
                try:
                    result = result_queue.get(block=False)
                    if len(result) == 2:
                        sim_index, score = result
                        debug_log("Got result for simulation %d: %s" % (sim_index, score))
                        
                        all_scores.append(score)
                        if score >= actual_score:
                            count_greater_equal += 1
                        
                        # Calculate p-value
                        p_value = float(count_greater_equal) / len(all_scores)
                        
                        # Update progress file for each simulation
                        with open(progress_file, 'a') as f:
                            f.write("%d,%s,%s\n" % (sim_index, score, p_value))
                        
                        results_processed += 1
                    else:
                        sim_index, score, error_msg = result
                        debug_log("Error in simulation %d: %s" % (sim_index, error_msg))
                        results_processed += 1
                except Exception as e:
                    debug_log("Error processing result: %s" % str(e))
            
            # Update completed count
            completed += results_processed
            debug_log("Batch complete. Total completed: %d/%d" % (completed, num_simulations))
            
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
            
            progress_str = "Progress: %.1f%% (%d/%d) - p-value: %.6f - Elapsed: %.1fs - Est. remaining: %.1fs" % (
                progress_pct, completed, num_simulations, p_value, elapsed_time, est_remaining_time)
            debug_log(progress_str)
            
            # Update status file with current progress
            with open(status_file, 'a') as f:
                f.write("%s\n" % progress_str)
            
            # Flush all logs to disk
            sys.stdout.flush()
            logging.shutdown()
            logging.root.handlers = [
                logging.FileHandler('debug.log'),
                logging.StreamHandler(sys.stdout)
            ]
    
    except Exception as e:
        error_msg = "ERROR in main simulation loop: %s\n%s" % (str(e), traceback.format_exc())
        debug_log(error_msg)
        # Log any errors
        with open(status_file, 'a') as f:
            f.write("ERROR: %s\n" % str(e))
        raise
    
    # Print final progress
    debug_log("Completed %d/%d simulations in %.1f seconds" % (completed, num_simulations, elapsed_time))
    
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
    
    # Return results
    results = {
        'constant': constant,
        'actual_score': actual_score,
        'surrogate_scores': all_scores,
        'p_value': p_value,
        'z_score': z_score,
        'significant': significant,
        'num_simulations': len(all_scores)
    }
    
    debug_log("Results: %s" % results)
    return results

def run_test_for_dataset(data, constants, output_dir, config):
    """
    Run the Information Architecture Test for a single dataset.
    """
    debug_log("Running test for dataset with output directory: %s" % output_dir)
    
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
        debug_log("Testing constant: %s = %s" % (name, value))
        
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
    
    # Update dataset status file
    with open(status_file, 'a') as f:
        f.write("\nAll tests completed at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    return results

if __name__ == "__main__":
    # Configuration
    config = {
        'num_simulations': 100,  # Start with a small number for debugging
        'constants': {
            'phi': 1.61803398875,  # Just test one constant for debugging
        },
        'parallel_processing': True,
        'max_workers': 8,
        'output_dir': "../results/information_architecture_debug"
    }
    
    debug_log("Starting Information Architecture Test (Debug Version)")
    debug_log("Configuration: %s" % config)
    
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
        debug_log("Running Information Architecture Test on WMAP data (%d simulations)" % config['num_simulations'])
        
        # Update master status file
        with open(master_status_file, 'a') as f:
            f.write("\nStarting WMAP data analysis at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Load WMAP data
        debug_log("Loading WMAP data")
        wmap_data = load_wmap_power_spectrum()
        debug_log("WMAP data loaded: %d data points" % len(wmap_data))
        
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
        
        # Print overall summary
        debug_log("Information Architecture Test Complete")
        debug_log("Results saved to: %s" % config['output_dir'])
        
        # Update master status file
        with open(master_status_file, 'a') as f:
            f.write("\nAll tests completed successfully at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    except Exception as e:
        error_msg = "ERROR in main program: %s\n%s" % (str(e), traceback.format_exc())
        debug_log(error_msg)
        # Log any errors to the master status file
        with open(master_status_file, 'a') as f:
            f.write("\nERROR: %s\n" % str(e))
            f.write("Test failed at %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        raise
