#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fractal Analysis Test for WMAP and Planck CMB data.

This script implements the Fractal Analysis Test, which uses the Hurst exponent
to evaluate fractal behavior in the CMB power spectrum. Of particular interest
is the relationship with the Golden Ratio, where a Hurst exponent of H = φ-1 
(approximately 0.618) represents optimal fractal behavior.

Author: Stephanie Alexander
Version: 2.0 (Enhanced with improved statistical analysis)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import multiprocessing
from functools import partial
import logging
import pickle
import json
from pathlib import Path

# Import utility functions
from .fractal_utils import (
    preprocess_data, 
    calculate_hurst_exponent, 
    generate_surrogate,
    calculate_phi_optimality,
    plot_fractal_results
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FractalAnalysis")

# Mathematical constants
PHI = 1.618033988749895  # Golden ratio


def load_wmap_power_spectrum(file_path):
    """
    Load WMAP CMB power spectrum data.
    
    Parameters:
    -----------
    file_path : str
        Path to the WMAP data file
    
    Returns:
    --------
    tuple
        (ell, power, error) arrays
    """
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Mean multipole moment
        power = data[:, 3]  # Power spectrum value
        error = data[:, 4]  # Error
        logger.info(f"Loaded WMAP data from {file_path}")
        return ell, power, error
    except Exception as e:
        logger.error(f"Error loading WMAP power spectrum: {str(e)}")
        return None, None, None


def load_planck_power_spectrum(file_path):
    """
    Load Planck CMB power spectrum data.
    
    Parameters:
    -----------
    file_path : str
        Path to the Planck data file
    
    Returns:
    --------
    tuple
        (ell, power, error) arrays
    """
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Multipole moment
        power = data[:, 1]  # Power spectrum value
        # Use average of asymmetric error bars as the error
        lower_error = data[:, 2]  # Lower error bound
        upper_error = data[:, 3]  # Upper error bound
        error = (abs(lower_error) + abs(upper_error)) / 2.0
        logger.info(f"Loaded Planck data from {file_path}")
        return ell, power, error
    except Exception as e:
        logger.error(f"Error loading Planck power spectrum: {str(e)}")
        return None, None, None


def _run_simulation(_, data, max_lag):
    """
    Run a single simulation for Monte Carlo analysis.
    
    Parameters:
    -----------
    _ : any
        Placeholder parameter for compatibility with map
    data : numpy.ndarray
        Original data array
    max_lag : int
        Maximum lag for Hurst exponent calculation
    
    Returns:
    --------
    float
        Hurst exponent of surrogate data
    """
    # Generate surrogate data
    surrogate = generate_surrogate(data)
    
    # Calculate Hurst exponent
    hurst, _, _ = calculate_hurst_exponent(surrogate, max_lag=max_lag)
    
    return hurst


def run_monte_carlo_parallel(data, n_simulations=10000, max_lag=20, num_processes=None):
    """
    Run Monte Carlo simulations in parallel to assess the significance of the Hurst exponent.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array
    n_simulations : int, optional
        Number of simulations, by default 10000
    max_lag : int, optional
        Maximum lag for R/S analysis, by default 20
    num_processes : int, optional
        Number of processes to use for parallelization, by default None (use all available)
    
    Returns:
    --------
    tuple
        (p_value, phi_optimality, actual_hurst, sim_hursts, log_lags, log_rs)
    """
    # Calculate actual Hurst exponent
    actual_hurst, log_lags, log_rs = calculate_hurst_exponent(data, max_lag=max_lag)
    logger.info(f"Actual Hurst exponent: {actual_hurst:.6f}")
    
    # Set up parallel processing
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Create simulation wrapper function
    sim_function = partial(_run_simulation, data=data, max_lag=max_lag)
    
    # Run simulations in chunks to show progress
    sim_hursts = []
    chunk_size = 100
    num_chunks = int(np.ceil(n_simulations / chunk_size))
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for i in range(num_chunks):
            current_chunk = min(chunk_size, n_simulations - i * chunk_size)
            logger.info(f"Running chunk {i+1}/{num_chunks} ({current_chunk} simulations)")
            
            # Run current chunk of simulations
            chunk_results = pool.map(sim_function, range(current_chunk))
            sim_hursts.extend(chunk_results)
    
    # Convert to numpy array
    sim_hursts = np.array(sim_hursts)
    
    # Calculate p-value (two-tailed test)
    if np.abs(actual_hurst - PHI + 1) < np.abs(np.mean(sim_hursts) - PHI + 1):
        # Actual value is closer to phi-1 than the mean of simulations
        p_value = np.mean(np.abs(sim_hursts - PHI + 1) <= np.abs(actual_hurst - PHI + 1))
    else:
        # Mean of simulations is closer to phi-1
        p_value = np.mean(np.abs(sim_hursts - PHI + 1) >= np.abs(actual_hurst - PHI + 1))
    
    # Calculate Golden Ratio optimality
    phi_optimality = calculate_phi_optimality(actual_hurst, sim_hursts)
    
    return p_value, phi_optimality, actual_hurst, sim_hursts, log_lags, log_rs


def run_fractal_analysis(data, output_dir, name, n_simulations=10000, max_lag=20, parallel=True, num_processes=None, random_seed=None):
    """
    Run fractal analysis on the provided data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array
    output_dir : str
        Directory to save results
    name : str
        Name for results files (e.g., 'wmap' or 'planck')
    n_simulations : int, optional
        Number of simulations, by default 10000
    max_lag : int, optional
        Maximum lag for R/S analysis, by default 20
    parallel : bool, optional
        Whether to use parallel processing, by default True
    num_processes : int, optional
        Number of processes for parallel computing, by default None (use all available)
    random_seed : int, optional
        Seed for random number generator, by default None
    
    Returns:
    --------
    dict
        Dictionary of results
    """
    logger.info(f"Starting fractal analysis for {name.upper()}")
    logger.info(f"Parameters: n_simulations={n_simulations}, max_lag={max_lag}")
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        logger.info(f"Using random seed: {random_seed}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Monte Carlo simulations
    if parallel:
        logger.info(f"Running parallel Monte Carlo with {num_processes or multiprocessing.cpu_count()} processes")
        p_value, phi_optimality, actual_hurst, sim_hursts, log_lags, log_rs = run_monte_carlo_parallel(
            data, n_simulations, max_lag, num_processes
        )
    else:
        logger.info("Running sequential Monte Carlo")
        # Implementation of sequential Monte Carlo would go here
        raise NotImplementedError("Sequential Monte Carlo not implemented, use parallel=True")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output paths
    plot_path = os.path.join(output_dir, f"fractal_analysis_{name}_{timestamp}.png")
    results_path = os.path.join(output_dir, f"fractal_analysis_{name}_{timestamp}.pkl")
    summary_path = os.path.join(output_dir, f"fractal_analysis_{name}_{timestamp}_summary.txt")
    
    # Plot results
    title = f"Fractal Analysis: {name.upper()} CMB Power Spectrum"
    plot_fractal_results(
        log_lags, log_rs, p_value, phi_optimality, 
        sim_hursts, actual_hurst, title, plot_path
    )
    
    # Save results
    results = {
        'name': name,
        'timestamp': timestamp,
        'actual_hurst': actual_hurst,
        'p_value': p_value,
        'phi_optimality': phi_optimality,
        'log_lags': log_lags.tolist(),
        'log_rs': log_rs.tolist(),
        'sim_hursts': sim_hursts.tolist(),
        'n_simulations': n_simulations,
        'max_lag': max_lag,
        'significant': p_value < 0.05
    }
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Write summary
    with open(summary_path, 'w') as f:
        f.write(f"Fractal Analysis Summary: {name.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of simulations: {n_simulations}\n")
        f.write(f"Maximum lag: {max_lag}\n\n")
        f.write(f"Actual Hurst exponent: {actual_hurst:.6f}\n")
        f.write(f"P-value: {p_value:.6f}\n")
        f.write(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}\n")
        f.write(f"Golden Ratio optimality: {phi_optimality:.6f}\n\n")
        
        # Add interpretation
        f.write("Interpretation:\n")
        f.write("-" * 50 + "\n")
        
        if actual_hurst > 0.5:
            f.write("The Hurst exponent indicates persistent behavior (H > 0.5), ")
            f.write("suggesting long-range correlations and fractal structure in the power spectrum.\n\n")
        elif actual_hurst < 0.5:
            f.write("The Hurst exponent indicates anti-persistent behavior (H < 0.5), ")
            f.write("suggesting a tendency to revert to the mean and less predictable patterns.\n\n")
        else:
            f.write("The Hurst exponent indicates random walk behavior (H = 0.5), ")
            f.write("suggesting no long-range correlations in the power spectrum.\n\n")
        
        # Comparison to Golden Ratio
        phi_minus_one = PHI - 1
        f.write(f"Proximity to Golden Ratio (φ-1 = {phi_minus_one:.6f}):\n")
        
        if abs(actual_hurst - phi_minus_one) < 0.05:
            f.write("The Hurst exponent is very close to φ-1 (within 0.05), ")
            f.write("suggesting optimal fractal behavior associated with the Golden Ratio.\n\n")
        else:
            f.write(f"The Hurst exponent differs from φ-1 by {abs(actual_hurst - phi_minus_one):.6f}, ")
            if actual_hurst > phi_minus_one:
                f.write("indicating stronger persistence than the Golden Ratio optimum.\n\n")
            else:
                f.write("indicating less persistence than the Golden Ratio optimum.\n\n")
        
        # Statistical significance
        if p_value < 0.05:
            f.write(f"The result is statistically significant (p = {p_value:.6f} < 0.05), ")
            f.write("indicating that the observed fractal behavior is unlikely to be due to chance.\n\n")
        else:
            f.write(f"The result is not statistically significant (p = {p_value:.6f} >= 0.05), ")
            f.write("suggesting that the observed fractal behavior could be due to chance.\n\n")
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Summary saved to {summary_path}")
    
    return results


def compare_results(wmap_results, planck_results, output_dir):
    """
    Compare fractal analysis results between WMAP and Planck data.
    
    Parameters:
    -----------
    wmap_results : dict
        WMAP analysis results
    planck_results : dict
        Planck analysis results
    output_dir : str
        Directory to save comparison results
    
    Returns:
    --------
    dict
        Dictionary of comparison results
    """
    logger.info("Comparing WMAP and Planck fractal analysis results")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output paths
    plot_path = os.path.join(output_dir, f"fractal_comparison_{timestamp}.png")
    summary_path = os.path.join(output_dir, f"fractal_comparison_{timestamp}_summary.txt")
    
    # Extract results
    wmap_h = wmap_results['actual_hurst']
    planck_h = planck_results['actual_hurst']
    wmap_p = wmap_results['p_value']
    planck_p = planck_results['p_value']
    wmap_phi = wmap_results['phi_optimality']
    planck_phi = planck_results['phi_optimality']
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Hurst exponents
    datasets = ['WMAP', 'Planck']
    hurst_values = [wmap_h, planck_h]
    phi_values = [wmap_phi, planck_phi]
    
    ax1.bar(datasets, hurst_values, color=['blue', 'red'], alpha=0.7)
    ax1.axhline(y=0.5, color='green', linestyle='--', label='H = 0.5 (Random)')
    ax1.axhline(y=PHI-1, color='orange', linestyle='--', label=f'H = {PHI-1:.4f} (φ-1)')
    
    for i, v in enumerate(hurst_values):
        ax1.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    ax1.set_ylabel('Hurst Exponent')
    ax1.set_title('Hurst Exponent Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot phi-optimality
    ax2.bar(datasets, phi_values, color=['blue', 'red'], alpha=0.7)
    ax2.axhline(y=1.0, color='gray', linestyle='--', label='Random baseline')
    
    for i, v in enumerate(phi_values):
        ax2.text(i, v + 0.05, f'{v:.4f}', ha='center')
    
    ax2.set_ylabel('φ-Optimality Score')
    ax2.set_title('Golden Ratio Optimality Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set main title
    plt.suptitle('WMAP vs. Planck Fractal Analysis Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate comparison metrics
    hurst_diff = abs(wmap_h - planck_h)
    hurst_ratio = max(wmap_h, planck_h) / min(wmap_h, planck_h) if min(wmap_h, planck_h) > 0 else float('inf')
    phi_diff = abs(wmap_phi - planck_phi)
    phi_ratio = max(wmap_phi, planck_phi) / min(wmap_phi, planck_phi) if min(wmap_phi, planck_phi) > 0 else float('inf')
    
    # Create comparison results
    comparison = {
        'timestamp': timestamp,
        'wmap_hurst': wmap_h,
        'planck_hurst': planck_h,
        'wmap_p_value': wmap_p,
        'planck_p_value': planck_p,
        'wmap_phi_optimality': wmap_phi,
        'planck_phi_optimality': planck_phi,
        'hurst_difference': hurst_diff,
        'hurst_ratio': hurst_ratio,
        'phi_difference': phi_diff,
        'phi_ratio': phi_ratio
    }
    
    # Write summary
    with open(summary_path, 'w') as f:
        f.write("Fractal Analysis Comparison: WMAP vs. Planck\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Hurst Exponent Comparison:\n")
        f.write("-" * 50 + "\n")
        f.write(f"WMAP Hurst exponent: {wmap_h:.6f}\n")
        f.write(f"Planck Hurst exponent: {planck_h:.6f}\n")
        f.write(f"Absolute difference: {hurst_diff:.6f}\n")
        f.write(f"Ratio (max/min): {hurst_ratio:.6f}\n\n")
        
        f.write("Statistical Significance:\n")
        f.write("-" * 50 + "\n")
        f.write(f"WMAP p-value: {wmap_p:.6f} ({'Significant' if wmap_p < 0.05 else 'Not significant'})\n")
        f.write(f"Planck p-value: {planck_p:.6f} ({'Significant' if planck_p < 0.05 else 'Not significant'})\n\n")
        
        f.write("Golden Ratio Optimality:\n")
        f.write("-" * 50 + "\n")
        f.write(f"WMAP φ-optimality: {wmap_phi:.6f}\n")
        f.write(f"Planck φ-optimality: {planck_phi:.6f}\n")
        f.write(f"Absolute difference: {phi_diff:.6f}\n")
        f.write(f"Ratio (max/min): {phi_ratio:.6f}\n\n")
        
        # Add interpretation
        f.write("Interpretation:\n")
        f.write("-" * 50 + "\n")
        
        # Hurst exponent comparison
        if abs(wmap_h - planck_h) < 0.05:
            f.write("The Hurst exponents for WMAP and Planck are very similar (difference < 0.05), ")
            f.write("suggesting consistent fractal behavior across both datasets.\n\n")
        else:
            f.write("There is a notable difference between the Hurst exponents for WMAP and Planck, ")
            if wmap_h > planck_h:
                f.write("with WMAP showing stronger fractal persistence than Planck.\n\n")
            else:
                f.write("with Planck showing stronger fractal persistence than WMAP.\n\n")
        
        # Statistical significance comparison
        if wmap_p < 0.05 and planck_p < 0.05:
            f.write("Both datasets show statistically significant fractal behavior, ")
            f.write("strongly supporting the presence of scale-invariant patterns in the CMB.\n\n")
        elif wmap_p < 0.05:
            f.write("Only the WMAP data shows statistically significant fractal behavior, ")
            f.write("suggesting the Planck observations may contain more instrumental effects or noise.\n\n")
        elif planck_p < 0.05:
            f.write("Only the Planck data shows statistically significant fractal behavior, ")
            f.write("suggesting the higher resolution of Planck may better capture the fractal structure.\n\n")
        else:
            f.write("Neither dataset shows statistically significant fractal behavior, ")
            f.write("suggesting the observed patterns could be due to chance or noise.\n\n")
        
        # Golden Ratio optimality comparison
        if wmap_phi > 1.1 and planck_phi > 1.1:
            f.write("Both datasets show above-random alignment with Golden Ratio optimality, ")
            if abs(wmap_phi - planck_phi) < 0.1:
                f.write("with very similar levels of alignment across datasets.\n\n")
            elif wmap_phi > planck_phi:
                f.write("with WMAP showing stronger alignment than Planck.\n\n")
            else:
                f.write("with Planck showing stronger alignment than WMAP.\n\n")
        elif wmap_phi > 1.1:
            f.write("Only the WMAP data shows above-random alignment with Golden Ratio optimality.\n\n")
        elif planck_phi > 1.1:
            f.write("Only the Planck data shows above-random alignment with Golden Ratio optimality.\n\n")
        else:
            f.write("Neither dataset shows strong alignment with Golden Ratio optimality.\n\n")
    
    logger.info(f"Comparison summary saved to {summary_path}")
    
    return comparison


def main():
    """Main function to run the fractal analysis test."""
    parser = argparse.ArgumentParser(description='Run Fractal Analysis Test on CMB data')
    parser.add_argument('--wmap_file', type=str, default='data/wmap/wmap_tt_spectrum_9yr_v5.txt',
                      help='Path to WMAP power spectrum data file')
    parser.add_argument('--planck_file', type=str, default='data/planck/planck_tt_spectrum_2018.txt',
                      help='Path to Planck power spectrum data file')
    parser.add_argument('--output_dir', type=str, default='results/fractal_analysis',
                      help='Directory to save results')
    parser.add_argument('--n_simulations', type=int, default=10000,
                      help='Number of Monte Carlo simulations')
    parser.add_argument('--max_lag', type=int, default=20,
                      help='Maximum lag for Hurst exponent calculation')
    parser.add_argument('--processes', type=int, default=None,
                      help='Number of processes for parallel computation')
    parser.add_argument('--random_seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--skip_wmap', action='store_true',
                      help='Skip WMAP analysis')
    parser.add_argument('--skip_planck', action='store_true',
                      help='Skip Planck analysis')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results
    wmap_results = None
    planck_results = None
    
    # Process WMAP data if not skipped
    if not args.skip_wmap:
        logger.info("Loading WMAP data...")
        wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(args.wmap_file)
        
        if wmap_power is not None:
            # Preprocess data
            wmap_processed = preprocess_data(wmap_power, normalize=True)
            
            # Run fractal analysis
            wmap_results = run_fractal_analysis(
                wmap_processed, 
                args.output_dir, 
                'wmap', 
                n_simulations=args.n_simulations,
                max_lag=args.max_lag,
                parallel=True,
                num_processes=args.processes,
                random_seed=args.random_seed
            )
    
    # Process Planck data if not skipped
    if not args.skip_planck:
        logger.info("Loading Planck data...")
        planck_ell, planck_power, planck_error = load_planck_power_spectrum(args.planck_file)
        
        if planck_power is not None:
            # Preprocess data
            planck_processed = preprocess_data(planck_power, normalize=True)
            
            # Run fractal analysis
            planck_results = run_fractal_analysis(
                planck_processed, 
                args.output_dir, 
                'planck', 
                n_simulations=args.n_simulations,
                max_lag=args.max_lag,
                parallel=True,
                num_processes=args.processes,
                random_seed=args.random_seed
            )
    
    # Compare results if both datasets were analyzed
    if wmap_results is not None and planck_results is not None:
        comparison = compare_results(wmap_results, planck_results, args.output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
