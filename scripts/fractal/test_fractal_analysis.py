#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fractal Analysis Test for WMAP and Planck CMB data.

This script implements the Fractal Analysis Test, which uses the Hurst exponent
to evaluate fractal behavior in the CMB power spectrum.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import argparse
import multiprocessing
from functools import partial

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_wmap_power_spectrum(file_path):
    """Load WMAP CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Mean multipole moment
        power = data[:, 3]  # Power spectrum value
        error = data[:, 4]  # Error
        return ell, power, error
    except Exception as e:
        print("Error loading WMAP power spectrum: {}".format(str(e)))
        return None, None, None


def load_planck_power_spectrum(file_path):
    """Load Planck CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Multipole moment
        power = data[:, 1]  # Power spectrum value
        # Use average of asymmetric error bars as the error
        lower_error = data[:, 2]  # Lower error bound
        upper_error = data[:, 3]  # Upper error bound
        error = (abs(lower_error) + abs(upper_error)) / 2.0
        return ell, power, error
    except Exception as e:
        print("Error loading Planck power spectrum: {}".format(str(e)))
        return None, None, None


def preprocess_data(data, smooth=False, smooth_window=5, normalize=True, detrend=False):
    """Preprocess data for analysis."""
    processed_data = data.copy()
    
    # Apply smoothing if requested
    if smooth:
        window = np.ones(smooth_window) / smooth_window
        processed_data = np.convolve(processed_data, window, mode='same')
    
    # Remove linear trend if requested
    if detrend:
        processed_data = signal.detrend(processed_data)
    
    # Normalize if requested
    if normalize:
        processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
    
    return processed_data


def calculate_hurst_exponent(data, max_lag=20):
    """
    Calculate the Hurst exponent using the rescaled range (R/S) analysis.
    
    The Hurst exponent (H) measures the long-term memory of a time series:
    - H = 0.5: Random walk (Brownian motion)
    - 0 < H < 0.5: Anti-persistent series
    - 0.5 < H < 1: Persistent series (fractal behavior)
    
    Args:
        data (numpy.ndarray): Input data array
        max_lag (int): Maximum lag for R/S analysis
        
    Returns:
        tuple: (hurst_exponent, log_lags, log_rs)
    """
    # Ensure data is at least 2*max_lag long
    if len(data) < 2*max_lag:
        max_lag = len(data) // 2
    
    # Calculate R/S values for different lags
    lags = range(2, max_lag + 1)
    rs_values = []
    
    for lag in lags:
        # Split data into chunks of size lag
        n_chunks = len(data) // lag
        if n_chunks == 0:
            continue
            
        rs_chunk_values = []
        for i in range(n_chunks):
            chunk = data[i*lag:(i+1)*lag]
            
            # Calculate mean and standard deviation
            mean = np.mean(chunk)
            std = np.std(chunk)
            
            if std == 0:
                continue
                
            # Calculate cumulative deviation from mean
            cumsum = np.cumsum(chunk - mean)
            
            # Calculate range (max - min of cumulative sum)
            r = np.max(cumsum) - np.min(cumsum)
            
            # Calculate rescaled range (R/S)
            rs = r / std if std > 0 else 0
            rs_chunk_values.append(rs)
        
        if rs_chunk_values:
            rs_values.append(np.mean(rs_chunk_values))
    
    # Filter out any potential NaN or zero values
    valid_indices = [i for i, rs in enumerate(rs_values) if rs > 0]
    valid_lags = [lags[i] for i in valid_indices]
    valid_rs = [rs_values[i] for i in valid_indices]
    
    if len(valid_lags) < 2:
        return 0.5, [], []  # Return default Hurst exponent
    
    # Log-log plot to find Hurst exponent
    log_lags = np.log10(valid_lags)
    log_rs = np.log10(valid_rs)
    
    # Linear regression to find Hurst exponent
    hurst_exponent, _ = np.polyfit(log_lags, log_rs, 1)
    
    return hurst_exponent, log_lags, log_rs


def run_monte_carlo(data, n_simulations=10000, max_lag=20):
    """
    Run Monte Carlo simulations to assess the significance of the Hurst exponent.
    
    Args:
        data (numpy.ndarray): Input data array
        n_simulations (int): Number of simulations
        max_lag (int): Maximum lag for R/S analysis
        
    Returns:
        tuple: (p_value, phi_optimality, actual_hurst, sim_hursts, log_lags, log_rs)
    """
    # Calculate actual Hurst exponent
    actual_hurst, log_lags, log_rs = calculate_hurst_exponent(data, max_lag)
    
    # Run simulations
    sim_hursts = []
    for i in range(n_simulations):
        if i % 10 == 0:
            print("  Simulation {}/{}".format(i, n_simulations))
        # Create random permutation of the data
        sim_data = np.random.permutation(data)
        sim_hurst, _, _ = calculate_hurst_exponent(sim_data, max_lag)
        sim_hursts.append(sim_hurst)
    
    # Calculate p-value
    # For Hurst exponent, we're interested in values significantly > 0.5 (persistent)
    # or significantly < 0.5 (anti-persistent)
    p_value = np.mean([1 if abs(sim - 0.5) >= abs(actual_hurst - 0.5) else 0 for sim in sim_hursts])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    # For Hurst exponent, values closer to 1 are more "organized" (persistent)
    # and values closer to 0 are more "anti-organized" (anti-persistent)
    sim_mean = np.mean(sim_hursts)
    sim_std = np.std(sim_hursts)
    if sim_std == 0:
        phi_optimality = 0
    else:
        # Adjust for the fact that 0.5 is the neutral point
        deviation_from_random = actual_hurst - 0.5
        phi_optimality = deviation_from_random / 0.5  # Scale to -1 to 1 range
        phi_optimality = np.clip(phi_optimality, -1, 1)  # Ensure within bounds
    
    return p_value, phi_optimality, actual_hurst, sim_hursts, log_lags, log_rs


def run_monte_carlo_parallel(data, n_simulations=10000, max_lag=20, num_processes=None):
    """
    Run Monte Carlo simulations in parallel to assess the significance of the Hurst exponent.
    
    Args:
        data (numpy.ndarray): Input data array
        n_simulations (int): Number of simulations
        max_lag (int): Maximum lag for R/S analysis
        num_processes (int): Number of processes to use for parallelization
        
    Returns:
        tuple: (p_value, phi_optimality, actual_hurst, sim_hursts, log_lags, log_rs)
    """
    # Calculate actual Hurst exponent
    actual_hurst, log_lags, log_rs = calculate_hurst_exponent(data, max_lag)
    
    # Run simulations in parallel
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print("Using {} processes for parallel computation".format(num_processes))
    
    pool = multiprocessing.Pool(processes=num_processes)
    sim_hursts = pool.map(partial(_run_simulation, data=data, max_lag=max_lag), range(n_simulations))
    pool.close()
    pool.join()
    
    # Calculate p-value
    # For Hurst exponent, we're interested in values significantly > 0.5 (persistent)
    # or significantly < 0.5 (anti-persistent)
    p_value = np.mean([1 if abs(sim - 0.5) >= abs(actual_hurst - 0.5) else 0 for sim in sim_hursts])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    # For Hurst exponent, values closer to 1 are more "organized" (persistent)
    # and values closer to 0 are more "anti-organized" (anti-persistent)
    sim_mean = np.mean(sim_hursts)
    sim_std = np.std(sim_hursts)
    if sim_std == 0:
        phi_optimality = 0
    else:
        # Adjust for the fact that 0.5 is the neutral point
        deviation_from_random = actual_hurst - 0.5
        phi_optimality = deviation_from_random / 0.5  # Scale to -1 to 1 range
        phi_optimality = np.clip(phi_optimality, -1, 1)  # Ensure within bounds
    
    return p_value, phi_optimality, actual_hurst, sim_hursts, log_lags, log_rs


def _run_simulation(_, data, max_lag):
    # Create random permutation of the data
    sim_data = np.random.permutation(data)
    sim_hurst, _, _ = calculate_hurst_exponent(sim_data, max_lag)
    return sim_hurst


def run_fractal_analysis(data, output_dir, name, n_simulations=10000, max_lag=20, parallel=False, num_processes=None):
    """Run fractal analysis on the provided data."""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run Monte Carlo simulations
    if parallel:
        print("Running {} Monte Carlo simulations in parallel for {} data...".format(n_simulations, name))
        p_value, phi_optimality, actual_hurst, sim_hursts, log_lags, log_rs = run_monte_carlo_parallel(
            data, n_simulations=n_simulations, max_lag=max_lag, num_processes=num_processes)
    else:
        print("Running {} Monte Carlo simulations sequentially for {} data...".format(n_simulations, name))
        p_value, phi_optimality, actual_hurst, sim_hursts, log_lags, log_rs = run_monte_carlo(
            data, n_simulations=n_simulations, max_lag=max_lag)
    
    # Plot results
    plot_path = os.path.join(output_dir, '{}_fractal_analysis.png'.format(name.lower()))
    plot_fractal_results(
        log_lags, log_rs, p_value, phi_optimality, sim_hursts, actual_hurst,
        "{} Fractal Analysis".format(name), plot_path
    )
    
    # Save results
    results = {
        'p_value': p_value,
        'phi_optimality': phi_optimality,
        'actual_hurst': actual_hurst,
        'sim_hursts': sim_hursts,
        'log_lags': log_lags,
        'log_rs': log_rs
    }
    
    return results


def plot_fractal_results(log_lags, log_rs, p_value, phi_optimality, 
                        sim_hursts, actual_hurst, title, output_path):
    """Plot fractal analysis results."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot R/S analysis
        if len(log_lags) > 0 and len(log_rs) > 0:
            ax1.scatter(log_lags, log_rs, c='blue', alpha=0.7)
            
            # Add regression line
            if len(log_lags) >= 2:
                m, b = np.polyfit(log_lags, log_rs, 1)
                x_range = np.linspace(min(log_lags), max(log_lags), 100)
                ax1.plot(x_range, m * x_range + b, 'r-', linewidth=2)
        
        ax1.set_title('Rescaled Range (R/S) Analysis')
        ax1.set_xlabel('Log10(Lag)')
        ax1.set_ylabel('Log10(R/S)')
        ax1.grid(True)
        
        # Add Hurst exponent
        ax1.text(0.05, 0.95, 'Hurst Exponent = {:.4f}'.format(actual_hurst), 
                transform=ax1.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot simulation results
        if len(sim_hursts) > 0:
            ax2.hist(sim_hursts, bins=min(30, len(sim_hursts)//3), alpha=0.7, color='gray', label='Random Simulations')
            ax2.axvline(actual_hurst, color='r', linestyle='--', linewidth=2, 
                       label='Actual Hurst: {:.4f}'.format(actual_hurst))
            ax2.axvline(0.5, color='g', linestyle=':', linewidth=2, 
                       label='Random Walk: 0.5')
        
        ax2.set_title('Monte Carlo Simulations')
        ax2.set_xlabel('Hurst Exponent')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True)
        
        # Add text with results
        plt.figtext(0.5, 0.01, 'P-value: {:.4f} | Phi-Optimality: {:.4f}'.format(p_value, phi_optimality), 
                   ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print("Warning: Error in plotting fractal results: {}".format(str(e)))
        print("Continuing with analysis...")


def compare_results(wmap_results, planck_results, output_dir):
    """Compare fractal analysis results between WMAP and Planck data."""
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate differences
        hurst_diff = abs(wmap_results['actual_hurst'] - planck_results['actual_hurst'])
        phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
        
        # Save comparison to file
        comparison_path = os.path.join(output_dir, 'fractal_analysis_comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write('Fractal Analysis Test Comparison: WMAP vs Planck\n')
            f.write('=' * 50 + '\n\n')
            
            f.write('WMAP Hurst Exponent: {:.6f}\n'.format(wmap_results['actual_hurst']))
            f.write('WMAP P-value: {:.6f}\n'.format(wmap_results['p_value']))
            f.write('WMAP Phi-Optimality: {:.6f}\n'.format(wmap_results['phi_optimality']))
            f.write('WMAP Significant: {}\n\n'.format(wmap_results['p_value'] < 0.05))
            
            f.write('Planck Hurst Exponent: {:.6f}\n'.format(planck_results['actual_hurst']))
            f.write('Planck P-value: {:.6f}\n'.format(planck_results['p_value']))
            f.write('Planck Phi-Optimality: {:.6f}\n'.format(planck_results['phi_optimality']))
            f.write('Planck Significant: {}\n\n'.format(planck_results['p_value'] < 0.05))
            
            f.write('Difference in Hurst Exponent: {:.6f}\n'.format(hurst_diff))
            f.write('Difference in Phi-Optimality: {:.6f}\n'.format(phi_diff))
        
        # Create comparison plot
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Bar chart of Hurst exponent and phi-optimality
            metrics = ['Hurst Exponent', 'Phi-Optimality']
            wmap_values = [wmap_results['actual_hurst'], wmap_results['phi_optimality']]
            planck_values = [planck_results['actual_hurst'], planck_results['phi_optimality']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, wmap_values, width, label='WMAP', color='blue', alpha=0.7)
            ax1.bar(x + width/2, planck_values, width, label='Planck', color='red', alpha=0.7)
            
            ax1.set_ylabel('Value')
            ax1.set_title('Fractal Analysis: WMAP vs Planck')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            
            # Add reference line for Hurst exponent = 0.5
            ax1.axhline(0.5, color='g', linestyle=':', linewidth=1, alpha=0.7)
            
            # Add text with p-values
            for i, metric in enumerate(metrics):
                ax1.text(i - width/2, wmap_values[i] + 0.02, 
                        'p={:.4f}'.format(wmap_results["p_value"]), 
                        ha='center', va='bottom', color='blue', fontweight='bold')
                ax1.text(i + width/2, planck_values[i] + 0.02, 
                        'p={:.4f}'.format(planck_results["p_value"]), 
                        ha='center', va='bottom', color='red', fontweight='bold')
            
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot 2: R/S analysis for both datasets
            if len(wmap_results['log_lags']) > 0:
                ax2.scatter(wmap_results['log_lags'], wmap_results['log_rs'], 
                           c='blue', alpha=0.7, label='WMAP')
                
                # Add regression line
                if len(wmap_results['log_lags']) >= 2:
                    m, b = np.polyfit(wmap_results['log_lags'], wmap_results['log_rs'], 1)
                    x_range = np.linspace(min(wmap_results['log_lags']), max(wmap_results['log_lags']), 100)
                    ax2.plot(x_range, m * x_range + b, 'b-', linewidth=2)
            
            if len(planck_results['log_lags']) > 0:
                ax2.scatter(planck_results['log_lags'], planck_results['log_rs'], 
                           c='red', alpha=0.7, label='Planck')
                
                # Add regression line
                if len(planck_results['log_lags']) >= 2:
                    m, b = np.polyfit(planck_results['log_lags'], planck_results['log_rs'], 1)
                    x_range = np.linspace(min(planck_results['log_lags']), max(planck_results['log_lags']), 100)
                    ax2.plot(x_range, m * x_range + b, 'r-', linewidth=2)
            
            ax2.set_title('R/S Analysis Comparison')
            ax2.set_xlabel('Log10(Lag)')
            ax2.set_ylabel('Log10(R/S)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            comparison_plot_path = os.path.join(output_dir, 'fractal_analysis_comparison.png')
            plt.savefig(comparison_plot_path)
            plt.close()
        except Exception as e:
            print("Warning: Error in creating comparison plot: {}".format(str(e)))
            print("Continuing with analysis...")
        
        print("\nComparison Results:")
        print("  Difference in Hurst Exponent: {:.6f}".format(hurst_diff))
        print("  Difference in Phi-Optimality: {:.6f}".format(phi_diff))
        print("  Comparison saved to: {}".format(comparison_path))
    except Exception as e:
        print("Error in comparing results: {}".format(str(e)))
        print("Continuing with analysis...")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fractal Analysis Test")
    parser.add_argument('--wmap-file', type=str, help='Path to WMAP power spectrum file')
    parser.add_argument('--planck-file', type=str, help='Path to Planck power spectrum file')
    parser.add_argument('--wmap-only', action='store_true', help='Process only WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Process only Planck data')
    parser.add_argument('--n-simulations', type=int, default=10000, 
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--max-lag', type=int, default=20, 
                        help='Maximum lag for R/S analysis')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--smooth-window', type=int, default=5, 
                        help='Window size for smoothing')
    parser.add_argument('--normalize', action='store_true', help='Apply normalization to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results. Default: results/fractal_analysis_TIMESTAMP')
    parser.add_argument('--parallel', action='store_true', help='Run Monte Carlo simulations in parallel')
    parser.add_argument('--num-processes', type=int, default=None, 
                        help='Number of processes to use for parallel computation (default: all available cores)')
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(parent_dir, "results", "fractal_analysis_{}".format(timestamp))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up WMAP file path
    if args.wmap_file:
        wmap_file = args.wmap_file
    else:
        wmap_file = os.path.join(parent_dir, "data", "wmap", "wmap_tt_spectrum_9yr_v5.txt")
    
    # Set up Planck file path
    if args.planck_file:
        planck_file = args.planck_file
    else:
        planck_file = os.path.join(parent_dir, "data", "planck", "COM_PowerSpect_CMB-TT-full_R3.01.txt")
    
    # Process WMAP data if requested
    if not args.planck_only:
        # Load WMAP data
        print("Loading WMAP data from {}".format(wmap_file))
        wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(wmap_file)
        if wmap_ell is None:
            print("Error loading WMAP data. Skipping WMAP analysis.")
        else:
            # Preprocess WMAP data
            print("Preprocessing WMAP data...")
            wmap_processed = preprocess_data(
                wmap_power, 
                smooth=args.smooth, 
                smooth_window=args.smooth_window,
                normalize=args.normalize,
                detrend=args.detrend
            )
            
            # Run fractal analysis on WMAP data
            print("\nRunning Fractal Analysis on WMAP data...")
            print("Number of simulations: {}".format(args.n_simulations))
            if args.parallel:
                print("Using parallel processing with {} processes".format(
                    args.num_processes if args.num_processes else "all available"))
            wmap_results = run_fractal_analysis(
                wmap_processed, 
                os.path.join(output_dir, 'wmap'), 
                'WMAP', 
                n_simulations=args.n_simulations,
                max_lag=args.max_lag,
                parallel=args.parallel,
                num_processes=args.num_processes
            )
    
    # Process Planck data if requested
    if not args.wmap_only:
        # Load Planck data
        print("Loading Planck data from {}".format(planck_file))
        planck_ell, planck_power, planck_error = load_planck_power_spectrum(planck_file)
        if planck_ell is None:
            print("Error loading Planck data. Skipping Planck analysis.")
        else:
            # Preprocess Planck data
            print("Preprocessing Planck data...")
            planck_processed = preprocess_data(
                planck_power, 
                smooth=args.smooth, 
                smooth_window=args.smooth_window,
                normalize=args.normalize,
                detrend=args.detrend
            )
            
            # Run fractal analysis on Planck data
            print("\nRunning Fractal Analysis on Planck data...")
            print("Number of simulations: {}".format(args.n_simulations))
            if args.parallel:
                print("Using parallel processing with {} processes".format(
                    args.num_processes if args.num_processes else "all available"))
            planck_results = run_fractal_analysis(
                planck_processed, 
                os.path.join(output_dir, 'planck'), 
                'Planck',
                n_simulations=args.n_simulations,
                max_lag=args.max_lag,
                parallel=args.parallel,
                num_processes=args.num_processes
            )
    
    # Compare results if both datasets were analyzed
    if not args.wmap_only and not args.planck_only and 'wmap_results' in locals() and 'planck_results' in locals():
        print("\nComparing WMAP and Planck results...")
        compare_results(wmap_results, planck_results, os.path.join(output_dir, 'comparison'))
    
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
