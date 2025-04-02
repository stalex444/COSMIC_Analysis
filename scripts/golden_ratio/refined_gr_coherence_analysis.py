#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Refined GR-Specific Coherence Analysis
This script provides a more detailed analysis of the Golden Ratio coherence results,
focusing on surrogate data distributions and statistical validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import os
import pickle
import sys

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from golden_ratio.test_gr_specific_coherence import (
    load_wmap_power_spectrum, 
    load_planck_power_spectrum,
    find_golden_ratio_pairs,
    calculate_coherence, 
    simulate_coherence
)

def generate_surrogate_power(power):
    """
    Generate a surrogate power spectrum by randomizing phases while preserving the amplitude.
    This is a custom implementation for the refined coherence analysis.
    
    Parameters:
    power (numpy.ndarray): Power spectrum values
    
    Returns:
    numpy.ndarray: Surrogate power spectrum
    """
    # Convert power to complex domain via FFT
    n = len(power)
    # Add small positive offset to avoid zeros
    min_positive = np.min(power[power > 0]) / 10 if np.any(power > 0) else 1e-6
    fft_data = np.fft.rfft(np.maximum(power, min_positive))
    
    # Generate random phases but keep the same amplitudes
    amplitudes = np.abs(fft_data)
    random_phases = np.random.uniform(0, 2*np.pi, len(fft_data))
    surrogate_fft = amplitudes * np.exp(1j * random_phases)
    
    # Convert back to the time domain
    surrogate_data = np.fft.irfft(surrogate_fft, n)
    
    # Ensure all values are positive (CMB power spectra should be positive)
    return np.abs(surrogate_data)

def load_surrogate_results(filepath):
    """
    Load pickled surrogate results from the GR-Specific Coherence test.
    
    Parameters:
    filepath (str): Path to the pickle file containing surrogate results
    
    Returns:
    dict: Dictionary containing surrogate test results
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def analyze_surrogate_distribution(surrogate_data, actual_coherence, dataset_name):
    """
    Analyze the distribution of coherence values in surrogate datasets.
    
    Parameters:
    surrogate_data (list or array): List of mean coherence values from surrogate datasets
    actual_coherence (float): Mean coherence value from actual CMB data
    dataset_name (str): Name of the dataset (e.g., "WMAP" or "Planck")
    
    Returns:
    dict: Dictionary with analysis results
    """
    # Basic statistics
    surr_mean = np.mean(surrogate_data)
    surr_std = np.std(surrogate_data)
    surr_median = np.median(surrogate_data)
    surr_min = np.min(surrogate_data)
    surr_max = np.max(surrogate_data)
    
    # Calculate percentile of actual value in surrogate distribution
    percentile = stats.percentileofscore(surrogate_data, actual_coherence)
    
    # Calculate z-score
    z_score = (actual_coherence - surr_mean) / surr_std if surr_std > 0 else 0
    
    # Calculate p-value (two-tailed)
    p_value = 1 - percentile/100 if percentile > 50 else percentile/100
    p_value = p_value * 2  # two-tailed
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot histogram of surrogate values
    plt.subplot(2, 1, 1)
    sns.histplot(surrogate_data, kde=True, bins=30)
    plt.axvline(actual_coherence, color='red', linestyle='dashed', linewidth=2, label=f'Actual CMB ({actual_coherence:.4f})')
    plt.axvline(surr_mean, color='green', linestyle='dotted', linewidth=2, label=f'Surrogate Mean ({surr_mean:.4f})')
    plt.title(f'Distribution of GR-Specific Coherence in Surrogate Data - {dataset_name}')
    plt.xlabel('Mean Coherence')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot QQ plot to check normality
    plt.subplot(2, 1, 2)
    stats.probplot(surrogate_data, dist="norm", plot=plt)
    plt.title(f'QQ Plot for Surrogate Distribution - {dataset_name}')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    out_dir = f'results/refined_gr_coherence_analysis_{dataset_name.lower()}'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/gr_coherence_surrogate_analysis_{dataset_name}.png')
    
    return {
        'surrogate_mean': surr_mean,
        'surrogate_std': surr_std,
        'surrogate_median': surr_median,
        'surrogate_min': surr_min,
        'surrogate_max': surr_max,
        'percentile': percentile,
        'z_score': z_score,
        'p_value': p_value
    }

def analyze_pair_distribution(pair_coherence_actual, pair_coherence_surrogates, dataset_name):
    """
    Analyze the distribution of coherence values for individual pairs.
    
    Parameters:
    pair_coherence_actual (array): Coherence values for each GR pair in actual data
    pair_coherence_surrogates (array): Coherence values for each GR pair in surrogate data
                                      Shape: (n_surrogates, n_pairs)
    dataset_name (str): Name of the dataset (e.g., "WMAP" or "Planck")
    
    Returns:
    dict: Dictionary with analysis results for pairs
    """
    n_pairs = len(pair_coherence_actual)
    
    # Count pairs with high coherence (>0.9)
    high_coherence_actual = sum(pair_coherence_actual > 0.9)
    high_coherence_actual_pct = high_coherence_actual / n_pairs * 100
    
    # Count pairs with high coherence in surrogates
    high_coherence_surr = np.sum(pair_coherence_surrogates > 0.9, axis=1)
    high_coherence_surr_pct = high_coherence_surr / n_pairs * 100
    
    # Calculate p-value for proportion of high coherence pairs
    p_value_prop = np.mean(high_coherence_surr_pct >= high_coherence_actual_pct)
    
    # Analyze top 5 pairs
    top5_indices = np.argsort(pair_coherence_actual)[-5:]
    top5_actual = pair_coherence_actual[top5_indices]
    top5_p_values = []
    
    for i, idx in enumerate(top5_indices):
        surrogate_values = pair_coherence_surrogates[:, idx]
        p_value = np.mean(surrogate_values >= pair_coherence_actual[idx])
        top5_p_values.append(p_value)
    
    # Create visualization of pair-wise comparison
    plt.figure(figsize=(14, 10))
    
    # Plot distribution of high coherence pairs
    plt.subplot(2, 2, 1)
    sns.histplot(high_coherence_surr_pct, kde=True, bins=30)
    plt.axvline(high_coherence_actual_pct, color='red', linestyle='dashed', linewidth=2, 
                label=f'Actual ({high_coherence_actual_pct:.1f}%)')
    plt.title(f'Distribution of % High Coherence Pairs (>0.9) - {dataset_name}')
    plt.xlabel('Percentage of Pairs with Coherence >0.9')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot top 5 pairs
    plt.subplot(2, 2, 2)
    x = np.arange(5)
    plt.bar(x, top5_actual)
    plt.axhline(np.mean(pair_coherence_surrogates), color='green', linestyle='dotted', 
                label='Surrogate Mean')
    plt.xticks(x, [f'Pair {i+1}' for i in range(5)])
    plt.ylabel('Coherence Value')
    plt.title(f'Top 5 Coherence Pairs - {dataset_name}')
    plt.legend()
    
    # Plot p-values for top 5 pairs
    plt.subplot(2, 2, 3)
    plt.bar(x, top5_p_values)
    plt.axhline(0.05, color='red', linestyle='dashed', label='p=0.05 threshold')
    plt.xticks(x, [f'Pair {i+1}' for i in range(5)])
    plt.ylabel('p-value')
    plt.title(f'p-values for Top 5 Coherence Pairs - {dataset_name}')
    plt.legend()
    
    # Plot all pairs: actual vs surrogate mean
    plt.subplot(2, 2, 4)
    x = np.arange(n_pairs)
    surrogate_means = np.mean(pair_coherence_surrogates, axis=0)
    surrogate_stds = np.std(pair_coherence_surrogates, axis=0)
    
    plt.errorbar(x, surrogate_means, yerr=surrogate_stds, fmt='o', alpha=0.5, 
                 label='Surrogate Mean ± Std')
    plt.scatter(x, pair_coherence_actual, color='red', label='Actual Data')
    plt.xlabel('Pair Index')
    plt.ylabel('Coherence Value')
    plt.title(f'Comparison of All Pairs: Actual vs Surrogate - {dataset_name}')
    plt.legend()
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    out_dir = f'results/refined_gr_coherence_analysis_{dataset_name.lower()}'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/gr_coherence_pair_analysis_{dataset_name}.png')
    
    return {
        'high_coherence_actual': high_coherence_actual,
        'high_coherence_actual_pct': high_coherence_actual_pct,
        'high_coherence_surr_mean_pct': np.mean(high_coherence_surr_pct),
        'high_coherence_surr_std_pct': np.std(high_coherence_surr_pct),
        'p_value_proportion': p_value_prop,
        'top5_indices': top5_indices,
        'top5_actual': top5_actual,
        'top5_p_values': top5_p_values
    }

def alternative_analysis(dataset_name, data_file, max_ell=None, n_surrogates=1000):
    """
    Implementation of the alternative approach that generates new surrogates.
    This focuses on scale-specific coherence analysis.
    
    Parameters:
    dataset_name (str): Name of the dataset (e.g., "WMAP" or "Planck")
    data_file (str): Path to the file containing actual CMB data
    max_ell (int, optional): Maximum ell value to consider. Set to None to use all data.
    n_surrogates (int): Number of surrogate datasets to generate
    """
    # Set max_ell based on dataset if not specified
    if max_ell is None:
        if 'planck' in dataset_name.lower():
            max_ell = 500
        else:
            max_ell = 1000
    
    print(f"Loading {dataset_name} data from {data_file}...")
    # Load actual CMB data
    if dataset_name.upper() == "WMAP":
        ell, power, _ = load_wmap_power_spectrum(data_file)
    elif dataset_name.upper() == "PLANCK":
        ell, power, _ = load_planck_power_spectrum(data_file)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if ell is None or power is None:
        print(f"Failed to load data from {data_file}")
        return
    
    # Apply max_ell limit
    if max_ell is not None:
        mask = ell <= max_ell
        ell = ell[mask]
        power = power[mask]
    
    # Define golden ratio pairs
    gr_pairs = find_golden_ratio_pairs(ell, max_ell=max_ell, max_pairs=50)
    print(f"Identified {len(gr_pairs)} golden ratio pairs")
    
    # Calculate coherence for actual data
    print(f"Calculating coherence for actual {dataset_name} data...")
    actual_coherence_values, actual_mean_coherence = calculate_coherence(power, ell, gr_pairs)
    
    # Generate surrogate datasets and calculate coherence
    print(f"Running {n_surrogates} simulations for {dataset_name}...")
    surrogate_coherence_values = []
    pair_coherence_surrogates = np.zeros((n_surrogates, len(actual_coherence_values)))
    
    # Create a surrogate power spectrum for each simulation
    for i in range(n_surrogates):
        if i % 100 == 0:
            print(f"  Simulation {i}/{n_surrogates}...")
        
        # Generate a surrogate power spectrum with randomized phases
        surrogate_power = generate_surrogate_power(power)
        
        # Calculate coherence for this surrogate
        try:
            # Get surrogate coherence values
            sim_coherence_values, sim_mean_coherence = calculate_coherence(surrogate_power, ell, gr_pairs)
            
            surrogate_coherence_values.append(sim_mean_coherence)
            
            # Store individual pair coherence values
            for j in range(min(len(sim_coherence_values), len(actual_coherence_values))):
                pair_coherence_surrogates[i, j] = sim_coherence_values[j]
                
        except Exception as e:
            print(f"Error in simulation {i}: {e}")
            # Insert NaN values for this simulation
            surrogate_coherence_values.append(np.nan)
            pair_coherence_surrogates[i, :] = np.nan
    
    # Remove NaN values
    surrogate_coherence_values = np.array(surrogate_coherence_values)
    valid_indices = ~np.isnan(surrogate_coherence_values)
    surrogate_coherence_values = surrogate_coherence_values[valid_indices]
    pair_coherence_surrogates = pair_coherence_surrogates[valid_indices, :]
    
    print(f"Completed {np.sum(valid_indices)} valid simulations out of {n_surrogates}")
    
    # Create output directory
    out_dir = f'results/refined_gr_coherence_analysis_{dataset_name.lower()}'
    os.makedirs(out_dir, exist_ok=True)
    
    # Save surrogate results for future use
    surrogate_results = {
        'surrogate_coherence': surrogate_coherence_values,
        'actual_coherence': actual_mean_coherence,
        'pair_coherence_actual': np.array(actual_coherence_values),
        'pair_coherence_surrogates': pair_coherence_surrogates,
        'gr_pairs': gr_pairs,
        'ell': ell,
        'power': power
    }
    
    with open(f'{out_dir}/surrogate_results.pkl', 'wb') as f:
        pickle.dump(surrogate_results, f)
    
    print(f"Saved surrogate results to {out_dir}/surrogate_results.pkl")
    
    # Analyze overall distribution
    overall_analysis = analyze_surrogate_distribution(
        surrogate_coherence_values, actual_mean_coherence, dataset_name)
    
    # Analyze pair distribution
    pair_analysis = analyze_pair_distribution(
        np.array(actual_coherence_values), pair_coherence_surrogates, dataset_name)
    
    # Create a report file
    with open(f'{out_dir}/refined_analysis_report.txt', 'w') as f:
        f.write(f"Refined GR-Specific Coherence Analysis for {dataset_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Overall Distribution Analysis\n")
        f.write("-"*40 + "\n")
        for key, value in overall_analysis.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nPair Distribution Analysis\n")
        f.write("-"*40 + "\n")
        for key, value in pair_analysis.items():
            if not isinstance(value, (list, np.ndarray)):
                f.write(f"  {key}: {value}\n")
        
        # Add interpretation
        f.write("\nInterpretation of Results\n")
        f.write("-"*40 + "\n")
        
        if overall_analysis['p_value'] < 0.05:
            f.write("The overall coherence pattern is STATISTICALLY SIGNIFICANT.\n")
        else:
            f.write("The overall coherence pattern is NOT statistically significant.\n")
        
        f.write(f"  - Actual coherence: {actual_mean_coherence:.4f}\n")
        f.write(f"  - Surrogate mean: {overall_analysis['surrogate_mean']:.4f}\n")
        f.write(f"  - Z-score: {overall_analysis['z_score']:.4f}\n")
        f.write(f"  - Two-tailed p-value: {overall_analysis['p_value']:.4f}\n\n")
        
        if pair_analysis['p_value_proportion'] < 0.05:
            f.write("The proportion of high coherence pairs is STATISTICALLY SIGNIFICANT.\n")
        else:
            f.write("The proportion of high coherence pairs is NOT statistically significant.\n")
        
        f.write(f"  - Actual % high coherence pairs: {pair_analysis['high_coherence_actual_pct']:.1f}%\n")
        f.write(f"  - Surrogate mean %: {pair_analysis['high_coherence_surr_mean_pct']:.1f}%\n")
        f.write(f"  - p-value: {pair_analysis['p_value_proportion']:.4f}\n\n")
        
        significant_pairs = [i for i, p in enumerate(pair_analysis['top5_p_values']) if p < 0.05]
        if significant_pairs:
            f.write("Significant individual pairs:\n")
            for i in significant_pairs:
                f.write(f"  - Pair {i+1}: p-value = {pair_analysis['top5_p_values'][i]:.4f}\n")
        else:
            f.write("No significant individual pairs found at α=0.05\n")
    
    print(f"Analysis complete. Report saved to {out_dir}/refined_analysis_report.txt")
    
    # Print summary to console
    print("\nSummary of Results:")
    print(f"  Actual Mean Coherence: {actual_mean_coherence:.4f}")
    print(f"  Surrogate Mean Coherence: {overall_analysis['surrogate_mean']:.4f}")
    print(f"  Z-score: {overall_analysis['z_score']:.4f}")
    print(f"  P-value: {overall_analysis['p_value']:.4f}")
    
    if overall_analysis['p_value'] < 0.05:
        print("  Overall: STATISTICALLY SIGNIFICANT")
    else:
        print("  Overall: NOT statistically significant")

def main():
    """Main function to run the analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Refined GR-Specific Coherence Analysis')
    parser.add_argument('--dataset', type=str, choices=['wmap', 'planck', 'both'], default='both',
                        help='Dataset to analyze (wmap, planck, or both)')
    parser.add_argument('--n-surrogates', type=int, default=1000,
                        help='Number of surrogate datasets to generate')
    parser.add_argument('--surrogate-path', type=str, default=None,
                        help='Path to existing surrogate results pickle file')
    
    args = parser.parse_args()
    
    # Define data file paths specific to the COSMIC_Analysis repository
    wmap_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             'data/wmap/wmap_tt_spectrum_9yr_v5.txt')
    planck_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                               'data/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt')
    
    # Verify that data files exist
    if not os.path.exists(wmap_file):
        print(f"Warning: WMAP data file not found at {wmap_file}")
        print("Searching for data file in common locations...")
        alt_locations = [
            '../data/wmap/wmap_tt_spectrum_9yr_v5.txt',
            '../../data/wmap/wmap_tt_spectrum_9yr_v5.txt',
            '/Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/data/wmap_tt_spectrum_9yr_v5.txt'
        ]
        for loc in alt_locations:
            if os.path.exists(loc):
                wmap_file = loc
                print(f"Found WMAP data at: {wmap_file}")
                break
    
    if not os.path.exists(planck_file):
        print(f"Warning: Planck data file not found at {planck_file}")
        print("Searching for data file in common locations...")
        alt_locations = [
            '../data/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt',
            '../../data/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt',
            '/Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/data/planck_tt_spectrum_2018.txt'
        ]
        for loc in alt_locations:
            if os.path.exists(loc):
                planck_file = loc
                print(f"Found Planck data at: {planck_file}")
                break
    
    print(f"Using WMAP data: {wmap_file}")
    print(f"Using Planck data: {planck_file}")
    
    if args.surrogate_path:
        # Use existing surrogate results
        print(f"Loading surrogate results from {args.surrogate_path}")
        try:
            results = load_surrogate_results(args.surrogate_path)
            
            # Extract needed data
            surrogate_coherence = results.get('surrogate_coherence', [])
            actual_coherence = results.get('actual_coherence', 0.0)
            pair_coherence_actual = results.get('pair_coherence_actual', [])
            pair_coherence_surrogates = results.get('pair_coherence_surrogates', [])
            dataset_name = "WMAP" if "wmap" in args.surrogate_path.lower() else "Planck"
            
            # Analyze distribution
            analyze_surrogate_distribution(
                surrogate_coherence, actual_coherence, dataset_name)
            
            # Analyze pair distribution if available
            if len(pair_coherence_actual) > 0 and len(pair_coherence_surrogates) > 0:
                analyze_pair_distribution(
                    pair_coherence_actual, pair_coherence_surrogates, dataset_name)
            
        except Exception as e:
            print(f"Error loading surrogate results: {e}")
            print("Falling back to alternative analysis...")
            
            if args.dataset in ['wmap', 'both']:
                alternative_analysis("WMAP", wmap_file, n_surrogates=args.n_surrogates)
            
            if args.dataset in ['planck', 'both']:
                alternative_analysis("Planck", planck_file, max_ell=500, n_surrogates=args.n_surrogates)
    else:
        # Run alternative analysis
        if args.dataset in ['wmap', 'both']:
            alternative_analysis("WMAP", wmap_file, n_surrogates=args.n_surrogates)
        
        if args.dataset in ['planck', 'both']:
            alternative_analysis("Planck", planck_file, max_ell=500, n_surrogates=args.n_surrogates)

if __name__ == "__main__":
    main()
