#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for Meta-Coherence Test on CMB data.

This module provides utility functions for the extended meta-coherence analysis,
which analyzes the coherence of local coherence measures across different scales 
in the CMB power spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import os
import multiprocessing
from functools import partial


def load_wmap_power_spectrum(file_path):
    """Load WMAP CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Mean multipole moment
        power = data[:, 3]  # Power spectrum value
        error = data[:, 4]  # Error
        return ell, power, error
    except Exception as e:
        print(f"Error loading WMAP power spectrum: {str(e)}")
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
        print(f"Error loading Planck power spectrum: {str(e)}")
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


def calculate_local_coherence(data, window_size=10, step_size=5, method="inverse_std"):
    """
    Calculate local coherence measures across the data using sliding windows.
    
    Args:
        data (numpy.ndarray): Input data array
        window_size (int): Size of the sliding window
        step_size (int): Step size for sliding the window
        method (str): Method to calculate local coherence:
                      "inverse_std" - inverse of the standard deviation of first differences
                      "autocorr" - autocorrelation at lag 1
                      "hurst" - simplified Hurst exponent estimation
                      "detrended_fluct" - simplified detrended fluctuation analysis
        
    Returns:
        tuple: (window_centers, local_coherence_values)
    """
    if len(data) < window_size:
        return [], []
    
    window_centers = []
    local_coherence_values = []
    
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i+window_size]
        window_center = i + window_size // 2
        window_centers.append(window_center)
        
        if method == "inverse_std":
            # Calculate local coherence as the inverse of the standard deviation
            # of the first differences (higher values indicate more coherence)
            diffs = np.diff(window)
            local_coherence = 1.0 / (np.std(diffs) + 1e-10)  # Add small value to avoid division by zero
        
        elif method == "autocorr":
            # Autocorrelation at lag 1
            if len(window) <= 1:
                local_coherence = 0
            else:
                # Normalize window
                norm_window = (window - np.mean(window)) / (np.std(window) + 1e-10)
                # Calculate autocorrelation at lag 1
                autocorr = np.correlate(norm_window[:-1], norm_window[1:])
                local_coherence = autocorr[0] / len(norm_window[:-1])
        
        elif method == "hurst":
            # Simplified Hurst exponent estimation
            # For a coherent time series, H approaches 1
            # For a random time series, H approaches 0.5
            if len(window) <= 1:
                local_coherence = 0.5  # Default to random
            else:
                # Calculate the range
                cumsum = np.cumsum(window - np.mean(window))
                R = np.max(cumsum) - np.min(cumsum)
                # Calculate the standard deviation
                S = np.std(window)
                if S == 0:
                    local_coherence = 0.5
                else:
                    # R/S should scale as n^H
                    RS = R / (S + 1e-10)
                    # Map to [0,1] range
                    H = np.log(RS) / np.log(len(window))
                    local_coherence = max(0, min(1, H))
        
        elif method == "detrended_fluct":
            # Simplified detrended fluctuation analysis
            if len(window) <= 3:
                local_coherence = 0.5  # Default to random
            else:
                # Get cumulative sum of mean-centered window
                Y = np.cumsum(window - np.mean(window))
                
                # Detrend the cumulative sum
                # Use linear regression to find trend
                X = np.arange(len(Y))
                coeffs = np.polyfit(X, Y, 1)
                trend = np.polyval(coeffs, X)
                
                # Calculate fluctuation
                F = np.sqrt(np.mean((Y - trend)**2))
                
                # For highly coherent signal, F will be small
                # Map to [0,1] range with inverse relationship
                if F == 0:
                    local_coherence = 1.0
                else:
                    # Map larger F to smaller coherence
                    local_coherence = 1.0 / (1.0 + F)
        
        else:
            raise ValueError(f"Unknown coherence calculation method: {method}")
        
        local_coherence_values.append(local_coherence)
    
    return np.array(window_centers), np.array(local_coherence_values)


def calculate_meta_coherence(local_coherence_values, method="cv"):
    """
    Calculate meta-coherence as a measure of how coherent the local coherence values are.
    
    Args:
        local_coherence_values (numpy.ndarray): Array of local coherence values
        method (str): Method to calculate meta-coherence:
                      "cv" - inverse of coefficient of variation
                      "autocorr" - autocorrelation at lag 1
                      "entropy" - negative normalized entropy (more order = higher value)
                      "trend_strength" - strength of trend in local coherence
        
    Returns:
        float: Meta-coherence value
    """
    if len(local_coherence_values) < 2:
        return 0.0
    
    if method == "cv":
        # Calculate meta-coherence as the inverse of the coefficient of variation
        # of the local coherence values (higher values indicate more meta-coherence)
        mean = np.mean(local_coherence_values)
        std = np.std(local_coherence_values)
        
        if mean == 0:
            return 0.0
        
        # Coefficient of variation (CV) = std / mean
        # Meta-coherence = 1 / CV = mean / std
        meta_coherence = mean / (std + 1e-10)  # Add small value to avoid division by zero
    
    elif method == "autocorr":
        # Autocorrelation at lag 1 (higher values indicate more meta-coherence)
        # Normalize local coherence values
        norm_lc = local_coherence_values - np.mean(local_coherence_values)
        if np.std(norm_lc) == 0:
            return 0.0
        norm_lc = norm_lc / np.std(norm_lc)
        
        # Calculate autocorrelation at lag 1
        acorr = np.correlate(norm_lc[:-1], norm_lc[1:])
        meta_coherence = acorr[0] / len(norm_lc[:-1])
    
    elif method == "entropy":
        # Entropy-based measure (lower entropy = higher meta-coherence)
        # Discretize the local coherence values into bins
        hist, _ = np.histogram(local_coherence_values, bins=min(10, len(local_coherence_values)//2 + 1))
        
        # Calculate probability distribution
        p = hist / np.sum(hist)
        p = p[p > 0]  # Remove zeros to avoid log(0)
        
        # Calculate entropy
        entropy = -np.sum(p * np.log2(p))
        
        # Normalize entropy to [0, 1] range
        max_entropy = np.log2(len(p))
        if max_entropy == 0:
            norm_entropy = 0
        else:
            norm_entropy = entropy / max_entropy
        
        # Meta-coherence is the opposite of normalized entropy (1 - norm_entropy)
        meta_coherence = 1 - norm_entropy
    
    elif method == "trend_strength":
        # Measure the strength of trend in local coherence values
        X = np.arange(len(local_coherence_values))
        
        # Linear regression
        try:
            coeffs = np.polyfit(X, local_coherence_values, 1)
            # Correlation between fitted trend and actual values
            trend = np.polyval(coeffs, X)
            corr = np.corrcoef(trend, local_coherence_values)[0, 1]
            
            # Take absolute value as we're interested in strength, not direction
            meta_coherence = abs(corr)
        except:
            meta_coherence = 0.0
    
    else:
        raise ValueError(f"Unknown meta-coherence calculation method: {method}")
    
    # Normalize to [0, 1] range using a sigmoid-like function
    normalized_meta_coherence = 2.0 / (1.0 + np.exp(-0.1 * meta_coherence)) - 1.0
    
    return normalized_meta_coherence


def run_monte_carlo_chunk(seed, chunk_idx, data, n_simulations, window_size, step_size, 
                         local_coherence_method, meta_coherence_method):
    """
    Run a chunk of Monte Carlo simulations for parallel processing.
    
    Args:
        seed: Random seed to use
        chunk_idx: Index of the current chunk
        data: Input data array
        n_simulations: Number of simulations in this chunk
        window_size: Size of the sliding window
        step_size: Step size for sliding the window
        local_coherence_method: Method to calculate local coherence
        meta_coherence_method: Method to calculate meta-coherence
        
    Returns:
        list: Simulated meta-coherence values
    """
    # Set the random seed for this process
    np.random.seed(seed + chunk_idx)
    
    # Run simulations
    sim_meta_coherences = []
    for i in range(n_simulations):
        # Create random permutation of the data
        sim_data = np.random.permutation(data)
        _, sim_local_coherence = calculate_local_coherence(
            sim_data, window_size=window_size, step_size=step_size, method=local_coherence_method)
        sim_meta_coherence = calculate_meta_coherence(sim_local_coherence, method=meta_coherence_method)
        sim_meta_coherences.append(sim_meta_coherence)
    
    return sim_meta_coherences


def run_monte_carlo_parallel(data, n_simulations=10000, window_size=10, step_size=5, num_processes=None,
                           local_coherence_method="inverse_std", meta_coherence_method="cv"):
    """
    Run Monte Carlo simulations in parallel to assess the significance of meta-coherence.
    
    Args:
        data (numpy.ndarray): Input data array
        n_simulations (int): Number of simulations
        window_size (int): Size of the sliding window
        step_size (int): Step size for sliding the window
        num_processes (int): Number of processes to use
        local_coherence_method (str): Method for calculating local coherence
        meta_coherence_method (str): Method for calculating meta-coherence
        
    Returns:
        tuple: (p_value, phi_optimality, actual_meta_coherence, sim_meta_coherences, 
                window_centers, local_coherence_values)
    """
    print(f"Running {n_simulations} Monte Carlo simulations to assess significance...")
    print(f"  Local coherence method: {local_coherence_method}")
    print(f"  Meta-coherence method: {meta_coherence_method}")
    
    # Calculate actual meta-coherence
    window_centers, local_coherence_values = calculate_local_coherence(
        data, window_size=window_size, step_size=step_size, method=local_coherence_method)
    actual_meta_coherence = calculate_meta_coherence(local_coherence_values, method=meta_coherence_method)
    
    print(f"  Actual meta-coherence: {actual_meta_coherence:.6f}")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print(f"  Using {num_processes} processes for parallel computation")
    
    # Divide simulations among processes
    chunk_size = n_simulations // num_processes
    remainder = n_simulations % num_processes
    chunks = [chunk_size] * num_processes
    for i in range(remainder):
        chunks[i] += 1
    
    # Set random seed
    seed = 42
    
    # Create pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Run simulations in parallel
    print("  Starting parallel Monte Carlo simulations...")
    results = []
    for i in range(num_processes):
        chunk_result = pool.apply_async(
            run_monte_carlo_chunk, 
            args=(seed, i, data, chunks[i], window_size, step_size, 
                  local_coherence_method, meta_coherence_method)
        )
        results.append(chunk_result)
    
    # Close pool and wait for all processes to finish
    pool.close()
    pool.join()
    
    # Collect results
    sim_meta_coherences = []
    for result in results:
        sim_meta_coherences.extend(result.get())
    
    # Calculate p-value
    p_value = np.mean([1 if sim >= actual_meta_coherence else 0 for sim in sim_meta_coherences])
    
    # Calculate phi-optimality (z-score normalized to [-1, 1])
    sim_mean = np.mean(sim_meta_coherences)
    sim_std = np.std(sim_meta_coherences)
    if sim_std == 0:
        phi_optimality = 0
    else:
        z_score = (actual_meta_coherence - sim_mean) / sim_std
        # Convert z-score to a value between -1 and 1 using a sigmoid-like function
        phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
    
    return (p_value, phi_optimality, actual_meta_coherence, sim_meta_coherences, 
            window_centers, local_coherence_values)


def plot_meta_coherence_results(window_centers, local_coherence, p_value, phi_optimality, 
                               sim_meta_coherences, actual_meta_coherence, title, output_path):
    """
    Plot meta-coherence analysis results.
    
    Args:
        window_centers (numpy.ndarray): Center positions of sliding windows
        local_coherence (numpy.ndarray): Local coherence values
        p_value (float): P-value from Monte Carlo simulations
        phi_optimality (float): Phi-optimality measure
        sim_meta_coherences (list): Simulated meta-coherence values
        actual_meta_coherence (float): Actual meta-coherence value
        title (str): Plot title
        output_path (str): Path to save the plot
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot local coherence values
    axs[0].plot(window_centers, local_coherence, 'b-', label='Local Coherence')
    axs[0].set_xlabel('Window Center Position')
    axs[0].set_ylabel('Local Coherence')
    axs[0].set_title(f'{title}\nLocal Coherence Values')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot histogram of simulated meta-coherence values
    axs[1].hist(sim_meta_coherences, bins=30, alpha=0.7, color='gray', 
                label=f'Surrogate Simulations (n={len(sim_meta_coherences)})')
    axs[1].axvline(actual_meta_coherence, color='r', linestyle='dashed', linewidth=2, 
                  label=f'Actual Meta-Coherence: {actual_meta_coherence:.6f}')
    
    # Add p-value and phi-optimality to the plot
    axs[1].text(0.05, 0.95, f'p-value: {p_value:.6f}', transform=axs[1].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axs[1].text(0.05, 0.87, f'Phi-Optimality: {phi_optimality:.6f}', transform=axs[1].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axs[1].set_xlabel('Meta-Coherence Value')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Distribution of Simulated Meta-Coherence Values\nSignificant: {p_value < 0.05}')
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def analyze_multiple_mathematical_constants(data, constants, window_size=10, step_size=5, 
                                          local_coherence_method="inverse_std", 
                                          meta_coherence_method="cv", 
                                          n_sims_per_constant=1000):
    """
    Analyze meta-coherence for regions related to multiple mathematical constants.
    
    Args:
        data (numpy.ndarray): Input data array
        constants (dict): Dictionary of constants {name: value}
        window_size (int): Window size for local coherence
        step_size (int): Step size for sliding window
        local_coherence_method (str): Method for local coherence
        meta_coherence_method (str): Method for meta-coherence
        n_sims_per_constant (int): Number of simulations per constant
    
    Returns:
        dict: Results for each constant
    """
    results = {}
    
    # Valid data length
    data_length = len(data)
    if data_length < 2 * window_size:
        print("Data length is too small for analysis")
        return results
    
    # Analyze each constant
    for const_name, const_value in constants.items():
        print(f"\nAnalyzing meta-coherence for {const_name} (value={const_value})")
        
        # Find regions related by the constant
        region_pairs = []
        
        # Only check a subset of potential start points to improve performance
        step = max(1, data_length // 20)
        for start in range(0, data_length - window_size, step):
            # Check if there's a region related by the constant
            related_start = int(start * const_value)
            if related_start + window_size <= data_length:
                region_pairs.append((start, related_start))
        
        print(f"  Found {len(region_pairs)} region pairs related by {const_name}")
        
        if not region_pairs:
            results[const_name] = {
                'meta_coherence': 0,
                'p_value': 1.0,
                'phi_optimality': 0,
                'significant': False
            }
            continue
            
        # Measure coherence between related regions
        coherence_values = []
        
        for start1, start2 in region_pairs:
            region1 = data[start1:start1+window_size]
            region2 = data[start2:start2+window_size]
            
            # Calculate correlation between regions
            correlation = np.corrcoef(region1, region2)[0, 1]
            if not np.isnan(correlation):
                coherence_values.append(abs(correlation))
        
        if not coherence_values:
            results[const_name] = {
                'meta_coherence': 0,
                'p_value': 1.0,
                'phi_optimality': 0,
                'significant': False
            }
            continue
        
        # Calculate meta-coherence for these coherence values
        meta_coherence = calculate_meta_coherence(np.array(coherence_values), method=meta_coherence_method)
        
        # Run simulations
        sim_meta_coherences = []
        for _ in range(n_sims_per_constant):
            # Generate surrogate data by permutation
            sim_data = np.random.permutation(data)
            
            # Calculate coherence values for surrogate
            sim_coherence_values = []
            for start1, start2 in region_pairs:
                sim_region1 = sim_data[start1:start1+window_size]
                sim_region2 = sim_data[start2:start2+window_size]
                
                sim_correlation = np.corrcoef(sim_region1, sim_region2)[0, 1]
                if not np.isnan(sim_correlation):
                    sim_coherence_values.append(abs(sim_correlation))
            
            if sim_coherence_values:
                sim_meta = calculate_meta_coherence(np.array(sim_coherence_values), method=meta_coherence_method)
                sim_meta_coherences.append(sim_meta)
        
        # Calculate p-value
        p_value = np.mean([1 if sim >= meta_coherence else 0 for sim in sim_meta_coherences])
        
        # Calculate phi-optimality
        sim_mean = np.mean(sim_meta_coherences)
        sim_std = np.std(sim_meta_coherences)
        if sim_std == 0:
            phi_optimality = 0
        else:
            z_score = (meta_coherence - sim_mean) / sim_std
            phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
        
        results[const_name] = {
            'meta_coherence': meta_coherence,
            'p_value': p_value,
            'phi_optimality': phi_optimality,
            'significant': p_value < 0.05,
            'coherence_values': coherence_values,
            'sim_meta_coherences': sim_meta_coherences
        }
        
        print(f"  Meta-coherence: {meta_coherence:.6f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  phi-optimality: {phi_optimality:.6f}")
        print(f"  Significant: {p_value < 0.05}")
    
    return results
