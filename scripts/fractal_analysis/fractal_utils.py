#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for Fractal Analysis of CMB data.

This module provides functions for calculating the Hurst exponent and
performing fractal analysis on CMB power spectrum data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FractalAnalysis")

# Mathematical constants
PHI = 1.618033988749895  # Golden ratio
PI = np.pi
E = np.e
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
LN2 = np.log(2)


def preprocess_data(data, smooth=False, smooth_window=5, normalize=True, detrend=False):
    """
    Preprocess data for fractal analysis.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array
    smooth : bool, optional
        Whether to apply smoothing, by default False
    smooth_window : int, optional
        Window size for smoothing, by default 5
    normalize : bool, optional
        Whether to normalize the data, by default True
    detrend : bool, optional
        Whether to remove linear trend, by default False
    
    Returns:
    --------
    numpy.ndarray
        Preprocessed data
    """
    processed_data = data.copy()
    
    # Detrend the data if requested
    if detrend:
        processed_data = signal.detrend(processed_data)
    
    # Apply smoothing if requested
    if smooth:
        if smooth_window % 2 == 0:
            smooth_window += 1  # Ensure window is odd
        processed_data = signal.savgol_filter(processed_data, smooth_window, 2)
    
    # Normalize the data if requested
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
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array
    max_lag : int, optional
        Maximum lag for R/S analysis, by default 20
    
    Returns:
    --------
    tuple
        (hurst_exponent, log_lags, log_rs)
    """
    # Ensure proper data length
    n = len(data)
    max_lag = min(max_lag, n // 4)
    
    # Create lag values
    lags = np.logspace(0, np.log10(max_lag), 20).astype(int)
    lags = np.unique(lags)  # Remove duplicates
    
    # Calculate R/S values for each lag
    rs_values = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # Skip if lag is too small
        if lag < 2:
            continue
            
        # Calculate R/S for current lag
        rs_values[i] = 0
        n_partitions = int(np.floor(n / lag))
        
        # Skip if not enough partitions
        if n_partitions < 1:
            continue
            
        # Calculate R/S for each partition and average
        for j in range(n_partitions):
            # Extract partition
            partition = data[j*lag:(j+1)*lag]
            
            # Calculate mean and standard deviation
            mean = np.mean(partition)
            std = np.std(partition)
            
            # Skip if std is too small
            if std < 1e-10:
                continue
                
            # Calculate cumulative deviate series
            cumsum = np.cumsum(partition - mean)
            
            # Calculate range and rescale
            r = np.max(cumsum) - np.min(cumsum)
            rs = r / std if std > 0 else 0
            
            # Add to average
            rs_values[i] += rs
            
        # Calculate average R/S value
        rs_values[i] /= n_partitions if n_partitions > 0 else 1
    
    # Remove any zero values
    valid_indices = (rs_values > 0) & (lags > 1)
    lags = lags[valid_indices]
    rs_values = rs_values[valid_indices]
    
    # Log-transform for linear regression
    log_lags = np.log10(lags)
    log_rs = np.log10(rs_values)
    
    # Linear regression to estimate Hurst exponent
    if len(log_lags) > 1:
        hurst_exponent, _, _, _, _ = stats.linregress(log_lags, log_rs)
    else:
        hurst_exponent = np.nan
    
    return hurst_exponent, log_lags, log_rs


def generate_surrogate(data, method="phase_randomization"):
    """
    Generate surrogate data for statistical testing.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Original data array
    method : str, optional
        Method for generating surrogates, by default "phase_randomization"
    
    Returns:
    --------
    numpy.ndarray
        Surrogate data with same power spectrum but randomized phases
    """
    if method == "phase_randomization":
        # Fourier transform
        fft_data = np.fft.fft(data)
        
        # Keep amplitudes but randomize phases
        amplitudes = np.abs(fft_data)
        phases = np.random.uniform(0, 2*np.pi, len(data))
        
        # Ensure conjugate symmetry for real output
        phases[0] = 0
        if len(data) % 2 == 0:
            phases[len(data)//2] = 0
            
        # Combine amplitudes and randomized phases
        fft_surrogate = amplitudes * np.exp(1j * phases)
        
        # Inverse Fourier transform and take real part
        surrogate = np.real(np.fft.ifft(fft_surrogate))
        
        # Ensure same mean and std as original data
        surrogate = (surrogate - np.mean(surrogate)) / np.std(surrogate)
        surrogate = surrogate * np.std(data) + np.mean(data)
        
        return surrogate
    
    elif method == "shuffle":
        # Simple shuffling of data points
        surrogate = np.random.permutation(data)
        return surrogate
    
    else:
        raise ValueError(f"Unknown surrogate method: {method}")


def calculate_phi_optimality(actual_hurst, sim_hursts, phi_target=PHI-1):
    """
    Calculate how optimal the fractal behavior is relative to the Golden Ratio.
    
    The Hurst exponent of H = 0.618... (phi-1) is of particular interest as it
    represents optimal complexity and self-similarity related to the Golden Ratio.
    
    Parameters:
    -----------
    actual_hurst : float
        Actual Hurst exponent from data
    sim_hursts : numpy.ndarray
        Array of Hurst exponents from simulations
    phi_target : float, optional
        Target value to compare with (phi-1 by default)
    
    Returns:
    --------
    float
        Phi optimality score (higher is better)
    """
    # Calculate distances to phi-1
    actual_distance = np.abs(actual_hurst - phi_target)
    sim_distances = np.abs(sim_hursts - phi_target)
    
    # Convert to optimality scores (higher is better)
    # Using a logarithmic scale and sigmoid function for stability
    actual_score = 1 / (1 + np.exp(actual_distance * 10))
    sim_scores = 1 / (1 + np.exp(sim_distances * 10))
    
    # Normalize by the mean of simulation scores
    mean_sim_score = np.mean(sim_scores)
    if mean_sim_score > 0:
        phi_optimality = actual_score / mean_sim_score
    else:
        phi_optimality = 0
    
    return phi_optimality


def plot_fractal_results(log_lags, log_rs, p_value, phi_optimality, 
                        sim_hursts, actual_hurst, title, output_path):
    """
    Plot fractal analysis results.
    
    Parameters:
    -----------
    log_lags : numpy.ndarray
        Log-transformed lag values
    log_rs : numpy.ndarray
        Log-transformed R/S values
    p_value : float
        P-value from statistical testing
    phi_optimality : float
        Phi optimality score
    sim_hursts : numpy.ndarray
        Array of Hurst exponents from simulations
    actual_hurst : float
        Actual Hurst exponent from data
    title : str
        Plot title
    output_path : str
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot log-log R/S analysis
    ax1.scatter(log_lags, log_rs, c='blue', marker='o')
    
    # Add linear fit line
    if len(log_lags) > 1:
        fit = np.polyfit(log_lags, log_rs, 1)
        fit_line = np.poly1d(fit)
        x_range = np.linspace(min(log_lags), max(log_lags), 100)
        ax1.plot(x_range, fit_line(x_range), 'r--', 
                label=f'H = {actual_hurst:.4f}')
    
    # Add reference lines for H=0.5 and H=phi-1
    if len(log_lags) > 1:
        mid_x = np.mean(log_lags)
        mid_y = np.mean(log_rs)
        
        # H=0.5 (random walk)
        h05_y = mid_y + 0.5 * (log_lags - mid_x)
        ax1.plot(log_lags, h05_y, 'g--', alpha=0.5, 
                label='H = 0.5 (Random)')
        
        # H=phi-1 (Golden Ratio optimality)
        h_phi_y = mid_y + (PHI-1) * (log_lags - mid_x)
        ax1.plot(log_lags, h_phi_y, 'orange', alpha=0.5, 
                label=f'H = {PHI-1:.4f} (φ-1)')
    
    ax1.set_xlabel('Log(Lag)')
    ax1.set_ylabel('Log(R/S)')
    ax1.set_title('Rescaled Range Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot distribution of simulated Hurst exponents
    ax2.hist(sim_hursts, bins=30, alpha=0.7, color='gray')
    
    # Add vertical line for actual H and phi-1
    ax2.axvline(actual_hurst, color='r', linestyle='-', 
               label=f'Actual H = {actual_hurst:.4f}')
    ax2.axvline(PHI-1, color='orange', linestyle='--', 
               label=f'φ-1 = {PHI-1:.4f}')
    
    # Add p-value and phi-optimality annotations
    significance = "Significant" if p_value < 0.05 else "Not significant"
    ax2.annotate(f'p-value: {p_value:.6f} ({significance})', 
                xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    ax2.annotate(f'φ-optimality: {phi_optimality:.4f}', 
                xy=(0.05, 0.87), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    ax2.set_xlabel('Hurst Exponent')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Simulated Hurst Exponents')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set main title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot saved to {output_path}")
