#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for loading and analyzing WMAP data.
"""

from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def load_wmap_power_spectrum(file_path):
    """
    Load WMAP CMB power spectrum data.
    
    Args:
        file_path (str): Path to the WMAP power spectrum file
            
    Returns:
        tuple: (ell, power, error) arrays
    """
    try:
        # Load the power spectrum data
        # WMAP power spectrum format: l, TT, error
        data = np.loadtxt(file_path, skiprows=1)
        
        ell = data[:, 0]  # Multipole moment
        power = data[:, 1]  # Power spectrum value (Dl = l(l+1)Cl/2π in μK²)
        error = data[:, 2]  # Error
        
        return ell, power, error
        
    except Exception as e:
        print("Error loading WMAP power spectrum: {}".format(str(e)))
        return None, None, None

def preprocess_data(data, smooth=False, smooth_window=5, normalize=True, detrend=False):
    """
    Preprocess data for analysis.
    
    Args:
        data (numpy.ndarray): Input data array
        smooth (bool): Whether to apply smoothing
        smooth_window (int): Window size for smoothing
        normalize (bool): Whether to normalize the data
        detrend (bool): Whether to remove linear trend
        
    Returns:
        numpy.ndarray: Preprocessed data
    """
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

def analyze_golden_ratio(data, n_shuffles=1000):
    """
    Simple test to analyze golden ratio patterns in the data.
    
    Args:
        data (numpy.ndarray): Input data array
        n_shuffles (int): Number of shuffles for significance testing
        
    Returns:
        dict: Analysis results
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Calculate correlations between data points separated by golden ratio
    correlations = []
    for i in range(len(data) - int(phi)):
        j = i + int(phi)
        correlations.append(data[i] * data[j])
    
    # Calculate mean correlation
    mean_correlation = np.mean(correlations)
    
    # Perform significance testing
    shuffled_correlations = []
    for _ in range(n_shuffles):
        shuffled_data = np.random.permutation(data)
        shuffle_corrs = []
        for i in range(len(shuffled_data) - int(phi)):
            j = i + int(phi)
            shuffle_corrs.append(shuffled_data[i] * shuffled_data[j])
        shuffled_correlations.append(np.mean(shuffle_corrs))
    
    # Calculate p-value
    p_value = np.mean([1 if abs(sc) >= abs(mean_correlation) else 0 for sc in shuffled_correlations])
    
    return {
        'correlation': mean_correlation,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def main():
    # Path to WMAP power spectrum file
    power_spectrum_file = 'wmap_data/raw_data/wmap_tt_spectrum_9yr_v5.txt'
    
    # Check if file exists
    if not os.path.exists(power_spectrum_file):
        print("Error: WMAP power spectrum file not found: {}".format(power_spectrum_file))
        return 1
    
    # Load WMAP power spectrum
    print("Loading WMAP power spectrum...")
    ell, power, error = load_wmap_power_spectrum(power_spectrum_file)
    
    if ell is None:
        print("Error loading WMAP power spectrum.")
        return 1
    
    print("Loaded WMAP power spectrum with {} multipoles".format(len(ell)))
    
    # Preprocess power spectrum data
    print("Preprocessing data...")
    processed_power = preprocess_data(
        power, 
        smooth=True, 
        smooth_window=5, 
        normalize=True, 
        detrend=True
    )
    
    # Analyze golden ratio patterns
    print("Analyzing golden ratio patterns...")
    results = analyze_golden_ratio(processed_power, n_shuffles=1000)
    
    print("\nAnalysis Results:")
    print("  Golden Ratio Correlation: {:.4f}".format(results['correlation']))
    print("  P-value: {:.4f}".format(results['p_value']))
    print("  Significant: {}".format(results['significant']))
    
    # Plot the power spectrum
    plt.figure(figsize=(12, 6))
    
    # Original power spectrum
    plt.subplot(2, 1, 1)
    plt.errorbar(ell, power, yerr=error, fmt='o', markersize=3, alpha=0.5)
    plt.xlabel('Multipole (ℓ)')
    plt.ylabel('Power (μK²)')
    plt.title('WMAP 9-year Temperature Power Spectrum')
    plt.grid(True, alpha=0.3)
    
    # Processed power spectrum
    plt.subplot(2, 1, 2)
    plt.plot(ell, processed_power)
    plt.xlabel('Multipole (ℓ)')
    plt.ylabel('Normalized Power')
    plt.title('Processed Power Spectrum')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'wmap_power_spectrum.png'), dpi=150)
    print("Plot saved to: {}".format(os.path.join(output_dir, 'wmap_power_spectrum.png')))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
