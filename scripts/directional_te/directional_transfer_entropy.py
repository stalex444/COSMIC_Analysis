#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

"""
Directional Transfer Entropy Test for CMB data.

This script implements a specialized test that analyzes the directionality
of information flow between different scales in the CMB power spectrum.
It measures both forward and reverse transfer entropy to determine if
information flows preferentially in one direction.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
import argparse
import time
import traceback
import multiprocessing
from functools import partial
import gc  # Garbage collection
import psutil  # For memory monitoring (pip install psutil)

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


def calculate_transfer_entropy(source, target, bins=10, delay=1, max_points=500):
    """
    Calculate transfer entropy from source to target time series.
    
    Args:
        source (numpy.ndarray): Source time series
        target (numpy.ndarray): Target time series
        bins (int): Number of bins for discretization
        delay (int): Time delay for information transfer
        max_points (int): Maximum number of points to use for calculation
        
    Returns:
        float: Transfer entropy value
    """
    # Ensure arrays are the same length and limit size for performance
    length = min(len(source), len(target), max_points)
    source = source[:length]
    target = target[:length]
    
    # Create delayed versions
    target_past = target[:-delay]
    target_future = target[delay:]
    source_past = source[:-delay]
    
    # Use fewer bins if we have limited data points
    actual_bins = min(bins, max(3, length // 10))
    
    # Discretize the data using binning
    s_bins = np.linspace(min(source_past), max(source_past), actual_bins+1)
    t_bins = np.linspace(min(target_past), max(target_past), actual_bins+1)
    tf_bins = np.linspace(min(target_future), max(target_future), actual_bins+1)
    
    # Ensure discretized values are within bounds (0 to bins-1)
    s_disc = np.clip(np.digitize(source_past, s_bins) - 1, 0, actual_bins-1)
    t_disc = np.clip(np.digitize(target_past, t_bins) - 1, 0, actual_bins-1)
    tf_disc = np.clip(np.digitize(target_future, tf_bins) - 1, 0, actual_bins-1)
    
    # Use numpy's histogram2d and histogramdd for faster joint probability calculation
    st_joint_counts, _, _ = np.histogram2d(s_disc, t_disc, bins=[actual_bins, actual_bins])
    tf_joint_counts, _, _ = np.histogram2d(t_disc, tf_disc, bins=[actual_bins, actual_bins])
    
    # For 3D histogram, we need to reshape the data
    stf_data = np.vstack([s_disc, t_disc, tf_disc]).T
    stf_joint_counts, _ = np.histogramdd(stf_data, bins=[actual_bins, actual_bins, actual_bins])
    
    # Normalize to get probabilities (add small epsilon to avoid division by zero)
    epsilon = 1e-10
    st_joint_prob = st_joint_counts / (np.sum(st_joint_counts) + epsilon)
    stf_joint_prob = stf_joint_counts / (np.sum(stf_joint_counts) + epsilon)
    tf_joint_prob = tf_joint_counts / (np.sum(tf_joint_counts) + epsilon)
    
    # Calculate transfer entropy using vectorized operations where possible
    te = 0
    for i in range(actual_bins):
        for j in range(actual_bins):
            for k in range(actual_bins):
                if stf_joint_prob[i, j, k] > epsilon and st_joint_prob[i, j] > epsilon and tf_joint_prob[j, k] > epsilon:
                    te += stf_joint_prob[i, j, k] * np.log2(stf_joint_prob[i, j, k] * tf_joint_prob[j, k] / 
                                                      (st_joint_prob[i, j] * tf_joint_prob[j, k]))
    
    return te


def calculate_directional_transfer_entropy(data1, data2, bins=10, delay=1, max_points=500):
    """
    Calculate directional transfer entropy between two time series.
    
    This function calculates both forward and reverse transfer entropy to
    determine the net direction of information flow.
    
    Args:
        data1: First time series
        data2: Second time series
        bins: Number of bins for discretization
        delay: Time delay for information transfer
        max_points: Maximum number of points to use for calculation
        
    Returns:
        tuple: (forward_te, reverse_te, net_te, direction)
        - forward_te: Transfer entropy from data1 to data2
        - reverse_te: Transfer entropy from data2 to data1
        - net_te: Net transfer entropy (forward_te - reverse_te)
        - direction: Direction of information flow (1 for forward, -1 for reverse, 0 for balanced)
    """
    # Calculate forward transfer entropy (data1 -> data2)
    forward_te = calculate_transfer_entropy(data1, data2, bins, delay, max_points)
    
    # Calculate reverse transfer entropy (data2 -> data1)
    reverse_te = calculate_transfer_entropy(data2, data1, bins, delay, max_points)
    
    # Calculate net transfer entropy
    net_te = forward_te - reverse_te
    
    # Determine direction of information flow
    if abs(net_te) < 1e-6:  # Close to zero, no clear direction
        direction = 0
    elif net_te > 0:  # Forward direction dominates
        direction = 1
    else:  # Reverse direction dominates
        direction = -1
    
    return forward_te, reverse_te, net_te, direction


def analyze_scale_pairs(data, scales, constants=None, bins=10, delay=1, max_points=500):
    """
    Analyze directional transfer entropy between all pairs of scales and
    identify which mathematical constants optimize the directionality.
    
    Args:
        data: Input data array
        scales: Number of scales to analyze
        constants: Dictionary of mathematical constants to test
        bins: Number of bins for discretization
        delay: Time delay for information transfer
        max_points: Maximum number of points to use for calculation
        
    Returns:
        dict: Results of the directional analysis
    """
    if constants is None:
        constants = {
            "phi": 1.618033988749895,  # Golden ratio
            "e": 2.718281828459045,    # Euler's number
            "pi": 3.141592653589793,   # Pi
            "sqrt2": 1.4142135623730951,  # Square root of 2
            "sqrt3": 1.7320508075688772,  # Square root of 3
            "ln2": 0.6931471805599453   # Natural logarithm of 2
        }
    
    # Split data into scales
    scale_size = len(data) // scales
    scale_data = [data[i*scale_size:(i+1)*scale_size] for i in range(scales)]
    
    # Create scale pairs
    scale_pairs = []
    for i in range(scales):
        for j in range(i+1, scales):  # Only pairs where j > i to avoid duplicates
            scale_pairs.append((i, j))
    
    # Calculate directional transfer entropy for each pair
    pair_results = []
    
    for i, j in scale_pairs:
        # Calculate the scale ratio
        ratio = (j + 0.5) / (i + 0.5)  # Use midpoints of scale ranges
        
        # Calculate directional TE
        forward_te, reverse_te, net_te, direction = calculate_directional_transfer_entropy(
            scale_data[i], scale_data[j], bins, delay, max_points
        )
        
        # Find which constant is closest to this ratio
        closest_constant = None
        min_distance = float('inf')
        
        for name, value in constants.items():
            distance = abs(ratio - value) / value  # Normalized distance
            if distance < min_distance:
                min_distance = distance
                closest_constant = name
        
        # Store results
        pair_results.append({
            'scales': (i, j),
            'ratio': ratio,
            'forward_te': forward_te,
            'reverse_te': reverse_te,
            'net_te': net_te,
            'direction': direction,
            'closest_constant': closest_constant,
            'constant_distance': min_distance
        })
    
    # Analyze results by constant
    constant_results = {}
    for name in constants:
        # Filter pairs where this constant is closest
        constant_pairs = [p for p in pair_results if p['closest_constant'] == name]
        
        if constant_pairs:
            # Calculate average metrics
            avg_forward_te = np.mean([p['forward_te'] for p in constant_pairs])
            avg_reverse_te = np.mean([p['reverse_te'] for p in constant_pairs])
            avg_net_te = np.mean([p['net_te'] for p in constant_pairs])
            
            # Count directions
            forward_count = sum(1 for p in constant_pairs if p['direction'] == 1)
            reverse_count = sum(1 for p in constant_pairs if p['direction'] == -1)
            balanced_count = sum(1 for p in constant_pairs if p['direction'] == 0)
            
            # Calculate directional ratio (-1 to 1)
            total = forward_count + reverse_count + balanced_count
            if total > 0:
                directional_ratio = (forward_count - reverse_count) / total
            else:
                directional_ratio = 0
            
            constant_results[name] = {
                'pairs_count': len(constant_pairs),
                'avg_forward_te': avg_forward_te,
                'avg_reverse_te': avg_reverse_te,
                'avg_net_te': avg_net_te,
                'forward_count': forward_count,
                'reverse_count': reverse_count,
                'balanced_count': balanced_count,
                'directional_ratio': directional_ratio
            }
        else:
            constant_results[name] = {
                'pairs_count': 0,
                'avg_forward_te': 0,
                'avg_reverse_te': 0,
                'avg_net_te': 0,
                'forward_count': 0,
                'reverse_count': 0,
                'balanced_count': 0,
                'directional_ratio': 0
            }
    
    # Calculate overall metrics
    overall_forward_te = np.mean([p['forward_te'] for p in pair_results])
    overall_reverse_te = np.mean([p['reverse_te'] for p in pair_results])
    overall_net_te = np.mean([p['net_te'] for p in pair_results])
    
    forward_count = sum(1 for p in pair_results if p['direction'] == 1)
    reverse_count = sum(1 for p in pair_results if p['direction'] == -1)
    balanced_count = sum(1 for p in pair_results if p['direction'] == 0)
    
    total_count = len(pair_results)
    if total_count > 0:
        overall_directional_ratio = (forward_count - reverse_count) / total_count
    else:
        overall_directional_ratio = 0
    
    # Find the dominant constant (highest absolute directional ratio)
    dominant_constant = max(constant_results.items(), 
                          key=lambda x: abs(x[1]['directional_ratio']) if x[1]['pairs_count'] > 0 else 0)
    
    # Compile results
    results = {
        'pair_results': pair_results,
        'constant_results': constant_results,
        'overall_forward_te': overall_forward_te,
        'overall_reverse_te': overall_reverse_te,
        'overall_net_te': overall_net_te,
        'forward_count': forward_count,
        'reverse_count': reverse_count,
        'balanced_count': balanced_count,
        'overall_directional_ratio': overall_directional_ratio,
        'dominant_constant': dominant_constant[0] if dominant_constant[1]['pairs_count'] > 0 else None
    }
    
    return results


def generate_surrogate_data(data, num_surrogates=1000):
    """Generate surrogate datasets for statistical validation."""
    surrogate_datasets = []
    
    for i in range(num_surrogates):
        # Create a permuted copy
        surrogate = np.random.permutation(data.copy())
        surrogate_datasets.append(surrogate)
    
    return surrogate_datasets


def run_monte_carlo_simulation(data, scales, constants=None, n_simulations=100, 
                            bins=10, delay=1, max_points=500):
    """
    Run Monte Carlo simulations to assess the statistical significance of directional transfer entropy.
    
    Args:
        data: Input data array
        scales: Number of scales to analyze
        constants: Dictionary of mathematical constants to test
        n_simulations: Number of simulations
        bins: Number of bins for discretization
        delay: Time delay for information transfer
        max_points: Maximum number of points to use for calculation
        
    Returns:
        dict: Results of the Monte Carlo simulations
    """
    # Analyze actual data
    print("Analyzing actual data...")
    actual_results = analyze_scale_pairs(data, scales, constants, bins, delay, max_points)
    
    # Run simulations
    print(f"Running {n_simulations} Monte Carlo simulations...")
    sim_directional_ratios = []
    sim_net_tes = []
    
    for i in range(n_simulations):
        if i % 10 == 0:
            print(f"  Simulation {i}/{n_simulations}...")
        
        # Generate surrogate data
        surrogate = np.random.permutation(data.copy())
        
        # Analyze surrogate data
        sim_results = analyze_scale_pairs(surrogate, scales, constants, bins, delay, max_points)
        
        # Store results
        sim_directional_ratios.append(sim_results['overall_directional_ratio'])
        sim_net_tes.append(sim_results['overall_net_te'])
    
    # Calculate p-values
    p_directional = np.mean([1 if abs(sim) >= abs(actual_results['overall_directional_ratio']) else 0 
                          for sim in sim_directional_ratios])
    
    p_net_te = np.mean([1 if abs(sim) >= abs(actual_results['overall_net_te']) else 0 
                    for sim in sim_net_tes])
    
    # Calculate z-scores
    if len(sim_directional_ratios) > 0:
        z_directional = ((actual_results['overall_directional_ratio'] - np.mean(sim_directional_ratios)) / 
                      np.std(sim_directional_ratios) if np.std(sim_directional_ratios) > 0 else 0)
    else:
        z_directional = 0
    
    if len(sim_net_tes) > 0:
        z_net_te = ((actual_results['overall_net_te'] - np.mean(sim_net_tes)) / 
                  np.std(sim_net_tes) if np.std(sim_net_tes) > 0 else 0)
    else:
        z_net_te = 0
    
    # Compile Monte Carlo results
    mc_results = {
        'p_directional': p_directional,
        'p_net_te': p_net_te,
        'z_directional': z_directional,
        'z_net_te': z_net_te,
        'sim_directional_ratios': sim_directional_ratios,
        'sim_net_tes': sim_net_tes
    }
    
    results = {
        'actual_results': actual_results,
        'mc_results': mc_results
    }
    
    return results


def run_directional_te_test(data, output_dir, name, scales=5, constants=None, 
                         n_simulations=100, bins=10, delay=1, max_points=500):
    """
    Run directional transfer entropy test on the provided data.
    
    Args:
        data: Input data array
        output_dir: Output directory for results
        name: Name of the dataset
        scales: Number of scales to analyze
        constants: Dictionary of mathematical constants to test
        n_simulations: Number of simulations
        bins: Number of bins for discretization
        delay: Time delay for information transfer
        max_points: Maximum number of points to use for calculation
        
    Returns:
        dict: Test results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running Directional Transfer Entropy Test on {name} data...")
    
    # Run Monte Carlo simulations
    results = run_monte_carlo_simulation(
        data, scales, constants, n_simulations, bins, delay, max_points
    )
    
    # Save results
    results_file = os.path.join(output_dir, f"{name.lower()}_directional_te_results.npz")
    np.savez_compressed(results_file, **{
        'overall_forward_te': results['actual_results']['overall_forward_te'],
        'overall_reverse_te': results['actual_results']['overall_reverse_te'],
        'overall_net_te': results['actual_results']['overall_net_te'],
        'forward_count': results['actual_results']['forward_count'],
        'reverse_count': results['actual_results']['reverse_count'],
        'balanced_count': results['actual_results']['balanced_count'],
        'overall_directional_ratio': results['actual_results']['overall_directional_ratio'],
        'dominant_constant': results['actual_results']['dominant_constant'],
        'p_directional': results['mc_results']['p_directional'],
        'p_net_te': results['mc_results']['p_net_te'],
        'z_directional': results['mc_results']['z_directional'],
        'z_net_te': results['mc_results']['z_net_te'],
        'sim_directional_ratios': results['mc_results']['sim_directional_ratios'],
        'sim_net_tes': results['mc_results']['sim_net_tes']
    })
    
    # Save detailed results as JSON
    import json
    detailed_results = {
        'pair_results': results['actual_results']['pair_results'],
        'constant_results': results['actual_results']['constant_results'],
        'overall_metrics': {
            'forward_te': results['actual_results']['overall_forward_te'],
            'reverse_te': results['actual_results']['overall_reverse_te'],
            'net_te': results['actual_results']['overall_net_te'],
            'forward_count': results['actual_results']['forward_count'],
            'reverse_count': results['actual_results']['reverse_count'],
            'balanced_count': results['actual_results']['balanced_count'],
            'directional_ratio': results['actual_results']['overall_directional_ratio'],
            'dominant_constant': results['actual_results']['dominant_constant']
        },
        'monte_carlo': {
            'p_directional': results['mc_results']['p_directional'],
            'p_net_te': results['mc_results']['p_net_te'],
            'z_directional': results['mc_results']['z_directional'],
            'z_net_te': results['mc_results']['z_net_te']
        }
    }
    
    # Convert NumPy arrays to lists for JSON serialization
    detailed_results_json = json.dumps(detailed_results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    with open(os.path.join(output_dir, f"{name.lower()}_detailed_results.json"), 'w') as f:
        f.write(detailed_results_json)
    
    # Create visualizations
    create_visualizations(results, output_dir, name)
    
    return results


def create_visualizations(results, output_dir, name):
    """Create visualizations of the directional transfer entropy results."""
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot directional flow matrix
        plt.figure(figsize=(10, 8))
        
        # Extract data
        pair_results = results['actual_results']['pair_results']
        
        # Create directional matrix
        if pair_results:
            scales = max(max(p['scales']) for p in pair_results) + 1
            dir_matrix = np.zeros((scales, scales))
            te_matrix = np.zeros((scales, scales))
            
            for p in pair_results:
                i, j = p['scales']
                dir_matrix[i, j] = p['direction']
                dir_matrix[j, i] = -p['direction']  # Opposite direction
                
                te_matrix[i, j] = p['net_te']
                te_matrix[j, i] = -p['net_te']  # Opposite value
            
            # Plot directional matrix
            plt.subplot(2, 2, 1)
            cmap = plt.cm.coolwarm
            im = plt.imshow(dir_matrix, cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar(im, label='Direction')
            plt.title(f'Direction of Information Flow\n({name} data)')
            plt.xlabel('To Scale')
            plt.ylabel('From Scale')
            plt.xticks(range(scales), [f'Scale {i+1}' for i in range(scales)])
            plt.yticks(range(scales), [f'Scale {i+1}' for i in range(scales)])
            
            # Plot transfer entropy matrix
            plt.subplot(2, 2, 2)
            im = plt.imshow(te_matrix, cmap=cmap)
            plt.colorbar(im, label='Net Transfer Entropy')
            plt.title(f'Net Transfer Entropy\n({name} data)')
            plt.xlabel('To Scale')
            plt.ylabel('From Scale')
            plt.xticks(range(scales), [f'Scale {i+1}' for i in range(scales)])
            plt.yticks(range(scales), [f'Scale {i+1}' for i in range(scales)])
            
            # Plot constant-specific results
            plt.subplot(2, 2, 3)
            constant_results = results['actual_results']['constant_results']
            constants = list(constant_results.keys())
            directional_ratios = [constant_results[c]['directional_ratio'] for c in constants]
            
            colors = ['blue' if ratio > 0 else 'red' for ratio in directional_ratios]
            plt.bar(constants, directional_ratios, color=colors)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.title('Directional Ratio by Constant')
            plt.ylabel('Directional Ratio (-1 to 1)')
            plt.xticks(rotation=45)
            
            # Plot Monte Carlo distribution
            plt.subplot(2, 2, 4)
            sim_ratios = results['mc_results']['sim_directional_ratios']
            actual_ratio = results['actual_results']['overall_directional_ratio']
            
            plt.hist(sim_ratios, bins=30, alpha=0.6, color='gray')
            plt.axvline(x=actual_ratio, color='r', linestyle='--', 
                      label=f'Actual: {actual_ratio:.4f}')
            plt.title('Monte Carlo Distribution\nDirectional Ratio')
            plt.xlabel('Directional Ratio')
            plt.ylabel('Frequency')
            plt.legend()
            
            # Add text with key results
            plt.figtext(0.5, 0.01, 
                      f"Overall Direction: {'Forward' if actual_ratio > 0 else 'Reverse'} " +
                      f"| p-value: {results['mc_results']['p_directional']:.4f} " +
                      f"| z-score: {results['mc_results']['z_directional']:.4f}", 
                      ha='center', fontsize=12, 
                      bbox=dict(facecolor='yellow', alpha=0.5))
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.savefig(os.path.join(output_dir, f"{name.lower()}_directional_te.png"))
            plt.close()
            
            # Create scale-specific plots
            plt.figure(figsize=(12, 10))
            
            # Group by scale
            scale_data = {}
            for s in range(scales):
                # Find pairs involving this scale
                scale_pairs = [p for p in pair_results if s in p['scales']]
                
                if scale_pairs:
                    # Calculate metrics
                    forward_count = sum(1 for p in scale_pairs 
                                    if (p['scales'][0] == s and p['direction'] == 1) or 
                                    (p['scales'][1] == s and p['direction'] == -1))
                    
                    reverse_count = sum(1 for p in scale_pairs 
                                     if (p['scales'][0] == s and p['direction'] == -1) or 
                                     (p['scales'][1] == s and p['direction'] == 1))
                    
                    balanced_count = sum(1 for p in scale_pairs if p['direction'] == 0)
                    
                    total = forward_count + reverse_count + balanced_count
                    scale_ratio = (forward_count - reverse_count) / total if total > 0 else 0
                    
                    # For calculating average net TE, we need to consider the direction relative to this scale
                    net_tes = []
                    for p in scale_pairs:
                        if p['scales'][0] == s:
                            net_tes.append(p['net_te'])
                        else:  # s is the second scale, reverse the sign
                            net_tes.append(-p['net_te'])
                    
                    avg_net_te = np.mean(net_tes) if net_tes else 0
                    
                    scale_data[s] = {
                        'forward_count': forward_count,
                        'reverse_count': reverse_count,
                        'balanced_count': balanced_count,
                        'scale_ratio': scale_ratio,
                        'avg_net_te': avg_net_te
                    }
            
            # Plot directional ratio by scale
            plt.subplot(2, 1, 1)
            scale_indices = sorted(scale_data.keys())
            ratios = [scale_data[s]['scale_ratio'] for s in scale_indices]
            
            colors = ['blue' if ratio > 0 else 'red' for ratio in ratios]
            plt.bar(scale_indices, ratios, color=colors)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.title(f'Directional Ratio by Scale ({name} data)')
            plt.xlabel('Scale')
            plt.ylabel('Directional Ratio (-1 to 1)')
            plt.xticks(scale_indices, [f'Scale {i+1}' for i in scale_indices])
            
            # Plot net TE by scale
            plt.subplot(2, 1, 2)
            net_tes = [scale_data[s]['avg_net_te'] for s in scale_indices]
            
            colors = ['blue' if te > 0 else 'red' for te in net_tes]
            plt.bar(scale_indices, net_tes, color=colors)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.title(f'Average Net Transfer Entropy by Scale ({name} data)')
            plt.xlabel('Scale')
            plt.ylabel('Net Transfer Entropy')
            plt.xticks(scale_indices, [f'Scale {i+1}' for i in scale_indices])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name.lower()}_scale_specific.png"))
            plt.close()
    
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        traceback.print_exc()


def compare_results(wmap_results, planck_results, output_dir):
    """Compare directional transfer entropy results between WMAP and Planck data."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract key metrics
        wmap_forward_te = wmap_results['actual_results']['overall_forward_te']
        wmap_reverse_te = wmap_results['actual_results']['overall_reverse_te']
        wmap_net_te = wmap_results['actual_results']['overall_net_te']
        wmap_ratio = wmap_results['actual_results']['overall_directional_ratio']
        
        planck_forward_te = planck_results['actual_results']['overall_forward_te']
        planck_reverse_te = planck_results['actual_results']['overall_reverse_te']
        planck_net_te = planck_results['actual_results']['overall_net_te']
        planck_ratio = planck_results['actual_results']['overall_directional_ratio']
        
        # Calculate differences
        forward_diff = abs(wmap_forward_te - planck_forward_te)
        reverse_diff = abs(wmap_reverse_te - planck_reverse_te)
        net_diff = abs(wmap_net_te - planck_net_te)
        ratio_diff = abs(wmap_ratio - planck_ratio)
        
        # Save comparison to file
        with open(os.path.join(output_dir, 'directional_te_comparison.txt'), 'w') as f:
            f.write('Directional Transfer Entropy Comparison: WMAP vs Planck\n')
            f.write('=' * 60 + '\n\n')
            
            f.write('WMAP Results:\n')
            f.write(f'  Forward TE: {wmap_forward_te:.6f}\n')
            f.write(f'  Reverse TE: {wmap_reverse_te:.6f}\n')
            f.write(f'  Net TE: {wmap_net_te:.6f}\n')
            f.write(f'  Directional Ratio: {wmap_ratio:.6f}\n')
            f.write(f'  P-value (directional): {wmap_results["mc_results"]["p_directional"]:.6f}\n')
            f.write(f'  Z-score (directional): {wmap_results["mc_results"]["z_directional"]:.6f}\n')
            f.write(f'  Dominant Constant: {wmap_results["actual_results"]["dominant_constant"]}\n\n')
            
            f.write('Planck Results:\n')
            f.write(f'  Forward TE: {planck_forward_te:.6f}\n')
            f.write(f'  Reverse TE: {planck_reverse_te:.6f}\n')
            f.write(f'  Net TE: {planck_net_te:.6f}\n')
            f.write(f'  Directional Ratio: {planck_ratio:.6f}\n')
            f.write(f'  P-value (directional): {planck_results["mc_results"]["p_directional"]:.6f}\n')
            f.write(f'  Z-score (directional): {planck_results["mc_results"]["z_directional"]:.6f}\n')
            f.write(f'  Dominant Constant: {planck_results["actual_results"]["dominant_constant"]}\n\n')
            
            f.write('Differences:\n')
            f.write(f'  Forward TE: {forward_diff:.6f}\n')
            f.write(f'  Reverse TE: {reverse_diff:.6f}\n')
            f.write(f'  Net TE: {net_diff:.6f}\n')
            f.write(f'  Directional Ratio: {ratio_diff:.6f}\n\n')
            
            # Compare dominant constants
            wmap_constant_results = wmap_results['actual_results']['constant_results']
            planck_constant_results = planck_results['actual_results']['constant_results']
            
            f.write('Constant-Specific Comparison:\n')
            for const in sorted(wmap_constant_results.keys()):
                wmap_const = wmap_constant_results[const]
                planck_const = planck_constant_results[const]
                
                f.write(f'  {const.upper()}:\n')
                f.write(f'    Directional Ratio - WMAP: {wmap_const["directional_ratio"]:.6f}, Planck: {planck_const["directional_ratio"]:.6f}\n')
                f.write(f'    Net TE - WMAP: {wmap_const["avg_net_te"]:.6f}, Planck: {planck_const["avg_net_te"]:.6f}\n\n')
        
        # Create comparison visualizations
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Bar chart of directional ratios
        plt.subplot(2, 2, 1)
        datasets = ['WMAP', 'Planck']
        ratios = [wmap_ratio, planck_ratio]
        
        colors = ['blue' if r > 0 else 'red' for r in ratios]
        plt.bar(datasets, ratios, color=colors)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('Overall Directional Ratio')
        plt.ylabel('Directional Ratio (-1 to 1)')
        
        # Add significance annotations
        for i, dataset in enumerate(datasets):
            p_value = wmap_results["mc_results"]["p_directional"] if i == 0 else planck_results["mc_results"]["p_directional"]
            sign = '*' if p_value < 0.05 else 'ns'
            plt.text(i, ratios[i] + 0.05 * np.sign(ratios[i]), 
                   f'p={p_value:.4f}\n{sign}', 
                   ha='center', va='center', fontweight='bold')
        
        # Plot 2: Bar chart of forward and reverse TE
        plt.subplot(2, 2, 2)
        width = 0.35
        x = np.arange(len(datasets))
        
        forward_values = [wmap_forward_te, planck_forward_te]
        reverse_values = [wmap_reverse_te, planck_reverse_te]
        
        plt.bar(x - width/2, forward_values, width, label='Forward TE')
        plt.bar(x + width/2, reverse_values, width, label='Reverse TE')
        
        plt.xlabel('Dataset')
        plt.ylabel('Transfer Entropy')
        plt.title('Forward vs Reverse Transfer Entropy')
        plt.xticks(x, datasets)
        plt.legend()
        
        # Plot 3: Directional ratio by constant, comparison
        plt.subplot(2, 2, 3)
        
        constants = sorted(wmap_constant_results.keys())
        wmap_const_ratios = [wmap_constant_results[c]['directional_ratio'] for c in constants]
        planck_const_ratios = [planck_constant_results[c]['directional_ratio'] for c in constants]
        
        x = np.arange(len(constants))
        width = 0.35
        
        plt.bar(x - width/2, wmap_const_ratios, width, label='WMAP')
        plt.bar(x + width/2, planck_const_ratios, width, label='Planck')
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Mathematical Constant')
        plt.ylabel('Directional Ratio')
        plt.title('Directional Ratio by Constant')
        plt.xticks(x, constants, rotation=45)
        plt.legend()
        
        # Plot 4: Summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        wmap_direction = 'Forward' if wmap_ratio > 0 else 'Reverse' if wmap_ratio < 0 else 'Balanced'
        planck_direction = 'Forward' if planck_ratio > 0 else 'Reverse' if planck_ratio < 0 else 'Balanced'
        
        summary_text = f"""
        Directional Transfer Entropy Analysis
        
        WMAP:
        - Dominant Direction: {wmap_direction}
        - Directional Ratio: {wmap_ratio:.4f}
        - Significance: p={wmap_results["mc_results"]["p_directional"]:.4f}
        - Dominant Constant: {wmap_results["actual_results"]["dominant_constant"]}
        
        Planck:
        - Dominant Direction: {planck_direction}
        - Directional Ratio: {planck_ratio:.4f}
        - Significance: p={planck_results["mc_results"]["p_directional"]:.4f}
        - Dominant Constant: {planck_results["actual_results"]["dominant_constant"]}
        
        Consistency: {'Same' if (wmap_ratio > 0) == (planck_ratio > 0) else 'Different'} directional tendency
        """
        
        plt.text(0.05, 0.95, summary_text, fontsize=12, 
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'directional_te_comparison.png'))
        plt.close()
        
        print(f"Comparison results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error comparing results: {str(e)}")
        traceback.print_exc()

def main():
    """Main function to run the Directional Transfer Entropy Test."""
    parser = argparse.ArgumentParser(description='Run Directional Transfer Entropy Test on CMB data')
    
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--n-simulations', type=int, default=1000, 
                        help='Number of Monte Carlo simulations for statistical validation')
    parser.add_argument('--scales', type=int, default=5, help='Number of scales to analyze')
    parser.add_argument('--bins', type=int, default=10, help='Number of bins for discretization')
    parser.add_argument('--delay', type=int, default=1, help='Time delay for information transfer')
    parser.add_argument('--max-points', type=int, default=500, 
                        help='Maximum number of points to use for transfer entropy calculation')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results (default: results/directional_te_TIMESTAMP)')
    parser.add_argument('--visualize', action='store_true', help='Generate additional visualizations')
    
    args = parser.parse_args()
    
    # Print start time
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Starting Directional Transfer Entropy Test at {timestamp}")
    
    # Path to data files
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    wmap_file = os.path.join(base_dir, 'data/wmap/wmap_tt_spectrum_9yr_v5.txt')
    planck_file = os.path.join(base_dir, 'data/planck/planck_tt_spectrum_2018.txt')
    
    # Check if files exist
    if not args.planck_only and not os.path.exists(wmap_file):
        print(f"Error: WMAP power spectrum file not found: {wmap_file}")
        return 1
    
    if not args.wmap_only and not os.path.exists(planck_file):
        print(f"Error: Planck power spectrum file not found: {planck_file}")
        print("Please make sure the Planck data is available.")
        return 1
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, 'results', f"directional_te_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Print parameters
    print("Parameters:")
    print(f"  - Number of simulations: {args.n_simulations}")
    print(f"  - Number of scales: {args.scales}")
    print(f"  - Number of bins: {args.bins}")
    print(f"  - Delay: {args.delay}")
    print(f"  - Max points: {args.max_points}")
    print(f"  - Output directory: {output_dir}")
    
    # Define mathematical constants
    constants = {
        "phi": 1.618033988749895,  # Golden ratio
        "e": 2.718281828459045,    # Euler's number
        "pi": 3.141592653589793,   # Pi
        "sqrt2": 1.4142135623730951,  # Square root of 2
        "sqrt3": 1.7320508075688772,  # Square root of 3
        "ln2": 0.6931471805599453   # Natural logarithm of 2
    }
    
    # Initialize results
    wmap_results = None
    planck_results = None
    
    # Process WMAP data
    if not args.planck_only:
        print("\n==================================================")
        print("Processing WMAP data")
        print("==================================================")
        
        # Load and preprocess WMAP data
        wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(wmap_file)
        if wmap_ell is None:
            print("Error loading WMAP power spectrum.")
            return 1
        
        print(f"Loaded WMAP power spectrum with {len(wmap_ell)} multipoles")
        
        wmap_processed = preprocess_data(wmap_power, normalize=True)
        
        # Run directional TE test on WMAP data
        print("Running directional transfer entropy test on WMAP data...")
        wmap_start_time = time.time()
        
        wmap_results = run_directional_te_test(
            wmap_processed, 
            os.path.join(output_dir, 'wmap'),
            'WMAP',
            scales=args.scales,
            constants=constants,
            n_simulations=args.n_simulations,
            bins=args.bins,
            delay=args.delay,
            max_points=args.max_points
        )
        
        wmap_elapsed = time.time() - wmap_start_time
        print(f"WMAP analysis completed in {wmap_elapsed:.2f} seconds")
    
    # Process Planck data
    if not args.wmap_only:
        print("\n==================================================")
        print("Processing Planck data")
        print("==================================================")
        
        # Load and preprocess Planck data
        planck_ell, planck_power, planck_error = load_planck_power_spectrum(planck_file)
        if planck_ell is None:
            print("Error loading Planck power spectrum.")
            return 1
        
        print(f"Loaded Planck power spectrum with {len(planck_ell)} multipoles")
        
        planck_processed = preprocess_data(planck_power, normalize=True)
        
        # Run directional TE test on Planck data
        print("Running directional transfer entropy test on Planck data...")
        planck_start_time = time.time()
        
        planck_results = run_directional_te_test(
            planck_processed,
            os.path.join(output_dir, 'planck'),
            'Planck',
            scales=args.scales,
            constants=constants,
            n_simulations=args.n_simulations,
            bins=args.bins,
            delay=args.delay,
            max_points=args.max_points
        )
        
        planck_elapsed = time.time() - planck_start_time
        print(f"Planck analysis completed in {planck_elapsed:.2f} seconds")
    
    # Compare results if both datasets were processed
    if wmap_results and planck_results:
        print("\n==================================================")
        print("Comparing WMAP and Planck results")
        print("==================================================")
        
        compare_results(wmap_results, planck_results, os.path.join(output_dir, 'comparison'))
    
    # Create summary file
    summary_file = os.path.join(output_dir, 'directional_te_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Directional Transfer Entropy Test Results\n")
        f.write("=======================================\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters: scales={args.scales}, bins={args.bins}, delay={args.delay}, "
              f"simulations={args.n_simulations}\n\n")
        
        if wmap_results:
            wmap_ratio = wmap_results['actual_results']['overall_directional_ratio']
            wmap_net_te = wmap_results['actual_results']['overall_net_te']
            wmap_p = wmap_results['mc_results']['p_directional']
            wmap_z = wmap_results['mc_results']['z_directional']
            wmap_dir = 'Forward' if wmap_ratio > 0 else 'Reverse' if wmap_ratio < 0 else 'Balanced'
            
            f.write("WMAP Results:\n")
            f.write(f"  - Dominant Direction: {wmap_dir}\n")
            f.write(f"  - Directional Ratio: {wmap_ratio:.6f}\n")
            f.write(f"  - Net Transfer Entropy: {wmap_net_te:.6f}\n")
            f.write(f"  - Statistical Significance: p={wmap_p:.6f}, z={wmap_z:.4f}\n")
            f.write(f"  - Dominant Constant: {wmap_results['actual_results']['dominant_constant']}\n\n")
        
        if planck_results:
            planck_ratio = planck_results['actual_results']['overall_directional_ratio']
            planck_net_te = planck_results['actual_results']['overall_net_te']
            planck_p = planck_results['mc_results']['p_directional']
            planck_z = planck_results['mc_results']['z_directional']
            planck_dir = 'Forward' if planck_ratio > 0 else 'Reverse' if planck_ratio < 0 else 'Balanced'
            
            f.write("Planck Results:\n")
            f.write(f"  - Dominant Direction: {planck_dir}\n")
            f.write(f"  - Directional Ratio: {planck_ratio:.6f}\n")
            f.write(f"  - Net Transfer Entropy: {planck_net_te:.6f}\n")
            f.write(f"  - Statistical Significance: p={planck_p:.6f}, z={planck_z:.4f}\n")
            f.write(f"  - Dominant Constant: {planck_results['actual_results']['dominant_constant']}\n\n")
        
        if wmap_results and planck_results:
            consistency = "Same" if (wmap_ratio > 0) == (planck_ratio > 0) else "Different"
            f.write(f"Directional Consistency: {consistency} dominant direction across datasets\n")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Summary available at: {summary_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
