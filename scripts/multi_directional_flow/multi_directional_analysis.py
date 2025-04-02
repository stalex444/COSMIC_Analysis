#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

"""
Multi-Scale Directional Flow Analysis

This script implements a specialized test to analyze multi-directional information flow
in CMB data, with particular focus on identifying scale-specific directional patterns
and transitions between information flow regimes.
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
import gc
import psutil
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Set the path to COSMIC_Analysis directory
cosmic_dir = os.path.dirname(parent_dir)

def convert_to_json_serializable(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {convert_to_json_serializable(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(i) for i in obj)
    else:
        return obj

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
    """
    Preprocess data for analysis.
    
    Args:
        data (numpy.ndarray): Input data
        smooth (bool): Whether to smooth the data
        smooth_window (int): Window size for smoothing
        normalize (bool): Whether to normalize the data
        detrend (bool): Whether to remove linear trend
        
    Returns:
        numpy.ndarray: Preprocessed data
    """
    # Make a copy to avoid modifying the original data
    result = np.array(data, dtype=float)
    
    # Replace NaN and inf values with interpolated values
    mask = np.isnan(result) | np.isinf(result)
    if np.any(mask):
        valid_indices = np.where(~mask)[0]
        if len(valid_indices) < 2:
            # If less than 2 valid points, replace with zeros
            result[mask] = 0
        else:
            interp_indices = np.where(mask)[0]
            result[interp_indices] = np.interp(interp_indices, valid_indices, result[valid_indices])
    
    # Detrend if requested
    if detrend:
        result = signal.detrend(result)
    
    # Smooth if requested
    if smooth and len(result) > smooth_window:
        result = np.convolve(result, np.ones(smooth_window)/smooth_window, mode='valid')
    
    # Normalize if requested
    if normalize and len(result) > 0:
        if np.max(result) != np.min(result):
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
        else:
            # If all values are the same, set to 0.5 (neutral value)
            result[:] = 0.5
    
    return result


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
    # Simple implementation for edge cases
    if len(source) <= delay + 2 or len(target) <= delay + 2:
        return 0.0
    
    # Ensure arrays are numpy arrays and have the same length
    source = np.array(source, dtype=float)
    target = np.array(target, dtype=float)
    min_len = min(len(source), len(target))
    source = source[:min_len]
    target = target[:min_len]
    
    # Trim data if needed
    if max_points and len(source) > max_points:
        step = max(1, len(source) // max_points)
        source = source[::step]
        target = target[::step]
    
    # Create time-delayed versions
    x_t = source[:-delay]        # X_t (past)
    y_t = target[:-delay]        # Y_t (past)
    y_t_plus = target[delay:]    # Y_t+1 (future)
    
    # Use simple binning with quantiles for discretization
    n_bins = min(bins, max(2, len(x_t) // 10))  # Adjust number of bins based on data size
    
    # Create more robust binning based on percentiles
    x_bins = np.percentile(x_t, np.linspace(0, 100, n_bins+1))
    y_bins = np.percentile(y_t, np.linspace(0, 100, n_bins+1))
    y_future_bins = np.percentile(y_t_plus, np.linspace(0, 100, n_bins+1))
    
    # Handle case where percentiles yield identical values
    if len(np.unique(x_bins)) < 2:
        x_bins = np.array([np.min(x_t) - 0.1, np.max(x_t) + 0.1])
    if len(np.unique(y_bins)) < 2:
        y_bins = np.array([np.min(y_t) - 0.1, np.max(y_t) + 0.1])
    if len(np.unique(y_future_bins)) < 2:
        y_future_bins = np.array([np.min(y_t_plus) - 0.1, np.max(y_t_plus) + 0.1])
    
    # Simple digitize that's guaranteed to be within bounds
    x_discrete = np.zeros(len(x_t), dtype=int)
    y_discrete = np.zeros(len(y_t), dtype=int)
    y_future_discrete = np.zeros(len(y_t_plus), dtype=int)
    
    for i in range(len(x_t)):
        for b in range(len(x_bins)-1):
            if x_t[i] >= x_bins[b] and (b == len(x_bins)-2 or x_t[i] < x_bins[b+1]):
                x_discrete[i] = b
                break
                
        for b in range(len(y_bins)-1):
            if y_t[i] >= y_bins[b] and (b == len(y_bins)-2 or y_t[i] < y_bins[b+1]):
                y_discrete[i] = b
                break
                
        for b in range(len(y_future_bins)-1):
            if y_t_plus[i] >= y_future_bins[b] and (b == len(y_future_bins)-2 or y_t_plus[i] < y_future_bins[b+1]):
                y_future_discrete[i] = b
                break
    
    # Count configurations (safely within the bin ranges)
    n_bins_actual = n_bins if n_bins > 1 else 1
    
    # Create frequency tables
    xy_count = np.zeros((n_bins_actual, n_bins_actual))
    y_future_y_count = np.zeros((n_bins_actual, n_bins_actual))
    xy_y_future_count = np.zeros((n_bins_actual, n_bins_actual, n_bins_actual))
    
    # Safely fill frequency tables
    for i in range(len(x_discrete)):
        x_idx = min(x_discrete[i], n_bins_actual-1)
        y_idx = min(y_discrete[i], n_bins_actual-1)
        y_future_idx = min(y_future_discrete[i], n_bins_actual-1)
        
        xy_count[x_idx, y_idx] += 1
        y_future_y_count[y_future_idx, y_idx] += 1
        xy_y_future_count[x_idx, y_idx, y_future_idx] += 1
    
    # Calculate probabilities
    total_samples = len(x_discrete)
    p_xy = xy_count / total_samples
    p_y_future_y = y_future_y_count / total_samples
    p_xy_y_future = xy_y_future_count / total_samples
    p_y = np.sum(p_xy, axis=0)
    
    # Calculate transfer entropy
    te = 0
    for x in range(n_bins_actual):
        for y in range(n_bins_actual):
            if p_xy[x, y] > 0 and p_y[y] > 0:
                for y_future in range(n_bins_actual):
                    if p_xy_y_future[x, y, y_future] > 0 and p_y_future_y[y_future, y] > 0:
                        te += p_xy_y_future[x, y, y_future] * np.log2(
                            p_xy_y_future[x, y, y_future] * p_y[y] / 
                            (p_xy[x, y] * p_y_future_y[y_future, y])
                        )
    
    return max(0, te)  # Ensure non-negative TE


def analyze_multi_scale_flow(data, num_scales=10, bins=10, delay=1, max_points=500):
    """
    Analyze information flow across multiple scales with fine granularity.
    
    Args:
        data: Input data array
        num_scales: Number of scales to divide the data into (higher for finer granularity)
        bins: Number of bins for discretization
        delay: Time delay for information transfer
        max_points: Maximum number of points for TE calculation
        
    Returns:
        dict: Results of multi-scale directional analysis
    """
    print(f"Analyzing information flow across {num_scales} scales...")
    
    results = {
        'scale_patterns': {},
        'forward_count': 0,
        'reverse_count': 0,
        'balanced_count': 0,
        'overall_ratio': 0.0,
        'flow_matrix': np.zeros((num_scales, num_scales)),
        'net_flow_by_scale': np.zeros(num_scales)
    }
    
    # Create scale data by resampling to different lengths
    scale_data = {}
    original_length = len(data)
    
    # Use a logarithmic scale for better coverage of different scales
    scale_points = np.logspace(np.log10(10), np.log10(original_length), num_scales, dtype=int)
    scale_points = np.unique(scale_points)  # Remove duplicates
    num_scales = len(scale_points)  # Update num_scales after removing duplicates
    
    # Ensure we have at least the minimum required data points for each scale
    min_scale_points = max(10, delay + 2)  # Need at least delay+2 points for TE calculation
    scale_points = np.maximum(scale_points, min_scale_points)
    
    # Create resampled data for each scale
    for i, num_points in enumerate(scale_points):
        # Resample data to the specified number of points
        indices = np.linspace(0, original_length-1, num_points, dtype=int)
        scale_data[i] = data[indices]
        
        # Apply additional preprocessing to ensure quality
        scale_data[i] = preprocess_data(scale_data[i], smooth=False, normalize=True)
    
    # Initialize flow matrix
    flow_matrix = np.zeros((num_scales, num_scales))
    
    # Calculate directional transfer entropy for all scale pairs
    pairs_count = num_scales * (num_scales - 1) // 2
    print(f"Calculating directional transfer entropy for {pairs_count} scale pairs...")
    
    pair_count = 0
    
    threshold = 1e-6  # Threshold for determining significant TE differences
    
    # For each pair of scales
    for i in range(num_scales):
        for j in range(i+1, num_scales):
            # Skip if either scale has too few points
            if len(scale_data[i]) <= delay + 1 or len(scale_data[j]) <= delay + 1:
                print(f"Skipping scales {i} and {j} - insufficient data points")
                flow_matrix[i, j] = 0
                flow_matrix[j, i] = 0
                continue
                
            # Calculate forward TE (from scale i to scale j)
            te_forward = calculate_transfer_entropy(
                scale_data[i], scale_data[j], bins=bins, delay=delay, max_points=max_points
            )
            
            # Calculate reverse TE (from scale j to scale i)
            te_reverse = calculate_transfer_entropy(
                scale_data[j], scale_data[i], bins=bins, delay=delay, max_points=max_points
            )
            
            # Store TE values in the flow matrix
            flow_matrix[i, j] = te_forward
            flow_matrix[j, i] = te_reverse
            
            # Count which direction has stronger TE
            diff = te_forward - te_reverse
            if abs(diff) < threshold:
                results['balanced_count'] += 1
            elif diff > 0:
                results['forward_count'] += 1
            else:
                results['reverse_count'] += 1
            
            pair_count += 1
    
    # Store the complete flow matrix
    results['flow_matrix'] = flow_matrix
    
    # Calculate net flow for each scale (outgoing - incoming)
    for i in range(num_scales):
        outgoing = np.sum(flow_matrix[i, :])
        incoming = np.sum(flow_matrix[:, i])
        results['net_flow_by_scale'][i] = outgoing - incoming
    
    # Analyze scale-specific patterns
    for i in range(num_scales):
        outgoing = np.sum(flow_matrix[i, :])
        incoming = np.sum(flow_matrix[:, i])
        
        if outgoing == 0 and incoming == 0:
            flow_type = 'Inactive'
            net_ratio = 0
        else:
            net_flow = outgoing - incoming
            if abs(net_flow) < threshold:
                flow_type = 'Balanced'
                net_ratio = 0
            elif net_flow > 0:
                flow_type = 'Source'
                net_ratio = net_flow / (outgoing + incoming) if (outgoing + incoming) > 0 else 0
            else:
                flow_type = 'Sink'
                net_ratio = net_flow / (outgoing + incoming) if (outgoing + incoming) > 0 else 0
        
        results['scale_patterns'][i] = {
            'outgoing': outgoing,
            'incoming': incoming,
            'net_flow': outgoing - incoming,
            'flow_type': flow_type,
            'net_ratio': net_ratio
        }
    
    # Calculate overall directional ratio
    total_forward = np.sum(np.triu(flow_matrix, k=1))
    total_reverse = np.sum(np.tril(flow_matrix, k=-1))
    
    if total_forward + total_reverse > 0:
        results['overall_ratio'] = (total_forward - total_reverse) / (total_forward + total_reverse)
    else:
        results['overall_ratio'] = 0
    
    # Analyze transitions between adjacent scales
    results['transitions'] = []
    for i in range(num_scales - 1):
        if (flow_matrix[i, i+1] > threshold or flow_matrix[i+1, i] > threshold) and \
           abs(flow_matrix[i, i+1] - flow_matrix[i+1, i]) > threshold:
            
            # Determine direction of stronger flow
            if flow_matrix[i, i+1] > flow_matrix[i+1, i]:
                direction = 'forward'
                strength = flow_matrix[i, i+1] / (flow_matrix[i, i+1] + flow_matrix[i+1, i]) \
                           if (flow_matrix[i, i+1] + flow_matrix[i+1, i]) > 0 else 0
            else:
                direction = 'reverse'
                strength = flow_matrix[i+1, i] / (flow_matrix[i, i+1] + flow_matrix[i+1, i]) \
                           if (flow_matrix[i, i+1] + flow_matrix[i+1, i]) > 0 else 0
            
            results['transitions'].append({
                'from_scale': i,
                'to_scale': i+1,
                'direction': direction,
                'strength': strength
            })
    
    return results


def generate_surrogate_data(data, num_surrogates=100):
    """Generate surrogate datasets for statistical validation."""
    print(f"Generating {num_surrogates} surrogate datasets...")
    surrogate_datasets = []
    
    for i in range(num_surrogates):
        if i % 10 == 0:
            print(f"  Generating surrogate {i+1}/{num_surrogates}...")
        
        # Create a permuted copy
        surrogate = np.random.permutation(data.copy())
        surrogate_datasets.append(surrogate)
    
    return surrogate_datasets


def calculate_scale_pattern_significance(actual_results, surrogate_results):
    """
    Calculate the statistical significance of scale patterns compared to surrogate data.
    
    Args:
        actual_results: Results from the actual data
        surrogate_results: List of results from surrogate datasets
        
    Returns:
        dict: Statistical significance of scale patterns
    """
    print("Calculating statistical significance of scale patterns...")
    
    # Extract actual scale patterns
    actual_patterns = actual_results['scale_patterns']
    
    # Initialize statistics storage
    scale_stats = {}
    
    # For each scale, calculate statistical significance
    for scale_idx in actual_patterns:
        # Extract actual metrics
        actual_outgoing = actual_patterns[scale_idx]['outgoing']
        actual_incoming = actual_patterns[scale_idx]['incoming']
        actual_ratio = actual_patterns[scale_idx]['net_ratio']
        
        # Collect surrogate metrics
        surrogate_outgoing = []
        surrogate_incoming = []
        surrogate_ratios = []
        
        for surr_result in surrogate_results:
            if scale_idx in surr_result['scale_patterns']:
                surr_pattern = surr_result['scale_patterns'][scale_idx]
                surrogate_outgoing.append(surr_pattern['outgoing'])
                surrogate_incoming.append(surr_pattern['incoming'])
                surrogate_ratios.append(surr_pattern['net_ratio'])
        
        # Calculate statistics if we have enough surrogate data
        if surrogate_ratios:
            # Outgoing TE
            out_mean = np.mean(surrogate_outgoing)
            out_std = np.std(surrogate_outgoing)
            out_z = (actual_outgoing - out_mean) / out_std if out_std > 0 else 0
            out_p = np.mean([1 if s >= actual_outgoing else 0 for s in surrogate_outgoing])
            
            # Incoming TE
            in_mean = np.mean(surrogate_incoming)
            in_std = np.std(surrogate_incoming)
            in_z = (actual_incoming - in_mean) / in_std if in_std > 0 else 0
            in_p = np.mean([1 if s >= actual_incoming else 0 for s in surrogate_incoming])
            
            # Net ratio
            ratio_mean = np.mean(surrogate_ratios)
            ratio_std = np.std(surrogate_ratios)
            ratio_z = (actual_ratio - ratio_mean) / ratio_std if ratio_std > 0 else 0
            ratio_p = np.mean([1 if abs(s) >= abs(actual_ratio) else 0 for s in surrogate_ratios])
            
            scale_stats[scale_idx] = {
                'outgoing_z': out_z,
                'outgoing_p': out_p,
                'incoming_z': in_z,
                'incoming_p': in_p,
                'ratio_z': ratio_z,
                'ratio_p': ratio_p,
                'significant': ratio_p < 0.05
            }
        else:
            scale_stats[scale_idx] = {
                'outgoing_z': 0,
                'outgoing_p': 1.0,
                'incoming_z': 0,
                'incoming_p': 1.0,
                'ratio_z': 0,
                'ratio_p': 1.0,
                'significant': False
            }
    
    return scale_stats


def calculate_transition_significance(actual_results, surrogate_results):
    """
    Calculate the statistical significance of flow transitions compared to surrogate data.
    
    Args:
        actual_results: Results from the actual data
        surrogate_results: List of results from surrogate datasets
        
    Returns:
        dict: Statistical significance of flow transitions
    """
    print("Calculating statistical significance of flow transitions...")
    
    # Extract actual transitions
    actual_transitions = actual_results['transitions']
    
    # Initialize statistics storage
    transition_stats = {}
    
    # For each actual transition, calculate statistical significance
    for i, transition in enumerate(actual_transitions):
        actual_strength = transition['strength']
        actual_scale = transition['to_scale']
        
        # Collect similar transitions from surrogate data
        surrogate_strengths = []
        
        for surr_result in surrogate_results:
            surr_transitions = surr_result['transitions']
            
            # Look for transitions at similar scales
            for surr_transition in surr_transitions:
                if abs(surr_transition['to_scale'] - actual_scale) / actual_scale < 0.2:
                    surrogate_strengths.append(surr_transition['strength'])
        
        # Calculate statistics if we have enough surrogate data
        if surrogate_strengths:
            # Transition strength
            strength_mean = np.mean(surrogate_strengths)
            strength_std = np.std(surrogate_strengths)
            strength_z = (actual_strength - strength_mean) / strength_std if strength_std > 0 else 0
            strength_p = np.mean([1 if s >= actual_strength else 0 for s in surrogate_strengths])
            
            transition_stats[i] = {
                'strength_z': strength_z,
                'strength_p': strength_p,
                'significant': strength_p < 0.05
            }
        else:
            transition_stats[i] = {
                'strength_z': 0,
                'strength_p': 1.0,
                'significant': False
            }
    
    return transition_stats


def create_flow_visualizations(results, output_dir, name):
    """Create visualizations of the multi-scale flow analysis results."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for plotting
        flow_matrix = results['actual_results']['flow_matrix']
        scale_boundaries = np.array(list(results['actual_results']['scale_patterns'].keys()))
        scale_patterns = results['actual_results']['scale_patterns']
        transitions = results['actual_results']['transitions']
        
        scale_stats = results['scale_stats']
        sig_source_scales = results['sig_source_scales']
        sig_sink_scales = results['sig_sink_scales']
        sig_balanced_scales = results['sig_balanced_scales']
        sig_transitions = results['sig_transitions']
        
        # Define size based on number of scales
        num_scales = flow_matrix.shape[0]
        fig_size = max(8, min(20, num_scales * 0.8))
        
        # Plot 1: Flow Matrix Heatmap
        plt.figure(figsize=(fig_size, fig_size))
        
        # Create custom colormap (blue-white-red)
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
        cmap = LinearSegmentedColormap.from_list('bwr', colors, N=101)
        
        # Find max absolute value for symmetric scale
        max_val = max(np.max(np.abs(flow_matrix)), 0.0001)
        
        # Plot heatmap
        im = plt.imshow(flow_matrix, cmap=cmap, vmin=-max_val, vmax=max_val)
        plt.colorbar(im, label='Transfer Entropy')
        
        plt.title(f'Information Flow Matrix ({name} Data)')
        plt.xlabel('To Scale')
        plt.ylabel('From Scale')
        
        # Add scale labels
        scale_labels = [f'Scale {i+1}' for i in range(num_scales)]
        plt.xticks(range(num_scales), scale_labels, rotation=45)
        plt.yticks(range(num_scales), scale_labels)
        
        # Add grid
        plt.grid(False)
        
        # Save Plot 1
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'flow_matrix.png'))
        plt.close()
        
        # Plot 2: Scale-Specific Flow Patterns
        plt.figure(figsize=(fig_size, fig_size//2))
        
        # Extract scale pattern data
        scale_indices = sorted(scale_patterns.keys())
        net_ratios = [scale_patterns[i]['net_ratio'] for i in scale_indices]
        
        # Determine bar colors based on flow type
        flow_types = [scale_patterns[i]['flow_type'] for i in scale_indices]
        colors = ['blue' if ft == 'Source' else 'red' if ft == 'Sink' else 'gray' for ft in flow_types]
        
        # Create bar chart with custom colors
        bars = plt.bar(scale_indices, net_ratios, color=colors, alpha=0.7)
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Mark significant scales with stars
        for i in scale_indices:
            if i in sig_source_scales or i in sig_sink_scales or i in sig_balanced_scales:
                plt.scatter(i, net_ratios[i], marker='*', s=150, color='gold', 
                            edgecolor='black', zorder=10)
        
        # Add labels and title
        plt.xlabel('Scale Index')
        plt.ylabel('Net Directional Ratio')
        plt.title(f'Scale-Specific Flow Patterns ({name} Data)')
        
        # Add legend
        source_patch = plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.7)
        sink_patch = plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.7)
        balanced_patch = plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.7)
        star = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                          markersize=15, markeredgecolor='black')
        
        plt.legend([source_patch, sink_patch, balanced_patch, star], 
                 ['Source (Outgoing)', 'Sink (Incoming)', 'Balanced', 'Statistically Significant'], 
                 loc='best')
        
        # Save Plot 2
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scale_patterns.png'))
        plt.close()
        
        # Plot 3: Flow Regime Transitions
        if transitions:
            plt.figure(figsize=(12, 6))
            
            # Extract data for plotting
            transition_scales = [t['to_scale'] for t in transitions]
            transition_strengths = [t['strength'] for t in transitions]
            
            # Create stem plot
            markerline, stemlines, baseline = plt.stem(transition_scales, transition_strengths)
            plt.setp(markerline, marker='o', markerfacecolor='blue')
            plt.setp(stemlines, color='blue', alpha=0.7)
            
            # Highlight significant transitions
            for i in sig_transitions:
                plt.scatter(transition_scales[i], transition_strengths[i], 
                           s=150, marker='*', color='red', zorder=10)
            
            plt.title(f'Flow Regime Transitions ({name} Data)')
            plt.xlabel('Scale Index')
            plt.ylabel('Transition Strength')
            
            # Add transition labels
            for i, transition in enumerate(transitions):
                if i in sig_transitions:
                    plt.text(transition_scales[i], transition_strengths[i] * 1.1, 
                           f"{transition['from_scale']} → {transition['to_scale']}", 
                           ha='center', va='bottom', fontsize=8, rotation=45,
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
            
            # Save Plot 3
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'flow_transitions.png'))
            plt.close()
        
        # Plot 4: Integrated Flow Analysis Summary
        plt.figure(figsize=(14, 10))
        
        # Overall metrics
        plt.subplot(2, 2, 1)
        metrics = ['Forward\nPairs', 'Reverse\nPairs', 'Balanced\nPairs']
        values = [results['actual_results']['forward_count'], 
                results['actual_results']['reverse_count'], 
                results['actual_results']['balanced_count']]
        
        colors = ['blue', 'red', 'gray']
        
        plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title('Flow Direction Counts')
        plt.ylabel('Count')
        
        # Overall ratio visualization
        plt.subplot(2, 2, 2)
        overall_ratio = results['actual_results']['overall_ratio']
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        arrow_width = 0.03
        arrow_head_width = 0.2
        
        if overall_ratio > 0:
            plt.arrow(0, 0, 0.7, 0, width=arrow_width*abs(overall_ratio), 
                    head_width=arrow_head_width*abs(overall_ratio), 
                    head_length=0.1, fc='blue', ec='blue', alpha=0.7)
        else:
            plt.arrow(0, 0, -0.7, 0, width=arrow_width*abs(overall_ratio), 
                    head_width=arrow_head_width*abs(overall_ratio), 
                    head_length=0.1, fc='red', ec='red', alpha=0.7)
        
        plt.text(0, 0.1, f'Overall Ratio: {overall_ratio:.4f}', 
               ha='center', va='bottom', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        plt.xlim(-1, 1)
        plt.ylim(-0.5, 0.5)
        plt.title('Dominant Flow Direction')
        plt.xticks([])
        plt.yticks([])
        
        # Flow regime diagram
        plt.subplot(2, 1, 2)
        
        # Create a simple diagram showing scale groups
        group_starts = []
        group_ends = []
        group_types = []
        
        for i in range(num_scales):
            if results['actual_results']['scale_patterns'][i]['flow_type'] != 'Inactive':
                group_starts.append(i)
                group_ends.append(i)
                group_types.append(results['actual_results']['scale_patterns'][i]['flow_type'])
        
        y_pos = np.arange(len(group_starts))
        
        for i, (start, end, flow_type) in enumerate(zip(group_starts, group_ends, group_types)):
            color = 'blue' if flow_type == 'Source' else 'red' if flow_type == 'Sink' else 'gray'
            plt.barh(i, end - start + 1, left=start, height=0.5, color=color, alpha=0.7)
            plt.text(start + (end - start + 1)/2, i, flow_type, 
                   ha='center', va='center', color='white', fontweight='bold')
        
        # Add transition markers
        for i, transition in enumerate(transitions):
            if i in sig_transitions:
                plt.axvline(x=transition['to_scale'], color='black', linestyle='--')
        
        plt.yticks([])
        plt.title('Flow Regime Diagram')
        plt.xlabel('Scale Index')
        
        # Add overall summary as text
        plt.figtext(0.5, 0.01, f"""
        Multi-Scale Directional Flow Analysis Summary
        ({name} Data)
        
        Overall Metrics:
        - Forward Pairs: {results['actual_results']['forward_count']}
        - Reverse Pairs: {results['actual_results']['reverse_count']}
        - Balanced Pairs: {results['actual_results']['balanced_count']}
        - Overall Ratio: {results['actual_results']['overall_ratio']:.4f}
        
        Significant Findings:
        - Source Scales: {sig_source_scales if sig_source_scales else 'None'}
        - Sink Scales: {sig_sink_scales if sig_sink_scales else 'None'}
        - Balanced Scales: {sig_balanced_scales if sig_balanced_scales else 'None'}
        
        Interpretation:
        - {"Multi-directional flow patterns detected" if sig_source_scales and sig_sink_scales else 
           "Predominantly forward flow" if results['actual_results']['overall_ratio'] > 0.3 else
           "Predominantly reverse flow" if results['actual_results']['overall_ratio'] < -0.3 else
           "Balanced bidirectional flow"}
        - {"With clear regime transitions" if sig_transitions else "Without significant transitions"}
        """, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Save Plot 4
        plt.tight_layout(rect=[0, 0.07, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'flow_analysis_summary.png'))
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        traceback.print_exc()


def run_multi_scale_flow_analysis(data, output_dir, name, num_scales=10, 
                                num_surrogates=100, bins=10, delay=1, max_points=500):
    """
    Run multi-scale flow analysis on the provided data.
    
    Args:
        data: Input data array
        output_dir: Output directory for results
        name: Name of the dataset
        num_scales: Number of scales to analyze
        num_surrogates: Number of surrogate datasets for validation
        bins: Number of bins for discretization
        delay: Time delay for information transfer
        max_points: Maximum number of points for TE calculation
        
    Returns:
        dict: Analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running Multi-Scale Flow Analysis on {name} data...")
    start_time = time.time()
    
    # Run analysis on actual data
    print("Analyzing actual data...")
    actual_results = analyze_multi_scale_flow(data, num_scales, bins, delay, max_points)
    
    # Generate surrogate datasets
    surrogate_datasets = generate_surrogate_data(data, num_surrogates)
    
    # Analyze surrogate datasets
    print(f"Analyzing {num_surrogates} surrogate datasets...")
    surrogate_results = []
    
    for i, surrogate in enumerate(surrogate_datasets):
        print(f"Analyzing surrogate {i+1}/{num_surrogates}...")
        surr_results = analyze_multi_scale_flow(surrogate, num_scales, bins, delay, max_points)
        surrogate_results.append(surr_results)
    
    # Calculate scale pattern significance
    scale_stats = calculate_scale_pattern_significance(actual_results, surrogate_results)
    
    # Calculate transition significance
    transition_stats = calculate_transition_significance(actual_results, surrogate_results)
    
    # Summarize significant patterns
    sig_source_scales = [s for s in scale_stats if scale_stats[s]['significant'] and 
                       actual_results['scale_patterns'][s]['flow_type'] == 'Source']
    
    sig_sink_scales = [s for s in scale_stats if scale_stats[s]['significant'] and 
                      actual_results['scale_patterns'][s]['flow_type'] == 'Sink']
    
    sig_balanced_scales = [s for s in scale_stats if scale_stats[s]['significant'] and 
                         actual_results['scale_patterns'][s]['flow_type'] == 'Balanced']
    
    sig_transitions = [i for i in transition_stats if transition_stats[i]['significant']]
    
    # Create integrated results
    results = {
        'actual_results': actual_results,
        'scale_stats': scale_stats,
        'transition_stats': transition_stats,
        'sig_source_scales': sig_source_scales,
        'sig_sink_scales': sig_sink_scales,
        'sig_balanced_scales': sig_balanced_scales,
        'sig_transitions': sig_transitions
    }
    
    # Save results
    print("Saving results...")
    results_file = os.path.join(output_dir, f"{name.lower()}_flow_analysis.npz")
    np.savez_compressed(results_file, 
                      flow_matrix=actual_results['flow_matrix'],
                      scale_boundaries=np.array(list(actual_results['scale_patterns'].keys())),
                      forward_count=actual_results['forward_count'],
                      reverse_count=actual_results['reverse_count'],
                      balanced_count=actual_results['balanced_count'],
                      overall_ratio=actual_results['overall_ratio'])
    
    # Save detailed results as JSON
    detailed_results = {
        'transitions': actual_results['transitions'],
        'scale_patterns': actual_results['scale_patterns'],
        'scale_stats': scale_stats,
        'transition_stats': transition_stats,
        'sig_source_scales': sig_source_scales,
        'sig_sink_scales': sig_sink_scales,
        'sig_balanced_scales': sig_balanced_scales,
        'sig_transitions': sig_transitions,
        'overall_metrics': {
            'forward_count': actual_results['forward_count'],
            'reverse_count': actual_results['reverse_count'],
            'balanced_count': actual_results['balanced_count'],
            'overall_ratio': actual_results['overall_ratio']
        },
        'runtime_info': {
            'runtime': time.time() - start_time,
            'num_scales': num_scales,
            'num_surrogates': num_surrogates,
            'bins': bins,
            'delay': delay,
            'max_points': max_points
        }
    }
    
    # Convert NumPy types to Python native types for JSON serialization
    serializable_results = convert_to_json_serializable(detailed_results)
    
    with open(os.path.join(output_dir, f"{name.lower()}_detailed_results.json"), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Create summary file
    summary_file = os.path.join(output_dir, f"{name.lower()}_flow_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Multi-Scale Flow Analysis Results for {name} Data\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Overall Directional Metrics:\n")
        f.write(f"  - Forward Pairs: {actual_results['forward_count']}\n")
        f.write(f"  - Reverse Pairs: {actual_results['reverse_count']}\n")
        f.write(f"  - Balanced Pairs: {actual_results['balanced_count']}\n")
        f.write(f"  - Overall Ratio: {actual_results['overall_ratio']:.6f}\n\n")
        
        f.write("Significant Findings:\n")
        if sig_source_scales:
            f.write(f"  - Information Source Scales: {sig_source_scales}\n")
        if sig_sink_scales:
            f.write(f"  - Information Sink Scales: {sig_sink_scales}\n")
        if sig_balanced_scales:
            f.write(f"  - Balanced Flow Scales: {sig_balanced_scales}\n")
        if not (sig_source_scales or sig_sink_scales or sig_balanced_scales):
            f.write("  - No scales show statistically significant flow patterns\n")
        f.write("\n")
        
        f.write("Flow Transitions:\n")
        for i, transition in enumerate(actual_results['transitions']):
            significance = "Significant" if i in sig_transitions else "Not significant"
            f.write(f"  - Transition at scale {transition['to_scale']}: "
                  f"{transition['from_scale']} → {transition['to_scale']} "
                  f"(Strength: {transition['strength']:.4f}, {significance})\n")
        if not actual_results['transitions']:
            f.write("  - No flow transitions detected\n")
        f.write("\n")
        
        f.write("Scale Groups:\n")
        for i in range(num_scales):
            if actual_results['scale_patterns'][i]['flow_type'] != 'Inactive':
                f.write(f"  - Scale {i+1}: {actual_results['scale_patterns'][i]['flow_type']} flow type\n")
        f.write("\n")
        
        f.write(f"Analysis completed in {time.time() - start_time:.2f} seconds\n")
    
    # Create visualizations
    create_flow_visualizations(results, output_dir, name)
    
    print(f"Analysis completed in {time.time() - start_time:.2f} seconds")
    return results


def main():
    """Main function to run the Multi-Scale Directional Flow Analysis."""
    parser = argparse.ArgumentParser(description='Run Multi-Scale Directional Flow Analysis on CMB data')
    
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--num-scales', type=int, default=10, 
                        help='Number of scales to analyze (higher for finer granularity)')
    parser.add_argument('--num-surrogates', type=int, default=100, 
                        help='Number of surrogate datasets for statistical validation')
    parser.add_argument('--bins', type=int, default=10, help='Number of bins for discretization')
    parser.add_argument('--delay', type=int, default=1, help='Time delay for information transfer')
    parser.add_argument('--max-points', type=int, default=500, 
                        help='Maximum number of points to use for transfer entropy calculation')
    parser.add_argument('--output-dir', default=None, 
                        help='Output directory for results (default: results/multi_scale_flow_TIMESTAMP)')
    
    args = parser.parse_args()
    
    # Print start time
    start_time = time.time()
    print(f"Starting Multi-Scale Directional Flow Analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Path to data files
    wmap_file = os.path.join(cosmic_dir, 'data/wmap/wmap_tt_spectrum_9yr_v5.txt')
    planck_file = os.path.join(cosmic_dir, 'data/planck/planck_tt_spectrum_2018.txt')
    
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
        output_dir = os.path.join(cosmic_dir, 'results', f"multi_scale_flow_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Print parameters
    print("Parameters:")
    print(f"  - Number of scales: {args.num_scales}")
    print(f"  - Number of surrogates: {args.num_surrogates}")
    print(f"  - Number of bins: {args.bins}")
    print(f"  - Delay: {args.delay}")
    print(f"  - Max points: {args.max_points}")
    print(f"  - Output directory: {output_dir}")
    
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
        
        # Run flow analysis on WMAP data
        print("Running multi-scale flow analysis on WMAP data...")
        wmap_start_time = time.time()
        
        wmap_results = run_multi_scale_flow_analysis(
            wmap_processed,
            os.path.join(output_dir, 'wmap'),
            'WMAP',
            num_scales=args.num_scales,
            num_surrogates=args.num_surrogates,
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
        
        # Run flow analysis on Planck data
        print("Running multi-scale flow analysis on Planck data...")
        planck_start_time = time.time()
        
        planck_results = run_multi_scale_flow_analysis(
            planck_processed,
            os.path.join(output_dir, 'planck'),
            'Planck',
            num_scales=args.num_scales,
            num_surrogates=args.num_surrogates,
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
        
        # Create comparison summary
        comparison_dir = os.path.join(output_dir, 'comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        comparison_file = os.path.join(comparison_dir, 'flow_comparison.txt')
        with open(comparison_file, 'w') as f:
            f.write("Multi-Scale Directional Flow Analysis Comparison\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("WMAP Results:\n")
            f.write(f"  - Overall Directional Ratio: {wmap_results['actual_results']['overall_ratio']:.6f}\n")
            f.write(f"  - Forward Pairs: {wmap_results['actual_results']['forward_count']}\n")
            f.write(f"  - Reverse Pairs: {wmap_results['actual_results']['reverse_count']}\n")
            f.write(f"  - Balanced Pairs: {wmap_results['actual_results']['balanced_count']}\n")
            
            if wmap_results['sig_source_scales']:
                f.write(f"  - Significant Source Scales: {wmap_results['sig_source_scales']}\n")
            if wmap_results['sig_sink_scales']:
                f.write(f"  - Significant Sink Scales: {wmap_results['sig_sink_scales']}\n")
            if wmap_results['sig_balanced_scales']:
                f.write(f"  - Significant Balanced Scales: {wmap_results['sig_balanced_scales']}\n")
            
            f.write("\nPlanck Results:\n")
            f.write(f"  - Overall Directional Ratio: {planck_results['actual_results']['overall_ratio']:.6f}\n")
            f.write(f"  - Forward Pairs: {planck_results['actual_results']['forward_count']}\n")
            f.write(f"  - Reverse Pairs: {planck_results['actual_results']['reverse_count']}\n")
            f.write(f"  - Balanced Pairs: {planck_results['actual_results']['balanced_count']}\n")
            
            if planck_results['sig_source_scales']:
                f.write(f"  - Significant Source Scales: {planck_results['sig_source_scales']}\n")
            if planck_results['sig_sink_scales']:
                f.write(f"  - Significant Sink Scales: {planck_results['sig_sink_scales']}\n")
            if planck_results['sig_balanced_scales']:
                f.write(f"  - Significant Balanced Scales: {planck_results['sig_balanced_scales']}\n")
            
            f.write("\nComparison Summary:\n")
            wmap_direction = "Forward" if wmap_results['actual_results']['overall_ratio'] > 0.1 else \
                           "Reverse" if wmap_results['actual_results']['overall_ratio'] < -0.1 else "Balanced"
            
            planck_direction = "Forward" if planck_results['actual_results']['overall_ratio'] > 0.1 else \
                             "Reverse" if planck_results['actual_results']['overall_ratio'] < -0.1 else "Balanced"
            
            consistency = "Same" if wmap_direction == planck_direction else "Different"
            
            f.write(f"  - WMAP Dominant Direction: {wmap_direction}\n")
            f.write(f"  - Planck Dominant Direction: {planck_direction}\n")
            f.write(f"  - Directional Consistency: {consistency}\n")
            
            # Compare significant scales
            wmap_sig_scales = set(wmap_results['sig_source_scales'] + wmap_results['sig_sink_scales'] + 
                               wmap_results['sig_balanced_scales'])
            
            planck_sig_scales = set(planck_results['sig_source_scales'] + planck_results['sig_sink_scales'] + 
                                 planck_results['sig_balanced_scales'])
            
            common_sig_scales = wmap_sig_scales.intersection(planck_sig_scales)
            
            if common_sig_scales:
                f.write(f"  - Common Significant Scales: {sorted(common_sig_scales)}\n")
            else:
                f.write("  - No common significant scales detected\n")
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        # WMAP flow pattern
        plt.subplot(2, 1, 1)
        wmap_scale_patterns = wmap_results['actual_results']['scale_patterns']
        wmap_scale_indices = sorted(wmap_scale_patterns.keys())
        wmap_net_ratios = [wmap_scale_patterns[i]['net_ratio'] for i in wmap_scale_indices]
        
        wmap_flow_types = [wmap_scale_patterns[i]['flow_type'] for i in wmap_scale_indices]
        wmap_colors = ['blue' if ft == 'Source' else 'red' if ft == 'Sink' else 'gray' 
                     for ft in wmap_flow_types]
        
        plt.bar(wmap_scale_indices, wmap_net_ratios, color=wmap_colors, alpha=0.7)
        
        # Add significance markers
        for i in wmap_scale_indices:
            if i in wmap_results['sig_source_scales'] or i in wmap_results['sig_sink_scales'] or \
               i in wmap_results['sig_balanced_scales']:
                plt.scatter(i, wmap_net_ratios[i], marker='*', s=150, color='gold', 
                           edgecolor='black', zorder=10)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('WMAP Scale-Specific Flow Patterns')
        plt.ylabel('Net Directional Ratio')
        
        # Planck flow pattern
        plt.subplot(2, 1, 2)
        planck_scale_patterns = planck_results['actual_results']['scale_patterns']
        planck_scale_indices = sorted(planck_scale_patterns.keys())
        planck_net_ratios = [planck_scale_patterns[i]['net_ratio'] for i in planck_scale_indices]
        
        planck_flow_types = [planck_scale_patterns[i]['flow_type'] for i in planck_scale_indices]
        planck_colors = ['blue' if ft == 'Source' else 'red' if ft == 'Sink' else 'gray' 
                       for ft in planck_flow_types]
        
        plt.bar(planck_scale_indices, planck_net_ratios, color=planck_colors, alpha=0.7)
        
        # Add significance markers
        for i in planck_scale_indices:
            if i in planck_results['sig_source_scales'] or i in planck_results['sig_sink_scales'] or \
               i in planck_results['sig_balanced_scales']:
                plt.scatter(i, planck_net_ratios[i], marker='*', s=150, color='gold', 
                           edgecolor='black', zorder=10)
        
        plt.title('Planck Scale-Specific Flow Patterns')
        plt.xlabel('Scale Index')
        plt.ylabel('Net Directional Ratio')
        
        # Add legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.7),
            plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.7),
            plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.7),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                      markersize=15, markeredgecolor='black')
        ]
        labels = ['Source (Outgoing)', 'Sink (Incoming)', 'Balanced', 'Statistically Significant']
        plt.legend(handles, labels, loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'flow_pattern_comparison.png'))
        plt.close()
        
        print(f"Comparison results saved to {comparison_dir}")
    
    # Create master summary
    summary_file = os.path.join(output_dir, 'multi_scale_flow_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Multi-Scale Directional Flow Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters: scales={args.num_scales}, surrogates={args.num_surrogates}, "
              f"bins={args.bins}, delay={args.delay}\n\n")
        
        if wmap_results:
            wmap_ratio = wmap_results['actual_results']['overall_ratio']
            wmap_dir = "Forward" if wmap_ratio > 0.1 else "Reverse" if wmap_ratio < -0.1 else "Balanced"
            
            f.write("WMAP Results:\n")
            f.write(f"  - Dominant Direction: {wmap_dir}\n")
            f.write(f"  - Directional Ratio: {wmap_ratio:.6f}\n")
            f.write(f"  - Forward Pairs: {wmap_results['actual_results']['forward_count']}\n")
            f.write(f"  - Reverse Pairs: {wmap_results['actual_results']['reverse_count']}\n")
            
            if wmap_results['sig_source_scales']:
                f.write(f"  - Significant Source Scales: {wmap_results['sig_source_scales']}\n")
            if wmap_results['sig_sink_scales']:
                f.write(f"  - Significant Sink Scales: {wmap_results['sig_sink_scales']}\n")
            
            f.write("\n")
        
        if planck_results:
            planck_ratio = planck_results['actual_results']['overall_ratio']
            planck_dir = "Forward" if planck_ratio > 0.1 else "Reverse" if planck_ratio < -0.1 else "Balanced"
            
            f.write("Planck Results:\n")
            f.write(f"  - Dominant Direction: {planck_dir}\n")
            f.write(f"  - Directional Ratio: {planck_ratio:.6f}\n")
            f.write(f"  - Forward Pairs: {planck_results['actual_results']['forward_count']}\n")
            f.write(f"  - Reverse Pairs: {planck_results['actual_results']['reverse_count']}\n")
            
            if planck_results['sig_source_scales']:
                f.write(f"  - Significant Source Scales: {planck_results['sig_source_scales']}\n")
            if planck_results['sig_sink_scales']:
                f.write(f"  - Significant Sink Scales: {planck_results['sig_sink_scales']}\n")
            
            f.write("\n")
        
        if wmap_results and planck_results:
            wmap_dir = "Forward" if wmap_ratio > 0.1 else "Reverse" if wmap_ratio < -0.1 else "Balanced"
            planck_dir = "Forward" if planck_ratio > 0.1 else "Reverse" if planck_ratio < -0.1 else "Balanced"
            consistency = "Same" if wmap_dir == planck_dir else "Different"
            
            f.write("Comparison Summary:\n")
            f.write(f"  - Directional Consistency: {consistency} dominant direction across datasets\n")
            
            # Multi-directional pattern interpretation
            wmap_has_sources = bool(wmap_results['sig_source_scales'])
            wmap_has_sinks = bool(wmap_results['sig_sink_scales'])
            planck_has_sources = bool(planck_results['sig_source_scales'])
            planck_has_sinks = bool(planck_results['sig_sink_scales'])
            
            if (wmap_has_sources and wmap_has_sinks) or (planck_has_sources and planck_has_sinks):
                f.write("  - Pattern: Multi-directional flow detected with both source and sink regions\n")
            elif wmap_dir != planck_dir:
                f.write("  - Pattern: Scale-dependent bidirectional flow (different directions at different scale ranges)\n")
            else:
                f.write(f"  - Pattern: Consistent {wmap_dir.lower()} directional tendency across all scales\n")
            
            f.write("\n")
            
            # Overall interpretation for theoretical implications
            f.write("Theoretical Implications:\n")
            if consistency == "Different" or ((wmap_has_sources and wmap_has_sinks) or (planck_has_sources and planck_has_sinks)):
                f.write("  - These results support the bidirectional information flow concept in Consciousness Field Theory\n")
                f.write("  - Different scales show different directional tendencies, consistent with the stratified universe model\n")
            else:
                f.write("  - These results show a predominant directional flow consistent across scales\n")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Summary available at: {summary_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
