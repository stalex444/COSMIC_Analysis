#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the Scale Transition Test.

This module provides utility functions for the Scale Transition Test, 
which analyzes scale boundaries where organizational principles change
in the CMB power spectrum.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import multiprocessing
from functools import partial
import traceback
from datetime import datetime


def load_wmap_power_spectrum(file_path):
    """
    Load WMAP CMB power spectrum data.
    
    Args:
        file_path (str): Path to WMAP power spectrum file
        
    Returns:
        tuple: (ell, power, error) arrays or (None, None, None) if loading fails
    """
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
    """
    Load Planck CMB power spectrum data.
    
    Args:
        file_path (str): Path to Planck power spectrum file
        
    Returns:
        tuple: (ell, power, error) arrays or (None, None, None) if loading fails
    """
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
        data (numpy.ndarray): Input data array
        smooth (bool): Whether to apply smoothing
        smooth_window (int): Window size for smoothing
        normalize (bool): Whether to normalize the data
        detrend (bool): Whether to remove linear trend
        
    Returns:
        numpy.ndarray: Processed data
    """
    processed_data = data.copy()
    
    # Apply smoothing if requested
    if smooth:
        window = np.ones(smooth_window) / float(smooth_window)
        processed_data = np.convolve(processed_data, window, mode='same')
    
    # Remove linear trend if requested
    if detrend:
        processed_data = signal.detrend(processed_data)
    
    # Normalize if requested
    if normalize:
        processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
    
    return processed_data


def calculate_local_complexity(data, window_size=10):
    """
    Calculate local complexity measures across the data.
    
    Args:
        data (numpy.ndarray): Input data array
        window_size (int): Size of the sliding window
        
    Returns:
        tuple: (complexity_values, window_centers)
    """
    complexity_values = []
    window_centers = []
    
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        
        # Calculate sample entropy as a measure of complexity
        # Bin the data for entropy calculation
        hist, _ = np.histogram(window, bins=min(10, window_size), density=True)
        # Add small constant to avoid log(0)
        hist = hist + 1e-10
        # Normalize
        hist = hist / np.sum(hist)
        # Calculate entropy
        entr = entropy(hist)
        
        complexity_values.append(entr)
        window_centers.append(i + window_size // 2)
    
    return np.array(complexity_values), np.array(window_centers)


def detect_scale_transitions(complexity, window_centers, n_clusters=3, timeout_seconds=30):
    """
    Detect scale transitions using clustering of complexity values.
    
    Args:
        complexity (numpy.ndarray): Complexity values
        window_centers (numpy.ndarray): Centers of the windows
        n_clusters (int): Number of clusters to find
        timeout_seconds (int): Maximum time in seconds to spend on clustering
        
    Returns:
        tuple: (transition_points, cluster_labels, best_n_clusters)
    """
    start_time = datetime.now()
    
    # Reshape for KMeans
    X = complexity.reshape(-1, 1)
    
    # Safety check for very small datasets
    if len(X) < max(10, n_clusters * 2):
        # Return simple division into n_clusters
        best_n_clusters = min(n_clusters, len(X))
        if best_n_clusters < 2:
            # Not enough data for transitions
            return [], np.zeros(len(X), dtype=int), 1
            
        # Simple equal division of data
        cluster_labels = np.zeros(len(X), dtype=int)
        segment_size = len(X) // best_n_clusters
        for i in range(1, best_n_clusters):
            cluster_labels[i*segment_size:] = i
            
        # Find transition points
        transition_points = []
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[i-1]:
                transition_points.append(window_centers[i])
                
        return transition_points, cluster_labels, best_n_clusters
    
    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    max_clusters = min(8, len(X) // 10)  # Limit on max clusters
    max_clusters = max(2, max_clusters)  # Ensure at least 2 clusters
    
    for k in range(2, max_clusters + 1):
        # Check for timeout
        if (datetime.now() - start_time).total_seconds() > timeout_seconds / 2:
            break
            
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append((k, score))
        except Exception:
            # Use previous k if available, otherwise use default
            break
    
    # Get best number of clusters
    if not silhouette_scores:
        best_n_clusters = n_clusters
    else:
        best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    
    # Apply KMeans with best number of clusters
    try:
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(X)
        
        # Identify transition points (boundary between clusters)
        transition_points = []
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[i-1]:
                transition_points.append(window_centers[i])
        
        return transition_points, cluster_labels, best_n_clusters
    except Exception as e:
        # Fallback to simple division
        cluster_labels = np.zeros(len(X), dtype=int)
        segment_size = len(X) // n_clusters
        for i in range(1, n_clusters):
            cluster_labels[i*segment_size:] = i
            
        # Find transition points
        transition_points = []
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[i-1]:
                transition_points.append(window_centers[i])
                
        return transition_points, cluster_labels, n_clusters


def analyze_golden_ratio_alignment(transition_points, ell):
    """
    Analyze alignment of transition points with golden ratio.
    
    Args:
        transition_points (list): Scale transition points
        ell (numpy.ndarray): Multipole moments
        
    Returns:
        tuple: (alignment_scores, mean_alignment, golden_ratio_significance)
    """
    if not transition_points or len(ell) < 3:
        return [], 0.0, 0.0
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Calculate all possible pairs of ℓ values
    alignment_scores = []
    
    for i in range(len(ell)):
        for j in range(i + 1, len(ell)):
            ratio = ell[j] / ell[i]
            # Calculate closeness to golden ratio (phi)
            phi_distance = abs(ratio - phi)
            
            # Check if transition point is within this range
            in_transition = False
            for tp in transition_points:
                if ell[i] <= tp <= ell[j]:
                    in_transition = True
                    break
            
            if in_transition:
                alignment_scores.append((ell[i], ell[j], ratio, phi_distance))
    
    # Calculate mean alignment score
    if alignment_scores:
        mean_alignment = np.mean([score[3] for score in alignment_scores])
        # Calculate significance (closer to 0 is better)
        golden_ratio_significance = 1.0 / (1.0 + mean_alignment)
    else:
        mean_alignment = float('inf')
        golden_ratio_significance = 0.0
    
    return alignment_scores, mean_alignment, golden_ratio_significance


def _run_simulation_wrapper(args):
    """Wrapper function for running a single simulation in multiprocessing.
    
    Args:
        args (tuple): Tuple containing (data, window_size, n_clusters, ell)
        
    Returns:
        tuple: (n_transitions, alignment_score)
    """
    data, window_size, n_clusters, ell = args
    return _run_single_monte_carlo(data, window_size, n_clusters, ell)


def _run_single_monte_carlo(data, window_size, n_clusters, ell):
    """
    Run a single Monte Carlo simulation for scale transition detection.
    
    Args:
        data (numpy.ndarray): Input data array
        window_size (int): Size of the sliding window
        n_clusters (int): Number of clusters for transition detection
        ell (numpy.ndarray): Multipole moments
        
    Returns:
        tuple: (n_transitions, alignment_score)
    """
    try:
        # Generate random permutation of the data
        np.random.shuffle(data)
        
        # Calculate complexity
        complexity_values, window_centers = calculate_local_complexity(data, window_size=window_size)
        
        # Detect transitions
        transition_points, _, _ = detect_scale_transitions(
            complexity_values, window_centers, n_clusters=n_clusters, timeout_seconds=10
        )
        
        # Calculate golden ratio alignment
        _, _, alignment_score = analyze_golden_ratio_alignment(transition_points, ell)
        
        return len(transition_points), alignment_score
    except Exception:
        # Return safe defaults on error
        return 0, 0.0


def run_monte_carlo_parallel(ell, power, n_simulations=10000, window_size=10, n_clusters=3, 
                           timeout_seconds=3600, num_processes=None, chunk_size=100):
    """
    Run Monte Carlo simulations in parallel to assess the significance of scale transitions.
    
    Args:
        ell (numpy.ndarray): Multipole moments
        power (numpy.ndarray): Power spectrum values
        n_simulations (int): Number of simulations
        window_size (int): Window size for complexity calculation
        n_clusters (int): Number of clusters for transition detection
        timeout_seconds (int): Maximum time in seconds to spend on simulations
        num_processes (int): Number of processes to use for parallelization
        chunk_size (int): Size of chunks for simulations to avoid memory issues
        
    Returns:
        tuple: (p_value, phi_optimality, actual_transitions, sim_transitions, 
                complexity_values, window_centers, cluster_labels, alignment_score)
    """
    start_time = datetime.now()
    
    # Preprocess the data
    processed_data = preprocess_data(power, smooth=True, smooth_window=3, normalize=True, detrend=True)
    
    # Calculate complexity for actual data
    complexity_values, window_centers = calculate_local_complexity(processed_data, window_size=window_size)
    
    # Detect transitions for actual data
    transition_points, cluster_labels, best_n_clusters = detect_scale_transitions(
        complexity_values, window_centers, n_clusters=n_clusters, timeout_seconds=60
    )
    
    # Calculate golden ratio alignment for actual data
    _, _, alignment_score = analyze_golden_ratio_alignment(transition_points, ell)
    
    print(f"Actual data:")
    print(f"  Number of transitions: {len(transition_points)}")
    print(f"  Transition points: {transition_points}")
    print(f"  Golden ratio alignment score: {alignment_score:.6f}")
    print(f"  Best number of clusters: {best_n_clusters}")
    
    # Run Monte Carlo simulations to assess significance
    print(f"Running {n_simulations} Monte Carlo simulations to assess significance...")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Using {num_processes} processes for parallel computation")
    
    # Run simulations in smaller chunks to avoid memory issues
    n_chunks = max(1, n_simulations // chunk_size)
    sim_n_transitions = []
    sim_alignment_scores = []
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for chunk_index in range(n_chunks):
            # Check for timeout
            if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                print(f"Timeout reached after {len(sim_n_transitions)} simulations.")
                break
            
            remaining = n_simulations - len(sim_n_transitions)
            if remaining <= 0:
                break
            
            chunk_size_actual = min(chunk_size, remaining)
            print(f"  Running chunk {chunk_index + 1}/{n_chunks} ({chunk_size_actual} simulations)...")
            
            # Prepare arguments for each simulation
            args_list = [(processed_data.copy(), window_size, best_n_clusters, ell) 
                        for _ in range(chunk_size_actual)]
            
            # Run simulations in parallel using the wrapper function
            results = pool.map(_run_simulation_wrapper, args_list)
            
            # Collect results
            for n_trans, align_score in results:
                sim_n_transitions.append(n_trans)
                sim_alignment_scores.append(align_score)
            
            print(f"  Completed {len(sim_n_transitions)}/{n_simulations} simulations")
    
    # Calculate p-value for number of transitions
    p_value_transitions = sum(1 for x in sim_n_transitions if x >= len(transition_points)) / len(sim_n_transitions)
    
    # Calculate p-value for alignment
    p_value_alignment = sum(1 for x in sim_alignment_scores if x >= alignment_score) / len(sim_alignment_scores)
    
    # Combine p-values (use the more stringent one)
    p_value = min(p_value_transitions, p_value_alignment)
    
    # Calculate phi-optimality
    if p_value == 0:
        # Avoid division by zero, use a very small p-value
        p_value = 1.0 / len(sim_n_transitions)
        phi_optimality = 1.0
    else:
        # Convert p-value to optimality score (-1 to 1)
        phi_optimality = 1.0 - 2.0 * p_value
    
    print(f"Results:")
    print(f"  p-value: {p_value:.6f}")
    print(f"  phi-optimality: {phi_optimality:.6f}")
    
    return (p_value, phi_optimality, transition_points, sim_n_transitions, 
            complexity_values, window_centers, cluster_labels, alignment_score)


def plot_scale_transition_results(ell, power, complexity_values, window_centers, 
                                 cluster_labels, transition_points, p_value, phi_optimality, 
                                 sim_n_transitions, actual_n_transitions, alignment_score,
                                 title, output_path):
    """
    Plot scale transition analysis results.
    
    Args:
        ell (numpy.ndarray): Multipole moments
        power (numpy.ndarray): Power spectrum values
        complexity_values (numpy.ndarray): Complexity values
        window_centers (numpy.ndarray): Centers of the windows
        cluster_labels (numpy.ndarray): Cluster labels
        transition_points (list): Scale transition points
        p_value (float): P-value from Monte Carlo simulation
        phi_optimality (float): Phi-optimality score
        sim_n_transitions (list): Number of transitions in simulations
        actual_n_transitions (int): Number of transitions in actual data
        alignment_score (float): Golden ratio alignment score
        title (str): Title for the plot
        output_path (str): Path to save the plot
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Plot power spectrum
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(ell, power, 'b-', alpha=0.7, label='Power Spectrum')
    
    # Add vertical lines for transition points
    for tp in transition_points:
        ax1.axvline(x=tp, color='r', linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Multipole ℓ')
    ax1.set_ylabel('Power Spectrum')
    ax1.set_title(f'{title}\nP-value: {p_value:.6f}, φ-Optimality: {phi_optimality:.6f}')
    ax1.legend()
    
    # Plot complexity with clusters
    ax2 = fig.add_subplot(3, 1, 2)
    
    # Plot complexity values
    ax2.plot(window_centers, complexity_values, 'k-', alpha=0.7, label='Complexity')
    
    # Plot cluster assignments with different colors
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        ax2.scatter(window_centers[mask], complexity_values[mask], 
                    color=colors[i], label=f'Cluster {label}', alpha=0.7)
    
    # Add vertical lines for transition points
    for tp in transition_points:
        ax2.axvline(x=tp, color='r', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Multipole ℓ')
    ax2.set_ylabel('Complexity')
    ax2.set_title(f'Scale Transitions: {len(transition_points)}, GR Alignment: {alignment_score:.6f}')
    ax2.legend()
    
    # Plot histogram of simulation results
    ax3 = fig.add_subplot(3, 1, 3)
    
    ax3.hist(sim_n_transitions, bins=max(10, max(sim_n_transitions) // 2), 
             alpha=0.7, label='Simulations')
    ax3.axvline(x=actual_n_transitions, color='r', linestyle='--', 
                label=f'Actual ({actual_n_transitions})', alpha=0.7)
    
    ax3.set_xlabel('Number of Transitions')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Monte Carlo Simulation Results')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
