#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Add Python 2.7 compatibility
from __future__ import division, print_function

"""
Scale Transition Test for WMAP and Planck CMB data.

This script implements the Scale Transition Test, which analyzes scale boundaries
where organizational principles change in the CMB power spectrum.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime
import argparse
import traceback
import multiprocessing
from functools import partial

# Check if running on Python 2
PY2 = sys.version_info[0] == 2

if PY2:
    # Python 2 compatibility
    range = xrange

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
    print("Starting scale transition detection...")
    start_time = datetime.now()
    
    # Reshape for KMeans
    X = complexity.reshape(-1, 1)
    
    # Safety check for very small datasets
    if len(X) < 10:
        print("Warning: Dataset too small for meaningful clustering (size: {})".format(len(X)))
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
    max_clusters = min(5, len(X) // 10)  # More conservative limit on max clusters
    max_clusters = max(2, max_clusters)  # Ensure at least 2 clusters
    
    print("Finding optimal number of clusters (max: {})...".format(max_clusters))
    
    for k in range(2, max_clusters + 1):
        # Check for timeout
        if (datetime.now() - start_time).total_seconds() > timeout_seconds / 2:
            print("Timeout reached during optimal cluster search. Using {} clusters.".format(k-1))
            break
            
        print("  Testing {} clusters...".format(k))
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append((k, score))
            print("  Silhouette score for {} clusters: {:.4f}".format(k, score))
        except Exception as e:
            print("  Error calculating silhouette score for {} clusters: {}".format(k, str(e)))
            # Use previous k if available, otherwise use default
            break
    
    # Get best number of clusters
    if not silhouette_scores:
        print("Warning: Could not calculate silhouette scores. Using default {} clusters.".format(n_clusters))
        best_n_clusters = n_clusters
    else:
        best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        print("Best number of clusters: {}".format(best_n_clusters))
    
    # Apply KMeans with best number of clusters
    try:
        print("Applying KMeans with {} clusters...".format(best_n_clusters))
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10, max_iter=100)
        cluster_labels = kmeans.fit_predict(X)
    except Exception as e:
        print("Error in KMeans clustering: {}".format(str(e)))
        # Fallback to simple division
        print("Falling back to simple division")
        cluster_labels = np.zeros(len(X), dtype=int)
        segment_size = len(X) // best_n_clusters
        for i in range(1, best_n_clusters):
            cluster_labels[i*segment_size:] = i
    
    # Find transition points between clusters
    transition_points = []
    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] != cluster_labels[i-1]:
            transition_points.append(window_centers[i])
    
    print("Found {} transition points".format(len(transition_points)))
    print("Scale transition detection completed in {:.2f} seconds".format(
        (datetime.now() - start_time).total_seconds()))
    
    return transition_points, cluster_labels, best_n_clusters


def analyze_golden_ratio_alignment(transition_points, ell):
    """
    Analyze alignment of transition points with golden ratio.
    
    Args:
        transition_points (list): Scale transition points
        ell (numpy.ndarray): Multipole moments
        
    Returns:
        tuple: (alignment_scores, mean_alignment, golden_ratio_significance)
    """
    if not transition_points:
        return [], 0, 0
    
    golden_ratio = (1 + np.sqrt(5)) / 2  # ~1.618
    
    # Calculate ratios between adjacent transition points
    ratios = []
    for i in range(len(transition_points) - 1):
        ell_index1 = np.argmin(np.abs(ell - transition_points[i]))
        ell_index2 = np.argmin(np.abs(ell - transition_points[i+1]))
        
        # Get actual ell values
        ell1 = ell[ell_index1]
        ell2 = ell[ell_index2]
        
        # Calculate ratio
        ratio = max(ell1, ell2) / float(min(ell1, ell2))
        ratios.append(ratio)
    
    # Calculate alignment with golden ratio
    alignment_scores = [abs(ratio - golden_ratio) / golden_ratio for ratio in ratios]
    mean_alignment = np.mean(alignment_scores) if alignment_scores else 1.0
    
    # Calculate significance (lower is better, 0 is perfect alignment)
    golden_ratio_significance = 1 - np.exp(-5 * mean_alignment)
    
    return alignment_scores, mean_alignment, golden_ratio_significance


def run_monte_carlo(ell, power, n_simulations=30, window_size=10, n_clusters=3, timeout_seconds=120):
    """
    Run Monte Carlo simulations to assess the significance of scale transitions.
    
    Args:
        ell (numpy.ndarray): Multipole moments
        power (numpy.ndarray): Power spectrum values
        n_simulations (int): Number of simulations
        window_size (int): Window size for complexity calculation
        n_clusters (int): Number of clusters for transition detection
        timeout_seconds (int): Maximum time in seconds to spend on simulations
        
    Returns:
        tuple: (p_value, phi_optimality, actual_transitions, sim_transitions, 
                complexity_values, window_centers, cluster_labels, alignment_score)
    """
    print("Starting Monte Carlo simulations for scale transition analysis...")
    start_time = datetime.now()
    
    # Calculate complexity and detect transitions for actual data
    print("Calculating complexity for actual data...")
    complexity_values, window_centers = calculate_local_complexity(power, window_size)
    
    print("Detecting scale transitions in actual data...")
    actual_transitions, cluster_labels, best_n_clusters = detect_scale_transitions(
        complexity_values, window_centers, n_clusters, timeout_seconds=min(30, timeout_seconds//4)
    )
    
    # Analyze golden ratio alignment
    print("Analyzing golden ratio alignment...")
    alignment_scores, mean_alignment, golden_ratio_significance = analyze_golden_ratio_alignment(
        actual_transitions, ell
    )
    
    # Calculate number of transitions
    actual_n_transitions = len(actual_transitions)
    print("Actual data has {} transitions".format(actual_n_transitions))
    
    # Check if we have enough time left for simulations
    elapsed_time = (datetime.now() - start_time).total_seconds()
    remaining_time = timeout_seconds - elapsed_time
    
    if remaining_time <= 0:
        print("Warning: Timeout reached before simulations could start.")
        # Return a conservative p-value
        return 0.5, 0.0, actual_transitions, [actual_n_transitions], complexity_values, window_centers, cluster_labels, mean_alignment
    
    # Run simulations
    print("Running {} Monte Carlo simulations...".format(n_simulations))
    sim_n_transitions = []
    significant_count = 0
    required_for_significance = 0.05 * n_simulations  # Number of simulations needed to exceed significance threshold
    
    for i in range(n_simulations):
        # Check for timeout every 5 simulations
        if i % 5 == 0:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if elapsed_time > timeout_seconds:
                print("Timeout reached after {} seconds and {} simulations.".format(
                    elapsed_time, i))
                break
            
            # Update progress more frequently
            print("  Simulation {}/{} ({:.1f}%)".format(i, n_simulations, 100.0 * i / n_simulations))
            
            # Update time estimate per simulation for future reference
            if i > 0:
                time_per_sim_estimate = elapsed_time / i
                print("  Estimated time per simulation: {:.2f} seconds".format(time_per_sim_estimate))
        
        # Create random permutation of the power spectrum
        sim_power = np.random.permutation(power)
        
        try:
            # Calculate complexity and detect transitions with a short timeout
            sim_complexity, sim_centers = calculate_local_complexity(sim_power, window_size)
            sim_transitions, _, _ = detect_scale_transitions(
                sim_complexity, sim_centers, best_n_clusters, timeout_seconds=5
            )
            
            # Calculate number of transitions
            n_transitions = len(sim_transitions)
            sim_n_transitions.append(n_transitions)
            
            # Check if simulation exceeds actual number of transitions
            if n_transitions >= actual_n_transitions:
                significant_count += 1
                
                # Early stopping if we've already exceeded the significance threshold
                if significant_count > required_for_significance:
                    print("  Early stopping at simulation {}: already exceeded significance threshold".format(i+1))
                    break
                    
        except Exception as e:
            print("  Error in simulation {}: {}".format(i, str(e)))
            # Add a neutral value to avoid biasing the results
            sim_n_transitions.append(actual_n_transitions)
    
    # Calculate p-value based on number of transitions
    if not sim_n_transitions:
        print("Warning: No valid simulations completed. Using conservative p-value of 0.5.")
        p_value = 0.5
    else:
        p_value = np.mean([1 if sim >= actual_n_transitions else 0 for sim in sim_n_transitions])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    if len(sim_n_transitions) < 2:
        print("Warning: Not enough simulations for reliable phi-optimality calculation.")
        phi_optimality = 0
    else:
        sim_mean = np.mean(sim_n_transitions)
        sim_std = np.std(sim_n_transitions)
        if sim_std == 0:
            phi_optimality = 0
        else:
            z_score = (actual_n_transitions - sim_mean) / sim_std
            # Convert z-score to a value between -1 and 1 using a sigmoid-like function
            phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
    
    # Report final statistics
    print("Completed {} simulations in {:.2f} seconds".format(
        len(sim_n_transitions), (datetime.now() - start_time).total_seconds()))
    print("p-value: {:.4f}, phi-optimality: {:.4f}".format(p_value, phi_optimality))
    
    return (p_value, phi_optimality, actual_transitions, sim_n_transitions, 
            complexity_values, window_centers, cluster_labels, mean_alignment)


def run_monte_carlo_parallel(ell, power, n_simulations=30, window_size=10, n_clusters=3, 
                           timeout_seconds=120, num_processes=None):
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
        
    Returns:
        tuple: (p_value, phi_optimality, actual_transitions, sim_transitions, 
                complexity_values, window_centers, cluster_labels, alignment_score)
    """
    print("Starting parallel Monte Carlo simulations for scale transition analysis...")
    start_time = datetime.now()
    
    # If num_processes is not specified, use all available cores
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Calculate complexity and detect transitions for actual data
    print("Calculating complexity for actual data...")
    complexity_values, window_centers = calculate_local_complexity(power, window_size)
    
    print("Detecting scale transitions in actual data...")
    actual_transitions, cluster_labels, best_n_clusters = detect_scale_transitions(
        complexity_values, window_centers, n_clusters, timeout_seconds=min(30, timeout_seconds//4)
    )
    
    # Analyze golden ratio alignment
    print("Analyzing golden ratio alignment...")
    alignment_scores, mean_alignment, golden_ratio_significance = analyze_golden_ratio_alignment(
        actual_transitions, ell
    )
    
    # Calculate number of transitions
    actual_n_transitions = len(actual_transitions)
    print("Actual data has {} transitions".format(actual_n_transitions))
    
    # Run simulations in parallel
    print("Running {} Monte Carlo simulations in parallel with {} processes...".format(n_simulations, num_processes))
    
    # Create chunks of simulations for each process - smaller chunks for better load balancing
    chunk_size = max(1, min(500, n_simulations // (num_processes * 2)))
    chunks = [range(i, min(i + chunk_size, n_simulations)) 
              for i in range(0, n_simulations, chunk_size)]
    
    print("Created {} chunks with approximately {} simulations per chunk".format(
        len(chunks), chunk_size))
    
    # Define a worker function that processes a chunk of simulations
    def process_chunk(chunk_indices):
        chunk_results = []
        for i in chunk_indices:
            try:
                # Create random permutation of the power spectrum
                sim_power = np.random.permutation(power)
                
                # Calculate complexity and detect transitions with a short timeout
                sim_complexity, sim_centers = calculate_local_complexity(sim_power, window_size)
                sim_transitions, _, _ = detect_scale_transitions(
                    sim_complexity, sim_centers, best_n_clusters, timeout_seconds=5
                )
                
                # Calculate number of transitions
                n_transitions = len(sim_transitions)
                chunk_results.append(n_transitions)
                
                # Report progress periodically
                if i % 50 == 0 and i > 0:
                    print("  Completed {} simulations in chunk".format(i))
                    
            except Exception as e:
                print("  Error in simulation {}: {}".format(i, str(e)))
                # Add a neutral value to avoid biasing the results
                chunk_results.append(actual_n_transitions)
        return chunk_results
    
    # Create and start processes
    processes = []
    result_queue = multiprocessing.Queue()
    
    for i, chunk in enumerate(chunks):
        process = multiprocessing.Process(
            target=lambda q, chunk_idx: q.put((chunk_idx, process_chunk(chunk))),
            args=(result_queue, i)
        )
        processes.append(process)
        process.start()
    
    # Collect results with early stopping capability
    results = [None] * len(chunks)
    completed_chunks = 0
    min_simulations = min(1000, n_simulations // 3)  # Minimum simulations before early stopping check
    sim_n_transitions = []
    
    # Track time for adaptive timeout
    chunk_start_time = datetime.now()
    
    while completed_chunks < len(chunks):
        # Check if we're approaching the timeout
        current_elapsed = (datetime.now() - start_time).total_seconds()
        if current_elapsed > timeout_seconds * 0.9:  # 90% of timeout reached
            print("Approaching timeout limit, stopping early after {} simulations".format(
                len([n for chunk in results if chunk is not None for n in chunk])))
            break
            
        # Get next completed chunk
        try:
            # Use a timeout to avoid blocking indefinitely
            chunk_idx, chunk_results = result_queue.get(timeout=10)
            results[chunk_idx] = chunk_results
            completed_chunks += 1
            
            # Update time estimate based on actual performance
            chunk_elapsed = (datetime.now() - chunk_start_time).total_seconds()
            chunk_start_time = datetime.now()
            
            # Check for early stopping criteria after minimum simulations
            if completed_chunks >= len(chunks) // 3:
                # Flatten current results
                current_results = [n for chunk in results if chunk is not None for n in chunk]
                if len(current_results) >= min_simulations:
                    # Calculate current p-value
                    current_p_value = np.mean([1 if sim >= actual_n_transitions else 0 for sim in current_results])
                    
                    # If p-value is very significant or very insignificant, we can stop early
                    if current_p_value < 0.01 or current_p_value > 0.99:
                        print("Early stopping criteria met with p-value: {:.4f} after {} simulations".format(
                            current_p_value, len(current_results)))
                        
                        # Terminate remaining processes
                        for process in processes:
                            if process.is_alive():
                                process.terminate()
                        
                        # Use current results
                        sim_n_transitions = current_results
                        break
            
            # Print progress
            if completed_chunks % max(1, len(chunks) // 10) == 0:
                print("Completed {}/{} chunks ({:.1f}%)".format(
                    completed_chunks, len(chunks), 100.0 * completed_chunks / len(chunks)))
                
        except Queue.Empty:  # Python 2 compatibility
            print("Waiting for results...")
            continue
    
    # Wait for all processes to finish if not early stopped
    if len(sim_n_transitions) == 0:
        # Terminate any remaining processes if we're near timeout
        current_elapsed = (datetime.now() - start_time).total_seconds()
        if current_elapsed > timeout_seconds * 0.9:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        else:
            for process in processes:
                process.join(timeout=max(1, (timeout_seconds - current_elapsed) / len(processes)))
        
        # Flatten results
        sim_n_transitions = [n for chunk in results if chunk is not None for n in chunk]
    
    # Calculate p-value based on number of transitions
    if not sim_n_transitions:
        print("Warning: No valid simulations completed. Using conservative p-value of 0.5.")
        p_value = 0.5
    else:
        p_value = np.mean([1 if sim >= actual_n_transitions else 0 for sim in sim_n_transitions])
    
    # Calculate phi-optimality (scaled between -1 and 1)
    if len(sim_n_transitions) < 2:
        print("Warning: Not enough simulations for reliable phi-optimality calculation.")
        phi_optimality = 0
    else:
        sim_mean = np.mean(sim_n_transitions)
        sim_std = np.std(sim_n_transitions)
        if sim_std == 0:
            phi_optimality = 0
        else:
            z_score = (actual_n_transitions - sim_mean) / sim_std
            # Convert z-score to a value between -1 and 1 using a sigmoid-like function
            phi_optimality = 2.0 / (1.0 + np.exp(-z_score)) - 1.0
    
    # Report final statistics
    print("Completed {} simulations in {:.2f} seconds".format(
        len(sim_n_transitions), (datetime.now() - start_time).total_seconds()))
    print("p-value: {:.6f}, phi-optimality: {:.6f}".format(p_value, phi_optimality))
    
    return (p_value, phi_optimality, actual_transitions, sim_n_transitions, 
            complexity_values, window_centers, cluster_labels, mean_alignment)


def plot_scale_transition_results(ell, power, complexity_values, window_centers, 
                                 cluster_labels, transition_points, p_value, phi_optimality, 
                                 sim_n_transitions, actual_n_transitions, alignment_score,
                                 title, output_path):
    """Plot scale transition analysis results."""
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot power spectrum with transition points
        ax1.plot(ell, power, 'b-', alpha=0.7, label='Power Spectrum')
        
        # Add vertical lines at transition points
        for tp in transition_points:
            ax1.axvline(tp, color='r', linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Multipole Moment (ℓ)')
        ax1.set_ylabel('Power')
        ax1.set_title('Power Spectrum with Scale Transitions')
        ax1.grid(True, alpha=0.3)
        
        # Add legend with transition points
        if transition_points:
            tp_str = ', '.join(['{:.1f}'.format(tp) for tp in transition_points])
            ax1.text(0.05, 0.95, 'Transition Points: {}'.format(tp_str), 
                    transform=ax1.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot complexity values with cluster labels
        scatter = ax2.scatter(window_centers, complexity_values, c=cluster_labels, 
                             cmap='viridis', alpha=0.7, s=30)
        
        # Add vertical lines at transition points
        for tp in transition_points:
            ax2.axvline(tp, color='r', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Window Center (ℓ)')
        ax2.set_ylabel('Complexity (Entropy)')
        ax2.set_title('Local Complexity with Cluster Labels')
        ax2.grid(True, alpha=0.3)
        
        # Add legend for clusters - compatible with older matplotlib versions
        try:
            # Try newer matplotlib method first
            legend1 = ax2.legend(*scatter.legend_elements(),
                                loc="upper right", title="Clusters")
            ax2.add_artist(legend1)
        except (AttributeError, TypeError):
            # Fallback for older matplotlib versions
            unique_labels = np.unique(cluster_labels)
            legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.viridis(i / float(max(1, len(unique_labels) - 1))), 
                                        markersize=10, label='Cluster {}'.format(i)) 
                             for i in range(len(unique_labels))]
            ax2.legend(handles=legend_handles, loc="upper right", title="Clusters")
        
        # Plot simulation results
        ax3.hist(sim_n_transitions, bins=min(20, max(sim_n_transitions) - min(sim_n_transitions) + 1), 
                alpha=0.7, color='gray', label='Random Simulations')
        ax3.axvline(actual_n_transitions, color='r', linestyle='--', linewidth=2, 
                   label='Actual: {}'.format(actual_n_transitions))
        
        ax3.set_title('Monte Carlo Simulations: Number of Scale Transitions')
        ax3.set_xlabel('Number of Transitions')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True)
        
        # Add text with results
        plt.figtext(0.5, 0.01, 
                   'P-value: {:.4f} | Phi-Optimality: {:.4f} | Golden Ratio Alignment: {:.4f}'.format(p_value, phi_optimality, alignment_score), 
                   ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print("Warning: Error in plotting scale transition results: {}".format(str(e)))
        print("Continuing with analysis...")


def visualize_results(results, output_path, title):
    """
    Visualize scale transition test results from a results dictionary.
    
    Args:
        results (dict): Dictionary containing test results
        output_path (str): Path to save the visualization
        title (str): Title for the plot
    """
    try:
        # Extract data from results dictionary
        ell = results['ell']
        power = results['power']
        complexity = results['complexity']
        window_centers = results['window_centers']
        cluster_labels = results['cluster_labels']
        transition_points = results['transitions']
        p_value = results['p_value']
        phi_optimality = results['phi_optimality']
        sim_transitions = results['sim_transitions']
        actual_n_transitions = results['n_transitions']
        alignment_score = results['alignment_score']
        
        # Call the existing plotting function
        plot_scale_transition_results(
            ell, power, complexity, window_centers, cluster_labels, 
            transition_points, p_value, phi_optimality, sim_transitions, 
            actual_n_transitions, alignment_score, title, output_path
        )
    except Exception as e:
        print("Error in visualize_results: {}".format(str(e)))
        traceback.print_exc()


def run_scale_transition_test(ell, power, output_dir, name, n_simulations=10000, 
                             window_size=10, n_clusters=3, timeout_seconds=3600, 
                             parallel=True, num_processes=None):
    """
    Run scale transition test on the provided data.
    
    Args:
        ell (numpy.ndarray): Array of multipole moments
        power (numpy.ndarray): Array of power spectrum values
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        n_simulations (int): Number of simulations for Monte Carlo
        window_size (int): Window size for complexity calculation
        n_clusters (int): Number of clusters for transition detection
        timeout_seconds (int): Maximum time in seconds to spend on simulations
        parallel (bool): Whether to use parallel processing
        num_processes (int): Number of processes to use for parallelization
        
    Returns:
        dict: Analysis results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Start timing
    start_time = datetime.now()
    
    try:
        # Run Monte Carlo simulations
        print("Running scale transition test for {} data...".format(name))
        
        if parallel:
            p_value, phi_optimality, transition_points, sim_n_transitions, complexity_values, window_centers, cluster_labels, alignment_score = run_monte_carlo_parallel(
                ell, power, n_simulations=n_simulations, window_size=window_size, 
                n_clusters=n_clusters, timeout_seconds=timeout_seconds, num_processes=num_processes
            )
        else:
            p_value, phi_optimality, transition_points, sim_n_transitions, complexity_values, window_centers, cluster_labels, alignment_score = run_monte_carlo(
                ell, power, n_simulations=n_simulations, window_size=window_size, 
                n_clusters=n_clusters, timeout_seconds=timeout_seconds
            )
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        print("Execution time: {:.2f} seconds".format(execution_time))
        
        # Plot results
        plot_path = os.path.join(output_dir, '{}_scale_transitions.png'.format(name.lower()))
        plot_scale_transition_results(
            ell, power, complexity_values, window_centers, cluster_labels, transition_points,
            p_value, phi_optimality, sim_n_transitions, len(transition_points), alignment_score,
            'Scale Transition Test: {} CMB Data'.format(name), plot_path
        )
        
        # Save results to file
        results_path = os.path.join(output_dir, '{}_scale_transitions.txt'.format(name.lower()))
        with open(results_path, 'w') as f:
            f.write('Scale Transition Test Results: {} CMB Data\n'.format(name))
            f.write('=' * 60 + '\n\n')
            f.write('Number of simulations: {}\n'.format(n_simulations))
            f.write('Window size: {}\n'.format(window_size))
            f.write('Number of clusters: {}\n'.format(n_clusters))
            f.write('Number of scale transitions: {}\n'.format(len(transition_points)))
            f.write('P-value: {:.6f}\n'.format(p_value))
            f.write('Phi-Optimality: {:.6f}\n'.format(phi_optimality))
            f.write('Golden Ratio Alignment Score: {:.6f}\n'.format(alignment_score))
            f.write('Execution time: {:.2f} seconds\n'.format(execution_time))
            
            # Determine significance
            significant = p_value < 0.05
            f.write('\nStatistical Significance: {}\n'.format('Significant' if significant else 'Not Significant'))
            
            if significant:
                f.write('The number of scale transitions is significantly different from what would be expected by chance.\n')
                f.write('This suggests structured organization across scales in the CMB power spectrum.\n')
            else:
                f.write('The number of scale transitions is not significantly different from what would be expected by chance.\n')
                f.write('This suggests no structured organization across scales in the CMB power spectrum.\n')
            
            f.write('\nScale Transition Points:\n')
            for i, point in enumerate(transition_points):
                f.write('  Transition {}: l = {:.1f}\n'.format(i+1, point))
        
        # Return results dictionary
        results = {
            'name': name,
            'n_simulations': n_simulations,
            'window_size': window_size,
            'n_clusters': n_clusters,
            'transition_points': transition_points,
            'complexity_values': complexity_values.tolist(),
            'window_centers': window_centers.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'p_value': p_value,
            'phi_optimality': phi_optimality,
            'alignment_score': alignment_score,
            'significant': p_value < 0.05,
            'execution_time': execution_time
        }
        
        return results
        
    except Exception as e:
        print("Error in scale transition test: {}".format(str(e)))
        traceback.print_exc()
        
        # Return partial results if available
        return {
            'name': name,
            'error': str(e),
            'n_simulations': n_simulations,
            'window_size': window_size,
            'n_clusters': n_clusters,
            'execution_time': (datetime.now() - start_time).total_seconds()
        }


def compare_results(wmap_results, planck_results, output_dir):
    """Compare scale transition test results between WMAP and Planck data."""
    try:
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Calculate differences
        n_transitions_diff = abs(wmap_results['n_transitions'] - planck_results['n_transitions'])
        phi_diff = abs(wmap_results['phi_optimality'] - planck_results['phi_optimality'])
        alignment_diff = abs(wmap_results['alignment_score'] - planck_results['alignment_score'])
        
        # Save comparison to file
        comparison_path = os.path.join(output_dir, 'scale_transition_comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write('Scale Transition Test Comparison: WMAP vs Planck\n')
            f.write('=' * 50 + '\n\n')
            
            f.write('WMAP Number of Transitions: {}\n'.format(wmap_results['n_transitions']))
            f.write('WMAP P-value: {:.6f}\n'.format(wmap_results["p_value"]))
            f.write('WMAP Phi-Optimality: {:.6f}\n'.format(wmap_results['phi_optimality']))
            f.write('WMAP Significant: {}\n'.format(wmap_results['significant']))
            f.write('WMAP Golden Ratio Alignment: {:.6f}\n\n'.format(wmap_results['alignment_score']))
            
            f.write('Planck Number of Transitions: {}\n'.format(planck_results['n_transitions']))
            f.write('Planck P-value: {:.6f}\n'.format(planck_results["p_value"]))
            f.write('Planck Phi-Optimality: {:.6f}\n'.format(planck_results['phi_optimality']))
            f.write('Planck Significant: {}\n'.format(planck_results['significant']))
            f.write('Planck Golden Ratio Alignment: {:.6f}\n\n'.format(planck_results['alignment_score']))
            
            f.write('Difference in Number of Transitions: {}\n'.format(n_transitions_diff))
            f.write('Difference in Phi-Optimality: {:.6f}\n'.format(phi_diff))
            f.write('Difference in Golden Ratio Alignment: {:.6f}\n'.format(alignment_diff))
        
        # Create comparison plot
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Bar chart of key metrics
            metrics = ['Number of Transitions', 'Phi-Optimality', 'GR Alignment']
            wmap_values = [wmap_results['n_transitions'], 
                          wmap_results['phi_optimality'], 
                          1 - wmap_results['alignment_score']]  # Invert alignment score for better visualization
            planck_values = [planck_results['n_transitions'], 
                            planck_results['phi_optimality'], 
                            1 - planck_results['alignment_score']]  # Invert alignment score
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, wmap_values, width, label='WMAP', color='blue', alpha=0.7)
            ax1.bar(x + width/2, planck_values, width, label='Planck', color='red', alpha=0.7)
            
            ax1.set_ylabel('Value')
            ax1.set_title('Scale Transition Analysis: WMAP vs Planck')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            
            # Add text with results
            ax1.text(0 - width/2, wmap_values[0] + 0.2, 
                    'p={:.4f}'.format(wmap_results["p_value"]), 
                    ha='center', va='bottom', color='blue', fontweight='bold')
            ax1.text(0 + width/2, planck_values[0] + 0.2, 
                    'p={:.4f}'.format(planck_results["p_value"]), 
                    ha='center', va='bottom', color='red', fontweight='bold')
            
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot 2: Comparison of transition points
            wmap_transitions = wmap_results['transition_points']
            planck_transitions = planck_results['transition_points']
            
            # Plot transition points
            if wmap_transitions:
                ax2.scatter(wmap_transitions, np.ones(len(wmap_transitions)) * 1, 
                           color='blue', s=100, alpha=0.7, label='WMAP Transitions')
                for tp in wmap_transitions:
                    ax2.text(tp, 1.05, '{:.1f}'.format(tp), ha='center', va='bottom', color='blue')
            
            if planck_transitions:
                ax2.scatter(planck_transitions, np.ones(len(planck_transitions)) * 0.5, 
                           color='red', s=100, alpha=0.7, label='Planck Transitions')
                for tp in planck_transitions:
                    ax2.text(tp, 0.55, '{:.1f}'.format(tp), ha='center', va='bottom', color='red')
            
            ax2.set_yticks([0.5, 1])
            ax2.set_yticklabels(['Planck', 'WMAP'])
            ax2.set_xlabel('Multipole Moment (ℓ)')
            ax2.set_title('Scale Transition Points')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            comparison_plot_path = os.path.join(output_dir, 'scale_transition_comparison.png')
            plt.savefig(comparison_plot_path)
            plt.close()
        except Exception as e:
            print("Warning: Error in creating comparison plot: {}".format(str(e)))
            print("Continuing with analysis...")
        
        print("\nComparison Results:")
        print("  Difference in Number of Transitions: {}".format(n_transitions_diff))
        print("  Difference in Phi-Optimality: {:.6f}".format(phi_diff))
        print("  Difference in Golden Ratio Alignment: {:.6f}".format(alignment_diff))
        print("  Comparison saved to: {}".format(comparison_path))
    except Exception as e:
        print("Error in comparing results: {}".format(str(e)))
        print("Continuing with analysis...")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Scale Transition Test on WMAP and Planck data')
    parser.add_argument('--wmap-only', action='store_true', help='Run analysis only on WMAP data')
    parser.add_argument('--planck-only', action='store_true', help='Run analysis only on Planck data')
    parser.add_argument('--n-simulations', type=int, default=10000, 
                        help='Number of simulations for Monte Carlo. Default: 10000')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Window size for complexity calculation. Default: 10')
    parser.add_argument('--n-clusters', type=int, default=3,
                        help='Number of clusters for transition detection. Default: 3')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Maximum time in seconds to spend on simulations. Default: 3600')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing to the data')
    parser.add_argument('--detrend', action='store_true', help='Apply detrending to the data')
    parser.add_argument('--output-dir', default=None, 
                        help='Directory to save results. Default: results/scale_transition_TIMESTAMP')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--num-processes', type=int, default=multiprocessing.cpu_count(),
                        help='Number of processes to use for parallel computation (default: all available cores)')
    
    args = parser.parse_args()
    
    # Start timing the execution
    start_time = datetime.now()
    print("Starting Scale Transition Test at {}".format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    
    # Path to data files
    wmap_file = 'wmap_data/raw_data/wmap_binned_tt_spectrum_9yr_v5.txt'
    planck_file = os.path.join(parent_dir, 'Cosmic_Consciousness_Analysis/planck_data/power_spectra/COM_PowerSpect_CMB-TT-full_R3.01.txt')
    
    # Check if files exist
    if not args.planck_only and not os.path.exists(wmap_file):
        print("Error: WMAP power spectrum file not found: {}".format(wmap_file))
        return 1
    
    if not args.wmap_only and not os.path.exists(planck_file):
        print("Error: Planck power spectrum file not found: {}".format(planck_file))
        print("Please make sure the Planck data is available in the Cosmic_Consciousness_Analysis repository.")
        return 1
    
    # Create output directory with timestamp
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('results', "scale_transition_{}".format(timestamp))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory: {}".format(output_dir))
    
    # Initialize results dictionaries
    wmap_results = None
    planck_results = None
    
    # Process WMAP data
    if not args.planck_only:
        print("\n" + "="*50)
        print("Processing WMAP data")
        print("="*50)
        
        try:
            # Check remaining time
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > args.timeout:
                print("Timeout reached after {:.1f} seconds. Skipping WMAP analysis.".format(elapsed))
            else:
                remaining_time = args.timeout - elapsed
                print("Time remaining: {:.1f} seconds".format(remaining_time))
                
                # Load and preprocess WMAP data
                wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(wmap_file)
                
                if wmap_ell is None:
                    print("Error: Failed to load WMAP data")
                    raise ValueError("WMAP data loading failed")
                
                if args.smooth:
                    print("Applying smoothing to WMAP data...")
                    wmap_power = preprocess_data(wmap_power, smooth=True)
                
                if args.detrend:
                    print("Applying detrending to WMAP data...")
                    wmap_power = preprocess_data(wmap_power, detrend=True)
                
                # Run Monte Carlo simulations
                wmap_p_value, wmap_phi_optimality, wmap_transitions, wmap_sim_transitions, \
                wmap_complexity, wmap_window_centers, wmap_cluster_labels, wmap_alignment = run_monte_carlo_parallel(
                    wmap_ell, wmap_power, 
                    n_simulations=args.n_simulations,
                    window_size=args.window_size,
                    n_clusters=args.n_clusters,
                    timeout_seconds=int(args.timeout * 0.8)  # Use 80% of remaining time
                )
                
                # Determine if result is significant
                wmap_significant = wmap_p_value < 0.05
                
                # Store results
                wmap_results = {
                    'ell': wmap_ell,
                    'power': wmap_power,
                    'p_value': wmap_p_value,
                    'phi_optimality': wmap_phi_optimality,
                    'transitions': wmap_transitions,
                    'n_transitions': len(wmap_transitions),
                    'sim_transitions': wmap_sim_transitions,
                    'complexity': wmap_complexity,
                    'window_centers': wmap_window_centers,
                    'cluster_labels': wmap_cluster_labels,
                    'alignment_score': wmap_alignment,
                    'significant': wmap_significant
                }
                
                # Generate visualizations if requested
                if args.visualize:
                    print("Generating WMAP visualizations...")
                    visualize_results(
                        wmap_results, 
                        os.path.join(output_dir, 'wmap_scale_transitions.png'),
                        title='WMAP Scale Transitions'
                    )
                    
                    # Generate histogram of simulation results
                    plt.figure(figsize=(10, 6))
                    plt.hist(wmap_sim_transitions, bins=10, alpha=0.7)
                    plt.axvline(len(wmap_transitions), color='red', linestyle='dashed', 
                                linewidth=2, label='Actual: {}'.format(len(wmap_transitions)))
                    plt.xlabel('Number of Transitions')
                    plt.ylabel('Frequency')
                    plt.title('WMAP: Distribution of Transitions in Simulations')
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, 'wmap_simulation_histogram.png'))
                    plt.close()
        
        except Exception as e:
            print("Error in WMAP analysis: {}".format(str(e)))
            traceback.print_exc()
    
    # Process Planck data
    if not args.wmap_only:
        print("\n" + "="*50)
        print("Processing Planck data")
        print("="*50)
        
        try:
            # Check remaining time
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > args.timeout:
                print("Timeout reached after {:.1f} seconds. Skipping Planck analysis.".format(elapsed))
            else:
                remaining_time = args.timeout - elapsed
                print("Time remaining: {:.1f} seconds".format(remaining_time))
                
                # Load and preprocess Planck data
                planck_ell, planck_power, planck_error = load_planck_power_spectrum(planck_file)
                
                if planck_ell is None:
                    print("Error: Failed to load Planck data")
                    raise ValueError("Planck data loading failed")
                
                if args.smooth:
                    print("Applying smoothing to Planck data...")
                    planck_power = preprocess_data(planck_power, smooth=True)
                
                if args.detrend:
                    print("Applying detrending to Planck data...")
                    planck_power = preprocess_data(planck_power, detrend=True)
                
                # Run Monte Carlo simulations
                planck_p_value, planck_phi_optimality, planck_transitions, planck_sim_transitions, \
                planck_complexity, planck_window_centers, planck_cluster_labels, planck_alignment = run_monte_carlo_parallel(
                    planck_ell, planck_power, 
                    n_simulations=args.n_simulations,
                    window_size=args.window_size,
                    n_clusters=args.n_clusters,
                    timeout_seconds=int(args.timeout * 0.8)  # Use 80% of remaining time
                )
                
                # Determine if result is significant
                planck_significant = planck_p_value < 0.05
                
                # Store results
                planck_results = {
                    'ell': planck_ell,
                    'power': planck_power,
                    'p_value': planck_p_value,
                    'phi_optimality': planck_phi_optimality,
                    'transitions': planck_transitions,
                    'n_transitions': len(planck_transitions),
                    'sim_transitions': planck_sim_transitions,
                    'complexity': planck_complexity,
                    'window_centers': planck_window_centers,
                    'cluster_labels': planck_cluster_labels,
                    'alignment_score': planck_alignment,
                    'significant': planck_significant
                }
                
                # Generate visualizations if requested
                if args.visualize:
                    print("Generating Planck visualizations...")
                    visualize_results(
                        planck_results, 
                        os.path.join(output_dir, 'planck_scale_transitions.png'),
                        title='Planck Scale Transitions'
                    )
                    
                    # Generate histogram of simulation results
                    plt.figure(figsize=(10, 6))
                    plt.hist(planck_sim_transitions, bins=10, alpha=0.7)
                    plt.axvline(len(planck_transitions), color='red', linestyle='dashed', 
                                linewidth=2, label='Actual: {}'.format(len(planck_transitions)))
                    plt.xlabel('Number of Transitions')
                    plt.ylabel('Frequency')
                    plt.title('Planck: Distribution of Transitions in Simulations')
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, 'planck_simulation_histogram.png'))
                    plt.close()
        
        except Exception as e:
            print("Error in Planck analysis: {}".format(str(e)))
            traceback.print_exc()
    
    # Write results to file
    print("\n" + "="*50)
    print("Writing results to file")
    print("="*50)
    
    try:
        with open(os.path.join(output_dir, 'scale_transition_results.txt'), 'w') as f:
            f.write('Scale Transition Test Results\n')
            f.write('=' * 50 + '\n\n')
            
            if wmap_results:
                f.write('WMAP Number of Transitions: {}\n'.format(wmap_results['n_transitions']))
                f.write('WMAP P-value: {:.6f}\n'.format(wmap_results["p_value"]))
                f.write('WMAP Phi-Optimality: {:.6f}\n'.format(wmap_results['phi_optimality']))
                f.write('WMAP Significant: {}\n'.format(wmap_results['significant']))
                f.write('WMAP Golden Ratio Alignment: {:.6f}\n\n'.format(wmap_results['alignment_score']))
            else:
                f.write('WMAP analysis not performed or failed\n\n')
                
            if planck_results:
                f.write('Planck Number of Transitions: {}\n'.format(planck_results['n_transitions']))
                f.write('Planck P-value: {:.6f}\n'.format(planck_results["p_value"]))
                f.write('Planck Phi-Optimality: {:.6f}\n'.format(planck_results['phi_optimality']))
                f.write('Planck Significant: {}\n'.format(planck_results['significant']))
                f.write('Planck Golden Ratio Alignment: {:.6f}\n\n'.format(planck_results['alignment_score']))
            else:
                f.write('Planck analysis not performed or failed\n\n')
                
            # Add comparison if both analyses were performed
            if wmap_results and planck_results:
                f.write('Comparison\n')
                f.write('-' * 50 + '\n')
                f.write('WMAP vs Planck Transitions: {} vs {}\n'.format(
                    wmap_results['n_transitions'], planck_results['n_transitions']))
                f.write('WMAP vs Planck P-value: {:.6f} vs {:.6f}\n'.format(
                    wmap_results['p_value'], planck_results['p_value']))
                f.write('WMAP vs Planck Phi-Optimality: {:.6f} vs {:.6f}\n'.format(
                    wmap_results['phi_optimality'], planck_results['phi_optimality']))
                
            # Add execution information
            total_time = (datetime.now() - start_time).total_seconds()
            f.write('\nExecution Information\n')
            f.write('-' * 50 + '\n')
            f.write('Total execution time: {:.2f} seconds\n'.format(total_time))
            f.write('Number of simulations: {}\n'.format(args.n_simulations))
            f.write('Window size: {}\n'.format(args.window_size))
            f.write('Initial clusters: {}\n'.format(args.n_clusters))
            f.write('Smoothing applied: {}\n'.format(args.smooth))
            f.write('Detrending applied: {}\n'.format(args.detrend))
            
        print("Results written to: {}".format(os.path.join(output_dir, 'scale_transition_results.txt')))
    except Exception as e:
        print("Error writing results to file: {}".format(str(e)))
    
    # Print final execution time
    total_time = (datetime.now() - start_time).total_seconds()
    print("\nTotal execution time: {:.2f} seconds".format(total_time))
    print("Results saved to: {}".format(output_dir))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
