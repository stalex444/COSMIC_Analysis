#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Add Python 2.7 compatibility
from __future__ import division, print_function

import os
import re
import sys
import glob
import json
import argparse
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (12, 8)

def find_result_files(base_dir, dataset_name):
    """
    Find scale transition result files for a specific dataset.
    
    Args:
        base_dir (str): Base directory to search in
        dataset_name (str): Name of the dataset ('wmap' or 'planck')
        
    Returns:
        list: List of file paths matching the pattern
    """
    pattern = os.path.join(base_dir, '**', dataset_name, f'{dataset_name}_scale_transition*.txt')
    return glob.glob(pattern, recursive=True)

def extract_transition_points(file_path):
    """
    Extract scale transition points from a result file.
    
    Args:
        file_path (str): Path to the result file
        
    Returns:
        tuple: (transition_points, p_value, phi_optimality, alignment_score, n_simulations)
    """
    transition_points = []
    p_value = None
    phi_optimality = None
    alignment_score = None
    n_simulations = None
    significant = False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # Extract number of simulations
            sim_match = re.search(r'Number of simulations: (\d+)', content)
            if sim_match:
                n_simulations = int(sim_match.group(1))
            
            # Extract p-value
            p_match = re.search(r'P-value: ([\d\.]+)', content)
            if p_match:
                p_value = float(p_match.group(1))
            
            # Extract phi-optimality
            phi_match = re.search(r'Phi-Optimality: ([\d\.]+)', content)
            if phi_match:
                phi_optimality = float(phi_match.group(1))
            
            # Extract alignment score
            align_match = re.search(r'Golden Ratio Alignment Score: ([\d\.]+)', content)
            if align_match:
                alignment_score = float(align_match.group(1))
            
            # Extract significance
            sig_match = re.search(r'Significant: (True|False)', content)
            if sig_match:
                significant = sig_match.group(1) == 'True'
            
            # Extract transition points
            transition_pattern = r'Transition (\d+): ℓ = ([\d\.]+)'
            for match in re.finditer(transition_pattern, content):
                transition_points.append(float(match.group(2)))
    
    except Exception as e:
        print(f"Error extracting data from {file_path}: {str(e)}")
        traceback.print_exc()
    
    return transition_points, p_value, phi_optimality, alignment_score, n_simulations, significant

def load_results(result_dir):
    """
    Load scale transition results from the specified directory.
    
    Args:
        result_dir (str): Directory containing the results
        
    Returns:
        tuple: (wmap_results, planck_results)
    """
    # Find result files
    wmap_files = find_result_files(result_dir, 'wmap')
    planck_files = find_result_files(result_dir, 'planck')
    
    # Sort files by modification time (newest first)
    wmap_files.sort(key=os.path.getmtime, reverse=True)
    planck_files.sort(key=os.path.getmtime, reverse=True)
    
    wmap_results = None
    planck_results = None
    
    # Extract data from the newest files
    if wmap_files:
        wmap_transitions, wmap_p, wmap_phi, wmap_align, wmap_sims, wmap_sig = extract_transition_points(wmap_files[0])
        wmap_results = {
            'transitions': wmap_transitions,
            'p_value': wmap_p,
            'phi_optimality': wmap_phi,
            'alignment_score': wmap_align,
            'n_simulations': wmap_sims,
            'significant': wmap_sig,
            'file_path': wmap_files[0]
        }
    
    if planck_files:
        planck_transitions, planck_p, planck_phi, planck_align, planck_sims, planck_sig = extract_transition_points(planck_files[0])
        planck_results = {
            'transitions': planck_transitions,
            'p_value': planck_p,
            'phi_optimality': planck_phi,
            'alignment_score': planck_align,
            'n_simulations': planck_sims,
            'significant': planck_sig,
            'file_path': planck_files[0]
        }
    
    return wmap_results, planck_results

def analyze_transition_distribution(transitions):
    """
    Analyze the distribution of scale transition points.
    
    Args:
        transitions (list): List of transition points
        
    Returns:
        dict: Distribution statistics
    """
    if not transitions:
        return None
    
    transitions = np.array(transitions)
    
    # Basic statistics
    stats = {
        'count': len(transitions),
        'min': np.min(transitions),
        'max': np.max(transitions),
        'mean': np.mean(transitions),
        'median': np.median(transitions),
        'std': np.std(transitions),
        'skewness': stats.skew(transitions),
        'kurtosis': stats.kurtosis(transitions)
    }
    
    # Calculate differences between adjacent transitions
    if len(transitions) > 1:
        sorted_transitions = np.sort(transitions)
        diffs = np.diff(sorted_transitions)
        stats['diff_mean'] = np.mean(diffs)
        stats['diff_median'] = np.median(diffs)
        stats['diff_std'] = np.std(diffs)
        stats['diff_min'] = np.min(diffs)
        stats['diff_max'] = np.max(diffs)
    
    return stats

def analyze_golden_ratio_alignment(transitions):
    """
    Analyze the alignment of transition points with the golden ratio.
    
    Args:
        transitions (list): List of transition points
        
    Returns:
        dict: Golden ratio alignment statistics
    """
    if not transitions or len(transitions) < 2:
        return None
    
    # Sort transitions
    sorted_transitions = np.sort(np.array(transitions))
    
    # Calculate ratios between adjacent transitions
    ratios = []
    for i in range(len(sorted_transitions) - 1):
        if sorted_transitions[i] > 0:  # Avoid division by zero
            ratio = sorted_transitions[i+1] / sorted_transitions[i]
            ratios.append(ratio)
    
    ratios = np.array(ratios)
    
    # Calculate deviation from golden ratio
    deviations = np.abs(ratios - PHI)
    
    # Find ratios close to golden ratio (within 5%)
    phi_threshold = 0.05 * PHI
    close_to_phi = deviations < phi_threshold
    
    # Calculate statistics
    stats = {
        'count': len(ratios),
        'mean_ratio': np.mean(ratios),
        'median_ratio': np.median(ratios),
        'std_ratio': np.std(ratios),
        'min_ratio': np.min(ratios),
        'max_ratio': np.max(ratios),
        'mean_deviation': np.mean(deviations),
        'median_deviation': np.median(deviations),
        'std_deviation': np.std(deviations),
        'phi_aligned_count': np.sum(close_to_phi),
        'phi_aligned_percentage': 100 * np.sum(close_to_phi) / len(ratios) if len(ratios) > 0 else 0
    }
    
    # Find sequences of consecutive transitions with golden ratio relationships
    phi_sequences = []
    current_sequence = []
    
    for i in range(len(ratios)):
        if deviations[i] < phi_threshold:
            if not current_sequence:
                current_sequence = [sorted_transitions[i], sorted_transitions[i+1]]
            else:
                current_sequence.append(sorted_transitions[i+1])
        else:
            if len(current_sequence) > 2:
                phi_sequences.append(current_sequence)
            current_sequence = []
    
    # Add the last sequence if it exists
    if len(current_sequence) > 2:
        phi_sequences.append(current_sequence)
    
    stats['phi_sequences'] = phi_sequences
    stats['phi_sequences_count'] = len(phi_sequences)
    
    return stats

def identify_transition_clusters(transitions, max_distance=10):
    """
    Identify clusters of transition points.
    
    Args:
        transitions (list): List of transition points
        max_distance (float): Maximum distance between points in a cluster
        
    Returns:
        list: List of clusters, where each cluster is a list of transition points
    """
    if not transitions or len(transitions) < 2:
        return []
    
    # Sort transitions
    sorted_transitions = np.sort(np.array(transitions))
    
    # Calculate distance matrix
    distances = squareform(pdist(sorted_transitions.reshape(-1, 1)))
    
    # Perform hierarchical clustering
    Z = linkage(distances, method='single')
    
    # Cut the dendrogram to form clusters
    cluster_indices = fcluster(Z, max_distance, criterion='distance')
    
    # Group points by cluster
    clusters = {}
    for i, cluster_idx in enumerate(cluster_indices):
        if cluster_idx not in clusters:
            clusters[cluster_idx] = []
        clusters[cluster_idx].append(sorted_transitions[i])
    
    # Convert to list of clusters
    cluster_list = [cluster for cluster in clusters.values()]
    
    return cluster_list

def plot_transition_distribution(transitions, title, output_path):
    """
    Plot the distribution of scale transition points.
    
    Args:
        transitions (list): List of transition points
        title (str): Plot title
        output_path (str): Path to save the plot
    """
    if not transitions:
        return
    
    plt.figure(figsize=(14, 8))
    
    # Histogram
    sns.histplot(transitions, bins=30, kde=True)
    
    plt.title(title)
    plt.xlabel('Multipole Moment (ℓ)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for min, max, mean, median
    plt.axvline(np.min(transitions), color='r', linestyle='--', alpha=0.7, label=f'Min: {np.min(transitions):.1f}')
    plt.axvline(np.max(transitions), color='g', linestyle='--', alpha=0.7, label=f'Max: {np.max(transitions):.1f}')
    plt.axvline(np.mean(transitions), color='b', linestyle='-', alpha=0.7, label=f'Mean: {np.mean(transitions):.1f}')
    plt.axvline(np.median(transitions), color='purple', linestyle='-', alpha=0.7, label=f'Median: {np.median(transitions):.1f}')
    
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_golden_ratio_analysis(transitions, title, output_path):
    """
    Plot the golden ratio analysis of transition points.
    
    Args:
        transitions (list): List of transition points
        title (str): Plot title
        output_path (str): Path to save the plot
    """
    if not transitions or len(transitions) < 2:
        return
    
    # Sort transitions
    sorted_transitions = np.sort(np.array(transitions))
    
    # Calculate ratios between adjacent transitions
    ratios = []
    for i in range(len(sorted_transitions) - 1):
        if sorted_transitions[i] > 0:  # Avoid division by zero
            ratio = sorted_transitions[i+1] / sorted_transitions[i]
            ratios.append(ratio)
    
    ratios = np.array(ratios)
    
    # Calculate deviation from golden ratio
    deviations = np.abs(ratios - PHI)
    
    plt.figure(figsize=(14, 10))
    
    # Create a 2x2 subplot grid
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Subplot 1: Ratio distribution
    ax1 = plt.subplot(gs[0, 0])
    sns.histplot(ratios, bins=20, kde=True, ax=ax1)
    ax1.axvline(PHI, color='r', linestyle='--', alpha=0.7, label=f'Golden Ratio (φ): {PHI:.4f}')
    ax1.axvline(np.mean(ratios), color='b', linestyle='-', alpha=0.7, label=f'Mean Ratio: {np.mean(ratios):.4f}')
    ax1.set_title('Distribution of Adjacent Transition Ratios')
    ax1.set_xlabel('Ratio (ℓ₂/ℓ₁)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Subplot 2: Deviation from golden ratio
    ax2 = plt.subplot(gs[0, 1])
    sns.histplot(deviations, bins=20, kde=True, ax=ax2)
    ax2.axvline(np.mean(deviations), color='b', linestyle='-', alpha=0.7, 
               label=f'Mean Deviation: {np.mean(deviations):.4f}')
    ax2.set_title('Deviation from Golden Ratio')
    ax2.set_xlabel('|Ratio - φ|')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Subplot 3: Scatter plot of adjacent multipoles
    ax3 = plt.subplot(gs[1, 0])
    ax3.scatter(sorted_transitions[:-1], sorted_transitions[1:], alpha=0.7)
    
    # Add golden ratio line
    x_min, x_max = ax3.get_xlim()
    ax3.plot([x_min, x_max], [x_min * PHI, x_max * PHI], 'r--', alpha=0.7, label=f'Golden Ratio (φ): {PHI:.4f}')
    
    ax3.set_title('Adjacent Transition Points')
    ax3.set_xlabel('Multipole Moment (ℓ₁)')
    ax3.set_ylabel('Multipole Moment (ℓ₂)')
    ax3.legend()
    
    # Subplot 4: Ratio vs. multipole
    ax4 = plt.subplot(gs[1, 1])
    ax4.scatter(sorted_transitions[:-1], ratios, alpha=0.7)
    ax4.axhline(PHI, color='r', linestyle='--', alpha=0.7, label=f'Golden Ratio (φ): {PHI:.4f}')
    ax4.set_title('Transition Ratio vs. Multipole')
    ax4.set_xlabel('Multipole Moment (ℓ₁)')
    ax4.set_ylabel('Ratio (ℓ₂/ℓ₁)')
    ax4.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_transition_comparison(wmap_transitions, planck_transitions, output_path):
    """
    Plot comparison of WMAP and Planck transition points.
    
    Args:
        wmap_transitions (list): List of WMAP transition points
        planck_transitions (list): List of Planck transition points
        output_path (str): Path to save the plot
    """
    if not wmap_transitions or not planck_transitions:
        return
    
    plt.figure(figsize=(14, 10))
    
    # Create a 2x2 subplot grid
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Subplot 1: Histogram comparison
    ax1 = plt.subplot(gs[0, 0])
    
    # Determine common bins
    all_transitions = np.concatenate([wmap_transitions, planck_transitions])
    bin_min = np.min(all_transitions)
    bin_max = np.max(all_transitions)
    bins = np.linspace(bin_min, bin_max, 30)
    
    # Plot histograms
    ax1.hist(wmap_transitions, bins=bins, alpha=0.5, label='WMAP')
    ax1.hist(planck_transitions, bins=bins, alpha=0.5, label='Planck')
    
    ax1.set_title('Distribution of Transition Points')
    ax1.set_xlabel('Multipole Moment (ℓ)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Subplot 2: Density comparison
    ax2 = plt.subplot(gs[0, 1])
    sns.kdeplot(wmap_transitions, label='WMAP', ax=ax2)
    sns.kdeplot(planck_transitions, label='Planck', ax=ax2)
    
    ax2.set_title('Density of Transition Points')
    ax2.set_xlabel('Multipole Moment (ℓ)')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    # Subplot 3: Box plot comparison
    ax3 = plt.subplot(gs[1, 0])
    data = [wmap_transitions, planck_transitions]
    ax3.boxplot(data, labels=['WMAP', 'Planck'])
    
    ax3.set_title('Distribution Statistics')
    ax3.set_ylabel('Multipole Moment (ℓ)')
    
    # Subplot 4: Cumulative distribution
    ax4 = plt.subplot(gs[1, 1])
    
    # Sort transitions
    sorted_wmap = np.sort(wmap_transitions)
    sorted_planck = np.sort(planck_transitions)
    
    # Calculate cumulative distributions
    wmap_y = np.arange(1, len(sorted_wmap) + 1) / len(sorted_wmap)
    planck_y = np.arange(1, len(sorted_planck) + 1) / len(sorted_planck)
    
    ax4.plot(sorted_wmap, wmap_y, label='WMAP')
    ax4.plot(sorted_planck, planck_y, label='Planck')
    
    ax4.set_title('Cumulative Distribution')
    ax4.set_xlabel('Multipole Moment (ℓ)')
    ax4.set_ylabel('Cumulative Probability')
    ax4.legend()
    
    plt.suptitle('Comparison of Scale Transitions: WMAP vs. Planck', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_analysis_report(wmap_results, planck_results, output_path):
    """
    Generate a comprehensive analysis report.
    
    Args:
        wmap_results (dict): WMAP analysis results
        planck_results (dict): Planck analysis results
        output_path (str): Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write('Scale Transition Analysis Report\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Analysis performed on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        # WMAP Analysis
        f.write('WMAP Scale Transition Analysis\n')
        f.write('-' * 50 + '\n\n')
        
        if wmap_results:
            wmap_transitions = wmap_results['transitions']
            wmap_dist_stats = analyze_transition_distribution(wmap_transitions)
            wmap_gr_stats = analyze_golden_ratio_alignment(wmap_transitions)
            
            f.write(f'Number of transitions: {len(wmap_transitions)}\n')
            f.write(f'P-value: {wmap_results["p_value"]:.6f}\n')
            f.write(f'Phi-optimality: {wmap_results["phi_optimality"]:.6f}\n')
            f.write(f'Alignment score: {wmap_results["alignment_score"]:.6f}\n')
            f.write(f'Significant: {wmap_results["significant"]}\n')
            f.write(f'Number of simulations: {wmap_results["n_simulations"]}\n\n')
            
            f.write('Distribution Statistics:\n')
            f.write(f'  Mean: {wmap_dist_stats["mean"]:.2f}\n')
            f.write(f'  Median: {wmap_dist_stats["median"]:.2f}\n')
            f.write(f'  Standard deviation: {wmap_dist_stats["std"]:.2f}\n')
            f.write(f'  Min: {wmap_dist_stats["min"]:.2f}\n')
            f.write(f'  Max: {wmap_dist_stats["max"]:.2f}\n')
            f.write(f'  Skewness: {wmap_dist_stats["skewness"]:.2f}\n')
            f.write(f'  Kurtosis: {wmap_dist_stats["kurtosis"]:.2f}\n\n')
            
            if wmap_gr_stats:
                f.write('Golden Ratio Analysis:\n')
                f.write(f'  Mean ratio: {wmap_gr_stats["mean_ratio"]:.4f}\n')
                f.write(f'  Median ratio: {wmap_gr_stats["median_ratio"]:.4f}\n')
                f.write(f'  Mean deviation from φ: {wmap_gr_stats["mean_deviation"]:.4f}\n')
                f.write(f'  Pairs aligned with φ: {wmap_gr_stats["phi_aligned_count"]} ({wmap_gr_stats["phi_aligned_percentage"]:.2f}%)\n')
                f.write(f'  Number of φ sequences: {wmap_gr_stats["phi_sequences_count"]}\n\n')
        else:
            f.write('No WMAP results available.\n\n')
        
        # Planck Analysis
        f.write('Planck Scale Transition Analysis\n')
        f.write('-' * 50 + '\n\n')
        
        if planck_results:
            planck_transitions = planck_results['transitions']
            planck_dist_stats = analyze_transition_distribution(planck_transitions)
            planck_gr_stats = analyze_golden_ratio_alignment(planck_transitions)
            
            f.write(f'Number of transitions: {len(planck_transitions)}\n')
            f.write(f'P-value: {planck_results["p_value"]:.6f}\n')
            f.write(f'Phi-optimality: {planck_results["phi_optimality"]:.6f}\n')
            f.write(f'Alignment score: {planck_results["alignment_score"]:.6f}\n')
            f.write(f'Significant: {planck_results["significant"]}\n')
            f.write(f'Number of simulations: {planck_results["n_simulations"]}\n\n')
            
            f.write('Distribution Statistics:\n')
            f.write(f'  Mean: {planck_dist_stats["mean"]:.2f}\n')
            f.write(f'  Median: {planck_dist_stats["median"]:.2f}\n')
            f.write(f'  Standard deviation: {planck_dist_stats["std"]:.2f}\n')
            f.write(f'  Min: {planck_dist_stats["min"]:.2f}\n')
            f.write(f'  Max: {planck_dist_stats["max"]:.2f}\n')
            f.write(f'  Skewness: {planck_dist_stats["skewness"]:.2f}\n')
            f.write(f'  Kurtosis: {planck_dist_stats["kurtosis"]:.2f}\n\n')
            
            if planck_gr_stats:
                f.write('Golden Ratio Analysis:\n')
                f.write(f'  Mean ratio: {planck_gr_stats["mean_ratio"]:.4f}\n')
                f.write(f'  Median ratio: {planck_gr_stats["median_ratio"]:.4f}\n')
                f.write(f'  Mean deviation from φ: {planck_gr_stats["mean_deviation"]:.4f}\n')
                f.write(f'  Pairs aligned with φ: {planck_gr_stats["phi_aligned_count"]} ({planck_gr_stats["phi_aligned_percentage"]:.2f}%)\n')
                f.write(f'  Number of φ sequences: {planck_gr_stats["phi_sequences_count"]}\n\n')
        else:
            f.write('No Planck results available.\n\n')
        
        # Comparative Analysis
        f.write('Comparative Analysis: WMAP vs. Planck\n')
        f.write('-' * 50 + '\n\n')
        
        if wmap_results and planck_results:
            wmap_transitions = wmap_results['transitions']
            planck_transitions = planck_results['transitions']
            
            f.write(f'Difference in number of transitions: {len(planck_transitions) - len(wmap_transitions)}\n')
            f.write(f'Ratio of transition counts (Planck/WMAP): {len(planck_transitions) / len(wmap_transitions):.2f}\n')
            f.write(f'Difference in phi-optimality: {planck_results["phi_optimality"] - wmap_results["phi_optimality"]:.6f}\n')
            f.write(f'Difference in alignment score: {planck_results["alignment_score"] - wmap_results["alignment_score"]:.6f}\n\n')
            
            # Calculate overlap in transition points
            wmap_set = set([int(x) for x in wmap_transitions])
            planck_set = set([int(x) for x in planck_transitions])
            overlap = wmap_set.intersection(planck_set)
            
            f.write(f'Number of common transition points: {len(overlap)}\n')
            f.write(f'Percentage of WMAP transitions in Planck: {100 * len(overlap) / len(wmap_set):.2f}%\n')
            f.write(f'Percentage of Planck transitions in WMAP: {100 * len(overlap) / len(planck_set):.2f}%\n\n')
            
            f.write('Common transition points:\n')
            for point in sorted(overlap):
                f.write(f'  ℓ = {point}\n')
            f.write('\n')
            
            # Interpretation
            f.write('Interpretation:\n')
            if planck_results["significant"] and not wmap_results["significant"]:
                f.write('  The Planck data shows significant scale transitions while the WMAP data does not.\n')
                f.write('  This suggests that the higher resolution of Planck data reveals structural features\n')
                f.write('  that are not detectable in the lower-resolution WMAP data.\n\n')
            elif wmap_results["significant"] and not planck_results["significant"]:
                f.write('  The WMAP data shows significant scale transitions while the Planck data does not.\n')
                f.write('  This unexpected result may indicate that the scale transitions in WMAP data\n')
                f.write('  could be artifacts or that the Planck data processing has removed certain features.\n\n')
            elif wmap_results["significant"] and planck_results["significant"]:
                f.write('  Both datasets show significant scale transitions, confirming the presence\n')
                f.write('  of distinct organizational regimes at different scales in the CMB power spectrum.\n')
                f.write('  The higher number of transitions in Planck data reflects its higher resolution.\n\n')
            else:
                f.write('  Neither dataset shows significant scale transitions, suggesting that\n')
                f.write('  the CMB power spectrum does not exhibit distinct organizational regimes\n')
                f.write('  beyond what would be expected by chance.\n\n')
            
            # Golden ratio interpretation
            wmap_gr_stats = analyze_golden_ratio_alignment(wmap_transitions)
            planck_gr_stats = analyze_golden_ratio_alignment(planck_transitions)
            
            if wmap_gr_stats and planck_gr_stats:
                f.write('Golden Ratio Interpretation:\n')
                if wmap_gr_stats["phi_aligned_percentage"] > 10 or planck_gr_stats["phi_aligned_percentage"] > 10:
                    f.write('  There is evidence of golden ratio alignment in the scale transitions,\n')
                    f.write('  particularly in the {}.\n'.format(
                        'Planck data' if planck_gr_stats["phi_aligned_percentage"] > wmap_gr_stats["phi_aligned_percentage"] else 'WMAP data'
                    ))
                    f.write('  This suggests an underlying mathematical structure that may be\n')
                    f.write('  related to the golden ratio in the organization of the CMB.\n\n')
                else:
                    f.write('  There is limited evidence of golden ratio alignment in the scale transitions.\n')
                    f.write('  The observed alignments are likely due to chance rather than\n')
                    f.write('  an underlying mathematical structure related to the golden ratio.\n\n')
        else:
            f.write('Cannot perform comparative analysis because results are missing for one or both datasets.\n\n')
        
        # Conclusion
        f.write('Conclusion\n')
        f.write('-' * 50 + '\n\n')
        
        if wmap_results and planck_results:
            f.write('The scale transition analysis reveals ')
            if planck_results["significant"]:
                f.write('significant structural organization in the Planck CMB power spectrum, ')
                if wmap_results["significant"]:
                    f.write('which is also detected in the WMAP data but with fewer transition points. ')
                else:
                    f.write('which is not detected in the lower-resolution WMAP data. ')
            else:
                if wmap_results["significant"]:
                    f.write('significant structural organization in the WMAP CMB power spectrum, ')
                    f.write('which is unexpectedly not detected in the higher-resolution Planck data. ')
                else:
                    f.write('no significant structural organization in either the WMAP or Planck CMB power spectra. ')
            
            # Golden ratio conclusion
            wmap_gr_stats = analyze_golden_ratio_alignment(wmap_transitions)
            planck_gr_stats = analyze_golden_ratio_alignment(planck_transitions)
            
            if wmap_gr_stats and planck_gr_stats:
                if (wmap_gr_stats["phi_aligned_percentage"] > 10 or 
                    planck_gr_stats["phi_aligned_percentage"] > 10):
                    f.write('There is evidence of golden ratio alignment in the transition points, ')
                    f.write('suggesting a possible mathematical structure underlying the CMB organization. ')
                else:
                    f.write('There is limited evidence of golden ratio alignment in the transition points. ')
            
            f.write('\n\nThis analysis complements the golden ratio coherence findings, providing a more ')
            f.write('comprehensive understanding of the structural organization of the cosmic microwave background radiation.')
        else:
            f.write('Cannot draw conclusions because results are missing for one or both datasets.')

def main():
    """
    Main function to run the scale transition analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze scale transition results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing the scale transition results')
    parser.add_argument('--output-dir', type=str, default="../results/scale_transition_analysis",
                        help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Loading results from: {args.results_dir}")
    wmap_results, planck_results = load_results(args.results_dir)
    
    if not wmap_results and not planck_results:
        print("No results found. Make sure the results directory is correct.")
        return
    
    print("Generating analysis report...")
    report_path = os.path.join(args.output_dir, 'scale_transition_analysis_report.txt')
    generate_analysis_report(wmap_results, planck_results, report_path)
    print(f"Report saved to: {report_path}")
    
    # Generate plots if results are available
    if wmap_results:
        wmap_transitions = wmap_results['transitions']
        print(f"Generating WMAP plots ({len(wmap_transitions)} transition points)...")
        
        # Distribution plot
        wmap_dist_path = os.path.join(args.output_dir, 'wmap_transition_distribution.png')
        plot_transition_distribution(wmap_transitions, 'WMAP Scale Transition Distribution', wmap_dist_path)
        print(f"WMAP distribution plot saved to: {wmap_dist_path}")
        
        # Golden ratio analysis plot
        wmap_gr_path = os.path.join(args.output_dir, 'wmap_golden_ratio_analysis.png')
        plot_golden_ratio_analysis(wmap_transitions, 'WMAP Golden Ratio Analysis', wmap_gr_path)
        print(f"WMAP golden ratio analysis plot saved to: {wmap_gr_path}")
    
    if planck_results:
        planck_transitions = planck_results['transitions']
        print(f"Generating Planck plots ({len(planck_transitions)} transition points)...")
        
        # Distribution plot
        planck_dist_path = os.path.join(args.output_dir, 'planck_transition_distribution.png')
        plot_transition_distribution(planck_transitions, 'Planck Scale Transition Distribution', planck_dist_path)
        print(f"Planck distribution plot saved to: {planck_dist_path}")
        
        # Golden ratio analysis plot
        planck_gr_path = os.path.join(args.output_dir, 'planck_golden_ratio_analysis.png')
        plot_golden_ratio_analysis(planck_transitions, 'Planck Golden Ratio Analysis', planck_gr_path)
        print(f"Planck golden ratio analysis plot saved to: {planck_gr_path}")
    
    # Generate comparison plots if both results are available
    if wmap_results and planck_results:
        print("Generating comparison plots...")
        comparison_path = os.path.join(args.output_dir, 'wmap_planck_comparison.png')
        plot_transition_comparison(wmap_results['transitions'], planck_results['transitions'], comparison_path)
        print(f"Comparison plot saved to: {comparison_path}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
