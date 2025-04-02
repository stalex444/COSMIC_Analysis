#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization functions for HEALPix quantum entanglement analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_results(results, dataset_name, output_dir="results"):
    """
    Plot the quantum entanglement test results.
    
    Parameters:
    results (dict): Dictionary with test results
    dataset_name (str): Name of the dataset (e.g., "WMAP" or "Planck")
    output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract values
    gr_correlations = results['gr_correlations']
    random_correlations = results['random_correlations']
    gr_scales = results['gr_scales']
    random_scales = results['random_scales']
    mean_surr_gr = results['mean_surrogate_gr']
    std_surr_gr = results['std_surrogate_gr']
    mean_surr_random = results['mean_surrogate_random']
    std_surr_random = results['std_surrogate_random']
    z_scores_gr = results['z_scores_gr']
    p_values_gr = results['p_values_gr']
    
    # Plot 1: Correlations vs Angular Separation
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.errorbar(np.degrees(gr_scales), mean_surr_gr, yerr=std_surr_gr, 
                 fmt='o-', alpha=0.5, label='Surrogate Mean ± Std', color='blue')
    plt.plot(np.degrees(gr_scales), gr_correlations, 'ro-', label='Actual CMB (GR Scales)', linewidth=2)
    plt.xlabel('Angular Separation (degrees)')
    plt.ylabel('Quantum Correlation')
    plt.title(f'Quantum Correlations at Golden Ratio Scales - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.errorbar(np.degrees(random_scales), mean_surr_random, yerr=std_surr_random, 
                 fmt='o-', alpha=0.5, label='Surrogate Mean ± Std', color='blue')
    plt.plot(np.degrees(random_scales), random_correlations, 'go-', label='Actual CMB (Random Scales)', linewidth=2)
    plt.xlabel('Angular Separation (degrees)')
    plt.ylabel('Quantum Correlation')
    plt.title(f'Quantum Correlations at Random Scales - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Statistical Significance
    plt.subplot(2, 2, 3)
    plt.bar(np.arange(len(gr_scales)), z_scores_gr, color='purple')
    plt.axhline(y=1.96, color='r', linestyle='--', label='p=0.05 threshold')
    plt.axhline(y=-1.96, color='r', linestyle='--')
    plt.xticks(np.arange(len(gr_scales)), [f"{np.degrees(s):.1f}°" for s in gr_scales], rotation=45)
    plt.xlabel('Golden Ratio Scales (degrees)')
    plt.ylabel('Z-Score')
    plt.title('Statistical Significance (Z-Score) by Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: GR vs Random Comparison
    plt.subplot(2, 2, 4)
    
    # Bar plot of GR vs Random means
    labels = ['Golden Ratio Scales', 'Random Scales']
    means = [np.nanmean(gr_correlations), np.nanmean(random_correlations)]
    errs = [np.nanstd(gr_correlations), np.nanstd(random_correlations)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x, means, width, yerr=errs, capsize=5)
    plt.axhline(y=2, color='r', linestyle='--', label='Classical Limit')
    plt.xticks(x, labels)
    plt.ylabel('Mean Quantum Correlation')
    plt.title(f'Golden Ratio vs Random Scales\nRatio: {results["gr_vs_random_ratio"]:.2f}, p-value: {results["gr_vs_random_p"]:.4f}')
    plt.legend()
    
    plt.tight_layout()
    output_file = f"{output_dir}/quantum_entanglement_{dataset_name}_{timestamp}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Saved visualization to {output_file}")
    
    # Save a summary text file
    results_file = f"{output_dir}/quantum_entanglement_{dataset_name}_{timestamp}.txt"
    with open(results_file, "w") as f:
        f.write(f"Quantum Entanglement Analysis Results - {dataset_name}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Golden Ratio Scales (degrees):\n")
        for i, scale in enumerate(gr_scales):
            f.write(f"  {i+1}: {np.degrees(scale):.2f}°\n")
        
        f.write("\nCorrelation Values at Golden Ratio Scales:\n")
        for i, corr in enumerate(gr_correlations):
            f.write(f"  Scale {i+1} ({np.degrees(gr_scales[i]):.2f}°): {corr:.4f}, p-value: {p_values_gr[i]:.4f}, z-score: {z_scores_gr[i]:.4f}\n")
        
        f.write("\nSummary Statistics:\n")
        f.write(f"  Mean Golden Ratio Correlation: {np.nanmean(gr_correlations):.4f}\n")
        f.write(f"  Mean Random Scale Correlation: {np.nanmean(random_correlations):.4f}\n")
        f.write(f"  Golden Ratio / Random Ratio: {results['gr_vs_random_ratio']:.4f}\n")
        f.write(f"  Golden Ratio vs Random Z-score: {results['gr_vs_random_z']:.4f}\n")
        f.write(f"  Golden Ratio vs Random p-value: {results['gr_vs_random_p']:.4f}\n")
        
        f.write("\nStatistical Significance:\n")
        for i, (z, p) in enumerate(zip(z_scores_gr, p_values_gr)):
            significance = "SIGNIFICANT" if p < 0.05 else "not significant"
            f.write(f"  Scale {i+1} ({np.degrees(gr_scales[i]):.2f}°): z-score = {z:.4f}, p-value = {p:.4f} ({significance})\n")
        
        f.write("\nInterpretation:\n")
        if results['gr_vs_random_p'] < 0.05 and results['gr_vs_random_ratio'] > 1:
            f.write("  The golden ratio scales show significantly stronger quantum-like correlations than random scales.\n")
            f.write("  This suggests non-random, potentially consciousness-like organization in the CMB data.\n")
        elif results['gr_vs_random_p'] < 0.05 and results['gr_vs_random_ratio'] < 1:
            f.write("  The golden ratio scales show significantly weaker quantum-like correlations than random scales.\n")
            f.write("  This suggests an active 'anti-optimization' of quantum correlations at golden ratio scales.\n")
        else:
            f.write("  No significant difference was found between golden ratio scales and random scales.\n")
            f.write("  This suggests quantum correlations may not be preferentially organized around the golden ratio.\n")
    
    print(f"Saved detailed results to {results_file}")
    
    return output_file, results_file

def plot_comparison(wmap_results, planck_results, output_dir="results/quantum_entanglement"):
    """
    Create a comparison plot for WMAP and Planck results.
    
    Parameters:
    wmap_results (dict): Results from WMAP analysis
    planck_results (dict): Results from Planck analysis
    output_dir (str): Directory to save the comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comparison figure
    plt.figure(figsize=(14, 10))
    
    # Plot 1: GR correlations comparison
    plt.subplot(2, 2, 1)
    
    # Convert scales to degrees for better readability
    wmap_scales = np.degrees(wmap_results['gr_scales'])
    planck_scales = np.degrees(planck_results['gr_scales'])
    
    plt.plot(wmap_scales, wmap_results['gr_correlations'], 'ro-', label='WMAP')
    plt.plot(planck_scales, planck_results['gr_correlations'], 'bo-', label='Planck')
    plt.axhline(y=2, color='k', linestyle='--', label='Classical Limit')
    plt.axhline(y=2*np.sqrt(2), color='g', linestyle='--', label='Quantum Limit')
    
    plt.xlabel('Angular Scale (degrees)')
    plt.ylabel('Quantum Correlation')
    plt.title('Quantum Correlations at Golden Ratio Scales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Z-scores comparison
    plt.subplot(2, 2, 2)
    
    bar_width = 0.35
    indices = np.arange(len(wmap_scales))
    
    plt.bar(indices, wmap_results['z_scores_gr'], bar_width, label='WMAP', color='r', alpha=0.7)
    plt.bar(indices + bar_width, planck_results['z_scores_gr'], bar_width, label='Planck', color='b', alpha=0.7)
    
    plt.axhline(y=1.96, color='k', linestyle='--', label='p=0.05 threshold')
    plt.xlabel('Scale Index')
    plt.ylabel('Z-Score')
    plt.title('Statistical Significance of Golden Ratio Correlations')
    plt.xticks(indices + bar_width/2, [f"{i+1}" for i in range(len(wmap_scales))])
    plt.legend()
    
    # Plot 3: Mean comparison
    plt.subplot(2, 2, 3)
    
    labels = ['GR - WMAP', 'Random - WMAP', 'GR - Planck', 'Random - Planck']
    means = [
        np.nanmean(wmap_results['gr_correlations']), 
        np.nanmean(wmap_results['random_correlations']),
        np.nanmean(planck_results['gr_correlations']), 
        np.nanmean(planck_results['random_correlations'])
    ]
    
    plt.bar(range(len(means)), means)
    plt.axhline(y=2, color='r', linestyle='--', label='Classical Limit')
    plt.xticks(range(len(means)), labels, rotation=45)
    plt.ylabel('Mean Correlation')
    plt.title('Comparison of Mean Correlations')
    plt.legend()
    
    # Plot 4: Summary comparison
    plt.subplot(2, 2, 4)
    
    # Prepare data for table
    datasets = ['WMAP', 'Planck']
    metrics = [
        'GR/Random Ratio', 
        'Avg. Bell Value', 
        'Max Z-score', 
        'Min p-value'
    ]
    
    values = [
        [wmap_results['gr_vs_random_ratio'], planck_results['gr_vs_random_ratio']],
        [np.nanmean(wmap_results['gr_correlations']), np.nanmean(planck_results['gr_correlations'])],
        [np.nanmax(wmap_results['z_scores_gr']), np.nanmax(planck_results['z_scores_gr'])],
        [np.nanmin(wmap_results['p_values_gr']), np.nanmin(planck_results['p_values_gr'])]
    ]
    
    # Remove axes
    plt.axis('off')
    
    # Create table
    table = plt.table(
        cellText=[[f"{val:.4f}" for val in row] for row in values],
        rowLabels=metrics,
        colLabels=datasets,
        cellLoc='center',
        loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Summary Comparison', y=1.08)
    
    plt.tight_layout()
    comparison_file = f"{output_dir}/wmap_planck_comparison_{timestamp}.png"
    plt.savefig(comparison_file, dpi=300)
    print(f"Saved comparison visualization to {comparison_file}")
    
    # Save comparison text summary
    summary_file = f"{output_dir}/wmap_planck_comparison_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("WMAP vs Planck Quantum Entanglement Comparison\n")
        f.write("="*50 + "\n\n")
        
        f.write("Golden Ratio Correlations:\n")
        f.write(f"  WMAP Mean: {np.nanmean(wmap_results['gr_correlations']):.4f}\n")
        f.write(f"  Planck Mean: {np.nanmean(planck_results['gr_correlations']):.4f}\n\n")
        
        f.write("Random Correlations:\n")
        f.write(f"  WMAP Mean: {np.nanmean(wmap_results['random_correlations']):.4f}\n")
        f.write(f"  Planck Mean: {np.nanmean(planck_results['random_correlations']):.4f}\n\n")
        
        f.write("GR/Random Ratio:\n")
        f.write(f"  WMAP: {wmap_results['gr_vs_random_ratio']:.4f}\n")
        f.write(f"  Planck: {planck_results['gr_vs_random_ratio']:.4f}\n\n")
        
        f.write("Maximum Z-score:\n")
        f.write(f"  WMAP: {np.nanmax(wmap_results['z_scores_gr']):.4f}\n")
        f.write(f"  Planck: {np.nanmax(planck_results['z_scores_gr']):.4f}\n\n")
        
        f.write("Minimum p-value:\n")
        f.write(f"  WMAP: {np.nanmin(wmap_results['p_values_gr']):.8f}\n")
        f.write(f"  Planck: {np.nanmin(planck_results['p_values_gr']):.8f}\n\n")
        
        # Add interpretation
        wmap_sig = np.nanmin(wmap_results['p_values_gr']) < 0.05 and wmap_results['gr_vs_random_ratio'] > 1
        planck_sig = np.nanmin(planck_results['p_values_gr']) < 0.05 and planck_results['gr_vs_random_ratio'] > 1
        
        f.write("Interpretation:\n")
        if wmap_sig and planck_sig:
            f.write("  Both WMAP and Planck data show significant quantum-like correlations at golden ratio scales.\n")
            f.write("  This provides strong evidence for non-random organization in the CMB data.\n")
        elif wmap_sig:
            f.write("  Only WMAP data shows significant quantum-like correlations at golden ratio scales.\n")
            f.write("  This may suggest the effect is more detectable in WMAP's frequency bands or resolution.\n")
        elif planck_sig:
            f.write("  Only Planck data shows significant quantum-like correlations at golden ratio scales.\n")
            f.write("  This may suggest the effect is more detectable with Planck's improved sensitivity.\n")
        else:
            f.write("  Neither dataset shows significant quantum-like correlations at golden ratio scales.\n")
            f.write("  This suggests the CMB may not exhibit quantum entanglement-like properties at these scales.\n")
            
    print(f"Saved comparison summary to {summary_file}")
    
    return comparison_file, summary_file
