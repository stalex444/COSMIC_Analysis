#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis of Golden Ratio Pairs from GR-Specific Coherence Test
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns

def extract_pairs_from_file(file_path):
    """Extract golden ratio pairs and coherence values from results file."""
    pairs = []
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Try different regex patterns to match the format
        patterns = [
            r"Pair \d+: l1 = (\d+\.\d+|\d+), l2 = (\d+\.\d+|\d+), Coherence = (\d+\.\d+)",
            r"Pair \d+: ell1 = (\d+\.\d+|\d+), ell2 = (\d+\.\d+|\d+), Coherence = (\d+\.\d+)",
            r"Pair \d+:\s+l1 = (\d+\.\d+|\d+),\s+l2 = (\d+\.\d+|\d+),\s+Coherence = (\d+\.\d+)",
            r"Pair \d+:\s+ell1 = (\d+\.\d+|\d+),\s+ell2 = (\d+\.\d+|\d+),\s+Coherence = (\d+\.\d+)"
        ]
        
        matches = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                break
        
        if not matches:
            print("Warning: Could not extract pairs from {}".format(file_path))
            print("Please check the file format and update the regex patterns.")
            return pairs
        
        for match in matches:
            ell1 = float(match[0])
            ell2 = float(match[1])
            coherence = float(match[2])
            pairs.append((ell1, ell2, coherence))
    
    return pairs

def analyze_pairs(pairs, dataset_name):
    """Analyze the golden ratio pairs."""
    ell1_values = [p[0] for p in pairs]
    ell2_values = [p[1] for p in pairs]
    coherence_values = [p[2] for p in pairs]
    
    # Calculate ratios
    ratios = [ell2/ell1 for ell1, ell2, _ in pairs]
    
    # Calculate statistics
    mean_coherence = np.mean(coherence_values)
    median_coherence = np.median(coherence_values)
    std_coherence = np.std(coherence_values)
    min_coherence = np.min(coherence_values)
    max_coherence = np.max(coherence_values)
    
    # Count highly coherent pairs (>0.9)
    high_coherence_count = sum(1 for c in coherence_values if c > 0.9)
    high_coherence_percentage = high_coherence_count / len(coherence_values) * 100
    
    # Calculate mean ratio and deviation from golden ratio
    mean_ratio = np.mean(ratios)
    golden_ratio = (1 + np.sqrt(5)) / 2  # Approximately 1.618
    ratio_deviation = np.abs(mean_ratio - golden_ratio)
    
    # Check for correlation between multipole value and coherence
    corr_ell1_coherence, p_ell1 = pearsonr(ell1_values, coherence_values)
    corr_ell2_coherence, p_ell2 = pearsonr(ell2_values, coherence_values)
    
    # Print results
    print("\n{} Golden Ratio Pairs Analysis".format(dataset_name))
    print("="*50)
    print("Number of pairs: {}".format(len(pairs)))
    print("Mean coherence: {:.6f}".format(mean_coherence))
    print("Median coherence: {:.6f}".format(median_coherence))
    print("Standard deviation: {:.6f}".format(std_coherence))
    print("Min coherence: {:.6f}".format(min_coherence))
    print("Max coherence: {:.6f}".format(max_coherence))
    print("Pairs with coherence > 0.9: {} ({:.2f}%)".format(high_coherence_count, high_coherence_percentage))
    print("Mean ratio (l2/l1): {:.6f}".format(mean_ratio))
    print("Deviation from golden ratio: {:.6f}".format(ratio_deviation))
    print("Correlation between l1 and coherence: {:.6f} (p={:.6f})".format(corr_ell1_coherence, p_ell1))
    print("Correlation between l2 and coherence: {:.6f} (p={:.6f})".format(corr_ell2_coherence, p_ell2))
    
    return {
        'ell1_values': ell1_values,
        'ell2_values': ell2_values,
        'coherence_values': coherence_values,
        'ratios': ratios,
        'mean_coherence': mean_coherence,
        'median_coherence': median_coherence,
        'std_coherence': std_coherence,
        'high_coherence_count': high_coherence_count,
        'high_coherence_percentage': high_coherence_percentage,
        'mean_ratio': mean_ratio,
        'ratio_deviation': ratio_deviation,
        'corr_ell1_coherence': corr_ell1_coherence,
        'p_ell1': p_ell1,
        'corr_ell2_coherence': corr_ell2_coherence,
        'p_ell2': p_ell2
    }

def plot_coherence_distribution(wmap_results, planck_results, output_dir):
    """Plot the distribution of coherence values."""
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    plt.hist(wmap_results['coherence_values'], bins=10, alpha=0.5, label='WMAP')
    plt.hist(planck_results['coherence_values'], bins=10, alpha=0.5, label='Planck')
    
    plt.xlabel('Coherence Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Coherence Values in Golden Ratio Pairs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(output_dir, 'coherence_distribution.png')
    plt.savefig(output_path)
    plt.close()
    print("Saved coherence distribution plot to {}".format(output_path))

def plot_coherence_vs_multipole(wmap_results, planck_results, output_dir):
    """Plot coherence vs multipole moment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # WMAP plot
    ax1.scatter(wmap_results['ell1_values'], wmap_results['coherence_values'], alpha=0.7, label='l1')
    ax1.scatter(wmap_results['ell2_values'], wmap_results['coherence_values'], alpha=0.7, label='l2')
    
    # Add trend lines
    z1 = np.polyfit(wmap_results['ell1_values'], wmap_results['coherence_values'], 1)
    p1 = np.poly1d(z1)
    ax1.plot(sorted(wmap_results['ell1_values']), p1(sorted(wmap_results['ell1_values'])), 
             linestyle='--', color='blue', alpha=0.5)
    
    z2 = np.polyfit(wmap_results['ell2_values'], wmap_results['coherence_values'], 1)
    p2 = np.poly1d(z2)
    ax1.plot(sorted(wmap_results['ell2_values']), p2(sorted(wmap_results['ell2_values'])), 
             linestyle='--', color='orange', alpha=0.5)
    
    ax1.set_xlabel('Multipole Moment (l)')
    ax1.set_ylabel('Coherence Value')
    ax1.set_title('WMAP: Coherence vs Multipole Moment')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Planck plot
    ax2.scatter(planck_results['ell1_values'], planck_results['coherence_values'], alpha=0.7, label='l1')
    ax2.scatter(planck_results['ell2_values'], planck_results['coherence_values'], alpha=0.7, label='l2')
    
    # Add trend lines
    z1 = np.polyfit(planck_results['ell1_values'], planck_results['coherence_values'], 1)
    p1 = np.poly1d(z1)
    ax2.plot(sorted(planck_results['ell1_values']), p1(sorted(planck_results['ell1_values'])), 
             linestyle='--', color='blue', alpha=0.5)
    
    z2 = np.polyfit(planck_results['ell2_values'], planck_results['coherence_values'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(sorted(planck_results['ell2_values']), p2(sorted(planck_results['ell2_values'])), 
             linestyle='--', color='orange', alpha=0.5)
    
    ax2.set_xlabel('Multipole Moment (l)')
    ax2.set_ylabel('Coherence Value')
    ax2.set_title('Planck: Coherence vs Multipole Moment')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'coherence_vs_multipole.png')
    plt.savefig(output_path)
    plt.close()
    print("Saved coherence vs multipole plot to {}".format(output_path))

def plot_ratio_distribution(wmap_results, planck_results, output_dir):
    """Plot the distribution of l2/l1 ratios."""
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    plt.hist(wmap_results['ratios'], bins=20, alpha=0.5, label='WMAP')
    plt.hist(planck_results['ratios'], bins=20, alpha=0.5, label='Planck')
    
    # Add vertical line for golden ratio
    golden_ratio = (1 + np.sqrt(5)) / 2
    plt.axvline(x=golden_ratio, color='r', linestyle='--', 
                label='Golden Ratio (approx. {:.3f})'.format(golden_ratio))
    
    plt.xlabel('Ratio (l2/l1)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Multipole Ratios in Golden Ratio Pairs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = os.path.join(output_dir, 'ratio_distribution.png')
    plt.savefig(output_path)
    plt.close()
    print("Saved ratio distribution plot to {}".format(output_path))

def plot_heatmap(wmap_pairs, planck_pairs, output_dir):
    """Create a heatmap of coherence values across multipole pairs."""
    # Create dataframes for heatmaps
    wmap_df = pd.DataFrame([(p[0], p[1], p[2]) for p in wmap_pairs], 
                          columns=['ell1', 'ell2', 'coherence'])
    planck_df = pd.DataFrame([(p[0], p[1], p[2]) for p in planck_pairs], 
                            columns=['ell1', 'ell2', 'coherence'])
    
    # Create pivot tables
    wmap_pivot = wmap_df.pivot_table(index='ell1', columns='ell2', values='coherence', aggfunc='mean')
    planck_pivot = planck_df.pivot_table(index='ell1', columns='ell2', values='coherence', aggfunc='mean')
    
    # Plot heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(wmap_pivot, cmap='viridis', ax=ax1, cbar_kws={'label': 'Coherence'})
    ax1.set_title('WMAP: Coherence Heatmap')
    ax1.set_xlabel('l2')
    ax1.set_ylabel('l1')
    
    sns.heatmap(planck_pivot, cmap='viridis', ax=ax2, cbar_kws={'label': 'Coherence'})
    ax2.set_title('Planck: Coherence Heatmap')
    ax2.set_xlabel('l2')
    ax2.set_ylabel('l1')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'coherence_heatmap.png')
    plt.savefig(output_path)
    plt.close()
    print("Saved coherence heatmap to {}".format(output_path))

def find_common_pairs(wmap_pairs, planck_pairs):
    """Find common golden ratio pairs between WMAP and Planck datasets."""
    wmap_ell_pairs = [(p[0], p[1]) for p in wmap_pairs]
    planck_ell_pairs = [(p[0], p[1]) for p in planck_pairs]
    
    common_pairs = []
    for i, wp in enumerate(wmap_ell_pairs):
        for j, pp in enumerate(planck_ell_pairs):
            # Check if the pairs are close enough (exact matches are unlikely due to binning differences)
            if abs(wp[0] - pp[0]) <= 1 and abs(wp[1] - pp[1]) <= 1:
                common_pairs.append((wp, pp, wmap_pairs[i][2], planck_pairs[j][2]))
    
    print("\nCommon Golden Ratio Pairs Analysis")
    print("="*50)
    print("Number of common pairs: {}".format(len(common_pairs)))
    
    if common_pairs:
        print("\nCommon Pairs Details:")
        print("WMAP (l1, l2) | Planck (l1, l2) | WMAP Coherence | Planck Coherence")
        print("-"*75)
        
        for wp, pp, wc, pc in common_pairs:
            print("({}, {}) | ({}, {}) | {:.6f} | {:.6f}".format(wp[0], wp[1], pp[0], pp[1], wc, pc))
        
        # Calculate average coherence difference
        coherence_diffs = [abs(wc - pc) for _, _, wc, pc in common_pairs]
        avg_coherence_diff = np.mean(coherence_diffs)
        print("\nAverage absolute coherence difference: {:.6f}".format(avg_coherence_diff))
    
    return common_pairs

def main():
    # Define paths
    results_dir = "../results/gr_specific_coherence_20250323_164550"
    wmap_file = os.path.join(results_dir, 'wmap/wmap_gr_specific_coherence.txt')
    planck_file = os.path.join(results_dir, 'planck/planck_gr_specific_coherence.txt')
    
    # Create output directory
    output_dir = os.path.join(results_dir, 'additional_analysis')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract pairs
    wmap_pairs = extract_pairs_from_file(wmap_file)
    planck_pairs = extract_pairs_from_file(planck_file)
    
    # Analyze pairs
    wmap_results = analyze_pairs(wmap_pairs, 'WMAP')
    planck_results = analyze_pairs(planck_pairs, 'Planck')
    
    # Find common pairs
    common_pairs = find_common_pairs(wmap_pairs, planck_pairs)
    
    # Create visualizations
    plot_coherence_distribution(wmap_results, planck_results, output_dir)
    plot_coherence_vs_multipole(wmap_results, planck_results, output_dir)
    plot_ratio_distribution(wmap_results, planck_results, output_dir)
    
    try:
        plot_heatmap(wmap_pairs, planck_pairs, output_dir)
    except Exception as e:
        print("Warning: Could not create heatmap: {}".format(e))
    
    # Save summary to file
    with open(os.path.join(output_dir, 'gr_pairs_analysis_summary.txt'), 'w') as f:
        f.write("Golden Ratio Pairs Analysis Summary\n")
        f.write("="*50 + "\n\n")
        
        f.write("WMAP Analysis\n")
        f.write("-"*50 + "\n")
        f.write("Number of pairs: {}\n".format(len(wmap_pairs)))
        f.write("Mean coherence: {:.6f}\n".format(wmap_results['mean_coherence']))
        f.write("Median coherence: {:.6f}\n".format(wmap_results['median_coherence']))
        f.write("Standard deviation: {:.6f}\n".format(wmap_results['std_coherence']))
        f.write("Min coherence: {:.6f}\n".format(min(wmap_results['coherence_values'])))
        f.write("Max coherence: {:.6f}\n".format(max(wmap_results['coherence_values'])))
        f.write("Pairs with coherence > 0.9: {} ({:.2f}%)\n".format(wmap_results['high_coherence_count'], wmap_results['high_coherence_percentage']))
        f.write("Mean ratio (l2/l1): {:.6f}\n".format(wmap_results['mean_ratio']))
        f.write("Deviation from golden ratio: {:.6f}\n".format(wmap_results['ratio_deviation']))
        f.write("Correlation between l1 and coherence: {:.6f} (p={:.6f})\n".format(wmap_results['corr_ell1_coherence'], wmap_results['p_ell1']))
        f.write("Correlation between l2 and coherence: {:.6f} (p={:.6f})\n\n".format(wmap_results['corr_ell2_coherence'], wmap_results['p_ell2']))
        
        f.write("Planck Analysis\n")
        f.write("-"*50 + "\n")
        f.write("Number of pairs: {}\n".format(len(planck_pairs)))
        f.write("Mean coherence: {:.6f}\n".format(planck_results['mean_coherence']))
        f.write("Median coherence: {:.6f}\n".format(planck_results['median_coherence']))
        f.write("Standard deviation: {:.6f}\n".format(planck_results['std_coherence']))
        f.write("Min coherence: {:.6f}\n".format(min(planck_results['coherence_values'])))
        f.write("Max coherence: {:.6f}\n".format(max(planck_results['coherence_values'])))
        f.write("Pairs with coherence > 0.9: {} ({:.2f}%)\n".format(planck_results['high_coherence_count'], planck_results['high_coherence_percentage']))
        f.write("Mean ratio (l2/l1): {:.6f}\n".format(planck_results['mean_ratio']))
        f.write("Deviation from golden ratio: {:.6f}\n".format(planck_results['ratio_deviation']))
        f.write("Correlation between l1 and coherence: {:.6f} (p={:.6f})\n".format(planck_results['corr_ell1_coherence'], planck_results['p_ell1']))
        f.write("Correlation between l2 and coherence: {:.6f} (p={:.6f})\n\n".format(planck_results['corr_ell2_coherence'], planck_results['p_ell2']))
        
        f.write("Common Pairs Analysis\n")
        f.write("-"*50 + "\n")
        f.write("Number of common pairs: {}\n".format(len(common_pairs)))
        
        if common_pairs:
            f.write("\nCommon Pairs Details:\n")
            f.write("WMAP (l1, l2) | Planck (l1, l2) | WMAP Coherence | Planck Coherence\n")
            f.write("-"*75 + "\n")
            
            for wp, pp, wc, pc in common_pairs:
                f.write("({}, {}) | ({}, {}) | {:.6f} | {:.6f}\n".format(wp[0], wp[1], pp[0], pp[1], wc, pc))
            
            coherence_diffs = [abs(wc - pc) for _, _, wc, pc in common_pairs]
            avg_coherence_diff = np.mean(coherence_diffs)
            f.write("\nAverage absolute coherence difference: {:.6f}\n".format(avg_coherence_diff))
    
    print("\nAnalysis complete. Results saved to {}".format(output_dir))

if __name__ == "__main__":
    main()
