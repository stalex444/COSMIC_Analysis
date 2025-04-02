#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-validate correlation analysis results between WMAP and Planck data.
This script compares the Transfer Entropy and Integrated Information metrics
calculated from both WMAP and Planck data to validate the findings.
"""

from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import json
import glob
from astropy.io import fits

# Import the MetricCorrelationAnalysis class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_correlation_analysis import MetricCorrelationAnalysis

def load_wmap_data(data_dir="wmap_data/raw_data", data_type='power_spectrum'):
    """
    Load WMAP data for analysis.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing WMAP data
    data_type : str
        Type of WMAP data to load ('power_spectrum' or 'binned_power_spectrum')
        
    Returns:
    --------
    data : numpy.ndarray
        WMAP data
    """
    # Define paths to WMAP data files
    if data_type == 'power_spectrum':
        # Load WMAP power spectrum data
        file_path = os.path.join(data_dir, 'wmap_tt_spectrum_9yr_v5.txt')
    elif data_type == 'binned_power_spectrum':
        # Load WMAP binned power spectrum data
        file_path = os.path.join(data_dir, 'wmap_binned_tt_spectrum_9yr_v5.txt')
    else:
        print("Unknown data type: %s" % data_type)
        return None
    
    try:
        # Check if file exists and is not an HTML error page
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('<!DOCTYPE HTML') or first_line.startswith('<html'):
                print("Error: WMAP file appears to be an HTML error page.")
                return None
        
        # Load data - expected format is multipole (l), power (Cl), error
        # Skip comment lines (lines starting with #)
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip().startswith('#'):
                    values = [float(x) for x in line.strip().split()]
                    data.append(values)
        
        data = np.array(data)
        
        # Extract multipole and power values
        if data_type == 'power_spectrum':
            multipoles = data[:, 0]  # multipole moment l
            power = data[:, 1]       # power spectrum value
        else:  # binned_power_spectrum
            multipoles = data[:, 0]  # mean multipole moment
            power = data[:, 3]       # power spectrum value
        
        print("Loaded WMAP %s with %d data points" % (data_type, len(multipoles)))
        return power  # Return just the power spectrum values
            
    except Exception as e:
        print("Error loading WMAP data: %s" % str(e))
        return None

def load_planck_data(data_dir="planck_data/raw_data", data_type='power_spectrum'):
    """
    Load Planck data for analysis.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing Planck data
    data_type : str
        Type of Planck data to load ('power_spectrum' or 'binned_power_spectrum')
        
    Returns:
    --------
    data : numpy.ndarray
        Planck data
    """
    # Define paths to Planck data files
    if data_type == 'power_spectrum':
        # Load Planck power spectrum data
        file_path = os.path.join(data_dir, 'planck_tt_spectrum_2018_R3.01.txt')
    elif data_type == 'binned_power_spectrum':
        # Load Planck binned power spectrum data
        file_path = os.path.join(data_dir, 'planck_binned_tt_spectrum_2018_R3.01.txt')
    else:
        print("Unknown data type: %s" % data_type)
        return None
    
    try:
        # Check if file exists and is not an HTML error page
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('<!DOCTYPE HTML') or first_line.startswith('<html'):
                print("Error: Planck file appears to be an HTML error page.")
                return None
        
        # Load data - expected format is multipole (l), power (Cl), error
        # Skip comment lines (lines starting with #)
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip().startswith('#'):
                    values = [float(x) for x in line.strip().split()]
                    data.append(values)
        
        data = np.array(data)
        
        # Extract multipole and power values (format may vary)
        # For Planck power spectrum, typically column 1 is D_l = l(l+1)C_l/(2π)
        if data_type == 'power_spectrum':
            multipoles = data[:, 0]  # multipole moment l
            power = data[:, 1]       # power spectrum value
        else:  # binned_power_spectrum
            multipoles = data[:, 0]  # mean multipole moment
            power = data[:, 1]       # power spectrum value
        
        print("Loaded Planck %s with %d data points" % (data_type, len(multipoles)))
        return power  # Return just the power spectrum values
            
    except Exception as e:
        print("Error loading Planck data: %s" % str(e))
        return None

def run_cross_validation(wmap_data, planck_data, n_surrogates=10000, n_gr_levels=10, output_dir="cross_validation_results"):
    """
    Run cross-validation between WMAP and Planck data.
    
    Parameters:
    -----------
    wmap_data : numpy.ndarray
        WMAP data
    planck_data : numpy.ndarray
        Planck data
    n_surrogates : int
        Number of surrogate datasets to generate
    n_gr_levels : int
        Number of GR levels to test
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    results : dict
        Cross-validation results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize analyzers for WMAP and Planck data
    wmap_analyzer = MetricCorrelationAnalysis(
        output_dir=os.path.join(output_dir, "wmap"),
        n_surrogates=n_surrogates,
        n_gr_levels=n_gr_levels
    )
    
    planck_analyzer = MetricCorrelationAnalysis(
        output_dir=os.path.join(output_dir, "planck"),
        n_surrogates=n_surrogates,
        n_gr_levels=n_gr_levels
    )
    
    # Run manipulation experiment on WMAP data
    print("Running manipulation experiment on WMAP data...")
    wmap_results = wmap_analyzer.run_manipulation_experiment(wmap_data)
    
    # Run manipulation experiment on Planck data
    print("Running manipulation experiment on Planck data...")
    planck_results = planck_analyzer.run_manipulation_experiment(planck_data)
    
    # Save results
    wmap_df = pd.DataFrame(wmap_results)
    planck_df = pd.DataFrame(planck_results)
    
    wmap_df.to_csv(os.path.join(output_dir, "wmap_results.csv"), index=False)
    planck_df.to_csv(os.path.join(output_dir, "planck_results.csv"), index=False)
    
    # Compare results
    compare_results(wmap_df, planck_df, output_dir)
    
    return {
        "wmap_results": wmap_results,
        "planck_results": planck_results
    }

def compare_results(wmap_df, planck_df, output_dir):
    """
    Compare results from WMAP and Planck data.
    
    Parameters:
    -----------
    wmap_df : pandas.DataFrame
        WMAP results
    planck_df : pandas.DataFrame
        Planck results
    output_dir : str
        Directory to save comparison results
    """
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # Ensure both DataFrames have the same GR levels
    gr_levels = sorted(set(wmap_df['gr_level']).intersection(set(planck_df['gr_level'])))
    wmap_df = wmap_df[wmap_df['gr_level'].isin(gr_levels)]
    planck_df = planck_df[planck_df['gr_level'].isin(gr_levels)]
    
    # Sort by GR level
    wmap_df = wmap_df.sort_values('gr_level')
    planck_df = planck_df.sort_values('gr_level')
    
    # Calculate correlations
    te_corr, te_p_value = stats.pearsonr(wmap_df['avg_te'], planck_df['avg_te'])
    ii_corr, ii_p_value = stats.pearsonr(wmap_df['avg_ii'], planck_df['avg_ii'])
    
    # Create comparison plots
    
    # Transfer Entropy comparison
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        wmap_df['gr_level'],
        wmap_df['avg_te'],
        yerr=wmap_df['std_te'],
        fmt='o-',
        linewidth=2,
        markersize=8,
        capsize=5,
        label='WMAP'
    )
    plt.errorbar(
        planck_df['gr_level'],
        planck_df['avg_te'],
        yerr=planck_df['std_te'],
        fmt='s--',
        linewidth=2,
        markersize=8,
        capsize=5,
        label='Planck'
    )
    plt.title('Transfer Entropy vs. GR Level: WMAP vs. Planck', fontsize=16)
    plt.xlabel('GR Level', fontsize=14)
    plt.ylabel('Transfer Entropy (bits)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add correlation information
    plt.figtext(
        0.5, 0.01,
        'Correlation between WMAP and Planck TE: r = %.3f (p = %.3e)' % (te_corr, te_p_value),
        ha='center',
        fontsize=12
    )
    
    # Save the plot
    output_file = os.path.join(comparison_dir, 'te_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Integrated Information comparison
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        wmap_df['gr_level'],
        wmap_df['avg_ii'],
        yerr=wmap_df['std_ii'],
        fmt='o-',
        linewidth=2,
        markersize=8,
        capsize=5,
        label='WMAP',
        color='orange'
    )
    plt.errorbar(
        planck_df['gr_level'],
        planck_df['avg_ii'],
        yerr=planck_df['std_ii'],
        fmt='s--',
        linewidth=2,
        markersize=8,
        capsize=5,
        label='Planck',
        color='green'
    )
    plt.title('Integrated Information vs. GR Level: WMAP vs. Planck', fontsize=16)
    plt.xlabel('GR Level', fontsize=14)
    plt.ylabel('Integrated Information (bits)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add correlation information
    plt.figtext(
        0.5, 0.01,
        'Correlation between WMAP and Planck II: r = %.3f (p = %.3e)' % (ii_corr, ii_p_value),
        ha='center',
        fontsize=12
    )
    
    # Save the plot
    output_file = os.path.join(comparison_dir, 'ii_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation scatter plots
    plt.figure(figsize=(10, 8))
    
    # Transfer Entropy correlation
    plt.subplot(2, 1, 1)
    for i, gr_level in enumerate(gr_levels):
        wmap_te = wmap_df[wmap_df['gr_level'] == gr_level]['avg_te'].values[0]
        planck_te = planck_df[planck_df['gr_level'] == gr_level]['avg_te'].values[0]
        plt.scatter(wmap_te, planck_te, s=100, label='GR=%.2f' % gr_level)
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(wmap_df['avg_te'], planck_df['avg_te'])
    x = np.array([min(wmap_df['avg_te']), max(wmap_df['avg_te'])])
    plt.plot(x, slope * x + intercept, 'r--', label='Regression line')
    
    plt.title('WMAP vs. Planck Transfer Entropy (r = %.3f, p = %.3e)' % (te_corr, te_p_value), fontsize=14)
    plt.xlabel('WMAP Transfer Entropy (bits)', fontsize=12)
    plt.ylabel('Planck Transfer Entropy (bits)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Integrated Information correlation
    plt.subplot(2, 1, 2)
    for i, gr_level in enumerate(gr_levels):
        wmap_ii = wmap_df[wmap_df['gr_level'] == gr_level]['avg_ii'].values[0]
        planck_ii = planck_df[planck_df['gr_level'] == gr_level]['avg_ii'].values[0]
        plt.scatter(wmap_ii, planck_ii, s=100, label='GR=%.2f' % gr_level, color='orange')
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(wmap_df['avg_ii'], planck_df['avg_ii'])
    x = np.array([min(wmap_df['avg_ii']), max(wmap_df['avg_ii'])])
    plt.plot(x, slope * x + intercept, 'g--', label='Regression line')
    
    plt.title('WMAP vs. Planck Integrated Information (r = %.3f, p = %.3e)' % (ii_corr, ii_p_value), fontsize=14)
    plt.xlabel('WMAP Integrated Information (bits)', fontsize=12)
    plt.ylabel('Planck Integrated Information (bits)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(comparison_dir, 'correlation_scatter.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comparison report
    generate_comparison_report(wmap_df, planck_df, te_corr, te_p_value, ii_corr, ii_p_value, comparison_dir)

def generate_comparison_report(wmap_df, planck_df, te_corr, te_p_value, ii_corr, ii_p_value, output_dir):
    """
    Generate a report comparing WMAP and Planck results.
    
    Parameters:
    -----------
    wmap_df : pandas.DataFrame
        WMAP results
    planck_df : pandas.DataFrame
        Planck results
    te_corr : float
        Correlation coefficient for Transfer Entropy
    te_p_value : float
        p-value for Transfer Entropy correlation
    ii_corr : float
        Correlation coefficient for Integrated Information
    ii_p_value : float
        p-value for Integrated Information correlation
    output_dir : str
        Directory to save the report
    """
    report_file = os.path.join(output_dir, 'cross_validation_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=== WMAP vs. Planck Cross-Validation Report ===\n\n")
        
        # Summary statistics
        f.write("=== Summary Statistics ===\n")
        f.write("Number of GR levels analyzed: %d\n" % len(wmap_df))
        f.write("GR levels: %s\n" % ", ".join(["%.2f" % x for x in sorted(wmap_df['gr_level'])]))
        f.write("\n")
        
        # Transfer Entropy correlation
        f.write("=== Transfer Entropy Correlation ===\n")
        f.write("Pearson correlation coefficient: %.3f\n" % te_corr)
        f.write("p-value: %.3e\n" % te_p_value)
        f.write("Interpretation: %s\n" % (
            "Strong positive correlation" if te_corr > 0.7 else
            "Moderate positive correlation" if te_corr > 0.3 else
            "Weak positive correlation" if te_corr > 0 else
            "No correlation" if te_corr == 0 else
            "Weak negative correlation" if te_corr > -0.3 else
            "Moderate negative correlation" if te_corr > -0.7 else
            "Strong negative correlation"
        ))
        f.write("\n")
        
        # Integrated Information correlation
        f.write("=== Integrated Information Correlation ===\n")
        f.write("Pearson correlation coefficient: %.3f\n" % ii_corr)
        f.write("p-value: %.3e\n" % ii_p_value)
        f.write("Interpretation: %s\n" % (
            "Strong positive correlation" if ii_corr > 0.7 else
            "Moderate positive correlation" if ii_corr > 0.3 else
            "Weak positive correlation" if ii_corr > 0 else
            "No correlation" if ii_corr == 0 else
            "Weak negative correlation" if ii_corr > -0.3 else
            "Moderate negative correlation" if ii_corr > -0.7 else
            "Strong negative correlation"
        ))
        f.write("\n")
        
        # Detailed comparison
        f.write("=== Detailed Comparison by GR Level ===\n")
        f.write("GR Level | WMAP TE      | Planck TE    | Diff TE      | WMAP II      | Planck II    | Diff II\n")
        f.write("---------|--------------|--------------|--------------|--------------|--------------|------------\n")
        
        for i, gr_level in enumerate(sorted(wmap_df['gr_level'])):
            wmap_row = wmap_df[wmap_df['gr_level'] == gr_level].iloc[0]
            planck_row = planck_df[planck_df['gr_level'] == gr_level].iloc[0]
            
            wmap_te = wmap_row['avg_te']
            planck_te = planck_row['avg_te']
            diff_te = wmap_te - planck_te
            
            wmap_ii = wmap_row['avg_ii']
            planck_ii = planck_row['avg_ii']
            diff_ii = wmap_ii - planck_ii
            
            f.write("%.2f     | %.4f±%.4f | %.4f±%.4f | %.4f      | %.4f±%.4f | %.4f±%.4f | %.4f\n" % (
                gr_level,
                wmap_te, wmap_row['std_te'],
                planck_te, planck_row['std_te'],
                diff_te,
                wmap_ii, wmap_row['std_ii'],
                planck_ii, planck_row['std_ii'],
                diff_ii
            ))
        
        f.write("\n")
        
        # Conclusion
        f.write("=== Conclusion ===\n")
        if te_corr > 0.7 and te_p_value < 0.05 and ii_corr > 0.7 and ii_p_value < 0.05:
            f.write("There is a strong and significant correlation between WMAP and Planck results for both Transfer Entropy and Integrated Information metrics. This provides strong cross-validation evidence for the observed patterns in cosmic microwave background radiation.\n")
        elif (te_corr > 0.5 and te_p_value < 0.05) or (ii_corr > 0.5 and ii_p_value < 0.05):
            f.write("There is a moderate to strong correlation between WMAP and Planck results for at least one of the metrics. This provides partial cross-validation evidence for the observed patterns in cosmic microwave background radiation.\n")
        else:
            f.write("The correlation between WMAP and Planck results is weak or not statistically significant. Further investigation is needed to understand the discrepancies between the two datasets.\n")
    
    print("Generated cross-validation report: %s" % report_file)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Cross-validate WMAP and Planck data")
    parser.add_argument("--wmap-data-dir", default="wmap_data/raw_data",
                        help="Directory containing WMAP data")
    parser.add_argument("--planck-data-dir", default="planck_data/raw_data",
                        help="Directory containing Planck data")
    parser.add_argument("--data-type", choices=['power_spectrum', 'binned_power_spectrum'],
                        default='power_spectrum', help="Type of data to use")
    parser.add_argument("--n-surrogates", type=int, default=10000,
                        help="Number of surrogate datasets to generate")
    parser.add_argument("--n-gr-levels", type=int, default=10,
                        help="Number of GR levels to test")
    parser.add_argument("--output-dir", default="cross_validation_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Load WMAP data
    wmap_data = load_wmap_data(args.wmap_data_dir, args.data_type)
    if wmap_data is None:
        print("Error loading WMAP data. Please run download_wmap_data_lambda.py first.")
        sys.exit(1)
    
    # Load Planck data
    planck_data = load_planck_data(args.planck_data_dir, args.data_type)
    if planck_data is None:
        print("Error loading Planck data. Please run download_planck_data.py first.")
        sys.exit(1)
    
    # Run cross-validation
    results = run_cross_validation(
        wmap_data, planck_data,
        n_surrogates=args.n_surrogates,
        n_gr_levels=args.n_gr_levels,
        output_dir=args.output_dir
    )
    
    print("\nCross-validation complete!")
    print("Results saved to: %s" % args.output_dir)

if __name__ == "__main__":
    main()
