#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze and visualize the results of the correlation analysis on WMAP data.
This script loads the results from the correlation analysis and generates
comprehensive visualizations and statistical analyses.
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

def load_results(results_dir):
    """
    Load results from the correlation analysis.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing the results
        
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing the results
    """
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print("Error: Results directory does not exist: %s" % results_dir)
        return None
    
    # Check if results CSV file exists
    results_csv = os.path.join(results_dir, 'correlation_results.csv')
    if os.path.exists(results_csv):
        print("Loading results from CSV file: %s" % results_csv)
        return pd.read_csv(results_csv)
    
    # If CSV doesn't exist, try to load from JSON files
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    if not json_files:
        print("Error: No results files found in directory: %s" % results_dir)
        return None
    
    print("Loading results from %d JSON files..." % len(json_files))
    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save as CSV for future use
    results_df.to_csv(results_csv, index=False)
    print("Saved results to CSV file: %s" % results_csv)
    
    return results_df

def create_correlation_plot(results_df, output_dir):
    """
    Create a plot showing the correlation between Transfer Entropy and
    Integrated Information across different GR levels.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing the results
    output_dir : str
        Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot with GR level as color
    scatter = plt.scatter(
        results_df['avg_te'], 
        results_df['avg_ii'],
        c=results_df['gr_level'],
        cmap='viridis',
        s=100,
        alpha=0.8
    )
    
    # Add error bars
    for i, row in results_df.iterrows():
        plt.errorbar(
            row['avg_te'], 
            row['avg_ii'],
            xerr=row['std_te'],
            yerr=row['std_ii'],
            fmt='none',
            ecolor='gray',
            alpha=0.5
        )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('GR Level', fontsize=14)
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(results_df['avg_te'], results_df['avg_ii'])
    
    # Add correlation information
    plt.title(
        'Correlation between Transfer Entropy and Integrated Information\n'
        'Pearson r = %.3f (p = %.3e)',
        fontsize=16
    )
    
    plt.xlabel('Transfer Entropy (bits)', fontsize=14)
    plt.ylabel('Integrated Information (bits)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'correlation_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("Saved correlation plot to: %s" % output_file)
    
    plt.close()

def create_gr_level_plots(results_df, output_dir):
    """
    Create plots showing how Transfer Entropy and Integrated Information
    vary with GR level.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing the results
    output_dir : str
        Directory to save the plots
    """
    # Sort by GR level
    results_df = results_df.sort_values('gr_level')
    
    # Create plot for Transfer Entropy
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        results_df['gr_level'],
        results_df['avg_te'],
        yerr=results_df['std_te'],
        fmt='o-',
        linewidth=2,
        markersize=8,
        capsize=5
    )
    plt.title('Transfer Entropy vs. GR Level', fontsize=16)
    plt.xlabel('GR Level', fontsize=14)
    plt.ylabel('Transfer Entropy (bits)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'te_vs_gr_level.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("Saved TE vs. GR level plot to: %s" % output_file)
    
    plt.close()
    
    # Create plot for Integrated Information
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        results_df['gr_level'],
        results_df['avg_ii'],
        yerr=results_df['std_ii'],
        fmt='o-',
        linewidth=2,
        markersize=8,
        capsize=5,
        color='orange'
    )
    plt.title('Integrated Information vs. GR Level', fontsize=16)
    plt.xlabel('GR Level', fontsize=14)
    plt.ylabel('Integrated Information (bits)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'ii_vs_gr_level.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("Saved II vs. GR level plot to: %s" % output_file)
    
    plt.close()

def create_distribution_plots(results_df, output_dir):
    """
    Create plots showing the distribution of Transfer Entropy and
    Integrated Information values for each GR level.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing the results
    output_dir : str
        Directory to save the plots
    """
    # Check if we have the detailed values
    if 'te_values' not in results_df.columns or 'ii_values' not in results_df.columns:
        print("Warning: Detailed TE and II values not available, skipping distribution plots")
        return
    
    # Create directory for distribution plots
    dist_dir = os.path.join(output_dir, 'distributions')
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    
    # For each GR level, create distribution plots
    for i, row in results_df.iterrows():
        gr_level = row['gr_level']
        
        # Convert string representations of lists to actual lists
        if isinstance(row['te_values'], str):
            te_values = json.loads(row['te_values'].replace("'", '"'))
        else:
            te_values = row['te_values']
            
        if isinstance(row['ii_values'], str):
            ii_values = json.loads(row['ii_values'].replace("'", '"'))
        else:
            ii_values = row['ii_values']
        
        # Create plot for Transfer Entropy distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(te_values, kde=True)
        plt.title('Distribution of Transfer Entropy Values (GR Level = %.2f)' % gr_level, fontsize=14)
        plt.xlabel('Transfer Entropy (bits)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Save the plot
        output_file = os.path.join(dist_dir, 'te_dist_gr_%.2f.png' % gr_level)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        # Create plot for Integrated Information distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(ii_values, kde=True, color='orange')
        plt.title('Distribution of Integrated Information Values (GR Level = %.2f)' % gr_level, fontsize=14)
        plt.xlabel('Integrated Information (bits)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Save the plot
        output_file = os.path.join(dist_dir, 'ii_dist_gr_%.2f.png' % gr_level)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    print("Saved distribution plots to: %s" % dist_dir)

def create_heatmap(results_df, output_dir):
    """
    Create a heatmap showing the relationship between GR level,
    Transfer Entropy, and Integrated Information.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing the results
    output_dir : str
        Directory to save the plot
    """
    # Create a pivot table
    pivot_df = results_df.pivot_table(
        index='gr_level',
        values=['avg_te', 'avg_ii'],
        aggfunc='mean'
    )
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        linewidths=0.5
    )
    plt.title('Heatmap of Metrics vs. GR Level', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'metrics_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("Saved heatmap to: %s" % output_file)
    
    plt.close()

def generate_report(results_df, output_dir):
    """
    Generate a comprehensive report of the correlation analysis results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing the results
    output_dir : str
        Directory to save the report
    """
    # Create report file
    report_file = os.path.join(output_dir, 'correlation_analysis_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=== WMAP Correlation Analysis Report ===\n\n")
        
        # Summary statistics
        f.write("=== Summary Statistics ===\n")
        f.write("Number of GR levels analyzed: %d\n" % len(results_df))
        f.write("GR levels: %s\n" % ", ".join(["%.2f" % x for x in sorted(results_df['gr_level'])]))
        f.write("\n")
        
        # Overall correlation
        corr, p_value = stats.pearsonr(results_df['avg_te'], results_df['avg_ii'])
        f.write("=== Overall Correlation ===\n")
        f.write("Pearson correlation coefficient: %.3f\n" % corr)
        f.write("p-value: %.3e\n" % p_value)
        f.write("Interpretation: %s\n" % (
            "Strong positive correlation" if corr > 0.7 else
            "Moderate positive correlation" if corr > 0.3 else
            "Weak positive correlation" if corr > 0 else
            "No correlation" if corr == 0 else
            "Weak negative correlation" if corr > -0.3 else
            "Moderate negative correlation" if corr > -0.7 else
            "Strong negative correlation"
        ))
        f.write("\n")
        
        # Detailed results for each GR level
        f.write("=== Detailed Results by GR Level ===\n")
        for i, row in results_df.sort_values('gr_level').iterrows():
            f.write("GR Level: %.2f\n" % row['gr_level'])
            f.write("  Transfer Entropy: %.3f ± %.3f bits\n" % (row['avg_te'], row['std_te']))
            f.write("  Integrated Information: %.3f ± %.3f bits\n" % (row['avg_ii'], row['std_ii']))
            f.write("\n")
        
        # Regression analysis
        f.write("=== Regression Analysis ===\n")
        X = results_df['gr_level'].values.reshape(-1, 1)
        
        # TE regression
        te_model = stats.linregress(results_df['gr_level'], results_df['avg_te'])
        f.write("Transfer Entropy vs. GR Level:\n")
        f.write("  Slope: %.3f\n" % te_model.slope)
        f.write("  Intercept: %.3f\n" % te_model.intercept)
        f.write("  R-squared: %.3f\n" % te_model.rvalue**2)
        f.write("  p-value: %.3e\n" % te_model.pvalue)
        f.write("\n")
        
        # II regression
        ii_model = stats.linregress(results_df['gr_level'], results_df['avg_ii'])
        f.write("Integrated Information vs. GR Level:\n")
        f.write("  Slope: %.3f\n" % ii_model.slope)
        f.write("  Intercept: %.3f\n" % ii_model.intercept)
        f.write("  R-squared: %.3f\n" % ii_model.rvalue**2)
        f.write("  p-value: %.3e\n" % ii_model.pvalue)
        f.write("\n")
        
        # Conclusion
        f.write("=== Conclusion ===\n")
        if corr > 0.5 and p_value < 0.05:
            f.write("There is a significant positive correlation between Transfer Entropy and Integrated Information in the WMAP data.\n")
        elif corr < -0.5 and p_value < 0.05:
            f.write("There is a significant negative correlation between Transfer Entropy and Integrated Information in the WMAP data.\n")
        else:
            f.write("There is no significant correlation between Transfer Entropy and Integrated Information in the WMAP data.\n")
        
        if te_model.pvalue < 0.05:
            f.write("Transfer Entropy shows a significant %s with increasing GR level.\n" % ("increase" if te_model.slope > 0 else "decrease"))
        else:
            f.write("Transfer Entropy does not show a significant relationship with GR level.\n")
            
        if ii_model.pvalue < 0.05:
            f.write("Integrated Information shows a significant %s with increasing GR level.\n" % ("increase" if ii_model.slope > 0 else "decrease"))
        else:
            f.write("Integrated Information does not show a significant relationship with GR level.\n")
    
    print("Generated report: %s" % report_file)
    
    return report_file

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze correlation analysis results")
    parser.add_argument("--results-dir", default="./correlation_results_wmap_real",
                        help="Directory containing the results")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save the analysis (default: same as results-dir)")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load results
    results_df = load_results(args.results_dir)
    if results_df is None:
        sys.exit(1)
    
    print("Loaded results with %d rows" % len(results_df))
    
    # Create visualizations
    create_correlation_plot(results_df, args.output_dir)
    create_gr_level_plots(results_df, args.output_dir)
    create_distribution_plots(results_df, args.output_dir)
    create_heatmap(results_df, args.output_dir)
    
    # Generate report
    report_file = generate_report(results_df, args.output_dir)
    
    print("\nAnalysis complete!")
    print("Results saved to: %s" % args.output_dir)
    print("Report: %s" % report_file)

if __name__ == "__main__":
    main()
