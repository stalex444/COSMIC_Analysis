#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extended Meta-Coherence Test for WMAP and Planck CMB data.

This script implements an enhanced version of the Meta-Coherence Test,
which analyzes the coherence of local coherence measures across different scales
in the CMB power spectrum, with additional features:

1. Multiple mathematical constants (φ, √2, √3, π, e, etc.)
2. Multiple coherence calculation methods
3. Improved statistical significance with 10,000 simulations
4. Multi-scale analysis
5. Cross-correlation between different metrics
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from datetime import datetime
import argparse
import multiprocessing
from functools import partial
import json
import logging

# Import utility functions
from meta_coherence_utils import (
    load_wmap_power_spectrum, load_planck_power_spectrum, preprocess_data,
    calculate_local_coherence, calculate_meta_coherence, run_monte_carlo_parallel,
    plot_meta_coherence_results, analyze_multiple_mathematical_constants
)


def setup_logging(output_dir):
    """Set up logging configuration."""
    log_path = os.path.join(output_dir, "extended_meta_coherence.log")
    
    # Create a logger
    logger = logging.getLogger('ExtendedMetaCoherence')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def run_extended_meta_coherence_test(data, output_dir, name, n_simulations=10000, 
                                    window_sizes=[10, 20, 30], step_size=5, 
                                    local_coherence_methods=["inverse_std", "autocorr"],
                                    meta_coherence_methods=["cv", "entropy"],
                                    constants=None, 
                                    parallel=True, num_processes=None):
    """
    Run extended meta-coherence test on the provided data.
    
    Args:
        data (numpy.ndarray): Data to analyze
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        n_simulations (int): Number of simulations for Monte Carlo
        window_sizes (list): Sizes of the sliding windows to test
        step_size (int): Step size for sliding the window
        local_coherence_methods (list): Methods for calculating local coherence
        meta_coherence_methods (list): Methods for calculating meta-coherence
        constants (dict): Dictionary of mathematical constants to analyze
        parallel (bool): Whether to use parallel processing
        num_processes (int): Number of processes to use for parallelization
        
    Returns:
        dict: Analysis results
    """
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting extended meta-coherence analysis for {name}")
    logger.info(f"Parameters: n_simulations={n_simulations}, window_sizes={window_sizes}, "
               f"step_size={step_size}")
    logger.info(f"Local coherence methods: {local_coherence_methods}")
    logger.info(f"Meta-coherence methods: {meta_coherence_methods}")
    
    # Create dataset directory
    dataset_dir = os.path.join(output_dir, name.lower())
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Set default mathematical constants if not provided
    if constants is None:
        constants = {
            "golden_ratio": (1 + np.sqrt(5)) / 2,  # ~1.618
            "sqrt2": np.sqrt(2),                   # ~1.414
            "sqrt3": np.sqrt(3),                   # ~1.732
            "pi": np.pi,                           # ~3.142
            "e": np.e,                             # ~2.718
            "ln2": np.log(2),                      # ~0.693
        }
    
    logger.info(f"Mathematical constants: {constants}")
    
    # Initialize results dictionary
    results = {
        "dataset": name,
        "n_simulations": n_simulations,
        "window_sizes": window_sizes,
        "step_size": step_size,
        "local_coherence_methods": local_coherence_methods,
        "meta_coherence_methods": meta_coherence_methods,
        "constants": constants,
        "results_by_method": {},
    }
    
    # Run analysis for each combination of window size and methods
    for window_size in window_sizes:
        window_results = {}
        
        for local_method in local_coherence_methods:
            for meta_method in meta_coherence_methods:
                method_key = f"{local_method}_{meta_method}_w{window_size}"
                logger.info(f"Running analysis with: window_size={window_size}, "
                           f"local_method={local_method}, meta_method={meta_method}")
                
                # Run Monte Carlo simulations
                if parallel:
                    (p_value, phi_optimality, actual_meta_coherence, 
                     sim_meta_coherences, window_centers, local_coherence_values) = run_monte_carlo_parallel(
                        data, n_simulations=n_simulations, window_size=window_size, step_size=step_size,
                        num_processes=num_processes, local_coherence_method=local_method,
                        meta_coherence_method=meta_method
                    )
                else:
                    # For simplicity, we'll just use the parallel version with 1 process
                    (p_value, phi_optimality, actual_meta_coherence, 
                     sim_meta_coherences, window_centers, local_coherence_values) = run_monte_carlo_parallel(
                        data, n_simulations=n_simulations, window_size=window_size, step_size=step_size,
                        num_processes=1, local_coherence_method=local_method,
                        meta_coherence_method=meta_method
                    )
                
                # Create plot
                plot_title = f"{name} Meta-Coherence (w={window_size}, {local_method}, {meta_method})"
                plot_path = os.path.join(dataset_dir, f"meta_coherence_{method_key}.png")
                plot_meta_coherence_results(
                    window_centers, local_coherence_values, p_value, phi_optimality,
                    sim_meta_coherences, actual_meta_coherence, plot_title, plot_path
                )
                
                # Store results
                window_results[method_key] = {
                    "p_value": p_value,
                    "phi_optimality": phi_optimality,
                    "actual_meta_coherence": actual_meta_coherence,
                    "significant": p_value < 0.05,
                }
                
                logger.info(f"Results for {method_key}:")
                logger.info(f"  p-value: {p_value:.6f}")
                logger.info(f"  phi-optimality: {phi_optimality:.6f}")
                logger.info(f"  meta-coherence: {actual_meta_coherence:.6f}")
                logger.info(f"  significant: {p_value < 0.05}")
        
        results["results_by_method"].update(window_results)
    
    # Analyze multiple mathematical constants
    logger.info("Analyzing meta-coherence for multiple mathematical constants")
    constants_results = analyze_multiple_mathematical_constants(
        data, constants, window_size=window_sizes[0], step_size=step_size, 
        local_coherence_method=local_coherence_methods[0],
        meta_coherence_method=meta_coherence_methods[0],
        n_sims_per_constant=min(1000, n_simulations // len(constants))
    )
    
    # Store mathematical constants results
    results["constants_results"] = {}
    for const_name, const_result in constants_results.items():
        results["constants_results"][const_name] = {
            "meta_coherence": const_result["meta_coherence"],
            "p_value": const_result["p_value"],
            "phi_optimality": const_result["phi_optimality"],
            "significant": const_result["significant"],
        }
    
    # Generate summary text report
    summary_path = os.path.join(dataset_dir, f"{name.lower()}_extended_meta_coherence.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Extended Meta-Coherence Test Results: {name} CMB Data\n")
        f.write("==================================================\n\n")
        
        f.write("Meta-Coherence Results by Method:\n")
        for method_key, method_result in results["results_by_method"].items():
            f.write(f"  {method_key}:\n")
            f.write(f"    Meta-Coherence: {method_result['actual_meta_coherence']:.6f}\n")
            f.write(f"    P-value: {method_result['p_value']:.6f}\n")
            f.write(f"    Phi-Optimality: {method_result['phi_optimality']:.6f}\n")
            f.write(f"    Significant: {method_result['significant']}\n\n")
        
        f.write("Mathematical Constants Meta-Coherence Results:\n")
        for const_name, const_result in results["constants_results"].items():
            f.write(f"  {const_name}:\n")
            f.write(f"    Meta-Coherence: {const_result['meta_coherence']:.6f}\n")
            f.write(f"    P-value: {const_result['p_value']:.6f}\n")
            f.write(f"    Phi-Optimality: {const_result['phi_optimality']:.6f}\n")
            f.write(f"    Significant: {const_result['significant']}\n\n")
        
        f.write("\nAnalysis Parameters:\n")
        f.write(f"  Number of Simulations: {n_simulations}\n")
        f.write(f"  Window Sizes: {window_sizes}\n")
        f.write(f"  Step Size: {step_size}\n")
        f.write(f"  Local Coherence Methods: {local_coherence_methods}\n")
        f.write(f"  Meta-Coherence Methods: {meta_coherence_methods}\n")
    
    # Save full results as JSON
    results_json_path = os.path.join(dataset_dir, f"{name.lower()}_extended_meta_coherence_results.json")
    with open(results_json_path, 'w') as f:
        # Convert NumPy arrays and types to Python native types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            else:
                return obj
        
        json_results = convert_for_json(results)
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {summary_path} and {results_json_path}")
    
    return results


def compare_results(wmap_results, planck_results, output_dir):
    """
    Compare extended meta-coherence test results between WMAP and Planck data.
    
    Args:
        wmap_results (dict): Results from WMAP data analysis
        planck_results (dict): Results from Planck data analysis
        output_dir (str): Directory to save comparison results
    """
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info("Comparing WMAP and Planck extended meta-coherence results")
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Initialize comparison results
    comparison = {
        "meta_coherence_comparison": {},
        "mathematical_constants_comparison": {}
    }
    
    # Compare meta-coherence results by method
    for method_key in wmap_results["results_by_method"].keys():
        if method_key in planck_results["results_by_method"]:
            wmap_result = wmap_results["results_by_method"][method_key]
            planck_result = planck_results["results_by_method"][method_key]
            
            comparison["meta_coherence_comparison"][method_key] = {
                "wmap_meta_coherence": wmap_result["actual_meta_coherence"],
                "planck_meta_coherence": planck_result["actual_meta_coherence"],
                "wmap_p_value": wmap_result["p_value"],
                "planck_p_value": planck_result["p_value"],
                "wmap_phi_optimality": wmap_result["phi_optimality"],
                "planck_phi_optimality": planck_result["phi_optimality"],
                "wmap_significant": wmap_result["significant"],
                "planck_significant": planck_result["significant"],
            }
    
    # Compare mathematical constants results
    for const_name in wmap_results["constants_results"].keys():
        if const_name in planck_results["constants_results"]:
            wmap_const = wmap_results["constants_results"][const_name]
            planck_const = planck_results["constants_results"][const_name]
            
            comparison["mathematical_constants_comparison"][const_name] = {
                "wmap_meta_coherence": wmap_const["meta_coherence"],
                "planck_meta_coherence": planck_const["meta_coherence"],
                "wmap_p_value": wmap_const["p_value"],
                "planck_p_value": planck_const["p_value"],
                "wmap_phi_optimality": wmap_const["phi_optimality"],
                "planck_phi_optimality": planck_const["phi_optimality"],
                "wmap_significant": wmap_const["significant"],
                "planck_significant": planck_const["significant"],
                "relative_difference": abs(wmap_const["meta_coherence"] - planck_const["meta_coherence"]) / 
                                     max(wmap_const["meta_coherence"], planck_const["meta_coherence"]) 
                                     if max(wmap_const["meta_coherence"], planck_const["meta_coherence"]) > 0 else 0
            }
    
    # Generate summary text report
    comparison_path = os.path.join(comparison_dir, "extended_meta_coherence_comparison.txt")
    with open(comparison_path, 'w') as f:
        f.write("Extended Meta-Coherence Test Comparison: WMAP vs Planck\n")
        f.write("====================================================\n\n")
        
        f.write("Meta-Coherence Comparison by Method:\n")
        for method_key, method_comp in comparison["meta_coherence_comparison"].items():
            f.write(f"  {method_key}:\n")
            f.write(f"    WMAP Meta-Coherence: {method_comp['wmap_meta_coherence']:.6f} (p={method_comp['wmap_p_value']:.6f})\n")
            f.write(f"    Planck Meta-Coherence: {method_comp['planck_meta_coherence']:.6f} (p={method_comp['planck_p_value']:.6f})\n")
            f.write(f"    WMAP Significant: {method_comp['wmap_significant']}\n")
            f.write(f"    Planck Significant: {method_comp['planck_significant']}\n\n")
        
        f.write("Mathematical Constants Meta-Coherence Comparison:\n")
        # Sort by relative difference
        sorted_constants = sorted(
            comparison["mathematical_constants_comparison"].items(),
            key=lambda x: x[1]["relative_difference"],
            reverse=True
        )
        
        for const_name, const_comp in sorted_constants:
            f.write(f"  {const_name}:\n")
            f.write(f"    WMAP Meta-Coherence: {const_comp['wmap_meta_coherence']:.6f} (p={const_comp['wmap_p_value']:.6f})\n")
            f.write(f"    Planck Meta-Coherence: {const_comp['planck_meta_coherence']:.6f} (p={const_comp['planck_p_value']:.6f})\n")
            f.write(f"    Relative Difference: {const_comp['relative_difference']:.2%}\n")
            f.write(f"    WMAP Significant: {const_comp['wmap_significant']}\n")
            f.write(f"    Planck Significant: {const_comp['planck_significant']}\n\n")
        
        # Overall conclusion
        wmap_significant_methods = sum(1 for m in comparison["meta_coherence_comparison"].values() if m["wmap_significant"])
        planck_significant_methods = sum(1 for m in comparison["meta_coherence_comparison"].values() if m["planck_significant"])
        wmap_significant_constants = sum(1 for c in comparison["mathematical_constants_comparison"].values() if c["wmap_significant"])
        planck_significant_constants = sum(1 for c in comparison["mathematical_constants_comparison"].values() if c["planck_significant"])
        
        f.write("\nSummary:\n")
        f.write(f"  WMAP has significant meta-coherence in {wmap_significant_methods} out of {len(comparison['meta_coherence_comparison'])} methods.\n")
        f.write(f"  Planck has significant meta-coherence in {planck_significant_methods} out of {len(comparison['meta_coherence_comparison'])} methods.\n")
        f.write(f"  WMAP has significant mathematical constant meta-coherence in {wmap_significant_constants} out of {len(comparison['mathematical_constants_comparison'])} constants.\n")
        f.write(f"  Planck has significant mathematical constant meta-coherence in {planck_significant_constants} out of {len(comparison['mathematical_constants_comparison'])} constants.\n")
    
    # Create comparison bar charts
    plt.figure(figsize=(14, 8))
    
    # Plot meta-coherence comparison by method
    method_keys = list(comparison["meta_coherence_comparison"].keys())
    wmap_values = [comparison["meta_coherence_comparison"][key]["wmap_meta_coherence"] for key in method_keys]
    planck_values = [comparison["meta_coherence_comparison"][key]["planck_meta_coherence"] for key in method_keys]
    
    x = np.arange(len(method_keys))
    width = 0.35
    
    plt.subplot(2, 1, 1)
    bars1 = plt.bar(x - width/2, wmap_values, width, label='WMAP')
    bars2 = plt.bar(x + width/2, planck_values, width, label='Planck')
    
    plt.xlabel('Method')
    plt.ylabel('Meta-Coherence Value')
    plt.title('Meta-Coherence Comparison by Method')
    plt.xticks(x, method_keys, rotation=45, ha='right')
    plt.legend()
    
    # Add significance indicators
    for i, key in enumerate(method_keys):
        if comparison["meta_coherence_comparison"][key]["wmap_significant"]:
            plt.text(i - width/2, wmap_values[i] + 0.01, '*', ha='center', va='bottom', fontsize=12)
        if comparison["meta_coherence_comparison"][key]["planck_significant"]:
            plt.text(i + width/2, planck_values[i] + 0.01, '*', ha='center', va='bottom', fontsize=12)
    
    # Plot mathematical constants comparison
    const_names = list(comparison["mathematical_constants_comparison"].keys())
    wmap_const_values = [comparison["mathematical_constants_comparison"][name]["wmap_meta_coherence"] for name in const_names]
    planck_const_values = [comparison["mathematical_constants_comparison"][name]["planck_meta_coherence"] for name in const_names]
    
    x = np.arange(len(const_names))
    
    plt.subplot(2, 1, 2)
    bars1 = plt.bar(x - width/2, wmap_const_values, width, label='WMAP')
    bars2 = plt.bar(x + width/2, planck_const_values, width, label='Planck')
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Meta-Coherence Value')
    plt.title('Meta-Coherence Comparison by Mathematical Constant')
    plt.xticks(x, const_names, rotation=45, ha='right')
    plt.legend()
    
    # Add significance indicators
    for i, name in enumerate(const_names):
        if comparison["mathematical_constants_comparison"][name]["wmap_significant"]:
            plt.text(i - width/2, wmap_const_values[i] + 0.01, '*', ha='center', va='bottom', fontsize=12)
        if comparison["mathematical_constants_comparison"][name]["planck_significant"]:
            plt.text(i + width/2, planck_const_values[i] + 0.01, '*', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "extended_meta_coherence_comparison.png"), dpi=300)
    plt.close()
    
    logger.info(f"Comparison complete. Results saved to {comparison_path}")
    
    return comparison


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extended Meta-Coherence Test for WMAP and Planck CMB data"
    )
    parser.add_argument("--wmap-file", type=str, required=True,
                       help="Path to WMAP power spectrum file")
    parser.add_argument("--planck-file", type=str, required=True,
                       help="Path to Planck power spectrum file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save results (default: 'extended_meta_coherence_<timestamp>')")
    parser.add_argument("--n-simulations", type=int, default=10000,
                       help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--window-sizes", type=str, default="10,20,30",
                       help="Comma-separated list of window sizes (default: '10,20,30')")
    parser.add_argument("--step-size", type=int, default=5,
                       help="Step size for sliding window (default: 5)")
    parser.add_argument("--num-processes", type=int, default=None,
                       help="Number of processes for parallel computation (default: all available cores)")
    
    args = parser.parse_args()
    
    # Process arguments
    window_sizes = [int(s) for s in args.window_sizes.split(',')]
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("results", f"extended_meta_coherence_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load WMAP data
    print(f"Loading WMAP data from {args.wmap_file}")
    ell_wmap, power_wmap, error_wmap = load_wmap_power_spectrum(args.wmap_file)
    if ell_wmap is None:
        print(f"Error loading WMAP data from {args.wmap_file}")
        return 1
    
    # Load Planck data
    print(f"Loading Planck data from {args.planck_file}")
    ell_planck, power_planck, error_planck = load_planck_power_spectrum(args.planck_file)
    if ell_planck is None:
        print(f"Error loading Planck data from {args.planck_file}")
        return 1
    
    # Preprocess data
    wmap_data = preprocess_data(power_wmap, smooth=True, smooth_window=3, normalize=True, detrend=True)
    planck_data = preprocess_data(power_planck, smooth=True, smooth_window=3, normalize=True, detrend=True)
    
    # Define local coherence and meta-coherence methods to use
    local_coherence_methods = ["inverse_std", "autocorr", "hurst"]
    meta_coherence_methods = ["cv", "entropy", "autocorr"]
    
    # Define mathematical constants to analyze
    constants = {
        "golden_ratio": (1 + np.sqrt(5)) / 2,  # ~1.618
        "sqrt2": np.sqrt(2),                   # ~1.414
        "sqrt3": np.sqrt(3),                   # ~1.732
        "pi": np.pi,                           # ~3.142
        "e": np.e,                             # ~2.718
        "ln2": np.log(2),                      # ~0.693
    }
    
    # Run analysis on WMAP data
    print("\nRunning extended meta-coherence analysis on WMAP data...")
    wmap_results = run_extended_meta_coherence_test(
        wmap_data, args.output_dir, "WMAP", n_simulations=args.n_simulations,
        window_sizes=window_sizes, step_size=args.step_size,
        local_coherence_methods=local_coherence_methods,
        meta_coherence_methods=meta_coherence_methods,
        constants=constants, num_processes=args.num_processes
    )
    
    # Run analysis on Planck data
    print("\nRunning extended meta-coherence analysis on Planck data...")
    planck_results = run_extended_meta_coherence_test(
        planck_data, args.output_dir, "Planck", n_simulations=args.n_simulations,
        window_sizes=window_sizes, step_size=args.step_size,
        local_coherence_methods=local_coherence_methods,
        meta_coherence_methods=meta_coherence_methods,
        constants=constants, num_processes=args.num_processes
    )
    
    # Compare results
    print("\nComparing WMAP and Planck results...")
    compare_results(wmap_results, planck_results, args.output_dir)
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
