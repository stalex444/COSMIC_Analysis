#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scale Transition Test for WMAP and Planck CMB data.

This script implements the Scale Transition Test, which analyzes scale boundaries
where organizational principles change in the CMB power spectrum. The test
identifies transitions between different organizational regimes and analyzes
their alignment with mathematical constants, particularly the golden ratio.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json
import logging
import multiprocessing

# Import utility functions
from scale_transition_utils import (
    load_wmap_power_spectrum, load_planck_power_spectrum, preprocess_data,
    calculate_local_complexity, detect_scale_transitions, analyze_golden_ratio_alignment,
    run_monte_carlo_parallel, plot_scale_transition_results
)


def setup_logging(output_dir):
    """Set up logging configuration."""
    log_path = os.path.join(output_dir, "scale_transition.log")
    
    # Create a logger
    logger = logging.getLogger('ScaleTransition')
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
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting scale transition analysis for {name}")
    logger.info(f"Parameters: n_simulations={n_simulations}, window_size={window_size}, "
               f"n_clusters={n_clusters}")
    
    # Create dataset directory
    dataset_dir = os.path.join(output_dir, name.lower())
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define different window sizes to try
    window_sizes = [window_size]
    if window_size != 55:
        window_sizes.append(55)  # Add window size 55 which showed strong sqrt2 specialization
    
    # Store results for each window size
    all_results = {}
    
    for ws in window_sizes:
        logger.info(f"Running analysis with window size {ws}")
        
        # Run Monte Carlo simulations
        if parallel:
            (p_value, phi_optimality, transition_points, sim_n_transitions, complexity_values,
             window_centers, cluster_labels, alignment_score) = run_monte_carlo_parallel(
                ell, power, n_simulations=n_simulations, window_size=ws, n_clusters=n_clusters,
                timeout_seconds=timeout_seconds, num_processes=num_processes
            )
        else:
            # For simplicity, still use the parallel version but with 1 process
            (p_value, phi_optimality, transition_points, sim_n_transitions, complexity_values,
             window_centers, cluster_labels, alignment_score) = run_monte_carlo_parallel(
                ell, power, n_simulations=n_simulations, window_size=ws, n_clusters=n_clusters,
                timeout_seconds=timeout_seconds, num_processes=1
            )
        
        # Plot results
        plot_path = os.path.join(dataset_dir, f"scale_transition_w{ws}.png")
        plot_title = f"{name} Scale Transition Analysis (Window Size {ws})"
        
        plot_scale_transition_results(
            ell, power, complexity_values, window_centers, cluster_labels, transition_points,
            p_value, phi_optimality, sim_n_transitions, len(transition_points), alignment_score,
            plot_title, plot_path
        )
        
        # Store results
        results = {
            "window_size": ws,
            "p_value": p_value,
            "phi_optimality": phi_optimality,
            "transition_points": transition_points,
            "num_transitions": len(transition_points),
            "alignment_score": alignment_score,
            "significant": p_value < 0.05,
        }
        
        all_results[f"window_{ws}"] = results
        
        logger.info(f"Results for window size {ws}:")
        logger.info(f"  p-value: {p_value:.6f}")
        logger.info(f"  phi-optimality: {phi_optimality:.6f}")
        logger.info(f"  number of transitions: {len(transition_points)}")
        logger.info(f"  transition points: {transition_points}")
        logger.info(f"  alignment score: {alignment_score:.6f}")
        logger.info(f"  significant: {p_value < 0.05}")
    
    # Store all window size results
    all_results["dataset"] = name
    all_results["n_simulations"] = n_simulations
    
    # Generate summary text report
    summary_path = os.path.join(dataset_dir, f"{name.lower()}_scale_transition.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Scale Transition Test Results: {name} CMB Data\n")
        f.write("===========================================\n\n")
        
        for ws, results in all_results.items():
            if ws.startswith("window_"):
                f.write(f"Window Size {results['window_size']}:\n")
                f.write(f"  p-value: {results['p_value']:.6f}\n")
                f.write(f"  phi-optimality: {results['phi_optimality']:.6f}\n")
                f.write(f"  number of transitions: {results['num_transitions']}\n")
                f.write(f"  transition points: {results['transition_points']}\n")
                f.write(f"  alignment score: {results['alignment_score']:.6f}\n")
                f.write(f"  significant: {results['significant']}\n\n")
        
        f.write("\nAnalysis Parameters:\n")
        f.write(f"  Number of Simulations: {n_simulations}\n")
        f.write(f"  Base Window Size: {window_size}\n")
        f.write(f"  Number of Clusters: {n_clusters}\n")
    
    # Save full results as JSON
    results_json_path = os.path.join(dataset_dir, f"{name.lower()}_scale_transition_results.json")
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
        
        json_results = convert_for_json(all_results)
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {summary_path} and {results_json_path}")
    
    return all_results


def compare_results(wmap_results, planck_results, output_dir):
    """
    Compare scale transition test results between WMAP and Planck data.
    
    Args:
        wmap_results (dict): Results from WMAP data analysis
        planck_results (dict): Results from Planck data analysis
        output_dir (str): Directory to save comparison results
    """
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info("Comparing WMAP and Planck scale transition results")
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Initialize comparison results
    comparison = {
        "scale_transition_comparison": {},
        "window_size_comparison": {}
    }
    
    # Compare window size results
    for ws_key in wmap_results.keys():
        if ws_key.startswith("window_") and ws_key in planck_results:
            wmap_result = wmap_results[ws_key]
            planck_result = planck_results[ws_key]
            window_size = wmap_result["window_size"]
            
            comparison["window_size_comparison"][ws_key] = {
                "window_size": window_size,
                "wmap_p_value": wmap_result["p_value"],
                "planck_p_value": planck_result["p_value"],
                "wmap_phi_optimality": wmap_result["phi_optimality"],
                "planck_phi_optimality": planck_result["phi_optimality"],
                "wmap_num_transitions": wmap_result["num_transitions"],
                "planck_num_transitions": planck_result["num_transitions"],
                "wmap_alignment_score": wmap_result["alignment_score"],
                "planck_alignment_score": planck_result["alignment_score"],
                "wmap_significant": wmap_result["significant"],
                "planck_significant": planck_result["significant"],
            }
    
    # Generate summary text report
    comparison_path = os.path.join(comparison_dir, "scale_transition_comparison.txt")
    with open(comparison_path, 'w') as f:
        f.write("Scale Transition Test Comparison: WMAP vs Planck\n")
        f.write("=============================================\n\n")
        
        f.write("Comparison by Window Size:\n")
        for ws_key, ws_comp in comparison["window_size_comparison"].items():
            window_size = ws_comp["window_size"]
            f.write(f"  Window Size {window_size}:\n")
            f.write(f"    WMAP p-value: {ws_comp['wmap_p_value']:.6f}\n")
            f.write(f"    Planck p-value: {ws_comp['planck_p_value']:.6f}\n")
            f.write(f"    WMAP phi-optimality: {ws_comp['wmap_phi_optimality']:.6f}\n")
            f.write(f"    Planck phi-optimality: {ws_comp['planck_phi_optimality']:.6f}\n")
            f.write(f"    WMAP transitions: {ws_comp['wmap_num_transitions']}\n")
            f.write(f"    Planck transitions: {ws_comp['planck_num_transitions']}\n")
            f.write(f"    WMAP alignment score: {ws_comp['wmap_alignment_score']:.6f}\n")
            f.write(f"    Planck alignment score: {ws_comp['planck_alignment_score']:.6f}\n")
            f.write(f"    WMAP significant: {ws_comp['wmap_significant']}\n")
            f.write(f"    Planck significant: {ws_comp['planck_significant']}\n\n")
        
        # Overall conclusion
        wmap_significant = sum(1 for ws in comparison["window_size_comparison"].values() if ws["wmap_significant"])
        planck_significant = sum(1 for ws in comparison["window_size_comparison"].values() if ws["planck_significant"])
        
        f.write("\nSummary:\n")
        f.write(f"  WMAP has significant scale transitions in {wmap_significant} out of {len(comparison['window_size_comparison'])} window sizes.\n")
        f.write(f"  Planck has significant scale transitions in {planck_significant} out of {len(comparison['window_size_comparison'])} window sizes.\n")
        
        # Special focus on window size 55 which showed strong sqrt2 specialization
        if "window_55" in comparison["window_size_comparison"]:
            ws55 = comparison["window_size_comparison"]["window_55"]
            f.write("\nSpecial focus on window size 55 (previously showed strong sqrt2 specialization):\n")
            f.write(f"  WMAP p-value: {ws55['wmap_p_value']:.6f}\n")
            f.write(f"  Planck p-value: {ws55['planck_p_value']:.6f}\n")
            f.write(f"  WMAP significant: {ws55['wmap_significant']}\n")
            f.write(f"  Planck significant: {ws55['planck_significant']}\n")
    
    # Create comparison bar charts
    plt.figure(figsize=(14, 10))
    
    # Plot p-values comparison by window size
    window_sizes = [ws_comp["window_size"] for ws_comp in comparison["window_size_comparison"].values()]
    wmap_p_values = [ws_comp["wmap_p_value"] for ws_comp in comparison["window_size_comparison"].values()]
    planck_p_values = [ws_comp["planck_p_value"] for ws_comp in comparison["window_size_comparison"].values()]
    
    x = np.arange(len(window_sizes))
    width = 0.35
    
    plt.subplot(2, 1, 1)
    bars1 = plt.bar(x - width/2, wmap_p_values, width, label='WMAP')
    bars2 = plt.bar(x + width/2, planck_p_values, width, label='Planck')
    
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance threshold (p=0.05)')
    
    plt.xlabel('Window Size')
    plt.ylabel('P-value')
    plt.title('P-value Comparison by Window Size')
    plt.xticks(x, window_sizes)
    plt.legend()
    
    # Plot alignment scores comparison by window size
    wmap_align = [ws_comp["wmap_alignment_score"] for ws_comp in comparison["window_size_comparison"].values()]
    planck_align = [ws_comp["planck_alignment_score"] for ws_comp in comparison["window_size_comparison"].values()]
    
    plt.subplot(2, 1, 2)
    bars1 = plt.bar(x - width/2, wmap_align, width, label='WMAP')
    bars2 = plt.bar(x + width/2, planck_align, width, label='Planck')
    
    plt.xlabel('Window Size')
    plt.ylabel('Golden Ratio Alignment Score')
    plt.title('Alignment Score Comparison by Window Size')
    plt.xticks(x, window_sizes)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "scale_transition_comparison.png"), dpi=300)
    plt.close()
    
    logger.info(f"Comparison complete. Results saved to {comparison_path}")
    
    return comparison


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Scale Transition Test for WMAP and Planck CMB data"
    )
    parser.add_argument("--wmap-file", type=str, required=True,
                       help="Path to WMAP power spectrum file")
    parser.add_argument("--planck-file", type=str, required=True,
                       help="Path to Planck power spectrum file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save results (default: 'scale_transition_<timestamp>')")
    parser.add_argument("--n-simulations", type=int, default=10000,
                       help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--window-size", type=int, default=10,
                       help="Window size for complexity calculation (default: 10)")
    parser.add_argument("--n-clusters", type=int, default=3,
                       help="Number of clusters for transition detection (default: 3)")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Maximum time in seconds to spend on simulations (default: 3600)")
    parser.add_argument("--num-processes", type=int, default=None,
                       help="Number of processes for parallel computation (default: all available cores)")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel processing")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("results", f"scale_transition_{timestamp}")
    
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
    
    # Process arguments
    parallel = not args.no_parallel
    
    # Run analysis on WMAP data
    print("\nRunning scale transition analysis on WMAP data...")
    wmap_results = run_scale_transition_test(
        ell_wmap, power_wmap, args.output_dir, "WMAP", n_simulations=args.n_simulations,
        window_size=args.window_size, n_clusters=args.n_clusters, timeout_seconds=args.timeout,
        parallel=parallel, num_processes=args.num_processes
    )
    
    # Run analysis on Planck data
    print("\nRunning scale transition analysis on Planck data...")
    planck_results = run_scale_transition_test(
        ell_planck, power_planck, args.output_dir, "Planck", n_simulations=args.n_simulations,
        window_size=args.window_size, n_clusters=args.n_clusters, timeout_seconds=args.timeout,
        parallel=parallel, num_processes=args.num_processes
    )
    
    # Compare results
    print("\nComparing WMAP and Planck results...")
    compare_results(wmap_results, planck_results, args.output_dir)
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
