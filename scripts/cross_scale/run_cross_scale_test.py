#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runner script for the cross-scale correlation test.

This script executes the cross-scale correlation test with 10,000 simulations
on both WMAP and Planck CMB data.
"""

import os
import sys
import time
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cross_scale.test_cross_scale import run_cross_scale_test

def main():
    """
    Main function to run the cross-scale correlation test with command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run cross-scale correlation test on CMB data")
    parser.add_argument("--wmap", action="store_true", help="Run test on WMAP data")
    parser.add_argument("--planck", action="store_true", help="Run test on Planck data")
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations (default: 10000)")
    parser.add_argument("--window", type=int, default=5, help="Window size (default: 5)")
    parser.add_argument("--output", type=str, default="results/cross_scale", 
                      help="Output directory (default: results/cross_scale)")
    
    args = parser.parse_args()
    
    # Set default to run both datasets if none specified
    if not (args.wmap or args.planck):
        args.wmap = True
        args.planck = True
    
    start_time = time.time()
    print("\n====================================================")
    print("CROSS-SCALE CORRELATION TEST")
    print("Started at: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("====================================================")
    print("Number of simulations: {}".format(args.sims))
    print("Window size: {}".format(args.window))
    print("Output directory: {}".format(args.output))
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    results = {}
    
    # Run test on WMAP data
    if args.wmap:
        print("\n--- Running Cross-Scale Test on WMAP Data ---")
        wmap_file = "data/wmap/wmap_tt_spectrum_9yr_v5.txt"
        if not os.path.exists(wmap_file):
            print("Error: WMAP power spectrum file not found at {}".format(wmap_file))
            # Try alternative locations
            alternative_wmap_files = [
                "data/wmap_tt_spectrum_9yr_v5.txt",
                "../data/wmap/wmap_tt_spectrum_9yr_v5.txt",
                "../data/wmap_tt_spectrum_9yr_v5.txt"
            ]
            
            for alt_file in alternative_wmap_files:
                if os.path.exists(alt_file):
                    wmap_file = alt_file
                    print("Using alternative WMAP file: {}".format(wmap_file))
                    break
            else:
                print("No WMAP data file found. Skipping WMAP analysis.")
                args.wmap = False
        
        if args.wmap:
            wmap_output_dir = os.path.join(args.output, "wmap")
            wmap_start = time.time()
            results["wmap"] = run_cross_scale_test(
                wmap_file, 
                wmap_output_dir,
                n_simulations=args.sims,
                window_size=args.window
            )
            print("WMAP analysis completed in {:.2f} minutes".format((time.time() - wmap_start) / 60))
    
    # Run test on Planck data
    if args.planck:
        print("\n--- Running Cross-Scale Test on Planck Data ---")
        planck_file = "data/planck/planck_tt_spectrum_2018.txt"
        if not os.path.exists(planck_file):
            print("Error: Planck power spectrum file not found at {}".format(planck_file))
            # Try alternative locations
            alternative_planck_files = [
                "data/planck_tt_spectrum_2018.txt",
                "../data/planck/planck_tt_spectrum_2018.txt",
                "../data/planck_tt_spectrum_2018.txt",
                "data/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt",
                "../data/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt"
            ]
            
            for alt_file in alternative_planck_files:
                if os.path.exists(alt_file):
                    planck_file = alt_file
                    print("Using alternative Planck file: {}".format(planck_file))
                    break
            else:
                print("No Planck data file found. Skipping Planck analysis.")
                args.planck = False
        
        if args.planck:
            planck_output_dir = os.path.join(args.output, "planck")
            planck_start = time.time()
            results["planck"] = run_cross_scale_test(
                planck_file, 
                planck_output_dir,
                n_simulations=args.sims,
                window_size=args.window
            )
            print("Planck analysis completed in {:.2f} minutes".format((time.time() - planck_start) / 60))
    
    # Print summary
    print("\n====================================================")
    print("CROSS-SCALE CORRELATION TEST SUMMARY")
    print("====================================================")
    print("Total run time: {:.2f} minutes".format((time.time() - start_time) / 60))
    
    if "wmap" in results and results["wmap"]:
        wmap_results = results["wmap"]["results"]
        print("\nWMAP Results:")
        print("  Phi-Related Correlation: {:.6f}".format(wmap_results["mean_phi_corr"]))
        print("  Random Correlation: {:.6f}".format(wmap_results["mean_random_corr"]))
        print("  Correlation Ratio: {:.6f}x".format(wmap_results["corr_ratio"]))
        print("  p-value: {:.6f}".format(wmap_results["p_value"]))
        print("  Phi Optimality: {:.6f}".format(wmap_results["phi_optimality"]))
    
    if "planck" in results and results["planck"]:
        planck_results = results["planck"]["results"]
        print("\nPlanck Results:")
        print("  Phi-Related Correlation: {:.6f}".format(planck_results["mean_phi_corr"]))
        print("  Random Correlation: {:.6f}".format(planck_results["mean_random_corr"]))
        print("  Correlation Ratio: {:.6f}x".format(planck_results["corr_ratio"]))
        print("  p-value: {:.6f}".format(planck_results["p_value"]))
        print("  Phi Optimality: {:.6f}".format(planck_results["phi_optimality"]))
    
    print("\nResults saved to: {}".format(args.output))
    print("\nAnalysis completed at: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

if __name__ == "__main__":
    main()
