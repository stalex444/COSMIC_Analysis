#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runner script for the predictive power test.

This script executes the predictive power test with 10,000 simulations
on both WMAP and Planck CMB data.
"""

import os
import sys
import time
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictive_power.test_predictive_power import run_predictive_power_test
from predictive_power.utilities import load_power_spectrum

def main():
    """
    Main function to run the predictive power test with command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run predictive power test on CMB data")
    parser.add_argument("--wmap", action="store_true", help="Run test on WMAP data")
    parser.add_argument("--planck", action="store_true", help="Run test on Planck data")
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations (default: 10000)")
    parser.add_argument("--tolerance", type=float, default=0.08, 
                      help="Base tolerance for matching (default: 0.08)")
    parser.add_argument("--forward", type=int, default=6,
                      help="Number of forward predictions (default: 6)")
    parser.add_argument("--backward", type=int, default=6,
                      help="Number of backward predictions (default: 6)")
    parser.add_argument("--output", type=str, default="results/predictive_power", 
                      help="Output directory (default: results/predictive_power)")
    
    args = parser.parse_args()
    
    # Set default to run both datasets if none specified
    if not (args.wmap or args.planck):
        args.wmap = True
        args.planck = True
    
    start_time = time.time()
    print("\n====================================================")
    print("PREDICTIVE POWER TEST")
    print("Started at: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("====================================================")
    print("Number of simulations: {}".format(args.sims))
    print("Base tolerance: {:.2f}".format(args.tolerance))
    print("Forward predictions: {}".format(args.forward))
    print("Backward predictions: {}".format(args.backward))
    print("Output directory: {}".format(args.output))
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Create timestamp for output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Track results
    results = {}
    
    # Run test on WMAP data
    if args.wmap:
        wmap_start = time.time()
        wmap_file = "data/wmap/wmap_tt_spectrum_9yr_v5.txt"
        
        if not os.path.exists(wmap_file):
            print("Error: WMAP data file not found at {}".format(wmap_file))
            print("Checking alternative locations...")
            
            alt_locations = [
                "../data/wmap_tt_spectrum_9yr_v5.txt",
                "data/wmap_tt_spectrum_9yr_v5.txt"
            ]
            
            for alt in alt_locations:
                if os.path.exists(alt):
                    wmap_file = alt
                    print("Found WMAP data at {}".format(wmap_file))
                    break
            else:
                print("No WMAP data file found. Skipping WMAP analysis.")
                args.wmap = False
        
        if args.wmap:
            print("\n========== ANALYZING WMAP DATA ==========")
            wmap_output_dir = os.path.join(args.output, "wmap_{}".format(timestamp))
            
            # Load WMAP data
            print("Loading WMAP power spectrum...")
            ell, power = load_power_spectrum(wmap_file)
            print("Loaded {} data points".format(len(ell)))
            
            # Run test
            results["wmap"] = run_predictive_power_test(
                ell, 
                power, 
                n_simulations=args.sims,
                base_tolerance=args.tolerance,
                n_forward=args.forward,
                n_backward=args.backward,
                output_dir=wmap_output_dir
            )
            print("WMAP analysis completed in {:.2f} minutes".format((time.time() - wmap_start) / 60))
    
    # Run test on Planck data
    if args.planck:
        planck_start = time.time()
        planck_file = "data/planck/planck_tt_spectrum_2018.txt"
        
        if not os.path.exists(planck_file):
            print("Error: Planck data file not found at {}".format(planck_file))
            print("Checking alternative locations...")
            
            alt_locations = [
                "../data/planck_tt_spectrum_2018.txt",
                "data/planck_tt_spectrum_2018.txt"
            ]
            
            for alt in alt_locations:
                if os.path.exists(alt):
                    planck_file = alt
                    print("Found Planck data at {}".format(planck_file))
                    break
            else:
                print("No Planck data file found. Skipping Planck analysis.")
                args.planck = False
        
        if args.planck:
            print("\n========== ANALYZING PLANCK DATA ==========")
            planck_output_dir = os.path.join(args.output, "planck_{}".format(timestamp))
            
            # Load Planck data
            print("Loading Planck power spectrum...")
            ell, power = load_power_spectrum(planck_file)
            print("Loaded {} data points".format(len(ell)))
            
            # Run test
            results["planck"] = run_predictive_power_test(
                ell, 
                power, 
                n_simulations=args.sims,
                base_tolerance=args.tolerance,
                n_forward=args.forward,
                n_backward=args.backward,
                output_dir=planck_output_dir
            )
            print("Planck analysis completed in {:.2f} minutes".format((time.time() - planck_start) / 60))
    
    # Compare results if both datasets were analyzed
    if args.wmap and args.planck:
        print("\n========== COMPARING WMAP AND PLANCK RESULTS ==========")
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        metrics = ["gr_match_rate", "p_value"]
        titles = ["GR Match Rate", "p-value"]
        colors = ['blue', 'red']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(2, 2, i+1)
            values = [results["wmap"][metric], results["planck"][metric]]
            bars = plt.bar(["WMAP", "Planck"], values, color=colors, alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      '{:.4f}'.format(height), ha='center', va='bottom')
            
            if metric == "p_value":
                plt.axhline(y=0.05, color='black', linestyle='--', alpha=0.5)
            
            plt.ylabel(title)
            plt.title(title)
        
        # Plot match ratio comparison
        plt.subplot(2, 2, 3)
        wmap_ratio = results["wmap"]["gr_match_rate"] / np.mean(results["wmap"]["random_match_rates"])
        planck_ratio = results["planck"]["gr_match_rate"] / np.mean(results["planck"]["random_match_rates"])
        values = [wmap_ratio, planck_ratio]
        
        bars = plt.bar(["WMAP", "Planck"], values, color=colors, alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                  '{:.2f}x'.format(height), ha='center', va='bottom')
        
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        plt.ylabel("Match Ratio")
        plt.title("GR vs Random Match Ratio\n(Higher = Better Prediction)")
        
        # Save comparison figure
        comparison_output = os.path.join(args.output, "comparison_{}.png".format(timestamp))
        plt.tight_layout()
        plt.savefig(comparison_output)
        print("Comparison plot saved to: {}".format(comparison_output))
    
    print("\n====================================================")
    print("PREDICTIVE POWER TEST SUMMARY")
    print("====================================================")
    print("Total run time: {:.2f} minutes".format((time.time() - start_time) / 60))
    
    if args.wmap:
        print("\nWMAP Results:")
        wmap_data = results["wmap"]
        print("  Golden Ratio Match Rate: {:.4f}".format(wmap_data["gr_match_rate"]))
        print("  Mean Random Match Rate:  {:.4f}".format(np.mean(wmap_data["random_match_rates"])))
        print("  Match Ratio: {:.4f}x".format(wmap_data["gr_match_rate"] / np.mean(wmap_data["random_match_rates"])))
        print("  p-value: {:.4f}".format(wmap_data["p_value"]))
    
    if args.planck:
        print("\nPlanck Results:")
        planck_data = results["planck"]
        print("  Golden Ratio Match Rate: {:.4f}".format(planck_data["gr_match_rate"]))
        print("  Mean Random Match Rate:  {:.4f}".format(np.mean(planck_data["random_match_rates"])))
        print("  Match Ratio: {:.4f}x".format(planck_data["gr_match_rate"] / np.mean(planck_data["random_match_rates"])))
        print("  p-value: {:.4f}".format(planck_data["p_value"]))
    
    print("\nResults saved to: {}".format(args.output))
    print("\nAnalysis completed at: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

if __name__ == "__main__":
    main()
