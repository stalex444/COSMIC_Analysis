#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Information Architecture Test with 10,000 simulations

This script runs the Information Architecture Test on both WMAP and Planck
cosmic microwave background data with 10,000 Monte Carlo simulations.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Import the Information Architecture Test
from scripts.info_architecture.archive.information_architecture_test import InformationArchitectureTest, load_wmap_power_spectrum, load_planck_power_spectrum

def ensure_dir_exists(dir_path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == "__main__":
    # Configuration
    config = {
        'num_simulations': 10000,
        'constants': {
            'phi': 1.61803398875,
            'sqrt2': 1.41421356237,
            'sqrt3': 1.73205080757,
            'ln2': 0.693147180559945,
            'e': 2.71828182846,
            'pi': 3.14159265359
        },
        'parallel_processing': True,
        'early_stopping': True,
        'min_simulations': 1000,
        'significance_threshold': 0.01,
        'max_workers': 8,  # Adjust based on available CPU cores
        'output_dir': "../results/information_architecture_10k"
    }
    
    # Create output directory
    ensure_dir_exists(config['output_dir'])
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create test instance
    ia_test = InformationArchitectureTest(config)
    
    # Run test on WMAP data
    print("\n" + "="*50)
    print("Running Information Architecture Test on WMAP data (%d simulations)" % config['num_simulations'])
    print("="*50)
    
    # Load WMAP data
    wmap_data = load_wmap_power_spectrum()
    
    # Run test for WMAP data
    wmap_output_dir = os.path.join(config['output_dir'], 'wmap')
    ensure_dir_exists(wmap_output_dir)
    
    wmap_results = ia_test.run_full_test(
        wmap_data, 
        constants=config['constants'],
        num_simulations=config['num_simulations'],
        output_dir=wmap_output_dir
    )
    
    # Save WMAP results summary
    ia_test.generate_summary_report(wmap_results, wmap_output_dir)
    ia_test.visualize_results(wmap_results, wmap_output_dir)
    
    # Run test on Planck data
    print("\n" + "="*50)
    print("Running Information Architecture Test on Planck data (%d simulations)" % config['num_simulations'])
    print("="*50)
    
    # Load Planck data
    planck_data = load_planck_power_spectrum()
    
    # Run test for Planck data
    planck_output_dir = os.path.join(config['output_dir'], 'planck')
    ensure_dir_exists(planck_output_dir)
    
    planck_results = ia_test.run_full_test(
        planck_data, 
        constants=config['constants'],
        num_simulations=config['num_simulations'],
        output_dir=planck_output_dir
    )
    
    # Save Planck results summary
    ia_test.generate_summary_report(planck_results, planck_output_dir)
    ia_test.visualize_results(planck_results, planck_output_dir)
    
    # Print overall summary
    print("\n" + "="*50)
    print("INFORMATION ARCHITECTURE TEST RESULTS SUMMARY")
    print("="*50)
    
    print("\nWMAP Results:")
    for constant_name, result in wmap_results.items():
        significance = "Significant" if result['significant'] else "Not Significant"
        print("- %s: Score %.6f, p-value %.6f (%s)" % 
              (constant_name, result['actual_score'], result['p_value'], significance))
    
    print("\nPlanck Results:")
    for constant_name, result in planck_results.items():
        significance = "Significant" if result['significant'] else "Not Significant"
        print("- %s: Score %.6f, p-value %.6f (%s)" % 
              (constant_name, result['actual_score'], result['p_value'], significance))
    
    print("\nTest completed successfully. Results saved to: %s" % config['output_dir'])
