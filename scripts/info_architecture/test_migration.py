#!/usr/bin/env python
"""
Quick Test Runner for Information Architecture Test After Migration

This script runs a small version of the Information Architecture Test 
to verify that the code works properly in the new repository structure.
"""

import os
import sys
import time
from datetime import datetime

# Import the main module
from cmb_info_architecture import run_information_architecture_test

def main():
    """Run a quick test of the information architecture code."""
    start_time = time.time()
    
    # Set up paths
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    wmap_data_file = os.path.join(repo_root, "data/wmap/wmap_tt_spectrum_9yr_v5.txt")
    output_dir = os.path.join(repo_root, "results/info_architecture/migration_test")
    
    # Check if data file exists
    if not os.path.exists(wmap_data_file):
        print(f"Error: WMAP data file not found at {wmap_data_file}")
        return 1
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"Running Information Architecture Test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data file: {wmap_data_file}")
    print(f"Output directory: {output_dir}")
    print(f"Running with 50 simulations (reduced for testing)")
    
    # Define the constants to test
    constants = {
        'phi': (1 + 5**0.5) / 2,  # Golden ratio
        'sqrt2': 2**0.5,          # Square root of 2
        'sqrt3': 3**0.5,          # Square root of 3
        'ln2': np.log(2),         # Natural log of 2
        'e': np.e,                # Euler's number
        'pi': np.pi               # Pi
    }
    
    # Run the test with reduced simulations for quick verification
    results = run_information_architecture_test(
        data_file=wmap_data_file,
        data_type='power_spectrum',
        constants=constants,
        n_simulations=50,  # Small number for quick testing
        scale_method='conventional',
        output_dir=output_dir,
        debug=True
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.2f} seconds")
    
    # Display key results
    if results:
        print("\nKey Results:")
        for constant, score in results['architecture_scores'].items():
            p_value = results['p_values'][constant]
            print(f"  - {constant}: Score = {score:.4f}, p-value = {p_value:.6f}")
        
        print(f"  - Phi Optimality: {results['phi_optimality']:.4f}")
        
        # Show a few key scales with high scores
        print("\nHighest Scoring Scales:")
        for constant in constants.keys():
            if 'scale_scores' in results and constant in results['scale_scores']:
                scale_scores = results['scale_scores'][constant]
                if scale_scores:
                    max_scale = max(scale_scores.items(), key=lambda x: x[1])
                    print(f"  - {constant}: Scale {max_scale[0]} = {max_scale[1]:.4f}")
    
    return 0

if __name__ == "__main__":
    import numpy as np  # Import here to ensure compatibility with run_information_architecture_test
    sys.exit(main())
