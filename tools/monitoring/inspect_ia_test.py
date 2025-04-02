#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inspect Information Architecture Test
This script inspects the Information Architecture Test to identify performance bottlenecks.
"""

import os
import sys
import time
import numpy as np
from scripts.info_architecture.archive.information_architecture_test import InformationArchitectureTest, load_wmap_power_spectrum

def main():
    """Main function to inspect the Information Architecture Test"""
    print("Inspecting Information Architecture Test...")
    
    # Load WMAP data
    print("Loading WMAP data...")
    start_time = time.time()
    wmap_data = load_wmap_power_spectrum()
    load_time = time.time() - start_time
    print("WMAP data loaded: %d data points in %.2f seconds" % (len(wmap_data), load_time))
    
    # Create test instance
    config = {}
    test = InformationArchitectureTest(config)
    
    # Test calculate_architecture_score
    print("\nTesting calculate_architecture_score...")
    constant = 1.61803398875  # Golden Ratio (phi)
    
    start_time = time.time()
    score = test.calculate_architecture_score(wmap_data, constant)
    calc_time = time.time() - start_time
    print("Architecture score: %s (calculated in %.2f seconds)" % (score, calc_time))
    
    # Test single simulation
    print("\nTesting single simulation...")
    start_time = time.time()
    
    # Generate surrogate data (shuffle the original data)
    np.random.seed(42)
    surrogate_data = np.random.permutation(wmap_data)
    
    # Calculate architecture score for surrogate data
    surrogate_score = test.calculate_architecture_score(surrogate_data, constant)
    sim_time = time.time() - start_time
    print("Surrogate score: %s (calculated in %.2f seconds)" % (surrogate_score, sim_time))
    
    # Estimate time for 10,000 simulations
    est_time_10k = sim_time * 10000
    est_time_10k_parallel = est_time_10k / 8  # Assuming 8 cores
    
    print("\nPerformance Estimates:")
    print("Single simulation time: %.2f seconds" % sim_time)
    print("Estimated time for 10,000 simulations (sequential): %.2f seconds (%.2f hours)" % 
          (est_time_10k, est_time_10k / 3600))
    print("Estimated time for 10,000 simulations (8 cores): %.2f seconds (%.2f hours)" % 
          (est_time_10k_parallel, est_time_10k_parallel / 3600))
    
    # Check if there are any optimizations we can make
    print("\nPotential Optimizations:")
    
    # Check if we're recalculating the same values repeatedly
    print("1. Caching intermediate results could improve performance")
    print("2. Reducing I/O operations (less frequent progress updates)")
    print("3. Optimizing the architecture score calculation")
    print("4. Using numpy vectorized operations where possible")
    
    # Check the implementation of calculate_architecture_score
    print("\nInspecting calculate_architecture_score implementation...")
    try:
        source_code = inspect.getsource(test.calculate_architecture_score)
        print(source_code)
    except:
        print("Could not inspect source code. Please check the implementation manually.")
    
    # Check the implementation of measure_information_architecture
    print("\nInspecting measure_information_architecture implementation...")
    try:
        source_code = inspect.getsource(test.measure_information_architecture)
        print(source_code)
    except:
        print("Could not inspect source code. Please check the implementation manually.")
    
    print("\nInspection complete.")

if __name__ == "__main__":
    main()
