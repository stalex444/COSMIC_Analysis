#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test module for GR-specific coherence analysis.

This module validates the GR-specific coherence analysis by comparing
the results with expected values from the research paper.
"""

from __future__ import print_function, division
import os
import sys
import json
import numpy as np
import unittest
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the module to test
from test_gr_specific_coherence import (
    load_wmap_power_spectrum,
    find_golden_ratio_pairs,
    calculate_coherence,
    run_monte_carlo,
    run_gr_specific_coherence_test
)

def safe_makedirs(directory):
    """Create directory if it doesn't exist (compatible with older Python versions)."""
    if not os.path.exists(directory):
        os.makedirs(directory)

class TestGRCoherence(unittest.TestCase):
    """Test cases for GR-specific coherence analysis."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Define paths to test data
        cls.data_dir = os.path.join(parent_dir, "data")
        cls.wmap_data_path = os.path.join(cls.data_dir, "wmap_tt_spectrum_9yr_v5.txt")
        
        # Load data if available
        if os.path.exists(cls.wmap_data_path):
            cls.ell, cls.power, cls.error = load_wmap_power_spectrum(cls.wmap_data_path)
            cls.data_loaded = cls.ell is not None and cls.power is not None
        else:
            cls.data_loaded = False
            print("Warning: Test data not found at %s" % cls.wmap_data_path)
        
        # Expected values from the research paper
        cls.expected_values = {
            "gr_coherence": 0.896,  # Expected GR-specific coherence
            "p_value": 0.00001,     # Expected p-value (should be less than this)
            "phi_optimality": 6.0,  # Expected phi optimality (sigma value)
            "tolerance": 0.05       # Tolerance for numerical comparisons (5%)
        }
        
        # Save test results
        cls.results = {}
    
    def test_data_loading(self):
        """Test that data can be loaded correctly."""
        self.assertTrue(self.data_loaded, "Failed to load WMAP data")
        self.assertIsNotNone(self.ell, "Multipole moments (ell) not loaded")
        self.assertIsNotNone(self.power, "Power spectrum not loaded")
        self.assertIsNotNone(self.error, "Error values not loaded")
        
        # Basic sanity checks on the data
        self.assertGreater(len(self.ell), 0, "No multipole moments loaded")
        self.assertEqual(len(self.ell), len(self.power), "Mismatch between ell and power array lengths")
        self.assertEqual(len(self.ell), len(self.error), "Mismatch between ell and error array lengths")
    
    def test_find_golden_ratio_pairs(self):
        """Test that golden ratio pairs can be found."""
        if not self.data_loaded:
            self.skipTest("Data not loaded, skipping test")
        
        # Find golden ratio pairs with a small max_ell for quick testing
        max_ell = 500
        max_pairs = 20
        gr_pairs = find_golden_ratio_pairs(
            self.ell, 
            max_ell=max_ell, 
            max_pairs=max_pairs, 
            use_efficient=True,
            timeout_seconds=10
        )
        
        # Check that pairs were found
        self.assertIsNotNone(gr_pairs, "Golden ratio pairs not found")
        self.assertGreater(len(gr_pairs), 0, "No golden ratio pairs found")
        
        # Check that pairs follow the golden ratio relationship
        golden_ratio = (1 + np.sqrt(5)) / 2  # ~1.618
        tolerance = 0.05
        
        for ell1, ell2 in gr_pairs:
            ratio = ell2 / ell1
            self.assertLess(abs(ratio - golden_ratio), tolerance, 
                           "Pair (%s, %s) does not have golden ratio relationship" % (ell1, ell2))
        
        # Save result
        self.results["gr_pairs_count"] = len(gr_pairs)
    
    def test_calculate_coherence(self):
        """Test the calculation of GR-specific coherence."""
        if not self.data_loaded:
            self.skipTest("Data not loaded, skipping test")
        
        # Find golden ratio pairs
        max_ell = 500
        max_pairs = 20
        gr_pairs = find_golden_ratio_pairs(
            self.ell, 
            max_ell=max_ell, 
            max_pairs=max_pairs, 
            use_efficient=True,
            timeout_seconds=10
        )
        
        # Calculate coherence
        coherence_values, mean_coherence = calculate_coherence(
            self.power, self.ell, gr_pairs, max_pairs_to_process=20
        )
        
        # Check that coherence was calculated
        self.assertIsNotNone(coherence_values, "Coherence values not calculated")
        self.assertIsNotNone(mean_coherence, "Mean coherence not calculated")
        self.assertGreater(len(coherence_values), 0, "No coherence values calculated")
        
        # Save result
        self.results["mean_coherence"] = mean_coherence
    
    def test_monte_carlo_significance(self):
        """Test the Monte Carlo significance assessment."""
        if not self.data_loaded:
            self.skipTest("Data not loaded, skipping test")
        
        # Run a reduced version of the Monte Carlo simulation for testing
        p_value, phi_optimality, actual_coherence, sim_coherences, gr_pairs, coherence_values = run_monte_carlo(
            self.power, self.ell, n_simulations=10, max_ell=500, use_efficient=True, max_pairs=20
        )
        
        # Check that results were calculated
        self.assertIsNotNone(p_value, "p-value not calculated")
        self.assertIsNotNone(phi_optimality, "Phi optimality not calculated")
        self.assertIsNotNone(actual_coherence, "Actual coherence not calculated")
        self.assertIsNotNone(sim_coherences, "Simulation coherences not calculated")
        
        # Check that p-value is within expected range
        self.assertLessEqual(p_value, 1.0, "p-value exceeds 1.0")
        self.assertGreaterEqual(p_value, 0.0, "p-value is negative")
        
        # Save results
        self.results["p_value"] = p_value
        self.results["phi_optimality"] = phi_optimality
        self.results["actual_coherence"] = actual_coherence
    
    def test_full_gr_coherence_analysis(self):
        """Test the full GR-specific coherence analysis pipeline."""
        if not self.data_loaded:
            self.skipTest("Data not loaded, skipping test")
        
        # Create a temporary output directory
        output_dir = os.path.join(parent_dir, "test_output")
        safe_makedirs(output_dir)
        
        # Run the full analysis with reduced parameters for testing
        results = run_gr_specific_coherence_test(
            self.ell, self.power, output_dir, "test_wmap", 
            n_simulations=5, max_ell=500, use_efficient=True, max_pairs=10
        )
        
        # Check that results were calculated
        self.assertIsNotNone(results, "No results returned from analysis")
        self.assertIn("p_value", results, "p-value not in results")
        self.assertIn("phi_optimality", results, "Phi optimality not in results")
        self.assertIn("mean_coherence", results, "Mean coherence not in results")
        
        # Save results
        self.results["full_analysis"] = {
            "p_value": results["p_value"],
            "phi_optimality": results["phi_optimality"],
            "actual_coherence": results.get("mean_coherence", 0.0)
        }
    
    def test_compare_with_expected_values(self):
        """Compare calculated values with expected values from the research paper."""
        if not self.data_loaded or not hasattr(self, "results") or not self.results:
            self.skipTest("Previous tests did not run successfully, skipping comparison")
        
        # Get actual values from previous tests
        actual_coherence = self.results.get("actual_coherence")
        p_value = self.results.get("p_value")
        phi_optimality = self.results.get("phi_optimality")
        
        if actual_coherence is None or p_value is None or phi_optimality is None:
            self.skipTest("Required values not calculated in previous tests")
        
        # Compare with expected values
        expected_coherence = self.expected_values["gr_coherence"]
        expected_p_value = self.expected_values["p_value"]
        expected_phi_optimality = self.expected_values["phi_optimality"]
        tolerance = self.expected_values["tolerance"]
        
        # Check coherence
        self.assertLess(
            abs(actual_coherence - expected_coherence) / expected_coherence, 
            tolerance,
            "GR coherence (%s) differs from expected (%s) by more than %s%%" % (
                actual_coherence, expected_coherence, tolerance*100)
        )
        
        # Check p-value (should be less than expected)
        self.assertLessEqual(
            p_value, 
            expected_p_value,
            "p-value (%s) is greater than expected (%s)" % (p_value, expected_p_value)
        )
        
        # Check phi optimality (should be greater than expected)
        self.assertGreaterEqual(
            phi_optimality, 
            expected_phi_optimality,
            "Phi optimality (%s) is less than expected (%s)" % (phi_optimality, expected_phi_optimality)
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests and save results."""
        # Save test results to a JSON file
        if hasattr(cls, "results") and cls.results:
            output_dir = os.path.join(parent_dir, "test_output")
            safe_makedirs(output_dir)
            
            results_file = os.path.join(output_dir, "gr_coherence_test_results.json")
            with open(results_file, 'w') as f:
                json.dump({
                    "test_results": cls.results,
                    "expected_values": cls.expected_values,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
            print("Test results saved to %s" % results_file)

if __name__ == "__main__":
    unittest.main()
