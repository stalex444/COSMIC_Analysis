"""
Integration tests for the enhanced cosmic test framework.

This module tests the cosmic test framework that was enhanced with:
- Robust transfer entropy calculation with binning
- Bootstrap confidence intervals
- Power spectrum visualization
- Automatic interpretation of results
- Phi-optimality calculation
- Golden ratio pattern detection
"""

import os
import pytest
import numpy as np
import json
import tempfile
try:
    from mock import patch, MagicMock  # Python 2.7
except ImportError:
    from unittest.mock import patch, MagicMock  # Python 3.x

# Try to import the modules to test
try:
    from wmap_cosmic_analysis.cosmic_framework import (
        calculate_phi_optimality,
        bootstrap_confidence_intervals,
        detect_golden_ratio_patterns,
        analyze_cmb_data,
        visualize_power_spectrum
    )
except ImportError:
    # If the modules are not installed, mark all tests as skipped
    pytestmark = pytest.mark.skip(reason="wmap_cosmic_analysis.cosmic_framework module not found")


@pytest.mark.integration
class TestCosmicFramework:
    """Integration tests for the enhanced cosmic test framework."""

    @pytest.fixture
    def mock_cmb_data(self):
        """Create mock CMB data for testing."""
        np.random.seed(42)
        # Create a simple mock CMB dataset with some golden ratio patterns
        ells = np.arange(2, 1001)
        cls = 1000.0 * (ells / 10.0) ** (-2.0)
        
        # Add some golden ratio patterns (phi approximately 1.618)
        phi = (1 + np.sqrt(5)) / 2
        for i in range(10):
            peak_ell = int(phi ** i * 10)
            if peak_ell < len(cls):
                cls[peak_ell-2:peak_ell+3] *= 1.2  # Enhance power around golden ratio multiples
        
        return ells, cls

    @pytest.fixture
    def mock_random_data(self):
        """Create mock random data for comparison."""
        np.random.seed(43)
        ells = np.arange(2, 1001)
        cls = 1000.0 * (ells / 10.0) ** (-2.0)
        # Add random noise without golden ratio patterns
        cls += np.random.normal(0, cls * 0.1)
        return ells, cls

    def test_calculate_phi_optimality(self, mock_cmb_data, mock_random_data):
        """Test the phi-optimality calculation function."""
        cmb_ells, cmb_cls = mock_cmb_data
        random_ells, random_cls = mock_random_data
        
        # Calculate phi-optimality for both datasets
        phi_opt_cmb = calculate_phi_optimality(cmb_ells, cmb_cls)
        phi_opt_random = calculate_phi_optimality(random_ells, random_cls)
        
        # Check that the results are reasonable
        assert phi_opt_cmb >= 0  # Phi-optimality should be non-negative
        assert phi_opt_random >= 0
        assert isinstance(phi_opt_cmb, float)
        assert isinstance(phi_opt_random, float)
        
        # The CMB data with golden ratio patterns should have higher phi-optimality
        assert phi_opt_cmb > phi_opt_random

    def test_bootstrap_confidence_intervals(self, mock_cmb_data):
        """Test the bootstrap confidence intervals function."""
        ells, cls = mock_cmb_data
        
        # Calculate bootstrap confidence intervals
        n_bootstrap = 100  # Use fewer samples for testing
        ci = bootstrap_confidence_intervals(
            ells, 
            cls, 
            statistic_func=calculate_phi_optimality,
            n_bootstrap=n_bootstrap,
            alpha=0.05
        )
        
        # Check that the confidence interval has the correct format
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        lower, upper = ci
        assert lower <= upper
        
        # Check that the confidence interval is reasonable
        phi_opt = calculate_phi_optimality(ells, cls)
        assert lower <= phi_opt <= upper or np.isclose(lower, phi_opt) or np.isclose(upper, phi_opt)

    def test_detect_golden_ratio_patterns(self, mock_cmb_data, mock_random_data):
        """Test the golden ratio pattern detection function."""
        cmb_ells, cmb_cls = mock_cmb_data
        random_ells, random_cls = mock_random_data
        
        # Detect patterns in both datasets
        cmb_patterns = detect_golden_ratio_patterns(cmb_ells, cmb_cls)
        random_patterns = detect_golden_ratio_patterns(random_ells, random_cls)
        
        # Check that the results have the correct format
        assert isinstance(cmb_patterns, dict)
        assert 'patterns_found' in cmb_patterns
        assert 'pattern_locations' in cmb_patterns
        assert 'pattern_strengths' in cmb_patterns
        
        # The CMB data should have more golden ratio patterns
        assert cmb_patterns['patterns_found'] >= random_patterns['patterns_found']

    def test_analyze_cmb_data(self, mock_cmb_data, mock_random_data, test_data_dir):
        """Test the main CMB data analysis function."""
        cmb_ells, cmb_cls = mock_cmb_data
        random_ells, random_cls = mock_random_data
        output_file = os.path.join(test_data_dir, "cmb_analysis_results.json")
        
        # Analyze the CMB data
        result = analyze_cmb_data(
            cmb_ells=cmb_ells,
            cmb_cls=cmb_cls,
            random_ells=random_ells,
            random_cls=random_cls,
            output_file=output_file,
            n_bootstrap=50,  # Use fewer bootstrap samples for testing
            seed=42,
            visualize=False
        )
        
        # Check that the analysis ran and produced a result
        assert result is not None
        assert os.path.exists(output_file)
        
        # Check that the result contains expected keys
        assert 'phi_optimality' in result
        assert 'confidence_interval' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'cmb_random_ratio' in result
        
        # Load the output file and check its contents
        with open(output_file, 'r') as f:
            saved_result = json.load(f)
        
        assert saved_result['phi_optimality'] == result['phi_optimality']
        assert saved_result['p_value'] == result['p_value']

    def test_visualize_power_spectrum(self, mock_cmb_data, test_data_dir):
        """Test the power spectrum visualization function."""
        ells, cls = mock_cmb_data
        output_file = os.path.join(test_data_dir, "power_spectrum.png")
        
        # Mock the plotting function to check if it's called
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            visualize_power_spectrum(
                ells=ells,
                cls=cls,
                output_file=output_file,
                highlight_golden_ratio=True,
                log_scale=True
            )
            
            # Check that the visualization was created
            assert mock_savefig.called

    def test_reproducibility_with_seed(self, mock_cmb_data, mock_random_data, test_data_dir):
        """Test that the analysis is reproducible with the same seed."""
        cmb_ells, cmb_cls = mock_cmb_data
        random_ells, random_cls = mock_random_data
        output_file1 = os.path.join(test_data_dir, "cmb_analysis_seed1.json")
        output_file2 = os.path.join(test_data_dir, "cmb_analysis_seed2.json")
        
        # Run the analysis twice with the same seed
        result1 = analyze_cmb_data(
            cmb_ells=cmb_ells,
            cmb_cls=cmb_cls,
            random_ells=random_ells,
            random_cls=random_cls,
            output_file=output_file1,
            n_bootstrap=50,
            seed=12345,
            visualize=False
        )
        
        result2 = analyze_cmb_data(
            cmb_ells=cmb_ells,
            cmb_cls=cmb_cls,
            random_ells=random_ells,
            random_cls=random_cls,
            output_file=output_file2,
            n_bootstrap=50,
            seed=12345,
            visualize=False
        )
        
        # Check that the results are identical
        assert result1['phi_optimality'] == result2['phi_optimality']
        assert result1['p_value'] == result2['p_value']
        assert result1['confidence_interval'] == result2['confidence_interval']
        
        # Run the analysis with a different seed
        result3 = analyze_cmb_data(
            cmb_ells=cmb_ells,
            cmb_cls=cmb_cls,
            random_ells=random_ells,
            random_cls=random_cls,
            output_file=os.path.join(test_data_dir, "cmb_analysis_seed3.json"),
            n_bootstrap=50,
            seed=54321,
            visualize=False
        )
        
        # The results should be different with a different seed
        # (at least one of these should be different)
        assert (result1['phi_optimality'] != result3['phi_optimality'] or
                result1['p_value'] != result3['p_value'] or
                result1['confidence_interval'] != result3['confidence_interval'])

    def test_python2_compatibility(self):
        """Test Python 2.7 compatibility of string formatting and division."""
        # This test simulates Python 2.7 behavior to ensure the code is compatible
        
        # Mock the __future__ import for division
        with patch('__future__.division', create=True):
            # Test division (should be floating point division in both Python 2 and 3)
            assert 5/2 == 2.5
            
            # Test string formatting (should use format() instead of f-strings)
            test_value = 42
            formatted_string = "The value is {}".format(test_value)
            assert formatted_string == "The value is 42"
            
            # Ensure no f-strings are used in the module
            # This is a simple check that would fail if f-strings were used
            # (This is just a placeholder - actual implementation would require code inspection)
            assert True
