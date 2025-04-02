"""
Unit tests for the data validation module.
"""

import os
import json
import pytest
import tempfile
import numpy as np
import sys
from pathlib import Path

# Import the mock from conftest.py
from conftest import mock_healpy

# Mock the healpy module before importing data_validation
sys.modules['healpy'] = mock_healpy

# Import the module to test
try:
    from wmap_data.data_validation import DataValidator
except ImportError:
    # If the module is not installed, try importing directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from wmap_data.data_validation import DataValidator


@pytest.mark.unit
class TestDataValidator:
    """Test suite for the DataValidator class."""

    def test_init(self):
        """Test initialization of DataValidator."""
        validator = DataValidator()
        assert hasattr(validator, 'default_config')
        assert hasattr(validator, 'validation_results')
        assert isinstance(validator.validation_results, dict)
        assert 'passed' in validator.validation_results

    def test_validate_power_spectrum(self, test_power_spectrum, test_metadata):
        """Test validating a power spectrum."""
        validator = DataValidator()
        
        # Validate the power spectrum
        result = validator.validate_power_spectrum(test_power_spectrum, test_metadata)
        
        # Check that the validation passed
        assert result['passed'] is True
        assert len(result['errors']) == 0

    def test_validate_map_data(self, test_cmb_map, test_metadata):
        """Test validating a HEALPix map."""
        validator = DataValidator()
        
        # Validate the map data
        result = validator.validate_map_data(test_cmb_map, test_metadata)
        
        # Check that the validation passed
        assert result['passed'] is True
        assert len(result['errors']) == 0
        
        # Test with invalid map data
        invalid_map = test_cmb_map.copy()
        invalid_map[0] = np.nan  # Add NaN values
        
        result = validator.validate_map_data(invalid_map, test_metadata)
        
        # Check that the validation failed
        assert result['passed'] is False
        assert len(result['errors']) > 0

    def test_validate_simulation_data(self, test_simulation_data, test_metadata):
        """Test validating simulation data."""
        validator = DataValidator()
        
        # Create proper simulation data structure expected by the validator
        sim_data = {
            'simulations': test_simulation_data['simulations'],
            'parameters': test_simulation_data['parameters']
        }
        
        # Add required metadata fields
        complete_metadata = test_metadata.copy()
        complete_metadata.update({
            'seed': 12345,
            'model': 'test_model',
            'parameters': {'param1': 1, 'param2': 2}
        })
        
        # Validate the simulation data
        result = validator.validate_simulation_data(sim_data, complete_metadata)
        
        # Check that the validation passed
        assert result['passed'] is True
        assert len(result['errors']) == 0
        
        # Test with invalid simulation data (missing simulations key)
        invalid_sim = {}
        
        result = validator.validate_simulation_data(invalid_sim, complete_metadata)
        
        # Check that the validation failed
        assert result['passed'] is False
        assert len(result['errors']) > 0

    def test_validate_file_exists(self, test_data_file):
        """Test validating file existence."""
        validator = DataValidator()
        
        # Validate that the file exists
        result = validator.validate_file_exists(test_data_file)
        
        # Check that the validation passed
        assert result['passed'] is True
        assert len(result['errors']) == 0
        
        # Test with nonexistent file
        result = validator.validate_file_exists("nonexistent_file.txt")
        
        # Check that the validation failed
        assert result['passed'] is False
        assert len(result['errors']) > 0

    def test_validate_file_format(self, test_data_file):
        """Test validating file format."""
        validator = DataValidator()
        
        # Validate the file format
        result = validator.validate_file_format(test_data_file, expected_format="npy")
        
        # Check that the validation passed
        assert result['passed'] is True
        assert len(result['errors']) == 0
        
        # Test with incorrect format
        result = validator.validate_file_format(test_data_file, expected_format="fits")
        
        # Check that the validation failed
        assert result['passed'] is False
        assert len(result['errors']) > 0

    def test_validate_data_consistency(self, test_cmb_map):
        """Test validating data consistency."""
        validator = DataValidator()
        
        # Create a copy of the data for consistency check
        consistency_check = test_cmb_map.copy()
        
        # Validate data consistency
        result = validator.validate_data_consistency(test_cmb_map, consistency_check)
        
        # Check that the validation passed
        assert result['passed'] is True
        assert len(result['errors']) == 0
        
        # Test with inconsistent data
        inconsistent_data = test_cmb_map.copy() * 2  # Multiply by 2 to make it inconsistent
        
        result = validator.validate_data_consistency(test_cmb_map, inconsistent_data)
        
        # Check that the validation failed
        assert result['passed'] is False
        assert len(result['errors']) > 0

    def test_validate_data_range(self, test_cmb_map):
        """Test validating data range."""
        validator = DataValidator()
        
        # Get the min and max values of the test data
        min_val = np.min(test_cmb_map)
        max_val = np.max(test_cmb_map)
        
        # Validate data range
        result = validator.validate_data_range(
            test_cmb_map,
            min_value=min_val - 1,  # Set range slightly wider than actual data
            max_value=max_val + 1
        )
        
        # Check that the validation passed
        assert result['passed'] is True
        assert len(result['errors']) == 0
        
        # Test with out-of-range data
        result = validator.validate_data_range(
            test_cmb_map,
            min_value=min_val + 1,  # Set min higher than actual min
            max_value=max_val + 1
        )
        
        # Check that the validation failed
        assert result['passed'] is False
        assert len(result['errors']) > 0

    def test_validate_with_custom_function(self, test_cmb_map):
        """Test validating with a custom function."""
        validator = DataValidator()
        
        # Define custom validation functions
        def custom_pass(data):
            """Custom validation function that always passes."""
            return True
            
        def custom_fail(data):
            """Custom validation function that always fails."""
            return False
            
        def custom_dict(data):
            """Custom validation function that returns a dict."""
            return {'test': 'value'}
            
        def custom_exception(data):
            """Custom validation function that raises an exception."""
            raise ValueError("Test exception")
            
        # Test with function that passes
        result = validator.validate_with_custom_function(
            test_cmb_map, 
            custom_pass
        )
        
        assert result['passed'] is True
        assert len(result['errors']) == 0
        
        # Test with function that fails
        result = validator.validate_with_custom_function(
            test_cmb_map, 
            custom_fail
        )
        
        assert result['passed'] is False
        assert len(result['errors']) > 0
        
        # Test with function that returns a dict
        result = validator.validate_with_custom_function(
            test_cmb_map, 
            custom_dict
        )
        
        assert result['passed'] is True
        assert len(result['errors']) == 0
        
        # Test with function that raises an exception
        result = validator.validate_with_custom_function(
            test_cmb_map, 
            custom_exception
        )
        
        assert result['passed'] is False
        assert len(result['errors']) > 0

    def test_generate_validation_report(self):
        """Test generating a validation report."""
        validator = DataValidator()
        
        # Add some validation results
        validator.validation_results = {
            'passed': True,
            'errors': [],
            'warnings': ['This is a warning'],
            'info': ['This is an info message']
        }
        
        # Generate reports in different formats
        text_report = validator.generate_validation_report(output_format="text")
        json_report = validator.generate_validation_report(output_format="json")
        html_report = validator.generate_validation_report(output_format="html")
        
        # Check that the reports were generated
        assert isinstance(text_report, str)
        assert "PASSED" in text_report
        assert "WARNING" in text_report
        assert "INFO" in text_report
        
        assert isinstance(json_report, str)
        assert "passed" in json_report
        assert "warnings" in json_report
        assert "info" in json_report
        
        assert isinstance(html_report, str)
        assert "PASSED" in html_report
        assert "warning" in html_report
        assert "info" in html_report

    def test_reset_validation_results(self):
        """Test resetting validation results."""
        validator = DataValidator()
        
        # Add some validation results
        validator.validation_results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Reset the results
        validator._reset_validation_results()
        
        # Check that the results were reset
        assert validator.validation_results == {
            'passed': False,
            'errors': [],
            'warnings': [],
            'info': []
        }

    def test_validate_metadata(self):
        """Test validating metadata."""
        validator = DataValidator()
        
        # Create test metadata
        metadata = {
            'title': 'Test Data',
            'description': 'Test description',
            'creator': 'Test User',
            'creation_date': '2023-01-01'
        }
        
        # Define required fields
        required_fields = ['title', 'description', 'creator']
        
        # Reset validation results
        validator._reset_validation_results()
        
        # Validate the metadata
        validator._validate_metadata(metadata, required_fields)
        
        # Check that the validation passed (no errors added)
        assert len(validator.validation_results['errors']) == 0
        
        # Test with missing required field
        incomplete_metadata = metadata.copy()
        del incomplete_metadata['title']
        
        # Reset validation results
        validator._reset_validation_results()
        
        # Validate the metadata
        validator._validate_metadata(incomplete_metadata, required_fields)
        
        # Check that the validation failed (errors added)
        assert len(validator.validation_results['errors']) > 0
