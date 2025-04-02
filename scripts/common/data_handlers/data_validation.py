"""
Data Validation Module for WMAP Cosmic Analysis

This module provides tools for validating data used in the WMAP Cosmic Analysis framework.
It includes functions for checking data integrity, format, and consistency to ensure
reliable analysis results.
"""

import os
import numpy as np
import logging
import json
# Import typing but don't use type annotations in function signatures for Python 2.7 compatibility
from typing import Dict, List, Union, Optional, Any, Tuple, Callable

# Try to import healpy, but provide a fallback if it's not available
try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    # Create a mock healpy module for basic functionality
    class MockHealpy:
        @staticmethod
        def nside2npix(nside):
            return 12 * nside * nside
    
    hp = MockHealpy()
    HEALPY_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Class for validating data in the WMAP Cosmic Analysis framework.
    
    This class provides methods for checking data integrity, format, and consistency
    to ensure reliable analysis results.
    """
    
    def __init__(self, validation_config=None):
        """
        Initialize the DataValidator object.
        
        Args:
            validation_config (dict, optional): Configuration for validation rules. If None, uses default rules.
        """
        # Default validation configuration
        self.default_config = {
            'power_spectrum': {
                'min_ell': 2,
                'max_ell': 2500,
                'min_value': 0,
                'max_value': 10000,
                'allowed_nans': 0,
                'allowed_infs': 0,
                'required_metadata': ['source', 'version', 'date']
            },
            'map_data': {
                'nside_values': [32, 64, 128, 256, 512, 1024, 2048],
                'allowed_nans': 0,
                'allowed_infs': 0,
                'required_metadata': ['source', 'version', 'date', 'nside', 'ordering']
            },
            'simulation': {
                'min_simulations': 10,
                'max_simulations': 10000,
                'required_metadata': ['seed', 'model', 'parameters']
            }
        }
        
        # Use provided configuration or default
        self.validation_config = validation_config or self.default_config
        
        # Initialize validation results
        self.validation_results = {
            'passed': False,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
    def validate_power_spectrum(self, data, metadata=None):
        """
        Validate a power spectrum dataset.
        
        Args:
            data (numpy.ndarray): Power spectrum data as a numpy array
            metadata (dict, optional): Metadata associated with the power spectrum
            
        Returns:
            dict: Dictionary with validation results
        """
        # Reset validation results
        self._reset_validation_results()
        
        # Get configuration for power spectrum validation
        config = self.validation_config['power_spectrum']
        
        # Check data type
        if not isinstance(data, np.ndarray):
            self.validation_results['errors'].append("Data is not a numpy array: {}".format(type(data)))
            
        # Check data shape
        if len(data.shape) != 1:
            self.validation_results['errors'].append("Power spectrum should be 1D, got shape {}".format(data.shape))
            
        # Check data length
        if len(data) < config['min_ell']:
            self.validation_results['errors'].append("Power spectrum too short: {} < {}".format(len(data), config['min_ell']))
            
        if len(data) > config['max_ell']:
            self.validation_results['warnings'].append("Power spectrum unusually long: {} > {}".format(len(data), config['max_ell']))
            
        # Check for NaN values
        nan_count = np.isnan(data).sum()
        if nan_count > config['allowed_nans']:
            self.validation_results['errors'].append("Power spectrum contains {} NaN values (max allowed: {})".format(nan_count, config['allowed_nans']))
            
        # Check for infinite values
        inf_count = np.isinf(data).sum()
        if inf_count > config['allowed_infs']:
            self.validation_results['errors'].append("Power spectrum contains {} infinite values (max allowed: {})".format(inf_count, config['allowed_infs']))
            
        # Check value range
        if np.any(data < config['min_value']):
            self.validation_results['warnings'].append("Power spectrum contains values below {}".format(config['min_value']))
            
        if np.any(data > config['max_value']):
            self.validation_results['warnings'].append("Power spectrum contains values above {}".format(config['max_value']))
            
        # Check metadata if provided
        if metadata:
            self._validate_metadata(metadata, config['required_metadata'])
            
        # Set passed flag
        self.validation_results['passed'] = len(self.validation_results['errors']) == 0
        
        return self.validation_results
        
    def validate_map_data(self, map_data, metadata=None):
        """
        Validate a HEALPix map dataset.
        
        Args:
            map_data (numpy.ndarray): HEALPix map data as a numpy array
            metadata (dict, optional): Metadata associated with the map
            
        Returns:
            dict: Dictionary with validation results
        """
        # Reset validation results
        self._reset_validation_results()
        
        # Get configuration for map data validation
        config = self.validation_config['map_data']
        
        # Check data type
        if not isinstance(map_data, np.ndarray):
            self.validation_results['errors'].append("Data is not a numpy array: {}".format(type(map_data)))
            
        # Check if data length is valid for a HEALPix map
        npix = len(map_data)
        valid_npix = False
        valid_nside = None
        
        for nside in config['nside_values']:
            if npix == hp.nside2npix(nside):
                valid_npix = True
                valid_nside = nside
                break
                
        if not valid_npix:
            self.validation_results['errors'].append("Map size {} is not valid for a HEALPix map".format(npix))
        else:
            self.validation_results['info'].append("Map has nside = {}".format(valid_nside))
            
        # Check for NaN values
        nan_count = np.isnan(map_data).sum()
        if nan_count > config['allowed_nans']:
            self.validation_results['errors'].append("Map contains {} NaN values (max allowed: {})".format(nan_count, config['allowed_nans']))
            
        # Check for infinite values
        inf_count = np.isinf(map_data).sum()
        if inf_count > config['allowed_infs']:
            self.validation_results['errors'].append("Map contains {} infinite values (max allowed: {})".format(inf_count, config['allowed_infs']))
            
        # Check metadata if provided
        if metadata:
            self._validate_metadata(metadata, config['required_metadata'])
            
            # Check if nside in metadata matches the detected nside
            if 'nside' in metadata and valid_nside is not None:
                if metadata['nside'] != valid_nside:
                    self.validation_results['errors'].append("Metadata nside ({}) does not match detected nside ({})".format(metadata['nside'], valid_nside))
                    
        # Set passed flag
        self.validation_results['passed'] = len(self.validation_results['errors']) == 0
        
        return self.validation_results
        
    def validate_simulation_data(self, data, metadata=None):
        """
        Validate simulation data.
        
        Args:
            data (dict): Simulation data as a dictionary
            metadata (dict, optional): Metadata associated with the simulation
            
        Returns:
            dict: Dictionary with validation results
        """
        # Reset validation results
        self._reset_validation_results()
        
        # Get configuration for simulation validation
        config = self.validation_config['simulation']
        
        # Check data type
        if not isinstance(data, dict):
            self.validation_results['errors'].append("Data is not a dictionary: {}".format(type(data)))
            return self.validation_results
            
        # Check if required keys are present
        required_keys = ['simulations', 'parameters']
        for key in required_keys:
            if key not in data:
                self.validation_results['errors'].append("Required key '{}' not found in simulation data".format(key))
                
        # Check number of simulations
        if 'simulations' in data:
            num_simulations = len(data['simulations'])
            if num_simulations < config['min_simulations']:
                self.validation_results['errors'].append("Too few simulations: {} < {}".format(num_simulations, config['min_simulations']))
                
            if num_simulations > config['max_simulations']:
                self.validation_results['warnings'].append("Unusually large number of simulations: {} > {}".format(num_simulations, config['max_simulations']))
                
        # Check metadata if provided
        if metadata:
            self._validate_metadata(metadata, config['required_metadata'])
            
        # Set passed flag
        self.validation_results['passed'] = len(self.validation_results['errors']) == 0
        
        return self.validation_results
        
    def validate_file_exists(self, file_path):
        """
        Validate that a file exists.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict: Dictionary with validation results
        """
        # Reset validation results
        self._reset_validation_results()
        
        # Check if file exists
        if not os.path.exists(file_path):
            self.validation_results['errors'].append("File does not exist: {}".format(file_path))
        else:
            self.validation_results['info'].append("File exists: {}".format(file_path))
            
            # Check if file is readable
            try:
                with open(file_path, 'r') as f:
                    pass
                self.validation_results['info'].append("File is readable: {}".format(file_path))
            except Exception as e:
                self.validation_results['errors'].append("File is not readable: {} ({})".format(file_path, str(e)))
                
        # Set passed flag
        self.validation_results['passed'] = len(self.validation_results['errors']) == 0
        
        return self.validation_results
        
    def validate_file_format(self, file_path, expected_format):
        """
        Validate that a file has the expected format.
        
        Args:
            file_path (str): Path to the file
            expected_format (str): Expected file format (extension)
            
        Returns:
            dict: Dictionary with validation results
        """
        # Reset validation results
        self._reset_validation_results()
        
        # Check if file exists
        if not os.path.exists(file_path):
            self.validation_results['errors'].append("File does not exist: {}".format(file_path))
            return self.validation_results
            
        # Check file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip('.')
        
        if ext != expected_format.lower():
            self.validation_results['errors'].append("File has wrong format: expected {}, got {}".format(expected_format, ext))
        else:
            self.validation_results['info'].append("File has correct format: {}".format(expected_format))
            
        # Set passed flag
        self.validation_results['passed'] = len(self.validation_results['errors']) == 0
        
        return self.validation_results
        
    def validate_with_custom_function(self, data, validation_func, func_args=None):
        """
        Validate data using a custom validation function.
        
        Args:
            data: Data to validate
            validation_func: Custom validation function
            func_args (dict, optional): Additional arguments for the validation function
            
        Returns:
            dict: Dictionary with validation results
        """
        # Reset validation results
        self._reset_validation_results()
        
        # Call the custom validation function
        try:
            func_args = func_args or {}
            result = validation_func(data, **func_args)
            
            # If the function returns a boolean, interpret it as pass/fail
            if isinstance(result, bool):
                if result:
                    self.validation_results['info'].append("Custom validation passed")
                else:
                    self.validation_results['errors'].append("Custom validation failed")
                    
            # If the function returns a dictionary, merge it with our results
            elif isinstance(result, dict):
                for key in ['errors', 'warnings', 'info']:
                    if key in result and isinstance(result[key], list):
                        self.validation_results[key].extend(result[key])
                        
            # Otherwise, assume the function raised exceptions for failures
            else:
                self.validation_results['info'].append("Custom validation passed")
                
        except Exception as e:
            self.validation_results['errors'].append("Custom validation failed: {}".format(str(e)))
            
        # Set passed flag
        self.validation_results['passed'] = len(self.validation_results['errors']) == 0
        
        return self.validation_results
        
    def validate_data_consistency(self, data1, data2, tolerance=1e-6):
        """
        Validate consistency between two datasets.
        
        Args:
            data1 (numpy.ndarray): First dataset
            data2 (numpy.ndarray): Second dataset
            tolerance (float, optional): Tolerance for numerical differences
            
        Returns:
            dict: Dictionary with validation results
        """
        # Reset validation results
        self._reset_validation_results()
        
        # Check data types
        if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
            self.validation_results['errors'].append("Both datasets must be numpy arrays")
            return self.validation_results
            
        # Check data shapes
        if data1.shape != data2.shape:
            self.validation_results['errors'].append("Datasets have different shapes: {} vs {}".format(data1.shape, data2.shape))
            return self.validation_results
            
        # Check data consistency
        diff = np.abs(data1 - data2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        if max_diff > tolerance:
            self.validation_results['errors'].append("Datasets differ by more than tolerance: max diff = {}, tolerance = {}".format(max_diff, tolerance))
        else:
            self.validation_results['info'].append("Datasets are consistent: max diff = {}, mean diff = {}".format(max_diff, mean_diff))
            
        # Set passed flag
        self.validation_results['passed'] = len(self.validation_results['errors']) == 0
        
        return self.validation_results
        
    def validate_data_range(self, data, min_value, max_value):
        """
        Validate that data values are within a specified range.
        
        Args:
            data (numpy.ndarray): Data to validate
            min_value (float): Minimum allowed value
            max_value (float): Maximum allowed value
            
        Returns:
            dict: Dictionary with validation results
        """
        # Reset validation results
        self._reset_validation_results()
        
        # Check data type
        if not isinstance(data, np.ndarray):
            self.validation_results['errors'].append("Data is not a numpy array: {}".format(type(data)))
            return self.validation_results
            
        # Check data range
        below_min = np.sum(data < min_value)
        above_max = np.sum(data > max_value)
        
        if below_min > 0:
            self.validation_results['errors'].append("{} values below minimum ({})".format(below_min, min_value))
            
        if above_max > 0:
            self.validation_results['errors'].append("{} values above maximum ({})".format(above_max, max_value))
            
        if below_min == 0 and above_max == 0:
            self.validation_results['info'].append("All values within range [{}, {}]".format(min_value, max_value))
            
        # Set passed flag
        self.validation_results['passed'] = len(self.validation_results['errors']) == 0
        
        return self.validation_results
        
    def generate_validation_report(self, output_format='text'):
        """
        Generate a report of the validation results.
        
        Args:
            output_format (str, optional): Format of the report ('text', 'json', or 'html')
            
        Returns:
            str: Report as a string in the specified format
        """
        if output_format == 'text':
            return self._generate_text_report()
        elif output_format == 'json':
            return json.dumps(self.validation_results, indent=2)
        elif output_format == 'html':
            return self._generate_html_report()
        else:
            raise ValueError("Unsupported output format: {}".format(output_format))
            
    def _generate_text_report(self):
        """
        Generate a text report of the validation results.
        
        Returns:
            str: Text report as a string
        """
        report = ["DATA VALIDATION REPORT\n"]
        
        # Add overall status
        status = "PASSED" if self.validation_results['passed'] else "FAILED"
        report.append("Status: {}\n".format(status))
        
        # Add errors
        if self.validation_results['errors']:
            report.append("ERRORS:")
            for i, error in enumerate(self.validation_results['errors']):
                report.append("  {}. {}".format(i+1, error))
            report.append("")
            
        # Add warnings
        if self.validation_results['warnings']:
            report.append("WARNINGS:")
            for i, warning in enumerate(self.validation_results['warnings']):
                report.append("  {}. {}".format(i+1, warning))
            report.append("")
            
        # Add info
        if self.validation_results['info']:
            report.append("INFO:")
            for i, info in enumerate(self.validation_results['info']):
                report.append("  {}. {}".format(i+1, info))
            report.append("")
            
        return "\n".join(report)
        
    def _generate_html_report(self):
        """
        Generate an HTML report of the validation results.
        
        Returns:
            str: HTML report as a string
        """
        status = "PASSED" if self.validation_results['passed'] else "FAILED"
        status_color = "green" if self.validation_results['passed'] else "red"
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .status {{ font-weight: bold; color: {status_color}; }}
                .error {{ color: red; }}
                .warning {{ color: orange; }}
                .info {{ color: blue; }}
                ul {{ list-style-type: none; padding-left: 20px; }}
            </style>
        </head>
        <body>
            <h1>Data Validation Report</h1>
            <p>Status: <span class="status">{status}</span></p>
        """.format(status=status, status_color=status_color)
        
        # Add errors
        if self.validation_results['errors']:
            html += "<h2>Errors</h2><ul>"
            for error in self.validation_results['errors']:
                html += "<li class='error'>{}</li>".format(error)
            html += "</ul>"
            
        # Add warnings
        if self.validation_results['warnings']:
            html += "<h2>Warnings</h2><ul>"
            for warning in self.validation_results['warnings']:
                html += "<li class='warning'>{}</li>".format(warning)
            html += "</ul>"
            
        # Add info
        if self.validation_results['info']:
            html += "<h2>Info</h2><ul>"
            for info in self.validation_results['info']:
                html += "<li class='info'>{}</li>".format(info)
            html += "</ul>"
            
        html += """
        </body>
        </html>
        """
        
        return html
        
    def _reset_validation_results(self):
        """
        Reset the validation results.
        """
        self.validation_results = {
            'passed': False,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
    def _validate_metadata(self, metadata, required_keys):
        """
        Validate metadata by checking for required keys.
        
        Args:
            metadata (dict): Metadata to validate
            required_keys (list): List of required keys
        """
        # Check if metadata is a dictionary
        if not isinstance(metadata, dict):
            self.validation_results['errors'].append("Metadata is not a dictionary: {}".format(type(metadata)))
            return
            
        # Check for required keys
        for key in required_keys:
            if key not in metadata:
                self.validation_results['errors'].append("Required metadata key '{}' not found".format(key))
            elif metadata[key] is None or (isinstance(metadata[key], str) and metadata[key].strip() == ''):
                self.validation_results['errors'].append("Metadata key '{}' has empty value".format(key))


# Example usage
if __name__ == "__main__":
    # Create a data validator
    validator = DataValidator()
    
    # Generate some test data
    power_spectrum = np.random.normal(size=2000)
    
    # Validate the power spectrum
    results = validator.validate_power_spectrum(power_spectrum, metadata={
        'source': 'WMAP',
        'version': 'WMAP9',
        'date': '2013-12-21'
    })
    
    # Generate a validation report
    report = validator.generate_validation_report()
    print(report)
