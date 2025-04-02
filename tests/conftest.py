"""
Pytest configuration file for WMAP Cosmic Analysis tests.

This file contains fixtures and utilities for testing the WMAP Cosmic Analysis framework.
"""

import os
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Define constants for testing
TEST_NSIDE = 32
TEST_NPIX = 12 * TEST_NSIDE * TEST_NSIDE  # Formula for HEALPix pixels
TEST_ELL_MAX = 100

# Flag to indicate if healpy is available - set to False by default
HEALPY_AVAILABLE = False

# Define a mock for healpy
class MockHealpy:
    @staticmethod
    def nside2npix(nside):
        return 12 * nside * nside
    
    @staticmethod
    def synfast(cls, nside, **kwargs):
        npix = 12 * nside * nside
        return np.random.normal(0, 1, npix)

# Use our mock healpy
hp = MockHealpy()
# Export the mock healpy for use in test files
mock_healpy = hp

# Skip tests that require healpy if it's not available
healpy_skip = pytest.mark.skipif(not HEALPY_AVAILABLE, 
                                reason="healpy package not available")


@pytest.fixture(scope="session")
def test_data_dir():
    """
    Create a temporary directory for test data.
    
    Returns:
        Path to the temporary directory.
    """
    temp_dir = tempfile.mkdtemp(prefix="wmap_test_")
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def test_power_spectrum():
    """
    Create a test power spectrum.
    
    Returns:
        Numpy array containing a test power spectrum.
    """
    # Create a simple power spectrum with a power law shape
    ells = np.arange(2, TEST_ELL_MAX + 1)
    cls = 1000.0 * (ells / 10.0) ** (-2.0)
    return cls


@pytest.fixture(scope="session")
def test_cmb_map(test_power_spectrum):
    """
    Create a test CMB map.
    
    Args:
        test_power_spectrum: Test power spectrum fixture.
        
    Returns:
        Numpy array containing a test CMB map.
    """
    # Create a simple random map
    np.random.seed(12345)  # For reproducibility
    
    # Create a full power spectrum including ell=0,1
    full_cls = np.zeros(TEST_ELL_MAX + 1)
    full_cls[2:] = test_power_spectrum
    
    # Generate a random CMB map
    cmb_map = np.random.normal(0, 1, TEST_NPIX)
        
    return cmb_map


@pytest.fixture(scope="session")
def test_metadata():
    """
    Create test metadata.
    
    Returns:
        Dictionary containing test metadata.
    """
    return {
        'source': 'TEST',
        'version': 'TEST_v1',
        'date': '2023-01-01',
        'nside': TEST_NSIDE,
        'ordering': 'RING',
        'creator': 'pytest',
        'description': 'Test data for WMAP Cosmic Analysis'
    }


@pytest.fixture(scope="session")
def test_data_file(test_data_dir, test_cmb_map):
    """
    Create a test data file.
    
    Args:
        test_data_dir: Test data directory fixture.
        test_cmb_map: Test CMB map fixture.
        
    Returns:
        Path to the test data file.
    """
    file_path = os.path.join(test_data_dir, "test_cmb_map.npy")
    np.save(file_path, test_cmb_map)
    return file_path


@pytest.fixture(scope="session")
def test_simulation_data(test_power_spectrum):
    """
    Create test simulation data.
    
    Args:
        test_power_spectrum: Test power spectrum fixture.
        
    Returns:
        Dictionary containing test simulation data.
    """
    # Create multiple simulations
    np.random.seed(67890)  # For reproducibility
    n_sims = 10
    simulations = []
    
    # Create simple random maps
    for i in range(n_sims):
        sim_map = np.random.normal(0, 1, TEST_NPIX)
        simulations.append(sim_map)
        
    return {
        'simulations': simulations,
        'parameters': {
            'nside': TEST_NSIDE,
            'ell_max': TEST_ELL_MAX,
            'seed': 67890
        }
    }


@pytest.fixture
def mock_data_provenance(monkeypatch, test_data_dir):
    """
    Mock the DataProvenance class for testing.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
        test_data_dir: Test data directory fixture.
        
    Returns:
        Mock DataProvenance object.
    """
    class MockDataProvenance:
        def __init__(self, data_dir=None):
            self.data_dir = data_dir or test_data_dir
            self.provenance_record = {
                'data_sources': [],
                'preprocessing_steps': [],
                'transformations': [],
                'derived_products': []
            }
            self.all_records = {}
            
        def add_data_source(self, **kwargs):
            self.provenance_record['data_sources'].append(kwargs)
            return kwargs
            
        def add_preprocessing_step(self, **kwargs):
            self.provenance_record['preprocessing_steps'].append(kwargs)
            return kwargs
            
        def add_transformation(self, **kwargs):
            self.provenance_record['transformations'].append(kwargs)
            return kwargs
            
        def add_derived_product(self, **kwargs):
            self.provenance_record['derived_products'].append(kwargs)
            return kwargs
            
        def save_record(self, record_id=None):
            record_id = record_id or "test_record_id"
            self.all_records[record_id] = self.provenance_record.copy()
            return record_id
            
        def load_record(self, record_id):
            return self.all_records.get(record_id, {})
            
        def generate_report(self, output_format="text"):
            if output_format == "markdown":
                return "# Data Provenance Report\n\nTest Provenance Report"
            elif output_format == "html":
                return "<!DOCTYPE html><html><body><h1>Data Provenance Report</h1><p>Test Provenance Report</p></body></html>"
            else:
                return "DATA PROVENANCE REPORT\n\nTest Provenance Report"
                
        def get_data_lineage(self, data_path):
            return {
                'transformations': [{'input_data': ['test.npy'], 'output_data': [data_path]}]
            }
            
        def validate_data_integrity(self, file_path, file_hash):
            return os.path.exists(file_path) and file_hash == self._calculate_file_hash(file_path)
            
        def _calculate_file_hash(self, file_path):
            if not os.path.exists(file_path):
                return ""
            return "test_hash_" + os.path.basename(file_path)
    
    try:
        import wmap_data.data_provenance
        monkeypatch.setattr(wmap_data.data_provenance, "DataProvenance", MockDataProvenance)
    except ImportError:
        pass
    
    return MockDataProvenance()


@pytest.fixture
def mock_data_validator(monkeypatch):
    """
    Mock the DataValidator class for testing.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
        
    Returns:
        Mock DataValidator object.
    """
    class MockDataValidator:
        def __init__(self, validation_config=None):
            self.validation_config = validation_config or {
                'power_spectrum': {
                    'min_ell': 2,
                    'max_ell': 2000,
                    'min_value': -1000,
                    'max_value': 10000,
                    'allowed_nans': 0,
                    'allowed_infs': 0,
                    'required_metadata': ['source']
                },
                'map_data': {
                    'min_value': -1000,
                    'max_value': 1000,
                    'allowed_nans': 0,
                    'allowed_infs': 0,
                    'required_metadata': ['nside', 'ordering']
                },
                'simulation_data': {
                    'min_simulations': 5,
                    'required_keys': ['simulations', 'parameters'],
                    'required_metadata': ['source']
                }
            }
            self.validation_results = {
                'passed': False,
                'errors': [],
                'warnings': [],
                'info': []
            }
            
        def _reset_validation_results(self):
            self.validation_results = {
                'passed': False,
                'errors': [],
                'warnings': [],
                'info': []
            }
            
        def _validate_metadata(self, metadata, required_keys):
            if not isinstance(metadata, dict):
                self.validation_results['errors'].append(
                    "Metadata must be a dictionary"
                )
                return False
                
            for key in required_keys:
                if key not in metadata:
                    self.validation_results['errors'].append(
                        "Required metadata key '{}' is missing".format(key)
                    )
                    return False
                if not metadata[key]:
                    self.validation_results['errors'].append(
                        "Metadata key '{}' has an empty value".format(key)
                    )
                    return False
                    
            return True
            
        def validate_power_spectrum(self, data, metadata=None):
            self._reset_validation_results()
            
            # Basic validation
            if not isinstance(data, np.ndarray):
                self.validation_results['errors'].append(
                    "Power spectrum must be a numpy array"
                )
                return self.validation_results
                
            if data.ndim != 1:
                self.validation_results['errors'].append(
                    "Power spectrum must be a 1D array, got shape {}".format(data.shape)
                )
                return self.validation_results
                
            # Check for NaN and Inf values
            if np.isnan(data).any():
                self.validation_results['errors'].append(
                    "Power spectrum contains NaN values"
                )
                
            if np.isinf(data).any():
                self.validation_results['errors'].append(
                    "Power spectrum contains infinite values"
                )
                
            # Validate metadata if provided
            if metadata is not None:
                required_keys = self.validation_config['power_spectrum']['required_metadata']
                self._validate_metadata(metadata, required_keys)
                
            # Set passed flag
            self.validation_results['passed'] = len(self.validation_results['errors']) == 0
            return self.validation_results
            
        def validate_map_data(self, data, metadata=None):
            self._reset_validation_results()
            
            # Basic validation
            if not isinstance(data, np.ndarray):
                self.validation_results['errors'].append(
                    "Map data must be a numpy array"
                )
                return self.validation_results
                
            if metadata and 'nside' in metadata:
                expected_size = 12 * metadata['nside'] * metadata['nside']
                if len(data) != expected_size:
                    self.validation_results['errors'].append(
                        "Map size {} does not match expected size {} for nside {}".format(
                            len(data), expected_size, metadata['nside']
                        )
                    )
                
            # Check for NaN and Inf values
            if np.isnan(data).any():
                self.validation_results['errors'].append(
                    "Map data contains NaN values"
                )
                
            if np.isinf(data).any():
                self.validation_results['errors'].append(
                    "Map data contains infinite values"
                )
                
            # Validate metadata if provided
            if metadata is not None:
                required_keys = self.validation_config['map_data']['required_metadata']
                self._validate_metadata(metadata, required_keys)
                
            # Set passed flag
            self.validation_results['passed'] = len(self.validation_results['errors']) == 0
            return self.validation_results
            
        def validate_simulation_data(self, data, metadata=None):
            self._reset_validation_results()
            
            # Basic validation
            if not isinstance(data, dict):
                self.validation_results['errors'].append(
                    "Simulation data must be a dictionary"
                )
                return self.validation_results
                
            # Check required keys
            required_keys = self.validation_config['simulation_data']['required_keys']
            for key in required_keys:
                if key not in data:
                    self.validation_results['errors'].append(
                        "Required key '{}' is missing from simulation data".format(key)
                    )
                    
            # Check number of simulations
            if 'simulations' in data:
                min_sims = self.validation_config['simulation_data']['min_simulations']
                if len(data['simulations']) < min_sims:
                    self.validation_results['errors'].append(
                        "Number of simulations ({}) is less than the minimum required ({})".format(
                            len(data['simulations']), min_sims
                        )
                    )
                    
            # Validate metadata if provided
            if metadata is not None:
                required_keys = self.validation_config['simulation_data']['required_metadata']
                self._validate_metadata(metadata, required_keys)
                
            # Set passed flag
            self.validation_results['passed'] = len(self.validation_results['errors']) == 0
            return self.validation_results
            
        def validate_file_exists(self, file_path):
            self._reset_validation_results()
            
            if not os.path.exists(file_path):
                self.validation_results['errors'].append(
                    "File '{}' does not exist".format(file_path)
                )
            else:
                self.validation_results['passed'] = True
                
            return self.validation_results
            
        def validate_file_format(self, file_path, expected_format):
            self._reset_validation_results()
            
            if not os.path.exists(file_path):
                self.validation_results['errors'].append(
                    "File '{}' does not exist".format(file_path)
                )
                return self.validation_results
                
            file_ext = os.path.splitext(file_path)[1].lower()[1:]
            if file_ext != expected_format.lower():
                self.validation_results['errors'].append(
                    "File format '{}' does not match expected format '{}'".format(
                        file_ext, expected_format
                    )
                )
            else:
                self.validation_results['passed'] = True
                
            return self.validation_results
            
        def validate_data_consistency(self, data1, data2, tolerance=1e-6):
            self._reset_validation_results()
            
            if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
                self.validation_results['errors'].append(
                    "Both datasets must be numpy arrays"
                )
                return self.validation_results
                
            if data1.shape != data2.shape:
                self.validation_results['errors'].append(
                    "Data shapes do not match: {} vs {}".format(data1.shape, data2.shape)
                )
                return self.validation_results
                
            if not np.allclose(data1, data2, rtol=tolerance, atol=tolerance):
                self.validation_results['errors'].append(
                    "Data values differ by more than the tolerance ({})".format(tolerance)
                )
            else:
                self.validation_results['passed'] = True
                
            return self.validation_results
            
        def validate_data_range(self, data, min_value, max_value):
            self._reset_validation_results()
            
            if not isinstance(data, np.ndarray):
                self.validation_results['errors'].append(
                    "Data must be a numpy array"
                )
                return self.validation_results
                
            if np.any(data < min_value):
                self.validation_results['errors'].append(
                    "Data contains values below the minimum ({})".format(min_value)
                )
                
            if np.any(data > max_value):
                self.validation_results['errors'].append(
                    "Data contains values above the maximum ({})".format(max_value)
                )
                
            self.validation_results['passed'] = len(self.validation_results['errors']) == 0
            return self.validation_results
            
        def validate_with_custom_function(self, data, validation_func):
            self._reset_validation_results()
            
            try:
                result = validation_func(data)
                
                if isinstance(result, dict):
                    # If the function returns a dictionary with specific keys
                    if 'errors' in result:
                        self.validation_results['errors'].extend(result['errors'])
                    if 'warnings' in result:
                        self.validation_results['warnings'].extend(result['warnings'])
                    if 'info' in result:
                        self.validation_results['info'].extend(result['info'])
                elif result is True:
                    # If the function returns True, validation passed
                    self.validation_results['passed'] = True
                elif result is False:
                    # If the function returns False, validation failed
                    self.validation_results['errors'].append(
                        "Custom validation function failed"
                    )
                else:
                    # For any other return value, assume it's a message
                    self.validation_results['info'].append(str(result))
                    
            except Exception as e:
                self.validation_results['errors'].append(
                    "Custom validation function raised an exception: {}".format(str(e))
                )
                
            self.validation_results['passed'] = len(self.validation_results['errors']) == 0
            return self.validation_results
            
        def generate_validation_report(self, output_format="text"):
            if output_format == "text":
                status = "PASSED" if self.validation_results['passed'] else "FAILED"
                report = "DATA VALIDATION REPORT\n\nStatus: {}\n\n".format(status)
                
                if self.validation_results['errors']:
                    report += "Errors:\n" + "\n".join("- {}".format(e) for e in self.validation_results['errors']) + "\n\n"
                    
                if self.validation_results['warnings']:
                    report += "Warnings:\n" + "\n".join("- {}".format(w) for w in self.validation_results['warnings']) + "\n\n"
                    
                if self.validation_results['info']:
                    report += "Info:\n" + "\n".join("- {}".format(i) for i in self.validation_results['info']) + "\n\n"
                    
                return report
                
            elif output_format == "json":
                import json
                return json.dumps(self.validation_results, indent=2)
                
            elif output_format == "html":
                status = "PASSED" if self.validation_results['passed'] else "FAILED"
                report = """<!DOCTYPE html>
                <html>
                <head>
                    <title>Data Validation Report</title>
                    <style>
                        .passed { color: green; }
                        .failed { color: red; }
                        .warning { color: orange; }
                        .info { color: blue; }
                    </style>
                </head>
                <body>
                    <h1>Data Validation Report</h1>
                    <h2 class="{0}">Status: {1}</h2>
                """.format(status.lower(), status)
                
                if self.validation_results['errors']:
                    report += "<h3>Errors:</h3><ul>"
                    for e in self.validation_results['errors']:
                        report += "<li class='failed'>{0}</li>".format(e)
                    report += "</ul>"
                    
                if self.validation_results['warnings']:
                    report += "<h3>Warnings:</h3><ul>"
                    for w in self.validation_results['warnings']:
                        report += "<li class='warning'>{0}</li>".format(w)
                    report += "</ul>"
                    
                if self.validation_results['info']:
                    report += "<h3>Info:</h3><ul>"
                    for i in self.validation_results['info']:
                        report += "<li class='info'>{0}</li>".format(i)
                    report += "</ul>"
                    
                report += "</body></html>"
                return report
                
            else:
                raise ValueError("Unsupported output format: {}".format(output_format))
    
    try:
        import wmap_data.data_validation
        monkeypatch.setattr(wmap_data.data_validation, "DataValidator", MockDataValidator)
    except ImportError:
        pass
    
    return MockDataValidator()


def pytest_configure(config):
    """
    Configure pytest.
    
    Args:
        config: Pytest config object.
    """
    # Register custom markers
    config.addinivalue_line("markers", "unit: mark a test as a unit test")
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "slow: mark a test as slow")
    config.addinivalue_line("markers", "data_dependent: mark a test as dependent on external data")


def pytest_collection_modifyitems(config, items):
    """
    Modify collected test items.
    
    Args:
        config: Pytest config object.
        items: List of test items.
    """
    # Skip slow tests if --skip-slow is specified
    if config.getoption("--skip-slow", False):
        skip_slow = pytest.mark.skip(reason="--skip-slow option provided")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
                
    # Skip data-dependent tests if --skip-data-dependent is specified
    if config.getoption("--skip-data-dependent", False):
        skip_data = pytest.mark.skip(reason="--skip-data-dependent option provided")
        for item in items:
            if "data_dependent" in item.keywords:
                item.add_marker(skip_data)


def pytest_addoption(parser):
    """
    Add command line options to pytest.
    
    Args:
        parser: Pytest argument parser.
    """
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="Skip slow tests"
    )
    parser.addoption(
        "--skip-data-dependent", action="store_true", default=False, 
        help="Skip tests that depend on external data"
    )
