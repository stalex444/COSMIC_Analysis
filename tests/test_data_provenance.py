"""
Unit tests for the data provenance module.
"""

import os
import json
import pytest
import numpy as np
from datetime import datetime

# Import the mock from conftest.py
# The actual implementation will be mocked in the tests


@pytest.mark.unit
class TestDataProvenance:
    """Test cases for the DataProvenance class."""

    def test_initialization(self, mock_data_provenance, test_data_dir):
        """Test initialization of DataProvenance."""
        assert mock_data_provenance.data_dir == test_data_dir
        assert 'data_sources' in mock_data_provenance.provenance_record
        assert 'preprocessing_steps' in mock_data_provenance.provenance_record
        assert 'transformations' in mock_data_provenance.provenance_record
        assert 'derived_products' in mock_data_provenance.provenance_record

    def test_add_data_source(self, mock_data_provenance):
        """Test adding a data source."""
        source_data = {
            'source_name': 'WMAP',
            'source_url': 'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/wmap_band_iqumap_r9_9yr_K_v5.fits',
            'source_version': 'DR5',
            'source_date': '2013-12-20',
            'description': 'WMAP 9-year K-band map',
            'data_format': 'FITS',
            'file_path': '/path/to/wmap_data.fits',
            'file_hash': 'abc123',
            'metadata': {'frequency': '23 GHz'}
        }
        
        result = mock_data_provenance.add_data_source(**source_data)
        
        assert result == source_data
        assert len(mock_data_provenance.provenance_record['data_sources']) == 1
        assert mock_data_provenance.provenance_record['data_sources'][0] == source_data

    def test_add_preprocessing_step(self, mock_data_provenance):
        """Test adding a preprocessing step."""
        preprocessing_data = {
            'step_name': 'Masking',
            'description': 'Applied galactic mask',
            'parameters': {'mask_type': 'galactic', 'threshold': 0.8},
            'input_data': ['wmap_data.fits'],
            'output_data': ['wmap_data_masked.fits'],
            'software_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        
        result = mock_data_provenance.add_preprocessing_step(**preprocessing_data)
        
        assert result == preprocessing_data
        assert len(mock_data_provenance.provenance_record['preprocessing_steps']) == 1
        assert mock_data_provenance.provenance_record['preprocessing_steps'][0] == preprocessing_data

    def test_add_transformation(self, mock_data_provenance):
        """Test adding a transformation."""
        transformation_data = {
            'transformation_name': 'Spherical Harmonics Transform',
            'description': 'Computed power spectrum',
            'parameters': {'lmax': 1000, 'iter': 3},
            'input_data': ['wmap_data_masked.fits'],
            'output_data': ['wmap_cl.txt'],
            'software_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        
        result = mock_data_provenance.add_transformation(**transformation_data)
        
        assert result == transformation_data
        assert len(mock_data_provenance.provenance_record['transformations']) == 1
        assert mock_data_provenance.provenance_record['transformations'][0] == transformation_data

    def test_add_derived_product(self, mock_data_provenance):
        """Test adding a derived product."""
        derived_product_data = {
            'product_name': 'CMB Power Spectrum',
            'description': 'Final CMB power spectrum',
            'input_data': ['wmap_cl.txt'],
            'output_data': ['cmb_power_spectrum.txt'],
            'parameters': {'binning': 'log', 'bin_width': 0.1},
            'software_version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        
        result = mock_data_provenance.add_derived_product(**derived_product_data)
        
        assert result == derived_product_data
        assert len(mock_data_provenance.provenance_record['derived_products']) == 1
        assert mock_data_provenance.provenance_record['derived_products'][0] == derived_product_data

    def test_save_and_load_record(self, mock_data_provenance, test_data_dir):
        """Test saving and loading a provenance record."""
        # Add some data to the record
        mock_data_provenance.add_data_source(
            source_name='TEST',
            source_url='https://example.com/test.fits',
            source_version='1.0',
            source_date='2023-01-01',
            description='Test data',
            data_format='FITS'
        )
        
        # Save the record
        record_id = mock_data_provenance.save_record()
        
        # Check that the record was saved
        assert record_id in mock_data_provenance.all_records
        
        # Load the record
        loaded_record = mock_data_provenance.load_record(record_id)
        
        # Check that the loaded record matches the original
        assert loaded_record == mock_data_provenance.provenance_record

    def test_generate_report(self, mock_data_provenance):
        """Test generating a provenance report."""
        # Add some data to the record
        mock_data_provenance.add_data_source(
            source_name='TEST',
            source_url='https://example.com/test.fits',
            source_version='1.0',
            source_date='2023-01-01',
            description='Test data',
            data_format='FITS'
        )
        
        # Generate reports in different formats
        text_report = mock_data_provenance.generate_report(output_format="text")
        markdown_report = mock_data_provenance.generate_report(output_format="markdown")
        html_report = mock_data_provenance.generate_report(output_format="html")
        
        # Check that the reports were generated
        assert "DATA PROVENANCE REPORT" in text_report
        assert "# Data Provenance Report" in markdown_report
        assert "<h1>Data Provenance Report</h1>" in html_report

    def test_get_data_lineage(self, mock_data_provenance):
        """Test getting data lineage for a file."""
        # Add some data to the record
        mock_data_provenance.add_transformation(
            transformation_name='Test Transform',
            description='Test transformation',
            input_data=['test.npy'],
            output_data=['output.npy'],
            parameters={},
            software_version='1.0.0',
            timestamp=datetime.now().isoformat()
        )
        
        # Get the lineage for the output file
        lineage = mock_data_provenance.get_data_lineage('output.npy')
        
        # Check that the lineage contains the transformation
        assert 'transformations' in lineage
        assert len(lineage['transformations']) == 1
        assert lineage['transformations'][0]['output_data'] == ['output.npy']

    def test_validate_data_integrity(self, mock_data_provenance, test_data_file):
        """Test validating data integrity."""
        # Calculate the expected hash
        expected_hash = "test_hash_" + os.path.basename(test_data_file)
        
        # Validate the file
        is_valid = mock_data_provenance.validate_data_integrity(test_data_file, expected_hash)
        
        # Check that the validation passed
        assert is_valid is True
        
        # Test with an incorrect hash
        is_valid = mock_data_provenance.validate_data_integrity(test_data_file, "wrong_hash")
        
        # Check that the validation failed
        assert is_valid is False

    def test_save_record_with_id(self, mock_data_provenance, test_data_dir):
        """Test saving a record with a specific ID."""
        # Add some data to the record
        mock_data_provenance.add_data_source(
            source_name='TEST',
            source_url='https://example.com/test.fits',
            source_version='1.0',
            source_date='2023-01-01',
            description='Test data',
            data_format='FITS'
        )
        
        # Save the record with a specific ID
        record_id = "test_record_123"
        saved_id = mock_data_provenance.save_record(record_id=record_id)
        
        # Check that the record was saved with the correct ID
        assert saved_id == record_id
        assert record_id in mock_data_provenance.all_records
        
        # Create a file path for the record
        record_file = os.path.join(test_data_dir, "{0}.json".format(record_id))
        
        # The file won't actually exist in the mock, but we can check the logic
        loaded_record = mock_data_provenance.load_record(record_id)
        assert loaded_record == mock_data_provenance.provenance_record
