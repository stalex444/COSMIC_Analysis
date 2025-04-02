"""
Data Provenance Module for WMAP Cosmic Analysis

This module provides tools for tracking the provenance of data used in the WMAP Cosmic Analysis
framework. It records information about data sources, preprocessing steps, and transformations
to ensure reproducibility and transparency in the analysis pipeline.
"""

import os
import json
import datetime
import hashlib
import logging
import numpy as np
import pandas as pd
from collections import OrderedDict
# Import typing but don't use type annotations in function signatures for Python 2.7 compatibility
from typing import Dict, List, Union, Optional, Any, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class DataProvenance:
    """
    Class for tracking data provenance in the WMAP Cosmic Analysis framework.
    
    This class records information about data sources, preprocessing steps, and transformations
    to ensure reproducibility and transparency in the analysis pipeline.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the DataProvenance object.
        
        Args:
            data_dir: Directory to store provenance records. If None, uses the default
                      location in the project directory.
        """
        if data_dir is None:
            # Use default location in project directory
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(project_dir, 'data', 'provenance')
        else:
            self.data_dir = data_dir
            
        # Create directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Initialize provenance record
        self.provenance_record = OrderedDict({
            'data_sources': [],
            'preprocessing_steps': [],
            'transformations': [],
            'derived_products': []
        })
        
        # Load existing provenance records
        self.all_records = self._load_all_records()
        
    def _load_all_records(self):
        """
        Load all existing provenance records.
        
        Returns:
            Dictionary mapping record IDs to provenance records.
        """
        records = {}
        
        if not os.path.exists(self.data_dir):
            return records
            
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        record = json.load(f)
                        record_id = os.path.splitext(filename)[0]
                        records[record_id] = record
                except Exception as e:
                    logger.warning("Failed to load provenance record {}: {e}".format(filename))
                    
        return records
        
    def add_data_source(self, 
                       source_name, 
                       source_url, 
                       source_version,
                       source_date,
                       description,
                       data_format,
                       file_path=None,
                       file_hash=None,
                       metadata=None):
        """
        Add information about a data source.
        
        Args:
            source_name: Name of the data source (e.g., 'WMAP', 'Planck')
            source_url: URL where the data was obtained
            source_version: Version of the data (e.g., 'WMAP9', 'Planck 2018')
            source_date: Date when the data was published
            description: Description of the data
            data_format: Format of the data (e.g., 'FITS', 'CSV')
            file_path: Path to the data file (optional)
            file_hash: Hash of the data file for verification (optional)
            metadata: Additional metadata about the data source (optional)
        """
        if file_path and not file_hash:
            # Calculate file hash if file exists
            if os.path.exists(file_path):
                file_hash = self._calculate_file_hash(file_path)
                
        source_info = {
            'source_name': source_name,
            'source_url': source_url,
            'source_version': source_version,
            'source_date': source_date,
            'description': description,
            'data_format': data_format,
            'file_path': file_path,
            'file_hash': file_hash,
            'date_added': datetime.datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.provenance_record['data_sources'].append(source_info)
        logger.info("Added data source: {} ({source_version})".format(source_name))
        
    def add_preprocessing_step(self,
                              step_name,
                              description,
                              parameters,
                              input_data,
                              output_data,
                              code_version=None,
                              runtime=None,
                              metadata=None):
        """
        Add information about a preprocessing step.
        
        Args:
            step_name: Name of the preprocessing step
            description: Description of the preprocessing step
            parameters: Parameters used in the preprocessing step
            input_data: Input data for the preprocessing step (file path or ID)
            output_data: Output data from the preprocessing step (file path or ID)
            code_version: Version of the code used for preprocessing (optional)
            runtime: Runtime of the preprocessing step in seconds (optional)
            metadata: Additional metadata about the preprocessing step (optional)
        """
        step_info = {
            'step_name': step_name,
            'description': description,
            'parameters': parameters,
            'input_data': input_data if isinstance(input_data, list) else [input_data],
            'output_data': output_data if isinstance(output_data, list) else [output_data],
            'code_version': code_version,
            'runtime': runtime,
            'timestamp': datetime.datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.provenance_record['preprocessing_steps'].append(step_info)
        logger.info("Added preprocessing step: {}".format(step_name))
        
    def add_transformation(self,
                          transform_name,
                          description,
                          parameters,
                          input_data,
                          output_data,
                          code_version=None,
                          runtime=None,
                          metadata=None):
        """
        Add information about a data transformation.
        
        Args:
            transform_name: Name of the transformation
            description: Description of the transformation
            parameters: Parameters used in the transformation
            input_data: Input data for the transformation (file path or ID)
            output_data: Output data from the transformation (file path or ID)
            code_version: Version of the code used for transformation (optional)
            runtime: Runtime of the transformation in seconds (optional)
            metadata: Additional metadata about the transformation (optional)
        """
        transform_info = {
            'transform_name': transform_name,
            'description': description,
            'parameters': parameters,
            'input_data': input_data if isinstance(input_data, list) else [input_data],
            'output_data': output_data if isinstance(output_data, list) else [output_data],
            'code_version': code_version,
            'runtime': runtime,
            'timestamp': datetime.datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.provenance_record['transformations'].append(transform_info)
        logger.info("Added transformation: {}".format(transform_name))
        
    def add_derived_product(self,
                           product_name,
                           description,
                           source_data,
                           product_path,
                           creation_date=None,
                           creator=None,
                           version=None,
                           metadata=None):
        """
        Add information about a derived data product.
        
        Args:
            product_name: Name of the derived product
            description: Description of the derived product
            source_data: Source data used to create the derived product
            product_path: Path to the derived product
            creation_date: Date when the derived product was created (optional)
            creator: Creator of the derived product (optional)
            version: Version of the derived product (optional)
            metadata: Additional metadata about the derived product (optional)
        """
        if not creation_date:
            creation_date = datetime.datetime.now().isoformat()
            
        product_hash = None
        if os.path.exists(product_path):
            product_hash = self._calculate_file_hash(product_path)
            
        product_info = {
            'product_name': product_name,
            'description': description,
            'source_data': source_data if isinstance(source_data, list) else [source_data],
            'product_path': product_path,
            'product_hash': product_hash,
            'creation_date': creation_date,
            'creator': creator,
            'version': version,
            'metadata': metadata or {}
        }
        
        self.provenance_record['derived_products'].append(product_info)
        logger.info("Added derived product: {}".format(product_name))
        
    def save_record(self, record_id=None):
        """
        Save the current provenance record to a JSON file.
        
        Args:
            record_id: ID for the provenance record. If None, a unique ID is generated.
            
        Returns:
            ID of the saved provenance record.
        """
        if record_id is None:
            # Generate a unique ID based on timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            record_id = "provenance_{}".format(timestamp)
            
        # Add metadata to the record
        self.provenance_record['record_id'] = record_id
        self.provenance_record['timestamp'] = datetime.datetime.now().isoformat()
        
        # Save the record to a JSON file
        filepath = os.path.join(self.data_dir, "{}.json".format(record_id))
        with open(filepath, 'w') as f:
            json.dump(self.provenance_record, f, indent=2)
            
        # Add to all records
        self.all_records[record_id] = self.provenance_record
        
        logger.info("Saved provenance record: {}".format(record_id))
        return record_id
        
    def load_record(self, record_id):
        """
        Load a provenance record from a JSON file.
        
        Args:
            record_id: ID of the provenance record to load.
            
        Returns:
            The loaded provenance record.
            
        Raises:
            FileNotFoundError: If the provenance record does not exist.
        """
        filepath = os.path.join(self.data_dir, "{}.json".format(record_id))
        
        if not os.path.exists(filepath):
            raise FileNotFoundError("Provenance record {} not found.".format(record_id))
            
        with open(filepath, 'r') as f:
            record = json.load(f)
            
        # Update the current provenance record
        self.provenance_record = record
        
        logger.info("Loaded provenance record: {}".format(record_id))
        return record
        
    def get_data_lineage(self, product_id):
        """
        Get the complete lineage of a data product.
        
        Args:
            product_id: ID of the data product.
            
        Returns:
            Dictionary containing the complete lineage of the data product.
        """
        lineage = {
            'data_sources': [],
            'preprocessing_steps': [],
            'transformations': [],
            'derived_products': []
        }
        
        # Search for the product in all records
        for record_id, record in self.all_records.items():
            # Check derived products
            for product in record['derived_products']:
                if product['product_name'] == product_id or product['product_path'] == product_id:
                    lineage['derived_products'].append(product)
                    
                    # Add source data to the search
                    for source in product['source_data']:
                        lineage.update(self.get_data_lineage(source))
                        
            # Check transformations
            for transform in record['transformations']:
                if product_id in transform['output_data']:
                    lineage['transformations'].append(transform)
                    
                    # Add input data to the search
                    for input_data in transform['input_data']:
                        lineage.update(self.get_data_lineage(input_data))
                        
            # Check preprocessing steps
            for step in record['preprocessing_steps']:
                if product_id in step['output_data']:
                    lineage['preprocessing_steps'].append(step)
                    
                    # Add input data to the search
                    for input_data in step['input_data']:
                        lineage.update(self.get_data_lineage(input_data))
                        
            # Check data sources
            for source in record['data_sources']:
                if source['file_path'] == product_id or source['source_name'] == product_id:
                    lineage['data_sources'].append(source)
                    
        return lineage
        
    def generate_report(self, output_format='markdown'):
        """
        Generate a report of the current provenance record.
        
        Args:
            output_format: Format of the report ('markdown', 'html', or 'text').
            
        Returns:
            Report as a string in the specified format.
        """
        if output_format == 'markdown':
            return self._generate_markdown_report()
        elif output_format == 'html':
            return self._generate_html_report()
        elif output_format == 'text':
            return self._generate_text_report()
        else:
            raise ValueError("Unsupported output format: {}".format(output_format))
            
    def _generate_markdown_report(self):
        """
        Generate a markdown report of the current provenance record.
        
        Returns:
            Markdown report as a string.
        """
        report = ["# Data Provenance Report\n"]
        
        # Add timestamp
        report.append("**Generated:** {}\n".format(datetime.datetime.now().isoformat()))
        
        # Add data sources
        report.append("## Data Sources\n")
        for i, source in enumerate(self.provenance_record['data_sources']):
            report.append("### {}. {source['source_name']} ({source['source_version']})\n".format(i+1))
            report.append("- **URL:** {}".format(source['source_url']))
            report.append("- **Date:** {}".format(source['source_date']))
            report.append("- **Format:** {}".format(source['data_format']))
            report.append("- **Description:** {}".format(source['description']))
            if source['file_path']:
                report.append("- **File Path:** {}".format(source['file_path']))
            if source['file_hash']:
                report.append("- **File Hash:** {}".format(source['file_hash']))
            report.append("\n")
            
        # Add preprocessing steps
        report.append("## Preprocessing Steps\n")
        for i, step in enumerate(self.provenance_record['preprocessing_steps']):
            report.append("### {}. {step['step_name']}\n".format(i+1))
            report.append("- **Description:** {}".format(step['description']))
            report.append("- **Parameters:** {}".format(json.dumps(step['parameters'], indent=2)))
            report.append("- **Input Data:** {}".format(', '.join(step['input_data'])))
            report.append("- **Output Data:** {}".format(', '.join(step['output_data'])))
            report.append("- **Timestamp:** {}".format(step['timestamp']))
            if step['runtime']:
                report.append("- **Runtime:** {} seconds".format(step['runtime']))
            report.append("\n")
            
        # Add transformations
        report.append("## Transformations\n")
        for i, transform in enumerate(self.provenance_record['transformations']):
            report.append("### {}. {transform['transform_name']}\n".format(i+1))
            report.append("- **Description:** {}".format(transform['description']))
            report.append("- **Parameters:** {}".format(json.dumps(transform['parameters'], indent=2)))
            report.append("- **Input Data:** {}".format(', '.join(transform['input_data'])))
            report.append("- **Output Data:** {}".format(', '.join(transform['output_data'])))
            report.append("- **Timestamp:** {}".format(transform['timestamp']))
            if transform['runtime']:
                report.append("- **Runtime:** {} seconds".format(transform['runtime']))
            report.append("\n")
            
        # Add derived products
        report.append("## Derived Products\n")
        for i, product in enumerate(self.provenance_record['derived_products']):
            report.append("### {}. {product['product_name']}\n".format(i+1))
            report.append("- **Description:** {}".format(product['description']))
            report.append("- **Source Data:** {}".format(', '.join(product['source_data'])))
            report.append("- **Product Path:** {}".format(product['product_path']))
            report.append("- **Creation Date:** {}".format(product['creation_date']))
            if product['creator']:
                report.append("- **Creator:** {}".format(product['creator']))
            if product['version']:
                report.append("- **Version:** {}".format(product['version']))
            if product['product_hash']:
                report.append("- **Product Hash:** {}".format(product['product_hash']))
            report.append("\n")
            
        return "\n".join(report)
        
    def _generate_html_report(self):
        """
        Generate an HTML report of the current provenance record.
        
        Returns:
            str: HTML report as a string
        """
        # First generate markdown report
        markdown_report = self._generate_markdown_report()
        
        # Convert markdown to simple HTML
        html_content = markdown_report.replace('\n', '<br>').replace('## ', '<h2>').replace('### ', '<h3>').replace('# ', '<h1>').replace('</h1>', '</h1><hr>')
        
        # HTML template
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Data Provenance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; border-bottom: 1px solid #3498db; padding-bottom: 5px; }
        h3 { color: #2980b9; }
        pre { background-color: #f8f8f8; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div id="content">
        {0}
    </div>
</body>
</html>"""
        
        # Format the template with the content
        html_report = html_template.format(html_content)
        
        return html_report
        
    def _generate_text_report(self):
        """
        Generate a text report of the current provenance record.
        
        Returns:
            Text report as a string.
        """
        report = ["DATA PROVENANCE REPORT\n"]
        
        # Add timestamp
        report.append("Generated: {}\n".format(datetime.datetime.now().isoformat()))
        
        # Add data sources
        report.append("DATA SOURCES\n" + "="*12 + "\n")
        for i, source in enumerate(self.provenance_record['data_sources']):
            report.append("{}. {source['source_name']} ({source['source_version']})".format(i+1))
            report.append("   URL: {}".format(source['source_url']))
            report.append("   Date: {}".format(source['source_date']))
            report.append("   Format: {}".format(source['data_format']))
            report.append("   Description: {}".format(source['description']))
            if source['file_path']:
                report.append("   File Path: {}".format(source['file_path']))
            if source['file_hash']:
                report.append("   File Hash: {}".format(source['file_hash']))
            report.append("")
            
        # Add preprocessing steps
        report.append("PREPROCESSING STEPS\n" + "="*19 + "\n")
        for i, step in enumerate(self.provenance_record['preprocessing_steps']):
            report.append("{}. {step['step_name']}".format(i+1))
            report.append("   Description: {}".format(step['description']))
            report.append("   Parameters: {}".format(json.dumps(step['parameters'], indent=2)))
            report.append("   Input Data: {}".format(', '.join(step['input_data'])))
            report.append("   Output Data: {}".format(', '.join(step['output_data'])))
            report.append("   Timestamp: {}".format(step['timestamp']))
            if step['runtime']:
                report.append("   Runtime: {} seconds".format(step['runtime']))
            report.append("")
            
        # Add transformations
        report.append("TRANSFORMATIONS\n" + "="*14 + "\n")
        for i, transform in enumerate(self.provenance_record['transformations']):
            report.append("{}. {transform['transform_name']}".format(i+1))
            report.append("   Description: {}".format(transform['description']))
            report.append("   Parameters: {}".format(json.dumps(transform['parameters'], indent=2)))
            report.append("   Input Data: {}".format(', '.join(transform['input_data'])))
            report.append("   Output Data: {}".format(', '.join(transform['output_data'])))
            report.append("   Timestamp: {}".format(transform['timestamp']))
            if transform['runtime']:
                report.append("   Runtime: {} seconds".format(transform['runtime']))
            report.append("")
            
        # Add derived products
        report.append("DERIVED PRODUCTS\n" + "="*16 + "\n")
        for i, product in enumerate(self.provenance_record['derived_products']):
            report.append("{}. {product['product_name']}".format(i+1))
            report.append("   Description: {}".format(product['description']))
            report.append("   Source Data: {}".format(', '.join(product['source_data'])))
            report.append("   Product Path: {}".format(product['product_path']))
            report.append("   Creation Date: {}".format(product['creation_date']))
            if product['creator']:
                report.append("   Creator: {}".format(product['creator']))
            if product['version']:
                report.append("   Version: {}".format(product['version']))
            if product['product_hash']:
                report.append("   Product Hash: {}".format(product['product_hash']))
            report.append("")
            
        return "\n".join(report)
        
    def _calculate_file_hash(self, file_path):
        """
        Calculate the SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            SHA-256 hash of the file as a hexadecimal string.
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                
        return sha256_hash.hexdigest()
        
    def validate_data_integrity(self, file_path, expected_hash):
        """
        Validate the integrity of a data file by comparing its hash with an expected hash.
        
        Args:
            file_path: Path to the data file.
            expected_hash: Expected hash of the data file.
            
        Returns:
            True if the file hash matches the expected hash, False otherwise.
        """
        if not os.path.exists(file_path):
            logger.warning("File not found: {}".format(file_path))
            return False
            
        file_hash = self._calculate_file_hash(file_path)
        
        if file_hash != expected_hash:
            logger.warning("Hash mismatch for {}. Expected: {expected_hash}, Got: {file_hash}".format(file_path))
            return False
            
        return True
        
    def get_data_sources(self):
        """
        Get all data sources in the current provenance record.
        
        Returns:
            List of data sources.
        """
        return self.provenance_record['data_sources']
        
    def get_preprocessing_steps(self):
        """
        Get all preprocessing steps in the current provenance record.
        
        Returns:
            List of preprocessing steps.
        """
        return self.provenance_record['preprocessing_steps']
        
    def get_transformations(self):
        """
        Get all transformations in the current provenance record.
        
        Returns:
            List of transformations.
        """
        return self.provenance_record['transformations']
        
    def get_derived_products(self):
        """
        Get all derived products in the current provenance record.
        
        Returns:
            List of derived products.
        """
        return self.provenance_record['derived_products']


# Example usage
if __name__ == "__main__":
    # Create a data provenance object
    provenance = DataProvenance()
    
    # Add a data source
    provenance.add_data_source(
        source_name="WMAP",
        source_url="https://lambda.gsfc.nasa.gov/product/wmap/dr5/",
        source_version="WMAP9",
        source_date="2013-12-21",
        description="WMAP 9-year power spectrum data",
        data_format="FITS",
        file_path="/path/to/wmap_power_spectrum.fits"
    )
    
    # Add a preprocessing step
    provenance.add_preprocessing_step(
        step_name="Smoothing",
        description="Apply Gaussian smoothing to the power spectrum",
        parameters={"window_size": 5, "sigma": 1.0},
        input_data="/path/to/wmap_power_spectrum.fits",
        output_data="/path/to/wmap_power_spectrum_smoothed.fits"
    )
    
    # Add a transformation
    provenance.add_transformation(
        transform_name="Normalization",
        description="Normalize the power spectrum",
        parameters={"method": "min-max"},
        input_data="/path/to/wmap_power_spectrum_smoothed.fits",
        output_data="/path/to/wmap_power_spectrum_normalized.fits"
    )
    
    # Add a derived product
    provenance.add_derived_product(
        product_name="WMAP Golden Ratio Analysis",
        description="Analysis of golden ratio patterns in the WMAP power spectrum",
        source_data="/path/to/wmap_power_spectrum_normalized.fits",
        product_path="/path/to/wmap_golden_ratio_analysis.json"
    )
    
    # Save the provenance record
    record_id = provenance.save_record()
    
    # Generate a report
    report = provenance.generate_report(output_format="markdown")
    print(report)
