#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

"""
Download WMAP data for the WMAP Cosmic Analysis project.

This script downloads WMAP CMB data from NASA's LAMBDA archive,
including power spectrum data and maps.
"""

import os
import sys
import argparse
import urllib
try:
    # Python 3
    from urllib.request import urlretrieve
    from urllib.error import URLError
except ImportError:
    # Python 2
    from urllib import urlretrieve
    from urllib2 import URLError
import hashlib
import json
import time

# Import data provenance tracking if available
try:
    from wmap_data.data_provenance import DataProvenanceTracker
    HAS_PROVENANCE = True
except ImportError:
    HAS_PROVENANCE = False

# WMAP data URLs
WMAP_DATA_URLS = {
    'power_spectrum': {
        'tt': 'https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/spectra/wmap_tt_spectrum_9yr_v5.txt',
        'te': 'https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/spectra/wmap_te_spectrum_9yr_v5.txt',
        'ee': 'https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/spectra/wmap_ee_spectrum_9yr_v5.txt',
        'bb': 'https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/spectra/wmap_bb_spectrum_9yr_v5.txt'
    },
    'maps': {
        'ILC': 'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_9yr_ilc_v5.fits',
        'K_band': 'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_smth_imap_r9_9yr_K_v5.fits',
        'Ka_band': 'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_smth_imap_r9_9yr_Ka_v5.fits',
        'Q_band': 'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_smth_imap_r9_9yr_Q_v5.fits',
        'V_band': 'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_smth_imap_r9_9yr_V_v5.fits',
        'W_band': 'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_smth_imap_r9_9yr_W_v5.fits'
    },
    'ancillary': {
        'beam_transfer': 'https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/beams/wmap_ampl_bl_9yr_v5.txt',
        'masks': 'https://lambda.gsfc.nasa.gov/data/map/dr5/ancillary/masks/wmap_temperature_analysis_mask_r9_9yr_v5.fits'
    }
}

# Expected file sizes (in bytes) for verification
EXPECTED_SIZES = {
    'wmap_tt_spectrum_9yr_v5.txt': 35000,
    'wmap_te_spectrum_9yr_v5.txt': 35000,
    'wmap_ee_spectrum_9yr_v5.txt': 35000,
    'wmap_bb_spectrum_9yr_v5.txt': 35000,
    'wmap_9yr_ilc_v5.fits': 3000000,
    'wmap_band_smth_imap_r9_9yr_K_v5.fits': 3000000,
    'wmap_band_smth_imap_r9_9yr_Ka_v5.fits': 3000000,
    'wmap_band_smth_imap_r9_9yr_Q_v5.fits': 3000000,
    'wmap_band_smth_imap_r9_9yr_V_v5.fits': 3000000,
    'wmap_band_smth_imap_r9_9yr_W_v5.fits': 3000000,
    'wmap_ampl_bl_9yr_v5.txt': 50000,
    'wmap_temperature_analysis_mask_r9_9yr_v5.fits': 1000000
}

def calculate_md5(file_path):
    """
    Calculate the MD5 hash of a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: MD5 hash of the file
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, output_path, max_retries=3):
    """
    Download a file from a URL with retry logic.
    
    Args:
        url (str): URL to download
        output_path (str): Path to save the downloaded file
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            print("Downloading {} to {}...".format(url, output_path))
            urlretrieve(url, output_path)
            
            # Verify file size
            file_name = os.path.basename(output_path)
            if file_name in EXPECTED_SIZES:
                file_size = os.path.getsize(output_path)
                expected_size = EXPECTED_SIZES[file_name]
                
                # Allow for some variation in file size (±10%)
                if file_size < expected_size * 0.9 or file_size > expected_size * 1.1:
                    print("Warning: File size mismatch for {}".format(file_name))
                    print("Expected: {} bytes, Got: {} bytes".format(expected_size, file_size))
                    print("Retrying download...")
                    continue
            
            print("Download successful!")
            return True
        except URLError as e:
            print("Error downloading {}: {}".format(url, str(e)))
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print("Retrying in {} seconds...".format(wait_time))
                time.sleep(wait_time)
            else:
                print("Max retries reached. Download failed.")
                return False

def download_wmap_data(data_types, output_dir, register_provenance=True):
    """
    Download WMAP data of the specified types.
    
    Args:
        data_types (list): List of data types to download ('power_spectrum', 'maps', 'ancillary')
        output_dir (str): Directory to save the downloaded data
        register_provenance (bool): Whether to register the downloaded files in the provenance system
        
    Returns:
        dict: Dictionary of downloaded files with their metadata
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize provenance tracker if available
    provenance_tracker = None
    if HAS_PROVENANCE and register_provenance:
        provenance_file = os.path.join(output_dir, 'provenance.json')
        provenance_tracker = DataProvenanceTracker(provenance_file)
    
    # Dictionary to store downloaded files and their metadata
    downloaded_files = {}
    
    # Download each data type
    for data_type in data_types:
        if data_type not in WMAP_DATA_URLS:
            print("Unknown data type: {}".format(data_type))
            continue
        
        # Create subdirectory for this data type
        type_dir = os.path.join(output_dir, data_type)
        os.makedirs(type_dir, exist_ok=True)
        
        # Download each file for this data type
        for name, url in WMAP_DATA_URLS[data_type].items():
            file_name = os.path.basename(url)
            output_path = os.path.join(type_dir, file_name)
            
            # Skip if file already exists
            if os.path.exists(output_path):
                print("{} already exists, skipping...".format(output_path))
                
                # Register in provenance system if needed
                if provenance_tracker is not None:
                    file_id = provenance_tracker.register_file(
                        output_path,
                        source_type='download',
                        source_details='WMAP data from LAMBDA archive',
                        metadata={
                            'url': url,
                            'data_type': data_type,
                            'name': name,
                            'md5': calculate_md5(output_path)
                        }
                    )
                    downloaded_files[output_path] = {
                        'url': url,
                        'data_type': data_type,
                        'name': name,
                        'md5': calculate_md5(output_path),
                        'file_id': file_id
                    }
                continue
            
            # Download the file
            success = download_file(url, output_path)
            
            if success:
                # Calculate MD5 hash
                md5_hash = calculate_md5(output_path)
                
                # Register in provenance system if needed
                if provenance_tracker is not None:
                    file_id = provenance_tracker.register_file(
                        output_path,
                        source_type='download',
                        source_details='WMAP data from LAMBDA archive',
                        metadata={
                            'url': url,
                            'data_type': data_type,
                            'name': name,
                            'md5': md5_hash
                        }
                    )
                    downloaded_files[output_path] = {
                        'url': url,
                        'data_type': data_type,
                        'name': name,
                        'md5': md5_hash,
                        'file_id': file_id
                    }
                else:
                    downloaded_files[output_path] = {
                        'url': url,
                        'data_type': data_type,
                        'name': name,
                        'md5': md5_hash
                    }
    
    # Save metadata about downloaded files
    metadata_file = os.path.join(output_dir, 'download_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(downloaded_files, f, indent=2)
    
    return downloaded_files

def main():
    """Main function to parse arguments and download data."""
    parser = argparse.ArgumentParser(description='Download WMAP data for analysis')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the downloaded data')
    parser.add_argument('--power_spectrum', action='store_true',
                        help='Download power spectrum data')
    parser.add_argument('--maps', action='store_true',
                        help='Download map data')
    parser.add_argument('--ancillary', action='store_true',
                        help='Download ancillary data (beam transfer functions, masks)')
    parser.add_argument('--all', action='store_true',
                        help='Download all data types')
    parser.add_argument('--no_provenance', action='store_true',
                        help='Do not register downloads in the provenance system')
    
    args = parser.parse_args()
    
    # Determine which data types to download
    data_types = []
    if args.all:
        data_types = list(WMAP_DATA_URLS.keys())
    else:
        if args.power_spectrum:
            data_types.append('power_spectrum')
        if args.maps:
            data_types.append('maps')
        if args.ancillary:
            data_types.append('ancillary')
    
    # Default to power_spectrum if no types specified
    if not data_types:
        data_types = ['power_spectrum']
    
    # Download the data
    download_wmap_data(data_types, args.output_dir, not args.no_provenance)

if __name__ == '__main__':
    main()
