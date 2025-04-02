#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download WMAP data files from NASA's LAMBDA website with updated URLs.
"""

from __future__ import print_function
import os
import sys
import time
try:
    # Python 3
    from urllib.request import urlretrieve
    from urllib.error import URLError
except ImportError:
    # Python 2
    from urllib import urlretrieve
    from urllib2 import URLError

# Base URL for WMAP data
LAMBDA_BASE_URL = "https://lambda.gsfc.nasa.gov/data/map/dr5"

# WMAP data files to download - Updated URLs based on current LAMBDA structure
WMAP_FILES = {
    # ILC Map (Internal Linear Combination)
    "ILC_MAP": {
        "url": "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_ilc_9yr_v5.fits",
        "description": "WMAP 9-year ILC (Internal Linear Combination) Map"
    },
    # Temperature power spectrum
    "POWER_SPECTRUM": {
        "url": "https://lambda.gsfc.nasa.gov/data/map/dr5/powspec/wmap_tt_spectrum_9yr_v5.txt",
        "description": "WMAP 9-year Temperature Power Spectrum"
    },
    # Analysis mask
    "ANALYSIS_MASK": {
        "url": "https://lambda.gsfc.nasa.gov/data/map/dr5/ancillary/masks/wmap_temperature_analysis_mask_r9_9yr_v5.fits",
        "description": "WMAP 9-year Analysis Mask"
    },
    # K-band map (23 GHz)
    "K_BAND": {
        "url": "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/K_IQU_9yr_r9_imap_v5.fits",
        "description": "WMAP 9-year K-band (23 GHz) Map"
    },
    # Ka-band map (33 GHz)
    "KA_BAND": {
        "url": "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/Ka_IQU_9yr_r9_imap_v5.fits",
        "description": "WMAP 9-year Ka-band (33 GHz) Map"
    },
    # Q-band map (41 GHz)
    "Q_BAND": {
        "url": "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/Q_IQU_9yr_r9_imap_v5.fits",
        "description": "WMAP 9-year Q-band (41 GHz) Map"
    },
    # V-band map (61 GHz)
    "V_BAND": {
        "url": "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/V_IQU_9yr_r9_imap_v5.fits",
        "description": "WMAP 9-year V-band (61 GHz) Map"
    },
    # W-band map (94 GHz)
    "W_BAND": {
        "url": "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/W_IQU_9yr_r9_imap_v5.fits",
        "description": "WMAP 9-year W-band (94 GHz) Map"
    }
}


def download_file(url, output_path, description=None):
    """
    Download a file from a URL to the specified output path.
    
    Args:
        url (str): URL to download from
        output_path (str): Path to save the file to
        description (str): Description of the file (optional)
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    if description:
        print("Downloading {}...".format(description))
    else:
        print("Downloading {}...".format(os.path.basename(url)))
    
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Download with progress reporting
        def report_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write("\r...%d%%" % percent)
            sys.stdout.flush()
        
        urlretrieve(url, output_path, reporthook=report_progress)
        print("\nDownload complete: {}".format(output_path))
        return True
    
    except URLError as e:
        print("\nError downloading {}: {}".format(url, e))
        return False
    except Exception as e:
        print("\nUnexpected error: {}".format(e))
        return False


def download_wmap_data(data_types=None, output_dir="wmap_data/raw_data"):
    """
    Download specified WMAP data files.
    
    Args:
        data_types (list): List of data types to download (from WMAP_FILES keys)
                          If None, download all data types
        output_dir (str): Directory to save the data to
        
    Returns:
        dict: Dictionary of downloaded files with their paths
    """
    if data_types is None:
        data_types = list(WMAP_FILES.keys())
    elif isinstance(data_types, str):
        data_types = [data_types]
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Download each file
    downloaded_files = {}
    for data_type in data_types:
        if data_type not in WMAP_FILES:
            print("Warning: Unknown data type '{}'".format(data_type))
            continue
        
        file_info = WMAP_FILES[data_type]
        url = file_info["url"]
        description = file_info["description"]
        output_path = os.path.join(output_dir, os.path.basename(url))
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print("File already exists: {}".format(output_path))
            downloaded_files[data_type] = output_path
            continue
        
        # Download file
        success = download_file(url, output_path, description)
        if success:
            downloaded_files[data_type] = output_path
        
        # Add a small delay between downloads to avoid overwhelming the server
        time.sleep(1)
    
    return downloaded_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download WMAP data files")
    parser.add_argument("--data-types", nargs="+", choices=list(WMAP_FILES.keys()),
                        help="Data types to download (default: all)")
    parser.add_argument("--output-dir", default="wmap_data/raw_data",
                        help="Directory to save the data to (default: wmap_data/raw_data)")
    parser.add_argument("--list", action="store_true",
                        help="List available data types and exit")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available WMAP data types:")
        for data_type, info in WMAP_FILES.items():
            print("  {:<15} - {}".format(data_type, info["description"]))
        sys.exit(0)
    
    print("Downloading WMAP data...")
    downloaded_files = download_wmap_data(args.data_types, args.output_dir)
    
    print("\nDownload summary:")
    for data_type, file_path in downloaded_files.items():
        print("  {:<15} - {}".format(data_type, file_path))
    
    print("\nTotal files downloaded: {}".format(len(downloaded_files)))
