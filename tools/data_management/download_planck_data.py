#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download Planck data files from ESA's Planck Legacy Archive for cross-validation
with WMAP analysis results.
"""

from __future__ import print_function
import os
import sys
import time
import shutil
try:
    # Python 3
    from urllib.request import urlretrieve, urlopen
    from urllib.error import URLError
except ImportError:
    # Python 2
    from urllib import urlretrieve
    from urllib2 import URLError, urlopen

# Direct URLs to Planck data files
PLANCK_DIRECT_URLS = {
    "POWER_SPECTRUM": {
        "url": "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt",
        "description": "Planck 2018 TT power spectrum",
        "local_filename": "planck_tt_spectrum_2018_R3.01.txt"
    },
    "CMB_MAP": {
        "url": "https://pla.esac.esa.int/pla/aio/product-action?MAP.FILE_ID=COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "description": "Planck 2018 CMB SMICA map",
        "local_filename": "planck_cmb_smica_2018_R3.00.fits"
    },
    "BINNED_POWER_SPECTRUM": {
        "url": "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-binned_R3.01.txt",
        "description": "Planck 2018 binned TT power spectrum",
        "local_filename": "planck_binned_tt_spectrum_2018_R3.01.txt"
    }
}

def download_file(url, output_path, description=None, force=True):
    """
    Download a file from a URL to the specified output path.
    
    Args:
        url (str): URL to download from
        output_path (str): Path to save the file to
        description (str): Description of the file (optional)
        force (bool): Force download even if file exists
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    if os.path.exists(output_path) and not force:
        print("File already exists: {}".format(output_path))
        return True
    
    if description:
        print("Downloading {}...".format(description))
    else:
        print("Downloading {}...".format(os.path.basename(url)))
    
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Remove existing file if force is True
        if os.path.exists(output_path) and force:
            print("Removing existing file: {}".format(output_path))
            os.remove(output_path)
        
        # First check if the URL is valid
        try:
            response = urlopen(url)
            if response.getcode() != 200:
                print("Error: URL returned status code {}".format(response.getcode()))
                return False
        except Exception as e:
            print("Error checking URL {}: {}".format(url, e))
            print("Trying direct download anyway...")
        
        # Download the file
        urlretrieve(url, output_path)
        print("Download complete: {}".format(output_path))
        
        # Verify file is not an HTML error page
        if output_path.endswith('.txt'):
            with open(output_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('<!DOCTYPE HTML') or first_line.startswith('<html'):
                    print("Error: Downloaded file is an HTML error page. URL may be incorrect.")
                    return False
        
        return True
    
    except URLError as e:
        print("Error downloading {}: {}".format(url, e))
        return False
    except Exception as e:
        print("Unexpected error: {}".format(e))
        return False


def download_planck_data(data_types=None, output_dir="planck_data/raw_data", force=True):
    """
    Download Planck data files.
    
    Args:
        data_types (list): List of data types to download (from PLANCK_DIRECT_URLS keys)
                          If None, download all data types
        output_dir (str): Directory to save the data to
        force (bool): Force download even if file exists
        
    Returns:
        dict: Dictionary of downloaded files with their paths
    """
    if data_types is None:
        data_types = list(PLANCK_DIRECT_URLS.keys())
    elif isinstance(data_types, str):
        data_types = [data_types]
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Download each file
    downloaded_files = {}
    for data_type in data_types:
        if data_type not in PLANCK_DIRECT_URLS:
            print("Warning: Unknown data type '{}'".format(data_type))
            continue
        
        file_info = PLANCK_DIRECT_URLS[data_type]
        url = file_info["url"]
        description = file_info["description"]
        local_filename = file_info.get("local_filename", os.path.basename(url))
        output_path = os.path.join(output_dir, local_filename)
        
        # Download file
        success = download_file(url, output_path, description, force=force)
        if success:
            downloaded_files[data_type] = output_path
        
        # Add a small delay between downloads to avoid overwhelming the server
        time.sleep(1)
    
    return downloaded_files


def verify_planck_data(output_dir="planck_data/raw_data"):
    """
    Verify that Planck data files are valid.
    
    Args:
        output_dir (str): Directory where data files are stored
        
    Returns:
        dict: Dictionary of verification results
    """
    verification_results = {}
    
    for data_type, file_info in PLANCK_DIRECT_URLS.items():
        local_filename = file_info.get("local_filename", os.path.basename(file_info["url"]))
        file_path = os.path.join(output_dir, local_filename)
        
        if not os.path.exists(file_path):
            verification_results[data_type] = "File not found"
            continue
        
        # Verify text files are not HTML error pages
        if file_path.endswith('.txt'):
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('<!DOCTYPE HTML') or first_line.startswith('<html'):
                        verification_results[data_type] = "HTML error page"
                    else:
                        verification_results[data_type] = "Valid"
            except Exception as e:
                verification_results[data_type] = "Error: {}".format(e)
        
        # Verify FITS files have valid headers
        elif file_path.endswith('.fits'):
            try:
                # Try to import astropy
                try:
                    from astropy.io import fits
                    has_astropy = True
                except ImportError:
                    has_astropy = False
                
                if has_astropy:
                    # Try to open the FITS file
                    with fits.open(file_path) as hdul:
                        if len(hdul) > 0:
                            verification_results[data_type] = "Valid"
                        else:
                            verification_results[data_type] = "Empty FITS file"
                else:
                    # If astropy is not available, just check file size
                    file_size = os.path.getsize(file_path)
                    if file_size > 1000:  # Arbitrary threshold
                        verification_results[data_type] = "Likely valid (file size: {} bytes)".format(file_size)
                    else:
                        verification_results[data_type] = "Suspicious file size: {} bytes".format(file_size)
            except Exception as e:
                verification_results[data_type] = "Error: {}".format(e)
        
        # Other file types
        else:
            file_size = os.path.getsize(file_path)
            verification_results[data_type] = "Unknown format, size: {} bytes".format(file_size)
    
    return verification_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Planck data files from ESA's Planck Legacy Archive")
    parser.add_argument("--data-types", nargs="+", choices=list(PLANCK_DIRECT_URLS.keys()),
                        help="Data types to download (default: all)")
    parser.add_argument("--output-dir", default="planck_data/raw_data",
                        help="Directory to save the data to (default: planck_data/raw_data)")
    parser.add_argument("--no-force", action="store_true",
                        help="Don't force re-download if file exists")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify files, don't download")
    
    args = parser.parse_args()
    
    if args.verify_only:
        print("Verifying Planck data files...")
        verification_results = verify_planck_data(args.output_dir)
        
        print("\nVerification results:")
        for data_type, result in verification_results.items():
            file_info = PLANCK_DIRECT_URLS[data_type]
            print("  {:<20} - {:<10} - {}".format(
                data_type, result, file_info["description"]))
    else:
        print("Downloading Planck data files...")
        downloaded_files = download_planck_data(
            args.data_types, args.output_dir, force=not args.no_force)
        
        print("\nDownload summary:")
        for data_type, file_path in downloaded_files.items():
            print("  {:<20} - {}".format(data_type, file_path))
        
        print("\nTotal files downloaded: {}".format(len(downloaded_files)))
        
        # Verify downloaded files
        print("\nVerifying downloaded files...")
        verification_results = verify_planck_data(args.output_dir)
        
        print("\nVerification results:")
        for data_type, result in verification_results.items():
            if data_type in downloaded_files:
                file_info = PLANCK_DIRECT_URLS[data_type]
                print("  {:<20} - {:<10} - {}".format(
                    data_type, result, file_info["description"]))
