#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMB Map Downloader

This script provides a robust way to download CMB maps from various sources
with proper error handling, retry logic, and progress tracking.

It provides download options for:
- WMAP 9-year ILC maps
- Planck 2018 SMICA maps
- Alternative sources when primary sources fail
"""

import os
import sys
import time
import argparse
import requests
from tqdm import tqdm

# Configuration - Update URLs if needed
SOURCES = {
    'wmap': [
        'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_iqusmap_r9_9yr_Ka_v5.fits',
        'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_iqusmap_r9_9yr_K_v5.fits',
        'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_iqusmap_r9_9yr_Q_v5.fits',
        'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_iqusmap_r9_9yr_V_v5.fits',
        'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_iqusmap_r9_9yr_W_v5.fits',
        'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_ilc_9yr_v5.fits',
    ],
    'planck': [
        'https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_1024_R2.02_full.fits',
        'https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R2.02_full.fits',
        'https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CMB_IQU-smica-field-Int_2048_R3.00.fits',
        'https://lambda.gsfc.nasa.gov/data/suborbital/Planck/plc2018/HFI_SkyMap_143_2048_R3.01_full.fits',
    ],
    'alternative': [
        # NASA LAMBDA mirror
        'https://lambda.gsfc.nasa.gov/product/planck/current/cmb_maps_fits.cfm',
        # ESA Planck Legacy Archive
        'https://pla.esac.esa.int',
        # IRSA mirror
        'https://irsa.ipac.caltech.edu/data/Planck/release_3',
    ]
}

def download_file(url, target_path, description="file", max_retries=3, timeout=30):
    """
    Download a file with robust error handling and progress tracking
    
    Parameters:
    -----------
    url : str
        URL to download from
    target_path : str
        Path to save the file to
    description : str
        Description of the file for logging
    max_retries : int
        Maximum number of retry attempts
    timeout : int
        Connection timeout in seconds
    
    Returns:
    --------
    bool: True if download was successful, False otherwise
    """
    if os.path.exists(target_path):
        print(f"{description} already exists at {target_path}")
        
        # Verify file integrity by checking size
        file_size = os.path.getsize(target_path)
        if file_size < 1000000:  # Less than 1MB is suspicious for a HEALPix map
            print(f"WARNING: Existing file is suspiciously small ({file_size} bytes). It may be incomplete.")
            choice = input("Do you want to re-download this file? (y/n): ").lower()
            if choice != 'y':
                return True
            print(f"Removing existing file and re-downloading...")
            os.remove(target_path)
        else:
            return True
    
    print(f"Attempting to download {description} from {url}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            # Start the download
            with requests.get(url, stream=True, timeout=timeout) as response:
                if response.status_code == 403 or response.status_code == 401:
                    print(f"Access denied (HTTP {response.status_code}). This URL may require authentication.")
                    return False
                
                response.raise_for_status()  # Raise error for bad responses
                
                # Get file size if available
                total_size = int(response.headers.get('content-length', 0))
                
                if total_size < 1000000 and 'fits' in url.lower():  # For FITS files, expect at least 1MB
                    print(f"WARNING: This download appears too small for a HEALPix map ({total_size} bytes).")
                    print("It may be an error page rather than the actual data.")
                    choice = input("Continue anyway? (y/n): ").lower()
                    if choice != 'y':
                        return False
                
                block_size = 1024 * 1024  # 1 MB
                
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
                
                with open(target_path, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                
                progress_bar.close()
                
                # Verify the download
                if total_size > 0 and os.path.getsize(target_path) < total_size * 0.9:
                    print(f"WARNING: Downloaded file is smaller than expected ({os.path.getsize(target_path)} vs {total_size} bytes)")
                    print("The download may have been interrupted or truncated.")
                    if attempt < max_retries - 1:
                        print(f"Retrying download (attempt {attempt+2}/{max_retries})...")
                        os.remove(target_path)
                        continue
                
                print(f"Successfully downloaded {description} to {target_path}")
                return True
                
        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            print(f"Error during download (attempt {attempt+1}/{max_retries}): {e}")
            if os.path.exists(target_path):
                os.remove(target_path)  # Remove partial download
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download after {max_retries} attempts.")
                return False
    
    return False

def list_available_sources():
    """Print available data sources"""
    print("\nAvailable data sources:")
    print("-" * 40)
    
    print("\nWMAP Sources:")
    for i, url in enumerate(SOURCES['wmap']):
        print(f"  {i+1}. {os.path.basename(url)}")
    
    print("\nPlanck Sources:")
    for i, url in enumerate(SOURCES['planck']):
        print(f"  {i+1}. {os.path.basename(url)}")
    
    print("\nAlternative Sources (manual download required):")
    for i, url in enumerate(SOURCES['alternative']):
        print(f"  {i+1}. {url}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download CMB maps from various sources")
    parser.add_argument("--data-dir", type=str, default="../../data",
                       help="Directory to save downloaded maps")
    parser.add_argument("--wmap", action="store_true",
                       help="Download WMAP 9-year ILC map")
    parser.add_argument("--planck", action="store_true",
                       help="Download Planck 2018 SMICA map")
    parser.add_argument("--all", action="store_true",
                       help="Try to download all available maps")
    parser.add_argument("--list", action="store_true",
                       help="List available data sources and exit")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_sources()
        return 0
    
    if not (args.wmap or args.planck or args.all):
        print("No download options selected. Use --wmap, --planck, or --all")
        parser.print_help()
        return 1
    
    # Create data directories
    data_dir = os.path.abspath(args.data_dir)
    wmap_dir = os.path.join(data_dir, "wmap")
    planck_dir = os.path.join(data_dir, "planck")
    
    os.makedirs(wmap_dir, exist_ok=True)
    os.makedirs(planck_dir, exist_ok=True)
    
    # Download WMAP data
    if args.wmap or args.all:
        print("\n=== Downloading WMAP Data ===")
        for url in SOURCES['wmap']:
            filename = os.path.basename(url)
            target_path = os.path.join(wmap_dir, filename)
            success = download_file(url, target_path, f"WMAP {filename}")
            if success and 'ilc' in filename.lower():
                print(f"Successfully downloaded main WMAP ILC map to {target_path}")
                # If we successfully got the ILC map and not doing --all, we can stop
                if not args.all:
                    break
    
    # Download Planck data
    if args.planck or args.all:
        print("\n=== Downloading Planck Data ===")
        success = False
        for url in SOURCES['planck']:
            filename = os.path.basename(url)
            target_path = os.path.join(planck_dir, filename)
            if download_file(url, target_path, f"Planck {filename}"):
                success = True
                print(f"Successfully downloaded Planck map to {target_path}")
                # If we're not doing --all, we can stop after first success
                if not args.all:
                    break
        
        if not success:
            print("\nAll automatic downloads failed. Consider manual download from these sources:")
            for url in SOURCES['alternative']:
                print(f"  - {url}")
            print("\nAfter downloading, place files in:")
            print(f"  - WMAP: {wmap_dir}")
            print(f"  - Planck: {planck_dir}")
    
    print("\nDownload operations completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
