#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct downloader for Planck and WMAP CMB maps from NASA's LAMBDA archive
"""

import os
import sys
import urllib.request
import urllib.error
import ssl
import time
from tqdm import tqdm

# LAMBDA direct links to CMB maps
DATA_URLS = {
    "wmap": {
        "ilc": "https://lambda.gsfc.nasa.gov/data/map/dr5/ilc_map/wmap_ilc_9yr_v5.fits",
        "temperature": "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_band_iqusmap_r9_9yr_V_v5.fits"
    },
    "planck": {
        "commander": "https://lambda.gsfc.nasa.gov/data/planck/data/commander_dx11d2_nside256.fits", 
        "smica": "https://lambda.gsfc.nasa.gov/data/planck/data/smica_nside1024_rj.fits",
        "nilc": "https://lambda.gsfc.nasa.gov/data/planck/data/nilc_nside1024_rj.fits"
    }
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file with progress bar, handling SSL and redirects"""
    try:
        # Create SSL context that ignores certificate validation (needed for some servers)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Add headers to simulate a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Create the request
        req = urllib.request.Request(url, headers=headers)
        
        # Open the URL with our SSL context
        with urllib.request.urlopen(req, context=ssl_context) as response:
            file_size = int(response.info().get('Content-Length', -1))
            
            # Check if we're getting a small error page instead of the actual file
            if 'fits' in url.lower() and file_size > 0 and file_size < 50000:
                print(f"Warning: The file size ({file_size} bytes) seems too small for a FITS file.")
                print("This might be an error page rather than the actual data.")
                return False
            
            # Set up progress bar
            with DownloadProgressBar(unit='B', unit_scale=True, 
                                    miniters=1, desc=os.path.basename(output_path)) as t:
                
                with open(output_path, 'wb') as f:
                    block_size = 8192  # 8 KB
                    downloaded = 0
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        downloaded += len(buffer)
                        t.update_to(b=1, bsize=len(buffer), tsize=file_size)
            
            # Verify file size if we know what it should be
            if file_size > 0 and os.path.getsize(output_path) != file_size:
                print("Warning: Downloaded file size doesn't match expected size!")
                print(f"Expected: {file_size} bytes, Got: {os.path.getsize(output_path)} bytes")
                return False
            
            return True
                
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        if e.code == 403:
            print("This URL requires authentication or is not publicly accessible.")
        return False
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    """Download Planck and WMAP data from NASA LAMBDA"""
    # Set up output directories
    data_dir = os.path.abspath("../../data")
    wmap_dir = os.path.join(data_dir, "wmap")
    planck_dir = os.path.join(data_dir, "planck")
    
    os.makedirs(wmap_dir, exist_ok=True)
    os.makedirs(planck_dir, exist_ok=True)
    
    # Download WMAP ILC map
    print("\n=== Downloading WMAP ILC map ===")
    wmap_ilc_path = os.path.join(wmap_dir, "wmap_ilc_9yr_v5.fits")
    if os.path.exists(wmap_ilc_path) and os.path.getsize(wmap_ilc_path) > 1000000:
        print(f"WMAP ILC map already exists at {wmap_ilc_path}")
    else:
        print(f"Downloading WMAP ILC map to {wmap_ilc_path}")
        success = download_url(DATA_URLS["wmap"]["ilc"], wmap_ilc_path)
        if success:
            print("Successfully downloaded WMAP ILC map")
        else:
            print("Failed to download WMAP ILC map")
    
    # Download Planck SMICA map
    print("\n=== Downloading Planck SMICA map ===")
    planck_smica_path = os.path.join(planck_dir, "smica_nside1024_rj.fits")
    if os.path.exists(planck_smica_path) and os.path.getsize(planck_smica_path) > 1000000:
        print(f"Planck SMICA map already exists at {planck_smica_path}")
    else:
        print(f"Downloading Planck SMICA map to {planck_smica_path}")
        success = download_url(DATA_URLS["planck"]["smica"], planck_smica_path)
        if success:
            print("Successfully downloaded Planck SMICA map")
        else:
            print("Failed to download Planck SMICA map")
            
            # Try alternative downloads
            print("\n=== Trying alternative Planck maps ===")
            for map_type, url in DATA_URLS["planck"].items():
                if map_type == "smica":
                    continue  # Skip SMICA as we already tried it
                    
                output_path = os.path.join(planck_dir, f"{map_type}_nside1024.fits")
                print(f"Downloading Planck {map_type.upper()} map to {output_path}")
                success = download_url(url, output_path)
                if success:
                    print(f"Successfully downloaded Planck {map_type.upper()} map")
                    break
                else:
                    print(f"Failed to download Planck {map_type.upper()} map")
    
    print("\nDownload operations completed")
    
    # Check what we have
    print("\n=== Available Maps ===")
    for directory, name in [(wmap_dir, "WMAP"), (planck_dir, "Planck")]:
        print(f"\n{name} Maps:")
        for file in os.listdir(directory):
            if file.endswith(".fits"):
                file_path = os.path.join(directory, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                print(f"  - {file} ({file_size:.2f} MB)")

if __name__ == "__main__":
    main()
