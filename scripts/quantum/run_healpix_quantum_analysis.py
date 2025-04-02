#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run HEALPix quantum entanglement analysis on CMB data.

This script:
1. Loads HEALPix CMB data from WMAP and/or Planck
2. Runs quantum entanglement tests focusing on golden ratio scale relationships
3. Generates comprehensive visualizations and reports
"""

import os
import sys
import time
import argparse
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from tqdm import tqdm
import logging

# Import local modules
from healpix_quantum_analysis import load_healpix_map, apply_mask
from healpix_quantum_gr_test import test_with_surrogates
from healpix_quantum_visualization import plot_results, plot_comparison

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map download URLs
WMAP_URL = "https://lambda.gsfc.nasa.gov/data/wmap/dr5/ilc/wmap_ilc_9yr_v5.fits"
PLANCK_URL = "https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_ID=COM_CMB_IQU-smica_2048_R3.00_full.fits"

def download_file(url, target_path, description="file"):
    """Download a file with progress tracking"""
    if os.path.exists(target_path):
        logger.info(f"{description} already downloaded at {target_path}")
        return target_path
    
    logger.info(f"Downloading {description} from {url}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Start the download
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error for bad responses
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(target_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        progress_bar.close()
        
        logger.info(f"Successfully downloaded {description} to {target_path}")
        return target_path
    
    except Exception as e:
        logger.error(f"Error downloading {description}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)  # Remove partial download
        return None

def downgrade_map_if_needed(map_data, max_nside=256):
    """Downgrade high-resolution maps to save memory and computation time"""
    current_nside = hp.get_nside(map_data)
    if current_nside > max_nside:
        logger.info(f"Downgrading map from NSIDE={current_nside} to NSIDE={max_nside}")
        return hp.ud_grade(map_data, max_nside)
    return map_data

def run_analysis(map_path, dataset_name, base_scale=1.0, max_depth=5, n_samples=1000, 
                n_surrogates=10, mask_value=None, max_nside=256, output_dir=None):
    """
    Run quantum entanglement analysis on a CMB map
    
    Parameters:
    map_path (str): Path to the HEALPix map file
    dataset_name (str): Name of the dataset (e.g., "WMAP" or "Planck")
    base_scale (float): Base angular scale in degrees
    max_depth (int): Number of golden ratio scales to test
    n_samples (int): Number of random points to sample
    n_surrogates (int): Number of surrogate maps to generate
    mask_value (float): Value to use for masking extreme values
    max_nside (int): Maximum NSIDE parameter (will downgrade if higher)
    output_dir (str): Directory to save results
    
    Returns:
    dict: Results dictionary
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"../../results/quantum_entanglement_{dataset_name.lower()}_{timestamp}"
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, f"{dataset_name.lower()}_analysis_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting quantum entanglement analysis for {dataset_name}")
    
    # Load the map
    logger.info(f"Loading {dataset_name} map from {map_path}")
    map_data, nside = load_healpix_map(map_path)
    
    if map_data is None:
        logger.error(f"Failed to load {dataset_name} map. Skipping analysis.")
        return None
    
    # Create a simple visualization of the original map
    hp.mollview(map_data, title=f"{dataset_name} CMB Map", unit="mK")
    plt.savefig(os.path.join(output_dir, f"{dataset_name.lower()}_original_map.png"), dpi=300)
    plt.close()
    
    # Apply mask if requested
    if mask_value is not None:
        logger.info(f"Applying mask with value {mask_value}")
        map_data = apply_mask(map_data, mask_value=mask_value)
        
        # Visualize masked map
        hp.mollview(map_data, title=f"{dataset_name} CMB Map (Masked)", unit="mK")
        plt.savefig(os.path.join(output_dir, f"{dataset_name.lower()}_masked_map.png"), dpi=300)
        plt.close()
    
    # Downgrade if needed
    if max_nside is not None:
        map_data = downgrade_map_if_needed(map_data, max_nside)
    
    # Run the analysis
    logger.info(f"Running quantum entanglement test with {n_surrogates} surrogates")
    start_time = time.time()
    
    results = test_with_surrogates(
        map_data, 
        n_surrogates=n_surrogates, 
        base_scale=base_scale, 
        max_depth=max_depth, 
        n_samples=n_samples
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    # Add metadata to results
    results['dataset'] = dataset_name
    results['execution_time'] = elapsed_time
    results['base_scale'] = base_scale
    results['max_depth'] = max_depth
    results['n_samples'] = n_samples
    results['n_surrogates'] = n_surrogates
    
    # Create visualization
    logger.info("Creating visualization...")
    plot_results(results, dataset_name, output_dir)
    
    # Remove file handler
    logger.removeHandler(file_handler)
    
    return results

def main():
    """Main function to parse arguments and run analysis"""
    parser = argparse.ArgumentParser(description="Run HEALPix Quantum Entanglement Analysis on CMB Data")
    
    # Dataset selection
    parser.add_argument("--wmap-only", action="store_true", help="Only analyze WMAP data")
    parser.add_argument("--planck-only", action="store_true", help="Only analyze Planck data")
    
    # Analysis parameters
    parser.add_argument("--base-scale", type=float, default=1.0,
                       help="Base angular scale in degrees (default: 1.0)")
    parser.add_argument("--max-depth", type=int, default=5,
                       help="Number of golden ratio scales to test (default: 5)")
    parser.add_argument("--n-samples", type=int, default=5000,
                       help="Number of random points to sample (default: 5000)")
    parser.add_argument("--n-surrogates", type=int, default=10,
                       help="Number of surrogate maps to generate (default: 10)")
    
    # Technical parameters
    parser.add_argument("--mask-value", type=float, default=None,
                       help="Mask values above this threshold (default: None)")
    parser.add_argument("--max-nside", type=int, default=256,
                       help="Maximum NSIDE parameter (default: 256)")
    
    # Output control
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                       help="Output directory for results (default: auto-generated)")
    
    # Custom map files
    parser.add_argument("--wmap-file", type=str, default=None,
                       help="Custom path to WMAP FITS file")
    parser.add_argument("--planck-file", type=str, default=None,
                       help="Custom path to Planck FITS file")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with minimal resources")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"../../results/quantum_entanglement_{timestamp}"
    
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, "analysis_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting HEALPix Quantum Entanglement Analysis")
    logger.info(f"Output directory: {output_dir}")
    
    # If debug mode, use minimal resources
    if args.debug:
        logger.info("Running in DEBUG mode with minimal resources")
        args.n_samples = 100
        args.n_surrogates = 2
        args.max_depth = 3
        args.max_nside = 32
    
    # Log parameters
    logger.info(f"Parameters:")
    logger.info(f"  Base scale: {args.base_scale} degrees")
    logger.info(f"  Max depth: {args.max_depth}")
    logger.info(f"  Number of samples: {args.n_samples}")
    logger.info(f"  Number of surrogates: {args.n_surrogates}")
    logger.info(f"  Max NSIDE: {args.max_nside}")
    if args.mask_value is not None:
        logger.info(f"  Mask value: {args.mask_value}")
    
    # Data directories
    data_dir = os.path.abspath("../../data")
    wmap_dir = os.path.join(data_dir, "wmap")
    planck_dir = os.path.join(data_dir, "planck")
    
    os.makedirs(wmap_dir, exist_ok=True)
    os.makedirs(planck_dir, exist_ok=True)
    
    # File paths for downloaded data
    wmap_file = args.wmap_file if args.wmap_file else os.path.join(wmap_dir, "wmap_ilc_9yr_v5.fits")
    planck_file = args.planck_file if args.planck_file else os.path.join(planck_dir, "COM_CMB_IQU-smica_2048_R3.00_full.fits")
    
    # Run WMAP analysis
    wmap_results = None
    if not args.planck_only:
        # Download WMAP data if needed
        if not os.path.exists(wmap_file) and not args.wmap_file:
            wmap_file = download_file(WMAP_URL, wmap_file, "WMAP ILC 9-year map")
        
        if wmap_file and os.path.exists(wmap_file):
            wmap_output = os.path.join(output_dir, "wmap")
            wmap_results = run_analysis(
                wmap_file, "WMAP", 
                base_scale=args.base_scale,
                max_depth=args.max_depth,
                n_samples=args.n_samples,
                n_surrogates=args.n_surrogates,
                mask_value=args.mask_value,
                max_nside=args.max_nside,
                output_dir=wmap_output
            )
        else:
            logger.error("WMAP file not found and could not be downloaded")
    
    # Run Planck analysis
    planck_results = None
    if not args.wmap_only:
        # Download Planck data if needed
        if not os.path.exists(planck_file) and not args.planck_file:
            planck_file = download_file(PLANCK_URL, planck_file, "Planck SMICA map")
        
        if planck_file and os.path.exists(planck_file):
            planck_output = os.path.join(output_dir, "planck")
            planck_results = run_analysis(
                planck_file, "Planck", 
                base_scale=args.base_scale,
                max_depth=args.max_depth,
                n_samples=args.n_samples,
                n_surrogates=args.n_surrogates,
                mask_value=args.mask_value,
                max_nside=args.max_nside,
                output_dir=planck_output
            )
        else:
            logger.error("Planck file not found and could not be downloaded")
    
    # Compare results
    if wmap_results and planck_results:
        logger.info("Creating comparison between WMAP and Planck results")
        plot_comparison(wmap_results, planck_results, output_dir)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    # Remove file handler
    logger.removeHandler(file_handler)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.exception("Unhandled exception")
        sys.exit(1)
