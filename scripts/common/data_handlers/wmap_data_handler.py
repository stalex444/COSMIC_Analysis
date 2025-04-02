#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WMAP Data Handler

This module provides functions for downloading, loading, and preprocessing
WMAP (Wilkinson Microwave Anisotropy Probe) CMB data for use in the
Cosmic Consciousness Analysis framework.
"""

from __future__ import print_function
import os
import sys
import numpy as np
import urllib2
import healpy as hp
from astropy.io import fits

# Python 2.7 compatibility
try:
    input = raw_input
except NameError:
    pass

# URLs for WMAP data files
WMAP_DATA_URLS = {
    'ILC_MAP': 'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_ilc_9yr_v5.fits',
    'MASK': 'https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/wmap_temperature_analysis_mask_r9_9yr_v5.fits',
    'POWER_SPECTRUM': 'https://lambda.gsfc.nasa.gov/data/map/dr5/powspec/wmap_tt_spectrum_9yr_v5.txt'
}

# Output directory structure
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_directories():
    """Create the necessary directories if they don't exist."""
    if not os.path.exists(OUTPUT_DIR):
        print("Creating directory: {}".format(OUTPUT_DIR))
        os.makedirs(OUTPUT_DIR)


def download_wmap_data(data_type='POWER_SPECTRUM', output_dir=None):
    """
    Download WMAP data files.
    
    Args:
        data_type (str): Type of data to download ('ILC_MAP', 'MASK', or 'POWER_SPECTRUM')
        output_dir (str): Directory to save the downloaded files
        
    Returns:
        str: Path to the downloaded file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if data_type not in WMAP_DATA_URLS:
        raise ValueError("Invalid data type. Choose from: {}".format(", ".join(WMAP_DATA_URLS.keys())))
        
    url = WMAP_DATA_URLS[data_type]
    filename = os.path.basename(url)
    output_path = os.path.join(output_dir, filename)
    
    # Check if file already exists
    if os.path.exists(output_path):
        print("File already exists: {}".format(output_path))
        return output_path
        
    print("Downloading {} data from {}...".format(data_type, url))
    
    try:
        # Open the URL
        response = urllib2.urlopen(url)
        total_size = int(response.info().getheader('Content-Length', 0))
        
        # Download the file with progress reporting
        bytes_downloaded = 0
        block_size = 8192
        with open(output_path, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                
                bytes_downloaded += len(buffer)
                f.write(buffer)
                
                # Calculate and display progress
                if total_size > 0:
                    percent = bytes_downloaded * 100.0 / total_size
                    status = "\r[{:.2f}%] [{}/{}]".format(
                        percent, bytes_downloaded, total_size)
                    sys.stdout.write(status)
                    sys.stdout.flush()
        
        print("\nDownload complete: {}".format(output_path))
        return output_path
        
    except Exception as e:
        print("Error downloading {}: {}".format(data_type, str(e)))
        return None


def load_wmap_power_spectrum(file_path=None):
    """
    Load WMAP CMB power spectrum data.
    
    Args:
        file_path (str): Path to the WMAP power spectrum file
            If None, will attempt to download the file
            
    Returns:
        tuple: (ell, power, error_low, error_high) arrays
    """
    if file_path is None or not os.path.exists(file_path):
        print("WMAP power spectrum file not found. Attempting to download...")
        file_path = download_wmap_data('POWER_SPECTRUM')
        
    if file_path is None or not os.path.exists(file_path):
        raise FileNotFoundError("Could not find or download WMAP power spectrum file")
        
    try:
        # Load the power spectrum data
        # WMAP power spectrum format: l, TT, error
        data = np.loadtxt(file_path, skiprows=1)
        
        ell = data[:, 0]  # Multipole moment
        power = data[:, 1]  # Power spectrum value (Dl = l(l+1)Cl/2π in μK²)
        error = data[:, 2]  # Error
        
        # For compatibility with Planck format, create symmetric error bars
        error_low = power - error
        error_high = power + error
        
        return ell, power, error_low, error_high
        
    except Exception as e:
        print("Error loading WMAP power spectrum: {}".format(str(e)))
        return None, None, None, None


def load_wmap_map(map_path=None, mask_path=None):
    """
    Load WMAP CMB temperature map.
    
    Parameters:
    -----------
    map_path : str
        Path to the WMAP ILC map FITS file
        If None, will attempt to download the file
    mask_path : str, optional
        Path to the WMAP analysis mask
        If None, will attempt to download the file
        
    Returns:
    --------
    cmb_map : numpy.ndarray
        HEALPix map of CMB temperature fluctuations
    """
    # Download or use existing map file
    if map_path is None or not os.path.exists(map_path):
        print("WMAP ILC map file not found. Attempting to download...")
        map_path = download_wmap_data('ILC_MAP')
        
    if map_path is None or not os.path.exists(map_path):
        raise FileNotFoundError("Could not find or download WMAP ILC map file")
    
    # Load the ILC map
    try:
        wmap_map = hp.read_map(map_path)
        
        # Apply mask if provided or downloaded
        if mask_path is not None or mask_path is None:
            if mask_path is None or not os.path.exists(mask_path):
                print("WMAP mask file not found. Attempting to download...")
                mask_path = download_wmap_data('MASK')
                
            if mask_path is not None and os.path.exists(mask_path):
                mask = hp.read_map(mask_path)
                wmap_map = wmap_map * mask  # Apply mask (zeros out masked regions)
        
        return wmap_map
        
    except Exception as e:
        print("Error loading WMAP map: {}".format(str(e)))
        return None


def healpix_to_timeseries(healpix_map, n_samples=4096):
    """
    Convert a HEALPix map to a 1D time series for analysis.
    
    Parameters:
    -----------
    healpix_map : numpy.ndarray
        HEALPix map
    n_samples : int
        Number of samples to extract
        
    Returns:
    --------
    timeseries : numpy.ndarray
        1D array of values
    """
    # Remove masked pixels (NaN or zero values)
    valid_pixels = healpix_map[np.isfinite(healpix_map) & (healpix_map != 0)]
    
    # If we have enough valid pixels, sample them
    if len(valid_pixels) >= n_samples:
        # Randomly sample pixels to create time series
        indices = np.random.choice(len(valid_pixels), n_samples, replace=False)
        timeseries = valid_pixels[indices]
    else:
        # If not enough valid pixels, use all and pad if necessary
        timeseries = valid_pixels
        if len(timeseries) < n_samples:
            # Pad by repeating with small random variations
            n_pad = n_samples - len(timeseries)
            pad = np.random.normal(np.mean(timeseries), np.std(timeseries) * 0.1, n_pad)
            timeseries = np.concatenate([timeseries, pad])
    
    # Normalize
    timeseries = (timeseries - np.mean(timeseries)) / np.std(timeseries)
    
    return timeseries


def preprocess_wmap_data(data, smooth=False, smooth_window=5, normalize=True, detrend=False):
    """
    Preprocess WMAP data for analysis.
    
    Args:
        data (numpy.ndarray): Input data array
        smooth (bool): Whether to apply smoothing
        smooth_window (int): Window size for smoothing
        normalize (bool): Whether to normalize the data
        detrend (bool): Whether to remove linear trend
        
    Returns:
        numpy.ndarray: Preprocessed data
    """
    # Make a copy to avoid modifying the original
    processed_data = np.copy(data)
    
    # Remove NaN values
    if np.any(np.isnan(processed_data)):
        processed_data = processed_data[~np.isnan(processed_data)]
    
    # Detrend (remove linear trend)
    if detrend:
        from scipy import signal
        processed_data = signal.detrend(processed_data)
    
    # Smooth the data
    if smooth:
        from scipy.ndimage import gaussian_filter1d
        processed_data = gaussian_filter1d(processed_data, smooth_window)
    
    # Normalize to zero mean and unit variance
    if normalize:
        processed_data = (processed_data - np.mean(processed_data)) / np.std(processed_data)
    
    return processed_data


def get_wmap_data(use_power_spectrum=True, n_samples=4096, **kwargs):
    """
    Get WMAP data ready for analysis.
    
    Args:
        use_power_spectrum (bool): Whether to use power spectrum (True) or map (False)
        n_samples (int): Number of samples to extract from map (if use_power_spectrum=False)
        **kwargs: Additional preprocessing parameters
        
    Returns:
        numpy.ndarray: Processed WMAP data ready for analysis
    """
    if use_power_spectrum:
        # Get power spectrum
        ell, power, _, _ = load_wmap_power_spectrum()
        if power is None:
            raise ValueError("Failed to load WMAP power spectrum")
        
        # Use power values as data
        data = power
    else:
        # Get map and convert to time series
        wmap_map = load_wmap_map()
        if wmap_map is None:
            raise ValueError("Failed to load WMAP map")
        
        # Convert map to time series
        data = healpix_to_timeseries(wmap_map, n_samples=n_samples)
    
    # Preprocess
    processed_data = preprocess_wmap_data(data, **kwargs)
    
    return processed_data


def compare_wmap_planck(wmap_data, planck_data):
    """
    Compare WMAP and Planck datasets.
    
    Args:
        wmap_data (numpy.ndarray): Processed WMAP data
        planck_data (numpy.ndarray): Processed Planck data
        
    Returns:
        dict: Comparison metrics
    """
    # Ensure both datasets have the same length
    min_length = min(len(wmap_data), len(planck_data))
    wmap_data = wmap_data[:min_length]
    planck_data = planck_data[:min_length]
    
    # Calculate basic statistics
    wmap_mean = np.mean(wmap_data)
    planck_mean = np.mean(planck_data)
    wmap_std = np.std(wmap_data)
    planck_std = np.std(planck_data)
    
    # Calculate correlation
    correlation = np.corrcoef(wmap_data, planck_data)[0, 1]
    
    # Calculate power spectrum for both
    from scipy import signal
    wmap_f, wmap_psd = signal.welch(wmap_data, nperseg=min(1024, min_length//4))
    planck_f, planck_psd = signal.welch(planck_data, nperseg=min(1024, min_length//4))
    
    # Calculate power ratio
    power_ratio = np.sum(wmap_psd) / np.sum(planck_psd)
    
    # Return comparison metrics
    return {
        'correlation': correlation,
        'mean_ratio': wmap_mean / planck_mean if planck_mean != 0 else float('inf'),
        'std_ratio': wmap_std / planck_std if planck_std != 0 else float('inf'),
        'power_ratio': power_ratio,
        'wmap_stats': {
            'mean': wmap_mean,
            'std': wmap_std,
            'min': np.min(wmap_data),
            'max': np.max(wmap_data)
        },
        'planck_stats': {
            'mean': planck_mean,
            'std': planck_std,
            'min': np.min(planck_data),
            'max': np.max(planck_data)
        }
    }


if __name__ == "__main__":
    """
    Main function to demonstrate usage of the WMAP data handler.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='WMAP Data Handler')
    parser.add_argument('--download', action='store_true', help='Download WMAP data')
    parser.add_argument('--type', choices=['POWER_SPECTRUM', 'ILC_MAP', 'MASK', 'ALL'], 
                        default='POWER_SPECTRUM', help='Type of data to download')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess and display data')
    parser.add_argument('--use-map', action='store_true', help='Use map instead of power spectrum')
    parser.add_argument('--smooth', action='store_true', help='Apply smoothing')
    parser.add_argument('--smooth-window', type=int, default=5, help='Smoothing window size')
    parser.add_argument('--n-samples', type=int, default=4096, help='Number of samples to extract from map')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    # Download data if requested
    if args.download:
        if args.type == 'ALL':
            for data_type in WMAP_DATA_URLS:
                download_wmap_data(data_type, args.output_dir)
        else:
            download_wmap_data(args.type, args.output_dir)
    
    # Preprocess and display data if requested
    if args.preprocess:
        try:
            # Get data
            data = get_wmap_data(
                use_power_spectrum=not args.use_map,
                n_samples=args.n_samples,
                smooth=args.smooth,
                smooth_window=args.smooth_window
            )
            
            print("\nData statistics:")
            print("  Mean: {:.4f}".format(np.mean(data)))
            print("  Std: {:.4f}".format(np.std(data)))
            print("  Min: {:.4f}".format(np.min(data)))
            print("  Max: {:.4f}".format(np.max(data)))
            print("  Shape: {}".format(data.shape))
            
            # Plot if matplotlib is available
            try:
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(10, 6))
                
                if not args.use_map:
                    # Plot power spectrum
                    ell, power, error_low, error_high = load_wmap_power_spectrum()
                    plt.errorbar(ell[:100], power[:100], 
                                yerr=[power[:100]-error_low[:100], error_high[:100]-power[:100]],
                                fmt='o', markersize=3, capsize=0, alpha=0.7)
                    plt.xlabel('Multipole moment (ℓ)')
                    plt.ylabel('Power (μK²)')
                    plt.title('WMAP 9-year CMB Power Spectrum')
                    plt.grid(True, alpha=0.3)
                else:
                    # Plot time series
                    plt.plot(data[:1000])
                    plt.xlabel('Sample')
                    plt.ylabel('Normalized amplitude')
                    plt.title('WMAP CMB Map (as time series)')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save or show
                if args.output_dir:
                    plot_path = os.path.join(args.output_dir, 
                                            'wmap_data_preview.png')
                    plt.savefig(plot_path, dpi=150)
                    print("\nPlot saved to: {}".format(plot_path))
                else:
                    plt.show()
                    
            except ImportError:
                print("\nMatplotlib not available for plotting.")
                
        except Exception as e:
            print("Error processing data: {}".format(str(e)))
