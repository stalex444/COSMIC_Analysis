#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose the structure of FITS files and extract CMB data correctly
"""

import sys
import os
import numpy as np
import healpy as hp
from astropy.io import fits

def inspect_fits_file(filepath):
    """
    Inspect the structure of a FITS file and print detailed information
    about each HDU (Header-Data Unit).
    
    Parameters:
    -----------
    filepath : str
        Path to the FITS file
    
    Returns:
    --------
    None
    """
    print(f"Inspecting FITS file: {filepath}")
    print("-" * 50)
    
    try:
        with fits.open(filepath) as hdul:
            print(f"File contains {len(hdul)} HDU(s):")
            
            for i, hdu in enumerate(hdul):
                print(f"\nHDU {i}:")
                print(f"  Type: {type(hdu).__name__}")
                print(f"  Name: {hdu.name}")
                
                # Print header information
                print("  Header information:")
                for key, value in list(hdu.header.items())[:10]:  # First 10 items
                    print(f"    {key}: {value}")
                
                if len(hdu.header) > 10:
                    print(f"    ... ({len(hdu.header) - 10} more header items)")
                
                # Print data information if present
                if hasattr(hdu, 'data') and hdu.data is not None:
                    if isinstance(hdu.data, np.ndarray):
                        print("  Data information:")
                        print(f"    Shape: {hdu.data.shape}")
                        print(f"    Type: {hdu.data.dtype}")
                        
                        # For 1D or 2D arrays, show some statistics
                        if hdu.data.ndim <= 2:
                            try:
                                print(f"    Min: {np.min(hdu.data)}")
                                print(f"    Max: {np.max(hdu.data)}")
                                print(f"    Mean: {np.mean(hdu.data)}")
                                print(f"    Non-zero elements: {np.count_nonzero(hdu.data)}")
                            except:
                                print("    Could not compute statistics on this array")
                    else:
                        print("  Data is present but not a numpy array")
                else:
                    print("  No data in this HDU")
                
                # Check for NSIDE if likely a HEALPix map
                if hasattr(hdu, 'data') and hdu.data is not None and isinstance(hdu.data, np.ndarray):
                    if hdu.data.ndim == 1 or (hdu.data.ndim == 2 and hdu.data.shape[0] == 1):
                        try:
                            # Get the actual 1D array
                            data_1d = hdu.data.ravel() if hdu.data.ndim == 2 else hdu.data
                            
                            # Try to determine NSIDE
                            nside = hp.npix2nside(len(data_1d))
                            print(f"  Likely a HEALPix map with NSIDE = {nside}, NPIX = {len(data_1d)}")
                            
                            # Check if there are NaNs or UNSEEN values
                            nan_count = np.sum(np.isnan(data_1d))
                            unseen_count = np.sum(data_1d == hp.UNSEEN)
                            
                            if nan_count > 0:
                                print(f"    Contains {nan_count} NaN values")
                            if unseen_count > 0:
                                print(f"    Contains {unseen_count} UNSEEN values (healpy.UNSEEN)")
                        except:
                            print("  Not a valid HEALPix map or could not determine parameters")
    
    except Exception as e:
        print(f"Error reading the FITS file: {e}")
        return

def extract_cmb_map(filepath, output_filepath=None):
    """
    Try to extract CMB temperature map from a FITS file and save it in a clean format
    that can be easily loaded with healpy, optionally downgrading resolution.
    
    Parameters:
    -----------
    filepath : str
        Path to the input FITS file
    output_filepath : str
        Path to save the extracted map. If None, will use the original name with "_clean" appended
    
    Returns:
    --------
    str: Path to the extracted map file or None if extraction failed
    """
    if output_filepath is None:
        dir_name = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        name_parts = os.path.splitext(base_name)
        output_filepath = os.path.join(dir_name, f"{name_parts[0]}_clean{name_parts[1]}")
    
    print(f"Attempting to extract CMB map from: {filepath}")
    print(f"Will save to: {output_filepath}")
    print("-" * 50)
    
    try:
        with fits.open(filepath) as hdul:
            # Strategies for different file formats
            
            # Strategy 1: First HDU is a simple 1D array (most common for HEALPix maps)
            if len(hdul) > 0 and hasattr(hdul[0], 'data') and hdul[0].data is not None:
                if isinstance(hdul[0].data, np.ndarray) and (hdul[0].data.ndim == 1 or 
                                                          (hdul[0].data.ndim == 2 and hdul[0].data.shape[0] == 1)):
                    print("Strategy 1: Using first HDU as a 1D array")
                    map_data = hdul[0].data.ravel()  # Convert to 1D if it's 2D with first dim = 1
                    
                    # Validate this looks like a HEALPix map
                    try:
                        nside = hp.npix2nside(len(map_data))
                        print(f"Found valid HEALPix map with NSIDE = {nside}")
                        hp.write_map(output_filepath, map_data, overwrite=True)
                        print(f"Successfully saved map to {output_filepath}")
                        return output_filepath
                    except:
                        print("Data doesn't appear to be a valid HEALPix map, trying other strategies")
            
            # Strategy 2: Look for specific HDU names commonly used for CMB maps
            cmb_hdu_names = ['CMB', 'TEMPERATURE', 'I', 'T', 'IQU']
            for i, hdu in enumerate(hdul):
                if hdu.name in cmb_hdu_names and hasattr(hdu, 'data') and hdu.data is not None:
                    print(f"Strategy 2: Found HDU named '{hdu.name}'")
                    
                    # Handle different dimensionality
                    if hdu.data.ndim == 1:
                        map_data = hdu.data
                    elif hdu.data.ndim == 2:
                        # For IQU maps, the first component is usually temperature (I)
                        print(f"Found 2D array with shape {hdu.data.shape}, using first component")
                        map_data = hdu.data[0]
                    elif hdu.data.ndim == 3 and hdu.data.shape[0] == 1:
                        # Handle [1, n, m] arrays
                        print(f"Found 3D array with shape {hdu.data.shape}, using [0, :, :] slice")
                        map_data = hdu.data[0].ravel()
                    else:
                        print(f"Unexpected data dimensionality: {hdu.data.shape}")
                        continue
                    
                    # Validate and save
                    try:
                        nside = hp.npix2nside(len(map_data))
                        print(f"Found valid HEALPix map with NSIDE = {nside}")
                        hp.write_map(output_filepath, map_data, overwrite=True)
                        print(f"Successfully saved map to {output_filepath}")
                        return output_filepath
                    except:
                        print(f"Data in HDU '{hdu.name}' doesn't appear to be a valid HEALPix map")
            
            # Strategy 3: Look at all HDUs and find arrays that could be HEALPix maps
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'data') and hdu.data is not None:
                    if isinstance(hdu.data, np.ndarray):
                        # Check if 1D or can be converted to 1D
                        if hdu.data.ndim == 1:
                            map_data = hdu.data
                        elif hdu.data.ndim == 2:
                            print(f"HDU {i}: Found 2D array with shape {hdu.data.shape}")
                            # If multi-column array, try each column
                            for j in range(hdu.data.shape[1]):
                                print(f"  Trying column {j}...")
                                try:
                                    col_data = hdu.data[:, j]
                                    nside = hp.npix2nside(len(col_data))
                                    print(f"  Column {j} is a valid HEALPix map with NSIDE = {nside}")
                                    hp.write_map(output_filepath, col_data, overwrite=True)
                                    print(f"Successfully saved map to {output_filepath}")
                                    return output_filepath
                                except:
                                    pass
                            
                            # If that didn't work, try flattening the whole array
                            print("  Trying flattened array...")
                            try:
                                flat_data = hdu.data.ravel()
                                nside = hp.npix2nside(len(flat_data))
                                print(f"  Flattened data is a valid HEALPix map with NSIDE = {nside}")
                                hp.write_map(output_filepath, flat_data, overwrite=True)
                                print(f"Successfully saved map to {output_filepath}")
                                return output_filepath
                            except:
                                print("  Flattened data doesn't appear to be a valid HEALPix map")
                        elif hdu.data.ndim == 3:
                            print(f"HDU {i}: Found 3D array with shape {hdu.data.shape}")
                            # Try the first slice (often I/T for IQU maps)
                            try:
                                slice_data = hdu.data[0].ravel()
                                nside = hp.npix2nside(len(slice_data))
                                print(f"  First slice is a valid HEALPix map with NSIDE = {nside}")
                                hp.write_map(output_filepath, slice_data, overwrite=True)
                                print(f"Successfully saved map to {output_filepath}")
                                return output_filepath
                            except:
                                print("  First slice doesn't appear to be a valid HEALPix map")
            
            print("Could not find a valid HEALPix map in this FITS file")
            return None
                
    except Exception as e:
        print(f"Error processing the FITS file: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_fits_format.py <path_to_fits_file> [extract_map]")
        return 1
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        return 1
    
    # Inspect the FITS file
    inspect_fits_file(filepath)
    
    # Extract map if requested
    if len(sys.argv) > 2 and sys.argv[2] == "extract_map":
        output_filepath = extract_cmb_map(filepath)
        if output_filepath:
            print(f"Successfully extracted CMB map to: {output_filepath}")
        else:
            print("Failed to extract CMB map")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
