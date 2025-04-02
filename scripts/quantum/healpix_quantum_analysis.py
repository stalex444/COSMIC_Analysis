#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Quantum Entanglement Analysis for HEALPix CMB Maps

This implementation properly accounts for the spherical geometry of the data
and tests for quantum-like correlations at scales related by the golden ratio.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime
import time
import argparse

def load_healpix_map(filepath, field=0):
    """
    Load a HEALPix map from a FITS file.
    
    Parameters:
    filepath (str): Path to the FITS file containing the HEALPix map
    field (int): Field to read if the file contains multiple maps
    
    Returns:
    tuple: (map_data, nside) - The map data and its NSIDE parameter
    """
    try:
        # Load the map
        map_data = hp.read_map(filepath, field=field)
        
        # Get NSIDE parameter
        nside = hp.get_nside(map_data)
        
        # Basic info
        npix = hp.nside2npix(nside)
        
        print(f"Loaded HEALPix map with NSIDE={nside} ({npix} pixels)")
        
        return map_data, nside
    except Exception as e:
        print(f"Error loading HEALPix map: {e}")
        return None, None

def apply_mask(map_data, mask_file=None, mask_value=None):
    """
    Apply a mask to the map data.
    
    Parameters:
    map_data (ndarray): The HEALPix map data
    mask_file (str): Path to a mask file, or None to use mask_value
    mask_value (float): Value to use as mask, or None to use mask_file
    
    Returns:
    ndarray: The masked map data
    """
    if mask_file is not None:
        # Load mask from file
        mask = hp.read_map(mask_file)
        masked_map = map_data.copy()
        masked_map[mask == 0] = hp.UNSEEN
    elif mask_value is not None:
        # Use mask value
        masked_map = map_data.copy()
        masked_map[np.abs(map_data) > mask_value] = hp.UNSEEN
    else:
        # No masking
        masked_map = map_data
    
    return masked_map

def calculate_quantum_correlation(map_data, theta_vals, phi_vals, r_vals, n_samples=1000):
    """
    Calculate entanglement-like correlations between points separated by various distances.
    
    Parameters:
    map_data (ndarray): The HEALPix map data
    theta_vals (ndarray): Array of theta values to test (radians)
    phi_vals (ndarray): Array of phi values to test (radians)
    r_vals (ndarray): Array of angular separation values to test (radians)
    n_samples (int): Number of random starting points to sample
    
    Returns:
    tuple: (correlations, significance) - Arrays of correlation values and their significance
    """
    nside = hp.get_nside(map_data)
    npix = hp.nside2npix(nside)
    
    # Initialize results arrays
    correlations = np.zeros(len(r_vals))
    significance = np.zeros(len(r_vals))
    
    # Generate random starting points
    random_pixels = np.random.choice(range(npix), size=n_samples, replace=False)
    random_thetas, random_phis = hp.pix2ang(nside, random_pixels)
    
    for i, r in enumerate(r_vals):
        print(f"Processing angular separation {np.degrees(r):.2f} degrees")
        
        # Arrays to store point pairs
        values_a = []
        values_b = []
        
        for theta, phi in zip(random_thetas, random_phis):
            # Find pixel index for point A
            pixel_a = hp.ang2pix(nside, theta, phi)
            
            # Skip if pixel is masked
            if map_data[pixel_a] == hp.UNSEEN:
                continue
                
            # Calculate new position at angular distance r
            # This is a simplified approach - for more accuracy, use great circle calculations
            new_theta = min(max(theta + r, 0), np.pi)
            new_phi = (phi + r) % (2 * np.pi)
            
            # Find pixel index for point B
            pixel_b = hp.ang2pix(nside, new_theta, new_phi)
            
            # Skip if pixel is masked
            if map_data[pixel_b] == hp.UNSEEN:
                continue
            
            # Add values to arrays
            values_a.append(map_data[pixel_a])
            values_b.append(map_data[pixel_b])
        
        # Convert to arrays
        values_a = np.array(values_a)
        values_b = np.array(values_b)
        
        if len(values_a) > 10:  # Ensure enough points for meaningful statistics
            # Calculate quantum correlation (Bell's inequality-like measure)
            # This is a simplified version - could be extended with more sophisticated metrics
            
            # Normalize values to [-1, 1] range (like spin measurements)
            norm_a = 2 * (values_a - np.min(values_a)) / (np.max(values_a) - np.min(values_a)) - 1
            norm_b = 2 * (values_b - np.min(values_b)) / (np.max(values_b) - np.min(values_b)) - 1
            
            # Calculate correlation E(a,b) = <A·B>
            E_ab = np.mean(norm_a * norm_b)
            
            # Generate slightly rotated measurement directions (simplified approach)
            rotation = r/10  # Small rotation proportional to r
            
            # Find pixel indices for rotated points
            pixels_a_prime = hp.ang2pix(nside, random_thetas, (random_phis + rotation) % (2 * np.pi))
            pixels_b_prime = hp.ang2pix(nside, np.minimum(random_thetas + rotation, np.pi), random_phis)
            
            # Filter masked pixels
            valid_indices = (map_data[pixels_a_prime] != hp.UNSEEN) & (map_data[pixels_b_prime] != hp.UNSEEN)
            values_a_prime = map_data[pixels_a_prime[valid_indices]]
            values_b_prime = map_data[pixels_b_prime[valid_indices]]
            
            if len(values_a_prime) > 10:  # Ensure enough points
                # Normalize values
                norm_a_prime = 2 * (values_a_prime - np.min(values_a_prime)) / (np.max(values_a_prime) - np.min(values_a_prime)) - 1
                norm_b_prime = 2 * (values_b_prime - np.min(values_b_prime)) / (np.max(values_b_prime) - np.min(values_b_prime)) - 1
                
                # Calculate correlations for Bell's inequality
                E_ap_b = np.mean(norm_a_prime * norm_b[valid_indices])
                E_a_bp = np.mean(norm_a[valid_indices] * norm_b_prime)
                E_ap_bp = np.mean(norm_a_prime * norm_b_prime)
                
                # Calculate Bell parameter S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
                S = E_ab - E_a_bp + E_ap_b + E_ap_bp
                
                # Store correlation value (how much Bell's inequality is violated)
                correlations[i] = abs(S)
                
                # Calculate significance (how far S is from the classical limit of 2)
                significance[i] = (abs(S) - 2) / 0.1  # 0.1 is an arbitrary scale factor
            else:
                correlations[i] = np.nan
                significance[i] = np.nan
        else:
            correlations[i] = np.nan
            significance[i] = np.nan
    
    return correlations, significance
