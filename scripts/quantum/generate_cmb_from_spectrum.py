#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a realistic CMB map from a power spectrum file,
optionally embedding golden ratio features
"""

import os
import sys
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import interp1d

def read_power_spectrum(filename):
    """
    Read a power spectrum from a file with columns: ell, C_ell
    
    Parameters:
    -----------
    filename : str
        Path to the power spectrum file
    
    Returns:
    --------
    tuple : (ell, cl) arrays
    """
    print(f"Reading power spectrum from {filename}")
    try:
        data = np.loadtxt(filename)
        if data.shape[1] >= 2:
            ell = data[:, 0].astype(int)
            cl = data[:, 1]
            return ell, cl
        else:
            raise ValueError("File doesn't have at least 2 columns (ell, C_ell)")
    except Exception as e:
        print(f"Error reading power spectrum: {e}")
        sys.exit(1)

def extend_power_spectrum(ell, cl, lmax):
    """
    Extend a power spectrum to cover ell=0 to lmax
    
    Parameters:
    -----------
    ell : array
        Multipole values
    cl : array
        Power spectrum values
    lmax : int
        Maximum multipole
    
    Returns:
    --------
    array : Extended power spectrum
    """
    # Create full ell range
    full_ell = np.arange(lmax + 1)
    
    # Create full power spectrum with zeros
    full_cl = np.zeros(lmax + 1)
    
    # Set monopole and dipole to zero (cosmological convention)
    full_cl[0] = 0.0
    full_cl[1] = 0.0
    
    # Fill in measured values
    mask = np.where((full_ell >= np.min(ell)) & (full_ell <= np.max(ell)))[0]
    
    # Use interpolation for values within the measured range
    interp_func = interp1d(ell, cl, bounds_error=False, fill_value=0.0)
    full_cl[mask] = interp_func(full_ell[mask])
    
    # For higher ell values, extrapolate using a power law
    if lmax > np.max(ell):
        # Use last few points to estimate power law
        high_idx = len(ell) - 10 if len(ell) > 10 else 0
        log_ell = np.log(ell[high_idx:])
        log_cl = np.log(cl[high_idx:])
        
        # Simple linear fit in log-log space
        coeffs = np.polyfit(log_ell, log_cl, 1)
        slope, intercept = coeffs
        
        # Fill in extrapolated values
        high_mask = np.where(full_ell > np.max(ell))[0]
        full_cl[high_mask] = np.exp(intercept + slope * np.log(full_ell[high_mask]))
    
    return full_cl

def add_golden_ratio_correlations(alm, lmax, gr_bias=0.2, seed=None):
    """
    Add correlations between multipoles related by the golden ratio
    
    Parameters:
    -----------
    alm : complex array
        Spherical harmonic coefficients
    lmax : int
        Maximum multipole
    gr_bias : float
        Strength of the golden ratio correlations (0-1)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    complex array : Modified alm
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Find multipole pairs related by golden ratio
    gr_pairs = []
    for l1 in range(2, lmax+1):
        l2 = int(round(l1 * phi))
        if l2 <= lmax:
            gr_pairs.append((l1, l2))
    
    print(f"Found {len(gr_pairs)} golden ratio multipole pairs")
    
    # Add correlations
    for l1, l2 in gr_pairs:
        # Get indices for these multipoles
        idx1 = hp.Alm.getidx(lmax, l1, np.arange(0, l1+1))
        idx2 = hp.Alm.getidx(lmax, l2, np.arange(0, l2+1))
        
        # Ensure we have indices to work with
        min_len = min(len(idx1), len(idx2))
        if min_len < 2:
            continue
            
        idx1 = idx1[:min_len]
        idx2 = idx2[:min_len]
        
        # Create correlation by sharing phase components
        phase1 = np.angle(alm[idx1])
        phase2 = np.angle(alm[idx2])
        
        # Blend phases based on bias strength
        common_phase = (phase1 + phase2) / 2
        
        new_phase1 = (1 - gr_bias) * phase1 + gr_bias * common_phase
        new_phase2 = (1 - gr_bias) * phase2 + gr_bias * common_phase
        
        # Update alms with new phases but preserve amplitudes
        amp1 = np.abs(alm[idx1])
        amp2 = np.abs(alm[idx2])
        
        alm[idx1] = amp1 * np.exp(1j * new_phase1)
        alm[idx2] = amp2 * np.exp(1j * new_phase2)
    
    return alm

def main():
    parser = argparse.ArgumentParser(description="Generate realistic CMB map from power spectrum")
    parser.add_argument("--spectrum", type=str, required=True,
                       help="Power spectrum file with columns: ell, C_ell")
    parser.add_argument("--output", type=str, default=None,
                       help="Output FITS file for the generated map")
    parser.add_argument("--nside", type=int, default=512,
                       help="HEALPix NSIDE parameter for the output map")
    parser.add_argument("--lmax", type=int, default=1500,
                       help="Maximum multipole for the generated map")
    parser.add_argument("--gr-bias", type=float, default=0.0,
                       help="Golden ratio correlation bias (0-1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Read power spectrum
    ell, cl = read_power_spectrum(args.spectrum)
    
    # Extend power spectrum to full range
    full_cl = extend_power_spectrum(ell, cl, args.lmax)
    
    print(f"Generating CMB map with NSIDE={args.nside}, LMAX={args.lmax}")
    
    # Generate random alms
    alm = hp.synalm(full_cl, lmax=args.lmax)
    
    # Add golden ratio correlations if requested
    if args.gr_bias > 0:
        print(f"Adding golden ratio correlations with bias={args.gr_bias}")
        alm = add_golden_ratio_correlations(alm, args.lmax, args.gr_bias, args.seed)
    
    # Convert to map
    cmb_map = hp.alm2map(alm, args.nside)
    
    # Set output filename if not provided
    if args.output is None:
        basename = os.path.basename(args.spectrum)
        basename = os.path.splitext(basename)[0]
        output_dir = os.path.dirname(args.spectrum)
        if args.gr_bias > 0:
            args.output = os.path.join(output_dir, f"{basename}_nside{args.nside}_gr{args.gr_bias}.fits")
        else:
            args.output = os.path.join(output_dir, f"{basename}_nside{args.nside}.fits")
    
    # Save map
    print(f"Saving map to {args.output}")
    hp.write_map(args.output, cmb_map, overwrite=True)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    hp.mollview(cmb_map, title=f"Simulated CMB Map (NSIDE={args.nside})", unit="mK")
    
    # Save visualization
    viz_file = os.path.splitext(args.output)[0] + ".png"
    plt.savefig(viz_file, dpi=300)
    print(f"Saved visualization to {viz_file}")
    
    print(f"Created CMB map with {len(cmb_map)} pixels")
    print(f"  Min value: {np.min(cmb_map)}")
    print(f"  Max value: {np.max(cmb_map)}")
    print(f"  Mean value: {np.mean(cmb_map)}")
    print(f"  Std dev: {np.std(cmb_map)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
