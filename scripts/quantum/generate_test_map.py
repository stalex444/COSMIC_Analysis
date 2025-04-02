#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate test HEALPix maps for quantum entanglement analysis testing.
Creates maps with and without embedded golden ratio correlations.
"""

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from datetime import datetime

def generate_test_map(nside=128, lmax=256, golden_ratio_bias=0.0, seed=42):
    """
    Generate a test HEALPix map with optional embedded golden ratio correlations.
    
    Parameters:
    -----------
    nside : int
        HEALPix nside parameter
    lmax : int
        Maximum multipole
    golden_ratio_bias : float
        Strength of golden ratio correlations (0-1)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple: (map_data, alm)
        The HEALPix map and its spherical harmonic coefficients
    """
    print(f"Generating test map with NSIDE={nside}, LMAX={lmax}")
    np.random.seed(seed)
    
    # Create power spectrum similar to CMB
    ell = np.arange(lmax+1)
    cl = np.zeros(lmax+1)
    cl[1:] = 1.0 / (1.0 + 0.1*ell[1:])**2  # Simple power law spectrum
    
    # Generate random alms
    alm = hp.synalm(cl, lmax=lmax)
    
    # If requested, add golden ratio correlations
    if golden_ratio_bias > 0:
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        print(f"Adding golden ratio correlations (bias={golden_ratio_bias})")
        
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
            
            new_phase1 = (1 - golden_ratio_bias) * phase1 + golden_ratio_bias * common_phase
            new_phase2 = (1 - golden_ratio_bias) * phase2 + golden_ratio_bias * common_phase
            
            # Update alms with new phases but preserve amplitudes
            amp1 = np.abs(alm[idx1])
            amp2 = np.abs(alm[idx2])
            
            alm[idx1] = amp1 * np.exp(1j * new_phase1)
            alm[idx2] = amp2 * np.exp(1j * new_phase2)
    
    # Convert to map
    map_data = hp.alm2map(alm, nside)
    
    return map_data, alm

def save_test_maps(output_dir="../../data/test_maps", nside=128, lmax=256):
    """
    Generate and save test maps with different levels of golden ratio correlations.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save maps
    nside : int
        HEALPix nside parameter
    lmax : int
        Maximum multipole
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate maps with different correlation strengths
    biases = [0.0, 0.1, 0.3]
    filenames = []
    
    for bias in biases:
        label = f"bias_{bias:.1f}"
        map_data, _ = generate_test_map(nside, lmax, bias, seed=42)
        
        # Save the map
        filename = os.path.join(output_dir, f"test_map_{label}_{timestamp}.fits")
        hp.write_map(filename, map_data, overwrite=True)
        filenames.append(filename)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        hp.mollview(map_data, title=f"Test Map (Golden Ratio Bias = {bias})")
        plt.savefig(os.path.join(output_dir, f"test_map_{label}_{timestamp}.png"), dpi=300)
        plt.close()
        
        print(f"Saved map with bias {bias} to {filename}")
    
    return filenames

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test HEALPix maps with golden ratio correlations")
    parser.add_argument("--nside", type=int, default=128, help="HEALPix nside parameter")
    parser.add_argument("--lmax", type=int, default=256, help="Maximum multipole")
    parser.add_argument("--output-dir", type=str, default="../../data/test_maps",
                       help="Directory to save test maps")
    
    args = parser.parse_args()
    
    filenames = save_test_maps(args.output_dir, args.nside, args.lmax)
    
    # Print command to run the analysis with these maps
    bias0 = filenames[0]  # No bias
    bias3 = filenames[2]  # 0.3 bias
    
    print("\nTo run the analysis with these maps, use these commands:")
    print(f"python run_healpix_quantum_analysis.py --wmap-file \"{bias0}\" --planck-file \"{bias3}\" --n-surrogates 10 --n-samples 1000")
    print("\nOr for faster testing:")
    print(f"python run_healpix_quantum_analysis.py --wmap-file \"{bias0}\" --planck-file \"{bias3}\" --debug")
