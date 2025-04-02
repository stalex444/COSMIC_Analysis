"""
CMB Map Simulator for Testing Purposes
This script generates a simulated CMB map that can be used to test 
retrocausality analysis functions.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os

def simulate_cmb_map(nside=256, lmax=500, output_file=None):
    """
    Generate a simulated CMB map with a typical power spectrum.
    
    Parameters:
    -----------
    nside : int
        HEALPix nside parameter defining resolution (must be power of 2)
    lmax : int
        Maximum multipole moment to generate
    output_file : str, optional
        Path to save the map as a FITS file
    
    Returns:
    --------
    cmb_map : array
        Simulated CMB map
    """
    print(f"Simulating CMB map with nside={nside}, lmax={lmax}")
    
    # Define a typical CMB power spectrum (simplified)
    ell = np.arange(lmax + 1)
    # Avoid division by zero at ell=0
    ell[0] = 1
    
    # Simple model of CMB temperature power spectrum
    # Based on a characteristic shape of the acoustic peaks
    amplitude = 1.0e-10
    peak_ell = 220  # First acoustic peak
    width = 200
    
    # Create a power spectrum with a main peak
    cl = amplitude * (ell/peak_ell)**2 * np.exp(-(ell-peak_ell)**2/width**2)
    cl[0:2] = 0  # Set monopole and dipole to zero
    
    # Print power spectrum info
    print(f"Power spectrum shape: {cl.shape}")
    print(f"Power spectrum max value: {np.max(cl)}")
    
    # Generate random coefficients with this power spectrum
    np.random.seed(42)  # For reproducibility
    alm = hp.synalm(cl, lmax=lmax)
    
    # Convert to map
    cmb_map = hp.alm2map(alm, nside=nside)
    
    # Add some Gaussian noise
    noise_level = 0.01 * np.std(cmb_map)
    noise = np.random.normal(0, noise_level, cmb_map.shape)
    cmb_map += noise
    
    print(f"Generated map with {len(cmb_map)} pixels")
    print(f"Map statistics: mean={np.mean(cmb_map):.2e}, std={np.std(cmb_map):.2e}")
    
    # Save to file if requested
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        hp.write_map(output_file, cmb_map, overwrite=True)
        print(f"Map saved to {output_file}")
    
    return cmb_map

def visualize_cmb_map(cmb_map, title="Simulated CMB Map", output_file=None):
    """Plot a visualization of the CMB map"""
    plt.figure(figsize=(10, 8))
    hp.mollview(cmb_map, title=title)
    hp.graticule()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate simulated CMB map for testing")
    parser.add_argument("--nside", type=int, default=128, 
                        help="HEALPix Nside parameter (power of 2)")
    parser.add_argument("--lmax", type=int, default=500,
                        help="Maximum multipole moment")
    parser.add_argument("--output", default="simulated_cmb_map.fits",
                        help="Output FITS file path")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualization of the map")
    
    args = parser.parse_args()
    
    # Generate the map
    cmb_map = simulate_cmb_map(nside=args.nside, lmax=args.lmax, output_file=args.output)
    
    # Visualize if requested
    if args.visualize:
        viz_file = os.path.splitext(args.output)[0] + ".png"
        visualize_cmb_map(cmb_map, output_file=viz_file)
    
    print("Simulation completed successfully")
