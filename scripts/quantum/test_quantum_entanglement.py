#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Quantum Entanglement Signature Test.
Generates simulated CMB data and runs the test.
"""

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from quantum_entanglement_test import quantum_entanglement_signature_test, visualize_results

def create_simulated_cmb(nside=512, lmax=1000, golden_ratio_bias=0.1, seed=42):
    """
    Create a simulated CMB map with optional bias towards golden ratio correlations.
    
    Parameters:
    ----------
    nside : int
        The HEALPix Nside parameter.
    lmax : int
        Maximum multipole to generate.
    golden_ratio_bias : float
        Strength of correlations between golden ratio related multipoles (0-1).
    seed : int
        Random seed for reproducibility.
        
    Returns:
    --------
    hp_map : ndarray
        HEALPix map of the simulated CMB.
    alm : complex ndarray
        Spherical harmonic coefficients of the simulated CMB.
    """
    print(f"Generating simulated CMB map with Nside={nside}, lmax={lmax}")
    np.random.seed(seed)
    
    # Create CAMB-like CMB power spectrum
    ell = np.arange(2, lmax+1)
    cls = np.zeros(lmax+1)
    
    # Simple model for CMB power spectrum
    cls[2:] = 1000 * (ell/(200))**(-0.9) * np.exp(-(ell/1000)**2)
    
    # Generate random alms from power spectrum
    alm = hp.synalm(cls, lmax=lmax, new=True)
    
    # Add golden ratio correlations if requested
    if golden_ratio_bias > 0:
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Find golden ratio related multipole pairs
        gr_pairs = []
        for l1 in range(2, lmax+1):
            l2 = int(round(l1 * phi))
            if l2 <= lmax:
                gr_pairs.append((l1, l2))
        
        # Add correlations between GR pairs
        for l1, l2 in gr_pairs:
            # Get indices for these multipoles
            idx1 = hp.Alm.getidx(lmax, l1, np.arange(0, l1+1))
            idx2 = hp.Alm.getidx(lmax, l2, np.arange(0, l2+1))
            
            # Limit to the available number of m-modes
            min_modes = min(len(idx1), len(idx2))
            if min_modes < 2:
                continue
                
            idx1 = idx1[:min_modes]
            idx2 = idx2[:min_modes]
            
            # Apply correlation
            common_phase = np.random.uniform(0, 2*np.pi, min_modes)
            phase1 = np.angle(alm[idx1])
            phase2 = np.angle(alm[idx2])
            
            # Create blend of original and common phase
            new_phase1 = (1-golden_ratio_bias) * phase1 + golden_ratio_bias * common_phase
            new_phase2 = (1-golden_ratio_bias) * phase2 + golden_ratio_bias * common_phase
            
            # Update alms with new phases
            amp1 = np.abs(alm[idx1])
            amp2 = np.abs(alm[idx2])
            alm[idx1] = amp1 * np.exp(1j * new_phase1)
            alm[idx2] = amp2 * np.exp(1j * new_phase2)
    
    # Convert to map
    hp_map = hp.alm2map(alm, nside)
    
    return hp_map, alm

def run_test_with_varying_bias():
    """Run tests with different levels of golden ratio bias to demonstrate sensitivity"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "../../results/quantum_entanglement_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Settings
    n_simulations = 100  # Low for quick testing
    biases = [0.0, 0.05, 0.1, 0.3]
    nside = 128  # Lower resolution for speed
    lmax = 250
    
    # Run tests with different bias levels
    results_collection = []
    
    for bias in biases:
        print(f"\n\nGenerating and testing data with Golden Ratio bias = {bias}")
        hp_map, alm = create_simulated_cmb(nside=nside, lmax=lmax, 
                                          golden_ratio_bias=bias, seed=42)
        
        # Save map for reference
        map_file = os.path.join(output_dir, f"simulated_cmb_bias_{bias:.2f}.fits")
        hp.write_map(map_file, hp_map, overwrite=True)
        print(f"Saved simulated map to {map_file}")
        
        # Run quantum entanglement test
        results = quantum_entanglement_signature_test(hp_map, n_simulations=n_simulations)
        results['bias'] = bias
        results_collection.append(results)
        
        # Save visualization
        fig = visualize_results(results)
        fig_file = os.path.join(output_dir, f"quantum_test_bias_{bias:.2f}.png")
        fig.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization to {fig_file}")
        
        # Save text results
        with open(os.path.join(output_dir, f"quantum_test_bias_{bias:.2f}.txt"), 'w') as f:
            f.write(f"QUANTUM ENTANGLEMENT TEST RESULTS - BIAS {bias:.2f}\n")
            f.write("="*50 + "\n\n")
            
            f.write("Bell Inequality Analysis:\n")
            f.write(f"Average Bell Value: {results['avg_bell_value']:.4f}\n")
            f.write(f"Surrogate Average Bell Value: {results['surrogate_mean']:.4f}\n")
            f.write(f"Z-score: {results['z_score']:.4f}\n")
            f.write(f"P-value: {results['p_value']:.8f}\n\n")
            
            f.write("Phi-optimality: {:.4f}\n".format(results['phi_optimality']))
            
            # Overall interpretation
            if results['p_value'] < 0.01 and results['phi_optimality'] > 0.5:
                interpretation = "STRONG EVIDENCE for quantum-like entanglement"
            elif results['p_value'] < 0.05 and results['phi_optimality'] > 0.2:
                interpretation = "MODERATE EVIDENCE for quantum-like entanglement"
            elif results['p_value'] < 0.1:
                interpretation = "WEAK EVIDENCE for quantum-like entanglement"
            else:
                interpretation = "NO SIGNIFICANT EVIDENCE for quantum-like entanglement"
                
            f.write(f"\nOverall Interpretation: {interpretation}\n")
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    x = biases
    y_bell = [res['avg_bell_value'] for res in results_collection]
    y_zscore = [res['z_score'] for res in results_collection]
    y_phi = [res['phi_optimality'] for res in results_collection]
    
    plt.plot(x, y_bell, 'o-', label='Bell Value')
    plt.plot(x, y_zscore, 's-', label='Z-Score')
    plt.plot(x, y_phi, '^-', label='Phi-Optimality')
    plt.axhline(y=2, color='r', linestyle='--', label='Classical Limit')
    
    plt.xlabel('Golden Ratio Bias')
    plt.ylabel('Value')
    plt.title('Quantum Entanglement Metrics vs. Golden Ratio Bias')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    comparison_file = os.path.join(output_dir, "bias_comparison.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {comparison_file}")
    
    return results_collection

if __name__ == "__main__":
    print("Running Quantum Entanglement Test with simulated data...")
    results = run_test_with_varying_bias()
    print("\nTest completed successfully.")
