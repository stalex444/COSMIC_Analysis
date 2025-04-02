#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quantum Entanglement Signature Test for CMB Data

This script implements a comprehensive test for quantum entanglement-like signatures
in the Cosmic Microwave Background (CMB) by analyzing correlations between
causally disconnected regions related by the golden ratio.

The test evaluates whether CMB data exhibits Bell inequality violations and quantum-like
non-local correlations that would be unexpected in purely classical systems.
"""

import numpy as np
import healpy as hp
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from astropy.io import fits
import os
import sys

def quantum_entanglement_signature_test(cmb_data, n_simulations=1000):
    """
    Test for quantum entanglement signatures in the CMB by analyzing correlations
    between causally disconnected regions related by the golden ratio.
    
    Parameters:
    -----------
    cmb_data : array-like
        The cosmic microwave background data (HEALPix map)
    n_simulations : int
        Number of Monte Carlo simulations for statistical validation
    
    Returns:
    --------
    dict
        Results including Bell inequality violations, non-locality metrics,
        and statistical significance
    """
    print("Starting Quantum Entanglement Signature Test...")
    start_time = time.time()
    
    # Constants
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Get HEALPix parameters
    nside = hp.get_nside(cmb_data)
    npix = hp.nside2npix(nside)
    print(f"Analyzing HEALPix map with nside={nside}, containing {npix} pixels")
    
    # Identify causally disconnected regions at the time of CMB formation
    # These are regions separated by angles larger than the horizon size at recombination
    # Approximately 2 degrees or ~0.035 radians
    horizon_angle = 2 * np.pi / 180
    
    # Step 1: Find pairs of pixels that are:
    #   1. Causally disconnected (separated by > horizon_angle)
    #   2. Related by golden ratio in their harmonic indices
    print("Identifying causally disconnected regions with golden ratio relationships...")
    
    # Get spherical harmonic coefficients (alm)
    lmax = 3 * nside - 1
    alm = hp.map2alm(cmb_data, lmax=lmax)
    
    # Reconstruct map from alm coefficients for verification
    reconstructed_map = hp.alm2map(alm, nside)
    
    # Calculate correlation between original and reconstructed maps
    # to verify the harmonic transformation is accurate
    corr = np.corrcoef(cmb_data, reconstructed_map)[0, 1]
    print(f"Verification correlation between original and reconstructed maps: {corr:.6f}")
    
    # Step 2: Identify golden ratio related multipole pairs
    print("Identifying golden ratio related multipole pairs...")
    gr_pairs = []
    
    # Test multipoles from 2 to lmax (skip monopole and dipole)
    for l1 in range(2, lmax // 2):
        # Find l2 value closest to l1 * PHI
        l2_ideal = l1 * PHI
        l2 = int(round(l2_ideal))
        
        # Only include if l2 is within range and close to ideal phi relationship
        if l2 <= lmax and abs(l2/l1 - PHI) < 0.1:
            gr_pairs.append((l1, l2))
    
    print(f"Found {len(gr_pairs)} multipole pairs with golden ratio relationships")
    
    # Step 3: Calculate Bell inequality violations for these pairs
    print("Calculating Bell inequality violations...")
    
    bell_values = []
    for l1, l2 in gr_pairs:
        # Extract spherical harmonic coefficients for these multipoles
        idx1 = hp.Alm.getidx(lmax, l1, np.arange(0, l1+1))
        idx2 = hp.Alm.getidx(lmax, l2, np.arange(0, l2+1))
        
        # Use coefficients as quantum states
        a1 = np.array([alm[i].real for i in idx1])
        a2 = np.array([alm[i].imag for i in idx1])
        b1 = np.array([alm[i].real for i in idx2])
        b2 = np.array([alm[i].imag for i in idx2])
        
        # Normalize
        a1 = a1 / np.sqrt(np.sum(a1**2)) if np.sum(a1**2) > 0 else a1
        a2 = a2 / np.sqrt(np.sum(a2**2)) if np.sum(a2**2) > 0 else a2
        b1 = b1 / np.sqrt(np.sum(b1**2)) if np.sum(b1**2) > 0 else b1
        b2 = b2 / np.sqrt(np.sum(b2**2)) if np.sum(b2**2) > 0 else b2
        
        # Calculate CHSH Bell parameter
        # S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)
        # Classical limit: |S| ≤ 2
        # Quantum limit: |S| ≤ 2√2 ≈ 2.83
        
        # Calculate correlation functions
        E_a1b1 = correlation_function(a1, b1)
        E_a1b2 = correlation_function(a1, b2)
        E_a2b1 = correlation_function(a2, b1)
        E_a2b2 = correlation_function(a2, b2)
        
        # CHSH value
        S = E_a1b1 - E_a1b2 + E_a2b1 + E_a2b2
        bell_values.append(abs(S))
    
    # Calculate average Bell violation
    avg_bell_value = np.mean(bell_values) if bell_values else 0
    max_bell_value = np.max(bell_values) if bell_values else 0
    
    # Count violations of classical limit (|S| > 2)
    classical_violations = sum(1 for v in bell_values if v > 2)
    classical_violation_rate = classical_violations / len(bell_values) if bell_values else 0
    
    # Count violations of quantum limit (|S| > 2√2)
    quantum_limit = 2 * np.sqrt(2)
    quantum_violations = sum(1 for v in bell_values if v > quantum_limit)
    quantum_violation_rate = quantum_violations / len(bell_values) if bell_values else 0
    
    print(f"Average Bell value: {avg_bell_value:.4f}")
    print(f"Maximum Bell value: {max_bell_value:.4f}")
    print(f"Classical limit violations: {classical_violations}/{len(bell_values)} ({classical_violation_rate:.2%})")
    print(f"Quantum limit violations: {quantum_violations}/{len(bell_values)} ({quantum_violation_rate:.2%})")
    
    # Step 4: Generate and analyze surrogate data
    print(f"Generating {n_simulations} surrogate datasets for comparison...")
    
    surrogate_bell_values = []
    surrogate_violation_rates = []
    
    for i in range(n_simulations):
        if i % 100 == 0 and i > 0:
            print(f"Completed {i}/{n_simulations} simulations")
        
        # Generate surrogate with same power spectrum but randomized phases
        rand_alm = randomize_phases(alm)
        
        # Calculate Bell values for surrogate
        surr_bell_values = []
        
        for l1, l2 in gr_pairs:
            # Extract spherical harmonic coefficients for these multipoles
            idx1 = hp.Alm.getidx(lmax, l1, np.arange(0, l1+1))
            idx2 = hp.Alm.getidx(lmax, l2, np.arange(0, l2+1))
            
            # Use coefficients as quantum states
            a1 = np.array([rand_alm[i].real for i in idx1])
            a2 = np.array([rand_alm[i].imag for i in idx1])
            b1 = np.array([rand_alm[i].real for i in idx2])
            b2 = np.array([rand_alm[i].imag for i in idx2])
            
            # Normalize
            a1 = a1 / np.sqrt(np.sum(a1**2)) if np.sum(a1**2) > 0 else a1
            a2 = a2 / np.sqrt(np.sum(a2**2)) if np.sum(a2**2) > 0 else a2
            b1 = b1 / np.sqrt(np.sum(b1**2)) if np.sum(b1**2) > 0 else b1
            b2 = b2 / np.sqrt(np.sum(b2**2)) if np.sum(b2**2) > 0 else b2
            
            # Calculate CHSH Bell parameter
            E_a1b1 = correlation_function(a1, b1)
            E_a1b2 = correlation_function(a1, b2)
            E_a2b1 = correlation_function(a2, b1)
            E_a2b2 = correlation_function(a2, b2)
            
            S = E_a1b1 - E_a1b2 + E_a2b1 + E_a2b2
            surr_bell_values.append(abs(S))
        
        # Calculate average Bell value for this surrogate
        surr_avg_bell = np.mean(surr_bell_values) if surr_bell_values else 0
        surrogate_bell_values.append(surr_avg_bell)
        
        # Calculate violation rate for this surrogate
        surr_violations = sum(1 for v in surr_bell_values if v > 2)
        surr_rate = surr_violations / len(surr_bell_values) if surr_bell_values else 0
        surrogate_violation_rates.append(surr_rate)
    
    # Calculate statistics
    surrogate_mean = np.mean(surrogate_bell_values)
    surrogate_std = np.std(surrogate_bell_values)
    
    # Z-score and p-value
    if surrogate_std > 0:
        z_score = (avg_bell_value - surrogate_mean) / surrogate_std
        p_value = 1 - stats.norm.cdf(z_score)
    else:
        z_score = float('inf') if avg_bell_value > surrogate_mean else float('-inf')
        p_value = 0 if avg_bell_value > surrogate_mean else 1
        
    # Calculate statistical significance for violation rates
    violation_mean = np.mean(surrogate_violation_rates)
    violation_std = np.std(surrogate_violation_rates)
    
    if violation_std > 0:
        violation_z = (classical_violation_rate - violation_mean) / violation_std
        violation_p = 1 - stats.norm.cdf(violation_z)
    else:
        violation_z = float('inf') if classical_violation_rate > violation_mean else float('-inf')
        violation_p = 0 if classical_violation_rate > violation_mean else 1
    
    # Step 5: Test for non-locality beyond Bell inequalities
    print("Testing for additional non-locality signatures...")
    
    # Calculate mutual information between golden ratio related multipoles
    mutual_info = calculate_mutual_information(alm, gr_pairs, lmax)
    
    # Generate mutual information for surrogate data
    surrogate_mi = []
    for i in range(min(100, n_simulations)):  # Use subset for computational efficiency
        rand_alm = randomize_phases(alm)
        mi = calculate_mutual_information(rand_alm, gr_pairs, lmax)
        surrogate_mi.append(mi)
    
    surrogate_mi_mean = np.mean(surrogate_mi)
    surrogate_mi_std = np.std(surrogate_mi)
    
    if surrogate_mi_std > 0:
        mi_z_score = (mutual_info - surrogate_mi_mean) / surrogate_mi_std
        mi_p_value = 1 - stats.norm.cdf(mi_z_score)
    else:
        mi_z_score = float('inf') if mutual_info > surrogate_mi_mean else float('-inf')
        mi_p_value = 0 if mutual_info > surrogate_mi_mean else 1
    
    # Step 6: Test phi-optimality
    print("Testing for phi-optimality...")
    
    # Compare with other constants
    constants = {
        "phi": PHI,          # Golden ratio
        "e": np.e,           # Euler's number
        "pi": np.pi,         # Pi
        "sqrt2": np.sqrt(2), # Square root of 2
        "sqrt3": np.sqrt(3), # Square root of 3
        "ln2": np.log(2)     # Natural logarithm of 2
    }
    
    # Test Bell violations for pairs related by other constants
    constant_violations = {}
    
    for name, value in constants.items():
        if name == "phi":
            constant_violations[name] = avg_bell_value
            continue
            
        # Find pairs related by this constant
        const_pairs = []
        for l1 in range(2, lmax // 2):
            l2_ideal = l1 * value
            l2 = int(round(l2_ideal))
            
            if l2 <= lmax and abs(l2/l1 - value) < 0.1:
                const_pairs.append((l1, l2))
        
        if not const_pairs:
            constant_violations[name] = 0
            continue
            
        # Calculate Bell values for these pairs
        const_bell_values = []
        
        for l1, l2 in const_pairs:
            idx1 = hp.Alm.getidx(lmax, l1, np.arange(0, l1+1))
            idx2 = hp.Alm.getidx(lmax, l2, np.arange(0, l2+1))
            
            a1 = np.array([alm[i].real for i in idx1])
            a2 = np.array([alm[i].imag for i in idx1])
            b1 = np.array([alm[i].real for i in idx2])
            b2 = np.array([alm[i].imag for i in idx2])
            
            # Normalize
            a1 = a1 / np.sqrt(np.sum(a1**2)) if np.sum(a1**2) > 0 else a1
            a2 = a2 / np.sqrt(np.sum(a2**2)) if np.sum(a2**2) > 0 else a2
            b1 = b1 / np.sqrt(np.sum(b1**2)) if np.sum(b1**2) > 0 else b1
            b2 = b2 / np.sqrt(np.sum(b2**2)) if np.sum(b2**2) > 0 else b2
            
            E_a1b1 = correlation_function(a1, b1)
            E_a1b2 = correlation_function(a1, b2)
            E_a2b1 = correlation_function(a2, b1)
            E_a2b2 = correlation_function(a2, b2)
            
            S = E_a1b1 - E_a1b2 + E_a2b1 + E_a2b2
            const_bell_values.append(abs(S))
        
        constant_violations[name] = np.mean(const_bell_values) if const_bell_values else 0
    
    # Calculate phi-optimality
    max_other = max([v for k, v in constant_violations.items() if k != "phi"])
    phi_optimality = (constant_violations["phi"] - surrogate_mean) / (max_other - surrogate_mean) if max_other > surrogate_mean else 1.0
    
    # Log all results
    print("\n" + "="*50)
    print("QUANTUM ENTANGLEMENT SIGNATURE TEST RESULTS")
    print("="*50)
    
    print(f"\nBell Inequality Analysis:")
    print(f"CMB Average Bell Value: {avg_bell_value:.4f}")
    print(f"Surrogate Average Bell Value: {surrogate_mean:.4f}")
    print(f"Ratio: {avg_bell_value/surrogate_mean:.2f}x")
    print(f"Z-score: {z_score:.4f}")
    print(f"P-value: {p_value:.8f}")
    
    print(f"\nClassical Limit Violations:")
    print(f"CMB Violation Rate: {classical_violation_rate:.2%}")
    print(f"Surrogate Violation Rate: {violation_mean:.2%}")
    print(f"Z-score: {violation_z:.4f}")
    print(f"P-value: {violation_p:.8f}")
    
    print(f"\nNon-Locality Metrics:")
    print(f"Mutual Information: {mutual_info:.4f}")
    print(f"Surrogate MI: {surrogate_mi_mean:.4f}")
    print(f"Z-score: {mi_z_score:.4f}")
    print(f"P-value: {mi_p_value:.8f}")
    
    print(f"\nConstant Comparison (Bell Values):")
    for name, value in constant_violations.items():
        print(f"{name}: {value:.4f}")
    
    print(f"\nPhi-optimality: {phi_optimality:.4f}")
    
    execution_time = time.time() - start_time
    print(f"\nTest completed in {execution_time:.2f} seconds")
    
    # Return comprehensive results
    results = {
        "bell_values": bell_values,
        "avg_bell_value": avg_bell_value,
        "max_bell_value": max_bell_value,
        "classical_violations": classical_violations,
        "classical_violation_rate": classical_violation_rate,
        "quantum_violations": quantum_violations,
        "quantum_violation_rate": quantum_violation_rate,
        "surrogate_bell_values": surrogate_bell_values,
        "surrogate_mean": surrogate_mean,
        "surrogate_std": surrogate_std,
        "z_score": z_score,
        "p_value": p_value,
        "violation_z": violation_z,
        "violation_p": violation_p,
        "mutual_info": mutual_info,
        "surrogate_mi_mean": surrogate_mi_mean,
        "surrogate_mi_std": surrogate_mi_std,
        "mi_z_score": mi_z_score,
        "mi_p_value": mi_p_value,
        "constant_violations": constant_violations,
        "phi_optimality": phi_optimality,
        "gr_pairs": gr_pairs,
        "execution_time": execution_time
    }
    
    return results

def correlation_function(a, b):
    """Calculate quantum correlation function between two sets of values"""
    if len(a) == 0 or len(b) == 0:
        return 0
        
    # Ensure equal lengths by truncating the longer array
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]
    
    # Normalize
    norm_a = np.sqrt(np.sum(a**2))
    norm_b = np.sqrt(np.sum(b**2))
    
    if norm_a == 0 or norm_b == 0:
        return 0
        
    a = a / norm_a
    b = b / norm_b
    
    # Calculate correlation (dot product of normalized vectors)
    correlation = np.sum(a * b)
    
    # Bound to [-1, 1]
    return max(min(correlation, 1.0), -1.0)

def randomize_phases(alm):
    """Generate a surrogate alm with the same power spectrum but randomized phases"""
    lmax = hp.Alm.getlmax(len(alm))
    
    # Create new alm with same amplitudes but random phases
    rand_alm = np.zeros_like(alm)
    
    # Loop through all (l,m) values
    for l in range(lmax + 1):
        for m in range(l + 1):
            idx = hp.Alm.getidx(lmax, l, m)
            
            # Get amplitude
            amp = np.abs(alm[idx])
            
            # Generate random phase
            phase = np.random.uniform(0, 2*np.pi)
            
            # Set new alm with same amplitude but random phase
            rand_alm[idx] = amp * np.exp(1j * phase)
    
    return rand_alm

def calculate_mutual_information(alm, gr_pairs, lmax, bins=20):
    """Calculate mutual information between golden ratio related multipoles"""
    if not gr_pairs:
        return 0
        
    mutual_infos = []
    
    for l1, l2 in gr_pairs:
        # Extract spherical harmonic coefficients for these multipoles
        idx1 = hp.Alm.getidx(lmax, l1, np.arange(0, l1+1))
        idx2 = hp.Alm.getidx(lmax, l2, np.arange(0, l2+1))
        
        # Get real parts (could also use magnitudes or combine real/imag)
        values1 = np.array([alm[i].real for i in idx1])
        values2 = np.array([alm[i].real for i in idx2])
        
        # Ensure equal lengths by truncating
        min_len = min(len(values1), len(values2))
        if min_len < 5:  # Skip if insufficient data
            continue
            
        values1 = values1[:min_len]
        values2 = values2[:min_len]
        
        # Calculate mutual information
        try:
            # Create 2D histogram for joint distribution
            hist_2d, x_edges, y_edges = np.histogram2d(values1, values2, bins=bins)
            
            # Create 1D histograms for marginal distributions
            hist_1, _ = np.histogram(values1, bins=x_edges)
            hist_2, _ = np.histogram(values2, bins=y_edges)
            
            # Convert to probabilities
            p_xy = hist_2d / np.sum(hist_2d)
            p_x = hist_1 / np.sum(hist_1)
            p_y = hist_2 / np.sum(hist_2)
            
            # Compute mutual information
            mutual_info = 0
            for i in range(bins):
                for j in range(bins):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mutual_info += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            mutual_infos.append(mutual_info)
        except:
            # Skip if calculation fails
            continue
    
    # Return average mutual information
    return np.mean(mutual_infos) if mutual_infos else 0

def visualize_results(results):
    """Create visualizations of quantum entanglement test results"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Bell value distribution
    ax1 = fig.add_subplot(2, 2, 1)
    if 'bell_values' in results and results['bell_values']:
        ax1.hist(results['bell_values'], bins=20, alpha=0.7, color='blue', density=True)
        ax1.axvline(x=2, color='r', linestyle='--', label='Classical Limit')
        ax1.axvline(x=2*np.sqrt(2), color='g', linestyle='--', label='Quantum Limit')
        ax1.axvline(x=results['avg_bell_value'], color='k', linewidth=2, label='CMB Average')
        ax1.set_xlabel('Bell Value |S|')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Distribution of Bell Values')
        ax1.legend()
    
    # 2. Statistical significance
    ax2 = fig.add_subplot(2, 2, 2)
    if 'surrogate_bell_values' in results and results['surrogate_bell_values']:
        ax2.hist(results['surrogate_bell_values'], bins=20, alpha=0.7, color='gray', density=True)
        ax2.axvline(x=results['avg_bell_value'], color='r', linewidth=2, label='CMB Data')
        ax2.set_xlabel('Average Bell Value')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Statistical Significance (Z-score: {:.2f})'.format(results['z_score']))
        ax2.legend()
        
        # Add statistical information
        stats_text = "Z-score: {:.4f}\nP-value: {:.8f}\n".format(results['z_score'], results['p_value'])
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Comparison across constants
    ax3 = fig.add_subplot(2, 2, 3)
    if 'constant_violations' in results:
        constants = list(results['constant_violations'].keys())
        values = [results['constant_violations'][k] for k in constants]
        
        # Find index of "phi" for highlighting
        phi_idx = constants.index("phi") if "phi" in constants else -1
        
        # Create color array with phi highlighted
        colors = ['gray'] * len(constants)
        if phi_idx >= 0:
            colors[phi_idx] = 'gold'
        
        ax3.bar(constants, values, color=colors)
        ax3.axhline(y=results['surrogate_mean'], color='r', linestyle='--', label='Random')
        ax3.set_xlabel('Mathematical Constant')
        ax3.set_ylabel('Average Bell Value')
        ax3.set_title('Bell Violations by Mathematical Constant')
        ax3.legend()
        
        # Add phi-optimality
        ax3.text(0.05, 0.95, "Phi-optimality: {:.4f}".format(results['phi_optimality']), 
                transform=ax3.transAxes, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Summary gauge
    ax4 = fig.add_subplot(2, 2, 4)
    create_summary_gauge(ax4, results)
    
    plt.tight_layout()
    return fig

def create_summary_gauge(ax, results):
    """Create a gauge visualization summarizing the quantum entanglement findings"""
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Draw gauge background (half-circle)
    theta = np.linspace(-np.pi, 0, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, 'k-', linewidth=2)
    
    # Add colored regions
    theta_classical = np.linspace(-np.pi, -2*np.pi/3, 50)
    x_classical = np.cos(theta_classical)
    y_classical = np.sin(theta_classical)
    ax.fill_between(x_classical, 0, y_classical, color='blue', alpha=0.3)
    
    theta_quantum = np.linspace(-2*np.pi/3, -np.pi/3, 50)
    x_quantum = np.cos(theta_quantum)
    y_quantum = np.sin(theta_quantum)
    ax.fill_between(x_quantum, 0, y_quantum, color='purple', alpha=0.3)
    
    theta_beyond = np.linspace(-np.pi/3, 0, 50)
    x_beyond = np.cos(theta_beyond)
    y_beyond = np.sin(theta_beyond)
    ax.fill_between(x_beyond, 0, y_beyond, color='red', alpha=0.3)
    
    # Add labels
    ax.text(-0.9, -0.3, "Classical", ha='center', va='center')
    ax.text(0, -0.3, "Quantum", ha='center', va='center')
    ax.text(0.9, -0.3, "Beyond\nQuantum", ha='center', va='center')
    
    # Convert z-score to angle for gauge
    # Map z-score range of [0, 10] to angle range of [-pi, 0]
    z_score = min(max(results.get('z_score', 0), 0), 10)  # Clamp to [0, 10]
    angle = -np.pi * (1 - z_score/10)
    
    # Draw gauge needle
    needle_x = 0.8 * np.cos(angle)
    needle_y = 0.8 * np.sin(angle)
    ax.plot([0, needle_x], [0, needle_y], 'k-', linewidth=3)
    ax.plot(0, 0, 'ko', markersize=10)
    
    # Add z-score and p-value
    z_text = "Z-score: {:.2f}".format(results.get('z_score', 0))
    p_text = "P-value: {:.8f}".format(results.get('p_value', 1))
    phi_text = "Phi-optimality: {:.4f}".format(results.get('phi_optimality', 0))
    
    ax.text(0, -0.6, z_text + "\n" + p_text + "\n" + phi_text, 
           ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add interpretation
    if results.get('p_value', 1) < 0.01 and results.get('phi_optimality', 0) > 0.5:
        interpretation = "STRONG EVIDENCE for quantum-like entanglement"
    elif results.get('p_value', 1) < 0.05 and results.get('phi_optimality', 0) > 0.2:
        interpretation = "MODERATE EVIDENCE for quantum-like entanglement"
    elif results.get('p_value', 1) < 0.1:
        interpretation = "WEAK EVIDENCE for quantum-like entanglement"
    else:
        interpretation = "NO SIGNIFICANT EVIDENCE for quantum-like entanglement"
        
    ax.text(0, -0.9, interpretation, ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.set_title('Quantum Entanglement Evidence Gauge')

# Function to load CMB data from a FITS file
def load_cmb_data(filename):
    """Load CMB data from a FITS file"""
    try:
        hdulist = fits.open(filename)
        data = hdulist[1].data['I_STOKES']  # For Planck SMICA map
        hdulist.close()
        return data
    except:
        try:
            data = hp.read_map(filename)
            return data
        except:
            print(f"Could not load {filename}. Make sure it's a valid FITS file or HEALPix map.")
            return None

# Main function to run the test
def main(cmb_file, n_simulations=1000, output_dir=None):
    """Run the quantum entanglement signature test on CMB data"""
    print(f"Loading CMB data from {cmb_file}...")
    cmb_data = load_cmb_data(cmb_file)
    
    if cmb_data is None:
        print("Could not load CMB data. Exiting.")
        return
    
    # Create output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/quantum_entanglement_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running quantum entanglement signature test with {n_simulations} simulations...")
    results = quantum_entanglement_signature_test(cmb_data, n_simulations)
    
    print("Creating visualization...")
    fig = visualize_results(results)
    
    output_file = os.path.join(output_dir, "quantum_entanglement_results.png")
    print(f"Saving visualization to {output_file}...")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Results saved to {output_file}")
    
    # Save detailed results to text file
    results_file = os.path.join(output_dir, "quantum_entanglement_results.txt")
    print(f"Saving detailed results to {results_file}...")
    
    with open(results_file, 'w') as f:
        f.write("QUANTUM ENTANGLEMENT SIGNATURE TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write("Bell Inequality Analysis:\n")
        f.write(f"CMB Average Bell Value: {results['avg_bell_value']:.4f}\n")
        f.write(f"Surrogate Average Bell Value: {results['surrogate_mean']:.4f}\n")
        f.write(f"Ratio: {results['avg_bell_value']/results['surrogate_mean']:.2f}x\n")
        f.write(f"Z-score: {results['z_score']:.4f}\n")
        f.write(f"P-value: {results['p_value']:.8f}\n\n")
        
        f.write("Classical Limit Violations:\n")
        f.write(f"CMB Violation Rate: {results['classical_violation_rate']:.2%}\n")
        f.write(f"Surrogate Violation Rate: {results['violation_mean']:.2%}\n")
        f.write(f"Z-score: {results['violation_z']:.4f}\n")
        f.write(f"P-value: {results['violation_p']:.8f}\n\n")
        
        f.write("Non-Locality Metrics:\n")
        f.write(f"Mutual Information: {results['mutual_info']:.4f}\n")
        f.write(f"Surrogate MI: {results['surrogate_mi_mean']:.4f}\n")
        f.write(f"Z-score: {results['mi_z_score']:.4f}\n")
        f.write(f"P-value: {results['mi_p_value']:.8f}\n\n")
        
        f.write("Constant Comparison (Bell Values):\n")
        for name, value in results['constant_violations'].items():
            f.write(f"{name}: {value:.4f}\n")
        
        f.write(f"\nPhi-optimality: {results['phi_optimality']:.4f}\n\n")
        
        # Add overall interpretation
        if results['p_value'] < 0.01 and results['phi_optimality'] > 0.5:
            interpretation = "STRONG EVIDENCE for quantum-like entanglement"
        elif results['p_value'] < 0.05 and results['phi_optimality'] > 0.2:
            interpretation = "MODERATE EVIDENCE for quantum-like entanglement"
        elif results['p_value'] < 0.1:
            interpretation = "WEAK EVIDENCE for quantum-like entanglement"
        else:
            interpretation = "NO SIGNIFICANT EVIDENCE for quantum-like entanglement"
            
        f.write(f"Overall Interpretation: {interpretation}\n")
        f.write(f"\nTest completed in {results['execution_time']:.2f} seconds\n")
    
    print(f"Detailed results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Quantum Entanglement Signature Test on CMB data")
    parser.add_argument("cmb_file", help="Path to CMB data file (FITS or HEALPix map)")
    parser.add_argument("-n", "--simulations", type=int, default=1000, 
                       help="Number of Monte Carlo simulations (default: 1000)")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                       help="Output directory for results (default: auto-generated)")
    
    args = parser.parse_args()
    main(args.cmb_file, args.simulations, args.output_dir)
