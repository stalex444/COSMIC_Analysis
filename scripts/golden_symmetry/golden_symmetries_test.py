#!/usr/bin/env python3
"""
Golden Symmetries Test Module.

This test analyzes symmetries in the CMB data related to the golden ratio
and compares with other mathematical constants.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys
import argparse
import time
from datetime import datetime

# Add parent directory to path to access utilities
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

def load_wmap_power_spectrum(file_path):
    """
    Load WMAP CMB power spectrum data.
    
    Args:
        file_path (str): Path to WMAP power spectrum data file
        
    Returns:
        tuple: (ell, power, error) arrays of multipole moments, power values, and errors
    """
    data = np.loadtxt(file_path)
    ell = data[:, 0]
    power = data[:, 1]
    error = data[:, 2]
    return ell, power, error

def load_planck_power_spectrum(file_path):
    """
    Load Planck CMB power spectrum data.
    
    Args:
        file_path (str): Path to Planck power spectrum data file
        
    Returns:
        tuple: (ell, power, error) arrays of multipole moments, power values, and errors
    """
    data = np.loadtxt(file_path)
    ell = data[:, 0]
    power = data[:, 1]
    error = data[:, 2]
    return ell, power, error

def ensure_dir_exists(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_phi_optimality(symmetry_ratio, baseline=1.0):
    """
    Calculate phi optimality measure.
    
    Args:
        symmetry_ratio (float): Ratio between observed and expected values
        baseline (float): Baseline value to compare against (default: 1.0)
        
    Returns:
        float: Phi optimality score between -1 and 1
    """
    if symmetry_ratio <= 0:
        return 0
    
    # Calculate raw optimality - how much better is the observed ratio compared to baseline
    raw_optimality = (symmetry_ratio - baseline) / baseline
    
    # Scale to range between -1 and 1 using a sigmoid-like function
    if raw_optimality > 0:
        phi_optimality = min(1.0, raw_optimality / 3.0)  # Positive values scaled by 1/3
    else:
        phi_optimality = max(-1.0, raw_optimality)  # Negative values capped at -1
    
    return phi_optimality

def interpret_phi_optimality(phi_optimality):
    """
    Interpret phi optimality score into text description.
    
    Args:
        phi_optimality (float): Phi optimality score between -1 and 1
        
    Returns:
        str: Text interpretation of phi optimality
    """
    if phi_optimality >= 0.75:
        return "strong"
    elif phi_optimality >= 0.5:
        return "moderate"
    elif phi_optimality >= 0.25:
        return "weak"
    elif phi_optimality >= 0:
        return "minimal"
    elif phi_optimality >= -0.25:
        return "minimal negative"
    elif phi_optimality >= -0.5:
        return "weak negative"
    elif phi_optimality >= -0.75:
        return "moderate negative"
    else:
        return "strong negative"

def run_golden_symmetries_test(ell, power, output_dir, name, n_simulations=1000):
    """
    Run the golden symmetries test.
    
    Parameters:
        ell (np.ndarray): Array of multipole moments
        power (np.ndarray): Array of power spectrum values
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        n_simulations (int): Number of Monte Carlo simulations
        
    Returns:
        dict: Test results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\n=== GOLDEN SYMMETRIES TEST ON {name} DATA ===")
    start_time = time.time()
    
    # Calculate golden ratio symmetries
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    phi_fold = []
    
    for i in range(len(ell)):
        l_value = ell[i]
        if l_value < 2 or i >= len(power):  # Skip extremely low multipoles
            continue
            
        power_value = power[i]
        if power_value <= 0:  # Skip negative or zero power values
            continue
        
        # Find closest ell values to l*phi and l/phi
        l_phi = l_value * phi
        idx_phi = np.abs(ell - l_phi).argmin()
        
        l_inv_phi = l_value / phi
        idx_inv_phi = np.abs(ell - l_inv_phi).argmin()
        
        if idx_phi < len(ell) and idx_inv_phi < len(ell):
            power_phi = power[idx_phi]
            power_inv_phi = power[idx_inv_phi]
            
            # Check for negative values before taking square root
            if power_phi <= 0 or power_inv_phi <= 0:
                continue
                
            # Calculate expected power based on geometric mean
            expected_power = np.sqrt(power_phi * power_inv_phi)
            symmetry_ratio = power_value / expected_power if expected_power != 0 else 1
            
            phi_fold.append(abs(1 - symmetry_ratio))
    
    # Calculate mean asymmetry for golden ratio
    mean_asymmetry = np.mean(phi_fold) if phi_fold else 1.0
    
    # Calculate alternative constants symmetries
    alternative_constants = [np.e, np.pi, np.sqrt(2)]
    alt_asymmetries = []
    
    for constant in alternative_constants:
        alt_fold = []
        for i in range(len(ell)):
            l_value = ell[i]
            if l_value < 2 or i >= len(power):  # Skip extremely low multipoles
                continue
                
            power_value = power[i]
            if power_value <= 0:  # Skip negative or zero power values
                continue
            
            # Find closest ell values to l*constant and l/constant
            l_const = l_value * constant
            idx_const = np.abs(ell - l_const).argmin()
            
            l_inv_const = l_value / constant
            idx_inv_const = np.abs(ell - l_inv_const).argmin()
            
            if idx_const < len(ell) and idx_inv_const < len(ell):
                power_const = power[idx_const]
                power_inv_const = power[idx_inv_const]
                
                # Check for negative values before taking square root
                if power_const <= 0 or power_inv_const <= 0:
                    continue
                
                # Calculate expected power based on geometric mean
                expected_power = np.sqrt(power_const * power_inv_const)
                symmetry_ratio = power_value / expected_power if expected_power != 0 else 1
                
                alt_fold.append(abs(1 - symmetry_ratio))
        
        alt_asymmetries.append(np.mean(alt_fold) if alt_fold else 1.0)
    
    # Calculate mean asymmetry for alternative constants
    mean_alternative = np.mean(alt_asymmetries) if alt_asymmetries else 1.0
    
    # Calculate z-score and p-value
    # Bootstrap method for p-value
    count_better = 0
    for _ in range(n_simulations):
        # Shuffle the power values
        shuffled_power = np.random.permutation(power)
        shuffled_asymmetry = 0
        n_points = 0
        
        for i in range(len(ell)):
            l_value = ell[i]
            if l_value < 2 or i >= len(shuffled_power):  # Skip extremely low multipoles
                continue
                
            power_value = shuffled_power[i]
            if power_value <= 0:  # Skip negative or zero power values
                continue
            
            # Find closest ell values to l*phi and l/phi
            l_phi = l_value * phi
            idx_phi = np.abs(ell - l_phi).argmin()
            
            l_inv_phi = l_value / phi
            idx_inv_phi = np.abs(ell - l_inv_phi).argmin()
            
            if idx_phi < len(ell) and idx_inv_phi < len(shuffled_power):
                power_phi = shuffled_power[idx_phi]
                power_inv_phi = shuffled_power[idx_inv_phi]
                
                # Check for negative values before taking square root
                if power_phi <= 0 or power_inv_phi <= 0:
                    continue
                
                # Calculate expected power based on geometric mean
                expected_power = np.sqrt(power_phi * power_inv_phi)
                symmetry_ratio = power_value / expected_power if expected_power != 0 else 1
                
                shuffled_asymmetry += abs(1 - symmetry_ratio)
                n_points += 1
        
        if n_points > 0:
            shuffled_asymmetry /= n_points
            if shuffled_asymmetry <= mean_asymmetry:
                count_better += 1
    
    p_value = (count_better + 1) / (n_simulations + 1)
    z_score = stats.norm.ppf(1 - p_value)
    
    # Calculate symmetry ratio (how much better is golden ratio compared to alternatives)
    symmetry_ratio = mean_alternative / mean_asymmetry if mean_asymmetry > 0 else 1.0
    
    # Calculate phi optimality
    phi_optimality = calculate_phi_optimality(symmetry_ratio, 1.0)
    
    # Interpret phi optimality
    phi_interpretation = interpret_phi_optimality(phi_optimality)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Define constants and their names
    constants = ["Golden Ratio (φ)", "e", "π", "√2"]
    
    # Combine asymmetries
    asymmetries = [mean_asymmetry] + alt_asymmetries
    
    # Plot bar chart
    plt.bar(constants, asymmetries, color=['gold', 'gray', 'gray', 'gray'], alpha=0.7)
    plt.ylabel('Mean Asymmetry (lower is better)')
    plt.title(f'Golden Ratio Symmetry Test - {name} Data')
    
    # Add phi optimality text
    plt.text(0.5, 0.9, f'Phi Optimality = {phi_optimality:.4f} ({phi_interpretation})', 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add p-value and symmetry ratio
    plt.text(0.5, 0.82, f'p-value = {p_value:.6f}', 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.text(0.5, 0.74, f'Symmetry ratio = {symmetry_ratio:.4f}', 
             horizontalalignment='center',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    
    # Save and show the figure
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualization_path = os.path.join(output_dir, f'golden_symmetries_test_{name.lower()}_{timestamp}.png')
    plt.savefig(visualization_path)
    plt.close()
    
    # Save results to file
    results_path = os.path.join(output_dir, f'golden_symmetries_results_{name.lower()}_{timestamp}.txt')
    
    with open(results_path, 'w') as f:
        f.write(f"=== GOLDEN SYMMETRIES TEST RESULTS ({name} DATA) ===\n\n")
        f.write(f"Mean asymmetry for golden ratio = {mean_asymmetry:.6f}\n")
        f.write(f"Mean asymmetry for alternative constants = {mean_alternative:.6f}\n\n")
        f.write(f"Z-score = {z_score:.4f}\n")
        f.write(f"P-value = {p_value:.6f}\n\n")
        f.write(f"Symmetry ratio = {symmetry_ratio:.4f}\n")
        f.write(f"Phi optimality = {phi_optimality:.4f}\n")
        f.write(f"Phi interpretation = {phi_interpretation}\n\n")
        f.write(f"Test completed in {time.time() - start_time:.2f} seconds\n")
        f.write(f"Test run with {n_simulations} simulations\n")
        f.write(f"Visualization saved to: {visualization_path}\n")
    
    print(f"Test completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved to: {results_path}")
    print(f"Visualization saved to: {visualization_path}")
    
    # Return results dictionary
    results = {
        'test_name': 'Golden Symmetries Test',
        'dataset': name,
        'mean_asymmetry': mean_asymmetry,
        'mean_alternative': mean_alternative,
        'z_score': z_score,
        'p_value': p_value,
        'symmetry_ratio': symmetry_ratio,
        'phi_optimality': phi_optimality,
        'phi_interpretation': phi_interpretation,
        'visualization_path': visualization_path,
        'results_path': results_path
    }
    
    return results

def main():
    """Run the Golden Symmetries Test on WMAP and Planck data."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Golden Symmetries Test on CMB data')
    
    # Define base directory for the COSMIC_Analysis project
    cosmic_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser.add_argument('--wmap-data', type=str, 
                        default=os.path.join(cosmic_dir, "data", "wmap", "wmap_tt_spectrum_9yr_v5.txt"),
                        help='Path to WMAP power spectrum data')
    parser.add_argument('--planck-data', type=str, 
                        default=os.path.join(cosmic_dir, "data", "planck", "planck_tt_spectrum_2018.txt"),
                        help='Path to Planck power spectrum data')
    parser.add_argument('--output-dir', type=str, 
                        default=os.path.join(cosmic_dir, "results", "golden_symmetry_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
                        help='Directory to save results')
    parser.add_argument('--n-simulations', type=int, default=10000,
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--wmap-only', action='store_true',
                        help='Run test only on WMAP data')
    parser.add_argument('--planck-only', action='store_true',
                        help='Run test only on Planck data')
    
    args = parser.parse_args()
    
    # Create output directories
    ensure_dir_exists(args.output_dir)
    wmap_output_dir = os.path.join(args.output_dir, 'wmap')
    planck_output_dir = os.path.join(args.output_dir, 'planck')
    ensure_dir_exists(wmap_output_dir)
    ensure_dir_exists(planck_output_dir)
    
    # Run tests
    if not args.planck_only:
        print("Loading WMAP power spectrum...")
        ell, power, error = load_wmap_power_spectrum(args.wmap_data)
        print(f"Loaded WMAP power spectrum with {len(ell)} multipoles")
        
        wmap_results = run_golden_symmetries_test(
            ell=ell,
            power=power,
            output_dir=wmap_output_dir,
            name="WMAP",
            n_simulations=args.n_simulations
        )
    
    if not args.wmap_only:
        print("Loading Planck power spectrum...")
        ell, power, error = load_planck_power_spectrum(args.planck_data)
        print(f"Loaded Planck power spectrum with {len(ell)} multipoles")
        
        planck_results = run_golden_symmetries_test(
            ell=ell,
            power=power,
            output_dir=planck_output_dir,
            name="Planck",
            n_simulations=args.n_simulations
        )
    
    # Print summary
    print("\n=== GOLDEN SYMMETRIES TEST SUMMARY ===")
    
    if not args.planck_only:
        print(f"\nWMAP Results:")
        print(f"  Phi optimality: {wmap_results['phi_optimality']:.4f} ({wmap_results['phi_interpretation']})")
        print(f"  P-value: {wmap_results['p_value']:.6f}")
        print(f"  Symmetry ratio: {wmap_results['symmetry_ratio']:.4f}")
    
    if not args.wmap_only:
        print(f"\nPlanck Results:")
        print(f"  Phi optimality: {planck_results['phi_optimality']:.4f} ({planck_results['phi_interpretation']})")
        print(f"  P-value: {planck_results['p_value']:.6f}")
        print(f"  Symmetry ratio: {planck_results['symmetry_ratio']:.4f}")
    
    print(f"\nOutput directories:")
    print(f"  WMAP results: {wmap_output_dir}")
    print(f"  Planck results: {planck_output_dir}")

if __name__ == "__main__":
    main()
