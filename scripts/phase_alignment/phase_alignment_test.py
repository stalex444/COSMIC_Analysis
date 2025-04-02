#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import healpy as hp
from scipy.fft import fft, ifft
from scipy.stats import circmean, circstd
import os
from datetime import datetime
import argparse

# Constants
PHI = 1.618033988749895  # Golden ratio
PI = np.pi

def load_cmb_map(filename):
    """Load CMB map from a file."""
    # Check if the file is a text file with power spectrum data
    if filename.endswith('.txt'):
        # Load power spectrum data from text file
        data = np.loadtxt(filename, comments='#')
        # Extract multipole moments (l) and power spectrum values
        ell = data[:, 0].astype(int)
        cl = data[:, 1]  # Using the TT power spectrum (column 2)
        
        # Create a dictionary mapping multipole moments to power spectrum values
        cl_dict = {l: c for l, c in zip(ell, cl)}
        
        # Return the power spectrum data instead of a map
        return {'ell': ell, 'cl': cl_dict, 'is_spectrum': True}
    else:
        # Try to load as a FITS file using a more basic approach
        try:
            # Use a simpler approach that avoids healpy's rotator module
            from astropy.io import fits
            hdul = fits.open(filename)
            # Assuming the CMB map is in the first extension
            cmb_map = hdul[1].data
            hdul.close()
            return {'map': cmb_map, 'is_spectrum': False, 'simple_map': True}
        except Exception as e:
            # Fall back to healpy if available
            try:
                import healpy as hp
                cmb_map = hp.read_map(filename)
                return {'map': cmb_map, 'is_spectrum': False}
            except Exception as e2:
                raise ValueError("Failed to load file {}: {}. Healpy error: {}".format(
                    filename, str(e), str(e2)))

def extract_spherical_harmonics(cmb_data, lmax=2000):
    """Extract or generate spherical harmonic coefficients."""
    # Check if we're working with a power spectrum or a map
    if cmb_data.get('is_spectrum', False):
        # We have a power spectrum, generate synthetic alm coefficients
        ell = cmb_data['ell']
        cl_dict = cmb_data['cl']
        
        # Create a full cl array up to lmax
        cl = np.zeros(lmax + 1)
        for l in range(min(lmax + 1, max(ell) + 1)):
            if l in cl_dict:
                # Convert from D_l to C_l (if needed)
                # D_l = l(l+1)/(2π) * C_l, so C_l = D_l * 2π/(l(l+1))
                if l > 0:  # Avoid division by zero
                    cl[l] = cl_dict[l] * 2 * np.pi / (l * (l + 1))
                else:
                    cl[l] = 0
        
        # Generate random alm coefficients with the given power spectrum
        np.random.seed(42)  # For reproducibility
        alm = hp.synalm(cl, lmax=lmax)
        
        return alm
    else:
        # We have a map, extract alm coefficients as before
        if cmb_data.get('simple_map', False):
            # Use a simpler approach that avoids healpy's rotator module
            from astropy.io import fits
            from healpy import pix2ang, ang2pix
            nside = int(np.sqrt(len(cmb_data['map']) / 12))
            theta, phi = pix2ang(nside, np.arange(len(cmb_data['map'])))
            alm = hp.map2alm(cmb_data['map'], lmax=lmax)
        else:
            alm = hp.map2alm(cmb_data['map'], lmax=lmax)
        return alm

def get_phase_information(alm, l_ranges):
    """
    Extract phase information for specified multipole ranges.
    
    Parameters:
    - alm: Spherical harmonic coefficients
    - l_ranges: List of (l_min, l_max) tuples defining the scale ranges
    
    Returns:
    - Dictionary mapping scale ranges to phase arrays
    """
    phase_info = {}
    
    # Calculate lmax based on the size of alm
    lmax = hp.Alm.getlmax(len(alm))
    
    for l_min, l_max in l_ranges:
        scale_key = "{}-{}".format(l_min, l_max)
        phases = []
        
        # Extract alm values for this multipole range
        for l in range(l_min, l_max + 1):
            for m in range(-l, l + 1):
                if m >= 0:  # healpy stores only m>=0
                    idx = hp.sphtfunc.Alm.getidx(lmax, l, abs(m))
                    if idx < len(alm):
                        coeff = alm[idx]
                        # Complex phase angle
                        phase = np.angle(coeff)
                        phases.append(phase)
        
        phase_info[scale_key] = np.array(phases)
    
    return phase_info

def compute_phase_coherence(phases1, phases2):
    """
    Compute phase coherence between two sets of phases.
    Returns a value between 0 (no coherence) and 1 (perfect coherence).
    """
    # Calculate phase differences
    phase_diffs = np.abs(phases1[:min(len(phases1), len(phases2))] - 
                         phases2[:min(len(phases1), len(phases2))])
    
    # Normalize to [0, pi]
    phase_diffs = np.mod(phase_diffs + PI, 2*PI) - PI
    
    # Compute coherence (1 = perfect alignment, 0 = random)
    coherence = 1 - np.mean(np.abs(phase_diffs) / PI)
    
    return coherence

def compute_phase_locking_value(phases1, phases2):
    """
    Compute phase-locking value between two sets of phases.
    PLV provides an alternative measure of phase synchronization.
    """
    min_length = min(len(phases1), len(phases2))
    if min_length == 0:
        return 0
        
    phases1 = phases1[:min_length]
    phases2 = phases2[:min_length]
    
    # Calculate complex phase difference
    complex_phase_diff = np.exp(1j * (phases1 - phases2))
    
    # Calculate PLV
    plv = np.abs(np.mean(complex_phase_diff))
    
    return plv

def generate_surrogate_phases(phases, n_surrogates=1000):
    """Generate phase-randomized surrogates while preserving amplitude structure."""
    surrogates = []
    
    for i in range(n_surrogates):
        # Create random phases with same length
        random_phases = np.random.uniform(0, 2*PI, size=len(phases))
        surrogates.append(random_phases)
    
    return surrogates

def calculate_gr_phase_coherence(phase_info, scale_pairs=None, use_plv=False):
    """
    Calculate phase coherence between scales related by golden ratio.
    
    Parameters:
    - phase_info: Dictionary mapping scale ranges to phase arrays
    - scale_pairs: List of scale pairs to check, or None to check all possible pairs
    - use_plv: If True, use Phase Locking Value instead of traditional coherence
    
    Returns:
    - Dictionary with coherence scores for each scale pair
    """
    scales = sorted([tuple(map(int, k.split('-'))) for k in phase_info.keys()])
    coherence_scores = {}
    
    # If no specific pairs provided, check all pairs
    if scale_pairs is None:
        scale_pairs = []
        for i, (min1, max1) in enumerate(scales[:-1]):
            for (min2, max2) in scales[i+1:]:
                # Calculate effective scale ratio (using midpoints)
                scale1 = (min1 + max1) / 2
                scale2 = (min2 + max2) / 2
                ratio = max(scale1, scale2) / min(scale1, scale2)
                
                # Check if close to golden ratio
                if abs(ratio - PHI) < 0.15:  # Tolerance parameter
                    scale_pairs.append(((min1, max1), (min2, max2)))
    
    # Calculate coherence for each pair
    for (min1, max1), (min2, max2) in scale_pairs:
        key1 = "{}-{}".format(min1, max1)
        key2 = "{}-{}".format(min2, max2)
        
        if key1 in phase_info and key2 in phase_info:
            if use_plv:
                coherence = compute_phase_locking_value(phase_info[key1], phase_info[key2])
            else:
                coherence = compute_phase_coherence(phase_info[key1], phase_info[key2])
            pair_key = "{}_x_{}".format(key1, key2)
            coherence_scores[pair_key] = coherence
    
    return coherence_scores

def run_phase_alignment_test(cmb_data, l_ranges, n_surrogates=1000, use_plv=False):
    """
    Run the full phase alignment test.
    
    Parameters:
    - cmb_data: The CMB data (either a map or a power spectrum)
    - l_ranges: List of (l_min, l_max) tuples defining the scale ranges
    - n_surrogates: Number of surrogate datasets to generate
    - use_plv: If True, use Phase Locking Value instead of traditional coherence
    
    Returns:
    - Dictionary of test results
    """
    # Extract spherical harmonics
    print("Extracting spherical harmonics...")
    alm = extract_spherical_harmonics(cmb_data)
    
    # Get phase information for each scale range
    print("Extracting phase information...")
    phase_info = get_phase_information(alm, l_ranges)
    
    # Find scale pairs related by golden ratio
    scales = sorted([tuple(map(int, k.split('-'))) for k in phase_info.keys()])
    gr_pairs = []
    
    print("Finding golden ratio scale pairs...")
    for i, (min1, max1) in enumerate(scales[:-1]):
        for (min2, max2) in scales[i+1:]:
            # Calculate effective scale ratio
            scale1 = (min1 + max1) / 2
            scale2 = (min2 + max2) / 2
            ratio = max(scale1, scale2) / min(scale1, scale2)
            
            # Check if close to golden ratio
            if abs(ratio - PHI) < 0.15:
                gr_pairs.append(((min1, max1), (min2, max2)))
                print("Found GR pair: ({}, {}) and ({}, {}), ratio: {:.4f}".format(min1, max1, min2, max2, ratio))
    
    # Calculate coherence for golden ratio pairs
    coherence_method = "Phase Locking Value" if use_plv else "Traditional Coherence"
    print("Calculating {} for golden ratio pairs...".format(coherence_method))
    gr_coherence = calculate_gr_phase_coherence(phase_info, gr_pairs, use_plv=use_plv)
    
    # Calculate mean coherence for all GR pairs
    mean_gr_coherence = np.mean(list(gr_coherence.values())) if gr_coherence else 0
    
    # Generate surrogate data and calculate statistics
    print("Generating {} surrogate datasets...".format(n_surrogates))
    surrogate_coherences = []
    
    for i in range(n_surrogates):
        if i % 100 == 0:
            print("Processing surrogate {}/{}".format(i, n_surrogates))
        
        surrogate_phase_info = {}
        for scale_key, phases in phase_info.items():
            # Generate randomized phases for this scale
            surrogate_phase_info[scale_key] = np.random.uniform(0, 2*PI, size=len(phases))
        
        # Calculate coherence for this surrogate
        surr_coherence = calculate_gr_phase_coherence(surrogate_phase_info, gr_pairs, use_plv=use_plv)
        mean_surr_coherence = np.mean(list(surr_coherence.values())) if surr_coherence else 0
        surrogate_coherences.append(mean_surr_coherence)
    
    # Calculate statistics
    surrogate_coherences = np.array(surrogate_coherences)
    mean_surrogate = np.mean(surrogate_coherences)
    std_surrogate = np.std(surrogate_coherences)
    
    # Calculate p-value (proportion of surrogates with coherence >= observed)
    p_value = np.sum(surrogate_coherences >= mean_gr_coherence) / n_surrogates
    
    # Calculate z-score
    z_score = (mean_gr_coherence - mean_surrogate) / std_surrogate if std_surrogate > 0 else 0
    
    # Calculate phi-optimality
    # Compare with coherence between scales related by other constants
    other_constants = {
        "e": 2.71828,
        "pi": 3.14159,
        "sqrt2": 1.41421,
        "sqrt3": 1.73205,
        "ln2": 0.69315
    }
    
    other_coherences = {}
    for const_name, const_value in other_constants.items():
        other_pairs = []
        for i, (min1, max1) in enumerate(scales[:-1]):
            for (min2, max2) in scales[i+1:]:
                scale1 = (min1 + max1) / 2
                scale2 = (min2 + max2) / 2
                ratio = max(scale1, scale2) / min(scale1, scale2)
                
                if abs(ratio - const_value) < 0.15:
                    other_pairs.append(((min1, max1), (min2, max2)))
        
        const_coherence = calculate_gr_phase_coherence(phase_info, other_pairs, use_plv=use_plv)
        other_coherences[const_name] = np.mean(list(const_coherence.values())) if const_coherence else 0
    
    # Calculate phi-optimality
    max_other_coherence = max(other_coherences.values()) if other_coherences else 0
    phi_optimality = (mean_gr_coherence - mean_surrogate) / (max_other_coherence - mean_surrogate) if max_other_coherence > mean_surrogate else 1.0
    
    # Compile results
    results = {
        "mean_gr_coherence": mean_gr_coherence,
        "mean_surrogate_coherence": mean_surrogate,
        "std_surrogate_coherence": std_surrogate,
        "z_score": z_score,
        "p_value": p_value,
        "phi_optimality": phi_optimality,
        "gr_pairs": gr_pairs,
        "gr_coherence": gr_coherence,
        "other_constants_coherence": other_coherences,
        "coherence_method": coherence_method
    }
    
    return results

def save_results(results, output_path, dataset_name):
    """Save test results to a file."""
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add coherence method to filename
    coherence_method = results.get("coherence_method", "Traditional Coherence")
    
    method_suffix = "plv" if "Locking" in coherence_method else "coherence"
    
    filename = os.path.join(output_path, "phase_alignment_{}_{}_{}.txt".format(
        dataset_name.lower(), method_suffix, timestamp))
    
    with open(filename, "w") as f:
        f.write("Phase Alignment Test Results for {}\n".format(dataset_name))
        f.write("Coherence Method: {}\n".format(coherence_method))
        f.write("Run at: {}\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        f.write("Overall Statistics:\n")
        f.write("Mean Golden Ratio Coherence: {:.6f}\n".format(results['mean_gr_coherence']))
        f.write("Mean Surrogate Coherence: {:.6f}\n".format(results['mean_surrogate_coherence']))
        f.write("Standard Deviation of Surrogate Coherence: {:.6f}\n".format(results['std_surrogate_coherence']))
        f.write("Z-score: {:.4f}\n".format(results['z_score']))
        f.write("P-value: {:.6f}\n".format(results['p_value']))
        f.write("Phi-optimality: {:.6f}\n\n".format(results['phi_optimality']))
        
        f.write("Other Constants Coherence:\n")
        for const_name, coherence in results["other_constants_coherence"].items():
            f.write("{}: {:.6f}\n".format(const_name, coherence))
        
        f.write("\nDetailed Golden Ratio Pair Results:\n")
        for pair_key, coherence in results["gr_coherence"].items():
            f.write("{}: {:.6f}\n".format(pair_key, coherence))
    
    print("Results saved to {}".format(filename))
    
    # Also generate visualization
    plot_filename = os.path.join(output_path, "phase_alignment_{}_{}_{}.png".format(
        dataset_name.lower(), method_suffix, timestamp))
    visualize_results(results, plot_filename, dataset_name)
    
    return filename

def visualize_results(results, filename, dataset_name):
    """Generate visualization of phase alignment test results."""
    plt.figure(figsize=(12, 8))
    
    # Get coherence method for title
    coherence_method = results.get("coherence_method", "Traditional Coherence")
    
    # Plot 1: Histogram of surrogate coherences with observed value
    plt.subplot(2, 2, 1)
    surrogate_mean = results["mean_surrogate_coherence"]
    surrogate_std = results["std_surrogate_coherence"]
    x = np.linspace(surrogate_mean - 4*surrogate_std, surrogate_mean + 4*surrogate_std, 100)
    plt.plot(x, 1/(surrogate_std * np.sqrt(2*np.pi)) * np.exp(-(x - surrogate_mean)**2 / (2*surrogate_std**2)), 'b-', label='Surrogate Distribution')
    plt.axvline(results["mean_gr_coherence"], color='r', linestyle='dashed', linewidth=2, label='Observed GR Coherence')
    plt.xlabel('Coherence')
    plt.ylabel('Probability Density')
    plt.title('Distribution of Surrogate Coherences')
    plt.legend()
    
    # Plot 2: Bar chart comparing GR with other constants
    plt.subplot(2, 2, 2)
    constants = list(results["other_constants_coherence"].keys()) + ["Golden Ratio"]
    coherences = list(results["other_constants_coherence"].values()) + [results["mean_gr_coherence"]]
    plt.bar(constants, coherences)
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Mean Phase Coherence')
    plt.title('Coherence by Mathematical Constant')
    plt.xticks(rotation=45)
    
    # Plot 3: Individual GR pair coherences
    plt.subplot(2, 2, 3)
    pair_keys = list(results["gr_coherence"].keys())
    pair_coherences = list(results["gr_coherence"].values())
    if pair_keys:  # Only if we have GR pairs
        plt.bar(pair_keys, pair_coherences)
        plt.xlabel('Scale Pairs')
        plt.ylabel('Coherence')
        plt.title('Coherence by Scale Pair')
        plt.xticks(rotation=90)
    
    # Plot 4: Summary statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = "Summary Statistics:\n\n"
    stats_text += "Coherence Method: {}\n\n".format(coherence_method)
    
    # GR coherence comparison
    wmap_gr = results["mean_gr_coherence"]
    stats_text += "Mean GR Coherence: {:.4f}\n".format(wmap_gr)
    stats_text += "Mean Surrogate Coherence: {:.4f}\n".format(results["mean_surrogate_coherence"])
    stats_text += "Z-score: {:.4f}\n".format(results["z_score"])
    stats_text += "P-value: {:.6f}\n".format(results["p_value"])
    stats_text += "Phi-optimality: {:.4f}\n\n".format(results["phi_optimality"])
    
    # Phi-optimality comparison
    stats_text += "Significant: {}\n".format("Yes" if results["p_value"] < 0.05 else "No")
    plt.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top')
    
    plt.suptitle("Phase Alignment Test Results for {} Dataset\n({})".format(dataset_name, coherence_method), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print("Visualization saved to {}".format(filename))

def compare_results(wmap_results, planck_results, output_file):
    """Compare WMAP and Planck phase alignment results."""
    # Create comparison visualization
    create_comparison_visualization(wmap_results, planck_results, output_file)
    
    # Get coherence method for output
    coherence_method = wmap_results.get("coherence_method", "Traditional Coherence")
    
    # Print comparison summary
    print("\nComparison of WMAP and Planck Results ({}):".format(coherence_method))
    print("=" * 50)
    
    print("\nGolden Ratio Coherence:")
    print("WMAP: {:.6f}".format(wmap_results["mean_gr_coherence"]))
    print("Planck: {:.6f}".format(planck_results["mean_gr_coherence"]))
    print("Difference: {:.6f}".format(planck_results["mean_gr_coherence"] - wmap_results["mean_gr_coherence"]))
    
    print("\nStatistical Significance:")
    print("WMAP p-value: {:.6f}".format(wmap_results["p_value"]))
    print("Planck p-value: {:.6f}".format(planck_results["p_value"]))
    
    print("\nPhi-optimality:")
    print("WMAP: {:.6f}".format(wmap_results["phi_optimality"]))
    print("Planck: {:.6f}".format(planck_results["phi_optimality"]))
    
    # Compare other constants
    print("\nOther Constants Comparison:")
    for const in wmap_results["other_constants_coherence"]:
        if const in planck_results["other_constants_coherence"]:
            wmap_val = wmap_results["other_constants_coherence"][const]
            planck_val = planck_results["other_constants_coherence"][const]
            diff = planck_val - wmap_val
            print("{}: WMAP={:.6f}, Planck={:.6f}, Diff={:.6f}".format(const, wmap_val, planck_val, diff))
    
    print("\nComparison visualization saved to {}".format(output_file))

def create_comparison_visualization(wmap_results, planck_results, filename):
    """Create visualization comparing WMAP and Planck results."""
    plt.figure(figsize=(14, 10))
    
    # Get coherence method for title
    coherence_method = wmap_results.get("coherence_method", "Traditional Coherence")
    
    # Plot 1: Compare GR coherence
    plt.subplot(2, 2, 1)
    datasets = ["WMAP", "Planck"]
    gr_coherences = [wmap_results["mean_gr_coherence"], planck_results["mean_gr_coherence"]]
    plt.bar(datasets, gr_coherences, color=['blue', 'red'])
    plt.ylabel('Golden Ratio Coherence')
    plt.title('Golden Ratio Coherence Comparison')
    
    # Add values on top of bars
    for i, v in enumerate(gr_coherences):
        plt.text(i, v + 0.01, "{:.4f}".format(v), ha='center')
    
    # Plot 2: Compare p-values
    plt.subplot(2, 2, 2)
    p_values = [wmap_results["p_value"], planck_results["p_value"]]
    plt.bar(datasets, p_values, color=['blue', 'red'])
    plt.ylabel('P-value')
    plt.title('Statistical Significance Comparison')
    
    # Add values on top of bars
    for i, v in enumerate(p_values):
        plt.text(i, v + 0.01, "{:.4f}".format(v), ha='center')
    
    # Plot 3: Compare coherence across constants
    plt.subplot(2, 2, 3)
    
    # Get common constants
    common_constants = set(wmap_results["other_constants_coherence"].keys()) & set(planck_results["other_constants_coherence"].keys())
    constants = list(common_constants) + ["Golden Ratio"]
    
    # Get values for each dataset
    wmap_values = [wmap_results["other_constants_coherence"].get(c, 0) for c in common_constants] + [wmap_results["mean_gr_coherence"]]
    planck_values = [planck_results["other_constants_coherence"].get(c, 0) for c in common_constants] + [planck_results["mean_gr_coherence"]]
    
    # Set up bar positions
    x = np.arange(len(constants))
    width = 0.35
    
    # Create grouped bars
    plt.bar(x - width/2, wmap_values, width, label='WMAP', color='blue')
    plt.bar(x + width/2, planck_values, width, label='Planck', color='red')
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Coherence')
    plt.title('Coherence by Constant Comparison')
    plt.xticks(x, constants, rotation=45)
    plt.legend()
    
    # Plot 4: Summary comparison
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Prepare comparison text
    comparison_text = "Comparison Summary:\n\n"
    comparison_text += "Coherence Method: {}\n\n".format(coherence_method)
    
    # GR coherence comparison
    wmap_gr = wmap_results["mean_gr_coherence"]
    planck_gr = planck_results["mean_gr_coherence"]
    diff_gr = planck_gr - wmap_gr
    comparison_text += "Golden Ratio Coherence:\n"
    comparison_text += "  WMAP: {:.4f}\n".format(wmap_gr)
    comparison_text += "  Planck: {:.4f}\n".format(planck_gr)
    comparison_text += "  Difference: {:.4f}\n\n".format(diff_gr)
    
    # P-value comparison
    wmap_p = wmap_results["p_value"]
    planck_p = planck_results["p_value"]
    comparison_text += "P-value:\n"
    comparison_text += "  WMAP: {:.6f} ({})\n".format(wmap_p, "Significant" if wmap_p < 0.05 else "Not Significant")
    comparison_text += "  Planck: {:.6f} ({})\n\n".format(planck_p, "Significant" if planck_p < 0.05 else "Not Significant")
    
    # Phi-optimality comparison
    wmap_phi = wmap_results["phi_optimality"]
    planck_phi = planck_results["phi_optimality"]
    diff_phi = planck_phi - wmap_phi
    comparison_text += "Phi-optimality:\n"
    comparison_text += "  WMAP: {:.4f}\n".format(wmap_phi)
    comparison_text += "  Planck: {:.4f}\n".format(planck_phi)
    comparison_text += "  Difference: {:.4f}\n".format(diff_phi)
    
    plt.text(0.1, 0.9, comparison_text, fontsize=10, verticalalignment='top')
    
    plt.suptitle("Comparison of Phase Alignment Results: WMAP vs. Planck\n({})".format(coherence_method), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print("Comparison visualization saved to {}".format(filename))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run phase alignment test on CMB data")
    parser.add_argument("--wmap-file", type=str, help="Path to WMAP data file")
    parser.add_argument("--planck-file", type=str, help="Path to Planck data file")
    parser.add_argument("--output-dir", type=str, default="../results/phase_alignment", help="Output directory for results")
    parser.add_argument("--num-simulations", type=int, default=1000, help="Number of surrogate simulations to run")
    parser.add_argument("--use-plv", action="store_true", help="Use Phase Locking Value instead of traditional coherence")
    parser.add_argument("--run-both-methods", action="store_true", help="Run tests with both coherence methods")
    return parser.parse_args()

def main():
    """Main function to run the phase alignment test."""
    args = parse_arguments()
    
    # Define output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Define multipole ranges to analyze
    # These ranges are based on typical CMB analysis patterns
    l_ranges = [
        (2, 10),     # Very large scales
        (10, 30),    # Large scales
        (30, 100),   # Medium scales
        (100, 300),  # Small scales
        (300, 500),  # Very small scales
        (500, 800),  # Ultra small scales
    ]
    
    # Run tests with both coherence methods if requested
    methods_to_run = [False, True] if args.run_both_methods else [args.use_plv]
    
    for use_plv in methods_to_run:
        method_name = "Phase Locking Value" if use_plv else "Traditional Coherence"
        print("\nRunning tests with {}...".format(method_name))
        
        # Process WMAP data if provided
        wmap_results = None
        if args.wmap_file:
            print("\nProcessing WMAP data...")
            wmap_data = load_cmb_map(args.wmap_file)
            wmap_output_dir = os.path.join(output_dir, "wmap")
            wmap_results = run_phase_alignment_test(wmap_data, l_ranges, args.num_simulations, use_plv=use_plv)
            save_results(wmap_results, wmap_output_dir, "WMAP")
        
        # Process Planck data if provided
        planck_results = None
        if args.planck_file:
            print("\nProcessing Planck data...")
            planck_data = load_cmb_map(args.planck_file)
            planck_output_dir = os.path.join(output_dir, "planck")
            planck_results = run_phase_alignment_test(planck_data, l_ranges, args.num_simulations, use_plv=use_plv)
            save_results(planck_results, planck_output_dir, "Planck")
        
        # Compare results if both datasets were processed
        if wmap_results and planck_results:
            comparison_dir = os.path.join(output_dir, "comparison")
            os.makedirs(comparison_dir, exist_ok=True)
            method_suffix = "plv" if use_plv else "coherence"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_file = os.path.join(comparison_dir, "comparison_{}_{}.png".format(method_suffix, timestamp))
            compare_results(wmap_results, planck_results, comparison_file)
    
    print("\nAll tests completed successfully.")

if __name__ == "__main__":
    main()
