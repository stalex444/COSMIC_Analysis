#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.stats import circmean, circstd
import os
from datetime import datetime
import argparse
from astropy.io import fits

# Constants
PHI = 1.618033988749895  # Golden ratio
PI = np.pi

def load_cmb_data(filename):
    """Load CMB data from a file."""
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
        # Try to load as a FITS file using astropy
        try:
            hdul = fits.open(filename)
            # Assuming the CMB map is in the first extension
            cmb_map = hdul[1].data
            hdul.close()
            return {'map': cmb_map, 'is_spectrum': False}
        except Exception as e:
            raise ValueError("Failed to load file {}: {}".format(filename, str(e)))

def generate_synthetic_alm(cl_dict, lmax=2000):
    """Generate synthetic alm coefficients from a power spectrum."""
    # Create a full cl array up to lmax
    cl = np.zeros(lmax + 1)
    for l in range(min(lmax + 1, max(cl_dict.keys()) + 1)):
        if l in cl_dict:
            cl[l] = cl_dict[l]
    
    # Generate random phases for each (l,m) mode
    alm = {}
    for l in range(1, lmax + 1):
        if cl[l] > 0:  # Only consider non-zero power
            for m in range(-l, l + 1):
                # Generate complex coefficient with random phase
                amplitude = np.sqrt(cl[l])
                phase = np.random.uniform(0, 2 * np.pi)
                alm[(l, m)] = amplitude * np.exp(1j * phase)
    
    return alm

def get_phase_information(alm, l_ranges):
    """Extract phase information for specified multipole ranges."""
    phase_info = {}
    
    for l_min, l_max in l_ranges:
        # Collect phases for this range
        phases = []
        for l in range(l_min, l_max + 1):
            for m in range(-l, l + 1):
                if (l, m) in alm:
                    # Extract phase from complex coefficient
                    phase = np.angle(alm[(l, m)])
                    phases.append(phase)
        
        if phases:
            phase_info[(l_min, l_max)] = np.array(phases)
    
    return phase_info

def compute_phase_coherence(phases1, phases2):
    """Compute phase coherence between two sets of phases."""
    # Ensure we're comparing equal-length arrays
    min_length = min(len(phases1), len(phases2))
    phases1 = phases1[:min_length]
    phases2 = phases2[:min_length]
    
    # Calculate phase differences
    phase_diff = phases1 - phases2
    
    # Wrap phase differences to [-pi, pi]
    phase_diff = np.angle(np.exp(1j * phase_diff))
    
    # Calculate mean resultant length (MRL) of phase differences
    # This is a measure of phase synchronization
    complex_phases = np.exp(1j * phase_diff)
    mrl = np.abs(np.mean(complex_phases))
    
    return mrl

def compute_phase_locking_value(phases1, phases2):
    """Compute phase-locking value between two sets of phases."""
    # Ensure we're comparing equal-length arrays
    min_length = min(len(phases1), len(phases2))
    phases1 = phases1[:min_length]
    phases2 = phases2[:min_length]
    
    # PLV is defined as |<exp(i*(phases1 - phases2))>|
    # where <> denotes average over all samples
    phase_diff = phases1 - phases2
    complex_diff = np.exp(1j * phase_diff)
    plv = np.abs(np.mean(complex_diff))
    
    return plv

def generate_surrogate_phases(phases, n_surrogates=1000):
    """Generate phase-randomized surrogates while preserving amplitude structure."""
    surrogates = []
    for _ in range(n_surrogates):
        # Create a random permutation of the phases
        surrogate = np.random.permutation(phases)
        surrogates.append(surrogate)
    return surrogates

def calculate_gr_phase_coherence(phase_info, scale_pairs=None, use_plv=False):
    """Calculate phase coherence between scales related by golden ratio."""
    coherence = {}
    gr_pairs = []
    
    # If no specific pairs are provided, find all pairs related by golden ratio
    if scale_pairs is None:
        scales = sorted(phase_info.keys())
        scale_pairs = []
        
        for i, (l_min1, l_max1) in enumerate(scales):
            for j, (l_min2, l_max2) in enumerate(scales[i+1:], i+1):
                # Check if the ratio of midpoints is close to golden ratio
                midpoint1 = (l_min1 + l_max1) / 2.0
                midpoint2 = (l_min2 + l_max2) / 2.0
                
                ratio = midpoint2 / midpoint1
                
                # Check if ratio is close to golden ratio (within 10%)
                if 0.9 * PHI <= ratio <= 1.1 * PHI:
                    scale_pairs.append(((l_min1, l_max1), (l_min2, l_max2)))
    
    # Calculate coherence for each scale pair
    for scale1, scale2 in scale_pairs:
        if scale1 in phase_info and scale2 in phase_info:
            phases1 = phase_info[scale1]
            phases2 = phase_info[scale2]
            
            # Use PLV or traditional coherence based on parameter
            if use_plv:
                coh = compute_phase_locking_value(phases1, phases2)
            else:
                coh = compute_phase_coherence(phases1, phases2)
            
            pair_key = "({},{}) - ({},{})".format(scale1[0], scale1[1], scale2[0], scale2[1])
            coherence[pair_key] = coh
            gr_pairs.append((scale1, scale2))
    
    return coherence, gr_pairs

def run_phase_alignment_test(cmb_data, l_ranges, n_surrogates=1000, use_plv=False):
    """Run the full phase alignment test."""
    # Generate or extract spherical harmonic coefficients
    if cmb_data.get('is_spectrum', False):
        alm = generate_synthetic_alm(cmb_data['cl'])
    else:
        # For real data, we would need healpy for proper spherical harmonic transform
        # Here we'll use a simplified approach for demonstration
        raise ValueError("This simplified version only supports power spectrum data")
    
    # Extract phase information for each scale range
    phase_info = get_phase_information(alm, l_ranges)
    
    # Calculate golden ratio phase coherence
    coherence_method = "Phase Locking Value" if use_plv else "Traditional Coherence"
    print("\nCalculating {} between scales related by golden ratio...".format(coherence_method))
    gr_coherence, gr_pairs = calculate_gr_phase_coherence(phase_info, use_plv=use_plv)
    
    # Calculate mean coherence across all GR pairs
    if gr_coherence:
        mean_gr_coherence = np.mean(list(gr_coherence.values()))
    else:
        mean_gr_coherence = 0
    
    # Calculate coherence for other mathematical constants for comparison
    other_constants = {
        "Square Root of 2": np.sqrt(2),
        "Square Root of 3": np.sqrt(3),
        "Natural Log of 2": np.log(2),
        "Euler's Number (e)": np.e,
        "Pi": np.pi
    }
    
    other_constants_coherence = {}
    
    for const_name, const_value in other_constants.items():
        print("Calculating {} for {}...".format(coherence_method, const_name))
        # Find scale pairs related by this constant
        scales = sorted(phase_info.keys())
        scale_pairs = []
        
        for i, (l_min1, l_max1) in enumerate(scales):
            for j, (l_min2, l_max2) in enumerate(scales[i+1:], i+1):
                midpoint1 = (l_min1 + l_max1) / 2.0
                midpoint2 = (l_min2 + l_max2) / 2.0
                
                ratio = midpoint2 / midpoint1
                
                # Check if ratio is close to the constant (within 10%)
                if 0.9 * const_value <= ratio <= 1.1 * const_value:
                    scale_pairs.append(((l_min1, l_max1), (l_min2, l_max2)))
        
        # Calculate coherence for these pairs
        const_coherence = {}
        for scale1, scale2 in scale_pairs:
            if scale1 in phase_info and scale2 in phase_info:
                phases1 = phase_info[scale1]
                phases2 = phase_info[scale2]
                
                if use_plv:
                    coh = compute_phase_locking_value(phases1, phases2)
                else:
                    coh = compute_phase_coherence(phases1, phases2)
                
                pair_key = "({},{}) - ({},{})".format(scale1[0], scale1[1], scale2[0], scale2[1])
                const_coherence[pair_key] = coh
        
        if const_coherence:
            other_constants_coherence[const_name] = np.mean(list(const_coherence.values()))
        else:
            other_constants_coherence[const_name] = 0
    
    # Generate surrogate datasets and calculate their coherence
    print("\nGenerating {} surrogate datasets...".format(n_surrogates))
    surrogate_coherences = []
    
    # For each GR pair, generate surrogates and calculate coherence
    for scale1, scale2 in gr_pairs:
        phases1 = phase_info[scale1]
        phases2 = phase_info[scale2]
        
        # Generate surrogate phases by randomizing phase2
        surrogate_phases2 = generate_surrogate_phases(phases2, n_surrogates)
        
        # Calculate coherence for each surrogate
        for surrogate in surrogate_phases2:
            if use_plv:
                coh = compute_phase_locking_value(phases1, surrogate)
            else:
                coh = compute_phase_coherence(phases1, surrogate)
            surrogate_coherences.append(coh)
    
    # Calculate statistics
    mean_surrogate_coherence = np.mean(surrogate_coherences)
    std_surrogate_coherence = np.std(surrogate_coherences)
    
    # Calculate z-score
    if std_surrogate_coherence > 0:
        z_score = (mean_gr_coherence - mean_surrogate_coherence) / std_surrogate_coherence
    else:
        z_score = 0
    
    # Calculate p-value (one-tailed test)
    p_value = 1 - norm.cdf(z_score)
    
    # Calculate phi-optimality (how much better GR is compared to other constants)
    other_coherences = list(other_constants_coherence.values())
    if other_coherences:
        mean_other_coherence = np.mean(other_coherences)
        if mean_other_coherence > 0:
            phi_optimality = mean_gr_coherence / mean_other_coherence
        else:
            phi_optimality = 0
    else:
        phi_optimality = 0
    
    # Compile results
    results = {
        "gr_coherence": gr_coherence,
        "gr_pairs": gr_pairs,
        "mean_gr_coherence": mean_gr_coherence,
        "other_constants_coherence": other_constants_coherence,
        "surrogate_coherences": surrogate_coherences,
        "mean_surrogate_coherence": mean_surrogate_coherence,
        "std_surrogate_coherence": std_surrogate_coherence,
        "z_score": z_score,
        "p_value": p_value,
        "phi_optimality": phi_optimality,
        "coherence_method": coherence_method
    }
    
    return results

def save_results(results, output_path, dataset_name):
    """Save test results to a file."""
    # Create output directory if it doesn't exist (Python 2.7 compatible)
    try:
        os.makedirs(output_path)
    except OSError:
        if not os.path.isdir(output_path):
            raise
            
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
    
    # Create histogram of surrogate coherences
    plt.hist(results["surrogate_coherences"], bins=30, alpha=0.7, density=True, label='Surrogate Distribution')
    
    # Add normal distribution curve
    x = np.linspace(surrogate_mean - 4*surrogate_std, surrogate_mean + 4*surrogate_std, 100)
    plt.plot(x, 1/(surrogate_std * np.sqrt(2*np.pi)) * np.exp(-(x - surrogate_mean)**2 / (2*surrogate_std**2)), 'b-')
    
    # Add observed value
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
    stats_text += "Mean GR Coherence: {:.4f}\n".format(results["mean_gr_coherence"])
    stats_text += "Mean Surrogate Coherence: {:.4f}\n".format(results["mean_surrogate_coherence"])
    stats_text += "Z-score: {:.4f}\n".format(results["z_score"])
    stats_text += "P-value: {:.6f}\n".format(results["p_value"])
    stats_text += "Phi-optimality: {:.4f}\n\n".format(results["phi_optimality"])
    
    # Significance
    stats_text += "Significant: {}\n".format("Yes" if results["p_value"] < 0.05 else "No")
    plt.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top')
    
    plt.suptitle("Phase Alignment Test Results for {} Dataset\n({})".format(dataset_name, coherence_method), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print("Visualization saved to {}".format(filename))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run phase alignment test on CMB data")
    parser.add_argument("--data-file", type=str, required=True, help="Path to CMB data file (power spectrum)")
    parser.add_argument("--output-dir", type=str, default="../results/phase_alignment", help="Output directory for results")
    parser.add_argument("--num-simulations", type=int, default=1000, help="Number of surrogate simulations to run")
    parser.add_argument("--use-plv", action="store_true", help="Use Phase Locking Value instead of traditional coherence")
    parser.add_argument("--run-both-methods", action="store_true", help="Run tests with both coherence methods")
    parser.add_argument("--dataset-name", type=str, default="CMB", help="Name of the dataset (e.g., WMAP, Planck)")
    return parser.parse_args()

def main():
    """Main function to run the phase alignment test."""
    args = parse_arguments()
    
    # Define multipole ranges to analyze
    l_ranges = [
        (2, 10),     # Very large scales
        (10, 30),    # Large scales
        (30, 100),   # Medium scales
        (100, 300),  # Small scales
        (300, 500),  # Very small scales
        (500, 800),  # Ultra small scales
    ]
    
    # Create output directory (Python 2.7 compatible)
    try:
        os.makedirs(args.output_dir)
    except OSError:
        if not os.path.isdir(args.output_dir):
            raise
            
    # Load data
    print("\nLoading data from {}...".format(args.data_file))
    cmb_data = load_cmb_data(args.data_file)
    
    # Run tests with both coherence methods if requested
    methods_to_run = [False, True] if args.run_both_methods else [args.use_plv]
    
    for use_plv in methods_to_run:
        method_name = "Phase Locking Value" if use_plv else "Traditional Coherence"
        print("\nRunning test with {}...".format(method_name))
        
        # Run the test
        results = run_phase_alignment_test(cmb_data, l_ranges, args.num_simulations, use_plv=use_plv)
        
        # Save results
        output_dir = os.path.join(args.output_dir, args.dataset_name.lower())
        save_results(results, output_dir, args.dataset_name)
    
    print("\nAll tests completed successfully.")

if __name__ == "__main__":
    # Import scipy.stats.norm here to avoid potential import issues
    from scipy.stats import norm
    
    # Make sure the base results directory exists (Python 2.7 compatible)
    try:
        os.makedirs("../results/phase_alignment")
    except OSError:
        if not os.path.isdir("../results/phase_alignment"):
            raise
    
    main()
