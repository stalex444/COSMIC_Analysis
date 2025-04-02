#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Set the path to COSMIC_Analysis directory
cosmic_dir = os.path.dirname(parent_dir)

# Constants
PHI = 1.618033988749895  # Golden ratio
PI = np.pi
E = np.e
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
LN2 = np.log(2)

def load_power_spectrum(dataset):
    """
    Load power spectrum data from file.
    
    Args:
        dataset (str): Name of the dataset ('WMAP' or 'Planck')
        
    Returns:
        dict: Dictionary with 'multipoles' and 'power' arrays, or None if file not found
    """
    # Define file paths using cosmic_dir
    if dataset == "WMAP":
        file_path = os.path.join(cosmic_dir, "data", "wmap", "wmap_tt_spectrum_9yr_v5.txt")
    elif dataset == "Planck":
        file_path = os.path.join(cosmic_dir, "data", "planck", "planck_tt_spectrum_2018.txt")
    else:
        print("Unknown dataset: {}".format(dataset))
        return None
    
    # Check if file exists
    if not os.path.exists(file_path):
        print("Power spectrum file not found: {}".format(file_path))
        return None
    
    # Load data
    try:
        print(f"Loading {dataset} power spectrum from: {file_path}")
        # For WMAP data format
        if dataset == "WMAP":
            data = np.loadtxt(file_path, skiprows=1)
            multipoles = data[:, 0]
            power = data[:, 1]
        # For Planck data format
        else:
            data = np.loadtxt(file_path, skiprows=1)
            multipoles = data[:, 0]
            power = data[:, 1]
            
        print(f"Loaded {dataset} power spectrum with {len(multipoles)} multipoles")
        return {'multipoles': multipoles, 'power': power}
    except Exception as e:
        print(f"Error loading power spectrum data: {str(e)}")
        return None

def define_scale_ranges():
    """
    Define multiple scale ranges to be analyzed.
    Returns a list of (min, max) tuples defining the multipole ranges.
    """
    return [
        (2, 10),      # Very large scales
        (11, 30),     # Large scales
        (31, 100),    # Medium-large scales
        (101, 300),   # Medium scales
        (301, 500),   # Medium-small scales
        (501, 800),   # Small scales
        (801, 1200),  # Very small scales (Planck only)
        (1201, 2000)  # Ultra small scales (Planck only)
    ]

def reconstruct_phases(power, multipoles):
    """
    Reconstruct phase information using statistical properties of the power spectrum.
    
    Parameters:
    - power: Power spectrum values C_l
    - multipoles: Corresponding multipole values l
    
    Returns:
    - Dictionary mapping multipole ranges to phase arrays
    """
    phase_info = {}
    
    # Define multipole ranges
    l_ranges = define_scale_ranges()
    
    for l_min, l_max in l_ranges:
        range_key = "%d-%d" % (l_min, l_max)
        # Filter to relevant multipoles
        mask = (multipoles >= l_min) & (multipoles <= l_max)
        if not np.any(mask):
            continue
            
        range_power = power[mask]
        range_multipoles = multipoles[mask]
        
        if len(range_power) < 2:
            # Not enough data points for this range
            continue
        
        # Calculate autocorrelation of power
        autocorr = np.correlate(range_power, range_power, mode='full')
        mid = len(autocorr) // 2
        autocorr = autocorr[mid:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Use autocorrelation structure to inform phase relationships
        # This creates phase relationships that respect the power spectrum structure
        phases = np.zeros(len(range_power))
        
        # Start with some seed phases
        phases[0] = 0
        if len(phases) > 1:
            phases[1] = PI/4
        
        # Generate remaining phases using autocorrelation structure
        for i in range(2, len(phases)):
            # Use autocorrelation as a weight for phase progression
            weight = np.abs(autocorr[i % len(autocorr)])
            # Create structured rather than random progression
            phase_step = PI * weight
            phases[i] = (phases[i-1] + phase_step) % (2*PI)
        
        phase_info[range_key] = phases
            
    return phase_info

def compute_traditional_coherence(phases1, phases2):
    """
    Compute traditional phase coherence between two sets of phases.
    Returns a value between 0 (no coherence) and 1 (perfect coherence).
    """
    # Ensure equal length comparison
    min_length = min(len(phases1), len(phases2))
    if min_length < 2:
        return 0
        
    phases1 = phases1[:min_length]
    phases2 = phases2[:min_length]
    
    # Calculate phase differences
    phase_diffs = np.abs(phases1 - phases2)
    
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
    # Ensure equal length comparison
    min_length = min(len(phases1), len(phases2))
    if min_length < 2:
        return 0
        
    phases1 = phases1[:min_length]
    phases2 = phases2[:min_length]
    
    # Calculate complex phase difference
    complex_phase_diff = np.exp(1j * (phases1 - phases2))
    
    # Calculate PLV
    plv = np.abs(np.mean(complex_phase_diff))
    
    return plv

def find_scale_pairs_for_constant(multipoles, constant, tolerance=0.01):
    """
    Identify pairs of scales whose ratio is close to a specific mathematical constant.
    
    Parameters:
    - multipoles: Array of multipole values
    - constant: The mathematical constant to test
    - tolerance: Tolerance for ratio matching (default: 0.01)
    
    Returns:
    - List of scale pairs (tuples of multipole ranges)
    """
    # Identify pairs of scales whose ratio is close to the constant
    scale_pairs = []
    
    # Increase the number of multipoles to check for better coverage
    max_check = min(2000, len(multipoles))
    
    # Use a more efficient approach for large multipole arrays
    multipole_set = set(multipoles[:max_check])
    
    # For each multipole, check if there's another multipole that forms a ratio close to the constant
    for i, l1 in enumerate(multipoles[:max_check]):
        # Calculate the target multipole value
        target = l1 * constant
        
        # Find the closest multipole to the target
        closest_multipoles = []
        
        # Check multipoles within the tolerance range
        lower_bound = target * (1 - tolerance)
        upper_bound = target * (1 + tolerance)
        
        for l2 in multipoles[:max_check]:
            if lower_bound <= l2 <= upper_bound:
                closest_multipoles.append(l2)
        
        # Add all valid pairs
        for l2 in closest_multipoles:
            ratio = float(l2) / float(l1)
            if abs(ratio - constant) <= tolerance:
                scale_pairs.append((l1, l2))
                print("    Found pair: ({}, {}), ratio: {:.4f}".format(l1, l2, ratio))
    
    print("  Total pairs found: {}".format(len(scale_pairs)))
    return scale_pairs

def calculate_coherence_for_constant(phase_info, scale_pairs, coherence_func):
    """
    Calculate phase coherence between scales related by a specific constant.
    
    Parameters:
    - phase_info: Dictionary mapping scale ranges to phase arrays
    - scale_pairs: List of scale pairs (tuples of multipole ranges)
    - coherence_func: Function to compute coherence
    
    Returns:
    - List of coherence values
    """
    coherence_values = []
    
    for scale1, scale2 in scale_pairs:
        # Find corresponding phase arrays
        phase1 = None
        phase2 = None
        for key, phases in phase_info.items():
            min_l, max_l = map(int, key.split('-'))
            if min_l <= scale1 <= max_l:
                phase1 = phases
            if min_l <= scale2 <= max_l:
                phase2 = phases
        
        if phase1 is not None and phase2 is not None:
            # Calculate coherence
            coherence = coherence_func(phase1, phase2)
            coherence_values.append(coherence)
    
    return coherence_values

def generate_surrogate_coherence(phase_info, scale_pairs, coherence_func, num_simulations):
    """
    Generate surrogate coherence values by randomizing phase relationships.
    
    Parameters:
    - phase_info: Dictionary mapping scale ranges to phase arrays
    - scale_pairs: List of scale pairs (tuples of multipole ranges)
    - coherence_func: Function to compute coherence
    - num_simulations: Number of surrogate datasets to generate
    
    Returns:
    - Array of surrogate coherence values
    """
    surrogate_values = []
    
    # Pre-compute scale to phase_info key mapping to avoid repeated lookups
    scale_to_key = {}
    for scale in set([s1 for s1, s2 in scale_pairs] + [s2 for s1, s2 in scale_pairs]):
        for key in phase_info.keys():
            min_l, max_l = map(int, key.split('-'))
            if min_l <= scale <= max_l:
                scale_to_key[scale] = key
                break
    
    # Sample a subset of scale pairs if there are too many
    if len(scale_pairs) > 1000:
        print("  Sampling 1000 scale pairs from {} total pairs for efficiency".format(len(scale_pairs)))
        sampled_pairs = np.random.choice(len(scale_pairs), size=1000, replace=False)
        sampled_scale_pairs = [scale_pairs[i] for i in sampled_pairs]
    else:
        sampled_scale_pairs = scale_pairs
    
    for i in range(num_simulations):
        if i % 100 == 0:  # Increased frequency of progress updates
            print("  Processing surrogate {}/{}...".format(i, num_simulations))
        
        # Create surrogate by randomizing phases
        surrogate_phase_info = {}
        for key, phases in phase_info.items():
            surrogate_phase_info[key] = np.random.uniform(0, 2*PI, size=len(phases))
        
        # Calculate coherence for each pair
        pair_coherences = []
        for scale1, scale2 in sampled_scale_pairs:
            # Get corresponding phase arrays using pre-computed mapping
            key1 = scale_to_key.get(scale1)
            key2 = scale_to_key.get(scale2)
            
            if key1 is not None and key2 is not None:
                phase1 = surrogate_phase_info[key1]
                phase2 = surrogate_phase_info[key2]
                # Calculate coherence
                coherence = coherence_func(phase1, phase2)
                pair_coherences.append(coherence)
        
        # Calculate mean across pairs
        if pair_coherences:
            surrogate_values.append(np.mean(pair_coherences))
    
    return np.array(surrogate_values)

def create_summary(results):
    """
    Create summary of phase alignment test results.
    
    Parameters:
    - results: Dictionary with results for each constant
    
    Returns:
    - Dictionary with summary statistics
    """
    summary = {
        'highest_coherence_constant': None,
        'highest_coherence_value': 0,
        'phi_coherence': 0
    }
    
    for const, result in results.items():
        if result['mean_coherence'] > summary['highest_coherence_value']:
            summary['highest_coherence_constant'] = const
            summary['highest_coherence_value'] = result['mean_coherence']
        
        if const == 'phi':
            summary['phi_coherence'] = result['mean_coherence']
    
    return summary

def run_phase_alignment_test_internal(power_spectrum, constants=None, coherence_method="Traditional", num_simulations=10000):
    """
    Run phase alignment test with comprehensive analysis.
    
    Parameters:
    - power_spectrum: Dictionary with 'multipoles' and 'power' or 'power_spectrum' arrays
    - constants: List of mathematical constants to test
    - coherence_method: Method for coherence calculation ('Traditional' or 'PLV')
    - num_simulations: Number of surrogate datasets to generate
    
    Returns:
    - Dictionary with results and summary
    """
    # Set default constants if not provided
    if constants is None:
        constants = [PHI, PI, E, SQRT2, SQRT3, LN2]
    
    # Choose coherence function based on method
    if coherence_method == "Traditional":
        coherence_func = compute_traditional_coherence
    elif coherence_method == "PLV":
        coherence_func = compute_phase_locking_value
    else:
        print("Unknown coherence method: {}. Using Traditional.".format(coherence_method))
        coherence_func = compute_traditional_coherence
    
    print("Reconstructing phases...")
    if 'power' in power_spectrum:
        multipoles = power_spectrum['multipoles']
        power = power_spectrum['power']
    elif 'power_spectrum' in power_spectrum:
        multipoles = power_spectrum['multipoles']
        power = power_spectrum['power_spectrum']
    else:
        raise ValueError("Power spectrum dictionary must contain 'power' or 'power_spectrum' key")
    
    phase_info = reconstruct_phases(power, multipoles)
    
    # Initialize results dictionary
    results = {}
    
    # Test each mathematical constant
    for constant in constants:
        const_name = constant_name(constant)
        print("Testing {}...".format(const_name))
        
        # Find scale pairs for this constant
        print("  Finding scale pairs for constant: {}".format(const_name))
        print("  Number of multipoles: {}".format(len(multipoles)))
        scale_pairs = find_scale_pairs_for_constant(multipoles, constant)
        
        if len(scale_pairs) < 5:
            print("Not enough scale pairs found for {}. Skipping.".format(const_name))
            results[const_name.split()[0].lower()] = {
                'constant': constant,
                'num_pairs': len(scale_pairs),
                'mean_coherence': 0,
                'p_value': 1.0,
                'z_score': 0,
                'significant': False
            }
            continue
        
        # Calculate coherence for this constant
        coherence_values = calculate_coherence_for_constant(
            phase_info, scale_pairs, coherence_func
        )
        
        # Calculate mean coherence
        mean_coherence = np.mean(coherence_values)
        
        # Generate surrogate coherence values
        print("Generating {} surrogate datasets...".format(num_simulations))
        surrogate_values = generate_surrogate_coherence(
            phase_info, scale_pairs, coherence_func, num_simulations
        )
        
        # Calculate p-value
        p_value = np.mean(surrogate_values >= mean_coherence)
        
        # Calculate z-score
        surrogate_mean = np.mean(surrogate_values)
        surrogate_std = np.std(surrogate_values)
        z_score = (mean_coherence - surrogate_mean) / surrogate_std
        
        # Store results
        const_key = const_name.split()[0].lower()
        results[const_key] = {
            'constant': constant,
            'num_pairs': len(scale_pairs),
            'mean_coherence': mean_coherence,
            'p_value': p_value,
            'z_score': z_score,
            'significant': p_value < 0.05,
            'surrogate_mean': surrogate_mean,
            'surrogate_std': surrogate_std,
            'surrogate_values': surrogate_values.tolist()[:1000]  # Store only 1000 values to save memory
        }
        
        print("{}: Mean Coherence = {:.6f}, p-value = {:.6f}, z-score = {:.4f}".format(
            const_name, mean_coherence, p_value, z_score
        ))
    
    # Create summary
    summary = create_summary(results)
    
    return {
        'results': results,
        'summary': summary
    }

def run_phase_alignment_test(wmap_data=None, planck_data=None, num_surrogates=10000):
    """
    Run the Phase Alignment Test on WMAP and Planck data.
    
    Parameters:
    - wmap_data: WMAP dataset (optional, will load from file if None)
    - planck_data: Planck dataset (optional, will load from file if None)
    - num_surrogates: Number of surrogate datasets to generate for statistical testing
    
    Returns:
    - Dictionary with results for both datasets and both methods
    """
    print("Running Phase Alignment Test with %d surrogates..." % num_surrogates)
    
    # Initialize results dictionary
    all_results = {}
    
    # Constants to test
    constants = [PHI, PI, E, SQRT2, SQRT3, LN2]
    
    # Process WMAP data
    if wmap_data is None:
        wmap_data = load_power_spectrum("WMAP")
        if wmap_data is None:
            print("Creating sample WMAP data...")
            wmap_data = create_sample_power_spectrum()
    
    # Process Planck data
    if planck_data is None:
        planck_data = load_power_spectrum("Planck")
        if planck_data is None:
            print("Creating sample Planck data...")
            planck_data = create_sample_power_spectrum()
    
    # Run Traditional method tests
    print("Running Traditional coherence method on WMAP data...")
    wmap_trad = run_phase_alignment_test_internal(
        wmap_data, constants, "Traditional", num_surrogates)
    all_results["WMAP_Traditional"] = wmap_trad
    
    print("Running Traditional coherence method on Planck data...")
    planck_trad = run_phase_alignment_test_internal(
        planck_data, constants, "Traditional", num_surrogates)
    all_results["Planck_Traditional"] = planck_trad
    
    # Run PLV method tests
    print("Running PLV coherence method on WMAP data...")
    wmap_plv = run_phase_alignment_test_internal(
        wmap_data, constants, "PLV", num_surrogates)
    all_results["WMAP_PLV"] = wmap_plv
    
    print("Running PLV coherence method on Planck data...")
    planck_plv = run_phase_alignment_test_internal(
        planck_data, constants, "PLV", num_surrogates)
    all_results["Planck_PLV"] = planck_plv
    
    return all_results

def create_sample_power_spectrum():
    """
    Create sample power spectrum data for testing.
    
    Returns:
        dict: Dictionary with 'multipoles' and 'power' arrays
    """
    # Create multipoles from 2 to 2000
    multipoles = np.arange(2, 2001)
    
    # Create power spectrum following a power law with noise
    power = 1000 * multipoles ** (-0.9) * (1 + 0.1 * np.random.randn(len(multipoles)))
    
    # Add features related to mathematical constants
    
    # Add a feature related to phi (Golden Ratio)
    phi_scale = int(100 * PHI)
    power[phi_scale-5:phi_scale+5] *= 1.5
    
    # Add a feature related to pi
    pi_scale = int(100 * PI)
    power[pi_scale-5:pi_scale+5] *= 1.4
    
    # Add a feature related to e
    e_scale = int(100 * E)
    power[e_scale-5:e_scale+5] *= 1.3
    
    # Add a feature related to sqrt(2)
    sqrt2_scale = int(100 * SQRT2)
    power[sqrt2_scale-5:sqrt2_scale+5] *= 1.45
    
    # Add a feature related to sqrt(3)
    sqrt3_scale = int(100 * SQRT3)
    power[sqrt3_scale-5:sqrt3_scale+5] *= 1.4
    
    # Add a feature related to ln(2)
    ln2_scale = int(100 * LN2)
    power[ln2_scale-5:ln2_scale+5] *= 1.35
    
    # Add specific scale pairs for each constant to ensure they're detected
    # For Phi (Golden Ratio)
    for i in range(20, 1000, 100):
        j = int(i * PHI)
        if j < len(power):
            power[i] *= 1.2
            power[j] *= 1.2
    
    # For Pi
    for i in range(30, 1000, 100):
        j = int(i * PI)
        if j < len(power):
            power[i] *= 1.2
            power[j] *= 1.2
    
    # For e
    for i in range(40, 1000, 100):
        j = int(i * E)
        if j < len(power):
            power[i] *= 1.2
            power[j] *= 1.2
    
    # For sqrt(2)
    for i in range(50, 1000, 100):
        j = int(i * SQRT2)
        if j < len(power):
            power[i] *= 1.2
            power[j] *= 1.2
    
    # For sqrt(3)
    for i in range(60, 1000, 100):
        j = int(i * SQRT3)
        if j < len(power):
            power[i] *= 1.2
            power[j] *= 1.2
    
    # For ln(2)
    for i in range(70, 1000, 100):
        j = int(i * LN2)
        if j < len(power):
            power[i] *= 1.2
            power[j] *= 1.2
    
    print("Created sample power spectrum with {} multipoles".format(len(multipoles)))
    return {'multipoles': multipoles, 'power': power}

def create_comparison_visualization(all_results, output_dir):
    """Create comprehensive comparison visualization."""
    plt.figure(figsize=(18, 12))
    
    constants = ["phi", "pi", "e", "sqrt2", "sqrt3", "ln2"]
    datasets = ["WMAP", "Planck"]
    methods = ["Traditional", "PLV"]
    
    # Plot 1: Compare coherence across datasets and methods
    plt.subplot(2, 2, 1)
    
    # Group data for plotting
    x_labels = []
    coherence_data = []
    colors = []
    
    for const in constants:
        for dataset in datasets:
            for method in methods:
                if (dataset in all_results and 
                    method in all_results[dataset] and 
                    all_results[dataset][method] is not None and
                    'results' in all_results[dataset][method] and
                    const in all_results[dataset][method]['results']):
                    
                    result = all_results[dataset][method]['results'][const]
                    x_labels.append("%s_%s_%s" % (const[:3], dataset[:1], method[:1]))
                    coherence_data.append(result['mean_coherence'])
                    
                    # Use different colors for different constants
                    if const == "phi":
                        colors.append('gold')
                    elif const == "pi":
                        colors.append('blue')
                    else:
                        colors.append('gray')
    
    plt.bar(range(len(x_labels)), coherence_data, color=colors)
    plt.xlabel('Constant_Dataset_Method')
    plt.ylabel('Mean Coherence')
    plt.title('Phase Coherence Comparison')
    
    # Format x-tick labels
    formatted_labels = []
    for x in x_labels:
        formatted_labels.append("%s_%s" % (x[:3], x[4]))
    plt.xticks(range(len(x_labels)), formatted_labels, rotation=90)
    
    # Highlight phi and pi
    plt.axhline(y=np.nanmean([d for d, l in zip(coherence_data, x_labels) if l.startswith('phi')]), 
               color='gold', linestyle='--', label='Phi Mean')
    plt.axhline(y=np.nanmean([d for d, l in zip(coherence_data, x_labels) if l.startswith('pi')]), 
               color='blue', linestyle='--', label='Pi Mean')
    plt.legend()
    
    # Plot 2: Compare z-scores
    plt.subplot(2, 2, 2)
    
    # Group data for plotting
    x_labels = []
    z_score_data = []
    colors = []
    
    for const in constants:
        for dataset in datasets:
            for method in methods:
                if (dataset in all_results and 
                    method in all_results[dataset] and 
                    all_results[dataset][method] is not None and
                    'results' in all_results[dataset][method] and
                    const in all_results[dataset][method]['results']):
                    
                    result = all_results[dataset][method]['results'][const]
                    x_labels.append("%s_%s_%s" % (const[:3], dataset[:1], method[:1]))
                    z_score_data.append(result['z_score'])
                    
                    # Use different colors for different constants
                    if const == "phi":
                        colors.append('gold')
                    elif const == "pi":
                        colors.append('blue')
                    else:
                        colors.append('gray')
    
    plt.bar(range(len(x_labels)), z_score_data, color=colors)
    plt.xlabel('Constant_Dataset_Method')
    plt.ylabel('Z-score')
    plt.title('Statistical Significance Comparison')
    
    # Format x-tick labels
    formatted_labels = []
    for x in x_labels:
        formatted_labels.append("%s_%s" % (x[:3], x[4]))
    plt.xticks(range(len(x_labels)), formatted_labels, rotation=90)
    
    plt.axhline(y=1.96, color='r', linestyle='--', label='p=0.05')
    plt.legend()
    
    # Plot 3: Number of scale pairs
    plt.subplot(2, 2, 3)
    
    # Group data for plotting
    x_labels = []
    pair_data = []
    colors = []
    
    for const in constants:
        for dataset in datasets:
            # Just use Traditional method for simplicity
            method = "Traditional"
            if (dataset in all_results and 
                method in all_results[dataset] and 
                all_results[dataset][method] is not None and
                'results' in all_results[dataset][method] and
                const in all_results[dataset][method]['results']):
                
                result = all_results[dataset][method]['results'][const]
                x_labels.append("%s_%s" % (const[:3], dataset[:1]))
                pair_data.append(result['num_pairs'])
                
                # Use different colors for different constants
                if const == "phi":
                    colors.append('gold')
                elif const == "pi":
                    colors.append('blue')
                else:
                    colors.append('gray')
    
    plt.bar(range(len(x_labels)), pair_data, color=colors)
    plt.xlabel('Constant_Dataset')
    plt.ylabel('Number of Scale Pairs')
    plt.title('Scale Pair Comparison')
    plt.xticks(range(len(x_labels)), x_labels, rotation=90)
    
    # Plot 4: Summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Determine highest coherence across all tests
    highest_coherence = {'value': 0, 'const': None, 'dataset': None, 'method': None}
    
    for dataset in datasets:
        for method in methods:
            if (dataset in all_results and 
                method in all_results[dataset] and 
                all_results[dataset][method] is not None and
                'summary' in all_results[dataset][method]):
                
                summary = all_results[dataset][method]['summary']
                if summary['highest_coherence_value'] > highest_coherence['value']:
                    highest_coherence = {
                        'value': summary['highest_coherence_value'],
                        'const': summary['highest_coherence_constant'],
                        'dataset': dataset,
                        'method': method
                    }
    
    # Get average z-scores for phi and pi
    phi_z_scores = []
    pi_z_scores = []
    
    for dataset in datasets:
        for method in methods:
            if (dataset in all_results and 
                method in all_results[dataset] and 
                all_results[dataset][method] is not None and
                'results' in all_results[dataset][method]):
                
                results = all_results[dataset][method]['results']
                if 'phi' in results:
                    phi_z_scores.append(results['phi']['z_score'])
                if 'pi' in results:
                    pi_z_scores.append(results['pi']['z_score'])
    
    avg_phi_z = np.mean(phi_z_scores) if phi_z_scores else float('nan')
    avg_pi_z = np.mean(pi_z_scores) if pi_z_scores else float('nan')
    
    summary_text = """
    Comprehensive Phase Alignment Test Summary
    
    Highest Coherence Overall:
    %s in %s using %s
    Value: %.6f
    
    Average Z-scores:
    - Phi (Golden Ratio): %.4f
    - Pi: %.4f
    
    Key Observations:
    - %s
    - %s
    - Scale pair analysis shows varying coverage across constants and datasets
    """ % (
        highest_coherence['const'],
        highest_coherence['dataset'],
        highest_coherence['method'],
        highest_coherence['value'],
        avg_phi_z,
        avg_pi_z,
        "Pi shows higher coherence than Phi across tests" if avg_pi_z > avg_phi_z else "Phi shows higher coherence than Pi across tests",
        "Some tests show statistically significant results (Z > 1.96)" if max(avg_phi_z, avg_pi_z) > 1.96 else "No tests show statistically significant results"
    )
    
    plt.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top')
    
    # Save the comparison visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "%s/phase_alignment_comparison_%s.png" % (output_dir, timestamp)
    plt.tight_layout()
    plt.savefig(filename)
    print("Comparison visualization saved to %s" % filename)

def run_all_tests(wmap_power_file, planck_power_file, output_dir):
    """
    Run all phase alignment tests on both datasets with both coherence methods.
    
    Parameters:
    - wmap_power_file: Path to WMAP power spectrum file
    - planck_power_file: Path to Planck power spectrum file
    - output_dir: Directory to save results
    
    Returns:
    - Dictionary with all test results
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Constants to test
    constants = [PHI, PI, E, SQRT2, SQRT3, LN2]
    
    # Load WMAP data
    print("Loading WMAP power spectrum from %s" % wmap_power_file)
    wmap_multipoles, wmap_power = load_power_spectrum(wmap_power_file)
    if wmap_multipoles is None:
        print("Failed to load WMAP data, using sample data")
        wmap_data = create_sample_power_spectrum()
    else:
        wmap_data = {
            'multipoles': wmap_multipoles,
            'power': wmap_power
        }
    
    # Load Planck data
    print("Loading Planck power spectrum from %s" % planck_power_file)
    planck_multipoles, planck_power = load_power_spectrum(planck_power_file)
    if planck_multipoles is None:
        print("Failed to load Planck data, using sample data")
        planck_data = create_sample_power_spectrum()
    else:
        planck_data = {
            'multipoles': planck_multipoles,
            'power': planck_power
        }
    
    all_results = {}
    
    # Run Traditional method tests
    print("Running Traditional coherence method on WMAP data...")
    wmap_trad = run_phase_alignment_test_internal(
        wmap_data, constants, "Traditional", 10000)
    all_results["WMAP_Traditional"] = wmap_trad
    
    print("Running Traditional coherence method on Planck data...")
    planck_trad = run_phase_alignment_test_internal(
        planck_data, constants, "Traditional", 10000)
    all_results["Planck_Traditional"] = planck_trad
    
    # Save results
    save_results(wmap_trad, "WMAP", "Traditional", output_dir)
    save_results(planck_trad, "Planck", "Traditional", output_dir)
    
    # Create visualizations
    create_visualization(wmap_trad, "WMAP", "Traditional", output_dir)
    create_visualization(planck_trad, "Planck", "Traditional", output_dir)
    
    # Run PLV method tests
    print("Running PLV coherence method on WMAP data...")
    wmap_plv = run_phase_alignment_test_internal(
        wmap_data, constants, "PLV", 10000)
    all_results["WMAP_PLV"] = wmap_plv
    
    print("Running PLV coherence method on Planck data...")
    planck_plv = run_phase_alignment_test_internal(
        planck_data, constants, "PLV", 10000)
    all_results["Planck_PLV"] = planck_plv
    
    # Save results
    save_results(wmap_plv, "WMAP", "PLV", output_dir)
    save_results(planck_plv, "Planck", "PLV", output_dir)
    
    # Create visualizations
    create_visualization(wmap_plv, "WMAP", "PLV", output_dir)
    create_visualization(planck_plv, "Planck", "PLV", output_dir)
    
    # Create comparison visualization
    create_comparison_visualization(all_results, output_dir)
    
    return all_results

def create_visualization(results, dataset, method, output_dir):
    """Create visualization of phase alignment test results."""
    plt.figure(figsize=(15, 10))
    
    # Filter constants that have enough data for visualization
    valid_constants = [k for k in results.keys() if 'surrogate_mean' in results[k]]
    
    if not valid_constants:
        print("No valid constants with sufficient data for visualization. Skipping visualization.")
        return
    
    # Sort constants by coherence value
    constants = sorted(valid_constants, 
                      key=lambda k: results[k]['mean_coherence'], 
                      reverse=True)
    
    # Plot 1: Coherence by constant
    plt.subplot(2, 2, 1)
    coherence_values = [results[k]['mean_coherence'] for k in constants]
    surrogate_means = [results[k]['surrogate_mean'] for k in constants]
    
    x = range(len(constants))
    bar_width = 0.35
    
    plt.bar([i - bar_width/2 for i in x], coherence_values, bar_width, label='Actual')
    plt.bar([i + bar_width/2 for i in x], surrogate_means, bar_width, label='Surrogate Mean')
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Mean Phase Coherence')
    plt.title('Phase Coherence by Mathematical Constant')
    plt.xticks(x, constants)
    plt.xticks(rotation=45)
    plt.legend()
    
    # Plot 2: Z-scores by constant
    plt.subplot(2, 2, 2)
    z_scores = [results[k]['z_score'] for k in constants]
    
    plt.bar(constants, z_scores)
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Z-score')
    plt.title('Statistical Significance (Z-score)')
    plt.axhline(y=1.96, color='r', linestyle='--', label='p=0.05')
    plt.axhline(y=2.58, color='g', linestyle='--', label='p=0.01')
    plt.legend()
    
    # Plot 3: Number of scale pairs by constant
    plt.subplot(2, 2, 3)
    num_pairs = [results[k]['num_pairs'] for k in constants]
    
    plt.bar(constants, num_pairs)
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Number of Scale Pairs')
    plt.title('Number of Scale Pairs by Constant')
    plt.xticks(rotation=45)
    
    # Plot 4: Summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Get phi results if available
    phi_results = results.get('phi', {})
    phi_coherence = phi_results.get('mean_coherence', 0)
    phi_p_value = phi_results.get('p_value', 1.0)
    phi_z_score = phi_results.get('z_score', 0)
    phi_num_pairs = phi_results.get('num_pairs', 0)
    
    # Get highest coherence
    highest_const = results.get('summary', {}).get('highest_coherence_constant', 'None')
    highest_value = results.get('summary', {}).get('highest_coherence_value', 0)
    
    summary_text = """
    Phase Alignment Test Summary
    Dataset: %s
    Method: %s coherence
    
    Highest Coherence: %s (%.6f)
    
    Golden Ratio (Phi) Results:
    - Coherence: %.6f
    - P-value: %.6f
    - Z-score: %.4f
    - Number of Scale Pairs: %d
    """ % (
        dataset,
        method,
        highest_const,
        highest_value,
        phi_coherence,
        phi_p_value,
        phi_z_score,
        phi_num_pairs
    )
    
    plt.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "%s/phase_alignment_%s_%s_%s.png" % (output_dir, dataset, method, timestamp)
    plt.savefig(filename)
    print("Visualization saved to %s" % filename)
    plt.close()

def save_results(results, dataset, method, output_dir):
    """Save results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = "%s/phase_alignment_%s_%s_%s.json" % (output_dir, dataset, method, timestamp)
    
    # Create a serializable version of the results
    serializable_results = {
        'summary': results['summary'],
        'constants': {}
    }
    
    for name, result in results['results'].items():
        # Convert boolean to integer (0 or 1) for JSON serialization in Python 2.7
        is_significant = 1 if result['significant'] else 0
        
        serializable_results['constants'][name] = {
            'constant': result['constant'],
            'mean_coherence': result['mean_coherence'],
            'surrogate_mean': result['surrogate_mean'],
            'surrogate_std': result['surrogate_std'],
            'p_value': result['p_value'],
            'z_score': result['z_score'],
            'significant': is_significant,  # Use integer instead of boolean
            'num_pairs': result['num_pairs'],
            # Don't include large arrays like surrogate_values
        }
    
    # Save to file
    with open(result_file, "w") as f:
        import json
        json.dump(serializable_results, f, indent=2)
    
    print("Results saved to %s" % result_file)

def main():
    """Run the enhanced phase alignment test."""
    print("Starting Enhanced Phase Alignment Test...")
    
    # Create a timestamp for the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output directory with timestamp
    output_dir = os.path.join(cosmic_dir, "results", f"phase_alignment_{timestamp}")
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            # Handle case where directory already exists
            pass
    
    print(f"Results will be saved to: {output_dir}")
    
    # Define constants to test
    constants = [PHI, PI, E, SQRT2, SQRT3, LN2]
    
    # Define datasets
    datasets = ["WMAP", "Planck"]
    
    # Define methods
    methods = ["Traditional", "PLV"]
    
    # Number of simulations - using 10,000 for robust statistical significance
    num_simulations = 10000
    print(f"Running with {num_simulations} surrogate simulations for statistical validation")
    
    # Store all results
    all_results = {}
    
    # Run tests for each dataset and method
    for dataset in datasets:
        all_results[dataset] = {}
        
        # Load power spectrum data
        power_spectrum = load_power_spectrum(dataset)
        if power_spectrum is None:
            print(f"Error: Could not load power spectrum data for {dataset}")
            continue
        
        for method in methods:
            print(f"Running {dataset} test with {method} method...")
            
            # Run the test
            results = run_phase_alignment_test_internal(
                power_spectrum=power_spectrum,
                constants=constants,
                coherence_method=method,
                num_simulations=num_simulations
            )
            
            all_results[dataset][method] = results
            
            # Create visualizations
            create_visualization(results['results'], dataset, method, output_dir)
            
            # Save results
            save_results(results, dataset, method, output_dir)
            
            print(f"{dataset} test with {method} method completed.")
    
    # Create comparison visualization
    create_comparison_visualization(all_results, output_dir)
    
    print(f"Enhanced Phase Alignment Test completed. Results saved to {output_dir}")
    
    # Create a simple summary file
    summary_file = os.path.join(output_dir, "phase_alignment_summary.txt")
    with open(summary_file, "w") as f:
        f.write("Phase Alignment Test Results Summary\n")
        f.write("==================================\n\n")
        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of surrogate simulations: {num_simulations}\n\n")
        
        for dataset in datasets:
            if dataset not in all_results:
                continue
                
            f.write(f"{dataset} Dataset Results:\n")
            f.write("-----------------------\n")
            
            for method in methods:
                if method not in all_results[dataset]:
                    continue
                    
                results = all_results[dataset][method]
                summary = results['summary']
                
                f.write(f"\n{method} Coherence Method:\n")
                
                for constant in constants:
                    const_name = constant_name(constant)
                    if const_name in summary:
                        const_summary = summary[const_name]
                        f.write(f"  {const_name}:\n")
                        f.write(f"    Mean coherence: {const_summary['mean_coherence']:.6f}\n")
                        f.write(f"    p-value: {const_summary['p_value']:.6f}\n")
                        f.write(f"    Significant: {'Yes' if const_summary['significant'] else 'No'}\n")
                        
                f.write("\n")
            
            f.write("\n")
    
    print(f"Summary saved to {summary_file}")

def constant_name(value):
    """Return a human-readable name for a mathematical constant."""
    if abs(value - PHI) < 1e-6:
        return "phi (Golden Ratio)"
    elif abs(value - PI) < 1e-6:
        return "pi (Pi)"
    elif abs(value - E) < 1e-6:
        return "e (Euler's Number)"
    elif abs(value - SQRT2) < 1e-6:
        return "sqrt(2)"
    elif abs(value - SQRT3) < 1e-6:
        return "sqrt(3)"
    elif abs(value - LN2) < 1e-6:
        return "ln(2)"
    else:
        return "%.6f" % value

if __name__ == "__main__":
    main()
