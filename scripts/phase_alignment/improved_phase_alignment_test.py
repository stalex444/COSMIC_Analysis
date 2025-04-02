import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime

# Constants
PHI = 1.618033988749895  # Golden ratio
PI = np.pi

def load_power_spectrum(filename):
    """
    Load CMB power spectrum data from text file.
    
    The file format is expected to be space-delimited with:
    - Comment lines starting with #
    - Column 1: multipole moment (l)
    - Column 2: power spectrum value (C_l)
    """
    # Load data, skipping comment lines
    data = np.loadtxt(filename, comments='#')
    
    # Extract multipoles (first column) and power (second column)
    multipoles = data[:, 0].astype(int)
    power = data[:, 1]
    
    return multipoles, power

def reconstruct_phases_from_correlations(power_spectrum, multipoles, method='statistical'):
    """
    Reconstruct phase information using statistical properties of the power spectrum.
    
    Instead of random phases, this uses the correlations between different multipoles
    to estimate more realistic phase relationships.
    
    Parameters:
    - power_spectrum: Power spectrum values C_l
    - multipoles: Corresponding multipole values l
    - method: Method to use for phase reconstruction ('statistical' or 'gradient')
    
    Returns:
    - Dictionary mapping multipole ranges to phase arrays
    """
    phase_info = {}
    
    # Define multipole ranges
    l_ranges = [
        (2, 10),      # Very large scales
        (11, 30),     # Large scales
        (31, 100),    # Medium-large scales
        (101, 300),   # Medium scales
        (301, 500),   # Medium-small scales
        (501, 1000),  # Small scales
    ]
    
    if method == 'statistical':
        # Use power spectrum correlations to guide phase relationships
        for l_min, l_max in l_ranges:
            range_key = "%d-%d" % (l_min, l_max)
            # Filter to relevant multipoles
            mask = (multipoles >= l_min) & (multipoles <= l_max)
            if not np.any(mask):
                continue
                
            range_power = power_spectrum[mask]
            range_multipoles = multipoles[mask]
            
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
            
    elif method == 'gradient':
        # Use power spectrum gradient to inform phase relationships
        for l_min, l_max in l_ranges:
            range_key = "%d-%d" % (l_min, l_max)
            # Filter to relevant multipoles
            mask = (multipoles >= l_min) & (multipoles <= l_max)
            if not np.any(mask):
                continue
                
            range_power = power_spectrum[mask]
            range_multipoles = multipoles[mask]
            
            # Calculate normalized power gradient
            if len(range_power) > 1:
                gradient = np.gradient(range_power)
                gradient = gradient / np.max(np.abs(gradient))
                
                # Use gradient to create phase relationships
                phases = np.cumsum(gradient) * PI
                phases = phases % (2*PI)
            else:
                phases = np.array([0.0])
            
            phase_info[range_key] = phases
    
    return phase_info

def compute_phase_coherence(phases1, phases2):
    """
    Compute phase coherence between two sets of phases.
    Returns a value between 0 (no coherence) and 1 (perfect coherence).
    """
    # Ensure equal length comparison
    min_length = min(len(phases1), len(phases2))
    if min_length == 0:
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

def calculate_constant_ratio_coherence(phase_info, constant_value, tolerance=0.15):
    """
    Calculate phase coherence between scales related by a specific constant.
    
    Parameters:
    - phase_info: Dictionary mapping scale ranges to phase arrays
    - constant_value: The mathematical constant to test (e.g., PHI, PI)
    - tolerance: Tolerance for ratio comparison
    
    Returns:
    - Dictionary with coherence scores and pairs
    """
    scales = sorted([tuple(map(int, k.split('-'))) for k in phase_info.keys()])
    coherence_scores = {}
    pairs = []
    
    # Find scale pairs related by the constant
    for i, (min1, max1) in enumerate(scales[:-1]):
        for (min2, max2) in scales[i+1:]:
            # Calculate effective scale ratio (using midpoints)
            scale1 = (min1 + max1) / 2
            scale2 = (min2 + max2) / 2
            ratio = max(scale1, scale2) / min(scale1, scale2)
            
            # Check if close to the constant
            if abs(ratio - constant_value) < tolerance:
                pairs.append(((min1, max1), (min2, max2)))
                
                # Calculate coherence
                key1 = "%d-%d" % (min1, max1)
                key2 = "%d-%d" % (min2, max2)
                
                if key1 in phase_info and key2 in phase_info:
                    coherence = compute_phase_coherence(phase_info[key1], phase_info[key2])
                    pair_key = "%s_x_%s" % (key1, key2)
                    coherence_scores[pair_key] = coherence
    
    return {
        'pairs': pairs,
        'coherence_scores': coherence_scores,
        'mean_coherence': np.mean(list(coherence_scores.values())) if coherence_scores else 0,
        'num_pairs': len(pairs)
    }

def run_phase_alignment_test(power_file, reconstruction_method='statistical', n_surrogates=1000):
    """
    Run phase alignment test with improved phase reconstruction.
    
    Parameters:
    - power_file: Path to power spectrum data file
    - reconstruction_method: Method for phase reconstruction ('statistical' or 'gradient')
    - n_surrogates: Number of surrogate datasets to generate
    
    Returns:
    - Dictionary with test results
    """
    print("Loading power spectrum from %s..." % power_file)
    multipoles, power = load_power_spectrum(power_file)
    
    print("Reconstructing phases using %s method..." % reconstruction_method)
    phase_info = reconstruct_phases_from_correlations(power, multipoles, method=reconstruction_method)
    
    # Test mathematical constants
    constants = {
        "phi": PHI,        # Golden ratio
        "e": 2.71828,      # e
        "pi": PI,          # pi
        "sqrt2": 1.41421,  # sqrt(2)
        "sqrt3": 1.73205,  # sqrt(3)
        "ln2": 0.69315     # ln(2)
    }
    
    constant_results = {}
    for name, value in constants.items():
        print("Testing coherence for %s..." % name)
        result = calculate_constant_ratio_coherence(phase_info, value)
        constant_results[name] = result
    
    # Generate surrogate data and calculate statistics
    print("Generating %d surrogate datasets..." % n_surrogates)
    surrogate_coherences = {name: [] for name in constants.keys()}
    
    for i in range(n_surrogates):
        if i % 100 == 0:
            print("Processing surrogate %d/%d..." % (i, n_surrogates))
        
        # Create surrogate with same power but randomized phases
        surrogate_phase_info = {}
        for range_key, phases in phase_info.items():
            # Generate randomized phases for this range
            surrogate_phase_info[range_key] = np.random.uniform(0, 2*PI, size=len(phases))
        
        # Calculate coherence for each constant in this surrogate
        for name, value in constants.items():
            result = calculate_constant_ratio_coherence(surrogate_phase_info, value)
            surrogate_coherences[name].append(result['mean_coherence'])
    
    # Calculate statistics for each constant
    print("Calculating statistics...")
    statistics = {}
    for name in constants.keys():
        # Convert to numpy array for calculations
        surrogate_values = np.array(surrogate_coherences[name])
        
        # Get actual coherence value
        actual_coherence = constant_results[name]['mean_coherence']
        
        # Calculate statistics
        mean_surrogate = np.mean(surrogate_values)
        std_surrogate = np.std(surrogate_values)
        
        # Calculate p-value
        p_value = np.sum(surrogate_values >= actual_coherence) / n_surrogates
        
        # Calculate z-score
        z_score = (actual_coherence - mean_surrogate) / std_surrogate if std_surrogate > 0 else 0
        
        statistics[name] = {
            'actual_coherence': actual_coherence,
            'mean_surrogate': mean_surrogate,
            'std_surrogate': std_surrogate,
            'p_value': p_value,
            'z_score': z_score,
            'num_pairs': constant_results[name]['num_pairs']
        }
    
    # Calculate phi-optimality
    max_coherence = max([stats['actual_coherence'] for stats in statistics.values()])
    min_coherence = min([stats['actual_coherence'] for stats in statistics.values()])
    range_coherence = max_coherence - min_coherence if max_coherence > min_coherence else 1
    
    for name in constants.keys():
        if name == 'phi':
            continue
        statistics[name]['phi_ratio'] = (statistics['phi']['actual_coherence'] - min_coherence) / range_coherence
    
    # Compile results
    results = {
        'constants': constants,
        'constant_results': constant_results,
        'statistics': statistics,
        'surrogate_coherences': surrogate_coherences,
        'phase_info': phase_info,
        'reconstruction_method': reconstruction_method
    }
    
    return results

def save_results(results, output_dir, dataset_name):
    """Save phase alignment test results to file."""
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir)
    except OSError:
        if not os.path.isdir(output_dir):
            raise
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "%s/phase_alignment_%s_%s_%s.txt" % (output_dir, dataset_name, results['reconstruction_method'], timestamp)
    
    with open(filename, "w") as f:
        f.write("Phase Alignment Test Results for %s\n" % dataset_name)
        f.write("Reconstruction Method: %s\n" % results['reconstruction_method'])
        f.write("Run at: %s\n\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        f.write("Results by Mathematical Constant:\n")
        f.write("--------------------------------\n")
        
        for name, stats in results['statistics'].items():
            f.write("\n%s\n" % name.upper())
            f.write("Actual Coherence: %.6f\n" % stats['actual_coherence'])
            f.write("Mean Surrogate Coherence: %.6f\n" % stats['mean_surrogate'])
            f.write("Standard Deviation: %.6f\n" % stats['std_surrogate'])
            f.write("Z-score: %.4f\n" % stats['z_score'])
            f.write("P-value: %.6f\n" % stats['p_value'])
            f.write("Number of Scale Pairs: %d\n" % stats['num_pairs'])
            if name != 'phi':
                f.write("Phi Ratio: %.6f\n" % stats['phi_ratio'])
        
        # Add phi-optimality section
        f.write("\nPhi-Optimality Analysis:\n")
        phi_stats = results['statistics']['phi']
        other_constants = [c for c in results['statistics'].keys() if c != 'phi']
        other_z_scores = [results['statistics'][c]['z_score'] for c in other_constants]
        max_other_z = max(other_z_scores) if other_z_scores else 0
        
        f.write("Phi Z-score: %.4f\n" % phi_stats['z_score'])
        f.write("Max Other Z-score: %.4f\n" % max_other_z)
        
        if max_other_z > 0:
            phi_optimality = phi_stats['z_score'] / max_other_z
            f.write("Phi-optimality (Z-score ratio): %.4f\n" % phi_optimality)
        else:
            f.write("Phi-optimality: N/A (no positive Z-scores for other constants)\n")
    
    print("Results saved to %s" % filename)
    
    # Create visualization
    plot_filename = "%s/phase_alignment_%s_%s_%s.png" % (output_dir, dataset_name, results['reconstruction_method'], timestamp)
    visualize_results(results, plot_filename, dataset_name)
    
    return filename

def visualize_results(results, filename, dataset_name):
    """Generate visualization of phase alignment test results."""
    plt.figure(figsize=(15, 10))
    
    # Create a color map for constants
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    const_colors = {name: colors[i % len(colors)] for i, name in enumerate(results['constants'].keys())}
    
    # Plot 1: Coherence by constant
    plt.subplot(2, 2, 1)
    constants = list(results['statistics'].keys())
    coherences = [results['statistics'][c]['actual_coherence'] for c in constants]
    surrogate_means = [results['statistics'][c]['mean_surrogate'] for c in constants]
    
    x = range(len(constants))
    bar_width = 0.35
    
    plt.bar([i - bar_width/2 for i in x], coherences, bar_width, label='Actual')
    plt.bar([i + bar_width/2 for i in x], surrogate_means, bar_width, label='Surrogate Mean')
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Mean Phase Coherence')
    plt.title('Phase Coherence by Mathematical Constant')
    plt.xticks(x, constants)
    plt.legend()
    
    # Plot 2: Z-scores by constant
    plt.subplot(2, 2, 2)
    z_scores = [results['statistics'][c]['z_score'] for c in constants]
    
    bars = plt.bar(constants, z_scores)
    for i, bar in enumerate(bars):
        bar.set_color(const_colors[constants[i]])
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Z-score')
    plt.title('Statistical Significance (Z-score)')
    plt.axhline(y=1.96, color='r', linestyle='--', label='p=0.05')
    plt.axhline(y=2.58, color='g', linestyle='--', label='p=0.01')
    
    # Plot 3: Pairs by constant
    plt.subplot(2, 2, 3)
    num_pairs = [results['statistics'][c]['num_pairs'] for c in constants]
    
    bars = plt.bar(constants, num_pairs)
    for i, bar in enumerate(bars):
        bar.set_color(const_colors[constants[i]])
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Number of Scale Pairs')
    plt.title('Number of Scale Pairs by Constant')
    
    # Plot 4: Summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Find the constants with the highest z-scores
    sorted_constants = sorted(constants, key=lambda c: results['statistics'][c]['z_score'], reverse=True)
    top_constant = sorted_constants[0]
    
    summary_text = """
    Phase Alignment Test Summary - %s
    Reconstruction Method: %s
    
    Most Significant Constant: %s
    - Z-score: %.4f
    - P-value: %.6f
    - Coherence: %.6f
    
    Golden Ratio (Phi) Results:
    - Z-score: %.4f
    - P-value: %.6f
    - Coherence: %.6f
    - Number of Scale Pairs: %d
    """ % (dataset_name, results['reconstruction_method'], top_constant, results['statistics'][top_constant]['z_score'], results['statistics'][top_constant]['p_value'], results['statistics'][top_constant]['actual_coherence'], results['statistics']['phi']['z_score'], results['statistics']['phi']['p_value'], results['statistics']['phi']['actual_coherence'], results['statistics']['phi']['num_pairs'])
    
    plt.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(filename)
    print("Visualization saved to %s" % filename)

def compare_results(wmap_results, planck_results, output_dir, method):
    """Compare WMAP and Planck phase alignment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "%s/phase_alignment_comparison_%s_%s.txt" % (output_dir, method, timestamp)
    
    with open(filename, "w") as f:
        f.write("Comparison of Phase Alignment Test Results: WMAP vs. Planck\n")
        f.write("Reconstruction Method: %s\n" % method)
        f.write("Generated at: %s\n\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Compare each constant
        for constant in wmap_results['statistics'].keys():
            wmap_stats = wmap_results['statistics'][constant]
            planck_stats = planck_results['statistics'][constant]
            
            f.write("\n%s\n" % constant.upper())
            f.write("Coherence - WMAP: %.6f, Planck: %.6f\n" % (wmap_stats['actual_coherence'], planck_stats['actual_coherence']))
            f.write("Z-score - WMAP: %.4f, Planck: %.4f\n" % (wmap_stats['z_score'], planck_stats['z_score']))
            f.write("P-value - WMAP: %.6f, Planck: %.6f\n" % (wmap_stats['p_value'], planck_stats['p_value']))
            f.write("Pairs - WMAP: %d, Planck: %d\n" % (wmap_stats['num_pairs'], planck_stats['num_pairs']))
            
            # Calculate differences
            coherence_diff = planck_stats['actual_coherence'] - wmap_stats['actual_coherence']
            z_score_diff = planck_stats['z_score'] - wmap_stats['z_score']
            
            f.write("Coherence Difference (Planck - WMAP): %.6f\n" % coherence_diff)
            f.write("Z-score Difference (Planck - WMAP): %.4f\n" % z_score_diff)
        
        # Add scale-dependent analysis
        f.write("\nScale-Dependent Analysis:\n")
        f.write("------------------------\n")
        
        # Identify the dominant constant at each scale in each dataset
        wmap_dominant = find_dominant_constants_by_scale(wmap_results)
        planck_dominant = find_dominant_constants_by_scale(planck_results)
        
        f.write("\nDominant Constants by Scale:\n")
        all_scales = sorted(set(list(wmap_dominant.keys()) + list(planck_dominant.keys())))
        
        for scale in all_scales:
            wmap_const = wmap_dominant.get(scale, "N/A")
            planck_const = planck_dominant.get(scale, "N/A")
            f.write("Scale %s:\n" % scale)
            f.write("  WMAP: %s\n" % wmap_const)
            f.write("  Planck: %s\n" % planck_const)
    
    print("Comparison saved to %s" % filename)
    
    # Create comparison visualization
    vis_filename = "%s/phase_alignment_comparison_%s_%s.png" % (output_dir, method, timestamp)
    create_comparison_visualization(wmap_results, planck_results, vis_filename)

def find_dominant_constants_by_scale(results):
    """Identify which constant dominates at each scale range."""
    dominant_constants = {}
    
    # Extract all scale pairs tested
    all_pairs = {}
    for const_name, const_result in results['constant_results'].items():
        for pair in const_result['pairs']:
            (min1, max1), (min2, max2) = pair
            scale1 = "%d-%d" % (min1, max1)
            scale2 = "%d-%d" % (min2, max2)
            
            if scale1 not in all_pairs:
                all_pairs[scale1] = {}
            if scale2 not in all_pairs:
                all_pairs[scale2] = {}
            
            pair_key = "%s_x_%s" % (scale1, scale2)
            if pair_key in const_result['coherence_scores']:
                coherence = const_result['coherence_scores'][pair_key]
                
                if const_name not in all_pairs[scale1]:
                    all_pairs[scale1][const_name] = []
                all_pairs[scale1][const_name].append(coherence)
                
                if const_name not in all_pairs[scale2]:
                    all_pairs[scale2][const_name] = []
                all_pairs[scale2][const_name].append(coherence)
    
    # Find dominant constant for each scale
    for scale, const_scores in all_pairs.items():
        if not const_scores:
            continue
            
        # Calculate mean coherence for each constant at this scale
        mean_coherences = {const: np.mean(scores) for const, scores in const_scores.items() if scores}
        
        # Find constant with highest coherence
        if mean_coherences:
            dominant_const = max(mean_coherences.items(), key=lambda x: x[1])
            dominant_constants[scale] = "%s (%.4f)" % (dominant_const[0], dominant_const[1])
    
    return dominant_constants

def create_comparison_visualization(wmap_results, planck_results, filename):
    """Create visualization comparing WMAP and Planck results."""
    plt.figure(figsize=(15, 10))
    
    # Get list of constants
    constants = list(wmap_results['statistics'].keys())
    
    # Plot 1: Compare coherence values
    plt.subplot(2, 2, 1)
    wmap_coherences = [wmap_results['statistics'][c]['actual_coherence'] for c in constants]
    planck_coherences = [planck_results['statistics'][c]['actual_coherence'] for c in constants]
    
    x = range(len(constants))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], wmap_coherences, width, label='WMAP')
    plt.bar([i + width/2 for i in x], planck_coherences, width, label='Planck')
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Coherence')
    plt.title('Phase Coherence by Dataset')
    plt.xticks(x, constants)
    plt.legend()
    
    # Plot 2: Compare Z-scores
    plt.subplot(2, 2, 2)
    wmap_zscores = [wmap_results['statistics'][c]['z_score'] for c in constants]
    planck_zscores = [planck_results['statistics'][c]['z_score'] for c in constants]
    
    plt.bar([i - width/2 for i in x], wmap_zscores, width, label='WMAP')
    plt.bar([i + width/2 for i in x], planck_zscores, width, label='Planck')
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Z-score')
    plt.title('Statistical Significance by Dataset')
    plt.xticks(x, constants)
    plt.legend()
    plt.axhline(y=1.96, color='r', linestyle='--', label='p=0.05')
    
    # Plot 3: Scale-dependent dominance
    plt.subplot(2, 2, 3)
    wmap_dominant = find_dominant_constants_by_scale(wmap_results)
    planck_dominant = find_dominant_constants_by_scale(planck_results)
    
    # Count how many times each constant dominates
    wmap_counts = {}
    planck_counts = {}
    
    for scale, const_info in wmap_dominant.items():
        const_name = const_info.split(' ')[0]
        if const_name not in wmap_counts:
            wmap_counts[const_name] = 0
        wmap_counts[const_name] += 1
    
    for scale, const_info in planck_dominant.items():
        const_name = const_info.split(' ')[0]
        if const_name not in planck_counts:
            planck_counts[const_name] = 0
        planck_counts[const_name] += 1
    
    # Plot dominance counts
    all_consts = sorted(set(list(wmap_counts.keys()) + list(planck_counts.keys())))
    x_consts = range(len(all_consts))
    
    wmap_dom_counts = [wmap_counts.get(c, 0) for c in all_consts]
    planck_dom_counts = [planck_counts.get(c, 0) for c in all_consts]
    
    plt.bar([i - width/2 for i in x_consts], wmap_dom_counts, width, label='WMAP')
    plt.bar([i + width/2 for i in x_consts], planck_dom_counts, width, label='Planck')
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Number of Scales Dominated')
    plt.title('Scale Domination by Constant')
    plt.xticks(x_consts, all_consts)
    plt.legend()
    
    # Plot 4: Summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Find the constants with the highest z-scores
    wmap_top = max(constants, key=lambda c: wmap_results['statistics'][c]['z_score'])
    planck_top = max(constants, key=lambda c: planck_results['statistics'][c]['z_score'])
    
    summary_text = """
    Phase Alignment Comparison Summary
    Reconstruction Method: %s
    
    WMAP Results:
    Most Significant Constant: %s
    - Z-score: %.4f
    - P-value: %.6f
    - Coherence: %.6f
    
    Golden Ratio (Phi) in WMAP:
    - Z-score: %.4f
    - P-value: %.6f
    
    Planck Results:
    Most Significant Constant: %s
    - Z-score: %.4f
    - P-value: %.6f
    - Coherence: %.6f
    
    Golden Ratio (Phi) in Planck:
    - Z-score: %.4f
    - P-value: %.6f
    """ % (wmap_results['reconstruction_method'], wmap_top, wmap_results['statistics'][wmap_top]['z_score'], wmap_results['statistics'][wmap_top]['p_value'], wmap_results['statistics'][wmap_top]['actual_coherence'], wmap_results['statistics']['phi']['z_score'], wmap_results['statistics']['phi']['p_value'], planck_top, planck_results['statistics'][planck_top]['z_score'], planck_results['statistics'][planck_top]['p_value'], planck_results['statistics'][planck_top]['actual_coherence'], planck_results['statistics']['phi']['z_score'], planck_results['statistics']['phi']['p_value'])
    
    plt.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(filename)
    print("Comparison visualization saved to %s" % filename)

if __name__ == "__main__":
    # Define output directory
    output_dir = "../results/phase_alignment_test"
    
    # Define paths to power spectrum files - update these to your actual file paths
    wmap_power_file = "../data/wmap_tt_spectrum_9yr_v5.txt"
    planck_power_file = "../data/planck_tt_spectrum_2018.txt"
    
    # Number of surrogate datasets
    n_surrogates = 10000
    
    # Run test with both reconstruction methods
    for method in ['statistical', 'gradient']:
        print("\nRunning Phase Alignment Test on WMAP data using %s method..." % method)
        wmap_results = run_phase_alignment_test(wmap_power_file, reconstruction_method=method, n_surrogates=n_surrogates)
        save_results(wmap_results, output_dir, "WMAP")
        
        print("\nRunning Phase Alignment Test on Planck data using %s method..." % method)
        planck_results = run_phase_alignment_test(planck_power_file, reconstruction_method=method, n_surrogates=n_surrogates)
        save_results(planck_results, output_dir, "Planck")
        
        # Compare WMAP and Planck results
        compare_results(wmap_results, planck_results, output_dir, method)
