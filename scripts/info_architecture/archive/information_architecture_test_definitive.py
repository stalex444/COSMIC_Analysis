"""
Definitive implementation of the Information Architecture Test for CMB analysis.
This implementation represents the validated methodology used in the published paper.

Author: Stephanie Alexander
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import datetime
import pickle
from scipy.signal import welch
from scipy.fft import fft, ifft

# Mathematical constants
PHI = 1.618033988749895  # Golden ratio
PI = np.pi
E = np.e
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
LN2 = np.log(2)

# Import healpy only when needed for map data
def import_healpy():
    try:
        import healpy as hp
        return hp
    except ImportError:
        print("Warning: healpy is not installed. Map data processing will not be available.")
        print("To install healpy: pip install healpy")
        return None

def load_cmb_data(filename, data_type='power_spectrum'):
    """
    Load CMB data from file, either power spectrum or map data.
    
    Parameters:
    - filename: Path to the data file
    - data_type: 'power_spectrum' or 'map'
    
    Returns:
    - Loaded data (power spectrum or map)
    """
    try:
        if data_type == 'power_spectrum':
            # Skip header lines (comments starting with #)
            data = np.loadtxt(filename, comments='#')
            # Assuming format is: multipole(l), power(C_l), ...
            # Only use the first two columns regardless of how many there are
            multipoles = data[:, 0].astype(int)
            power = data[:, 1]
            return multipoles, power
        elif data_type == 'map':
            hp = import_healpy()
            if hp is None:
                print("Error: healpy is required for map data processing.")
                return None
            cmb_map = hp.read_map(filename)
            return cmb_map
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None

def preprocess_cmb_data(data, data_type='power_spectrum'):
    """
    Preprocess CMB data for analysis.
    
    Parameters:
    - data: CMB data (power spectrum or map)
    - data_type: 'power_spectrum' or 'map'
    
    Returns:
    - Preprocessed data
    """
    if data_type == 'power_spectrum':
        multipoles, power = data
        
        # Remove any NaN or infinite values
        valid_indices = np.isfinite(power)
        multipoles = multipoles[valid_indices]
        power = power[valid_indices]
        
        # Normalize to zero mean and unit variance
        power = (power - np.mean(power)) / np.std(power)
        
        return multipoles, power
    
    elif data_type == 'map':
        hp = import_healpy()
        if hp is None:
            print("Error: healpy is required for map data processing.")
            return None
            
        cmb_map = data
        # Apply mask to remove galactic contamination
        mask = hp.mask_bad(cmb_map)
        cmb_map = hp.ma(cmb_map)
        
        # Normalize
        cmb_map = (cmb_map - np.mean(cmb_map)) / np.std(cmb_map)
        
        return cmb_map

def define_hierarchical_scales(data, method='conventional', n_scales=5):
    """
    Define hierarchical scales for analysis.
    
    Parameters:
    - data: CMB data (power spectrum or map)
    - method: Method for defining scales ('conventional', 'logarithmic', 'equal_width')
    - n_scales: Number of hierarchical scales to define
    
    Returns:
    - List of scale boundaries
    """
    if isinstance(data, tuple):
        # Power spectrum data
        multipoles, power = data
        
        if method == 'conventional':
            # Use conventional CMB multipole boundaries
            return [
                (2, 10),      # Very large scales
                (11, 30),     # Large scales
                (31, 100),    # Medium-large scales
                (101, 300),   # Medium scales
                (301, 1000)   # Small scales
            ]
        
        elif method == 'logarithmic':
            # Define logarithmically spaced scales
            l_min, l_max = np.min(multipoles), np.max(multipoles)
            log_bounds = np.logspace(np.log10(max(l_min, 2)), np.log10(l_max), n_scales+1)
            return [(int(log_bounds[i]), int(log_bounds[i+1])) for i in range(n_scales)]
        
        elif method == 'equal_width':
            # Define equally spaced scales
            l_min, l_max = np.min(multipoles), np.max(multipoles)
            bounds = np.linspace(l_min, l_max, n_scales+1)
            return [(int(bounds[i]), int(bounds[i+1])) for i in range(n_scales)]
    
    else:
        # Map data
        # For map data, define scale ranges based on multipoles
        if method == 'conventional':
            return [
                (2, 10),      # Very large scales
                (11, 30),     # Large scales
                (31, 100),    # Medium-large scales
                (101, 300),   # Medium scales
                (301, 1000)   # Small scales
            ]
        # Other methods would be implemented similarly

def calculate_architecture_score(data, scales, constant, data_type='power_spectrum'):
    """
    Calculate architecture score for a mathematical constant.
    
    Parameters:
    - data: CMB data (power spectrum or map)
    - scales: List of scale boundaries
    - constant: Mathematical constant value
    - data_type: 'power_spectrum' or 'map'
    
    Returns:
    - Architecture score (0-1)
    """
    if data_type == 'power_spectrum':
        multipoles, power = data
        
        # Calculate scores for pairs of scales
        pair_scores = []
        scale_midpoints = [np.mean(scale) for scale in scales]
        
        for i in range(len(scales) - 1):
            for j in range(i + 1, len(scales)):
                scale1 = scale_midpoints[i]
                scale2 = scale_midpoints[j]
                
                # Calculate ratio
                ratio = max(scale1, scale2) / min(scale1, scale2)
                
                # Calculate how close the ratio is to the constant
                similarity = 1 - min(abs(ratio - constant) / constant, 1)
                
                # Extract power within these scales
                power1 = extract_scale_power(multipoles, power, scales[i])
                power2 = extract_scale_power(multipoles, power, scales[j])
                
                # Calculate correlation between these scales
                if len(power1) > 0 and len(power2) > 0:
                    # Match lengths for correlation
                    min_len = min(len(power1), len(power2))
                    if min_len > 1:
                        corr, _ = stats.pearsonr(power1[:min_len], power2[:min_len])
                        # Weight correlation by similarity to constant
                        weighted_score = similarity * abs(corr)
                        pair_scores.append(weighted_score)
        
        # Average scores across all scale pairs
        if pair_scores:
            architecture_score = np.mean(pair_scores)
        else:
            architecture_score = 0
            
    elif data_type == 'map':
        # Implementation for map data would go here
        # This would involve spherical harmonic decomposition
        architecture_score = 0  # Placeholder
    
    return architecture_score

def extract_scale_power(multipoles, power, scale_range):
    """
    Extract power spectrum values within a specific scale range.
    
    Parameters:
    - multipoles: Array of multipole values
    - power: Array of power spectrum values
    - scale_range: Tuple (min_scale, max_scale)
    
    Returns:
    - Array of power values within the scale range
    """
    min_scale, max_scale = scale_range
    mask = (multipoles >= min_scale) & (multipoles <= max_scale)
    return power[mask]

def generate_surrogate(data, data_type='power_spectrum', method='phase_randomization'):
    """
    Generate surrogate data for statistical validation.
    
    Parameters:
    - data: CMB data (power spectrum or map)
    - data_type: 'power_spectrum' or 'map'
    - method: Method for surrogate generation ('phase_randomization', 'bootstrap')
    
    Returns:
    - Surrogate dataset
    """
    if data_type == 'power_spectrum':
        multipoles, power = data
        
        if method == 'phase_randomization':
            # Perform Fourier transform
            fft_vals = fft(power)
            
            # Preserve amplitudes but randomize phases
            amplitudes = np.abs(fft_vals)
            phases = np.angle(fft_vals)
            
            # Generate random phases while preserving conjugate symmetry
            random_phases = np.random.uniform(0, 2*np.pi, len(phases))
            random_phases[0] = 0  # DC component has zero phase
            if len(phases) % 2 == 0:
                random_phases[len(phases)//2] = 0  # Nyquist frequency has zero phase
            
            # Ensure conjugate symmetry for real output
            n = len(phases)
            midpoint = n // 2
            
            # For even-length signals
            if n % 2 == 0:
                random_phases[1:midpoint] = random_phases[1:midpoint]  # Redundant, but keeping for clarity
                random_phases[midpoint+1:] = -np.flip(random_phases[1:midpoint])
            # For odd-length signals
            else:
                random_phases[1:midpoint+1] = random_phases[1:midpoint+1]  # Redundant, but keeping for clarity
                random_phases[midpoint+1:] = -np.flip(random_phases[1:midpoint+1])
            
            # Reconstruct with original amplitudes but random phases
            fft_surrogate = amplitudes * np.exp(1j * random_phases)
            
            # Inverse FFT to get surrogate time series
            surrogate_power = np.real(ifft(fft_surrogate))
            
            return multipoles, surrogate_power
        
        elif method == 'bootstrap':
            # Simple bootstrap resampling with replacement
            indices = np.random.choice(len(power), len(power), replace=True)
            surrogate_power = power[indices]
            
            return multipoles, surrogate_power
    
    elif data_type == 'map':
        # Implementation for map data would go here
        pass

def analyze_layer_specialization(data, scales, constants, data_type='power_spectrum'):
    """
    Analyze which constants specialize in organizing which layers.
    
    Parameters:
    - data: CMB data
    - scales: List of scale boundaries
    - constants: Dictionary of mathematical constants
    - data_type: 'power_spectrum' or 'map'
    
    Returns:
    - Dictionary with layer specialization results
    """
    specialization = {}
    
    # Analyze each scale layer
    for i, scale in enumerate(scales):
        scale_specialization = {}
        
        # Score each constant for this layer
        max_score = 0
        max_constant = None
        
        for name, value in constants.items():
            # For each layer, calculate how well the constant organizes it
            layer_score = calculate_layer_organization(data, scale, value, data_type)
            scale_specialization[name] = layer_score
            
            if layer_score > max_score:
                max_score = layer_score
                max_constant = name
        
        # Calculate ratio of specialization (how much better the best constant is)
        avg_other_scores = np.mean([score for name, score in scale_specialization.items() if name != max_constant])
        specialization_ratio = max_score / avg_other_scores if avg_other_scores > 0 else 1
        
        specialization[f"Scale_{i+1}"] = {
            'range': scale,
            'dominant_constant': max_constant,
            'specialization_ratio': specialization_ratio,
            'all_scores': scale_specialization
        }
    
    # Analyze interlayer connections
    for i in range(len(scales) - 1):
        for j in range(i + 1, len(scales)):
            scale1 = scales[i]
            scale2 = scales[j]
            
            connection_specialization = {}
            max_score = 0
            max_constant = None
            
            for name, value in constants.items():
                # Calculate how well each constant organizes the connection
                connection_score = calculate_interlayer_connection(data, scale1, scale2, value, data_type)
                connection_specialization[name] = connection_score
                
                if connection_score > max_score:
                    max_score = connection_score
                    max_constant = name
            
            # Calculate ratio of specialization
            avg_other_scores = np.mean([score for name, score in connection_specialization.items() if name != max_constant])
            specialization_ratio = max_score / avg_other_scores if avg_other_scores > 0 else 1
            
            specialization[f"Connection_{i+1}_{j+1}"] = {
                'scales': (scale1, scale2),
                'dominant_constant': max_constant,
                'specialization_ratio': specialization_ratio,
                'all_scores': connection_specialization
            }
    
    return specialization

def calculate_layer_organization(data, scale, constant, data_type='power_spectrum'):
    """
    Calculate how well a constant organizes a specific layer.
    
    Parameters:
    - data: CMB data
    - scale: Scale boundaries
    - constant: Mathematical constant value
    - data_type: 'power_spectrum' or 'map'
    
    Returns:
    - Organization score for this layer (0-1)
    """
    if data_type == 'power_spectrum':
        multipoles, power = data
        
        # Extract power within this scale
        scale_power = extract_scale_power(multipoles, power, scale)
        
        if len(scale_power) < 2:
            return 0
        
        # Calculate autocorrelation and find peaks
        autocorr = np.correlate(scale_power, scale_power, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i)
        
        if not peaks:
            return 0
        
        # Calculate ratios between successive peaks
        peak_ratios = []
        for i in range(len(peaks) - 1):
            ratio = peaks[i+1] / peaks[i]
            peak_ratios.append(ratio)
        
        if not peak_ratios:
            return 0
        
        # Calculate similarity to constant
        similarities = [1 - min(abs(ratio - constant) / constant, 1) for ratio in peak_ratios]
        
        # Average similarity is the organization score
        organization_score = np.mean(similarities)
        
        return organization_score
    
    elif data_type == 'map':
        # Implementation for map data would go here
        return 0  # Placeholder

def calculate_interlayer_connection(data, scale1, scale2, constant, data_type='power_spectrum'):
    """
    Calculate how well a constant organizes the connection between two layers.
    
    Parameters:
    - data: CMB data
    - scale1: First scale boundaries
    - scale2: Second scale boundaries
    - constant: Mathematical constant value
    - data_type: 'power_spectrum' or 'map'
    
    Returns:
    - Connection score (0-1)
    """
    if data_type == 'power_spectrum':
        multipoles, power = data
        
        # Extract power within these scales
        power1 = extract_scale_power(multipoles, power, scale1)
        power2 = extract_scale_power(multipoles, power, scale2)
        
        if len(power1) < 2 or len(power2) < 2:
            return 0
        
        # Calculate cross-correlation
        min_len = min(len(power1), len(power2))
        power1 = power1[:min_len]
        power2 = power2[:min_len]
        
        cross_corr = np.correlate(power1, power2, mode='full')
        
        # Normalize cross-correlation
        norm = np.sqrt(np.sum(power1**2) * np.sum(power2**2))
        if norm > 0:
            cross_corr /= norm
        
        # Find peak in cross-correlation
        peak_idx = np.argmax(np.abs(cross_corr))
        lag = peak_idx - len(power1) + 1
        
        # Calculate ratio of scale midpoints
        scale1_mid = np.mean(scale1)
        scale2_mid = np.mean(scale2)
        scale_ratio = max(scale1_mid, scale2_mid) / min(scale1_mid, scale2_mid)
        
        # Calculate similarity between peak lag, scale ratio, and constant
        lag_similarity = 1 - min(abs(abs(lag) - constant) / constant, 1) if lag != 0 else 0
        scale_similarity = 1 - min(abs(scale_ratio - constant) / constant, 1)
        
        # Combined connection score
        connection_score = 0.5 * lag_similarity + 0.5 * scale_similarity
        
        return connection_score
    
    elif data_type == 'map':
        # Implementation for map data would go here
        return 0  # Placeholder

def run_information_architecture_test(data_file, data_type='power_spectrum', constants=None, 
                                      n_simulations=10000, scale_method='conventional',
                                      output_dir="../results/information_architecture"):
    """
    Runs the definitive Information Architecture Test on CMB data.
    
    Parameters:
    - data_file: Path to the CMB data file
    - data_type: Type of data ('power_spectrum' or 'map')
    - constants: Dictionary of mathematical constants to test (default includes standard set)
    - n_simulations: Number of Monte Carlo simulations for statistical validation
    - scale_method: Method for defining hierarchical scales
    - output_dir: Directory to save results
    
    Returns:
    - Dictionary with comprehensive test results
    """
    # Set default constants if none provided
    if constants is None:
        constants = {
            'phi': PHI,
            'pi': PI,
            'e': E,
            'sqrt2': SQRT2,
            'sqrt3': SQRT3,
            'ln2': LN2
        }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print(f"Loading data from {data_file}...")
    data = load_cmb_data(data_file, data_type)
    if data is None:
        print(f"Failed to load data from {data_file}. Aborting test.")
        return None
    
    print("Preprocessing data...")
    data = preprocess_cmb_data(data, data_type)
    
    # Define hierarchical scales
    print(f"Defining hierarchical scales using {scale_method} method...")
    scales = define_hierarchical_scales(data, method=scale_method)
    
    # Calculate architecture scores for each constant
    results = {}
    for name, value in constants.items():
        print(f"Testing {name}...")
        
        # Calculate actual architecture score
        actual_score = calculate_architecture_score(data, scales, value, data_type)
        
        # Generate surrogate datasets and calculate their scores
        surrogate_scores = []
        print("STARTING ACTUAL SIMULATIONS - THIS SHOULD TAKE HOURS, NOT MINUTES")
        start_time = datetime.datetime.now()
        for i in range(n_simulations):
            if i % 100 == 0:
                current_time = datetime.datetime.now()
                print(f"  Simulation {i}/{n_simulations}... - TIME: {current_time.strftime('%H:%M:%S')}")
            surrogate = generate_surrogate(data, data_type)
            score = calculate_architecture_score(surrogate, scales, value, data_type)
            surrogate_scores.append(score)
        end_time = datetime.datetime.now()
        print(f"COMPLETED ALL {n_simulations} SIMULATIONS - TIME: {end_time.strftime('%H:%M:%S')}")
        print(f"TOTAL RUNTIME: {end_time - start_time}")
        
        # Calculate p-value (one-tailed test)
        p_value = sum(s >= actual_score for s in surrogate_scores) / n_simulations
        
        # Calculate statistical metrics
        mean_surrogate = np.mean(surrogate_scores)
        std_surrogate = np.std(surrogate_scores)
        z_score = (actual_score - mean_surrogate) / std_surrogate if std_surrogate > 0 else 0
        
        results[name] = {
            'architecture_score': actual_score,
            'mean_surrogate': mean_surrogate,
            'std_surrogate': std_surrogate,
            'p_value': p_value,
            'z_score': z_score,
            'significant': p_value < 0.05,
            'surrogate_scores': surrogate_scores
        }
    
    # Analyze layer specialization
    print("Analyzing layer specialization...")
    layer_specialization = analyze_layer_specialization(data, scales, constants, data_type)
    results['layer_specialization'] = layer_specialization
    
    # Calculate phi-optimality for each constant
    best_score = max([results[name]['architecture_score'] for name in constants])
    worst_score = min([results[name]['architecture_score'] for name in constants])
    score_range = best_score - worst_score
    
    if 'phi' in results and score_range > 0:
        for name in constants:
            phi_score = results['phi']['architecture_score']
            constant_score = results[name]['architecture_score']
            phi_optimality = (phi_score - worst_score) / score_range
            results[name]['phi_optimality'] = phi_optimality
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{output_dir}/information_architecture_results_{timestamp}.pkl"
    print(f"Saving results to {result_file}...")
    
    with open(result_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Generate summary report
    summary_file = f"{output_dir}/summary_{timestamp}.txt"
    generate_summary_report(results, constants, scales, summary_file)
    
    # Create visualizations
    visualization_prefix = f"{output_dir}/visualization_{timestamp}"
    create_visualizations(results, constants, visualization_prefix)
    
    return results

def generate_summary_report(results, constants, scales, output_file):
    """
    Generate a human-readable summary report of test results.
    
    Parameters:
    - results: Dictionary with test results
    - constants: Dictionary of mathematical constants tested
    - scales: List of scale boundaries
    - output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        f.write("Information Architecture Test Results\n")
        f.write("===================================\n\n")
        
        f.write("Mathematical Constants Analysis:\n")
        f.write("-------------------------------\n")
        
        # Sort constants by architecture score
        sorted_constants = sorted(constants.keys(), 
                                 key=lambda x: results[x]['architecture_score'],
                                 reverse=True)
        
        for name in sorted_constants:
            result = results[name]
            f.write(f"\n{name.upper()}:\n")
            f.write(f"Architecture Score: {result['architecture_score']:.6f}\n")
            f.write(f"Mean Surrogate Score: {result['mean_surrogate']:.6f}\n")
            f.write(f"Standard Deviation: {result['std_surrogate']:.6f}\n")
            f.write(f"Z-score: {result['z_score']:.4f}\n")
            f.write(f"P-value: {result['p_value']:.6f}\n")
            f.write(f"Significant: {'Yes' if result['significant'] else 'No'}\n")
            
            if 'phi_optimality' in result:
                f.write(f"Phi-optimality: {result['phi_optimality']:.6f}\n")
        
        f.write("\n\nLayer Specialization Analysis:\n")
        f.write("-----------------------------\n")
        
        # Report layer specialization
        for layer, spec in results['layer_specialization'].items():
            if layer.startswith("Scale_"):
                f.write(f"\n{layer}: {spec['range']}\n")
                f.write(f"Dominant Constant: {spec['dominant_constant']}\n")
                f.write(f"Specialization Ratio: {spec['specialization_ratio']:.4f}\n")
        
        f.write("\n\nInterlayer Connection Analysis:\n")
        f.write("------------------------------\n")
        
        # Report interlayer connections
        for connection, spec in results['layer_specialization'].items():
            if connection.startswith("Connection_"):
                f.write(f"\n{connection}: {spec['scales']}\n")
                f.write(f"Dominant Constant: {spec['dominant_constant']}\n")
                f.write(f"Specialization Ratio: {spec['specialization_ratio']:.4f}\n")

def create_visualizations(results, constants, output_prefix):
    """
    Create visualizations of test results.
    
    Parameters:
    - results: Dictionary with test results
    - constants: Dictionary of mathematical constants tested
    - output_prefix: Prefix for output files
    """
    # Plot 1: Architecture scores by constant
    plt.figure(figsize=(10, 6))
    
    # Sort constants by architecture score
    sorted_constants = sorted(constants.keys(), 
                             key=lambda x: results[x]['architecture_score'],
                             reverse=True)
    
    scores = [results[name]['architecture_score'] for name in sorted_constants]
    surrogate_means = [results[name]['mean_surrogate'] for name in sorted_constants]
    
    # Create grouped bar chart
    x = np.arange(len(sorted_constants))
    width = 0.35
    
    plt.bar(x - width/2, scores, width, label='Actual Score')
    plt.bar(x + width/2, surrogate_means, width, label='Mean Surrogate Score')
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Architecture Score')
    plt.title('Information Architecture Scores by Mathematical Constant')
    plt.xticks(x, sorted_constants)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_scores.png")
    
    # Plot 2: Statistical significance (z-scores)
    plt.figure(figsize=(10, 6))
    
    z_scores = [results[name]['z_score'] for name in sorted_constants]
    
    plt.bar(x, z_scores)
    plt.axhline(y=1.96, color='r', linestyle='--', label='p=0.05 (two-tailed)')
    plt.axhline(y=1.645, color='g', linestyle='--', label='p=0.05 (one-tailed)')
    
    plt.xlabel('Mathematical Constant')
    plt.ylabel('Z-score')
    plt.title('Statistical Significance by Mathematical Constant')
    plt.xticks(x, sorted_constants)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_significance.png")
    
    # Plot 3: Layer specialization
    plt.figure(figsize=(12, 8))
    
    # Extract scales and their dominant constants
    scales = [key for key in results['layer_specialization'].keys() if key.startswith("Scale_")]
    dominant_constants = [results['layer_specialization'][scale]['dominant_constant'] for scale in scales]
    specialization_ratios = [results['layer_specialization'][scale]['specialization_ratio'] for scale in scales]
    
    # Create color map for constants
    constant_colors = {}
    colors = ['gold', 'blue', 'green', 'red', 'purple', 'brown']
    
    for i, name in enumerate(constants.keys()):
        constant_colors[name] = colors[i % len(colors)]
    
    # Plot with colored bars based on dominant constant
    bar_colors = [constant_colors.get(const, 'gray') for const in dominant_constants]
    
    plt.bar(scales, specialization_ratios, color=bar_colors)
    plt.xlabel('Scale Layer')
    plt.ylabel('Specialization Ratio')
    plt.title('Layer Specialization by Mathematical Constant')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0), 1, 1, color=color, label=name) 
                       for name, color in constant_colors.items()]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_layer_specialization.png")

def main():
    """
    Main function to run the Information Architecture Test.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Information Architecture Test on CMB data')
    parser.add_argument('--data_file', type=str, required=True, help='Path to CMB data file')
    parser.add_argument('--data_type', type=str, default='power_spectrum', choices=['power_spectrum', 'map'],
                       help='Type of CMB data')
    parser.add_argument('--n_simulations', type=int, default=10000, help='Number of Monte Carlo simulations')
    parser.add_argument('--scale_method', type=str, default='conventional', 
                       choices=['conventional', 'logarithmic', 'equal_width'],
                       help='Method for defining hierarchical scales')
    parser.add_argument('--output_dir', type=str, default="../results/information_architecture",
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run the test
    results = run_information_architecture_test(
        data_file=args.data_file,
        data_type=args.data_type,
        n_simulations=args.n_simulations,
        scale_method=args.scale_method,
        output_dir=args.output_dir
    )
    
    if results:
        print("Information Architecture Test completed successfully.")
    else:
        print("Information Architecture Test failed.")

if __name__ == "__main__":
    main()
