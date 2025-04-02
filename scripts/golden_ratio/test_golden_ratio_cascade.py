import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from multiprocessing import Pool, cpu_count
import time
import os
import argparse
from datetime import datetime

# Import data loading functions from existing codebase
def load_wmap_power_spectrum(file_path):
    """Load WMAP CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Mean multipole moment
        power = data[:, 3]  # Power spectrum value
        error = data[:, 4]  # Error
        return ell, power, error
    except Exception as e:
        print("Error loading WMAP power spectrum: {}".format(str(e)))
        return None, None, None

def load_planck_power_spectrum(file_path):
    """Load Planck CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Multipole moment
        power = data[:, 1]  # Power spectrum value
        # Use average of asymmetric error bars as the error
        lower_error = data[:, 2]  # Lower error bound
        upper_error = data[:, 3]  # Upper error bound
        error = (abs(lower_error) + abs(upper_error)) / 2.0
        return ell, power, error
    except Exception as e:
        print("Error loading Planck power spectrum: {}".format(str(e)))
        return None, None, None

# Helper function to ensure directory exists (compatible with older Python versions)
def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

class GoldenRatioCascadeTest:
    """
    Tests whether the Golden Ratio creates unique resonance cascade patterns across multiple scales
    in the cosmic microwave background radiation. This test examines if information flows more
    coherently through scales related by powers of phi than through other mathematical relationships.
    """
    
    def __init__(self, config=None):
        """Initialize the Golden Ratio Cascade Test with configuration parameters."""
        self.constants = {
            'phi': 1.618033988749895,  # Golden ratio
            'e': 2.718281828459045,    # Euler's number
            'pi': 3.141592653589793,   # Pi
            'sqrt2': 1.4142135623730951,  # Square root of 2
            'sqrt3': 1.7320508075688772,  # Square root of 3
            'ln2': 0.6931471805599453   # Natural log of 2
        }
        
        # Default configuration
        self.config = {
            'num_simulations': 10000,     # Number of Monte Carlo simulations
            'cascade_depth': 5,           # Number of scales in the cascade
            'significance_level': 0.05,   # Statistical significance threshold
            'parallel_processing': False,  # Whether to use parallel processing
            'batch_size': 1000,           # Batch size for parallel processing
            'early_stopping': True,       # Whether to stop early if significance is reached
            'timeout': 3600,              # Timeout in seconds (1 hour)
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Initialize results storage
        self.results = {}
        
    def generate_cascade_scales(self, base_scale, constant, depth):
        """
        Generate a cascade of scales related by powers of a mathematical constant.
        
        Args:
            base_scale (int): The starting scale index
            constant (float): The mathematical constant to use for scaling
            depth (int): The number of scales in the cascade
            
        Returns:
            list: A list of scale indices forming a cascade
        """
        cascade = [base_scale]
        
        # Generate ascending scales (base_scale * constant^n)
        for i in range(1, depth):
            next_scale = int(base_scale * (constant ** i))
            cascade.append(next_scale)
            
        return cascade
    
    def measure_cascade_coherence(self, data, cascade_scales):
        """
        Measure the coherence across a cascade of scales.
        
        Args:
            data: CMB power spectrum or time series
            cascade_scales: List of scale indices forming a cascade
            
        Returns:
            float: A measure of coherence across the cascade
        """
        # Extract data at cascade scales
        cascade_data = [data[scale] if scale < len(data) else 0 for scale in cascade_scales]
        
        # Calculate pairwise mutual information
        mutual_info = 0
        for i in range(len(cascade_scales) - 1):
            for j in range(i + 1, len(cascade_scales)):
                # Skip if scales are out of bounds
                if cascade_scales[i] >= len(data) or cascade_scales[j] >= len(data):
                    continue
                
                # Get windows around each scale
                window_size = min(10, len(data) // 20)
                if cascade_scales[i] < window_size or cascade_scales[j] < window_size or \
                   cascade_scales[i] >= len(data) - window_size or cascade_scales[j] >= len(data) - window_size:
                    continue
                
                window_i = data[cascade_scales[i] - window_size:cascade_scales[i] + window_size]
                window_j = data[cascade_scales[j] - window_size:cascade_scales[j] + window_size]
                
                # Calculate mutual information (approximated with correlation)
                try:
                    corr, _ = stats.pearsonr(window_i, window_j)
                    mutual_info += abs(corr)  # Use absolute correlation as proxy for mutual information
                except:
                    continue
        
        # Calculate cascade resonance as average mutual information
        if len(cascade_scales) <= 1:
            return 0
        
        num_pairs = (len(cascade_scales) * (len(cascade_scales) - 1)) / 2
        if num_pairs == 0:
            return 0
            
        return mutual_info / num_pairs
    
    def measure_cascade_transfer_entropy(self, data, cascade_scales):
        """
        Measure the transfer entropy across a cascade of scales.
        
        Args:
            data: CMB power spectrum or time series
            cascade_scales: List of scale indices forming a cascade
            
        Returns:
            float: A measure of transfer entropy across the cascade
        """
        total_transfer_entropy = 0
        count = 0
        
        # Calculate transfer entropy between consecutive scales in the cascade
        for i in range(len(cascade_scales) - 1):
            scale_i = cascade_scales[i]
            scale_j = cascade_scales[i + 1]
            
            # Skip if scales are out of bounds
            if scale_i >= len(data) or scale_j >= len(data):
                continue
            
            # Get windows around each scale
            window_size = min(20, len(data) // 20)
            if scale_i < window_size or scale_j < window_size or \
               scale_i >= len(data) - window_size or scale_j >= len(data) - window_size:
                continue
            
            window_i = data[scale_i - window_size:scale_i + window_size]
            window_j = data[scale_j - window_size:scale_j + window_size]
            
            # Calculate transfer entropy (using a simplified estimation)
            te = self._estimate_transfer_entropy(window_i, window_j)
            total_transfer_entropy += te
            count += 1
        
        # Return average transfer entropy
        if count == 0:
            return 0
            
        return total_transfer_entropy / count
    
    def _estimate_transfer_entropy(self, source, target, lag=1):
        """
        Estimate transfer entropy from source to target time series.
        
        This is a simplified implementation using conditional mutual information.
        
        Args:
            source: Source time series
            target: Target time series
            lag: Time lag
            
        Returns:
            float: Estimated transfer entropy
        """
        # Shift the target series by lag
        target_past = target[:-lag]
        target_future = target[lag:]
        source_past = source[:-lag]
        
        # Trim to same length
        min_length = min(len(target_past), len(target_future), len(source_past))
        target_past = target_past[:min_length]
        target_future = target_future[:min_length]
        source_past = source_past[:min_length]
        
        # Early return if not enough data
        if min_length < 10:
            return 0
        
        # Calculate H(Target_future | Target_past)
        try:
            # Calculate correlation as proxy for mutual information
            corr_target, _ = stats.pearsonr(target_past, target_future)
            h_target = 1 - abs(corr_target)  # Higher correlation means lower entropy
            
            # Calculate H(Target_future | Target_past, Source_past)
            # Use multiple regression as proxy
            X = np.column_stack((target_past, source_past))
            y = target_future
            
            # Check for constant columns
            if np.all(np.std(X, axis=0) > 1e-10):
                model = np.linalg.lstsq(X, y, rcond=None)[0]
                predictions = X.dot(model)
                residuals = y - predictions
                h_combined = np.std(residuals)
                
                # Transfer entropy is the reduction in uncertainty
                te = h_target - h_combined
                return max(0, te)  # Ensure non-negative
            else:
                return 0
        except:
            return 0
            
    def measure_cascade_resonance(self, data, cascade_scales):
        """
        Measure the resonance patterns across a cascade of scales.
        
        Args:
            data: CMB power spectrum or time series
            cascade_scales: List of scale indices forming a cascade
            
        Returns:
            float: A measure of resonance across the cascade
        """
        # Skip if any scale is out of bounds
        if any(scale >= len(data) for scale in cascade_scales):
            return 0
        
        # Extract data at cascade scales
        cascade_data = [data[scale] for scale in cascade_scales]
        
        # Calculate power spectrum of the cascade pattern
        try:
            # Calculate power spectrum
            f, Pxx = signal.welch(cascade_data, nperseg=min(len(cascade_data), 8))
            
            # Calculate the dominant frequency component
            dominant_idx = np.argmax(Pxx[1:]) + 1  # Skip DC component
            dominant_freq = f[dominant_idx]
            dominant_power = Pxx[dominant_idx]
            
            # Calculate the ratio of dominant frequency power to total power
            total_power = np.sum(Pxx)
            if total_power == 0:
                return 0
                
            resonance_score = dominant_power / total_power
            return resonance_score
        except:
            return 0
    
    def evaluate_cascade(self, data, base_scale, constant, depth=None):
        """
        Evaluate the resonance cascade for a specific constant and base scale.
        
        Args:
            data: CMB power spectrum or time series
            base_scale: The starting scale
            constant: The mathematical constant to use
            depth: The number of scales in the cascade (default: use config value)
            
        Returns:
            dict: A dictionary of cascade metrics
        """
        if depth is None:
            depth = self.config['cascade_depth']
        
        # Generate the cascade of scales
        cascade_scales = self.generate_cascade_scales(base_scale, constant, depth)
        
        # Measure different aspects of the cascade
        coherence = self.measure_cascade_coherence(data, cascade_scales)
        transfer_entropy = self.measure_cascade_transfer_entropy(data, cascade_scales)
        resonance = self.measure_cascade_resonance(data, cascade_scales)
        
        # Return comprehensive metrics
        return {
            'coherence': coherence,
            'transfer_entropy': transfer_entropy,
            'resonance': resonance,
            'composite_score': coherence * transfer_entropy * resonance,
            'scales': cascade_scales
        }
    
    def run_simulation(self, data, seed=None):
        """
        Run a single simulation for all constants using phase-randomized surrogate data.
        
        Args:
            data: Original CMB data
            seed: Random seed
            
        Returns:
            dict: Simulation results for all constants
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate phase-randomized surrogate
        surrogate = self._generate_surrogate(data)
        
        results = {}
        
        # Test each base scale
        base_scales = [10, 20, 50, 100, 200]
        for base_scale in base_scales:
            results[base_scale] = {}
            
            # Test each constant
            for const_name, const_value in self.constants.items():
                # Evaluate the cascade
                cascade_results = self.evaluate_cascade(surrogate, base_scale, const_value)
                results[base_scale][const_name] = cascade_results
        
        return results
    
    def _generate_surrogate(self, data):
        """
        Generate phase-randomized surrogate data preserving power spectrum.
        
        Args:
            data: Original time series
            
        Returns:
            ndarray: Surrogate time series
        """
        # Compute FFT
        fft_vals = np.fft.rfft(data)
        
        # Get amplitudes and randomize phases
        amplitudes = np.abs(fft_vals)
        phases = np.random.uniform(0, 2*np.pi, len(amplitudes))
        
        # Reconstruct with random phases
        fft_random = amplitudes * np.exp(1j * phases)
        
        # Inverse FFT to get surrogate time series
        surrogate = np.fft.irfft(fft_random, n=len(data))
        
        return surrogate
        
    def run_test(self, data):
        """
        Run the Golden Ratio Cascade Test on the provided data.
        
        Args:
            data: CMB power spectrum or time series
            
        Returns:
            dict: Test results
        """
        start_time = time.time()
        
        # Evaluate cascades for actual data
        actual_results = {}
        base_scales = [10, 20, 50, 100, 200]
        
        for base_scale in base_scales:
            actual_results[base_scale] = {}
            
            for const_name, const_value in self.constants.items():
                cascade_results = self.evaluate_cascade(data, base_scale, const_value)
                actual_results[base_scale][const_name] = cascade_results
        
        # Run simulations
        simulation_results = []
        num_simulations = self.config['num_simulations']
        
        # Set parallel_processing to False for compatibility
        self.config['parallel_processing'] = False
        
        # Run simulations sequentially
        for i in range(num_simulations):
            sim_result = self.run_simulation(data, seed=i)
            simulation_results.append(sim_result)
            
            # Check for early stopping
            if self.config['early_stopping'] and len(simulation_results) >= 1000:
                p_values = self._calculate_p_values(actual_results, simulation_results)
                min_p_value = min(p_value for base_p_values in p_values.values() 
                                for p_value in base_p_values.values())
                
                if min_p_value < self.config['significance_level'] / 10:
                    print("Early stopping at {} simulations due to significant result (p={:.6f})".format(
                        len(simulation_results), min_p_value))
                    break
            
            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > self.config['timeout']:
                print("Stopping simulations due to timeout ({:.1f}s)".format(elapsed_time))
                break
            
            # Print progress (every 100 simulations)
            if (i + 1) % 100 == 0 or i == num_simulations - 1:
                print("Completed {}/{} simulations ({:.1f}%)".format(
                    i + 1, num_simulations, 100 * (i + 1) / num_simulations))
        
        # Calculate p-values
        p_values = self._calculate_p_values(actual_results, simulation_results)
        
        # Store results
        self.results = {
            'actual_results': actual_results,
            'simulation_results': simulation_results,
            'p_values': p_values,
            'num_simulations': len(simulation_results),
            'elapsed_time': time.time() - start_time
        }
        
        return self.results
    
    def _calculate_p_values(self, actual_results, simulation_results):
        """
        Calculate p-values for each constant and base scale.
        
        Args:
            actual_results: Results from actual data
            simulation_results: Results from simulations
            
        Returns:
            dict: P-values for each constant and base scale
        """
        p_values = {}
        
        for base_scale in actual_results.keys():
            p_values[base_scale] = {}
            
            for const_name in self.constants.keys():
                # Get actual score
                actual_score = actual_results[base_scale][const_name]['composite_score']
                
                # Get simulation scores
                sim_scores = [sim_result[base_scale][const_name]['composite_score'] 
                             for sim_result in simulation_results]
                
                # Calculate p-value (proportion of simulations with score >= actual)
                count_exceeding = sum(1 for score in sim_scores if score >= actual_score)
                p_value = (count_exceeding + 1) / (len(sim_scores) + 1)  # Add 1 for Bayesian correction
                
                p_values[base_scale][const_name] = p_value
        
        return p_values
    
    def visualize_results(self, output_dir):
        """
        Visualize the test results.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            None
        """
        if not self.results:
            print("No results to visualize. Run the test first.")
            return
        
        # Create output directory if it doesn't exist
        ensure_dir_exists(output_dir)
        
        # 1. Visualize p-values for each constant and base scale
        self._visualize_p_values(output_dir)
        
        # 2. Visualize cascade patterns for the golden ratio vs other constants
        self._visualize_cascade_patterns(output_dir)
        
        # 3. Visualize distribution of simulation scores vs actual score
        self._visualize_score_distributions(output_dir)
        
    def _visualize_p_values(self, output_dir):
        """Visualize p-values for each constant and base scale."""
        p_values = self.results['p_values']
        
        # Create a heatmap of p-values
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        base_scales = sorted(p_values.keys())
        constants = list(self.constants.keys())
        
        p_value_matrix = np.zeros((len(base_scales), len(constants)))
        
        for i, base_scale in enumerate(base_scales):
            for j, const_name in enumerate(constants):
                p_value_matrix[i, j] = p_values[base_scale][const_name]
        
        # Create heatmap
        im = ax.imshow(p_value_matrix, cmap='viridis_r', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('p-value')
        
        # Add labels
        ax.set_xticks(np.arange(len(constants)))
        ax.set_yticks(np.arange(len(base_scales)))
        ax.set_xticklabels(constants)
        ax.set_yticklabels(base_scales)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add title and labels
        ax.set_title('P-values for Different Constants and Base Scales')
        ax.set_xlabel('Mathematical Constant')
        ax.set_ylabel('Base Scale')
        
        # Add text annotations
        for i in range(len(base_scales)):
            for j in range(len(constants)):
                text = ax.text(j, i, "{:.3f}".format(p_value_matrix[i, j]),
                              ha="center", va="center", color="w" if p_value_matrix[i, j] < 0.3 else "black")
        
        # Add significance threshold line
        ax.axhline(y=-0.5, color='r', linestyle='-')
        ax.text(len(constants)-1, -0.7, "p = {}".format(self.config['significance_level']), 
                color='r', ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'p_values_heatmap.png'), dpi=300)
        plt.close()
        
    def _visualize_cascade_patterns(self, output_dir):
        """Visualize cascade patterns for the golden ratio vs other constants."""
        actual_results = self.results['actual_results']
        
        # Ensure we have the original data
        if not hasattr(self, 'original_data'):
            print("Warning: Original data not available for cascade pattern visualization")
            return
        
        # Select a representative base scale
        base_scale = 50  # Middle value from [10, 20, 50, 100, 200]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot cascade patterns for each constant
        for const_name, const_value in self.constants.items():
            cascade_scales = actual_results[base_scale][const_name]['scales']
            
            # Skip if scales are out of range
            if max(cascade_scales) >= len(self.original_data):
                continue
                
            # Get values at cascade scales
            cascade_values = [self.original_data[scale] for scale in cascade_scales]
            
            # Plot with different style for golden ratio
            if const_name == 'phi':
                ax.plot(cascade_scales, cascade_values, 'o-', linewidth=3, markersize=10,
                       label="{} ({:.4f})".format(const_name, const_value))
            else:
                ax.plot(cascade_scales, cascade_values, 'o--', alpha=0.7,
                       label="{} ({:.4f})".format(const_name, const_value))
        
        # Add labels and title
        ax.set_title('Cascade Patterns for Different Constants (Base Scale = {})'.format(base_scale))
        ax.set_xlabel('Scale Index')
        ax.set_ylabel('Power Spectrum Value')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cascade_patterns.png'), dpi=300)
        plt.close()
        
    def _visualize_score_distributions(self, output_dir):
        """Visualize distribution of simulation scores vs actual score."""
        actual_results = self.results['actual_results']
        simulation_results = self.results['simulation_results']
        
        # Create a figure for each base scale and constant combination
        for base_scale in actual_results.keys():
            for const_name in self.constants.keys():
                # Get actual score
                actual_score = actual_results[base_scale][const_name]['composite_score']
                
                # Get simulation scores
                sim_scores = [sim_result[base_scale][const_name]['composite_score'] 
                             for sim_result in simulation_results]
                
                # Create histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot histogram of simulation scores
                ax.hist(sim_scores, bins=30, alpha=0.7, color='blue')
                
                # Add vertical line for actual score
                ax.axvline(x=actual_score, color='red', linestyle='--', linewidth=2,
                          label='Actual Score: {:.4f}'.format(actual_score))
                
                # Calculate and display p-value
                p_value = self.results['p_values'][base_scale][const_name]
                ax.text(0.95, 0.95, 'p-value: {:.4f}'.format(p_value),
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add labels and title
                ax.set_title('Score Distribution for {} (Base Scale = {})'.format(const_name, base_scale))
                ax.set_xlabel('Composite Score')
                ax.set_ylabel('Frequency')
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'score_dist_{}_{}.png'.format(const_name, base_scale)), dpi=300)
                plt.close()
    
    def generate_report(self, output_dir):
        """
        Generate a comprehensive report of the test results.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            str: Path to the report file
        """
        if not self.results:
            print("No results to report. Run the test first.")
            return None
        
        # Create output directory if it doesn't exist
        ensure_dir_exists(output_dir)
        
        # Prepare report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, 'golden_ratio_cascade_report_{}.txt'.format(timestamp))
        
        with open(report_file, 'w') as f:
            # Write header
            f.write("="*80 + "\n")
            f.write("GOLDEN RATIO CASCADE TEST REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Write test configuration
            f.write("TEST CONFIGURATION\n")
            f.write("-"*80 + "\n")
            for key, value in self.config.items():
                f.write("{}: {}\n".format(key, value))
            f.write("\n")
            
            # Write test summary
            f.write("TEST SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write("Number of simulations: {}\n".format(self.results['num_simulations']))
            f.write("Elapsed time: {:.2f} seconds\n".format(self.results['elapsed_time']))
            f.write("\n")
            
            # Write p-values
            f.write("P-VALUES\n")
            f.write("-"*80 + "\n")
            f.write("Base Scale | " + " | ".join("{:^10}".format(const) for const in self.constants.keys()) + "\n")
            f.write("-"*80 + "\n")
            
            for base_scale in sorted(self.results['p_values'].keys()):
                row = "{:^10} | ".format(base_scale)
                for const_name in self.constants.keys():
                    p_value = self.results['p_values'][base_scale][const_name]
                    row += "{:^10.4f} | ".format(p_value)
                f.write(row + "\n")
            f.write("\n")
            
            # Write significant findings
            f.write("SIGNIFICANT FINDINGS\n")
            f.write("-"*80 + "\n")
            
            significant_findings = []
            for base_scale in self.results['p_values'].keys():
                for const_name in self.constants.keys():
                    p_value = self.results['p_values'][base_scale][const_name]
                    if p_value < self.config['significance_level']:
                        significant_findings.append((base_scale, const_name, p_value))
            
            if significant_findings:
                for base_scale, const_name, p_value in sorted(significant_findings, key=lambda x: x[2]):
                    f.write("* Base Scale {}, Constant {}: p-value = {:.6f}\n".format(
                        base_scale, const_name, p_value))
            else:
                f.write("No significant findings at alpha = {}\n".format(self.config['significance_level']))
            f.write("\n")
            
            # Write conclusion
            f.write("CONCLUSION\n")
            f.write("-"*80 + "\n")
            
            # Check if golden ratio has the lowest p-value
            min_p_values = {}
            for const_name in self.constants.keys():
                min_p_values[const_name] = min(self.results['p_values'][base_scale][const_name] 
                                             for base_scale in self.results['p_values'].keys())
            
            sorted_constants = sorted(min_p_values.items(), key=lambda x: x[1])
            
            if sorted_constants[0][0] == 'phi':
                f.write("The Golden Ratio (phi) shows the strongest cascade pattern ")
                f.write("with the lowest p-value of {:.6f}.\n".format(sorted_constants[0][1]))
                
                if sorted_constants[0][1] < self.config['significance_level']:
                    f.write("This result is statistically significant at alpha = {}.\n".format(
                        self.config['significance_level']))
                else:
                    f.write("However, this result is not statistically significant ")
                    f.write("at alpha = {}.\n".format(self.config['significance_level']))
            else:
                f.write("The constant {} shows the strongest cascade pattern ".format(sorted_constants[0][0]))
                f.write("with the lowest p-value of {:.6f}, ".format(sorted_constants[0][1]))
                f.write("while the Golden Ratio (phi) has a p-value of {:.6f}.\n".format(min_p_values['phi']))
            
            # Add comparison to other constants
            f.write("\nRanking of constants by minimum p-value:\n")
            for i, (const_name, p_value) in enumerate(sorted_constants):
                f.write("{}. {}: {:.6f}\n".format(i+1, const_name, p_value))
        
        print("Report generated: {}".format(report_file))
        return report_file

def main():
    """Run the Golden Ratio Cascade Test on WMAP and Planck data."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run Golden Ratio Cascade Test on CMB data')
    
    # Define base directory for the COSMIC_Analysis project
    cosmic_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser.add_argument('--wmap-data', type=str, 
                        default=os.path.join(cosmic_dir, "data", "wmap", "wmap_tt_spectrum_9yr_v5.txt"),
                        help='Path to WMAP power spectrum data')
    parser.add_argument('--planck-data', type=str, 
                        default=os.path.join(cosmic_dir, "data", "planck", "planck_tt_spectrum_2018.txt"),
                        help='Path to Planck power spectrum data')
    parser.add_argument('--output-dir', type=str, 
                        default=os.path.join(cosmic_dir, "results", "golden_ratio_cascade_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
                        help='Directory to save results')
    parser.add_argument('--num-simulations', type=int, default=10000,
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='Disable early stopping')
    parser.add_argument('--cascade-depth', type=int, default=5,
                        help='Number of scales in the cascade')
    args = parser.parse_args()
    
    # Create output directories
    wmap_output_dir = os.path.join(args.output_dir, 'wmap_full')
    planck_output_dir = os.path.join(args.output_dir, 'planck')
    ensure_dir_exists(wmap_output_dir)
    ensure_dir_exists(planck_output_dir)
    
    # Configure test
    config = {
        'num_simulations': args.num_simulations,
        'cascade_depth': args.cascade_depth,
        'parallel_processing': not args.no_parallel,
        'early_stopping': not args.no_early_stopping
    }
    
    # Run test on WMAP data
    print("="*80)
    print("Running Golden Ratio Cascade Test on WMAP data")
    print("="*80)
    
    # Load WMAP data
    wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(args.wmap_data)
    if wmap_ell is None:
        print("Failed to load WMAP data. Exiting.")
        return
    
    # Initialize and run test
    config_wmap = {
        'num_simulations': args.num_simulations,
        'parallel_processing': not args.no_parallel,
        'early_stopping': False  # Disable early stopping for WMAP
    }
    wmap_test = GoldenRatioCascadeTest(config_wmap)
    wmap_test.original_data = wmap_power  # Store for visualization
    wmap_results = wmap_test.run_test(wmap_power)
    
    # Generate visualizations and report
    wmap_test.visualize_results(wmap_output_dir)
    wmap_report = wmap_test.generate_report(wmap_output_dir)
    
    # Run test on Planck data
    print("\n" + "="*80)
    print("Running Golden Ratio Cascade Test on Planck data")
    print("="*80)
    
    # Load Planck data
    planck_ell, planck_power, planck_error = load_planck_power_spectrum(args.planck_data)
    if planck_ell is None:
        print("Failed to load Planck data. Exiting.")
        return
    
    # Initialize and run test
    planck_test = GoldenRatioCascadeTest(config)
    planck_test.original_data = planck_power  # Store for visualization
    planck_results = planck_test.run_test(planck_power)
    
    # Generate visualizations and report
    planck_test.visualize_results(planck_output_dir)
    planck_report = planck_test.generate_report(planck_output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("Golden Ratio Cascade Test Summary")
    print("="*80)
    
    # Compare WMAP and Planck results
    print("\nWMAP Results:")
    min_wmap_p_value = min(p_value for base_p_values in wmap_results['p_values'].values() 
                          for p_value in base_p_values.values())
    print("Minimum p-value: {:.6f}".format(min_wmap_p_value))
    
    print("\nPlanck Results:")
    min_planck_p_value = min(p_value for base_p_values in planck_results['p_values'].values() 
                            for p_value in base_p_values.values())
    print("Minimum p-value: {:.6f}".format(min_planck_p_value))
    
    print("\nOutput directories:")
    print("WMAP results: {}".format(wmap_output_dir))
    print("Planck results: {}".format(planck_output_dir))
    
    print("\nReports:")
    print("WMAP report: {}".format(wmap_report))
    print("Planck report: {}".format(planck_report))

if __name__ == '__main__':
    main()
