#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orthogonality Test for WMAP and Planck CMB data.

This script implements the Orthogonality Test, which evaluates whether optimization
around different mathematical constants (e.g., phi, e, pi) is orthogonal in the
CMB power spectrum. This helps determine if golden ratio patterns are independent
from patterns related to other mathematical constants.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import json
from datetime import datetime
import argparse
from multiprocessing import Pool, cpu_count
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_wmap_power_spectrum(file_path):
    """Load WMAP CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Mean multipole moment
        power = data[:, 3]  # Power spectrum value
        error = data[:, 4]  # Error
        return ell, power, error
    except Exception as e:
        print("Error loading WMAP power spectrum: %s" % str(e))
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
        print("Error loading Planck power spectrum: %s" % str(e))
        return None, None, None


def preprocess_data(data, smooth=False, smooth_window=5, normalize=True, detrend=False):
    """Preprocess data for analysis."""
    # Make a copy to avoid modifying the original
    processed_data = np.copy(data)
    
    # Apply smoothing if requested
    if smooth:
        processed_data = np.convolve(
            processed_data, 
            np.ones(smooth_window)/smooth_window, 
            mode='same'
        )
    
    # Remove linear trend if requested
    if detrend:
        x = np.arange(len(processed_data))
        slope, intercept = np.polyfit(x, processed_data, 1)
        processed_data = processed_data - (slope * x + intercept)
    
    # Normalize to [0, 1] range if requested
    if normalize:
        min_val = np.min(processed_data)
        max_val = np.max(processed_data)
        if max_val > min_val:
            processed_data = (processed_data - min_val) / (max_val - min_val)
    
    return processed_data


class OrthogonalityTest:
    """
    Test whether optimization around different mathematical constants is orthogonal.
    This evaluates if the golden ratio (phi) patterns are independent from patterns
    related to other constants (e.g., e, pi, sqrt(2), etc.)
    """
    
    def __init__(self, config=None):
        """Initialize the orthogonality test with configuration parameters."""
        self.constants = {
            'phi': 1.618033988749895,
            'e': 2.718281828459045,
            'pi': 3.141592653589793,
            'sqrt2': 1.4142135623730951,
            'sqrt3': 1.7320508075688772,
            'ln2': 0.6931471805599453
        }
        
        # Default configuration
        self.config = {
            'num_simulations': 10000,
            'significance_level': 0.05,
            'parallel_processing': False,  # Always False for compatibility
            'batch_size': 1000,
            'early_stopping': True,
            'timeout': 3600,  # 1 hour timeout
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Initialize results storage
        self.results = {}
        
    def compute_optimization_vectors(self, data, scales):
        """
        Compute optimization vectors for each constant.
        Each vector represents how well the data optimizes around scales 
        related by that constant.
        
        Args:
            data: CMB power spectrum or time series
            scales: Array of scale indices to analyze
            
        Returns:
            Dictionary mapping constants to their optimization vectors
        """
        optimization_vectors = {}
        
        # For each mathematical constant
        for const_name, const_value in self.constants.items():
            vector = []
            
            # Calculate optimization at each scale
            for scale in scales:
                # Find related scale based on this constant
                related_scale = int(scale * const_value)
                if related_scale >= len(data) or related_scale < 0:
                    continue
                
                # Calculate correlation between these scales
                correlation = self._calculate_correlation(data, scale, related_scale)
                vector.append(correlation)
            
            # Normalize the vector
            if vector:
                vector = np.array(vector)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                optimization_vectors[const_name] = vector
        
        return optimization_vectors
    
    def _calculate_correlation(self, data, scale1, scale2):
        """Calculate correlation between two scales in the data."""
        # Extract windows centered on each scale with appropriate padding
        window_size = min(20, len(data) // 10)  # Adaptive window size
        
        # Ensure scales are within bounds
        if scale1 < window_size or scale2 < window_size or \
           scale1 >= len(data) - window_size or scale2 >= len(data) - window_size:
            return 0
        
        window1 = data[scale1 - window_size:scale1 + window_size]
        window2 = data[scale2 - window_size:scale2 + window_size]
        
        # Calculate correlation
        corr, _ = stats.pearsonr(window1, window2)
        return corr if not np.isnan(corr) else 0
    
    def calculate_orthogonality(self, optimization_vectors):
        """
        Calculate orthogonality between optimization vectors for different constants.
        
        Returns:
            DataFrame with orthogonality scores between each pair of constants
        """
        constants = list(optimization_vectors.keys())
        orthogonality_matrix = np.zeros((len(constants), len(constants)))
        
        for i, const1 in enumerate(constants):
            vector1 = optimization_vectors[const1]
            
            for j, const2 in enumerate(constants):
                if i == j:
                    orthogonality_matrix[i, j] = 1.0  # Self-orthogonality is 1
                    continue
                
                vector2 = optimization_vectors[const2]
                
                # Ensure vectors are the same length
                min_len = min(len(vector1), len(vector2))
                if min_len == 0:
                    orthogonality_matrix[i, j] = np.nan
                    continue
                
                # Calculate dot product (cosine similarity since vectors are normalized)
                dot_product = np.dot(vector1[:min_len], vector2[:min_len])
                
                # Orthogonality is 1 - |cosine similarity|
                orthogonality_matrix[i, j] = 1 - abs(dot_product)
        
        return pd.DataFrame(orthogonality_matrix, index=constants, columns=constants)
    
    def run_simulation(self, data, scales, seed=None):
        """Run a single simulation with phase-randomized surrogate data."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate phase-randomized surrogate
        surrogate = self._generate_surrogate(data)
        
        # Compute optimization vectors for surrogate
        opt_vectors = self.compute_optimization_vectors(surrogate, scales)
        
        # Calculate orthogonality matrix
        orthogonality = self.calculate_orthogonality(opt_vectors)
        
        return orthogonality
    
    def _generate_surrogate(self, data):
        """Generate phase-randomized surrogate data preserving power spectrum."""
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
    
    def run_test(self, data, scales):
        """
        Run the orthogonality test with the specified number of simulations.
        
        Args:
            data: CMB power spectrum or time series
            scales: Array of scale indices to analyze
            
        Returns:
            Dictionary containing test results including:
            - Observed orthogonality matrix
            - Mean surrogate orthogonality matrix
            - P-values for each pair of constants
            - Phi-optimality scores for orthogonality
        """
        # Compute optimization vectors for actual data
        actual_opt_vectors = self.compute_optimization_vectors(data, scales)
        
        # Calculate orthogonality matrix for actual data
        actual_orthogonality = self.calculate_orthogonality(actual_opt_vectors)
        
        # Store in results
        self.results['actual_orthogonality'] = actual_orthogonality
        
        # Run simulations
        num_sims = self.config['num_simulations']
        surrogate_results = []
        
        print("Running %d simulations sequentially..." % num_sims)
        for i in range(num_sims):
            sim_result = self.run_simulation(data, scales, seed=i)
            surrogate_results.append(sim_result)
            
            # Print progress
            if (i+1) % 10 == 0:
                print("Completed %d/%d simulations..." % (i+1, num_sims))
            
            # Early stopping check if enabled
            if self.config['early_stopping'] and (i+1) % 10 == 0:
                if self._check_significance(actual_orthogonality, surrogate_results):
                    print("Early stopping at simulation %d: already exceeded significance threshold" % (i+1))
                    break
        
        # Calculate p-values and phi-optimality
        p_values, phi_optimality = self._calculate_statistics(actual_orthogonality, surrogate_results)
        
        # Store results
        self.results['num_simulations'] = len(surrogate_results)
        self.results['p_values'] = p_values
        self.results['phi_optimality'] = phi_optimality
        
        return self.results
    
    def _check_significance(self, actual, surrogates, min_sims=30):
        """Check if we've already reached significance for early stopping."""
        if len(surrogates) < min_sims:
            return False
            
        # Get phi-related orthogonality values
        phi_orth_actual = self._extract_phi_orthogonality(actual)
        phi_orth_surr = [self._extract_phi_orthogonality(s) for s in surrogates]
        
        # Calculate preliminary p-value
        p_value = sum(s >= phi_orth_actual for s in phi_orth_surr) / len(phi_orth_surr)
        
        # Stop if already significant
        if p_value < self.config['significance_level'] / 10:  # More conservative
            print("Early stopping at simulation %d: already exceeded significance threshold" % len(surrogates))
            return True
        return False
    
    def _extract_phi_orthogonality(self, orth_matrix):
        """Extract mean orthogonality of phi to other constants."""
        if 'phi' not in orth_matrix.index:
            return 0
            
        phi_row = orth_matrix.loc['phi']
        other_constants = [c for c in phi_row.index if c != 'phi']
        
        if not other_constants:
            return 0
            
        return phi_row[other_constants].mean()
    
    def _calculate_statistics(self, actual, surrogates):
        """Calculate p-values and phi-optimality for orthogonality results."""
        # Initialize results
        constants = actual.index
        p_values = pd.DataFrame(1.0, index=constants, columns=constants)
        phi_optimality = pd.DataFrame(0.0, index=constants, columns=constants)
        
        # For each pair of constants
        for const1 in constants:
            for const2 in constants:
                if const1 == const2:
                    continue
                
                # Get actual orthogonality
                actual_orth = actual.loc[const1, const2]
                
                # Get surrogate orthogonalities
                surrogate_orths = [s.loc[const1, const2] for s in surrogates if const1 in s.index and const2 in s.index]
                
                if not surrogate_orths:
                    continue
                    
                # Calculate p-value (proportion of surrogates with higher orthogonality)
                p_value = sum(s >= actual_orth for s in surrogate_orths) / len(surrogate_orths)
                p_values.loc[const1, const2] = p_value
                
                # Calculate phi-optimality (percentile rank, from -1 to 1)
                sorted_orths = sorted(surrogate_orths)
                rank = sorted_orths.index(min(sorted_orths, key=lambda x: abs(x-actual_orth))) if actual_orth in sorted_orths else 0
                phi_optimality.loc[const1, const2] = 2 * (rank / len(sorted_orths)) - 1
        
        return p_values, phi_optimality
    
    def visualize_results(self, save_path=None):
        """Visualize the orthogonality test results."""
        if not self.results:
            print("No results to visualize. Run the test first.")
            return
            
        # Create figure with multiple subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot actual orthogonality matrix
        im0 = axes[0].imshow(self.results['actual_orthogonality'], cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title('Observed Orthogonality')
        plt.colorbar(im0, ax=axes[0])
        
        # Add labels
        constants = self.results['actual_orthogonality'].index
        for ax in axes[:2]:
            ax.set_xticks(range(len(constants)))
            ax.set_yticks(range(len(constants)))
            ax.set_xticklabels(constants)
            ax.set_yticklabels(constants)
        
        # Plot p-values
        im1 = axes[1].imshow(self.results['p_values'], cmap='coolwarm_r', vmin=0, vmax=0.2)
        axes[1].set_title('P-values')
        plt.colorbar(im1, ax=axes[1])
        
        # Plot phi-optimality
        im2 = axes[2].imshow(self.results['phi_optimality'], cmap='RdBu', vmin=-1, vmax=1)
        axes[2].set_title('Phi-Optimality')
        plt.colorbar(im2, ax=axes[2])
        
        axes[2].set_xticks(range(len(constants)))
        axes[2].set_yticks(range(len(constants)))
        axes[2].set_xticklabels(constants)
        axes[2].set_yticklabels(constants)
        
        plt.tight_layout()
        
        if save_path:
            # Ensure the directory exists
            ensure_dir_exists(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Visualization saved to %s" % save_path)
            
        plt.close()
        
    def generate_report(self):
        """Generate a text report summarizing the orthogonality test results."""
        if not self.results:
            return "No results available. Run the test first."
            
        report = ["Orthogonality Test Results", "========================="]
        
        # Add basic information
        report.append("Number of simulations: %d" % self.results['num_simulations'])
        
        # Add phi-related orthogonality results
        if 'phi' in self.results['actual_orthogonality'].index:
            phi_row = self.results['actual_orthogonality'].loc['phi']
            other_constants = [c for c in phi_row.index if c != 'phi']
            
            report.append("\nOrthogonality of Golden Ratio (phi) to other constants:")
            report.append("--------------------------------------------------")
            
            for const in other_constants:
                orth_value = phi_row[const]
                p_value = self.results['p_values'].loc['phi', const]
                phi_opt = self.results['phi_optimality'].loc['phi', const]
                
                significance = "significant" if p_value < 0.05 else "not significant"
                
                report.append("phi vs %s: Orthogonality = %.4f, p-value = %.6f (%s)" % (const, orth_value, p_value, significance))
                report.append("    Phi-optimality = %.4f" % phi_opt)
            
            # Calculate mean orthogonality
            mean_orth = phi_row[other_constants].mean()
            report.append("\nMean orthogonality of phi to all other constants: %.4f" % mean_orth)
            
            # Calculate overall significance
            min_p_value = self.results['p_values'].loc['phi', other_constants].min()
            report.append("Minimum p-value across all phi comparisons: %.6f" % min_p_value)
        
        # Add overall test conclusion
        report.append("\nTest Conclusion:")
        report.append("---------------")
        
        if 'phi' in self.results['actual_orthogonality'].index:
            min_p = self.results['p_values'].loc['phi', [c for c in phi_row.index if c != 'phi']].min()
            
            if min_p < 0.05:
                report.append("The golden ratio (phi) optimization is significantly orthogonal to at least")
                report.append("one other mathematical constant, suggesting its unique role in cosmic organization.")
            else:
                report.append("No statistically significant orthogonality detected between the golden ratio")
                report.append("and other mathematical constants.")
                
        return "\n".join(report)


def ensure_dir_exists(directory):
    """Create directory if it doesn't exist (compatible with older Python versions)."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_orthogonality_test(data, output_dir, name, n_simulations=1000, parallel=False, early_stopping=True):
    """
    Run orthogonality test on the provided data.
    
    Args:
        data (numpy.ndarray): Data to analyze
        output_dir (str): Directory to save results
        name (str): Name of the dataset (e.g., 'WMAP' or 'Planck')
        n_simulations (int): Number of simulations for Monte Carlo
        parallel (bool): Whether to use parallel processing (ignored, always False for compatibility)
        early_stopping (bool): Whether to enable early stopping
        
    Returns:
        dict: Analysis results
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    ensure_dir_exists(output_dir)
    
    # Define scales to analyze (all valid indices in the data)
    scales = np.arange(2, len(data) - 1)
    
    # Configure the test
    config = {
        'num_simulations': n_simulations,
        'parallel_processing': False,  # Always False for compatibility
        'early_stopping': early_stopping
    }
    
    # Create and run the test
    print("Running orthogonality test on %s data with %d simulations..." % (name, n_simulations))
    orthogonality_test = OrthogonalityTest(config)
    results = orthogonality_test.run_test(data, scales)
    
    # Generate and save report
    report = orthogonality_test.generate_report()
    report_path = os.path.join(output_dir, "%s_orthogonality_report.txt" % name.lower())
    with open(report_path, 'w') as f:
        f.write(report)
    print("Report saved to %s" % report_path)
    
    # Visualize and save results
    vis_path = os.path.join(output_dir, "%s_orthogonality_visualization.png" % name.lower())
    orthogonality_test.visualize_results(save_path=vis_path)
    
    # Save raw results as JSON
    results_json = {}
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            results_json[key] = value.to_dict()
        else:
            results_json[key] = value
    
    json_path = os.path.join(output_dir, "%s_orthogonality_results.json" % name.lower())
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print("Raw results saved to %s" % json_path)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print("Orthogonality test for %s completed in %.2f seconds" % (name, execution_time))
    
    # Add execution time to results
    results['execution_time'] = execution_time
    
    return results


def compare_results(wmap_results, planck_results, output_dir):
    """
    Compare orthogonality test results between WMAP and Planck data.
    
    Args:
        wmap_results (dict): Results from WMAP analysis
        planck_results (dict): Results from Planck analysis
        output_dir (str): Directory to save comparison results
    """
    # Create output directory if it doesn't exist
    ensure_dir_exists(output_dir)
    
    # Extract phi orthogonality values
    wmap_phi_orth = None
    planck_phi_orth = None
    
    if 'actual_orthogonality' in wmap_results and 'phi' in wmap_results['actual_orthogonality'].index:
        wmap_phi_row = wmap_results['actual_orthogonality'].loc['phi']
        wmap_other_constants = [c for c in wmap_phi_row.index if c != 'phi']
        wmap_phi_orth = wmap_phi_row[wmap_other_constants].mean()
    
    if 'actual_orthogonality' in planck_results and 'phi' in planck_results['actual_orthogonality'].index:
        planck_phi_row = planck_results['actual_orthogonality'].loc['phi']
        planck_other_constants = [c for c in planck_phi_row.index if c != 'phi']
        planck_phi_orth = planck_phi_row[planck_other_constants].mean()
    
    # Generate comparison report
    report = ["Comparison of WMAP and Planck Orthogonality Results", 
              "=============================================="]
    
    if wmap_phi_orth is not None and planck_phi_orth is not None:
        report.append("\nMean phi orthogonality in WMAP data: %.4f" % wmap_phi_orth)
        report.append("Mean phi orthogonality in Planck data: %.4f" % planck_phi_orth)
        report.append("Difference: %.4f" % abs(wmap_phi_orth - planck_phi_orth))
        
        if wmap_phi_orth > planck_phi_orth:
            report.append("\nThe golden ratio shows stronger orthogonality in WMAP data.")
            report.append("This suggests that phi-related patterns in the WMAP data are more")
            report.append("distinct from other mathematical constants compared to Planck data.")
        else:
            report.append("\nThe golden ratio shows stronger orthogonality in Planck data.")
            report.append("This suggests that phi-related patterns in the Planck data are more")
            report.append("distinct from other mathematical constants compared to WMAP data.")
    else:
        report.append("\nUnable to compare phi orthogonality between datasets.")
        report.append("One or both datasets may not have valid orthogonality results.")
    
    # Add p-value comparison
    if ('p_values' in wmap_results and 'p_values' in planck_results and
        'phi' in wmap_results['p_values'].index and 'phi' in planck_results['p_values'].index):
        
        report.append("\nP-value comparison for phi orthogonality:")
        report.append("----------------------------------------")
        
        wmap_min_p = wmap_results['p_values'].loc['phi', [c for c in wmap_phi_row.index if c != 'phi']].min()
        planck_min_p = planck_results['p_values'].loc['phi', [c for c in planck_phi_row.index if c != 'phi']].min()
        
        report.append("Minimum p-value in WMAP data: %.6f" % wmap_min_p)
        report.append("Minimum p-value in Planck data: %.6f" % planck_min_p)
        
        wmap_significant = wmap_min_p < 0.05
        planck_significant = planck_min_p < 0.05
        
        if wmap_significant and planck_significant:
            report.append("\nBoth datasets show statistically significant orthogonality")
            report.append("for the golden ratio compared to at least one other constant.")
        elif wmap_significant:
            report.append("\nOnly WMAP data shows statistically significant orthogonality")
            report.append("for the golden ratio compared to other constants.")
        elif planck_significant:
            report.append("\nOnly Planck data shows statistically significant orthogonality")
            report.append("for the golden ratio compared to other constants.")
        else:
            report.append("\nNeither dataset shows statistically significant orthogonality")
            report.append("for the golden ratio compared to other constants.")
    
    # Save comparison report
    report_path = os.path.join(output_dir, "wmap_planck_orthogonality_comparison.txt")
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    print("Comparison report saved to %s" % report_path)
    
    # Create comparison visualization
    if wmap_phi_orth is not None and planck_phi_orth is not None:
        plt.figure(figsize=(10, 6))
        
        # Bar chart comparing phi orthogonality
        datasets = ['WMAP', 'Planck']
        values = [wmap_phi_orth, planck_phi_orth]
        
        plt.bar(datasets, values, color=['blue', 'red'])
        plt.ylabel('Mean Orthogonality')
        plt.title('Comparison of Golden Ratio Orthogonality')
        plt.ylim(0, 1)
        
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, "%.4f" % v, ha='center')
        
        # Save visualization
        vis_path = os.path.join(output_dir, "wmap_planck_orthogonality_comparison.png")
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("Comparison visualization saved to %s" % vis_path)


def main():
    """Run orthogonality test on WMAP and Planck data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run orthogonality test on CMB data')
    parser.add_argument('--output-dir', type=str, default="../results/orthogonality_test",
                        help='Directory to save results')
    parser.add_argument('--n-simulations', type=int, default=1000,
                        help='Number of Monte Carlo simulations')
    parser.add_argument('--wmap-only', action='store_true',
                        help='Run test only on WMAP data')
    parser.add_argument('--planck-only', action='store_true',
                        help='Run test only on Planck data')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing (ignored, always disabled for compatibility)')
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='Disable early stopping and run all simulations')
    args = parser.parse_args()
    
    # Determine which datasets to analyze
    run_wmap = not args.planck_only
    run_planck = not args.wmap_only
    
    # Set up paths
    wmap_file = os.path.join('data', 'wmap_tt_spectrum_9yr_v5.txt')
    planck_file = os.path.join('data', 'planck_tt_spectrum_2018.txt')
    
    # Check if files exist
    if run_wmap and not os.path.exists(wmap_file):
        print("WMAP data file not found at %s" % wmap_file)
        print("Checking in project root directory...")
        wmap_file = 'wmap_tt_spectrum_9yr_v5.txt'
        if not os.path.exists(wmap_file):
            print("WMAP data file not found at %s" % wmap_file)
            run_wmap = False
    
    if run_planck and not os.path.exists(planck_file):
        print("Planck data file not found at %s" % planck_file)
        run_planck = False
    
    # Create output directory
    ensure_dir_exists(args.output_dir)
    
    # Initialize results
    wmap_results = None
    planck_results = None
    
    # Run test on WMAP data
    if run_wmap:
        print("Loading WMAP data from %s..." % wmap_file)
        ell, power, error = load_wmap_power_spectrum(wmap_file)
        
        if ell is not None and power is not None:
            # Preprocess data
            processed_power = preprocess_data(power, smooth=False, normalize=True)
            
            # Run test
            wmap_results = run_orthogonality_test(
                processed_power, 
                args.output_dir, 
                'WMAP', 
                n_simulations=args.n_simulations,
                parallel=False,  # Always False for compatibility
                early_stopping=not args.no_early_stopping
            )
        else:
            print("Failed to load WMAP data. Skipping WMAP analysis.")
    
    # Run test on Planck data
    if run_planck:
        print("Loading Planck data from %s..." % planck_file)
        ell, power, error = load_planck_power_spectrum(planck_file)
        
        if ell is not None and power is not None:
            # Preprocess data
            processed_power = preprocess_data(power, smooth=False, normalize=True)
            
            # Run test
            planck_results = run_orthogonality_test(
                processed_power, 
                args.output_dir, 
                'Planck', 
                n_simulations=args.n_simulations,
                parallel=False,  # Always False for compatibility
                early_stopping=not args.no_early_stopping
            )
        else:
            print("Failed to load Planck data. Skipping Planck analysis.")
    
    # Compare results if both datasets were analyzed
    if wmap_results is not None and planck_results is not None:
        compare_results(wmap_results, planck_results, args.output_dir)
    
    print("\nOrthogonality test complete. Results saved to %s" % args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
