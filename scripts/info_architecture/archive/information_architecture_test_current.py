#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Information Architecture Test for Cosmic Microwave Background Data

This test examines how different mathematical constants organize different aspects 
of the hierarchical information structure in the CMB. It's particularly focused on 
testing whether phi and sqrt2 specialize in organizing different layers of the 
information hierarchy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from datetime import datetime
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import pdist, squareform
import argparse
import os
import time
from copy import deepcopy
import multiprocessing
import json

def ensure_dir_exists(dir_path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_wmap_power_spectrum(file_path):
    """Load WMAP CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Mean multipole moment
        power = data[:, 3]  # Power spectrum value
        error = data[:, 4]  # Error
        return ell, power, error
    except Exception as e:
        print("No WMAP data file provided or loading failed. Creating sample WMAP data.")
        # Create sample data if loading fails
        ell = np.arange(2, 1201)
        power = 1000 * np.power(ell, -0.6) * (1 + 0.1 * np.sin(ell / 10))
        error = power * 0.05  # 5% error
        return ell, power, error

def load_planck_power_spectrum(file_path):
    """Load Planck CMB power spectrum data."""
    try:
        data = np.loadtxt(file_path, comments='#')
        ell = data[:, 0]  # Multipole moment
        power = data[:, 1]  # Power spectrum value
        # Use average of asymmetric error bars as the error
        error = (data[:, 2] + data[:, 3]) / 2
        return ell, power, error
    except Exception as e:
        print("No Planck data file provided or loading failed. Creating sample Planck data.")
        # Create sample data if loading fails
        ell = np.arange(2, 2501)
        power = 1000 * np.power(ell, -0.6) * (1 + 0.08 * np.sin(ell / 12))
        error = power * 0.03  # 3% error
        return ell, power, error

# Define a global function for parallel processing
def _run_simulation_global(args):
    """
    Global function for running a single simulation in parallel.
    
    Args:
        args: Tuple of (data, constant, seed)
        
    Returns:
        float: Architecture score for this simulation
    """
    data, constant, seed = args
    
    # Create surrogate data
    np.random.seed(seed)
    surrogate_data = np.random.permutation(data)
    
    # Calculate architecture score using a new test instance
    test = InformationArchitectureTest()
    return test.calculate_architecture_score(surrogate_data, constant)

class InformationArchitectureTest:
    """Test for measuring how mathematical constants organize cosmic information architecture."""
    
    def __init__(self, config=None):
        """
        Initialize the Information Architecture Test.
        
        Args:
            config: Test configuration
        """
        # Default configuration
        self.config = {
            'significance_level': 0.05,
            'num_simulations': 1000,
            'output_dir': "../results/information_architecture_test",
            'early_stopping': False,
            'parallel_processing': False,
            'num_processes': 4,
            'hierarchy_depth': 5
        }
        
        # Update with user configuration
        if config:
            self.config.update(config)
        
        # Mathematical constants to test
        self.constants = {
            'phi': 1.618033988749895,  # Golden ratio
            'sqrt2': 1.4142135623730951,  # Square root of 2
            'sqrt3': 1.7320508075688772,  # Square root of 3
            'ln2': 0.6931471805599453,  # Natural logarithm of 2
            'e': 2.718281828459045,  # Euler's number
            'pi': 3.141592653589793  # Pi
        }
        
        # Initialize cache for performance optimization
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def calculate_architecture_score(self, data, constant):
        """
        Calculate the architecture score for a given dataset and constant.
        
        Args:
            data: Data to analyze
            constant: Mathematical constant to test
            
        Returns:
            float: Architecture score
        """
        # Ensure data is a numpy array
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except:
                return 0.0
        
        # Check cache first
        try:
            data_hash = hash(data.tobytes()) if hasattr(data, 'tobytes') else hash(str(data))
            cache_key = ('architecture_score', data_hash, constant)
            
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            
            self.cache_misses += 1
        except:
            # If hashing fails, skip caching
            pass
        
        # Calculate power spectrum
        try:
            power_spectrum = np.abs(np.fft.rfft(data))**2
            
            # Normalize power spectrum
            power_spectrum = power_spectrum / np.sum(power_spectrum)
            
            # Calculate layer boundaries based on the constant
            n = len(power_spectrum)
            boundaries = []
            
            # Generate boundaries using the constant
            value = 1.0
            while value < n:
                boundaries.append(int(value))
                value *= constant
            
            # Ensure we have at least 2 boundaries
            if len(boundaries) < 2:
                boundaries = [0, n//2, n]
            
            # Calculate organization score
            organization_score = 0.0
            
            # Measure how well power is organized around these boundaries
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i+1]
                
                if start >= end or start >= n or end > n:
                    continue
                
                # Calculate power in this layer
                layer_power = np.sum(power_spectrum[start:end])
                
                # Calculate expected power (uniform distribution)
                expected_power = (end - start) / float(n)
                
                # Calculate ratio (how much this layer deviates from uniform)
                if expected_power > 0:
                    power_ratio = layer_power / expected_power
                    
                    # Contribution to score (closer to constant = higher score)
                    if power_ratio > 0:
                        contribution = np.exp(-np.abs(np.log(power_ratio) - np.log(constant)))
                        organization_score += contribution
            
            # Normalize score to 0-1 range
            if len(boundaries) > 1:
                organization_score /= (len(boundaries) - 1)
            
            # Cache the result
            try:
                if cache_key:
                    self.cache[cache_key] = organization_score
            except:
                pass
            
            return organization_score
            
        except Exception as e:
            print("Error calculating architecture score: {}".format(str(e)))
            return 0.0
    
    def shuffle_data(self, data, seed=None):
        """
        Shuffle data while preserving statistical properties.
        
        Args:
            data: Data to shuffle
            seed: Random seed for reproducibility
            
        Returns:
            Shuffled data with same statistical properties
        """
        # Ensure data is a numpy array
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except:
                return np.array([])
        
        # Use cache if available
        try:
            cache_key = "shuffle_{}".format(seed)
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            
            self.cache_misses += 1
        except:
            # If caching fails, continue without it
            pass
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Get data length
        N = len(data)
        
        try:
            # Create surrogate data using Fourier shuffling to preserve power spectrum
            # Convert to frequency domain
            fft_data = np.fft.rfft(data)
            
            # Get amplitudes and phases
            amplitudes = np.abs(fft_data)
            phases = np.angle(fft_data)
            
            # Randomize phases while preserving amplitudes
            random_phases = np.random.uniform(0, 2 * np.pi, len(phases))
            
            # Reconstruct with random phases
            fft_random = amplitudes * np.exp(1j * random_phases)
            
            # Convert back to time domain
            shuffled_data = np.fft.irfft(fft_random, n=N)
            
            # Normalize to match original data statistics
            shuffled_data = (shuffled_data - np.mean(shuffled_data)) / np.std(shuffled_data)
            shuffled_data = shuffled_data * np.std(data) + np.mean(data)
            
            # Cache result
            try:
                if cache_key:
                    self.cache[cache_key] = shuffled_data
            except:
                pass
            
            return shuffled_data
            
        except Exception as e:
            print("Error shuffling data: {}".format(str(e)))
            return np.copy(data)  # Return original data if shuffling fails
    
    def run_monte_carlo_simulation(self, data, constant, num_simulations=None):
        """
        Run Monte Carlo simulations to assess statistical significance.
        
        Args:
            data: Original data
            constant: Mathematical constant to test
            num_simulations: Number of simulations to run
            
        Returns:
            dict: Simulation results including p-value
        """
        # Ensure data is a numpy array
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except:
                return {"error": "Invalid data format"}
        
        if num_simulations is None:
            num_simulations = self.config['num_simulations']
        
        # Get constant name for display
        constant_name = None
        for name, value in self.constants.items():
            if abs(value - constant) < 1e-10:
                constant_name = name
                break
        
        if constant_name is None:
            constant_name = "Constant({:.6f})".format(constant)
        
        print("Running {} simulations for {}...".format(num_simulations, constant_name))
        
        # Calculate actual score for original data
        print("Calculating architecture score for original data...")
        actual_score = self.calculate_architecture_score(data, constant)
        print("Original data score: {:.6f}".format(actual_score))
        
        # Initialize results
        surrogate_scores = []
        
        # Create progress file
        progress_file = os.path.join(self.config['output_dir'], "{}_progress.txt".format(constant_name.lower().replace(' ', '_')))
        with open(progress_file, 'w') as f:
            f.write("Starting Monte Carlo simulation with {} surrogates for {}\n".format(num_simulations, constant_name))
            f.write("Original data score: {:.6f}\n".format(actual_score))
        
        # Always use sequential processing for reliability
        print("Using sequential processing for reliability...")
        start_time = time.time()
        
        for i in range(num_simulations):
            # Update progress periodically
            if i % 5 == 0 or i == num_simulations - 1:
                progress = 100.0 * (i + 1) / num_simulations
                elapsed_time = time.time() - start_time
                
                # Estimate remaining time
                if i > 0:
                    avg_time_per_sim = elapsed_time / (i + 1)
                    remaining_sims = num_simulations - (i + 1)
                    estimated_remaining = avg_time_per_sim * remaining_sims
                    
                    print("Progress: {:.1f}% ({}/{}) - Elapsed: {:.1f}s - Est. remaining: {:.1f}s".format(
                        progress, i + 1, num_simulations, elapsed_time, estimated_remaining))
                    
                    with open(progress_file, 'a') as f:
                        f.write("Progress: {:.1f}% ({}/{}) - Elapsed: {:.1f}s - Est. remaining: {:.1f}s\n".format(
                            progress, i + 1, num_simulations, elapsed_time, estimated_remaining))
                else:
                    print("Progress: {:.1f}% ({}/{}) - Elapsed: {:.1f}s".format(
                        progress, i + 1, num_simulations, elapsed_time))
                    
                    with open(progress_file, 'a') as f:
                        f.write("Progress: {:.1f}% ({}/{}) - Elapsed: {:.1f}s\n".format(
                            progress, i + 1, num_simulations, elapsed_time))
            
            # Run a single simulation
            seed = i + 1000  # Use a consistent seed pattern for reproducibility
            surrogate_data = self.shuffle_data(data, seed)
            score = self.calculate_architecture_score(surrogate_data, constant)
            surrogate_scores.append(score)
            
            # Check for early stopping if enabled
            if self.config['early_stopping'] and len(surrogate_scores) >= 100:
                # Calculate current p-value
                count_higher = sum(1 for s in surrogate_scores if s >= actual_score)
                current_p = float(count_higher) / len(surrogate_scores)
                
                # If clearly not significant, stop early
                if current_p > 0.1 and len(surrogate_scores) >= 1000:
                    print("Early stopping at {} simulations: p-value = {:.6f}".format(
                        len(surrogate_scores), current_p))
                    break
        
        # Calculate p-value
        count_higher = sum(1 for score in surrogate_scores if score >= actual_score)
        p_value = float(count_higher) / len(surrogate_scores)
        
        # Calculate z-score
        if len(surrogate_scores) > 1:
            mean_surrogate = np.mean(surrogate_scores)
            std_surrogate = np.std(surrogate_scores)
            if std_surrogate > 0:
                z_score = (actual_score - mean_surrogate) / std_surrogate
            else:
                z_score = 0.0
        else:
            z_score = 0.0
        
        # Print final results
        print("\nResults for {}:".format(constant_name))
        print("Actual score: {:.6f}".format(actual_score))
        print("Mean surrogate score: {:.6f}".format(np.mean(surrogate_scores)))
        print("P-value: {:.6f}".format(p_value))
        print("Z-score: {:.4f}".format(z_score))
        print("Significant: {}".format("Yes" if p_value < self.config['significance_level'] else "No"))
        print("Total time: {:.1f}s".format(time.time() - start_time))
        
        # Save final results to progress file
        with open(progress_file, 'a') as f:
            f.write("\nFinal Results:\n")
            f.write("Actual score: {:.6f}\n".format(actual_score))
            f.write("Mean surrogate score: {:.6f}\n".format(np.mean(surrogate_scores)))
            f.write("P-value: {:.6f}\n".format(p_value))
            f.write("Z-score: {:.4f}\n".format(z_score))
            f.write("Significant: {}\n".format("Yes" if p_value < self.config['significance_level'] else "No"))
            f.write("Total time: {:.1f}s\n".format(time.time() - start_time))
        
        # Return results
        return {
            'actual_score': actual_score,
            'surrogate_scores': surrogate_scores,
            'p_value': p_value,
            'z_score': z_score,
            'num_simulations_run': len(surrogate_scores),
            'significant': p_value < self.config['significance_level']
        }
    
    def run_full_test(self, data, output_dir=None):
        """
        Run the full Information Architecture Test on the data.
        
        Args:
            data: CMB power spectrum or time series
            output_dir: Directory to save results and visualizations
            
        Returns:
            dict: Complete test results
        """
        results = {}
        
        # Create a summary file
        if output_dir:
            ensure_dir_exists(output_dir)
            summary_file = os.path.join(output_dir, "summary_report.txt")
            with open(summary_file, 'w') as f:
                f.write("INFORMATION ARCHITECTURE TEST RESULTS\n")
                f.write("====================================\n\n")
                f.write("Individual Constant Results:\n\n")
        
        # Test each constant individually
        for constant_name, constant_value in self.constants.items():
            print("Testing {}...".format(constant_name))
            
            # Run Monte Carlo simulation
            simulation_results = self.run_monte_carlo_simulation(data, constant_value)
            
            # Store results
            results[constant_name] = simulation_results
            
            # Print summary
            print("  - Architecture score: {:.6f}".format(simulation_results['actual_score']))
            print("  - p-value: {:.6f}".format(simulation_results['p_value']))
            print("  - Significant: {}".format(simulation_results['significant']))
            print("  - Simulations run: {}".format(simulation_results['num_simulations_run']))
            
            # Write to summary file
            if output_dir:
                with open(summary_file, 'a') as f:
                    f.write("{}:\n".format(constant_name.upper()))
                    f.write("  Architecture Score: {:.6f}\n".format(simulation_results['actual_score']))
                    f.write("  p-value: {:.6f}\n".format(simulation_results['p_value']))
                    f.write("  Significant: {}\n".format(simulation_results['significant']))
                    f.write("  Simulations Run: {}\n\n".format(simulation_results['num_simulations_run']))
        
        # Generate visualizations if output directory is provided
        if output_dir:
            self.generate_visualizations(results, output_dir)
        
        # Store results
        self.results = results
        
        return results
    
    def generate_visualizations(self, results, output_dir):
        """
        Generate visualizations of test results.
        
        Args:
            results: Test results
            output_dir: Directory to save visualizations
        """
        ensure_dir_exists(output_dir)
        
        # 1. Bar chart of architecture scores for each constant
        plt.figure(figsize=(12, 6))
        constants = [c for c in self.constants.keys() if c in results]
        scores = [results[c]['actual_score'] for c in constants]
        p_values = [results[c]['p_value'] for c in constants]
        
        bars = plt.bar(constants, scores, alpha=0.7)
        
        # Highlight significant results
        for i, p in enumerate(p_values):
            if p < self.config['significance_level']:
                bars[i].set_color('red')
        
        plt.title('Information Architecture Scores by Mathematical Constant')
        plt.xlabel('Mathematical Constant')
        plt.ylabel('Architecture Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add p-values as text
        for i, (score, p) in enumerate(zip(scores, p_values)):
            plt.text(i, score + 0.01, "p={:.4f}".format(p), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'architecture_scores.png'), dpi=300)
        plt.close()
        
        # 3. Simulation histogram for phi and sqrt2
        for constant in ['phi', 'sqrt2']:
            if constant in results:
                plt.figure(figsize=(10, 6))
                
                simulation_scores = results[constant]['surrogate_scores']
                actual_score = results[constant]['actual_score']
                
                plt.hist(simulation_scores, bins=30, alpha=0.7, color='gray')
                plt.axvline(actual_score, color='red', linestyle='--', linewidth=2)
                
                plt.title('Monte Carlo Simulation Results for {}'.format(constant))
                plt.xlabel('Architecture Score')
                plt.ylabel('Frequency')
                plt.text(actual_score, plt.ylim()[1]*0.9, "Actual Score: {:.4f}".format(actual_score), 
                         ha='right', va='top', color='red')
                plt.text(actual_score, plt.ylim()[1]*0.8, "p-value: {:.4f}".format(results[constant]["p_value"]), 
                         ha='right', va='top', color='red')
                
                plt.grid(linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, '{}_simulation.png'.format(constant)), dpi=300)
                plt.close()
        
        # 4. Summary report
        with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
            f.write("INFORMATION ARCHITECTURE TEST RESULTS\n")
            f.write("====================================\n\n")
            
            f.write("Individual Constant Results:\n")
            for constant in constants:
                f.write("\n{}:\n".format(constant.upper()))
                f.write("  Architecture Score: {:.6f}\n".format(results[constant]['actual_score']))
                f.write("  p-value: {:.6f}\n".format(results[constant]['p_value']))
                f.write("  Significant: {}\n".format(results[constant]['p_value'] < self.config['significance_level']))
                f.write("  Simulations Run: {}\n".format(results[constant]['num_simulations_run']))
            
            f.write("\nTest Configuration:\n")
            for key, value in self.config.items():
                f.write("  {}: {}\n".format(key, value))

def run_information_architecture_test(wmap_data=None, planck_data=None, num_surrogates=100):
    """
    Run the Information Architecture Test on WMAP and Planck data.
    
    Args:
        wmap_data: WMAP data (optional)
        planck_data: Planck data (optional)
        num_surrogates: Number of surrogate datasets to generate
        
    Returns:
        dict: Test results
    """
    print("\nRunning Information Architecture Test with {} surrogates...".format(num_surrogates))
    
    # Ensure output directory exists
    output_dir = "../results/information_architecture_test"
    ensure_dir_exists(output_dir)
    
    # Configure test
    config = {
        'significance_level': 0.05,
        'num_simulations': num_surrogates,
        'output_dir': output_dir,
        'early_stopping': False,  # Ensure all simulations run to completion
        'parallel_processing': False,  # Disable parallel processing
        'num_processes': 4,
        'batch_size': 100,
        'timeout': 3600,              # Timeout in seconds (1 hour)
    }
    
    test = InformationArchitectureTest(config)
    
    # Initialize results
    results = {
        'wmap': {},
        'planck': {},
        'cache_performance': {
            'hits': 0,
            'misses': 0,
            'ratio': 0.0
        }
    }
    
    # Process WMAP data if provided
    if wmap_data is not None:
        print("Loading WMAP data...")
        
        # Test each constant
        print("Running Information Architecture Test on WMAP data...")
        
        for name, value in test.constants.items():
            print("\nTesting {}...".format(name))
            start_time = time.time()
            
            # Run test for this constant
            constant_results = test.run_monte_carlo_simulation(wmap_data, value)
            
            # Add to results
            results['wmap'][name] = constant_results
            
            # Print summary
            elapsed = time.time() - start_time
            print("Completed {} test in {:.2f} seconds ({:.2f} minutes)".format(
                name, elapsed, elapsed / 60.0))
            print("Score: {:.6f}, p-value: {:.6f}, significant: {}".format(
                constant_results['actual_score'],
                constant_results['p_value'],
                "Yes" if constant_results['significant'] else "No"
            ))
        
        # Save WMAP results
        wmap_results_file = os.path.join(output_dir, "wmap_results.json")
        with open(wmap_results_file, 'w') as f:
            json.dump(results['wmap'], f, indent=2)
        print("\nWMAP results saved to {}".format(wmap_results_file))
    
    # Process Planck data if provided
    if planck_data is not None:
        print("\nLoading Planck data...")
        
        # Test each constant
        print("Running Information Architecture Test on Planck data...")
        
        for name, value in test.constants.items():
            print("\nTesting {}...".format(name))
            start_time = time.time()
            
            # Run test for this constant
            constant_results = test.run_monte_carlo_simulation(planck_data, value)
            
            # Add to results
            results['planck'][name] = constant_results
            
            # Print summary
            elapsed = time.time() - start_time
            print("Completed {} test in {:.2f} seconds ({:.2f} minutes)".format(
                name, elapsed, elapsed / 60.0))
            print("Score: {:.6f}, p-value: {:.6f}, significant: {}".format(
                constant_results['actual_score'],
                constant_results['p_value'],
                "Yes" if constant_results['significant'] else "No"
            ))
        
        # Save Planck results
        planck_results_file = os.path.join(output_dir, "planck_results.json")
        with open(planck_results_file, 'w') as f:
            json.dump(results['planck'], f, indent=2)
        print("\nPlanck results saved to {}".format(planck_results_file))
    
    # Get cache performance
    if hasattr(test, 'cache_hits') and hasattr(test, 'cache_misses'):
        results['cache_performance']['hits'] = test.cache_hits
        results['cache_performance']['misses'] = test.cache_misses
        total = test.cache_hits + test.cache_misses
        if total > 0:
            results['cache_performance']['ratio'] = float(test.cache_hits) / total
    
    # Save combined results
    combined_results_file = os.path.join(output_dir, "combined_results.json")
    with open(combined_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nCombined results saved to {}".format(combined_results_file))
    
    return results

def create_sample_data(dataset):
    """Create sample data for testing."""
    if dataset == "wmap":
        # Create WMAP-like sample data
        multipoles = np.arange(2, 1201)
        power = 1000 * np.power(multipoles, -0.6) * (1 + 0.1 * np.sin(multipoles / 10.0))
        error = power * 0.05  # 5% error
        return power  # Return just the power spectrum as a numpy array
    else:
        # Create Planck-like sample data
        multipoles = np.arange(2, 2501)
        power = 1000 * np.power(multipoles, -0.65) * (1 + 0.08 * np.sin(multipoles / 12.0))
        error = power * 0.05  # 5% error
        return power  # Return just the power spectrum as a numpy array

def get_most_significant(constants_results):
    """Get the most significant constant from results."""
    min_p_value = 1.0
    most_significant = None
    
    for constant, results in constants_results.items():
        if results['p_value'] < min_p_value:
            min_p_value = results['p_value']
            most_significant = constant
    
    return most_significant

def count_significant(constants_results):
    """Count the number of significant constants."""
    return sum(1 for results in constants_results.values() if results['significant'])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Information Architecture Test on CMB data")
    
    # Data file arguments
    parser.add_argument("--wmap-file", help="Path to WMAP data file")
    parser.add_argument("--planck-file", help="Path to Planck data file")
    
    # Dataset selection arguments
    parser.add_argument("--wmap-only", action="store_true", 
                        help="Only process WMAP data")
    parser.add_argument("--planck-only", action="store_true",
                        help="Only process Planck data")
    
    # Optional arguments
    parser.add_argument("--output-dir", default="../results/information_architecture_test",
                        help="Directory to save results")
    parser.add_argument("--num-surrogates", type=int, default=1000,
                        help="Number of surrogate datasets to generate")
    parser.add_argument("--parallel", action="store_true", 
                        help="Enable parallel processing")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Enable early stopping if result is not significant")
    
    return parser.parse_args()

def main():
    """Run the Information Architecture Test on CMB data."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print configuration
    print("\nInformation Architecture Test Configuration:")
    print("-------------------------------------------")
    print("Number of surrogate datasets: {}".format(args.num_surrogates))
    print("Parallel processing: {}".format("Enabled" if args.parallel else "Disabled"))
    print("Early stopping: {}".format("Enabled" if args.early_stopping else "Disabled"))
    print("Output directory: {}".format(args.output_dir))
    print("")
    
    # Create output directory
    ensure_dir_exists(args.output_dir)
    
    # Initialize WMAP and Planck data as None
    wmap_data = None
    planck_data = None
    
    # Try to load WMAP data if file path provided
    if args.wmap_file and os.path.exists(args.wmap_file):
        print("Loading WMAP data from {}".format(args.wmap_file))
        ell, power, error = load_wmap_power_spectrum(args.wmap_file)
        if ell is not None:
            wmap_data = power
    
    # Try to load Planck data if file path provided
    if args.planck_file and os.path.exists(args.planck_file):
        print("Loading Planck data from {}".format(args.planck_file))
        ell, power, error = load_planck_power_spectrum(args.planck_file)
        if ell is not None:
            planck_data = power
    
    # If no data files provided or loading failed, create sample data
    if wmap_data is None:
        print("No WMAP data file provided or loading failed. Creating sample WMAP data.")
        wmap_data = create_sample_data("wmap")
    
    if planck_data is None:
        print("No Planck data file provided or loading failed. Creating sample Planck data.")
        planck_data = create_sample_data("planck")
    
    # Determine which datasets to process
    process_wmap = not args.planck_only
    process_planck = not args.wmap_only
    
    # Run the test
    start_time = time.time()
    results = run_information_architecture_test(
        wmap_data=wmap_data if process_wmap else None,
        planck_data=planck_data if process_planck else None,
        num_surrogates=args.num_surrogates
    )
    total_time = time.time() - start_time
    
    # Print summary
    print("\nTest Summary:")
    print("-------------")
    
    if process_wmap and 'wmap' in results:
        print("WMAP Results:")
        for constant, data in results['wmap'].items():
            print("- {} ({}): Score {:.6f}, p-value {:.6f}".format(
                constant,
                "Significant" if data['significant'] else "Not Significant",
                data['actual_score'],
                data['p_value']
            ))
    
    if process_planck and 'planck' in results:
        print("\nPlanck Results:")
        for constant, data in results['planck'].items():
            print("- {} ({}): Score {:.6f}, p-value {:.6f}".format(
                constant,
                "Significant" if data['significant'] else "Not Significant",
                data['actual_score'],
                data['p_value']
            ))
    
    if 'cache_performance' in results:
        print("\nCache Performance:")
        print("- Hit ratio: {:.2f}%".format(100 * results['cache_performance']['ratio']))
    
    print("\nTotal execution time: {:.2f} seconds ({:.2f} minutes)".format(
        total_time, total_time / 60.0))
    print("Results saved to: {}".format(args.output_dir))

if __name__ == "__main__":
    main()
