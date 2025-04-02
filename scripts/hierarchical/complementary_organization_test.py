import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft, stats
from scipy import signal
from datetime import datetime
import multiprocessing
from multiprocessing import Pool, cpu_count
import signal

# Import utility functions
def ensure_dir_exists(dir_path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Define data loading functions
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

class ComplementaryOrganizationTest:
    """
    Tests whether the Golden Ratio (phi) and Square Root of 2 (sqrt2) exhibit complementary
    organizational roles in the cosmic microwave background, with phi optimizing direct
    relationships between pairs of scales and sqrt2 optimizing cascade pathways.
    """
    
    def __init__(self, config=None):
        """Initialize the Complementary Organization Test with configuration parameters."""
        self.constants = {
            'phi': 1.618033988749895,  # Golden ratio
            'sqrt2': 1.4142135623730951,  # Square root of 2
            'e': 2.718281828459045,    # Euler's number
            'pi': 3.141592653589793,   # Pi
            'sqrt3': 1.7320508075688772,  # Square root of 3
            'ln2': 0.6931471805599453   # Natural log of 2
        }
        
        # Default configuration
        self.config = {
            'num_simulations': 10000,     # Number of Monte Carlo simulations
            'cascade_depth': 5,           # Number of scales in a cascade
            'significance_level': 0.05,   # Statistical significance threshold
            'parallel_processing': True,  # Whether to use parallel processing
            'batch_size': 1000,           # Batch size for parallel processing
            'early_stopping': False,       # Whether to stop early if significance is reached
            'timeout': 3600,              # Timeout in seconds (1 hour)
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Initialize results storage
        self.results = {}
        
    def measure_pairwise_optimization(self, data, constant):
        """
        Measure how well a constant optimizes direct relationships between pairs of scales.
        
        Args:
            data: CMB power spectrum or time series
            constant: Mathematical constant to test
            
        Returns:
            dict: Optimization metrics
        """
        pairwise_scores = []
        
        # Test for a range of scales
        base_scales = [10, 20, 50, 100, 200]
        for base_scale in base_scales:
            # Skip if base scale is out of bounds
            if base_scale >= len(data):
                continue
                
            # Calculate related scale based on constant
            related_scale = int(base_scale * constant)
            if related_scale >= len(data):
                continue
            
            # Measure transfer entropy between these scales
            transfer_entropy = self._estimate_transfer_entropy(
                data[base_scale-5:base_scale+5], 
                data[related_scale-5:related_scale+5]
            )
            
            # Measure coherence between these scales
            coherence = self._estimate_coherence(
                data[base_scale-10:base_scale+10], 
                data[related_scale-10:related_scale+10]
            )
            
            # Combine into a single pairwise optimization score
            pairwise_score = (transfer_entropy + coherence) / 2
            pairwise_scores.append(pairwise_score)
        
        # Return average and maximum scores
        if not pairwise_scores:
            return {
                'avg_pairwise_score': 0,
                'max_pairwise_score': 0,
                'num_pairs': 0
            }
            
        return {
            'avg_pairwise_score': np.mean(pairwise_scores),
            'max_pairwise_score': np.max(pairwise_scores),
            'num_pairs': len(pairwise_scores)
        }
    
    def measure_cascade_optimization(self, data, constant):
        """
        Measure how well a constant optimizes cascade pathways across multiple scales.
        
        Args:
            data: CMB power spectrum or time series
            constant: Mathematical constant to test
            
        Returns:
            dict: Optimization metrics
        """
        cascade_scores = []
        
        # Test for a range of scales
        base_scales = [10, 20, 50, 100, 200]
        for base_scale in base_scales:
            # Generate cascade of scales
            cascade = self._generate_cascade(base_scale, constant)
            
            # Skip if any scale is out of bounds
            if any(scale >= len(data) for scale in cascade):
                continue
            
            # Extract data at cascade scales
            cascade_data = [data[scale] for scale in cascade]
            
            # Measure cascade resonance
            resonance = self._estimate_resonance(cascade_data)
            
            # Measure cascade transfer entropy
            cascade_te = self._estimate_cascade_transfer_entropy(data, cascade)
            
            # Combine into a single cascade optimization score
            cascade_score = (resonance + cascade_te) / 2
            cascade_scores.append(cascade_score)
        
        # Return average and maximum scores
        if not cascade_scores:
            return {
                'avg_cascade_score': 0,
                'max_cascade_score': 0,
                'num_cascades': 0
            }
            
        return {
            'avg_cascade_score': np.mean(cascade_scores),
            'max_cascade_score': np.max(cascade_scores),
            'num_cascades': len(cascade_scores)
        }
    
    def measure_combined_optimization(self, data):
        """
        Measure the combined optimization when both phi and sqrt2 work together.
        
        Args:
            data: CMB power spectrum or time series
            
        Returns:
            dict: Combined optimization metrics
        """
        phi = self.constants['phi']
        sqrt2 = self.constants['sqrt2']
        
        # Measure individual optimization
        phi_pairwise = self.measure_pairwise_optimization(data, phi)
        sqrt2_cascade = self.measure_cascade_optimization(data, sqrt2)
        
        # Calculate combined effect (interaction between phi pairs and sqrt2 cascades)
        combined_score = 0
        count = 0
        
        # Test for interactions between phi-related pairs and sqrt2 cascades
        base_scales = [10, 20, 50, 100, 200]
        for base_scale in base_scales:
            # Get phi-related scale
            phi_scale = int(base_scale * phi)
            if phi_scale >= len(data):
                continue
                
            # Generate sqrt2 cascade starting from the phi-related scale
            cascade = self._generate_cascade(phi_scale, sqrt2)
            
            # Skip if any scale is out of bounds
            if any(scale >= len(data) for scale in cascade):
                continue
            
            # Measure how phi pairs interact with sqrt2 cascades
            interaction_score = self._measure_cross_interaction(
                data, base_scale, phi_scale, cascade
            )
            
            combined_score += interaction_score
            count += 1
        
        # Calculate average combined score
        avg_combined = combined_score / count if count > 0 else 0
        
        return {
            'phi_pairwise': phi_pairwise,
            'sqrt2_cascade': sqrt2_cascade,
            'combined_score': avg_combined,
            'num_interactions': count
        }

    def _measure_cross_interaction(self, data, base_scale, phi_scale, cascade):
        """
        Measure the interaction between a phi-related pair and a sqrt2 cascade.
        
        Args:
            data: CMB power spectrum or time series
            base_scale: Starting scale
            phi_scale: Scale related to base by phi
            cascade: Cascade of scales starting from phi_scale
            
        Returns:
            float: Interaction score
        """
        # Skip if scales are out of bounds
        if base_scale >= len(data) or phi_scale >= len(data) or any(scale >= len(data) for scale in cascade):
            return 0
        
        # Calculate information flow from base scale to the cascade
        base_window = data[base_scale-5:base_scale+5] if base_scale >= 5 else data[:base_scale+5]
        
        cascade_flows = []
        for scale in cascade:
            if scale >= len(data):
                continue
                
            scale_window = data[scale-5:scale+5] if scale >= 5 else data[:scale+5]
            flow = self._estimate_transfer_entropy(base_window, scale_window)
            cascade_flows.append(flow)
        
        # Calculate coherence across the entire system
        all_scales = [base_scale, phi_scale] + cascade
        all_scale_data = [data[scale] for scale in all_scales if scale < len(data)]
        
        system_coherence = self._estimate_system_coherence(all_scale_data)
        
        # Combine transfer entropy flows and system coherence
        avg_flow = np.mean(cascade_flows) if cascade_flows else 0
        
        return (avg_flow + system_coherence) / 2
    
    def _generate_cascade(self, base_scale, constant, depth=None):
        """
        Generate a cascade of scales related by a mathematical constant.
        
        Args:
            base_scale: Starting scale
            constant: Mathematical constant to use
            depth: Number of scales in cascade (default: use config value)
            
        Returns:
            list: Cascade of scales
        """
        if depth is None:
            depth = self.config['cascade_depth']
            
        cascade = [base_scale]
        for i in range(1, depth):
            next_scale = int(base_scale * (constant ** i))
            cascade.append(next_scale)
            
        return cascade
    
    def _estimate_transfer_entropy(self, source, target, lag=1):
        """
        Estimate transfer entropy from source to target time series.
        
        Args:
            source: Source time series
            target: Target time series
            lag: Time lag
            
        Returns:
            float: Estimated transfer entropy
        """
        # Early return if series are too short
        if len(source) < 5 or len(target) < 5:
            return 0
            
        # Trim to same length
        min_length = min(len(source), len(target))
        source = source[:min_length]
        target = target[:min_length]
        
        # Calculate lagged series
        if min_length <= lag:
            return 0
            
        source_past = source[:-lag]
        target_past = target[:-lag]
        target_future = target[lag:]
        
        # Trim to same length
        min_len = min(len(source_past), len(target_past), len(target_future))
        if min_len < 3:
            return 0
            
        source_past = source_past[:min_len]
        target_past = target_past[:min_len]
        target_future = target_future[:min_len]
        
        try:
            # Calculate H(Target_future | Target_past)
            corr_target, _ = stats.pearsonr(target_past, target_future)
            h_target = 1 - abs(corr_target)
            
            # Calculate H(Target_future | Target_past, Source_past)
            X = np.column_stack((target_past, source_past))
            y = target_future
            
            # Check for constant columns
            if np.all(np.std(X, axis=0) > 1e-10):
                model = np.linalg.lstsq(X, y, rcond=None)[0]
                predictions = X.dot(model)
                residuals = y - predictions
                h_combined = np.std(residuals)
                
                # Transfer entropy is the reduction in uncertainty
                te = max(0, h_target - h_combined)
                return te
            else:
                return 0
        except:
            return 0
    
    def _estimate_coherence(self, series1, series2):
        """
        Estimate coherence between two time series.
        
        Args:
            series1: First time series
            series2: Second time series
            
        Returns:
            float: Estimated coherence
        """
        # Early return if series are too short
        if len(series1) < 5 or len(series2) < 5:
            return 0
            
        # Trim to same length
        min_length = min(len(series1), len(series2))
        series1 = series1[:min_length]
        series2 = series2[:min_length]
        
        try:
            # Calculate coherence using correlation
            corr, _ = stats.pearsonr(series1, series2)
            return abs(corr)
        except:
            return 0
    
    def _estimate_resonance(self, cascade_data):
        """
        Estimate resonance pattern in a cascade of data points.
        
        Args:
            cascade_data: List of data points in a cascade
            
        Returns:
            float: Estimated resonance
        """
        if len(cascade_data) < 3:
            return 0
            
        try:
            # Calculate power spectrum
            f, Pxx = signal.welch(cascade_data, nperseg=min(len(cascade_data), 4))
            
            # Calculate the dominant frequency component
            dominant_idx = np.argmax(Pxx[1:]) + 1  # Skip DC component
            dominant_power = Pxx[dominant_idx]
            
            # Calculate the ratio of dominant frequency power to total power
            total_power = np.sum(Pxx)
            if total_power == 0:
                return 0
                
            resonance_score = dominant_power / total_power
            return resonance_score
        except:
            return 0
    
    def _estimate_cascade_transfer_entropy(self, data, cascade):
        """
        Estimate the transfer entropy through a cascade of scales.
        
        Args:
            data: Full data series
            cascade: List of scale indices
            
        Returns:
            float: Estimated cascade transfer entropy
        """
        total_te = 0
        count = 0
        
        for i in range(len(cascade) - 1):
            scale1 = cascade[i]
            scale2 = cascade[i + 1]
            
            if scale1 >= len(data) or scale2 >= len(data):
                continue
                
            window1 = data[scale1-5:scale1+5] if scale1 >= 5 else data[:scale1+5]
            window2 = data[scale2-5:scale2+5] if scale2 >= 5 else data[:scale2+5]
            
            te = self._estimate_transfer_entropy(window1, window2)
            total_te += te
            count += 1
        
        return total_te / count if count > 0 else 0
    
    def _estimate_system_coherence(self, scale_data):
        """
        Estimate the coherence across an entire system of scales.
        
        Args:
            scale_data: List of data values at different scales
            
        Returns:
            float: Estimated system coherence
        """
        if len(scale_data) < 3:
            return 0
            
        total_coherence = 0
        count = 0
        
        for i in range(len(scale_data)):
            for j in range(i + 1, len(scale_data)):
                coh = self._estimate_coherence([scale_data[i]], [scale_data[j]])
                total_coherence += coh
                count += 1
        
        return total_coherence / count if count > 0 else 0

    def run_test(self, data):
        """
        Run the Complementary Organization Test on the provided data.
        
        Args:
            data: CMB power spectrum or time series
            
        Returns:
            dict: Test results
        """
        print("Running Complementary Organization Test...")
        
        # Initialize results dictionary
        self.results = {
            'original_scores': {},
            'surrogate_results': [],
            'p_values': {},
            'num_simulations': 0
        }
        
        # Measure optimization on original data
        print("Measuring optimization on original data...")
        phi = self.constants['phi']
        sqrt2 = self.constants['sqrt2']
        
        # Store original data
        self.original_data = data
        
        # Measure individual optimization
        phi_pairwise = self.measure_pairwise_optimization(data, phi)
        sqrt2_cascade = self.measure_cascade_optimization(data, sqrt2)
        combined = self.measure_combined_optimization(data)
        
        # Store original scores
        self.results['original_scores'] = {
            'phi_pairwise': phi_pairwise,
            'sqrt2_cascade': sqrt2_cascade,
            'combined': combined
        }
        
        # Run Monte Carlo simulations
        print("Running {} Monte Carlo simulations...".format(self.config.get('num_simulations', 10000)))
        
        # Run simulations (sequential or parallel)
        if self.config.get('parallel_processing', True):
            self.results = self._run_parallel_simulations(data, self.results)
        else:
            self.results = self._run_sequential_simulations(data, self.results)
        
        return self.results
    
    def _run_sequential_simulations(self, data, results):
        """
        Run Monte Carlo simulations sequentially.
        
        Args:
            data: Original CMB power spectrum or time series
            results: Results dictionary to update
            
        Returns:
            dict: Updated results
        """
        phi = self.constants['phi']
        sqrt2 = self.constants['sqrt2']
        
        original_phi_score = results['original_scores']['phi_pairwise']['avg_pairwise_score']
        original_sqrt2_score = results['original_scores']['sqrt2_cascade']['avg_cascade_score']
        original_combined_score = results['original_scores']['combined']['combined_score']
        
        # Count simulations with scores exceeding original
        phi_pairwise_count = 0
        sqrt2_cascade_count = 0
        combined_count = 0
        
        # Set up early stopping
        early_stopping = self.config.get('early_stopping', False)
        min_simulations = 1000  # Minimum number of simulations before early stopping
        
        # Start simulation loop
        for i in range(self.config.get('num_simulations', 10000)):
            # Create phase-randomized surrogate
            surrogate = self._create_surrogate(data)
            
            # Measure optimization on surrogate
            surrogate_phi_pairwise = self.measure_pairwise_optimization(surrogate, phi)
            surrogate_sqrt2_cascade = self.measure_cascade_optimization(surrogate, sqrt2)
            surrogate_combined = self.measure_combined_optimization(surrogate)
            
            # Update counts
            if surrogate_phi_pairwise['avg_pairwise_score'] >= original_phi_score:
                phi_pairwise_count += 1
                
            if surrogate_sqrt2_cascade['avg_cascade_score'] >= original_sqrt2_score:
                sqrt2_cascade_count += 1
                
            if surrogate_combined['combined_score'] >= original_combined_score:
                combined_count += 1
            
            # Print progress
            if (i + 1) % 100 == 0:
                print("Completed {}/{} simulations".format(i + 1, self.config.get('num_simulations', 10000)))
                
                # Calculate current p-values
                phi_p = phi_pairwise_count / (i + 1)
                sqrt2_p = sqrt2_cascade_count / (i + 1)
                combined_p = combined_count / (i + 1)
                
                print("Current p-values: phi_pairwise={:.6f}, sqrt2_cascade={:.6f}, combined={:.6f}".format(
                    phi_p, sqrt2_p, combined_p
                ))
            
            # Check for early stopping if enabled and after minimum simulations
            if early_stopping:
                # Calculate current p-values
                phi_p = phi_pairwise_count / (i + 1)
                sqrt2_p = sqrt2_cascade_count / (i + 1)
                combined_p = combined_count / (i + 1)
                
                # Stop if all p-values are significant or clearly non-significant
                if (phi_p < 0.01 and sqrt2_p < 0.01 and combined_p < 0.01) or \
                   (phi_p > 0.1 and sqrt2_p > 0.1 and combined_p > 0.1):
                    print("Early stopping at {} simulations".format(i + 1))
                    break
        
        # Calculate final p-values
        total_simulations = i + 1
        results['simulation_counts'] = {
            'phi_pairwise': phi_pairwise_count,
            'sqrt2_cascade': sqrt2_cascade_count,
            'combined': combined_count
        }
        
        results['p_values'] = {
            'phi_pairwise': phi_pairwise_count / total_simulations,
            'sqrt2_cascade': sqrt2_cascade_count / total_simulations,
            'combined': combined_count / total_simulations
        }
        
        results['total_simulations'] = total_simulations
        
        return results
    
    def _run_parallel_simulations(self, data, results):
        """
        Run Monte Carlo simulations in parallel.
        
        Args:
            data: Original CMB power spectrum data
            results: Dictionary to store results
            
        Returns:
            dict: Updated results dictionary
        """
        print("Running simulations in parallel with {} processes".format(multiprocessing.cpu_count()))
        
        # Setup simulation parameters
        num_simulations = self.config.get('num_simulations', 10000)
        early_stopping = self.config.get('early_stopping', False)
        num_processes = min(multiprocessing.cpu_count(), 8)  # Limit to 8 processes max
        
        # Extract original scores
        original_phi_score = results['original_scores']['phi_pairwise']['avg_pairwise_score']
        original_sqrt2_score = results['original_scores']['sqrt2_cascade']['avg_cascade_score']
        original_combined_score = results['original_scores']['combined']['combined_score']
        
        original_scores = {
            'phi_pairwise': original_phi_score,
            'sqrt2_cascade': original_sqrt2_score,
            'combined': original_combined_score
        }
        
        # Prepare simulation arguments
        simulation_args = [(data, self.constants, original_scores) for _ in range(num_simulations)]
        
        # Track counts for p-value calculation
        phi_pairwise_count = 0
        sqrt2_cascade_count = 0
        combined_count = 0
        total_simulations = 0
        
        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=num_processes)
        
        try:
            # Submit all tasks to the pool
            results_iter = pool.imap(simulation_worker, simulation_args)
            
            # Process results as they complete
            for phi_exceeds, sqrt2_exceeds, combined_exceeds in results_iter:
                if phi_exceeds:
                    phi_pairwise_count += 1
                if sqrt2_exceeds:
                    sqrt2_cascade_count += 1
                if combined_exceeds:
                    combined_count += 1
                
                total_simulations += 1
                
                # Print progress periodically
                if total_simulations % 10 == 0 or total_simulations == num_simulations:
                    # Calculate current p-values
                    phi_p = phi_pairwise_count / total_simulations
                    sqrt2_p = sqrt2_cascade_count / total_simulations
                    combined_p = combined_count / total_simulations
                    
                    print("Completed {}/{} simulations. Current p-values: phi_pairwise={:.6f}, sqrt2_cascade={:.6f}, combined={:.6f}".format(
                        total_simulations, num_simulations, phi_p, sqrt2_p, combined_p
                    ))
                    
                    # Update results
                    results['p_values'] = {
                        'phi_pairwise': phi_p,
                        'sqrt2_cascade': sqrt2_p,
                        'combined': combined_p
                    }
                    
                    # Check for early stopping if enabled
                    if early_stopping and total_simulations >= 100:
                        significant_p_values = (
                            phi_p < 0.01 or 
                            sqrt2_p < 0.01 or 
                            combined_p < 0.01
                        )
                        
                        if significant_p_values:
                            print("Early stopping: Significant p-values detected after {} simulations.".format(
                                total_simulations
                            ))
                            break
                
                # Check if we've reached the maximum number of simulations
                if total_simulations >= num_simulations:
                    break
            
            # Store final results
            results['simulation_counts'] = {
                'phi_pairwise': phi_pairwise_count,
                'sqrt2_cascade': sqrt2_cascade_count,
                'combined': combined_count
            }
            
            results['p_values'] = {
                'phi_pairwise': phi_pairwise_count / total_simulations,
                'sqrt2_cascade': sqrt2_cascade_count / total_simulations,
                'combined': combined_count / total_simulations
            }
            
            results['num_simulations'] = total_simulations
            
            return results
        finally:
            # Clean up
            pool.close()
            pool.join()
    
    def _create_surrogate(self, data):
        """
        Create a phase-randomized surrogate of the data.
        
        Args:
            data: Original time series
            
        Returns:
            array: Phase-randomized surrogate
        """
        # Perform FFT
        fft_data = np.fft.rfft(data)
        
        # Randomize phases
        magnitudes = np.abs(fft_data)
        phases = np.angle(fft_data)
        random_phases = np.random.uniform(0, 2*np.pi, len(phases))
        
        # Keep DC component phase unchanged
        random_phases[0] = phases[0]
        
        # Create new FFT data with random phases
        new_fft_data = magnitudes * np.exp(1j * random_phases)
        
        # Perform inverse FFT
        surrogate = np.fft.irfft(new_fft_data, n=len(data))
        
        return surrogate

    def visualize_results(self, output_dir):
        """
        Generate visualizations of the test results.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            list: Paths to generated visualization files
        """
        ensure_dir_exists(output_dir)
        
        # List to store paths to generated files
        visualization_files = []
        
        # Visualize original data
        if hasattr(self, 'original_data'):
            data_fig_path = os.path.join(output_dir, 'original_data.png')
            plt.figure(figsize=(12, 6))
            plt.plot(self.original_data)
            plt.title('Original CMB Power Spectrum')
            plt.xlabel('Multipole Moment Index')
            plt.ylabel('Power')
            plt.grid(True, alpha=0.3)
            plt.savefig(data_fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files.append(data_fig_path)
        
        # Visualize p-values
        if 'p_values' in self.results:
            p_values_fig_path = os.path.join(output_dir, 'p_values.png')
            
            # Extract p-values
            p_values = [
                self.results['p_values']['phi_pairwise'],
                self.results['p_values']['sqrt2_cascade'],
                self.results['p_values']['combined']
            ]
            
            labels = ['Phi Pairwise', 'Sqrt2 Cascade', 'Combined']
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, p_values, color=['#3498db', '#2ecc71', '#9b59b6'])
            
            # Add significance threshold line
            plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='p=0.05')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        '{:.6f}'.format(height), ha='center', va='bottom')
            
            plt.title('Complementary Organization Test p-values')
            plt.ylabel('p-value')
            plt.ylim(0, 1.1)
            plt.grid(True, axis='y', alpha=0.3)
            plt.legend()
            plt.savefig(p_values_fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files.append(p_values_fig_path)
        
        # Visualize optimization scores
        if 'original_scores' in self.results:
            scores_fig_path = os.path.join(output_dir, 'optimization_scores.png')
            
            # Extract scores
            phi_score = self.results['original_scores']['phi_pairwise']['avg_pairwise_score']
            sqrt2_score = self.results['original_scores']['sqrt2_cascade']['avg_cascade_score']
            combined_score = self.results['original_scores']['combined']['combined_score']
            
            scores = [phi_score, sqrt2_score, combined_score]
            labels = ['Phi Pairwise', 'Sqrt2 Cascade', 'Combined']
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, scores, color=['#3498db', '#2ecc71', '#9b59b6'])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        '{:.4f}'.format(height), ha='center', va='bottom')
            
            plt.title('Complementary Organization Optimization Scores')
            plt.ylabel('Score')
            plt.grid(True, axis='y', alpha=0.3)
            plt.savefig(scores_fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files.append(scores_fig_path)
        
        # Visualize phi vs sqrt2 complementary roles
        if 'original_scores' in self.results:
            complementary_fig_path = os.path.join(output_dir, 'complementary_roles.png')
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Pairwise optimization by constant
            constants = list(self.constants.keys())
            pairwise_scores = []
            
            for constant in constants:
                score = self.measure_pairwise_optimization(self.original_data, self.constants[constant])
                pairwise_scores.append(score['avg_pairwise_score'])
            
            # Highlight phi and sqrt2
            colors = ['#3498db'] * len(constants)
            phi_idx = constants.index('phi')
            sqrt2_idx = constants.index('sqrt2')
            colors[phi_idx] = '#e74c3c'  # Highlight phi in red
            
            ax1.bar(constants, pairwise_scores, color=colors)
            ax1.set_title('Pairwise Optimization by Constant')
            ax1.set_ylabel('Pairwise Score')
            ax1.grid(True, axis='y', alpha=0.3)
            
            # Plot 2: Cascade optimization by constant
            cascade_scores = []
            
            for constant in constants:
                score = self.measure_cascade_optimization(self.original_data, self.constants[constant])
                cascade_scores.append(score['avg_cascade_score'])
            
            # Highlight phi and sqrt2
            colors = ['#3498db'] * len(constants)
            colors[sqrt2_idx] = '#e74c3c'  # Highlight sqrt2 in red
            
            ax2.bar(constants, cascade_scores, color=colors)
            ax2.set_title('Cascade Optimization by Constant')
            ax2.set_ylabel('Cascade Score')
            ax2.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(complementary_fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files.append(complementary_fig_path)
        
        return visualization_files
    
    def generate_report(self, output_dir):
        """
        Generate a comprehensive report of the test results.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            str: Path to the generated report file
        """
        ensure_dir_exists(output_dir)
        
        # Create report file path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(output_dir, 'complementary_organization_report_{}.txt'.format(timestamp))
        
        with open(report_path, 'w') as f:
            # Write header
            f.write('='*80 + '\n')
            f.write('COMPLEMENTARY ORGANIZATION TEST REPORT\n')
            f.write('='*80 + '\n\n')
            
            f.write('Generated on: {}\n\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            # Write test configuration
            f.write('-'*80 + '\n')
            f.write('TEST CONFIGURATION\n')
            f.write('-'*80 + '\n\n')
            
            for key, value in self.config.items():
                f.write('{}: {}\n'.format(key, value))
            
            # Write test results
            f.write('\n' + '-'*80 + '\n')
            f.write('TEST RESULTS\n')
            f.write('-'*80 + '\n\n')
            
            if 'total_simulations' in self.results:
                f.write('Total simulations: {}\n'.format(self.results['total_simulations']))
            
            if 'early_stopping' in self.results:
                f.write('Early stopping: {}\n'.format(self.results['early_stopping']))
            
            # Write p-values
            if 'p_values' in self.results:
                f.write('\n' + '-'*40 + '\n')
                f.write('P-VALUES\n')
                f.write('-'*40 + '\n\n')
                
                f.write('Phi pairwise optimization: {:.6f}\n'.format(
                    self.results['p_values']['phi_pairwise']))
                f.write('Sqrt2 cascade optimization: {:.6f}\n'.format(
                    self.results['p_values']['sqrt2_cascade']))
                f.write('Combined optimization: {:.6f}\n'.format(
                    self.results['p_values']['combined']))
                
                # Interpret p-values
                f.write('\nInterpretation:\n')
                
                phi_p = self.results['p_values']['phi_pairwise']
                sqrt2_p = self.results['p_values']['sqrt2_cascade']
                combined_p = self.results['p_values']['combined']
                
                if phi_p < 0.05 and sqrt2_p < 0.05:
                    f.write('The test results support the hypothesis that phi and sqrt2 have complementary '
                           'organizational roles in the cosmic microwave background. Phi optimizes direct '
                           'relationships between pairs of scales (p={:.6f}), while sqrt2 optimizes cascade '
                           'pathways across multiple scales (p={:.6f}).\n\n'.format(phi_p, sqrt2_p))
                elif phi_p < 0.05:
                    f.write('The test results partially support the complementary organization hypothesis. '
                           'Phi shows significant optimization of direct relationships between pairs of scales '
                           '(p={:.6f}), but sqrt2 does not show significant optimization of cascade pathways '
                           '(p={:.6f}).\n\n'.format(phi_p, sqrt2_p))
                elif sqrt2_p < 0.05:
                    f.write('The test results partially support the complementary organization hypothesis. '
                           'Sqrt2 shows significant optimization of cascade pathways across multiple scales '
                           '(p={:.6f}), but phi does not show significant optimization of direct relationships '
                           'between pairs of scales (p={:.6f}).\n\n'.format(sqrt2_p, phi_p))
                else:
                    f.write('The test results do not support the hypothesis that phi and sqrt2 have '
                           'complementary organizational roles in the cosmic microwave background. '
                           'Neither phi (p={:.6f}) nor sqrt2 (p={:.6f}) shows significant optimization.\n\n'.format(
                               phi_p, sqrt2_p))
                
                if combined_p < 0.05:
                    f.write('The combined optimization of phi and sqrt2 working together is statistically '
                           'significant (p={:.6f}), suggesting that these constants may interact in a '
                           'complementary manner to optimize information flow in the cosmic microwave '
                           'background.\n'.format(combined_p))
                else:
                    f.write('The combined optimization of phi and sqrt2 working together is not statistically '
                           'significant (p={:.6f}).\n'.format(combined_p))
            
            # Write original scores
            if 'original_scores' in self.results:
                f.write('\n' + '-'*40 + '\n')
                f.write('ORIGINAL SCORES\n')
                f.write('-'*40 + '\n\n')
                
                # Phi pairwise scores
                phi_scores = self.results['original_scores']['phi_pairwise']
                f.write('Phi pairwise optimization:\n')
                f.write('  Average score: {:.6f}\n'.format(phi_scores['avg_pairwise_score']))
                f.write('  Maximum score: {:.6f}\n'.format(phi_scores['max_pairwise_score']))
                f.write('  Number of pairs: {}\n'.format(phi_scores['num_pairs']))
                
                # Sqrt2 cascade scores
                sqrt2_scores = self.results['original_scores']['sqrt2_cascade']
                f.write('\nSqrt2 cascade optimization:\n')
                f.write('  Average score: {:.6f}\n'.format(sqrt2_scores['avg_cascade_score']))
                f.write('  Maximum score: {:.6f}\n'.format(sqrt2_scores['max_cascade_score']))
                f.write('  Number of cascades: {}\n'.format(sqrt2_scores['num_cascades']))
                
                # Combined scores
                combined_scores = self.results['original_scores']['combined']
                f.write('\nCombined optimization:\n')
                f.write('  Combined score: {:.6f}\n'.format(combined_scores['combined_score']))
                f.write('  Number of interactions: {}\n'.format(combined_scores['num_interactions']))
            
            # Write conclusion
            f.write('\n' + '-'*80 + '\n')
            f.write('CONCLUSION\n')
            f.write('-'*80 + '\n\n')
            
            if 'p_values' in self.results:
                phi_p = self.results['p_values']['phi_pairwise']
                sqrt2_p = self.results['p_values']['sqrt2_cascade']
                combined_p = self.results['p_values']['combined']
                
                if sqrt2_p < 0.05 and combined_p < 0.05:
                    f.write('The test results support the hypothesis that sqrt2 and combined optimization have complementary '
                           'organizational roles in the cosmic microwave background. sqrt2 optimizes cascade '
                           'pathways across multiple scales (p={:.6f}), while combined optimization shows significant interaction '
                           '(p={:.6f}).\n\n'.format(sqrt2_p, combined_p))
                elif sqrt2_p < 0.05:
                    f.write('The test results partially support the complementary organization hypothesis. '
                           'sqrt2 shows significant cascade optimization (p={:.6f}), but combined optimization does not show significant interaction '
                           '(p={:.6f}).\n\n'.format(sqrt2_p, combined_p))
                elif combined_p < 0.05:
                    f.write('The test results partially support the complementary organization hypothesis. '
                           'Combined optimization shows significant interaction (p={:.6f}), but sqrt2 does not show significant cascade optimization '
                           '(p={:.6f}).\n\n'.format(combined_p, sqrt2_p))
                else:
                    f.write('The test results do not support the hypothesis that sqrt2 and combined optimization have '
                           'complementary organizational roles in the cosmic microwave background. '
                           'Neither sqrt2 (p={:.6f}) nor combined optimization (p={:.6f}) shows significant optimization.\n\n'.format(
                               sqrt2_p, combined_p))
        
        print("Report generated: {}".format(report_path))
        return report_path

def simulation_worker(params):
    """
    Worker function for parallel simulations.
    
    Args:
        params: Tuple of parameters
            (data, constants, original_phi_score, original_sqrt2_score, original_combined_score)
        
    Returns:
        tuple: (phi_exceeds, sqrt2_exceeds, combined_exceeds)
    """
    data, constants, original_scores = params
    
    # Create phase-randomized surrogate
    # FFT the data
    fft_data = np.fft.rfft(data)
    
    # Randomize the phases
    magnitudes = np.abs(fft_data)
    phases = np.angle(fft_data)
    random_phases = np.random.uniform(0, 2 * np.pi, len(phases))
    fft_random = magnitudes * np.exp(1j * random_phases)
    
    # Inverse FFT to get surrogate data
    surrogate = np.fft.irfft(fft_random, len(data))
    
    # Measure optimization on surrogate
    phi = constants['phi']
    sqrt2 = constants['sqrt2']
    
    # Measure pairwise optimization for phi
    phi_pairwise_scores = []
    base_scales = [10, 20, 50, 100, 200]
    for base_scale in base_scales:
        phi_scale = int(base_scale * phi)
        if phi_scale >= len(surrogate):
            continue
        
        # Calculate correlation between base scale and phi-related scale
        if base_scale < len(surrogate) and phi_scale < len(surrogate):
            correlation = np.corrcoef(surrogate[base_scale], surrogate[phi_scale])[0, 1]
            phi_pairwise_scores.append(abs(correlation))
    
    surrogate_phi_score = np.mean(phi_pairwise_scores) if phi_pairwise_scores else 0
    
    # Measure cascade optimization for sqrt2
    sqrt2_cascade_scores = []
    for base_scale in base_scales:
        # Generate cascade
        cascade = []
        current_scale = base_scale
        for _ in range(5):  # Depth of cascade
            current_scale = int(current_scale * sqrt2)
            if current_scale >= len(surrogate):
                break
            cascade.append(current_scale)
        
        # Skip if cascade is too short
        if len(cascade) < 3:
            continue
        
        # Extract cascade data
        cascade_data = [surrogate[scale] for scale in cascade]
        
        # Calculate cascade coherence
        coherence_score = 0
        for i in range(len(cascade_data) - 1):
            correlation = np.corrcoef(cascade_data[i], cascade_data[i+1])[0, 1]
            coherence_score += abs(correlation)
        
        avg_coherence = coherence_score / (len(cascade_data) - 1)
        sqrt2_cascade_scores.append(avg_coherence)
    
    surrogate_sqrt2_score = np.mean(sqrt2_cascade_scores) if sqrt2_cascade_scores else 0
    
    # Calculate combined effect
    combined_score = 0
    count = 0
    
    for base_scale in base_scales:
        # Get phi-related scale
        phi_scale = int(base_scale * phi)
        if phi_scale >= len(surrogate):
            continue
            
        # Generate sqrt2 cascade starting from the phi-related scale
        cascade = []
        current_scale = phi_scale
        for _ in range(5):  # Depth of cascade
            current_scale = int(current_scale * sqrt2)
            if current_scale >= len(surrogate):
                break
            cascade.append(current_scale)
        
        # Skip if cascade is too short
        if len(cascade) < 2:
            continue
        
        # Measure interaction between base scale, phi scale, and sqrt2 cascade
        interaction_score = 0
        
        # Correlation between base scale and phi scale
        base_phi_corr = np.corrcoef(surrogate[base_scale], surrogate[phi_scale])[0, 1]
        
        # Average correlation between phi scale and cascade scales
        cascade_corr = 0
        for scale in cascade:
            if scale < len(surrogate):
                corr = np.corrcoef(surrogate[phi_scale], surrogate[scale])[0, 1]
                cascade_corr += abs(corr)
        
        avg_cascade_corr = cascade_corr / len(cascade) if len(cascade) > 0 else 0
        
        # Combined interaction score
        interaction_score = abs(base_phi_corr) * avg_cascade_corr
        
        combined_score += interaction_score
        count += 1
    
    surrogate_combined_score = combined_score / count if count > 0 else 0
    
    # Check if surrogate scores exceed original scores
    phi_exceeds = surrogate_phi_score >= original_scores['phi_pairwise']
    sqrt2_exceeds = surrogate_sqrt2_score >= original_scores['sqrt2_cascade']
    combined_exceeds = surrogate_combined_score >= original_scores['combined']
    
    return phi_exceeds, sqrt2_exceeds, combined_exceeds

def main():
    """Run the Complementary Organization Test on WMAP and Planck data."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the Complementary Organization Test')
    parser.add_argument('--num-simulations', type=int, default=10000, help='Number of Monte Carlo simulations to run')
    parser.add_argument('--output-dir', type=str, default="../results/complementary_organization", help='Output directory for results')
    parser.add_argument('--wmap-data', type=str, default="../data/wmap_tt_spectrum_9yr_v5.txt", help='Path to WMAP data file')
    parser.add_argument('--planck-data', type=str, default="../data/planck_tt_spectrum_2018.txt", help='Path to Planck data file')
    args = parser.parse_args()
    
    # Create output directories
    wmap_output_dir = os.path.join(args.output_dir, 'wmap')
    planck_output_dir = os.path.join(args.output_dir, 'planck')
    ensure_dir_exists(wmap_output_dir)
    ensure_dir_exists(planck_output_dir)
    
    # Print header
    print("="*80)
    print("Running Complementary Organization Test on WMAP data")
    print("="*80)
    
    # Configure the test
    wmap_config = {
        'num_simulations': args.num_simulations,
        'significance_level': 0.05,
        'cascade_depth': 5,
        'batch_size': 1000,
        'parallel_processing': True,
        'timeout': 3600,
        'early_stopping': False  # Disable early stopping to run all simulations
    }
    
    # Initialize the test
    wmap_test = ComplementaryOrganizationTest(wmap_config)
    
    # Load WMAP data
    wmap_ell, wmap_power, wmap_error = load_wmap_power_spectrum(args.wmap_data)
    
    # Run the test on WMAP data
    if wmap_ell is not None and wmap_power is not None:
        wmap_results = wmap_test.run_test(wmap_power)
        wmap_test.visualize_results(wmap_output_dir)
        wmap_report_path = wmap_test.generate_report(wmap_output_dir)
    else:
        print("Error: Could not load WMAP data.")
        wmap_results = None
    
    # Print header
    print("="*80)
    print("Running Complementary Organization Test on Planck data")
    print("="*80)
    
    # Configure the test for Planck
    planck_config = {
        'num_simulations': args.num_simulations,
        'significance_level': 0.05,
        'cascade_depth': 5,
        'batch_size': 1000,
        'parallel_processing': True,
        'timeout': 3600,
        'early_stopping': False  # Disable early stopping to run all simulations
    }
    
    # Initialize the test for Planck
    planck_test = ComplementaryOrganizationTest(planck_config)
    
    # Load Planck data
    planck_ell, planck_power, planck_error = load_planck_power_spectrum(args.planck_data)
    
    # Run the test on Planck data
    if planck_ell is not None and planck_power is not None:
        planck_results = planck_test.run_test(planck_power)
        planck_test.visualize_results(planck_output_dir)
        planck_report_path = planck_test.generate_report(planck_output_dir)
    else:
        print("Error: Could not load Planck data.")
        planck_results = None
    
    # Print summary
    print("="*80)
    print("Complementary Organization Test Summary")
    print("="*80)
    print()
    
    if wmap_results is not None:
        print("WMAP Results:")
        print("Phi pairwise p-value: {:.6f}".format(wmap_results['p_values']['phi_pairwise']))
        print("Sqrt2 cascade p-value: {:.6f}".format(wmap_results['p_values']['sqrt2_cascade']))
        print("Combined p-value: {:.6f}".format(wmap_results['p_values']['combined']))
        print()
    
    if planck_results is not None:
        print("Planck Results:")
        print("Phi pairwise p-value: {:.6f}".format(planck_results['p_values']['phi_pairwise']))
        print("Sqrt2 cascade p-value: {:.6f}".format(planck_results['p_values']['sqrt2_cascade']))
        print("Combined p-value: {:.6f}".format(planck_results['p_values']['combined']))
        print()
    
    if wmap_results is not None and planck_results is not None:
        print("Cross-Dataset Comparison:")
        
        # Compare sqrt2 cascade optimization
        if wmap_results['p_values']['sqrt2_cascade'] < 0.05 and planck_results['p_values']['sqrt2_cascade'] < 0.05:
            print("- Sqrt2 shows significant cascade optimization in both datasets")
        elif wmap_results['p_values']['sqrt2_cascade'] < 0.05:
            print("- Sqrt2 shows significant cascade optimization only in WMAP data")
        elif planck_results['p_values']['sqrt2_cascade'] < 0.05:
            print("- Sqrt2 shows significant cascade optimization only in Planck data")
        else:
            print("- Sqrt2 does not show significant cascade optimization in either dataset")
        
        # Compare phi pairwise optimization
        if wmap_results['p_values']['phi_pairwise'] < 0.05 and planck_results['p_values']['phi_pairwise'] < 0.05:
            print("- Phi shows significant pairwise optimization in both datasets")
        elif wmap_results['p_values']['phi_pairwise'] < 0.05:
            print("- Phi shows significant pairwise optimization only in WMAP data")
        elif planck_results['p_values']['phi_pairwise'] < 0.05:
            print("- Phi shows significant pairwise optimization only in Planck data")
        else:
            print("- Phi does not show significant pairwise optimization in either dataset")
        
        print()
    
    # Print output directories
    print("Output directories:")
    print("WMAP results: {}".format(wmap_output_dir))
    print("Planck results: {}".format(planck_output_dir))
    print()
    
    # Print report paths
    print("Reports:")
    if wmap_results is not None:
        print("WMAP report: {}".format(wmap_report_path))
    if planck_results is not None:
        print("Planck report: {}".format(planck_report_path))

if __name__ == '__main__':
    main()
