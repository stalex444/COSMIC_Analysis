#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test for correlation analysis between Transfer Entropy and Integrated Information.
This script analyzes how these two metrics respond to changes in golden ratio organization
in synthetic CMB-like data.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.signal import decimate
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import correlation
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')
import argparse

# For FITS file handling
try:
    from astropy.io import fits
except ImportError:
    print("Warning: astropy not installed. WMAP data analysis will not work.")
    print("Install with: pip install astropy")

# Set random seed for reproducibility
np.random.seed(42)

# Ensure output directory exists
if not os.path.exists('./correlation_results'):
    os.makedirs('./correlation_results')

# Function to process a single GR level (defined outside the class for multiprocessing)
def process_level(args):
    """
    Process a single GR level for the manipulation experiment.
    
    Parameters:
    -----------
    args : tuple
        (gr_level, n_surrogates, size, downsampling_factor) or
        (gr_level, n_surrogates, wmap_data)
        
    Returns:
    --------
    dict
        Results for this GR level
    """
    if len(args) == 4:
        gr_level, n_surrogates, size, downsampling_factor = args
        use_real_data = False
    elif len(args) == 3:
        gr_level, n_surrogates, wmap_data = args
        use_real_data = True
        # Set a default downsampling factor based on data size
        downsampling_factor = 1
        if wmap_data is not None and len(wmap_data) > 1000:
            downsampling_factor = len(wmap_data) // 1000
    else:
        raise ValueError("Invalid number of arguments")
    
    print("Processing GR level: %.2f" % gr_level)
    
    # Create an instance for this process
    analyzer = MetricCorrelationAnalysis()
    
    # Generate synthetic data with this level of GR organization
    if use_real_data:
        data = analyzer.add_gr_to_data(wmap_data, gr_level=gr_level)
    else:
        data = analyzer.generate_synthetic_cmb(size=size, gr_level=gr_level)
    
    # Calculate metrics for multiple surrogate datasets
    te_values = []
    ii_values = []
    
    for i in range(n_surrogates):
        # Create a surrogate by shuffling the data
        surrogate = np.random.permutation(data)
        
        # Calculate transfer entropy
        te = analyzer.calculate_transfer_entropy(surrogate, downsampling_factor=downsampling_factor)
        te_values.append(te)
        
        # Calculate integrated information
        ii = analyzer.calculate_integrated_information(surrogate)
        ii_values.append(ii)
    
    # Calculate statistics
    avg_te = np.nanmean(te_values)
    std_te = np.nanstd(te_values)
    avg_ii = np.nanmean(ii_values)
    std_ii = np.nanstd(ii_values)
    
    # Return results for this level
    return {
        'gr_level': gr_level,
        'avg_te': avg_te,
        'std_te': std_te,
        'avg_ii': avg_ii,
        'std_ii': std_ii,
        'te_values': te_values,
        'ii_values': ii_values
    }

class MetricCorrelationAnalysis:
    """
    Class for analyzing correlations between different metrics in CMB data analysis.
    Specifically focuses on Transfer Entropy and Integrated Information.
    """
    
    def __init__(self, output_dir=None, n_surrogates=10, n_gr_levels=5, parallel=False, num_processes=None, random_seed=42, use_real_data=False):
        """
        Initialize the MetricCorrelationAnalysis class.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save results. If None, use current directory.
        n_surrogates : int, optional
            Number of surrogate datasets to generate for each GR level.
        n_gr_levels : int, optional
            Number of golden ratio organization levels to test.
        parallel : bool, optional
            Whether to use parallel processing.
        num_processes : int, optional
            Number of processes to use for parallel processing.
            If None, use the number of CPU cores.
        random_seed : int, optional
            Random seed for reproducibility.
        use_real_data : bool, optional
            Whether to use real WMAP data instead of synthetic data.
        """
        self.output_dir = output_dir if output_dir else os.getcwd()
        self.n_surrogates = n_surrogates
        self.n_gr_levels = n_gr_levels
        self.parallel = parallel
        self.num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count()
        self.random_seed = random_seed
        self.use_real_data = use_real_data
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Set random seed
        np.random.seed(random_seed)
            
        print("Initialized MetricCorrelationAnalysis")
        print("Output directory: %s" % self.output_dir)
        print("Number of surrogates: %d" % n_surrogates)
        print("Number of GR levels: %d" % n_gr_levels)
        if use_real_data:
            print("Using real WMAP data for analysis")

    def generate_synthetic_cmb(self, size=128, gr_level=0.0):
        """
        Generate synthetic CMB-like data with controlled levels of golden ratio organization.
        
        Parameters:
        -----------
        size : int
            Size of the synthetic data
        gr_level : float
            Level of golden ratio organization (0 to 1)
            
        Returns:
        --------
        data : numpy.ndarray
            Synthetic CMB-like data
        """
        # Base random noise
        data = np.random.normal(0, 1, (size, size))
        
        # Add golden ratio organization if level > 0
        if gr_level > 0:
            # Golden ratio
            phi = (1 + np.sqrt(5)) / 2
            
            # Create golden ratio patterns at different scales
            gr_pattern = np.zeros((size, size))
            
            # Add patterns at phi-related scales (limit to fewer scales for efficiency)
            for scale in [int(size / (phi ** i)) for i in range(1, 4) if size / (phi ** i) > 5]:
                # Create a pattern at this scale
                pattern = np.zeros((size, size))
                
                # Add some structure at this scale
                for i in range(0, size, scale):
                    for j in range(0, size, scale):
                        # Add a coherent pattern with phi-based relationships
                        block_size = min(scale, size - max(i, j))
                        # Create a pattern with golden ratio proportions
                        block = np.random.normal(0, 1, (block_size, block_size))
                        
                        # Apply a phi-based filter (simplified for efficiency)
                        x, y = np.meshgrid(np.arange(block_size), np.arange(block_size))
                        phi_mask = np.sin(2 * np.pi * x / phi) * np.sin(2 * np.pi * y / phi)
                        block = block * (0.5 + 0.5 * phi_mask)
                        
                        pattern[i:i+block_size, j:j+block_size] = block
                
                gr_pattern += pattern / np.std(pattern)
            
            # Normalize the GR pattern
            if np.std(gr_pattern) > 0:
                gr_pattern = gr_pattern / np.std(gr_pattern)
            
            # Mix the random noise with the GR pattern according to the organization level
            data = (1 - gr_level) * data + gr_level * gr_pattern
        
        # Ensure the data has zero mean and unit variance
        data = (data - np.mean(data)) / np.std(data)
        
        return data

    def load_wmap_data(self, data_type='power_spectrum'):
        """
        Load WMAP data for analysis.
        
        Parameters:
        -----------
        data_type : str
            Type of WMAP data to load ('power_spectrum' or 'binned_power_spectrum')
            
        Returns:
        --------
        data : numpy.ndarray
            WMAP data
        """
        # Define paths to WMAP data files
        wmap_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wmap_data', 'raw_data')
        
        if data_type == 'power_spectrum':
            # Load WMAP power spectrum data
            file_path = os.path.join(wmap_data_dir, 'wmap_tt_spectrum_9yr_v5.txt')
            
            try:
                # Check if file exists and is not an HTML error page
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('<!DOCTYPE HTML') or first_line.startswith('<html'):
                        print("Error: WMAP power spectrum file appears to be an HTML error page.")
                        print("Please run download_wmap_data_lambda.py to download the data properly.")
                        return None
                
                # Load data - expected format is multipole (l), power (Cl), error
                # Skip comment lines (lines starting with #)
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if not line.strip().startswith('#'):
                            values = [float(x) for x in line.strip().split()]
                            data.append(values)
                
                data = np.array(data)
                
                # Extract multipole and power values
                multipoles = data[:, 0]  # multipole moment l
                power = data[:, 1]       # power spectrum value
                
                print("Loaded WMAP power spectrum with %d data points" % len(multipoles))
                return power  # Return just the power spectrum values
                
            except Exception as e:
                print("Error loading WMAP power spectrum: %s" % str(e))
                return None
                
        elif data_type == 'binned_power_spectrum':
            # Load WMAP binned power spectrum data
            file_path = os.path.join(wmap_data_dir, 'wmap_binned_tt_spectrum_9yr_v5.txt')
            
            try:
                # Check if file exists and is not an HTML error page
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('<!DOCTYPE HTML') or first_line.startswith('<html'):
                        print("Error: WMAP binned power spectrum file appears to be an HTML error page.")
                        print("Please run download_wmap_data_lambda.py to download the data properly.")
                        return None
                
                # Load data - expected format is bin, power (Cl), lower error, upper error
                # Skip comment lines (lines starting with #)
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if not line.strip().startswith('#'):
                            values = [float(x) for x in line.strip().split()]
                            data.append(values)
                
                data = np.array(data)
                
                # Extract bin and power values
                bins = data[:, 0]  # bin number
                power = data[:, 1]  # power spectrum value
                
                print("Loaded WMAP binned power spectrum with %d data points" % len(bins))
                return power  # Return just the power spectrum values
                
            except Exception as e:
                print("Error loading WMAP binned power spectrum: %s" % str(e))
                return None
        
        else:
            print("Unknown data type: %s" % data_type)
            return None

    def add_gr_to_data(self, data, gr_level):
        """
        Manipulate real WMAP data to introduce golden ratio organization.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Original WMAP data
        gr_level : float
            Level of golden ratio organization to introduce (0 to 1)
            
        Returns:
        --------
        manipulated_data : numpy.ndarray
            Manipulated data with golden ratio organization
        """
        # Make a copy of the data
        manipulated_data = data.copy()
        
        if gr_level == 0:
            # No manipulation
            return manipulated_data
            
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Number of data points
        n = len(data)
        
        # Calculate Fibonacci-like sequence based on data length
        fib_sequence = [1, 1]
        while fib_sequence[-1] < n:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
            
        # Remove the last element if it exceeds the data length
        if fib_sequence[-1] > n:
            fib_sequence.pop()
            
        # Calculate the indices where we'll introduce golden ratio organization
        indices = []
        for i in range(len(fib_sequence) - 1):
            indices.append(fib_sequence[i] - 1)  # Convert to 0-based indexing
            
        # Limit indices to valid range
        indices = [idx for idx in indices if idx < n]
        
        # Introduce golden ratio organization by amplifying values at Fibonacci indices
        for idx in indices:
            # Amplify the value based on gr_level
            manipulated_data[idx] = data[idx] * (1 + gr_level * (phi - 1))
            
        # Apply smoothing to blend the manipulated values
        if gr_level > 0:
            # Calculate sigma based on gr_level (higher gr_level = less smoothing)
            sigma = 1.0 * (1.0 - gr_level * 0.8)
            manipulated_data = gaussian_filter(manipulated_data, sigma=sigma)
            
        return manipulated_data

    def check_wmap_data_exists(self, filename):
        """
        Check if a WMAP data file exists and is valid.
        
        Parameters:
        -----------
        filename : str
            Name of the WMAP data file to check
            
        Returns:
        --------
        bool
            True if the file exists and is valid, False otherwise
        """
        # Define path to WMAP data file
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print("Error: WMAP data file '%s' does not exist." % filename)
            return False
            
        # Check if file is not empty
        if os.path.getsize(file_path) == 0:
            print("Error: WMAP data file '%s' is empty." % filename)
            return False
            
        # Check file content based on file type
        if filename.endswith('.txt'):
            try:
                # Check if file is not an HTML error page
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('<!DOCTYPE HTML') or first_line.startswith('<html'):
                        print("Error: WMAP data file '%s' appears to be an HTML error page." % filename)
                        return False
                    
                # For text files, check if they contain numerical data
                with open(file_path, 'r') as f:
                    for line in f:
                        if not line.strip().startswith('#'):
                            try:
                                # Try to parse a line as numbers
                                values = [float(x) for x in line.strip().split()]
                                if len(values) >= 2:  # At least multipole and power
                                    return True  # Found valid data
                            except ValueError:
                                continue
                            
                print("Error: WMAP data file '%s' does not contain valid numerical data." % filename)
                return False
                
            except Exception as e:
                print("Error checking WMAP data file '%s': %s" % (filename, str(e)))
                return False
                
        elif filename.endswith('.fits'):
            try:
                # Check if file is a valid FITS file
                from astropy.io import fits
                with fits.open(file_path) as hdul:
                    # Check if file has at least one HDU
                    if len(hdul) == 0:
                        print("Error: WMAP data file '%s' is not a valid FITS file (no HDUs)." % filename)
                        return False
                        
                    # Check if primary HDU has valid header
                    primary_hdu = hdul[0]
                    if not primary_hdu.header:
                        print("Error: WMAP data file '%s' has no valid header in primary HDU." % filename)
                        return False
                        
                return True
                
            except Exception as e:
                print("Error checking WMAP data file '%s': %s" % (filename, str(e)))
                return False
                
        elif filename.endswith('.tar.gz'):
            try:
                # Check if file is a valid tar.gz file
                import tarfile
                if tarfile.is_tarfile(file_path):
                    return True
                else:
                    print("Error: WMAP data file '%s' is not a valid tar.gz file." % filename)
                    return False
                    
            except Exception as e:
                print("Error checking WMAP data file '%s': %s" % (filename, str(e)))
                return False
                
        else:
            print("Error: Unknown file type for WMAP data file '%s'." % filename)
            return False
            
    def validate_all_wmap_data(self):
        """
        Validate all required WMAP data files.
        
        Returns:
        --------
        dict
            Validation results for each file
        """
        required_files = [
            'wmap_tt_spectrum_9yr_v5.txt',
            'wmap_binned_tt_spectrum_9yr_v5.txt',
            'wmap_ilc_9yr_v5.fits'
        ]
        
        results = {}
        all_valid = True
        
        for filename in required_files:
            valid = self.check_wmap_data_exists(filename)
            results[filename] = valid
            if not valid:
                all_valid = False
                
        if all_valid:
            print("All required WMAP data files are valid.")
        else:
            print("Some required WMAP data files are missing or invalid.")
            print("Please run validate_wmap_data.py --fix-issues to fix the issues.")
            
        return results

    def calculate_transfer_entropy(self, data, downsampling_factor=1):
        """
        Calculate transfer entropy between scales related by the golden ratio.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data
        downsampling_factor : int
            Factor by which to downsample the data
            
        Returns:
        --------
        te : float
            Transfer entropy
        """
        # Ensure data is 1D
        data = np.asarray(data).flatten()
        
        # Check if data is too small
        if len(data) < 10:
            return 0.0
        
        # Downsample data to create coarse-grained version
        if downsampling_factor > 1:
            # More efficient downsampling using array slicing
            coarse_data = data[::downsampling_factor]
        else:
            coarse_data = data
        
        # Ensure we have enough data points after downsampling
        if len(coarse_data) < 5:
            return 0.0
        
        # Create time-shifted versions for transfer entropy calculation
        x_t = data[:-1]  # X_t (current state)
        y_t = coarse_data[:-1]  # Y_t (coarse-grained current state)
        y_t_plus_1 = coarse_data[1:]  # Y_{t+1} (coarse-grained next state)
        
        # Ensure all arrays have the same length
        min_length = min(len(x_t), len(y_t), len(y_t_plus_1))
        x_t = x_t[:min_length]
        y_t = y_t[:min_length]
        y_t_plus_1 = y_t_plus_1[:min_length]
        
        try:
            # Calculate mutual information terms using KDE for more robust estimation
            # I(Y_{t+1}; X_t | Y_t) = I(Y_{t+1}; X_t, Y_t) - I(Y_{t+1}; Y_t)
            
            # Calculate I(Y_{t+1}; Y_t)
            mi_y_next_y = mutual_info_regression(
                y_t.reshape(-1, 1), y_t_plus_1, discrete_features=False, random_state=self.random_seed
            )[0]
            
            # Calculate I(Y_{t+1}; X_t, Y_t)
            # Stack X_t and Y_t as features
            xy_features = np.column_stack((x_t, y_t))
            mi_y_next_xy = mutual_info_regression(
                xy_features, y_t_plus_1, discrete_features=False, random_state=self.random_seed
            ).sum()
            
            # Transfer entropy is the conditional mutual information
            te = max(0, mi_y_next_xy - mi_y_next_y)
            
            return te
        except Exception as e:
            print("Error calculating transfer entropy: %s" % str(e))
            return 0.0

    def calculate_integrated_information(self, data, tau=1, k=2):
        """
        Calculate integrated information (Phi) for the data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data
        tau : int
            Time delay
        k : int
            Number of parts to divide the system into
            
        Returns:
        --------
        phi : float
            Integrated information value
        """
        # For 2D data, we'll flatten it for this calculation
        if len(data.shape) > 1:
            data = data.flatten()
            
        n = len(data)
        if n < 10:  # Need minimum data for meaningful calculation
            return 0.0
            
        # Create time-delayed version of the data
        X = data[:-tau]
        Y = data[tau:]
        
        # Ensure we have enough data points
        n = len(X)
        if n < 10:
            return 0.0
            
        # Normalize the data
        X = (X - np.mean(X)) / (np.std(X) + 1e-10)
        Y = (Y - np.mean(Y)) / (np.std(Y) + 1e-10)
        
        try:
            # Calculate the covariance matrices
            cov_X = np.cov(X, rowvar=False) if X.ndim > 1 else np.var(X)
            cov_Y = np.cov(Y, rowvar=False) if Y.ndim > 1 else np.var(Y)
            
            # For 1D data, these are just scalars
            if np.isscalar(cov_X):
                cov_X = np.array([[cov_X]])
            if np.isscalar(cov_Y):
                cov_Y = np.array([[cov_Y]])
                
            # Calculate the joint entropy of the system
            # For Gaussian variables, entropy is related to determinant of covariance
            if cov_X.size > 1:
                try:
                    h_X = 0.5 * np.log(np.linalg.det(cov_X) + 1e-10)
                except:
                    h_X = 0.5 * np.log(np.prod(np.diag(cov_X)) + 1e-10)
            else:
                h_X = 0.5 * np.log(cov_X[0, 0] + 1e-10)
                
            if cov_Y.size > 1:
                try:
                    h_Y = 0.5 * np.log(np.linalg.det(cov_Y) + 1e-10)
                except:
                    h_Y = 0.5 * np.log(np.prod(np.diag(cov_Y)) + 1e-10)
            else:
                h_Y = 0.5 * np.log(cov_Y[0, 0] + 1e-10)
            
            # Simplified approach: use mutual information as a proxy for integrated information
            # In a fully integrated system, the mutual information between past and future states
            # should be high
            
            # For 1D data, use a simple correlation-based approach
            corr = np.corrcoef(X, Y)[0, 1] if X.size > 1 and Y.size > 1 else 0
            mi = -0.5 * np.log(1 - corr**2 + 1e-10) if abs(corr) < 1 else 0
            
            # Scale the result to be in a reasonable range
            phi = np.tanh(mi)  # Constrain to [0, 1]
            
            return max(0, phi)  # Ensure non-negative
            
        except Exception as e:
            # Handle numerical issues
            return 0.0

    def run_manipulation_experiment(self):
        """
        Run the experiment manipulating the golden ratio organization level
        and measuring the response of different metrics.
        
        Returns:
        --------
        dict
            Results of the experiment
        """
        print("Running manipulation experiment...")
        
        # Define GR levels to test
        gr_levels = np.linspace(0, 1, self.n_gr_levels)
        
        # Initialize results list
        results = []
        
        # Load real WMAP data if using real data
        wmap_data = None
        if self.use_real_data:
            wmap_data = self.load_wmap_data(data_type='power_spectrum')
            if wmap_data is None:
                print("Failed to load WMAP data. Falling back to synthetic data.")
                self.use_real_data = False
        
        # Use parallel processing if enabled
        if self.parallel:
            try:
                print("Using %d cores for parallel processing" % self.num_processes)
                
                # Create argument tuples for each GR level
                args_list = []
                for level in gr_levels:
                    if self.use_real_data:
                        args_list.append((level, self.n_surrogates, wmap_data))
                    else:
                        args_list.append((level, self.n_surrogates, 128, 1))
                
                # Create a pool of workers
                pool = Pool(processes=self.num_processes)
                
                # Map the process_level function to the arguments
                results = pool.map(process_level, args_list)
                
                # Close the pool
                pool.close()
                pool.join()
                
            except Exception as e:
                print("Error in parallel processing: %s" % str(e))
                print("Falling back to sequential processing")
                results = []
        
        # Use sequential processing if parallel is disabled or failed
        if not self.parallel or not results:
            # Process each GR level
            for level in tqdm(gr_levels, desc="Processing GR levels"):
                if self.use_real_data:
                    results.append(process_level((level, self.n_surrogates, wmap_data)))
                else:
                    results.append(process_level((level, self.n_surrogates, 128, 1)))
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Add normalized columns
        # For TE
        if df_results['avg_te'].max() > 0:
            df_results['norm_te'] = (df_results['avg_te'] - df_results['avg_te'].min()) / (df_results['avg_te'].max() - df_results['avg_te'].min())
        else:
            df_results['norm_te'] = 0
            
        # For II
        if df_results['avg_ii'].max() > df_results['avg_ii'].min():
            df_results['norm_ii'] = (df_results['avg_ii'] - df_results['avg_ii'].min()) / (df_results['avg_ii'].max() - df_results['avg_ii'].min())
        else:
            df_results['norm_ii'] = 0
            
        # Calculate growth rates
        df_results['te_growth'] = df_results['avg_te'].pct_change()
        df_results['ii_growth'] = df_results['avg_ii'].pct_change()
        
        # Replace NaN with 0 for the first row
        df_results['te_growth'].fillna(0, inplace=True)
        df_results['ii_growth'].fillna(0, inplace=True)
        
        # Save results
        results_file = os.path.join(self.output_dir, 'manipulation_experiment_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(df_results, f)
            
        # Also save as CSV for easier inspection
        csv_file = os.path.join(self.output_dir, 'manipulation_experiment_results.csv')
        df_results.to_csv(csv_file, index=False)
        
        print("Experiment completed in %.2f seconds" % (time.time() - time.time()))
        
        return df_results

    def analyze_results(self, results=None):
        """
        Analyze the results of the manipulation experiment.
        
        Parameters:
        -----------
        results : pandas.DataFrame or dict, optional
            Results of the experiment. If None, load from file.
            
        Returns:
        --------
        dict
            Analysis results
        """
        # Load results if not provided
        if results is None:
            try:
                with open(os.path.join(self.output_dir, 'manipulation_experiment_results.pkl'), 'rb') as f:
                    loaded_results = pickle.load(f)
                    if isinstance(loaded_results, dict) and 'df_results' in loaded_results:
                        df_results = loaded_results['df_results']
                    else:
                        df_results = loaded_results
            except:
                raise ValueError("No results provided and could not load from file")
        else:
            # If results is already a DataFrame, use it directly
            if isinstance(results, pd.DataFrame):
                df_results = results
            # If results is a dict with 'df_results', use that
            elif isinstance(results, dict) and 'df_results' in results:
                df_results = results['df_results']
            else:
                df_results = results
        
        # Calculate correlations between metrics
        pearson_corr, pearson_p = stats.pearsonr(df_results['avg_te'], df_results['avg_ii'])
        try:
            spearman_corr, spearman_p = stats.spearmanr(df_results['avg_te'], df_results['avg_ii'])
        except:
            spearman_corr, spearman_p = np.nan, np.nan
            
        # Calculate mutual information between metrics
        try:
            # Discretize the data for MI calculation
            n_bins = min(10, len(df_results) // 2)
            if n_bins < 2:
                n_bins = 2
                
            te_binned = np.digitize(df_results['avg_te'], 
                                    np.linspace(df_results['avg_te'].min(), 
                                                df_results['avg_te'].max(), 
                                                n_bins))
            ii_binned = np.digitize(df_results['avg_ii'], 
                                    np.linspace(df_results['avg_ii'].min(), 
                                                df_results['avg_ii'].max(), 
                                                n_bins))
            
            mutual_info = mutual_info_score(te_binned, ii_binned)
        except:
            mutual_info = 0.0
        
        # Calculate correlation between growth rates
        try:
            # Calculate growth rates (derivative of metrics with respect to GR level)
            te_growth = np.diff(df_results['avg_te']) / np.diff(df_results['gr_level'])
            ii_growth = np.diff(df_results['avg_ii']) / np.diff(df_results['gr_level'])
            
            # Calculate correlation between growth rates
            growth_corr, growth_p = stats.pearsonr(te_growth, ii_growth)
        except:
            growth_corr, growth_p = np.nan, np.nan
        
        # Determine relationship type
        # 1. Independent: Different response curves with low correlation
        # 2. Complementary: Different but correlated response curves
        # 3. Redundant: Similar response curves with high correlation
        
        # Criteria for classification
        if abs(pearson_corr) < 0.3 or np.isnan(pearson_corr):
            relationship = "Independent (different response curves with low correlation)"
        elif abs(pearson_corr) >= 0.7:
            relationship = "Redundant (similar response curves with high correlation)"
        else:
            relationship = "Complementary (different but correlated response curves)"
        
        # Check if there's a threshold relationship
        # (one metric responds earlier/later than the other)
        threshold_relationship = "Could not determine threshold relationship"
        try:
            # Normalize curves for comparison
            norm_te = (df_results['avg_te'] - df_results['avg_te'].min()) / (df_results['avg_te'].max() - df_results['avg_te'].min() + 1e-10)
            norm_ii = (df_results['avg_ii'] - df_results['avg_ii'].min()) / (df_results['avg_ii'].max() - df_results['avg_ii'].min() + 1e-10)
            
            # Calculate difference between normalized curves
            curve_diff = norm_te - norm_ii
            max_diff = np.max(np.abs(curve_diff))
            
            # Check if one metric consistently leads the other
            if max_diff > 0.3:  # Significant difference
                # Find where the maximum difference occurs
                max_diff_idx = np.argmax(np.abs(curve_diff))
                gr_level_at_max_diff = df_results['gr_level'].iloc[max_diff_idx]
                
                if curve_diff[max_diff_idx] > 0:
                    threshold_relationship = "Transfer Entropy responds earlier (at GR level %.2f)" % gr_level_at_max_diff
                else:
                    threshold_relationship = "Integrated Information responds earlier (at GR level %.2f)" % gr_level_at_max_diff
        except:
            pass
        
        # Compile analysis results
        analysis_results = {
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'mutual_info': mutual_info,
            'growth_corr': growth_corr,
            'growth_p': growth_p,
            'relationship': relationship,
            'threshold_relationship': threshold_relationship,
            'max_curve_diff': max_diff if 'max_diff' in locals() else np.nan
        }
        
        # Save analysis results
        with open(os.path.join(self.output_dir, 'analysis_results.pkl'), 'wb') as f:
            pickle.dump(analysis_results, f)
            
        # Also save as text for easy inspection
        with open(os.path.join(self.output_dir, 'analysis_results.txt'), 'w') as f:
            f.write("Analysis Results\n")
            f.write("===============\n\n")
            f.write("Pearson Correlation: %.4f (p = %.6f)\n" % (pearson_corr, pearson_p))
            f.write("Spearman Correlation: %.4f (p = %.6f)\n" % (spearman_corr, spearman_p))
            f.write("Mutual Information: %.4f\n" % mutual_info)
            f.write("Growth Rate Correlation: %.4f (p = %.6f)\n" % (growth_corr, growth_p))
            f.write("Relationship Classification: %s\n" % relationship)
            f.write("Threshold Relationship: %s\n" % threshold_relationship)
            if 'max_diff' in locals():
                f.write("Maximum Normalized Curve Difference: %.4f\n" % max_diff)
        
        return analysis_results

    def create_visualizations(self, results=None, analysis=None):
        """
        Create visualizations of the results.
        
        Parameters:
        -----------
        results : pandas.DataFrame or dict, optional
            Results of the experiment. If None, load from file.
        analysis : dict, optional
            Analysis results. If None, load from file.
        """
        # Load results if not provided
        if results is None:
            try:
                with open(os.path.join(self.output_dir, 'manipulation_experiment_results.pkl'), 'rb') as f:
                    loaded_results = pickle.load(f)
                    if isinstance(loaded_results, dict) and 'df_results' in loaded_results:
                        df_results = loaded_results['df_results']
                        raw_results = loaded_results.get('raw_results', [])
                    else:
                        df_results = loaded_results
                        raw_results = []
            except:
                raise ValueError("No results provided and could not load from file")
        else:
            # If results is already a DataFrame, use it directly
            if isinstance(results, pd.DataFrame):
                df_results = results
                raw_results = []
            # If results is a dict with 'df_results', use that
            elif isinstance(results, dict) and 'df_results' in results:
                df_results = results['df_results']
                raw_results = results.get('raw_results', [])
            else:
                df_results = results
                raw_results = []
        
        # Load analysis if not provided
        if analysis is None:
            try:
                with open(os.path.join(self.output_dir, 'analysis_results.pkl'), 'rb') as f:
                    analysis = pickle.load(f)
            except:
                raise ValueError("No analysis provided and could not load from file")
        
        # Calculate growth rates if not present
        if 'te_growth' not in df_results.columns:
            try:
                df_results['te_growth'] = df_results['avg_te'].pct_change().fillna(0)
                df_results['ii_growth'] = df_results['avg_ii'].pct_change().fillna(0)
            except:
                # If we can't calculate growth rates, add dummy columns
                df_results['te_growth'] = 0
                df_results['ii_growth'] = 0
        
        # 1. Response curves plot
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['gr_level'], df_results['avg_te'], 'o-', label='Transfer Entropy')
        plt.fill_between(df_results['gr_level'], 
                         df_results['avg_te'] - df_results['std_te'], 
                         df_results['avg_te'] + df_results['std_te'], 
                         alpha=0.3)
        
        plt.plot(df_results['gr_level'], df_results['avg_ii'], 's-', label='Integrated Information')
        plt.fill_between(df_results['gr_level'], 
                         df_results['avg_ii'] - df_results['std_ii'], 
                         df_results['avg_ii'] + df_results['std_ii'], 
                         alpha=0.3)
        
        plt.xlabel('Golden Ratio Organization Level')
        plt.ylabel('Metric Value')
        plt.title('Response to Golden Ratio Organization')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'response_analysis.png'), dpi=300, bbox_inches='tight')
        
        # 2. Growth rate comparison (if we have enough data points)
        plt.figure(figsize=(10, 6))
        if len(df_results) > 2:
            plt.plot(df_results['gr_level'][1:], df_results['te_growth'][1:], 'o-', label='TE Growth Rate')
            plt.plot(df_results['gr_level'][1:], df_results['ii_growth'][1:], 's-', label='II Growth Rate')
            plt.xlabel('Golden Ratio Organization Level')
            plt.ylabel('Growth Rate')
            plt.title('Growth Rate Comparison (Correlation: %.4f)' % analysis['growth_corr'])
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Not enough data points for growth rate analysis', 
                     ha='center', va='center', fontsize=12)
        plt.grid(True)
        
        # 3. Correlation plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df_results['avg_te'], df_results['avg_ii'])
        plt.xlabel('Transfer Entropy')
        plt.ylabel('Integrated Information')
        title = 'Correlation: %.4f (p = %.6f)\n' % (analysis['pearson_corr'], analysis['pearson_p'])
        title += 'Relationship: %s' % analysis['relationship']
        plt.title(title)
        plt.grid(True)
        
        # 4. Individual surrogate plots (if raw results available)
        plt.figure(figsize=(12, 8))
        
        # Only plot if we have raw results with individual surrogate data
        if raw_results and len(raw_results) > 0 and 'te_values' in raw_results[0]:
            n_levels = len(raw_results)
            n_surrogates = min(10, len(raw_results[0]['te_values']))  # Limit to 10 surrogates for clarity
            
            # Plot TE for individual surrogates
            plt.subplot(2, 1, 1)
            for i in range(n_surrogates):
                te_values = [r['te_values'][i] if i < len(r['te_values']) else np.nan for r in raw_results]
                gr_levels = [r['gr_level'] for r in raw_results]
                plt.plot(gr_levels, te_values, 'o-', alpha=0.3, linewidth=1)
            
            # Plot average
            plt.plot(df_results['gr_level'], df_results['avg_te'], 'o-', color='black', linewidth=2, label='Average')
            plt.xlabel('Golden Ratio Organization Level')
            plt.ylabel('Transfer Entropy')
            plt.title('Individual Surrogate Transfer Entropy Responses')
            plt.legend()
            plt.grid(True)
            
            # Plot II for individual surrogates
            plt.subplot(2, 1, 2)
            for i in range(n_surrogates):
                ii_values = [r['ii_values'][i] if i < len(r['ii_values']) else np.nan for r in raw_results]
                gr_levels = [r['gr_level'] for r in raw_results]
                plt.plot(gr_levels, ii_values, 's-', alpha=0.3, linewidth=1)
            
            # Plot average
            plt.plot(df_results['gr_level'], df_results['avg_ii'], 's-', color='black', linewidth=2, label='Average')
            plt.xlabel('Golden Ratio Organization Level')
            plt.ylabel('Integrated Information')
            plt.title('Individual Surrogate Integrated Information Responses')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No individual surrogate data available', 
                     ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'individual_surrogates.png'), dpi=300, bbox_inches='tight')
        
        # 5. Summary visualization with multiple plots
        plt.figure(figsize=(14, 10))
        
        # Add a title
        plt.suptitle('Relationship Between Transfer Entropy and Integrated Information\n%s' % analysis['relationship'], 
                    fontsize=18)
        
        # Main response curve plot
        plt.subplot(2, 2, 1)
        plt.plot(df_results['gr_level'], df_results['avg_te'], 'o-', label='Transfer Entropy')
        plt.fill_between(df_results['gr_level'], 
                         df_results['avg_te'] - df_results['std_te'], 
                         df_results['avg_te'] + df_results['std_te'], 
                         alpha=0.3)
        
        plt.plot(df_results['gr_level'], df_results['avg_ii'], 's-', label='Integrated Information')
        plt.fill_between(df_results['gr_level'], 
                         df_results['avg_ii'] - df_results['std_ii'], 
                         df_results['avg_ii'] + df_results['std_ii'], 
                         alpha=0.3)
        
        plt.xlabel('Golden Ratio Organization Level')
        plt.ylabel('Metric Value')
        plt.title('Response to Golden Ratio Organization')
        plt.legend()
        plt.grid(True)
        
        # Normalized response curve plot
        plt.subplot(2, 2, 2)
        
        # Calculate normalized curves if not already present
        if 'norm_te' not in df_results.columns or 'norm_ii' not in df_results.columns:
            try:
                norm_te = (df_results['avg_te'] - df_results['avg_te'].min()) / (df_results['avg_te'].max() - df_results['avg_te'].min())
                norm_ii = (df_results['avg_ii'] - df_results['avg_ii'].min()) / (df_results['avg_ii'].max() - df_results['avg_ii'].min())
            except:
                norm_te = df_results['avg_te'] * 0
                norm_ii = df_results['avg_ii'] * 0
        else:
            norm_te = df_results['norm_te']
            norm_ii = df_results['norm_ii']
            
        plt.plot(df_results['gr_level'], norm_te, 'o-', label='Normalized TE')
        plt.plot(df_results['gr_level'], norm_ii, 's-', label='Normalized II')
        plt.xlabel('Golden Ratio Organization Level')
        plt.ylabel('Normalized Value')
        plt.title('Normalized Response Curves')
        plt.legend()
        plt.grid(True)
        
        # Correlation plot
        plt.subplot(2, 2, 3)
        plt.scatter(df_results['avg_te'], df_results['avg_ii'])
        
        plt.xlabel('Transfer Entropy')
        plt.ylabel('Integrated Information')
        title = 'Correlation: %.4f (p = %.6f)' % (analysis['pearson_corr'], analysis['pearson_p'])
        plt.title(title)
        plt.grid(True)
        
        # Summary findings
        plt.subplot(2, 2, 4)
        
        findings_text = """
Relationship Analysis:

1. Classification: %s
2. Pearson Correlation: %.4f (p = %.6f)
3. Spearman Correlation: %.4f (p = %.6f)
4. Mutual Information: %.4f
5. Threshold Relationship: %s
6. Max Curve Difference: %.4f
7. Growth Rate Correlation: %.4f (p = %.6f)

Key Findings:
   - %s
   - %s
   - %s
""" % (analysis['relationship'], analysis['pearson_corr'], analysis['pearson_p'], 
       analysis['spearman_corr'], analysis['spearman_p'], analysis['mutual_info'], 
       analysis['threshold_relationship'], analysis['max_curve_diff'], analysis['growth_corr'], 
       analysis['growth_p'], 
       'These metrics appear to capture similar aspects of golden ratio organization.' if analysis['pearson_corr'] > 0.7 else 'These metrics appear to capture different aspects of golden ratio organization.', 
       'They respond at similar thresholds, suggesting they detect the same critical level of organization.' if 'similar' in analysis['threshold_relationship'] else 'They have different response thresholds, suggesting they are sensitive to different levels of organization.', 
       'Their growth patterns are highly correlated, indicating similar sensitivity to changes in organization.' if analysis['growth_corr'] > 0.7 else 'Their growth patterns differ, indicating different sensitivity to changes in organization.')
        
        plt.text(0.1, 0.9, findings_text, fontsize=12, va='top', ha='left', 
                 transform=plt.gca().transAxes)
        plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        plt.savefig(os.path.join(self.output_dir, 'summary_findings.png'), dpi=300, bbox_inches='tight')
        
        print("Visualizations created and saved to output directory.")

def main():
    """Main function to run the correlation analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run correlation analysis between Transfer Entropy and Integrated Information')
    parser.add_argument('--n-surrogates', type=int, default=100, 
                        help='Number of surrogate datasets to generate for each level (default: 100)')
    parser.add_argument('--n-gr-levels', type=int, default=10, 
                        help='Number of golden ratio organization levels to test (default: 10)')
    parser.add_argument('--output-dir', type=str, default='./correlation_results', 
                        help='Directory to save results (default: ./correlation_results)')
    parser.add_argument('--no-parallel', action='store_true', 
                        help='Disable parallel processing')
    parser.add_argument('--random-seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--use-real-data', action='store_true', 
                        help='Use real WMAP data instead of synthetic data')
    
    args = parser.parse_args()
    
    # Create the analyzer
    analyzer = MetricCorrelationAnalysis(
        output_dir=args.output_dir,
        n_surrogates=args.n_surrogates,
        n_gr_levels=args.n_gr_levels,
        random_seed=args.random_seed,
        parallel=not args.no_parallel,
        use_real_data=args.use_real_data
    )
    
    # Run the experiment
    print("Starting correlation analysis with %d surrogates and %d GR levels" % (args.n_surrogates, args.n_gr_levels))
    print("Results will be saved to: %s" % args.output_dir)
    
    start_time = time.time()
    
    # Run the manipulation experiment
    results = analyzer.run_manipulation_experiment()
    
    # Analyze the results
    analysis = analyzer.analyze_results(results)
    
    # Create visualizations
    analyzer.create_visualizations(results, analysis)
    
    total_time = time.time() - start_time
    print("Analysis completed in %.2f seconds (%.2f minutes)" % (total_time, total_time/60))
    print("Relationship classification: %s" % analysis['relationship'])
    print("Pearson correlation: %.4f (p = %.6f)" % (analysis['pearson_corr'], analysis['pearson_p']))
    
    # Print path to results
    print("\nResults saved to: %s" % os.path.abspath(args.output_dir))
    print("Files generated:")
    print("  - %s" % os.path.join(args.output_dir, 'manipulation_experiment_results.csv'))
    print("  - %s" % os.path.join(args.output_dir, 'analysis_results.txt'))
    print("  - %s" % os.path.join(args.output_dir, 'response_analysis.png'))
    print("  - %s" % os.path.join(args.output_dir, 'individual_surrogates.png'))
    print("  - %s" % os.path.join(args.output_dir, 'summary_findings.png'))

if __name__ == "__main__":
    main()
