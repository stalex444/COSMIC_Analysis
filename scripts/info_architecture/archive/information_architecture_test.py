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
from multiprocessing import Pool, cpu_count
import time
import os
from datetime import datetime
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import pdist, squareform
import argparse
from astropy.io import fits

# Simple data loading functions that don't rely on healpy
def load_cmb_map(filename):
    """Load CMB map from a file."""
    # Check if the file is a text file with power spectrum data
    if filename.endswith('.txt'):
        # Load power spectrum data from text file
        data = np.loadtxt(filename, comments='#')
        # Extract multipole moments (l) and power spectrum values
        if data.ndim > 1:
            ell = data[:, 0].astype(int)
            cl = data[:, 1]  # Using the TT power spectrum (column 2)
            
            # Create a dictionary mapping multipole moments to power spectrum values
            cl_dict = {l: c for l, c in zip(ell, cl)}
            
            # Return the power spectrum data instead of a map
            return {'ell': ell, 'cl': cl_dict, 'is_spectrum': True}
        else:
            # Single column data
            return {'data': data, 'is_spectrum': True, 'single_column': True}
    else:
        # Try to load as a FITS file
        try:
            hdul = fits.open(filename)
            # Assuming the CMB map is in the first extension
            cmb_map = hdul[1].data
            hdul.close()
            return {'map': cmb_map, 'is_spectrum': False, 'simple_map': True}
        except Exception as e:
            raise ValueError("Failed to load file %s: %s" % (filename, str(e)))

def ensure_dir_exists(dir_path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_wmap_power_spectrum(filename=None):
    """
    Load WMAP power spectrum data.
    
    Args:
        filename: Path to WMAP power spectrum file
        
    Returns:
        numpy.ndarray: Power spectrum data
    """
    if filename is None or not os.path.exists(filename):
        # Default location
        filename = "../data/wmap_tt_spectrum_9yr_v5.txt"
        
    if not os.path.exists(filename):
        # Create sample data if file doesn't exist
        print("WMAP data file not found. Creating sample data...")
        return create_sample_data(seed=42)
    
    try:
        # Load data directly using numpy
        data = np.loadtxt(filename, comments='#')
        
        # Extract power spectrum
        if data.ndim > 1:
            # If data has multiple columns, use the second column (index 1)
            return data[:, 1]
        else:
            # If data has only one column, use it directly
            return data
    except Exception as e:
        print("Error loading WMAP data: %s" % str(e))
        print("Creating sample data instead...")
        return create_sample_data(seed=42)

def load_planck_power_spectrum(filename=None):
    """
    Load Planck power spectrum data.
    
    Args:
        filename: Path to Planck power spectrum file
        
    Returns:
        numpy.ndarray: Power spectrum data
    """
    if filename is None or not os.path.exists(filename):
        # Default location
        filename = "../data/planck_tt_spectrum_2018.txt"
        
    if not os.path.exists(filename):
        # Create sample data if file doesn't exist
        print("Planck data file not found. Creating sample data...")
        return create_sample_data(seed=43)
    
    try:
        # Load data directly using numpy
        data = np.loadtxt(filename, comments='#')
        
        # Extract power spectrum
        if data.ndim > 1:
            # If data has multiple columns, use the second column (index 1)
            return data[:, 1]
        else:
            # If data has only one column, use it directly
            return data
    except Exception as e:
        print("Error loading Planck data: %s" % str(e))
        print("Creating sample data instead...")
        return create_sample_data(seed=43)

def create_sample_data(n_points=2000, seed=None):
    """
    Create sample power spectrum data for testing.
    
    Args:
        n_points: Number of data points
        seed: Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Sample power spectrum
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create sample power spectrum with realistic features
    ell = np.arange(2, n_points+2)
    
    # CMB-like power spectrum: C_l ~ 1/l^2 with acoustic peaks
    base_spectrum = 1000 * (ell * (ell + 1))**(-1)
    
    # Add acoustic peaks
    peak_positions = [220, 540, 800, 1120]
    peak_heights = [5, 2.5, 1.2, 0.8]
    peak_widths = [80, 70, 60, 50]
    
    for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
        base_spectrum += height * np.exp(-0.5 * ((ell - pos) / width)**2)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, size=len(base_spectrum))
    spectrum = base_spectrum * (1 + noise)
    
    # Ensure positive values
    spectrum = np.maximum(spectrum, 0)
    
    return spectrum

# Helper functions for parallel processing
def _run_single_simulation(data, constant, seed, config):
    """Run a single Monte Carlo simulation."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create surrogate data by permuting the original data
    surrogate_data = np.random.permutation(data)
    
    # Create a temporary test instance to calculate the score
    temp_test = InformationArchitectureTest(config)
    
    # Calculate architecture score for surrogate data
    surrogate_score = temp_test.calculate_architecture_score(surrogate_data, constant)
    
    return surrogate_score

def _run_single_simulation_star(args):
    """Wrapper for _run_single_simulation to be used with multiprocessing.Pool.map"""
    return _run_single_simulation(*args)

class InformationArchitectureTest:
    """
    Information Architecture Test for Cosmic Microwave Background Data.
    
    This test examines how different mathematical constants organize different aspects 
    of the hierarchical information structure in the CMB.
    """
    
    def __init__(self, config=None):
        """Initialize the Information Architecture Test with configuration parameters."""
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
            'hierarchy_depth': 5,         # Number of hierarchical levels to analyze
            'window_size': 20,            # Window size for local analysis
            'significance_level': 0.05,   # Statistical significance threshold
            'parallel_processing': True,  # Whether to use parallel processing
            'batch_size': 1000,           # Batch size for parallel processing
            'early_stopping': True,       # Whether to stop early if significance is reached
            'timeout': 3600,              # Timeout in seconds (1 hour)
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Initialize results storage
        self.results = {}
        
        # Initialize cache for performance optimization
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def extract_hierarchical_layers(self, data):
        """
        Extract hierarchical layers from the data using multi-scale analysis.
        
        Args:
            data: CMB power spectrum or time series
            
        Returns:
            dict: Hierarchical layers at different scales
        """
        # Use cache if available
        try:
            data_hash = hash(data.tobytes()) if hasattr(data, 'tobytes') else hash(str(data))
            cache_key = ('hierarchical_layers', data_hash)
            
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            
            self.cache_misses += 1
        except:
            # If hashing fails, skip caching
            pass
        
        layers = {}
        
        # Define scale factors for hierarchical layers
        scales = [2, 4, 8, 16, 32]
        
        for scale in scales:
            # Extract data at this scale using wavelet transform or moving average
            if scale == 1:
                # Base layer is the original data
                layers[scale] = data
            else:
                # Apply coarse-graining to get higher-level structure
                layers[scale] = self._coarse_grain(data, scale)
                
        # Cache the result
        try:
            if 'cache_key' in locals():
                self.cache[cache_key] = layers
        except:
            pass
                
        return layers
    
    def _coarse_grain(self, data, scale):
        """
        Apply coarse-graining to extract structure at a specific scale.
        
        Args:
            data: Original time series
            scale: Scale factor for coarse-graining
            
        Returns:
            ndarray: Coarse-grained data at specified scale
        """
        # If scale is large relative to data, use smaller window
        window_size = min(scale, len(data) // 10)
        
        # Apply moving average
        weights = np.ones(window_size) / window_size
        coarse_data = np.convolve(data, weights, mode='valid')
        
        # Downsample to match scale
        step = max(1, window_size // 2)
        return coarse_data[::step]
    
    def measure_layer_organization(self, layers, constant):
        """
        Measure how well a constant organizes each hierarchical layer.
        
        Args:
            layers: Dictionary of hierarchical layers
            constant: Mathematical constant to test
            
        Returns:
            dict: Organization metrics for each layer
        """
        # Use cache if available
        try:
            layers_hash = hash(str(layers.keys()))
            cache_key = ('layer_organization', layers_hash, constant)
            
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            
            self.cache_misses += 1
        except:
            # If hashing fails, skip caching
            pass
        
        organization = {}
        
        for scale, layer in layers.items():
            if len(layer) < 10:
                # Skip layers with too little data
                continue
                
            # Measure organization at this layer using various metrics
            # 1. Self-similarity at constant-related points
            self_similarity = self._measure_self_similarity(layer, constant)
            
            # 2. Information flow patterns optimized by constant
            info_flow = self._measure_information_flow(layer, constant)
            
            # 3. Coherence structures organized by constant
            coherence = self._measure_coherence_structure(layer, constant)
            
            # Combine metrics
            organization[scale] = {
                'self_similarity': self_similarity,
                'information_flow': info_flow,
                'coherence': coherence,
                'combined_score': (self_similarity + info_flow + coherence) / 3
            }
        
        # Cache the result
        try:
            if 'cache_key' in locals():
                self.cache[cache_key] = organization
        except:
            pass
            
        return organization
    
    def _measure_self_similarity(self, layer, constant):
        """
        Measure self-similarity at points related by the constant.
        
        Args:
            layer: Data at a specific hierarchical layer
            constant: Mathematical constant to test
            
        Returns:
            float: Self-similarity score
        """
        if len(layer) < 20:
            return 0
            
        similarities = []
        
        for i in range(len(layer) // 4):  # Use first quarter as seeds
            j = int(i * constant)
            if j >= len(layer):
                continue
                
            # Calculate similarity between points spaced by the constant
            window_i = layer[max(0, i-5):min(len(layer), i+5)]
            window_j = layer[max(0, j-5):min(len(layer), j+5)]
            
            if len(window_i) < 3 or len(window_j) < 3:
                continue
                
            try:
                corr, _ = stats.pearsonr(window_i, window_j)
                similarities.append(abs(corr))
            except:
                continue
        
        # Return average similarity
        return np.mean(similarities) if similarities else 0
    
    def _measure_information_flow(self, layer, constant):
        """
        Measure information flow patterns optimized by the constant.
        
        Args:
            layer: Data at a specific hierarchical layer
            constant: Mathematical constant to test
            
        Returns:
            float: Information flow score
        """
        if len(layer) < 20:
            return 0
            
        flow_scores = []
        
        for i in range(len(layer) // 4):  # Use first quarter for efficiency
            j = int(i * constant)
            if j >= len(layer):
                continue
                
            # Calculate transfer entropy between points spaced by the constant
            window_i = layer[max(0, i-5):min(len(layer), i+5)]
            window_j = layer[max(0, j-5):min(len(layer), j+5)]
            
            if len(window_i) < 5 or len(window_j) < 5:
                continue
                
            te = self._estimate_transfer_entropy(window_i, window_j)
            flow_scores.append(te)
        
        # Return average flow
        return np.mean(flow_scores) if flow_scores else 0
    
    def _measure_coherence_structure(self, layer, constant):
        """
        Measure coherence structures organized by the constant.
        
        Args:
            layer: Data at a specific hierarchical layer
            constant: Mathematical constant to test
            
        Returns:
            float: Coherence structure score
        """
        if len(layer) < 30:
            return 0
            
        # Build a network of points related by the constant
        graph = nx.Graph()
        
        for i in range(len(layer)):
            graph.add_node(i, value=layer[i])
            
            j = int(i / constant)
            if 0 <= j < len(layer):
                # Add edge if correlation is significant
                window_i = layer[max(0, i-3):min(len(layer), i+3)]
                window_j = layer[max(0, j-3):min(len(layer), j+3)]
                
                if len(window_i) >= 3 and len(window_j) >= 3:
                    try:
                        corr, _ = stats.pearsonr(window_i, window_j)
                        if abs(corr) > 0.5:  # Only add significant edges
                            graph.add_edge(i, j, weight=abs(corr))
                    except:
                        pass
            
            j = int(i * constant)
            if 0 <= j < len(layer):
                # Add edge if correlation is significant
                window_i = layer[max(0, i-3):min(len(layer), i+3)]
                window_j = layer[max(0, j-3):min(len(layer), j+3)]
                
                if len(window_i) >= 3 and len(window_j) >= 3:
                    try:
                        corr, _ = stats.pearsonr(window_i, window_j)
                        if abs(corr) > 0.5:  # Only add significant edges
                            graph.add_edge(i, j, weight=abs(corr))
                    except:
                        pass
        
        # Calculate network metrics
        if graph.number_of_edges() == 0:
            return 0
            
        # Clustering coefficient measures local organization
        clustering = nx.average_clustering(graph, weight='weight')
        
        # Efficiency measures information flow capacity
        try:
            efficiency = nx.global_efficiency(graph)
        except:
            efficiency = 0
            
        # Modularity measures community structure
        try:
            communities = nx.community.greedy_modularity_communities(graph)
            modularity = nx.community.modularity(graph, communities)
        except:
            modularity = 0
        
        # Combine metrics
        return (clustering + efficiency + modularity) / 3
    
    def measure_interlayer_interactions(self, layers, constant):
        """
        Measure how well a constant organizes interactions between hierarchical layers.
        
        Args:
            layers: Dictionary of hierarchical layers
            constant: Mathematical constant to test
            
        Returns:
            dict: Interaction metrics between layers
        """
        # Use cache if available
        try:
            layers_hash = hash(str(layers.keys()))
            cache_key = ('interlayer_interactions', layers_hash, constant)
            
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            
            self.cache_misses += 1
        except:
            # If hashing fails, skip caching
            pass
        
        interactions = {}
        
        scales = sorted(layers.keys())
        for i in range(len(scales) - 1):
            scale_i = scales[i]
            scale_j = scales[i + 1]
            
            layer_i = layers[scale_i]
            layer_j = layers[scale_j]
            
            # Skip if either layer has too little data
            if len(layer_i) < 10 or len(layer_j) < 10:
                continue
            
            # Resample larger layer to match smaller one if needed
            if len(layer_i) > len(layer_j):
                indices = np.linspace(0, len(layer_i) - 1, len(layer_j), dtype=int)
                layer_i_resampled = layer_i[indices]
                flow_i_to_j = self._measure_interlayer_flow(layer_i_resampled, layer_j, constant)
            elif len(layer_j) > len(layer_i):
                indices = np.linspace(0, len(layer_j) - 1, len(layer_i), dtype=int)
                layer_j_resampled = layer_j[indices]
                flow_i_to_j = self._measure_interlayer_flow(layer_i, layer_j_resampled, constant)
            else:
                flow_i_to_j = self._measure_interlayer_flow(layer_i, layer_j, constant)
            
            interactions[(scale_i, scale_j)] = flow_i_to_j
        
        # Cache the result
        try:
            if 'cache_key' in locals():
                self.cache[cache_key] = interactions
        except:
            pass
            
        return interactions
    
    def _measure_interlayer_flow(self, layer_i, layer_j, constant):
        """
        Measure information flow between two hierarchical layers.
        
        Args:
            layer_i: Data from first layer
            layer_j: Data from second layer
            constant: Mathematical constant to test
            
        Returns:
            float: Interlayer flow score
        """
        # Ensure equal length
        min_len = min(len(layer_i), len(layer_j))
        layer_i = layer_i[:min_len]
        layer_j = layer_j[:min_len]
        
        if min_len < 20:
            return 0
        
        # Calculate correlations between points related by constant
        constant_correlations = []
        
        for i in range(min_len // 4):  # Use first quarter for efficiency
            j = int(i * constant) % min_len
            
            # Calculate correlation between layers at related points
            i_window = max(0, i - 3)
            j_window = max(0, j - 3)
            
            if i_window + 6 > min_len or j_window + 6 > min_len:
                continue
                
            window_i = layer_i[i_window:i_window + 6]
            window_j = layer_j[j_window:j_window + 6]
            
            try:
                corr, _ = stats.pearsonr(window_i, window_j)
                constant_correlations.append(abs(corr))
            except:
                continue
        
        # Calculate correlations between random points as baseline
        random_correlations = []
        
        for _ in range(len(constant_correlations)):
            i = np.random.randint(0, min_len - 6)
            j = np.random.randint(0, min_len - 6)
            
            window_i = layer_i[i:i + 6]
            window_j = layer_j[j:j + 6]
            
            try:
                corr, _ = stats.pearsonr(window_i, window_j)
                random_correlations.append(abs(corr))
            except:
                continue
        
        # Calculate advantage ratio
        if not random_correlations or np.mean(random_correlations) == 0:
            return 0
            
        return np.mean(constant_correlations) / np.mean(random_correlations)
    
    def measure_information_architecture(self, data, constant):
        """
        Measure how well a constant organizes the entire hierarchical information architecture.
        
        Args:
            data: CMB power spectrum or time series
            constant: Mathematical constant to test
            
        Returns:
            dict: Architecture metrics for the constant
        """
        # Use cache if available
        try:
            data_hash = hash(data.tobytes()) if hasattr(data, 'tobytes') else hash(str(data))
            cache_key = ('information_architecture', data_hash, constant)
            
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            
            self.cache_misses += 1
        except:
            # If hashing fails, skip caching
            pass
        
        # Extract hierarchical layers
        layers = self.extract_hierarchical_layers(data)
        
        # Measure organization within each layer
        layer_organization = self.measure_layer_organization(layers, constant)
        
        # Measure interactions between layers
        interlayer_interactions = self.measure_interlayer_interactions(layers, constant)
        
        # Calculate overall architecture metrics
        layer_scores = [layer['combined_score'] for layer in layer_organization.values()]
        interaction_scores = list(interlayer_interactions.values())
        
        if not layer_scores or not interaction_scores:
            result = {
                'mean_layer_score': 0,
                'mean_interaction_score': 0,
                'architecture_score': 0,
                'num_layers': len(layers),
                'num_interactions': len(interlayer_interactions)
            }
        else:
            mean_layer_score = np.mean(layer_scores)
            mean_interaction_score = np.mean(interaction_scores)
            
            # Calculate combined architecture score (weighted toward interactions)
            architecture_score = (mean_layer_score + 2 * mean_interaction_score) / 3
            
            result = {
                'mean_layer_score': mean_layer_score,
                'mean_interaction_score': mean_interaction_score,
                'architecture_score': architecture_score,
                'num_layers': len(layers),
                'num_interactions': len(interlayer_interactions),
                'layers': layer_organization,
                'interactions': interlayer_interactions
            }
        
        # Cache the result
        try:
            if 'cache_key' in locals():
                self.cache[cache_key] = result
        except:
            pass
        
        return result
    
    def measure_complementary_architecture(self, data):
        """
        Measure how phi and sqrt2 might organize different aspects of the information architecture.
        
        Args:
            data: CMB power spectrum or time series
            
        Returns:
            dict: Complementary architecture metrics
        """
        # Use cache if available
        try:
            data_hash = hash(data.tobytes()) if hasattr(data, 'tobytes') else hash(str(data))
            cache_key = ('complementary_architecture', data_hash)
            
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            
            self.cache_misses += 1
        except:
            # If hashing fails, skip caching
            pass
        
        # Measure individual architecture organization
        phi_architecture = self.measure_information_architecture(data, self.constants['phi'])
        sqrt2_architecture = self.measure_information_architecture(data, self.constants['sqrt2'])
        
        # Extract hierarchical layers
        layers = self.extract_hierarchical_layers(data)
        
        # Measure layer specialization (which constant organizes which layer better)
        layer_specialization = {}
        
        for scale in sorted(layers.keys()):
            if scale in phi_architecture.get('layers', {}) and scale in sqrt2_architecture.get('layers', {}):
                phi_score = phi_architecture['layers'][scale]['combined_score']
                sqrt2_score = sqrt2_architecture['layers'][scale]['combined_score']
                
                if phi_score > 0 and sqrt2_score > 0:
                    specialization_ratio = max(phi_score / sqrt2_score, sqrt2_score / phi_score)
                    specialization_constant = 'phi' if phi_score > sqrt2_score else 'sqrt2'
                    
                    layer_specialization[scale] = {
                        'specialization_ratio': specialization_ratio,
                        'specialized_constant': specialization_constant,
                        'phi_score': phi_score,
                        'sqrt2_score': sqrt2_score
                    }
        
        # Measure interaction specialization
        interaction_specialization = {}
        
        phi_interactions = phi_architecture.get('interactions', {})
        sqrt2_interactions = sqrt2_architecture.get('interactions', {})
        
        for interaction in set(phi_interactions.keys()) & set(sqrt2_interactions.keys()):
            phi_score = phi_interactions[interaction]
            sqrt2_score = sqrt2_interactions[interaction]
            
            if phi_score > 0 and sqrt2_score > 0:
                specialization_ratio = max(phi_score / sqrt2_score, sqrt2_score / phi_score)
                specialization_constant = 'phi' if phi_score > sqrt2_score else 'sqrt2'
                
                interaction_specialization[interaction] = {
                    'specialization_ratio': specialization_ratio,
                    'specialized_constant': specialization_constant,
                    'phi_score': phi_score,
                    'sqrt2_score': sqrt2_score
                }
        
        # Calculate complementary organization scores
        if layer_specialization:
            avg_layer_specialization = np.mean([l['specialization_ratio'] for l in layer_specialization.values()])
        else:
            avg_layer_specialization = 0
            
        if interaction_specialization:
            avg_interaction_specialization = np.mean([i['specialization_ratio'] for i in interaction_specialization.values()])
        else:
            avg_interaction_specialization = 0
        
        # Count the number of layers where each constant is specialized
        phi_specialized_layers = sum(1 for l in layer_specialization.values() if l['specialized_constant'] == 'phi')
        sqrt2_specialized_layers = sum(1 for l in layer_specialization.values() if l['specialized_constant'] == 'sqrt2')
        
        # Count the number of interactions where each constant is specialized
        phi_specialized_interactions = sum(1 for i in interaction_specialization.values() if i['specialized_constant'] == 'phi')
        sqrt2_specialized_interactions = sum(1 for i in interaction_specialization.values() if i['specialized_constant'] == 'sqrt2')
        
        # Calculate an overall complementary architecture score
        complementary_score = (avg_layer_specialization + avg_interaction_specialization) / 2
        
        result = {
            'phi_architecture': phi_architecture,
            'sqrt2_architecture': sqrt2_architecture,
            'layer_specialization': layer_specialization,
            'interaction_specialization': interaction_specialization,
            'avg_layer_specialization': avg_layer_specialization,
            'avg_interaction_specialization': avg_interaction_specialization,
            'phi_specialized_layers': phi_specialized_layers,
            'sqrt2_specialized_layers': sqrt2_specialized_layers,
            'phi_specialized_interactions': phi_specialized_interactions,
            'sqrt2_specialized_interactions': sqrt2_specialized_interactions,
            'complementary_score': complementary_score
        }
        
        # Cache the result
        try:
            if 'cache_key' in locals():
                self.cache[cache_key] = result
        except:
            pass
        
        return result
    
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

    def calculate_architecture_score(self, data, constant):
        """
        Calculate the architecture score for a given dataset and constant.
        
        Args:
            data: CMB power spectrum or time series
            constant: Mathematical constant to test
            
        Returns:
            float: Architecture score
        """
        # Use cache if available
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
        
        # Measure information architecture
        architecture = self.measure_information_architecture(data, constant)
        
        # Cache the result
        try:
            if 'cache_key' in locals():
                self.cache[cache_key] = architecture['architecture_score']
        except:
            pass
            
        return architecture['architecture_score']
    
    def run_monte_carlo_simulation(self, data, constant, num_simulations=None, output_dir=None):
        """
        Run Monte Carlo simulation to determine statistical significance.
        
        Args:
            data: Input data
            constant: Mathematical constant to test
            num_simulations: Number of simulations to run
            output_dir: Directory to save progress and results
            
        Returns:
            dict: Results including p-value and significance
        """
        if num_simulations is None:
            num_simulations = self.config.get('num_simulations', 1000)
            
        # Get actual score
        actual_score = self.calculate_architecture_score(data, constant)
        
        # Initialize progress tracking
        progress_file = None
        if output_dir:
            ensure_dir_exists(output_dir)
            progress_file = os.path.join(output_dir, "progress_%s_simulations.txt" % constant)
            
            # Write header to progress file
            with open(progress_file, 'w') as f:
                f.write("# Information Architecture Test - Monte Carlo Simulation\n")
                f.write("# Constant: %s\n" % constant)
                f.write("# Actual Score: %s\n" % actual_score)
                f.write("# Simulation Progress:\n")
                f.write("# Simulation,Score,p-value\n")
        
        # Count how many random scores are >= actual score
        count_greater_equal = 0
        all_scores = []
        
        # Use parallel processing if enabled
        if self.config.get('parallel_processing', True):
            # Initialize progress tracking
            start_time = time.time()
            last_update_time = start_time
            
            # Create a pool of workers
            num_workers = min(cpu_count(), self.config.get('max_workers', cpu_count()))
            pool = Pool(processes=num_workers)
            
            # Prepare arguments for parallel execution - include config for each worker
            args = [(data, constant, i, self.config) for i in range(num_simulations)]
            
            # Execute simulations in parallel with a chunksize that balances overhead and distribution
            chunksize = max(1, num_simulations // (num_workers * 4))
            
            # Track progress
            completed = 0
            
            # Process results as they come in
            for i, score in enumerate(pool.imap_unordered(_run_single_simulation_star, args, chunksize=chunksize)):
                all_scores.append(score)
                if score >= actual_score:
                    count_greater_equal += 1
                
                # Calculate p-value
                p_value = float(count_greater_equal) / (i + 1)
                
                # Update progress file
                if progress_file:
                    with open(progress_file, 'a') as f:
                        f.write("%d,%f,%f\n" % (i, score, p_value))
                
                # Print progress
                completed += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Update progress more frequently (every 5 simulations or every 5 seconds)
                if completed % 5 == 0 or (current_time - last_update_time) >= 5:
                    last_update_time = current_time
                    
                    # Calculate progress percentage and estimated time remaining
                    progress_pct = 100.0 * completed / num_simulations
                    if completed > 0:
                        avg_time_per_sim = elapsed_time / completed
                        remaining_sims = num_simulations - completed
                        est_remaining_time = avg_time_per_sim * remaining_sims
                    else:
                        est_remaining_time = 0
                    
                    # Create a text-based progress bar
                    bar_length = 30
                    filled_length = int(bar_length * completed // num_simulations)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    # Print progress with progress bar
                    print("\rProgress: [%s] %5.1f%% (%d/%d) - p-value: %.6f - Elapsed: %.1fs - Est. remaining: %.1fs" % 
                          (bar, progress_pct, completed, num_simulations, p_value, 
                           elapsed_time, est_remaining_time)),
                
                # Early stopping if p-value is already significant or definitely not significant
                if self.config.get('early_stopping', True) and i >= self.config.get('min_simulations', 100):
                    # If p-value is very low, we can stop early
                    if p_value < self.config.get('significance_threshold', 0.01) / 10:
                        print("\nEarly stopping: p-value is significantly low (%.6f)" % p_value)
                        break
                    
                    # If p-value is very high, we can also stop early
                    if p_value > 0.5 and (i + 1) >= num_simulations / 2:
                        print("\nEarly stopping: p-value is high (%.6f) after %d simulations" % (p_value, i + 1))
                        break
            
            # Close the pool
            pool.close()
            pool.join()
            
            # Print final progress
            print("\nCompleted %d/%d simulations in %.1f seconds" % (completed, num_simulations, elapsed_time))
            
        else:
            # Sequential processing
            start_time = time.time()
            last_update_time = start_time
            
            for i in range(num_simulations):
                # Run a single simulation
                score = _run_single_simulation(data, constant, i, self.config)
                all_scores.append(score)
                
                if score >= actual_score:
                    count_greater_equal += 1
                
                # Calculate p-value
                p_value = float(count_greater_equal) / (i + 1)
                
                # Update progress file
                if progress_file and i % 10 == 0:
                    with open(progress_file, 'a') as f:
                        f.write("%d,%f,%f\n" % (i, score, p_value))
                
                # Print progress
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Update progress more frequently (every 10 simulations or every 5 seconds)
                if i % 10 == 0 or (current_time - last_update_time) >= 5:
                    last_update_time = current_time
                    
                    # Calculate progress percentage and estimated time remaining
                    progress_pct = 100.0 * (i + 1) / num_simulations
                    if i > 0:
                        avg_time_per_sim = elapsed_time / (i + 1)
                        remaining_sims = num_simulations - (i + 1)
                        est_remaining_time = avg_time_per_sim * remaining_sims
                    else:
                        est_remaining_time = 0
                    
                    # Create a text-based progress bar
                    bar_length = 30
                    filled_length = int(bar_length * (i + 1) // num_simulations)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    # Print progress with progress bar
                    print("\rProgress: [%s] %5.1f%% (%d/%d) - p-value: %.6f - Elapsed: %.1fs - Est. remaining: %.1fs" % 
                          (bar, progress_pct, i + 1, num_simulations, p_value, 
                           elapsed_time, est_remaining_time)),
                
                # Early stopping if p-value is already significant or definitely not significant
                if self.config.get('early_stopping', True) and i >= self.config.get('min_simulations', 100):
                    # If p-value is very low, we can stop early
                    if p_value < self.config.get('significance_threshold', 0.01) / 10:
                        print("\nEarly stopping: p-value is significantly low (%.6f)" % p_value)
                        break
                    
                    # If p-value is very high, we can also stop early
                    if p_value > 0.5 and i >= num_simulations / 2:
                        print("\nEarly stopping: p-value is high (%.6f) after %d simulations" % (p_value, i + 1))
                        break
            
            # Print final progress
            print("\nCompleted %d/%d simulations in %.1f seconds" % (min(i + 1, num_simulations), num_simulations, elapsed_time))
        
        # Calculate final p-value
        p_value = float(count_greater_equal) / num_simulations
        
        # Calculate z-score
        mean_surrogate = np.mean(all_scores)
        std_surrogate = np.std(all_scores)
        z_score = (actual_score - mean_surrogate) / std_surrogate if std_surrogate > 0 else 0
        
        # Prepare results
        results = {
            'constant': constant,
            'actual_score': actual_score,
            'surrogate_scores': all_scores,
            'num_simulations': len(all_scores),
            'p_value': p_value,
            'z_score': z_score,
            'significant': p_value < self.config.get('significance_level', 0.05),
            'mean_surrogate': mean_surrogate,
            'std_surrogate': std_surrogate
        }
        
        # Save final results
        if output_dir:
            results_file = os.path.join(output_dir, "results_%s.txt" % constant)
            with open(results_file, 'w') as f:
                f.write("# Information Architecture Test - Results\n")
                f.write("# Constant: %s\n" % constant)
                f.write("# Actual Score: %s\n" % actual_score)
                f.write("# Number of Simulations: %s\n" % len(all_scores))
                f.write("# p-value: %s\n" % p_value)
                f.write("# z-score: %s\n" % z_score)
                f.write("# Significant: %s\n" % (p_value < self.config.get('significance_level', 0.05)))
                f.write("# Mean Surrogate Score: %s\n" % mean_surrogate)
                f.write("# Std Surrogate Score: %s\n" % std_surrogate)
        
        return results
    
    def run_full_test(self, data, constants=None, output_dir=None, num_simulations=None):
        """
        Run the full Information Architecture Test on the data.
        
        Args:
            data: CMB power spectrum or time series
            constants: Dictionary of constants to test
            output_dir: Directory to save results
            num_simulations: Number of Monte Carlo simulations to run
            
        Returns:
            dict: Test results for all constants
        """
        # Set default values
        if constants is None:
            constants = self.constants
            
        if num_simulations is None:
            num_simulations = self.config.get('num_simulations', 1000)
            
        # Ensure output directory exists
        if output_dir:
            ensure_dir_exists(output_dir)
            
        # Initialize results
        results = {}
        
        # Run test for each constant
        for name, value in constants.items():
            print("\n" + "="*50)
            print("Testing constant: %s = %s" % (name, value))
            print("="*50)
            
            # Create output directory for this constant
            constant_dir = os.path.join(output_dir, name) if output_dir else None
            if constant_dir:
                ensure_dir_exists(constant_dir)
                
            # Run Monte Carlo simulation
            constant_results = self.run_monte_carlo_simulation(
                data, value, num_simulations, constant_dir
            )
            
            # Store results
            results[name] = constant_results
            
            # Print results
            print("Results for %s:" % name)
            print("  Architecture Score: %.6f" % constant_results['actual_score'])
            print("  p-value: %.6f" % constant_results['p_value'])
            print("  z-score: %.2f" % constant_results['z_score'])
            print("  Significant: %s" % constant_results['significant'])
            
        # Generate summary report
        if output_dir:
            self.generate_summary_report(results, output_dir)
            self.visualize_results(results, output_dir)
            
        return results
    
    def generate_summary_report(self, results, output_dir):
        """
        Generate a summary report of the test results.
        
        Args:
            results: Dictionary of test results
            output_dir: Directory to save report
        """
        # Ensure output directory exists
        ensure_dir_exists(output_dir)
        
        # Create summary report
        report_file = os.path.join(output_dir, "summary_report.txt")
        with open(report_file, 'w') as f:
            f.write("# Information Architecture Test - Summary Report\n")
            f.write("# Date: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            f.write("# Number of Simulations: %s\n" % results[list(results.keys())[0]]['num_simulations'])
            f.write("\n")
            
            # Write table header
            f.write("Constant,Value,Architecture Score,p-value,z-score,Significant\n")
            
            # Write results for each constant
            for name, constant_results in results.items():
                f.write("%s,%s,%.6f,%.6f,%.2f,%s\n" % (name, self.constants[name], constant_results['actual_score'], constant_results['p_value'], constant_results['z_score'], constant_results['significant']))
    
    def visualize_results(self, results, output_dir):
        """
        Generate visualizations of the test results.
        
        Args:
            results: Dictionary of test results
            output_dir: Directory to save visualizations
        """
        # Ensure output directory exists
        ensure_dir_exists(output_dir)
        
        # Create bar chart of architecture scores
        plt.figure(figsize=(12, 6))
        
        # Extract data for plotting
        constant_names = list(results.keys())
        architecture_scores = [results[name]['actual_score'] for name in constant_names]
        p_values = [results[name]['p_value'] for name in constant_names]
        
        # Sort by architecture score
        sorted_indices = np.argsort(architecture_scores)[::-1]
        constant_names = [constant_names[i] for i in sorted_indices]
        architecture_scores = [architecture_scores[i] for i in sorted_indices]
        p_values = [p_values[i] for i in sorted_indices]
        
        # Create bar chart
        bars = plt.bar(constant_names, architecture_scores, color='skyblue')
        
        # Highlight significant results
        for i, p in enumerate(p_values):
            if p < self.config.get('significance_level', 0.05):
                bars[i].set_color('green')
        
        # Add labels and title
        plt.xlabel('Mathematical Constant')
        plt.ylabel('Architecture Score')
        plt.title('Information Architecture Test Results')
        
        # Add p-values as text
        for i, (score, p) in enumerate(zip(architecture_scores, p_values)):
            plt.text(i, score + 0.02, "p=%.4f" % p, ha='center')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "architecture_scores.png"), dpi=300)
        plt.close()
        
        # Create histogram of surrogate scores for each constant
        for name, constant_results in results.items():
            plt.figure(figsize=(10, 6))
            
            # Extract data for plotting
            surrogate_scores = constant_results['surrogate_scores']
            actual_score = constant_results['actual_score']
            p_value = constant_results['p_value']
            
            # Create histogram
            plt.hist(surrogate_scores, bins=30, alpha=0.7, color='skyblue')
            
            # Add vertical line for actual score
            plt.axvline(actual_score, color='red', linestyle='--', linewidth=2)
            
            # Add labels and title
            plt.xlabel('Architecture Score')
            plt.ylabel('Frequency')
            plt.title('Surrogate Distribution for %s (p=%.4f)' % (name, p_value))
            
            # Add legend
            plt.legend(['Actual Score', 'Surrogate Scores'])
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "surrogate_distribution_%s.png" % name), dpi=300)
            plt.close()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Information Architecture Test for CMB Data')
    
    parser.add_argument('--wmap-file', type=str, default="../data/wmap_tt_spectrum_9yr_v5.txt",
                        help='Path to WMAP power spectrum data file')
    
    parser.add_argument('--planck-file', type=str, default="../data/planck_tt_spectrum_2018.txt",
                        help='Path to Planck power spectrum data file')
    
    parser.add_argument('--output-dir', type=str, default="../results/information_architecture",
                        help='Directory to save results')
    
    parser.add_argument('--num-simulations', type=int, default=10000,
                        help='Number of Monte Carlo simulations to run')
    
    parser.add_argument('--early-stopping', action='store_true', default=True,
                        help='Enable early stopping')
    
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Enable parallel processing')
    
    return parser.parse_args()

def main():
    """Main function to run the Information Architecture Test."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Define constants to test
    constants = {
        'phi': (1 + 5**0.5) / 2,  # Golden ratio
        'sqrt2': 2**0.5,          # Square root of 2
        'e': np.e,                # Euler's number
        'pi': np.pi,              # Pi
        'sqrt3': 3**0.5,          # Square root of 3
        'ln2': np.log(2)          # Natural log of 2
    }
    
    # Create test configuration
    config = {
        'num_simulations': args.num_simulations,
        'parallel_processing': args.parallel,
        'early_stopping': args.early_stopping,
    }
    
    # Run test on WMAP data
    print("\n" + "="*50)
    print("Running Information Architecture Test on WMAP data")
    print("="*50)
    
    # Load WMAP data
    wmap_data = load_wmap_power_spectrum(args.wmap_file)
    if wmap_data is not None:
        # Create output directory for WMAP results
        wmap_output_dir = os.path.join(args.output_dir, 'wmap')
        if not os.path.exists(wmap_output_dir):
            os.makedirs(wmap_output_dir)
        
        # Create test instance
        wmap_test = InformationArchitectureTest(config)
        
        # Run test on WMAP data
        wmap_results = wmap_test.run_full_test(wmap_data, constants, wmap_output_dir, args.num_simulations)
    else:
        print("Error: Failed to load WMAP data.")
    
    # Run test on Planck data
    print("\n" + "="*50)
    print("Running Information Architecture Test on Planck data")
    print("="*50)
    
    # Load Planck data
    planck_data = load_planck_power_spectrum(args.planck_file)
    if planck_data is not None:
        # Create output directory for Planck results
        planck_output_dir = os.path.join(args.output_dir, 'planck')
        if not os.path.exists(planck_output_dir):
            os.makedirs(planck_output_dir)
        
        # Create test instance
        planck_test = InformationArchitectureTest(config)
        
        # Run test on Planck data
        planck_results = planck_test.run_full_test(planck_data, constants, planck_output_dir, args.num_simulations)
    else:
        print("Error: Failed to load Planck data.")
    
    # Compare WMAP and Planck results if both are available
    if 'wmap_results' in locals() and 'planck_results' in locals():
        # Create output directory for comparison results
        comparison_output_dir = os.path.join(args.output_dir, 'comparison')
        if not os.path.exists(comparison_output_dir):
            os.makedirs(comparison_output_dir)
        
        # Generate comparison report
        comparison_report_file = os.path.join(comparison_output_dir, "comparison_report.txt")
        with open(comparison_report_file, 'w') as f:
            f.write("# Information Architecture Test - WMAP vs Planck Comparison\n")
            f.write("# Date: %s\n" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            f.write("\n")
            
            # Write table header
            f.write("Constant,WMAP Score,WMAP p-value,Planck Score,Planck p-value,Consistent\n")
            
            # Write results for each constant
            for name in constants:
                wmap_score = wmap_results[name]['actual_score']
                wmap_p = wmap_results[name]['p_value']
                planck_score = planck_results[name]['actual_score']
                planck_p = planck_results[name]['p_value']
                
                # Check if results are consistent (both significant or both not significant)
                consistent = (wmap_p < 0.05) == (planck_p < 0.05)
                
                f.write("%s,%s,%s,%s,%s,%s\n" % (name, wmap_score, wmap_p, planck_score, planck_p, consistent))
        
        # Generate comparison visualization
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        constant_names = list(constants.keys())
        wmap_scores = [wmap_results[name]['actual_score'] for name in constant_names]
        planck_scores = [planck_results[name]['actual_score'] for name in constant_names]
        
        # Set up bar positions
        x = np.arange(len(constant_names))
        width = 0.35
        
        # Create grouped bar chart
        plt.bar(x - width/2, wmap_scores, width, label='WMAP', color='skyblue')
        plt.bar(x + width/2, planck_scores, width, label='Planck', color='lightgreen')
        
        # Add labels and title
        plt.xlabel('Mathematical Constant')
        plt.ylabel('Architecture Score')
        plt.title('Information Architecture Test: WMAP vs Planck')
        plt.xticks(x, constant_names)
        plt.legend()
        
        # Add p-values as text
        for i, name in enumerate(constant_names):
            wmap_p = wmap_results[name]['p_value']
            planck_p = planck_results[name]['p_value']
            
            plt.text(i - width/2, wmap_scores[i] + 0.02, "p=%.4f" % wmap_p, ha='center', fontsize=8)
            plt.text(i + width/2, planck_scores[i] + 0.02, "p=%.4f" % planck_p, ha='center', fontsize=8)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_output_dir, "wmap_vs_planck.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
