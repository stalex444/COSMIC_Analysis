#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golden ratio quantum correlation tests for CMB maps.
Part of the healpix quantum entanglement analysis.
"""

import numpy as np
import healpy as hp
from scipy import stats
from healpix_quantum_analysis import calculate_quantum_correlation

def golden_ratio_quantum_test(map_data, base_scale, max_depth=5, n_samples=1000):
    """
    Test for quantum correlations between scales related by the golden ratio.
    
    Parameters:
    map_data (ndarray): The HEALPix map data
    base_scale (float): The base angular scale to start from (in degrees)
    max_depth (int): How many levels of golden ratio scales to test
    n_samples (int): Number of random starting points to sample
    
    Returns:
    tuple: (gr_correlations, random_correlations) - Correlation values for GR and random scales
    """
    nside = hp.get_nside(map_data)
    
    # Define golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Convert base scale to radians
    base_scale_rad = np.radians(base_scale)
    
    # Create array of scales related by golden ratio
    gr_scales = np.array([base_scale_rad * phi**i for i in range(max_depth)])
    
    # Create array of random scales with same range but not related by GR
    min_scale = gr_scales[0]
    max_scale = gr_scales[-1]
    random_scales = np.random.uniform(min_scale, max_scale, size=max_depth)
    
    # Calculate quantum correlations for GR scales
    print("\nTesting Golden Ratio scales:")
    gr_correlations, gr_significance = calculate_quantum_correlation(
        map_data, np.linspace(0, np.pi, 10), np.linspace(0, 2*np.pi, 10), gr_scales, n_samples)
    
    # Calculate quantum correlations for random scales
    print("\nTesting Random scales:")
    random_correlations, random_significance = calculate_quantum_correlation(
        map_data, np.linspace(0, np.pi, 10), np.linspace(0, 2*np.pi, 10), random_scales, n_samples)
    
    return gr_correlations, gr_significance, random_correlations, random_significance, gr_scales, random_scales

def create_surrogate_map(map_data, n_surrogates=100):
    """
    Create surrogate maps that preserve power spectrum but randomize phases.
    
    Parameters:
    map_data (ndarray): The original HEALPix map data
    n_surrogates (int): Number of surrogate maps to generate
    
    Returns:
    list: List of surrogate maps
    """
    nside = hp.get_nside(map_data)
    
    # Convert map to spherical harmonic coefficients
    alm = hp.map2alm(map_data)
    
    # Get power spectrum
    cl = hp.alm2cl(alm)
    
    surrogate_maps = []
    
    for i in range(n_surrogates):
        print(f"Generating surrogate map {i+1}/{n_surrogates}")
        
        # Generate random phases while preserving power spectrum
        # Create new alms with same power spectrum but random phases
        new_alm = hp.synalm(cl, lmax=hp.Alm.getlmax(len(alm)))
        
        # Convert back to map
        surrogate_map = hp.alm2map(new_alm, nside)
        
        surrogate_maps.append(surrogate_map)
    
    return surrogate_maps

def test_with_surrogates(map_data, n_surrogates=100, base_scale=1.0, max_depth=5, n_samples=1000):
    """
    Run quantum entanglement test with surrogate maps for statistical comparison.
    
    Parameters:
    map_data (ndarray): The original HEALPix map data
    n_surrogates (int): Number of surrogate maps to generate
    base_scale (float): The base angular scale to start from (in degrees)
    max_depth (int): How many levels of golden ratio scales to test
    n_samples (int): Number of random starting points to sample
    
    Returns:
    dict: Dictionary with test results
    """
    # Generate surrogate maps
    surrogate_maps = create_surrogate_map(map_data, n_surrogates)
    
    # Test actual map
    print("\nTesting actual CMB map:")
    gr_corr, gr_sig, random_corr, random_sig, gr_scales, random_scales = golden_ratio_quantum_test(
        map_data, base_scale, max_depth, n_samples)
    
    # Test surrogate maps
    surrogate_gr_corrs = []
    surrogate_random_corrs = []
    
    for i, surr_map in enumerate(surrogate_maps):
        print(f"\nTesting surrogate map {i+1}/{n_surrogates}:")
        s_gr_corr, s_gr_sig, s_random_corr, s_random_sig, _, _ = golden_ratio_quantum_test(
            surr_map, base_scale, max_depth, n_samples)
        
        surrogate_gr_corrs.append(s_gr_corr)
        surrogate_random_corrs.append(s_random_corr)
    
    # Convert to arrays
    surrogate_gr_corrs = np.array(surrogate_gr_corrs)
    surrogate_random_corrs = np.array(surrogate_random_corrs)
    
    # Calculate mean and standard deviation for surrogates
    mean_surr_gr = np.nanmean(surrogate_gr_corrs, axis=0)
    std_surr_gr = np.nanstd(surrogate_gr_corrs, axis=0)
    
    mean_surr_random = np.nanmean(surrogate_random_corrs, axis=0)
    std_surr_random = np.nanstd(surrogate_random_corrs, axis=0)
    
    # Calculate z-scores
    z_scores_gr = (gr_corr - mean_surr_gr) / std_surr_gr
    z_scores_random = (random_corr - mean_surr_random) / std_surr_random
    
    # Calculate p-values (two-tailed)
    p_values_gr = 2 * (1 - stats.norm.cdf(np.abs(z_scores_gr)))
    p_values_random = 2 * (1 - stats.norm.cdf(np.abs(z_scores_random)))
    
    # Compare GR vs Random
    gr_vs_random_ratio = np.nanmean(gr_corr) / np.nanmean(random_corr)
    gr_vs_random_z = (np.nanmean(gr_corr) - np.nanmean(random_corr)) / np.nanstd(np.concatenate([gr_corr, random_corr]))
    gr_vs_random_p = 2 * (1 - stats.norm.cdf(np.abs(gr_vs_random_z)))
    
    # Compile results
    results = {
        'gr_correlations': gr_corr,
        'random_correlations': random_corr,
        'gr_scales': gr_scales,
        'random_scales': random_scales,
        'mean_surrogate_gr': mean_surr_gr,
        'std_surrogate_gr': std_surr_gr,
        'mean_surrogate_random': mean_surr_random,
        'std_surrogate_random': std_surr_random,
        'z_scores_gr': z_scores_gr,
        'z_scores_random': z_scores_random,
        'p_values_gr': p_values_gr,
        'p_values_random': p_values_random,
        'gr_vs_random_ratio': gr_vs_random_ratio,
        'gr_vs_random_z': gr_vs_random_z,
        'gr_vs_random_p': gr_vs_random_p
    }
    
    return results
