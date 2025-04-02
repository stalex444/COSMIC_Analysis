"""
Scale-specific Analysis for Information Architecture Test
Analyzes specific scales in CMB data for mathematical constant optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import logging
from scipy import stats
import argparse
import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from info_architecture.cmb_info_architecture import (
    load_cmb_data, preprocess_cmb_data, define_hierarchical_scales,
    calculate_architecture_score, generate_surrogate
)

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
PI = np.pi  # Pi
E = np.e  # Euler's number
SQRT2 = np.sqrt(2)  # Square root of 2
SQRT3 = np.sqrt(3)  # Square root of 3
LN2 = np.log(2)  # Natural log of 2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('CMB-ScaleAnalysis')

def analyze_scale_specialization(data, scale_center, data_type='power_spectrum', 
                               constants=None, n_simulations=1000):
    """
    Analyze specialization of a specific scale for various mathematical constants.
    
    Parameters:
    - data: CMB data
    - scale_center: Scale center to analyze
    - data_type: Type of CMB data
    - constants: Dictionary of constants to test
    - n_simulations: Number of simulations for statistical testing
    
    Returns:
    - Dictionary of results
    """
    # Constants to test
    if constants is None:
        constants = {
            'phi': PHI,
            'pi': PI,
            'e': E,
            'sqrt2': SQRT2,
            'sqrt3': SQRT3,
            'ln2': LN2
        }
    
    # Define hierarchical scales
    scales = define_hierarchical_scales(data, method='conventional')
    
    # Find the scale closest to the requested scale_center
    if isinstance(scale_center, int):
        if str(scale_center) not in scales:
            closest_scale = min(scales.keys(), key=lambda x: abs(int(x) - scale_center))
            logger.info(f"Scale {scale_center} not found, using closest scale {closest_scale}")
            scale_center = closest_scale
        else:
            scale_center = str(scale_center)
    
    results = {}
    
    # Analyze specialization for each constant
    for name, value in constants.items():
        logger.info(f"Analyzing scale {scale_center} specialization for {name}...")
        
        # Calculate architecture score for the constant at this scale
        score = calculate_architecture_score(
            data, {scale_center: scales[scale_center]}, value, data_type
        )
        
        # Generate surrogate datasets and calculate their scores
        surrogate_scores = []
        for i in range(n_simulations):
            if i % 100 == 0 and i > 0:
                logger.info(f"Completed {i}/{n_simulations} simulations for {name}")
            
            # Generate surrogate
            surrogate = generate_surrogate(data, data_type)
            
            # Calculate score for surrogate
            surrogate_score = calculate_architecture_score(
                surrogate, {scale_center: scales[scale_center]}, value, data_type
            )
            surrogate_scores.append(surrogate_score)
        
        # Calculate statistics
        surrogate_mean = np.mean(surrogate_scores)
        surrogate_std = np.std(surrogate_scores)
        
        # Calculate z-score
        z_score = (score - surrogate_mean) / surrogate_std if surrogate_std > 0 else 0
        
        # Calculate p-value (one-tailed for optimization)
        p_value = 1 - stats.norm.cdf(z_score) if surrogate_std > 0 else 1.0
        
        # Store results
        results[name] = {
            'scale': scale_center,
            'score': score,
            'surrogate_mean': surrogate_mean,
            'surrogate_std': surrogate_std,
            'z_score': z_score,
            'p_value': p_value
        }
        
        logger.info(f"Results for {name} at scale {scale_center}:")
        logger.info(f"  Score: {score:.4f}")
        logger.info(f"  Mean surrogate: {surrogate_mean:.4f}")
        logger.info(f"  Z-score: {z_score:.4f}")
        logger.info(f"  P-value: {p_value:.6f}")
    
    return results

def visualize_scale_results(results, output_dir):
    """Create visualizations of scale-specific results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    scale = list(results.values())[0]['scale']
    
    # Bar chart of z-scores
    plt.figure(figsize=(12, 6))
    constants = list(results.keys())
    z_scores = [results[c]['z_score'] for c in constants]
    
    bars = plt.bar(constants, z_scores)
    
    # Color bars based on significance
    for i, bar in enumerate(bars):
        if z_scores[i] > 1.96:
            bar.set_color('green')
        elif z_scores[i] < -1.96:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=1.96, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=-1.96, color='r', linestyle='--', alpha=0.5)
    
    plt.title(f'Mathematical Constant Specialization at Scale {scale}')
    plt.ylabel('Z-score')
    plt.xlabel('Constant')
    
    plt.savefig(f"{output_dir}/scale_{scale}_specialization_{timestamp}.png", dpi=300)
    plt.close()
    
    # Create text summary
    with open(f"{output_dir}/scale_{scale}_specialization_{timestamp}.txt", 'w') as f:
        f.write(f"=== SCALE {scale} SPECIALIZATION ANALYSIS ===\n\n")
        
        for constant in constants:
            f.write(f"=== {constant.upper()} ===\n")
            f.write(f"Score: {results[constant]['score']:.4f}\n")
            f.write(f"Mean surrogate: {results[constant]['surrogate_mean']:.4f}\n")
            f.write(f"Standard deviation: {results[constant]['surrogate_std']:.4f}\n")
            f.write(f"Z-score: {results[constant]['z_score']:.4f}\n")
            f.write(f"P-value: {results[constant]['p_value']:.6f}\n\n")
    
    return f"{output_dir}/scale_{scale}_specialization_{timestamp}.txt"

def compare_datasets(wmap_results, planck_results, output_dir):
    """Compare scale specialization between WMAP and Planck datasets"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    scale = list(wmap_results.values())[0]['scale']
    
    # Prepare comparison data
    constants = list(wmap_results.keys())
    wmap_z = [wmap_results[c]['z_score'] for c in constants]
    planck_z = [planck_results[c]['z_score'] for c in constants]
    
    # Create bar chart
    plt.figure(figsize=(14, 7))
    width = 0.35
    x = np.arange(len(constants))
    
    plt.bar(x - width/2, wmap_z, width, label='WMAP')
    plt.bar(x + width/2, planck_z, width, label='Planck')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=1.96, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=-1.96, color='r', linestyle='--', alpha=0.5)
    
    plt.title(f'WMAP vs Planck: Scale {scale} Specialization')
    plt.ylabel('Z-score')
    plt.xlabel('Constant')
    plt.xticks(x, constants)
    plt.legend()
    
    plt.savefig(f"{output_dir}/scale_{scale}_comparison_{timestamp}.png", dpi=300)
    plt.close()
    
    # Create text summary
    with open(f"{output_dir}/scale_{scale}_comparison_{timestamp}.txt", 'w') as f:
        f.write(f"=== WMAP VS PLANCK: SCALE {scale} SPECIALIZATION COMPARISON ===\n\n")
        
        for constant in constants:
            f.write(f"=== {constant.upper()} ===\n")
            f.write(f"WMAP Score: {wmap_results[constant]['score']:.4f}\n")
            f.write(f"WMAP Z-score: {wmap_results[constant]['z_score']:.4f}\n")
            f.write(f"WMAP P-value: {wmap_results[constant]['p_value']:.6f}\n\n")
            
            f.write(f"Planck Score: {planck_results[constant]['score']:.4f}\n")
            f.write(f"Planck Z-score: {planck_results[constant]['z_score']:.4f}\n")
            f.write(f"Planck P-value: {planck_results[constant]['p_value']:.6f}\n\n")
            
            f.write(f"Z-score difference: {abs(wmap_results[constant]['z_score'] - planck_results[constant]['z_score']):.4f}\n\n")
    
    return f"{output_dir}/scale_{scale}_comparison_{timestamp}.txt"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze scale-specific specialization in CMB data")
    parser.add_argument('--wmap_data', type=str, default='data/wmap/wmap_spectrum_processed.pkl',
                       help='Path to WMAP data')
    parser.add_argument('--planck_data', type=str, default='data/planck/planck_tt_spectrum_2018.txt',
                       help='Path to Planck data')
    parser.add_argument('--scale', type=int, default=55,
                       help='Scale center to analyze')
    parser.add_argument('--n_simulations', type=int, default=1000,
                       help='Number of simulations')
    parser.add_argument('--output_dir', type=str, default='../results/scale_specific',
                       help='Output directory')
    parser.add_argument('--data_type', type=str, default='power_spectrum',
                       choices=['power_spectrum', 'map'],
                       help='Type of CMB data')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and process WMAP data
    logger.info(f"Loading WMAP data from {args.wmap_data}")
    wmap_data = load_cmb_data(args.wmap_data, args.data_type)
    wmap_data = preprocess_cmb_data(wmap_data, args.data_type)
    
    # Analyze WMAP data
    logger.info(f"Analyzing scale {args.scale} in WMAP data...")
    wmap_results = analyze_scale_specialization(
        wmap_data, args.scale, args.data_type, n_simulations=args.n_simulations
    )
    
    # Visualize WMAP results
    wmap_summary = visualize_scale_results(wmap_results, f"{output_dir}/wmap")
    logger.info(f"WMAP analysis complete. Results saved to {wmap_summary}")
    
    # Load and process Planck data
    logger.info(f"Loading Planck data from {args.planck_data}")
    planck_data = load_cmb_data(args.planck_data, args.data_type)
    planck_data = preprocess_cmb_data(planck_data, args.data_type)
    
    # Analyze Planck data
    logger.info(f"Analyzing scale {args.scale} in Planck data...")
    planck_results = analyze_scale_specialization(
        planck_data, args.scale, args.data_type, n_simulations=args.n_simulations
    )
    
    # Visualize Planck results
    planck_summary = visualize_scale_results(planck_results, f"{output_dir}/planck")
    logger.info(f"Planck analysis complete. Results saved to {planck_summary}")
    
    # Compare datasets
    logger.info("Comparing WMAP and Planck results...")
    comparison_summary = compare_datasets(
        wmap_results, planck_results, output_dir
    )
    logger.info(f"Comparison complete. Results saved to {comparison_summary}")

if __name__ == "__main__":
    main()
