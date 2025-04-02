#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fetch WMAP and Planck CMB maps and analyze them with quantum entanglement test

This script:
1. Downloads official WMAP and Planck CMB maps if not already downloaded
2. Runs the quantum entanglement signature test on both datasets
3. Compares the results between the two datasets
4. Saves detailed results and visualizations
"""

import os
import sys
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import requests
from astropy.io import fits
import time
from tqdm import tqdm
import logging

# Import our quantum entanglement test implementation
from quantum_entanglement_test import quantum_entanglement_signature_test, visualize_results

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map download URLs
WMAP_URL = "https://lambda.gsfc.nasa.gov/data/wmap/dr5/ilc/wmap_ilc_9yr_v5.fits"
PLANCK_URL = "https://pla.esac.esa.int/pla-sl/data-action?MAP.MAP_ID=COM_CMB_IQU-smica_2048_R3.00_full.fits"  # May require authentication

def download_file(url, target_path, description="file"):
    """Download a file with progress tracking"""
    if os.path.exists(target_path):
        logger.info(f"{description} already downloaded at {target_path}")
        return target_path
    
    logger.info(f"Downloading {description} from {url}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Start the download
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error for bad responses
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(target_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        progress_bar.close()
        
        logger.info(f"Successfully downloaded {description} to {target_path}")
        return target_path
    
    except Exception as e:
        logger.error(f"Error downloading {description}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)  # Remove partial download
        return None

def load_cmb_map(filepath, field=0, nest=False):
    """Load a CMB map from FITS file using healpy"""
    try:
        logger.info(f"Loading CMB map from {filepath}")
        cmb_map = hp.read_map(filepath, field=field, nest=nest)
        nside = hp.get_nside(cmb_map)
        npix = hp.nside2npix(nside)
        logger.info(f"Loaded map with NSIDE={nside}, {npix} pixels")
        return cmb_map
    except Exception as e:
        logger.error(f"Error loading CMB map: {e}")
        return None

def downgrade_map_if_needed(cmb_map, max_nside=512):
    """Downgrade high-resolution maps to save memory if needed"""
    current_nside = hp.get_nside(cmb_map)
    if current_nside > max_nside:
        logger.info(f"Downgrading map from NSIDE={current_nside} to NSIDE={max_nside}")
        return hp.ud_grade(cmb_map, max_nside)
    return cmb_map

def run_analysis(map_path, name, n_simulations=1000, output_dir=None):
    """Run quantum entanglement signature test on a CMB map"""
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"../../results/quantum_entanglement_{name.lower()}_{timestamp}"
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the CMB map
    cmb_map = load_cmb_map(map_path)
    if cmb_map is None:
        logger.error(f"Failed to load {name} map. Skipping analysis.")
        return None
    
    # Downgrade if necessary for memory efficiency
    cmb_map = downgrade_map_if_needed(cmb_map)
    
    # Create a simple visualization of the map
    hp.mollview(cmb_map, title=f"{name} CMB Map", unit="mK")
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_map.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Run the quantum entanglement test
    logger.info(f"Running quantum entanglement test on {name} data with {n_simulations} simulations...")
    start_time = time.time()
    results = quantum_entanglement_signature_test(cmb_map, n_simulations=n_simulations)
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    # Add metadata
    results['dataset'] = name
    results['execution_time'] = elapsed_time
    
    # Create visualization
    logger.info("Creating visualization...")
    fig = visualize_results(results)
    output_file = os.path.join(output_dir, f"{name.lower()}_quantum_entanglement_results.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"{name.lower()}_quantum_entanglement_results.txt")
    logger.info(f"Saving detailed results to {results_file}")
    
    with open(results_file, 'w') as f:
        f.write(f"QUANTUM ENTANGLEMENT SIGNATURE TEST RESULTS - {name}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Bell Inequality Analysis:\n")
        f.write(f"CMB Average Bell Value: {results['avg_bell_value']:.4f}\n")
        f.write(f"Surrogate Average Bell Value: {results['surrogate_mean']:.4f}\n")
        f.write(f"Ratio: {results['avg_bell_value']/results['surrogate_mean']:.2f}x\n")
        f.write(f"Z-score: {results['z_score']:.4f}\n")
        f.write(f"P-value: {results['p_value']:.8f}\n\n")
        
        f.write("Classical Limit Violations:\n")
        f.write(f"CMB Violation Rate: {results['classical_violation_rate']:.2%}\n")
        f.write(f"Surrogate Violation Rate: {results['violation_mean']:.2%}\n")
        f.write(f"Z-score: {results['violation_z']:.4f}\n")
        f.write(f"P-value: {results['violation_p']:.8f}\n\n")
        
        f.write("Non-Locality Metrics:\n")
        f.write(f"Mutual Information: {results['mutual_info']:.4f}\n")
        f.write(f"Surrogate MI: {results['surrogate_mi_mean']:.4f}\n")
        f.write(f"Z-score: {results['mi_z_score']:.4f}\n")
        f.write(f"P-value: {results['mi_p_value']:.8f}\n\n")
        
        f.write("Constant Comparison (Bell Values):\n")
        for name, value in results['constant_violations'].items():
            f.write(f"{name}: {value:.4f}\n")
        
        f.write(f"\nPhi-optimality: {results['phi_optimality']:.4f}\n\n")
        
        # Add overall interpretation
        if results['p_value'] < 0.01 and results['phi_optimality'] > 0.5:
            interpretation = "STRONG EVIDENCE for quantum-like entanglement"
        elif results['p_value'] < 0.05 and results['phi_optimality'] > 0.2:
            interpretation = "MODERATE EVIDENCE for quantum-like entanglement"
        elif results['p_value'] < 0.1:
            interpretation = "WEAK EVIDENCE for quantum-like entanglement"
        else:
            interpretation = "NO SIGNIFICANT EVIDENCE for quantum-like entanglement"
            
        f.write(f"Overall Interpretation: {interpretation}\n")
        f.write(f"\nTest completed in {results['execution_time']:.2f} seconds\n")
    
    logger.info(f"Results saved to {output_file} and {results_file}")
    return results

def compare_results(wmap_results, planck_results, output_dir):
    """Create a comparison of WMAP and Planck results"""
    if wmap_results is None or planck_results is None:
        logger.error("Cannot compare results as one or both datasets failed analysis")
        return
    
    logger.info("Creating comparison visualizations...")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Bell Values
    ax = axes[0, 0]
    datasets = ['WMAP', 'Planck']
    bell_values = [wmap_results['avg_bell_value'], planck_results['avg_bell_value']]
    surrogate_means = [wmap_results['surrogate_mean'], planck_results['surrogate_mean']]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, bell_values, width, label='CMB Data')
    rects2 = ax.bar(x + width/2, surrogate_means, width, label='Random Surrogates')
    
    ax.set_ylabel('Bell Value')
    ax.set_title('Bell Values Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.axhline(y=2, color='r', linestyle='--', label='Classical Limit')
    
    # Z-scores
    ax = axes[0, 1]
    z_scores = [wmap_results['z_score'], planck_results['z_score']]
    ax.bar(datasets, z_scores)
    ax.set_ylabel('Z-Score')
    ax.set_title('Statistical Significance')
    
    # P-values (log scale)
    ax = axes[0, 2]
    p_values = [wmap_results['p_value'], planck_results['p_value']]
    ax.bar(datasets, p_values)
    ax.set_ylabel('P-Value')
    ax.set_title('P-Values')
    ax.set_yscale('log')
    
    # Phi-optimality
    ax = axes[1, 0]
    phi_opts = [wmap_results['phi_optimality'], planck_results['phi_optimality']]
    ax.bar(datasets, phi_opts)
    ax.set_ylabel('Phi-Optimality')
    ax.set_title('Golden Ratio Preference')
    
    # Classical Limit Violations
    ax = axes[1, 1]
    viol_rates = [wmap_results['classical_violation_rate'], planck_results['classical_violation_rate']]
    surr_rates = [wmap_results['violation_mean'], planck_results['violation_mean']]
    
    x = np.arange(len(datasets))
    
    rects1 = ax.bar(x - width/2, viol_rates, width, label='CMB Data')
    rects2 = ax.bar(x + width/2, surr_rates, width, label='Random Surrogates')
    
    ax.set_ylabel('Violation Rate')
    ax.set_title('Classical Limit Violation Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    # Mutual Information
    ax = axes[1, 2]
    mi_values = [wmap_results['mutual_info'], planck_results['mutual_info']]
    surr_mi = [wmap_results['surrogate_mi_mean'], planck_results['surrogate_mi_mean']]
    
    x = np.arange(len(datasets))
    
    rects1 = ax.bar(x - width/2, mi_values, width, label='CMB Data')
    rects2 = ax.bar(x + width/2, surr_mi, width, label='Random Surrogates')
    
    ax.set_ylabel('Mutual Information')
    ax.set_title('Mutual Information Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    plt.tight_layout()
    comparison_file = os.path.join(output_dir, "wmap_planck_comparison.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Write comparison text file
    comparison_txt = os.path.join(output_dir, "wmap_planck_comparison.txt")
    with open(comparison_txt, 'w') as f:
        f.write("QUANTUM ENTANGLEMENT TEST - WMAP vs PLANCK COMPARISON\n")
        f.write("="*60 + "\n\n")
        
        f.write("Bell Values:\n")
        f.write(f"  WMAP: {wmap_results['avg_bell_value']:.4f} (surrogates: {wmap_results['surrogate_mean']:.4f})\n")
        f.write(f"  Planck: {planck_results['avg_bell_value']:.4f} (surrogates: {planck_results['surrogate_mean']:.4f})\n\n")
        
        f.write("Statistical Significance:\n")
        f.write(f"  WMAP: Z-score = {wmap_results['z_score']:.4f}, p-value = {wmap_results['p_value']:.8f}\n")
        f.write(f"  Planck: Z-score = {planck_results['z_score']:.4f}, p-value = {planck_results['p_value']:.8f}\n\n")
        
        f.write("Phi-Optimality (Golden Ratio Preference):\n")
        f.write(f"  WMAP: {wmap_results['phi_optimality']:.4f}\n")
        f.write(f"  Planck: {planck_results['phi_optimality']:.4f}\n\n")
        
        f.write("Classical Limit Violations:\n")
        f.write(f"  WMAP: {wmap_results['classical_violation_rate']:.2%} (surrogates: {wmap_results['violation_mean']:.2%})\n")
        f.write(f"  Planck: {planck_results['classical_violation_rate']:.2%} (surrogates: {planck_results['violation_mean']:.2%})\n\n")
        
        f.write("Mutual Information:\n")
        f.write(f"  WMAP: {wmap_results['mutual_info']:.4f} (surrogates: {wmap_results['surrogate_mi_mean']:.4f})\n")
        f.write(f"  Planck: {planck_results['mutual_info']:.4f} (surrogates: {planck_results['surrogate_mi_mean']:.4f})\n\n")
        
        # Add interpretations
        f.write("Overall Interpretations:\n")
        
        # WMAP interpretation
        if wmap_results['p_value'] < 0.01 and wmap_results['phi_optimality'] > 0.5:
            wmap_interpretation = "STRONG EVIDENCE for quantum-like entanglement"
        elif wmap_results['p_value'] < 0.05 and wmap_results['phi_optimality'] > 0.2:
            wmap_interpretation = "MODERATE EVIDENCE for quantum-like entanglement"
        elif wmap_results['p_value'] < 0.1:
            wmap_interpretation = "WEAK EVIDENCE for quantum-like entanglement"
        else:
            wmap_interpretation = "NO SIGNIFICANT EVIDENCE for quantum-like entanglement"
        
        # Planck interpretation
        if planck_results['p_value'] < 0.01 and planck_results['phi_optimality'] > 0.5:
            planck_interpretation = "STRONG EVIDENCE for quantum-like entanglement"
        elif planck_results['p_value'] < 0.05 and planck_results['phi_optimality'] > 0.2:
            planck_interpretation = "MODERATE EVIDENCE for quantum-like entanglement"
        elif planck_results['p_value'] < 0.1:
            planck_interpretation = "WEAK EVIDENCE for quantum-like entanglement"
        else:
            planck_interpretation = "NO SIGNIFICANT EVIDENCE for quantum-like entanglement"
            
        f.write(f"  WMAP: {wmap_interpretation}\n")
        f.write(f"  Planck: {planck_interpretation}\n\n")
        
        # Add consensus
        if "STRONG" in wmap_interpretation and "STRONG" in planck_interpretation:
            consensus = "STRONG CONSENSUS on quantum-like entanglement"
        elif ("STRONG" in wmap_interpretation or "MODERATE" in wmap_interpretation) and \
             ("STRONG" in planck_interpretation or "MODERATE" in planck_interpretation):
            consensus = "MODERATE CONSENSUS on quantum-like entanglement"
        elif "NO SIGNIFICANT" not in wmap_interpretation and "NO SIGNIFICANT" not in planck_interpretation:
            consensus = "WEAK CONSENSUS on quantum-like entanglement"
        else:
            consensus = "NO CONSENSUS on quantum-like entanglement"
            
        f.write(f"Cross-dataset Consensus: {consensus}\n")
    
    logger.info(f"Comparison saved to {comparison_file} and {comparison_txt}")

def main():
    """Main function to download data and run analysis"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Download and analyze CMB maps for quantum entanglement")
    parser.add_argument("--wmap-only", action="store_true", help="Only analyze WMAP data")
    parser.add_argument("--planck-only", action="store_true", help="Only analyze Planck data")
    parser.add_argument("-n", "--simulations", type=int, default=1000, 
                       help="Number of Monte Carlo simulations (default: 1000)")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--max-nside", type=int, default=512,
                       help="Maximum NSIDE parameter for analysis (downgrade if higher)")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"../../results/quantum_entanglement_{timestamp}"
    
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging file
    log_file = os.path.join(output_dir, "analysis_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting CMB Quantum Entanglement Analysis")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of simulations: {args.simulations}")
    
    # Data directories
    data_dir = os.path.abspath("../../data")
    wmap_dir = os.path.join(data_dir, "wmap")
    planck_dir = os.path.join(data_dir, "planck")
    
    os.makedirs(wmap_dir, exist_ok=True)
    os.makedirs(planck_dir, exist_ok=True)
    
    # File paths for downloaded data
    wmap_file = os.path.join(wmap_dir, "wmap_ilc_9yr_v5.fits")
    planck_file = os.path.join(planck_dir, "COM_CMB_IQU-smica_2048_R3.00_full.fits")
    
    # Run WMAP analysis
    wmap_results = None
    if not args.planck_only:
        # Download WMAP data if needed
        logger.info("Processing WMAP data...")
        if not os.path.exists(wmap_file):
            wmap_file = download_file(WMAP_URL, wmap_file, "WMAP ILC 9-year map")
        
        if wmap_file:
            wmap_output = os.path.join(output_dir, "wmap")
            os.makedirs(wmap_output, exist_ok=True)
            wmap_results = run_analysis(wmap_file, "WMAP", n_simulations=args.simulations, 
                                       output_dir=wmap_output)
    
    # Run Planck analysis
    planck_results = None
    if not args.wmap_only:
        # Download Planck data if needed
        logger.info("Processing Planck data...")
        if not os.path.exists(planck_file):
            planck_file = download_file(PLANCK_URL, planck_file, "Planck SMICA map")
        
        if planck_file:
            planck_output = os.path.join(output_dir, "planck")
            os.makedirs(planck_output, exist_ok=True)
            planck_results = run_analysis(planck_file, "Planck", n_simulations=args.simulations, 
                                         output_dir=planck_output)
    
    # Compare results
    if wmap_results and planck_results:
        compare_results(wmap_results, planck_results, output_dir)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)
