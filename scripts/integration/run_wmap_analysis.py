#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WMAP Cosmic Consciousness Analysis

This script runs the Cosmic Consciousness Analysis framework on WMAP CMB data.
It can also compare results between WMAP and Planck data.
"""

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime

# Add parent directory to path for importing from Cosmic_Consciousness_Analysis
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cosmic_analysis_path = os.path.join(parent_dir, 'Cosmic_Consciousness_Analysis')
sys.path.append(cosmic_analysis_path)

# Import WMAP data handler
from scripts.common.data_handlers.wmap_data_handler import get_wmap_data, compare_wmap_planck

# Import from Cosmic Consciousness Analysis framework
try:
    from analysis.analysis import Analyzer, CosmicConsciousnessAnalyzer
    from core_framework.data_handler import get_simulated_data
    from planck_data.planck_data_handler import get_planck_data
    HAS_COSMIC_FRAMEWORK = True
except ImportError:
    print("Warning: Could not import Cosmic Consciousness Analysis framework.")
    print("Make sure the framework is installed at: {}".format(cosmic_analysis_path))
    HAS_COSMIC_FRAMEWORK = False


def run_analysis(data, output_dir, args):
    """
    Run the Cosmic Consciousness Analysis on the provided data.
    
    Args:
        data (numpy.ndarray): Data to analyze
        output_dir (str): Directory to save results
        args (argparse.Namespace): Command-line arguments
        
    Returns:
        dict: Analysis results
    """
    if not HAS_COSMIC_FRAMEWORK:
        print("Error: Cosmic Consciousness Analysis framework not found.")
        return None
    
    # Create analyzer
    analyzer = CosmicConsciousnessAnalyzer(
        data=data,
        output_dir=output_dir,
        phi_bias=args.phi_bias
    )
    
    # Determine which tests to run
    tests_to_run = []
    if args.golden_ratio:
        tests_to_run.append('golden_ratio')
    if args.coherence_analysis:
        tests_to_run.append('coherence_analysis')
    if args.gr_specific_coherence:
        tests_to_run.append('gr_specific_coherence')
    if args.hierarchical_organization:
        tests_to_run.append('hierarchical_organization')
    if args.information_integration:
        tests_to_run.append('information_integration')
    if args.scale_transition:
        tests_to_run.append('scale_transition')
    if args.resonance_analysis:
        tests_to_run.append('resonance_analysis')
    if args.fractal_analysis:
        tests_to_run.append('fractal_analysis')
    if args.meta_coherence:
        tests_to_run.append('meta_coherence')
    if args.transfer_entropy:
        tests_to_run.append('transfer_entropy')
    if args.all or not tests_to_run:
        tests_to_run = 'all'
    
    # Run analysis
    print("\nRunning analysis on {} data...".format(args.data_source))
    results = analyzer.run_analysis(
        tests=tests_to_run,
        visualize=not args.no_visualize,
        report=not args.no_report,
        parallel=not args.no_parallel,
        n_jobs=args.n_jobs
    )
    
    return results


def compare_datasets(wmap_results, planck_results, output_dir):
    """
    Compare results between WMAP and Planck datasets.
    
    Args:
        wmap_results (dict): Results from WMAP analysis
        planck_results (dict): Results from Planck analysis
        output_dir (str): Directory to save comparison results
        
    Returns:
        dict: Comparison metrics
    """
    print("\nComparing WMAP and Planck results...")
    
    # Extract phi optimality scores for each test
    comparison = {
        'phi_optimality': {},
        'significance': {},
        'correlation': {}
    }
    
    # Compare phi optimality for each test
    for test_name in wmap_results['phi_optimality']:
        if test_name in planck_results['phi_optimality']:
            wmap_phi = wmap_results['phi_optimality'][test_name]
            planck_phi = planck_results['phi_optimality'][test_name]
            
            # Handle different data types
            if isinstance(wmap_phi, dict) and isinstance(planck_phi, dict):
                # For tests that return dictionaries of phi values
                comparison['phi_optimality'][test_name] = {
                    'wmap': wmap_phi,
                    'planck': planck_phi,
                    'ratio': {k: wmap_phi[k]/planck_phi[k] if planck_phi[k] != 0 else float('inf') 
                             for k in wmap_phi if k in planck_phi}
                }
            else:
                # For tests that return single phi values
                comparison['phi_optimality'][test_name] = {
                    'wmap': wmap_phi,
                    'planck': planck_phi,
                    'ratio': wmap_phi/planck_phi if planck_phi != 0 else float('inf')
                }
    
    # Compare significance for each test
    for test_name in wmap_results['significance']:
        if test_name in planck_results['significance']:
            wmap_sig = wmap_results['significance'][test_name]
            planck_sig = planck_results['significance'][test_name]
            
            comparison['significance'][test_name] = {
                'wmap': wmap_sig,
                'planck': planck_sig,
                'ratio': wmap_sig/planck_sig if planck_sig != 0 else float('inf')
            }
    
    # Calculate overall correlation of phi optimality scores
    wmap_phi_values = []
    planck_phi_values = []
    
    for test_name in comparison['phi_optimality']:
        wmap_val = comparison['phi_optimality'][test_name]['wmap']
        planck_val = comparison['phi_optimality'][test_name]['planck']
        
        if not isinstance(wmap_val, dict):
            wmap_phi_values.append(wmap_val)
            planck_phi_values.append(planck_val)
    
    if wmap_phi_values and planck_phi_values:
        correlation = np.corrcoef(wmap_phi_values, planck_phi_values)[0, 1]
        comparison['correlation']['overall'] = correlation
    
    # Save comparison results
    comparison_file = os.path.join(output_dir, 'wmap_planck_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2, sort_keys=True, default=str)
    
    print("Comparison results saved to: {}".format(comparison_file))
    
    # Generate comparison visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up the figure
        plt.figure(figsize=(12, 8))
        
        # Plot phi optimality comparison
        test_names = []
        wmap_phis = []
        planck_phis = []
        
        for test_name in comparison['phi_optimality']:
            wmap_val = comparison['phi_optimality'][test_name]['wmap']
            planck_val = comparison['phi_optimality'][test_name]['planck']
            
            if not isinstance(wmap_val, dict):
                test_names.append(test_name)
                wmap_phis.append(wmap_val)
                planck_phis.append(planck_val)
        
        # Create bar chart
        x = np.arange(len(test_names))
        width = 0.35
        
        plt.bar(x - width/2, wmap_phis, width, label='WMAP')
        plt.bar(x + width/2, planck_phis, width, label='Planck')
        
        plt.xlabel('Test')
        plt.ylabel('Phi Optimality Score')
        plt.title('Comparison of Phi Optimality Scores: WMAP vs Planck')
        plt.xticks(x, test_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        comparison_plot = os.path.join(output_dir, 'wmap_planck_comparison.png')
        plt.savefig(comparison_plot, dpi=150)
        print("Comparison plot saved to: {}".format(comparison_plot))
        
    except ImportError:
        print("Matplotlib or seaborn not available for visualization.")
    
    return comparison


def main():
    """Main function to run WMAP analysis."""
    parser = argparse.ArgumentParser(description='WMAP Cosmic Consciousness Analysis')
    
    # Data source options
    parser.add_argument('--data-source', choices=['wmap', 'planck', 'simulated', 'both'], 
                        default='wmap', help='Data source to use')
    parser.add_argument('--data-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--wmap-file', type=str, default=None,
                        help='Path to WMAP data file (if not using default)')
    parser.add_argument('--planck-file', type=str, default=None,
                        help='Path to Planck data file (if not using default)')
    parser.add_argument('--use-power-spectrum', action='store_true',
                        help='Use power spectrum instead of map')
    parser.add_argument('--data-size', type=int, default=4096,
                        help='Size of data to analyze')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Preprocessing options
    parser.add_argument('--smooth', action='store_true',
                        help='Apply smoothing to the data')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Window size for smoothing')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize the data')
    parser.add_argument('--detrend', action='store_true',
                        help='Remove linear trend from data')
    
    # Analysis options
    parser.add_argument('--phi-bias', type=float, default=0.1,
                        help='Bias factor for golden ratio tests')
    parser.add_argument('--golden-ratio', action='store_true',
                        help='Run golden ratio test')
    parser.add_argument('--coherence-analysis', action='store_true',
                        help='Run coherence analysis test')
    parser.add_argument('--gr-specific-coherence', action='store_true',
                        help='Run GR-specific coherence test')
    parser.add_argument('--hierarchical-organization', action='store_true',
                        help='Run hierarchical organization test')
    parser.add_argument('--information-integration', action='store_true',
                        help='Run information integration test')
    parser.add_argument('--scale-transition', action='store_true',
                        help='Run scale transition test')
    parser.add_argument('--resonance-analysis', action='store_true',
                        help='Run resonance analysis test')
    parser.add_argument('--fractal-analysis', action='store_true',
                        help='Run fractal analysis test')
    parser.add_argument('--meta-coherence', action='store_true',
                        help='Run meta-coherence test')
    parser.add_argument('--transfer-entropy', action='store_true',
                        help='Run transfer entropy test')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests')
    
    # Output options
    parser.add_argument('--visualize', dest='no_visualize', action='store_false',
                        help='Generate visualizations')
    parser.add_argument('--no-visualize', dest='no_visualize', action='store_true',
                        help='Do not generate visualizations')
    parser.add_argument('--report', dest='no_report', action='store_false',
                        help='Generate detailed reports')
    parser.add_argument('--no-report', dest='no_report', action='store_true',
                        help='Do not generate reports')
    
    # Parallel processing options
    parser.add_argument('--parallel', dest='no_parallel', action='store_false',
                        help='Use parallel processing')
    parser.add_argument('--no-parallel', dest='no_parallel', action='store_true',
                        help='Do not use parallel processing')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs')
    
    # Comparison options
    parser.add_argument('--compare', action='store_true',
                        help='Compare WMAP and Planck results')
    
    # Set defaults
    parser.set_defaults(no_visualize=False, no_report=False, no_parallel=False)
    
    args = parser.parse_args()
    
    # Check if Cosmic Consciousness Analysis framework is available
    if not HAS_COSMIC_FRAMEWORK:
        print("Error: Cosmic Consciousness Analysis framework not found.")
        print("Please make sure it's installed at: {}".format(cosmic_analysis_path))
        return 1
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.data_dir, f"wmap_analysis_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save command-line arguments
    with open(os.path.join(output_dir, 'arguments.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Get data based on source
    wmap_data = None
    planck_data = None
    
    if args.data_source in ['wmap', 'both']:
        print("Loading WMAP data...")
        wmap_data = get_wmap_data(
            use_power_spectrum=args.use_power_spectrum,
            n_samples=args.data_size,
            smooth=args.smooth,
            smooth_window=args.smooth_window,
            normalize=args.normalize,
            detrend=args.detrend
        )
        print("WMAP data loaded: {} samples".format(len(wmap_data)))
    
    if args.data_source in ['planck', 'both']:
        print("Loading Planck data...")
        planck_data = get_planck_data(
            data_file=args.planck_file,
            smooth=args.smooth,
            smooth_window=args.smooth_window,
            normalize=args.normalize,
            detrend=args.detrend
        )
        print("Planck data loaded: {} samples".format(len(planck_data)))
    
    if args.data_source == 'simulated':
        print("Generating simulated data...")
        simulated_data = get_simulated_data(args.data_size)
        print("Simulated data generated: {} samples".format(len(simulated_data)))
        wmap_data = simulated_data  # Use simulated data as WMAP data
    
    # Run analysis
    wmap_results = None
    planck_results = None
    
    if wmap_data is not None:
        wmap_output_dir = os.path.join(output_dir, 'wmap')
        if not os.path.exists(wmap_output_dir):
            os.makedirs(wmap_output_dir)
        
        args.data_source = 'WMAP'
        wmap_results = run_analysis(wmap_data, wmap_output_dir, args)
    
    if planck_data is not None:
        planck_output_dir = os.path.join(output_dir, 'planck')
        if not os.path.exists(planck_output_dir):
            os.makedirs(planck_output_dir)
        
        args.data_source = 'Planck'
        planck_results = run_analysis(planck_data, planck_output_dir, args)
    
    # Compare results if both datasets were analyzed
    if args.compare and wmap_results and planck_results:
        comparison = compare_datasets(wmap_results, planck_results, output_dir)
        
        # Also compare raw data
        data_comparison = compare_wmap_planck(wmap_data, planck_data)
        
        # Save data comparison
        data_comparison_file = os.path.join(output_dir, 'wmap_planck_data_comparison.json')
        with open(data_comparison_file, 'w') as f:
            json.dump(data_comparison, f, indent=2, sort_keys=True, default=str)
        
        print("\nData comparison results:")
        print("  Correlation: {:.4f}".format(data_comparison['correlation']))
        print("  Power ratio (WMAP/Planck): {:.4f}".format(data_comparison['power_ratio']))
    
    print("\nAnalysis complete. Results saved to: {}".format(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
