import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import test modules
from enhanced_phase_alignment_test import run_phase_alignment_test
from scripts.info_architecture.archive.information_architecture_test import run_information_architecture_test
# Import other test modules as needed

# Import cross-validation framework
from scripts.cross_validation.cross_validation_framework import cross_validate_findings

def run_all_tests_and_cross_validate(wmap_data=None, planck_data=None, 
                                    num_surrogates=1000, output_dir="../results/integrated_tests"):
    """
    Run all tests and perform cross-validation.
    
    Parameters:
    - wmap_data: WMAP dataset (optional, will load from file if None)
    - planck_data: Planck dataset (optional, will load from file if None)
    - num_surrogates: Number of surrogate datasets to generate for statistical testing
    - output_dir: Directory to save results
    
    Returns:
    - Dictionary with all test results and cross-validation
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data if not provided
    if wmap_data is None:
        print("Loading WMAP data...")
        wmap_data = load_wmap_data()
    
    if planck_data is None:
        print("Loading Planck data...")
        planck_data = load_planck_data()
    
    # Initialize results dictionary
    all_results = {}
    
    # Run Phase Alignment Test
    print("Running Phase Alignment Test...")
    phase_results = run_phase_alignment_test(wmap_data, planck_data, num_surrogates)
    all_results['phase_alignment'] = standardize_phase_alignment_results(phase_results)
    
    # Run Information Architecture Test
    print("Running Information Architecture Test...")
    arch_results = run_information_architecture_test(wmap_data, planck_data, num_surrogates)
    all_results['information_architecture'] = standardize_info_arch_results(arch_results)
    
    # Add all other tests...
    # For example:
    # transfer_results = run_transfer_entropy_test(wmap_data, planck_data, num_surrogates)
    # all_results['transfer_entropy'] = standardize_transfer_entropy_results(transfer_results)
    
    # Perform cross-validation
    print("Performing cross-validation...")
    cross_val_results = cross_validate_findings(all_results)
    all_results['cross_validation'] = cross_val_results
    
    # Generate final report
    print("Generating comprehensive report...")
    final_report = generate_comprehensive_report(all_results)
    all_results['final_report'] = final_report
    
    # Save results
    save_integrated_results(all_results, output_dir)
    
    return all_results

def load_wmap_data():
    """Load WMAP data from file."""
    # Implementation depends on your data format
    # This is a placeholder
    try:
        # Try to load from standard location
        data_path = "wmap_data/wmap_processed_data.npy"
        if os.path.exists(data_path):
            return np.load(data_path)
        else:
            # Create sample data if file doesn't exist
            print("WMAP data file not found. Creating sample data...")
            return create_sample_data("WMAP")
    except Exception as e:
        print("Error loading WMAP data: %s" % e)
        return create_sample_data("WMAP")

def load_planck_data():
    """Load Planck data from file."""
    # Implementation depends on your data format
    # This is a placeholder
    try:
        # Try to load from standard location
        data_path = "planck_data/planck_processed_data.npy"
        if os.path.exists(data_path):
            return np.load(data_path)
        else:
            # Create sample data if file doesn't exist
            print("Planck data file not found. Creating sample data...")
            return create_sample_data("Planck")
    except Exception as e:
        print("Error loading Planck data: %s" % e)
        return create_sample_data("Planck")

def create_sample_data(dataset_name):
    """Create sample data for testing."""
    # Create a sample dataset with appropriate structure
    # This is a placeholder
    if dataset_name == "WMAP":
        return {"multipoles": np.arange(2, 2001), 
                "power_spectrum": np.random.exponential(scale=1.0, size=1999),
                "phases": np.random.uniform(0, 2*np.pi, size=1999)}
    else:  # Planck
        return {"multipoles": np.arange(2, 2501), 
                "power_spectrum": np.random.exponential(scale=0.8, size=2499),
                "phases": np.random.uniform(0, 2*np.pi, size=2499)}

def standardize_phase_alignment_results(phase_results):
    """
    Standardize phase alignment results to a common format for cross-validation.
    
    Parameters:
    - phase_results: Raw results from the phase alignment test
    
    Returns:
    - Standardized results dictionary
    """
    standardized = {
        'WMAP': {},
        'Planck': {},
        'constant_performance': {
            'WMAP': {},
            'Planck': {}
        }
    }
    
    # Process results for each dataset and method
    for dataset in ['WMAP', 'Planck']:
        for method in ['Traditional', 'PLV']:
            # Extract results for this dataset and method
            key = "%s_%s" % (dataset, method)
            if key in phase_results:
                results = phase_results[key]
                
                # Process each constant
                for constant, data in results.get('constants', {}).items():
                    # Store key metrics
                    if constant not in standardized[dataset]:
                        standardized[dataset][constant] = {}
                    
                    standardized[dataset][constant]['%s_p_value' % method] = data.get('p_value', 1.0)
                    standardized[dataset][constant]['%s_z_score' % method] = data.get('z_score', 0.0)
                    standardized[dataset][constant]['%s_coherence' % method] = data.get('mean_coherence', 0.0)
                    standardized[dataset][constant]['%s_significant' % method] = data.get('significant', 0) == 1
                    
                    # Update constant performance based on coherence and significance
                    coherence = data.get('mean_coherence', 0.0)
                    p_value = data.get('p_value', 1.0)
                    
                    # Calculate performance score (higher for higher coherence and lower p-value)
                    performance = coherence * (1.0 - min(p_value, 0.99))
                    standardized['constant_performance'][dataset][constant] = performance
    
    return standardized

def standardize_info_arch_results(arch_results):
    """
    Standardize information architecture results to a common format for cross-validation.
    
    Parameters:
    - arch_results: Raw results from the information architecture test
    
    Returns:
    - Standardized results dictionary
    """
    standardized = {
        'WMAP': {},
        'Planck': {},
        'constant_performance': {
            'WMAP': {},
            'Planck': {}
        },
        'layer_specialization': {
            'WMAP': {},
            'Planck': {}
        }
    }
    
    # Process results for each dataset
    for dataset in ['WMAP', 'Planck']:
        if dataset in arch_results:
            results = arch_results[dataset]
            
            # Process each constant
            for constant, data in results.get('constants', {}).items():
                # Store key metrics
                standardized[dataset][constant] = {
                    'score': data.get('score', 0.0),
                    'p_value': data.get('p_value', 1.0),
                    'significant': data.get('p_value', 1.0) < 0.05
                }
                
                # Update constant performance based on score and significance
                score = data.get('score', 0.0)
                p_value = data.get('p_value', 1.0)
                
                # Calculate performance score (higher for higher score and lower p-value)
                performance = score * (1.0 - min(p_value, 0.99))
                standardized['constant_performance'][dataset][constant] = performance
            
            # Process layer specialization if available
            if 'layer_specialization' in results:
                standardized['layer_specialization'][dataset] = results['layer_specialization']
    
    return standardized

def generate_comprehensive_report(all_results):
    """
    Generate a comprehensive report of all test results and cross-validation.
    
    Parameters:
    - all_results: Dictionary with all test results and cross-validation
    
    Returns:
    - Report dictionary
    """
    report = {
        'summary': {},
        'test_results': {},
        'cross_validation': {},
        'recommendations': []
    }
    
    # Summarize test results
    for test_name, results in all_results.items():
        if test_name != 'cross_validation' and test_name != 'final_report':
            report['test_results'][test_name] = summarize_test_results(test_name, results)
    
    # Include cross-validation summary
    if 'cross_validation' in all_results and 'summary' in all_results['cross_validation']:
        report['cross_validation'] = all_results['cross_validation']['summary']
    
    # Generate overall summary
    report['summary'] = generate_overall_summary(report)
    
    # Generate recommendations
    report['recommendations'] = generate_recommendations(all_results)
    
    return report

def summarize_test_results(test_name, results):
    """Summarize results for a specific test."""
    summary = {
        'significant_constants': {
            'WMAP': [],
            'Planck': []
        },
        'consistency': {}
    }
    
    # Identify significant constants for each dataset
    for dataset in ['WMAP', 'Planck']:
        if dataset in results:
            for constant, data in results[dataset].items():
                # Check if any metric indicates significance
                is_significant = False
                
                for key, value in data.items():
                    if 'significant' in key.lower() and value:
                        is_significant = True
                    elif 'p_value' in key.lower() and isinstance(value, (int, float)) and value < 0.05:
                        is_significant = True
                
                if is_significant:
                    summary['significant_constants'][dataset].append(constant)
    
    # Check cross-dataset consistency
    common_significant = set(summary['significant_constants']['WMAP']) & set(summary['significant_constants']['Planck'])
    summary['consistency']['common_significant_constants'] = list(common_significant)
    summary['consistency']['consistency_ratio'] = len(common_significant) / max(
        len(summary['significant_constants']['WMAP'] + summary['significant_constants']['Planck']), 1)
    
    return summary

def generate_overall_summary(report):
    """Generate overall summary from all test results and cross-validation."""
    summary = {
        'most_significant_constants': {},
        'cross_dataset_consistency': 0.0,
        'test_independence': 0.0
    }
    
    # Count significant findings for each constant
    constant_counts = {}
    
    for test_name, test_summary in report['test_results'].items():
        for dataset, constants in test_summary['significant_constants'].items():
            for constant in constants:
                if constant not in constant_counts:
                    constant_counts[constant] = 0
                constant_counts[constant] += 1
    
    # Identify most significant constants
    if constant_counts:
        max_count = max(constant_counts.values())
        most_significant = [c for c, count in constant_counts.items() if count == max_count]
        summary['most_significant_constants'] = {
            'constants': most_significant,
            'test_count': max_count
        }
    
    # Calculate cross-dataset consistency
    consistency_ratios = [test_summary['consistency']['consistency_ratio'] 
                         for test_summary in report['test_results'].values()]
    if consistency_ratios:
        summary['cross_dataset_consistency'] = sum(consistency_ratios) / len(consistency_ratios)
    
    # Include test independence from cross-validation
    if 'test_independence' in report['cross_validation']:
        test_independence = report['cross_validation']['test_independence']
        if 'overall_independence' in test_independence:
            summary['test_independence'] = 1.0 - test_independence['overall_independence']
    
    return summary

def generate_recommendations(all_results):
    """Generate recommendations based on all results."""
    recommendations = []
    
    # Check if cross-validation was performed
    if 'cross_validation' not in all_results:
        recommendations.append({
            'priority': 'high',
            'recommendation': 'Perform cross-validation to validate findings across tests and datasets.'
        })
        return recommendations
    
    cross_val = all_results['cross_validation']
    
    # Recommend focusing on high-confidence constants
    if 'multi_dimensional_validation' in cross_val['summary']:
        multi_val = cross_val['summary']['multi_dimensional_validation']
        if 'high_confidence_constants' in multi_val and multi_val['high_confidence_constants']:
            recommendations.append({
                'priority': 'high',
                'recommendation': "Focus further research on high-confidence constants: %s." % ', '.join(multi_val['high_confidence_constants'])
            })
    
    # Recommend additional tests if test independence is low
    if 'test_independence' in cross_val['summary']:
        test_independence = cross_val['summary']['test_independence']
        if 'overall_independence' in test_independence and test_independence['overall_independence'] > 0.7:
            recommendations.append({
                'priority': 'medium',
                'recommendation': "Develop additional independent test methods to provide more diverse validation approaches."
            })
    
    # Recommend investigating scale dependence
    if 'scale_dependence' in cross_val['summary']:
        scale_dependence = cross_val['summary']['scale_dependence']
        if 'scale_dependent_tests' in scale_dependence and scale_dependence['scale_dependent_tests']:
            recommendations.append({
                'priority': 'medium',
                'recommendation': "Investigate scale-dependent behavior in tests: %s." % ', '.join(scale_dependence['scale_dependent_tests'])
            })
    
    # Always recommend increasing sample size
    recommendations.append({
        'priority': 'low',
        'recommendation': "Increase the number of surrogate datasets to improve statistical power."
    })
    
    return recommendations

def save_integrated_results(all_results, output_dir):
    """Save integrated results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a serializable version of the results
    serializable_results = {}
    
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj)
        elif obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)
    
    # Make results serializable
    for key, value in all_results.items():
        if key != 'cross_validation' or (key == 'cross_validation' and 'visualizations' not in value):
            serializable_results[key] = make_serializable(value)
    
    # Save full results
    results_file = os.path.join(output_dir, "integrated_results_%s.json" % timestamp)
    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save report separately
    if 'final_report' in all_results:
        report_file = os.path.join(output_dir, "comprehensive_report_%s.json" % timestamp)
        with open(report_file, "w") as f:
            json.dump(make_serializable(all_results['final_report']), f, indent=2)
    
    # Save visualizations
    if 'cross_validation' in all_results and 'visualizations' in all_results['cross_validation']:
        for name, plt_obj in all_results['cross_validation']['visualizations'].items():
            viz_file = os.path.join(output_dir, "%s_%s.png" % (name, timestamp))
            plt_obj.savefig(viz_file)
            plt.close(plt_obj.figure)
    
    print("Results saved to %s" % output_dir)
    print("Full results: %s" % results_file)
    if 'final_report' in all_results:
        print("Comprehensive report: %s" % report_file)

if __name__ == "__main__":
    # Run all tests with 10,000 surrogates as requested
    print("Running all tests with 10,000 surrogate simulations...")
    run_all_tests_and_cross_validate(num_surrogates=10000)
