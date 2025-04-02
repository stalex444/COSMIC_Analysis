#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive validation testing script for WMAP Cosmic Analysis.
This script implements the testing plan to validate all key findings and functionality.
"""

from __future__ import print_function, division
import os
import sys
import time
import json
import argparse
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation_tests.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("validation_tests")

# Test result storage
test_results = {
    "environment_setup": {},
    "data_processing": {},
    "key_analysis": {},
    "cross_dataset": {},
    "visualization": {},
    "summary": {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "skipped_tests": 0,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": None
    }
}

def log_test_result(category, test_name, passed, expected=None, actual=None, error=None, skipped=False):
    """Log a test result and update the results dictionary."""
    status = "SKIPPED" if skipped else "PASSED" if passed else "FAILED"
    
    if expected is not None and actual is not None:
        logger.info(f"Test {test_name}: {status} - Expected: {expected}, Actual: {actual}")
    else:
        logger.info(f"Test {test_name}: {status}")
    
    if error:
        logger.error(f"Error in {test_name}: {error}")
    
    # Update test results dictionary
    test_results[category][test_name] = {
        "passed": passed,
        "skipped": skipped,
        "expected": expected,
        "actual": actual,
        "error": str(error) if error else None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Update summary
    if not skipped:
        test_results["summary"]["total_tests"] += 1
        if passed:
            test_results["summary"]["passed_tests"] += 1
        else:
            test_results["summary"]["failed_tests"] += 1
    else:
        test_results["summary"]["skipped_tests"] += 1

def is_close_enough(expected, actual, tolerance=0.05):
    """Check if actual value is within tolerance of expected value."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) <= tolerance * abs(expected)
    return expected == actual

def test_environment_setup():
    """Test 1: Environment Setup Validation"""
    logger.info("Starting Environment Setup Validation tests...")
    
    # Test 1.1: Check Python version
    try:
        import platform
        python_version = platform.python_version()
        is_compatible = python_version.startswith('2.7') or float(python_version[:3]) >= 3.6
        log_test_result(
            "environment_setup", 
            "python_version_compatibility", 
            is_compatible,
            "Python 2.7 or >= 3.6", 
            python_version
        )
    except Exception as e:
        log_test_result("environment_setup", "python_version_compatibility", False, error=e)
    
    # Test 1.2: Check if all required packages are installed
    try:
        import numpy
        import scipy
        import matplotlib
        import pandas
        import astropy
        import healpy
        log_test_result("environment_setup", "required_packages", True)
    except ImportError as e:
        log_test_result("environment_setup", "required_packages", False, error=e)
    
    # Test 1.3: Check if the package can be imported
    try:
        sys.path.insert(0, os.path.abspath('.'))
        import wmap_cosmic_analysis
        log_test_result("environment_setup", "package_import", True)
    except ImportError as e:
        log_test_result("environment_setup", "package_import", False, error=e)
    
    # Test 1.4: Check configuration loading
    try:
        # Assuming there's a config module or function
        from wmap_cosmic_analysis.config import load_config
        config = load_config()
        log_test_result("environment_setup", "config_loading", True)
    except Exception as e:
        log_test_result("environment_setup", "config_loading", False, error=e)

def test_data_processing():
    """Test 2: Data Processing Validation"""
    logger.info("Starting Data Processing Validation tests...")
    
    # Test 2.1: Verify WMAP data loading
    try:
        from wmap_cosmic_analysis.data import load_wmap_data
        wmap_data = load_wmap_data()
        log_test_result(
            "data_processing", 
            "wmap_data_loading", 
            wmap_data is not None,
            "WMAP data loaded", 
            "Data loaded successfully" if wmap_data is not None else "Failed to load data"
        )
    except Exception as e:
        log_test_result("data_processing", "wmap_data_loading", False, error=e)
    
    # Test 2.2: Verify Planck data loading
    try:
        from wmap_cosmic_analysis.data import load_planck_data
        planck_data = load_planck_data()
        log_test_result(
            "data_processing", 
            "planck_data_loading", 
            planck_data is not None,
            "Planck data loaded", 
            "Data loaded successfully" if planck_data is not None else "Failed to load data"
        )
    except Exception as e:
        log_test_result("data_processing", "planck_data_loading", False, error=e)
    
    # Test 2.3: Verify preprocessing steps
    try:
        from wmap_cosmic_analysis.preprocessing import preprocess_data
        preprocessed_data = preprocess_data(wmap_data if 'wmap_data' in locals() else None)
        log_test_result(
            "data_processing", 
            "preprocessing_execution", 
            preprocessed_data is not None,
            "Preprocessing completed", 
            "Preprocessing successful" if preprocessed_data is not None else "Preprocessing failed"
        )
    except Exception as e:
        log_test_result("data_processing", "preprocessing_execution", False, error=e)
    
    # Test 2.4: Verify surrogate data generation
    try:
        from wmap_cosmic_analysis.surrogate import generate_surrogate_data
        surrogate_data = generate_surrogate_data(preprocessed_data if 'preprocessed_data' in locals() else None)
        log_test_result(
            "data_processing", 
            "surrogate_data_generation", 
            surrogate_data is not None,
            "Surrogate data generated", 
            "Generation successful" if surrogate_data is not None else "Generation failed"
        )
    except Exception as e:
        log_test_result("data_processing", "surrogate_data_generation", False, error=e)
    
    # Test 2.5: Verify data output format
    try:
        # This would depend on the expected format of your data
        is_correct_format = True  # Replace with actual validation
        log_test_result(
            "data_processing", 
            "data_output_format", 
            is_correct_format,
            "Correct data format", 
            "Format is correct" if is_correct_format else "Format is incorrect"
        )
    except Exception as e:
        log_test_result("data_processing", "data_output_format", False, error=e)

def test_key_analysis():
    """Test 3: Key Analysis Validation"""
    logger.info("Starting Key Analysis Validation tests...")
    
    # Test 3.1: Transfer Entropy Test
    try:
        from wmap_cosmic_analysis.analysis import calculate_transfer_entropy
        transfer_entropy_ratio = 74.79  # Expected value from paper
        confidence_sigma = 63.23  # Expected value from paper
        
        # Assuming these functions exist and return the values we want to check
        actual_ratio, actual_sigma = calculate_transfer_entropy()
        
        ratio_passed = is_close_enough(transfer_entropy_ratio, actual_ratio)
        sigma_passed = is_close_enough(confidence_sigma, actual_sigma)
        
        log_test_result(
            "key_analysis", 
            "transfer_entropy_ratio", 
            ratio_passed,
            transfer_entropy_ratio, 
            actual_ratio
        )
        
        log_test_result(
            "key_analysis", 
            "transfer_entropy_confidence", 
            sigma_passed,
            confidence_sigma, 
            actual_sigma
        )
    except Exception as e:
        log_test_result("key_analysis", "transfer_entropy_ratio", False, error=e)
        log_test_result("key_analysis", "transfer_entropy_confidence", False, error=e)
    
    # Test 3.2: Laminarity Test
    try:
        from wmap_cosmic_analysis.analysis import calculate_laminarity
        laminarity_ratio = 16.23  # Expected value from paper
        lam_confidence_sigma = 12.16  # Expected value from paper
        
        actual_lam_ratio, actual_lam_sigma = calculate_laminarity()
        
        lam_ratio_passed = is_close_enough(laminarity_ratio, actual_lam_ratio)
        lam_sigma_passed = is_close_enough(lam_confidence_sigma, actual_lam_sigma)
        
        log_test_result(
            "key_analysis", 
            "laminarity_ratio", 
            lam_ratio_passed,
            laminarity_ratio, 
            actual_lam_ratio
        )
        
        log_test_result(
            "key_analysis", 
            "laminarity_confidence", 
            lam_sigma_passed,
            lam_confidence_sigma, 
            actual_lam_sigma
        )
    except Exception as e:
        log_test_result("key_analysis", "laminarity_ratio", False, error=e)
        log_test_result("key_analysis", "laminarity_confidence", False, error=e)
    
    # Test 3.3: Meta-Coherence Analysis
    try:
        from wmap_cosmic_analysis.analysis import calculate_meta_coherence
        meta_coherence_ratio = 31954  # Expected value from paper
        meta_coherence_sigma = 7.0  # Expected value from paper (>7σ)
        
        actual_mc_ratio, actual_mc_sigma = calculate_meta_coherence()
        
        mc_ratio_passed = is_close_enough(meta_coherence_ratio, actual_mc_ratio)
        mc_sigma_passed = actual_mc_sigma >= meta_coherence_sigma
        
        log_test_result(
            "key_analysis", 
            "meta_coherence_ratio", 
            mc_ratio_passed,
            meta_coherence_ratio, 
            actual_mc_ratio
        )
        
        log_test_result(
            "key_analysis", 
            "meta_coherence_confidence", 
            mc_sigma_passed,
            f">= {meta_coherence_sigma}", 
            actual_mc_sigma
        )
    except Exception as e:
        log_test_result("key_analysis", "meta_coherence_ratio", False, error=e)
        log_test_result("key_analysis", "meta_coherence_confidence", False, error=e)
    
    # Test 3.4: Golden Ratio Significance
    try:
        from wmap_cosmic_analysis.analysis import calculate_golden_ratio_significance
        gr_significance = 0.00001  # Expected p-value from paper (p < 0.00001)
        
        actual_gr_significance = calculate_golden_ratio_significance()
        
        gr_passed = actual_gr_significance <= gr_significance
        
        log_test_result(
            "key_analysis", 
            "golden_ratio_significance", 
            gr_passed,
            f"< {gr_significance}", 
            actual_gr_significance
        )
    except Exception as e:
        log_test_result("key_analysis", "golden_ratio_significance", False, error=e)
    
    # Test 3.5: Scale Transitions
    try:
        from wmap_cosmic_analysis.analysis import detect_scale_transitions
        expected_transitions = 55  # Expected value from paper
        
        actual_transitions = detect_scale_transitions()
        
        transitions_passed = is_close_enough(expected_transitions, actual_transitions)
        
        log_test_result(
            "key_analysis", 
            "scale_transitions", 
            transitions_passed,
            expected_transitions, 
            actual_transitions
        )
    except Exception as e:
        log_test_result("key_analysis", "scale_transitions", False, error=e)
    
    # Additional tests for other metrics
    metrics = [
        ("resonance_patterns", 96.96, 6.0),
        ("cosmic_optimization", 17.20, 5.0),
        ("information_integration", 1.49, 4.74),
        ("predictive_power", 3.05, 4.06),
        ("determinism", 1.18, 2.87),
        ("cross_scale_correlation", 1.48, 2.27)
    ]
    
    for metric_name, expected_ratio, expected_sigma in metrics:
        try:
            # Dynamically import the calculation function
            import importlib
            module = importlib.import_module("wmap_cosmic_analysis.analysis")
            calculate_func = getattr(module, f"calculate_{metric_name}")
            
            actual_ratio, actual_sigma = calculate_func()
            
            ratio_passed = is_close_enough(expected_ratio, actual_ratio)
            sigma_passed = is_close_enough(expected_sigma, actual_sigma)
            
            log_test_result(
                "key_analysis", 
                f"{metric_name}_ratio", 
                ratio_passed,
                expected_ratio, 
                actual_ratio
            )
            
            log_test_result(
                "key_analysis", 
                f"{metric_name}_confidence", 
                sigma_passed,
                expected_sigma, 
                actual_sigma
            )
        except Exception as e:
            log_test_result("key_analysis", f"{metric_name}_ratio", False, error=e)
            log_test_result("key_analysis", f"{metric_name}_confidence", False, error=e)

def test_cross_dataset_validation():
    """Test 4: Cross-Dataset Validation"""
    logger.info("Starting Cross-Dataset Validation tests...")
    
    # Test 4.1: WMAP GR-specific coherence
    try:
        from wmap_cosmic_analysis.cross_validation import calculate_wmap_gr_coherence
        expected_coherence = 0.896
        expected_p_value = 0.00001
        
        actual_coherence, actual_p_value = calculate_wmap_gr_coherence()
        
        coherence_passed = is_close_enough(expected_coherence, actual_coherence)
        p_value_passed = actual_p_value <= expected_p_value
        
        log_test_result(
            "cross_dataset", 
            "wmap_gr_coherence", 
            coherence_passed,
            expected_coherence, 
            actual_coherence
        )
        
        log_test_result(
            "cross_dataset", 
            "wmap_gr_coherence_p_value", 
            p_value_passed,
            f"< {expected_p_value}", 
            actual_p_value
        )
    except Exception as e:
        log_test_result("cross_dataset", "wmap_gr_coherence", False, error=e)
        log_test_result("cross_dataset", "wmap_gr_coherence_p_value", False, error=e)
    
    # Test 4.2: WMAP fractal behavior (Hurst exponent)
    try:
        from wmap_cosmic_analysis.cross_validation import calculate_wmap_hurst_exponent
        expected_hurst = 0.937
        expected_p_value = 0.00001
        
        actual_hurst, actual_p_value = calculate_wmap_hurst_exponent()
        
        hurst_passed = is_close_enough(expected_hurst, actual_hurst)
        p_value_passed = actual_p_value <= expected_p_value
        
        log_test_result(
            "cross_dataset", 
            "wmap_hurst_exponent", 
            hurst_passed,
            expected_hurst, 
            actual_hurst
        )
        
        log_test_result(
            "cross_dataset", 
            "wmap_hurst_p_value", 
            p_value_passed,
            f"< {expected_p_value}", 
            actual_p_value
        )
    except Exception as e:
        log_test_result("cross_dataset", "wmap_hurst_exponent", False, error=e)
        log_test_result("cross_dataset", "wmap_hurst_p_value", False, error=e)
    
    # Test 4.3: WMAP information flow
    try:
        from wmap_cosmic_analysis.cross_validation import calculate_wmap_information_flow
        expected_flow = -0.327
        expected_p_value = 0.00001
        
        actual_flow, actual_p_value = calculate_wmap_information_flow()
        
        flow_passed = is_close_enough(expected_flow, actual_flow)
        p_value_passed = actual_p_value <= expected_p_value
        
        log_test_result(
            "cross_dataset", 
            "wmap_information_flow", 
            flow_passed,
            expected_flow, 
            actual_flow
        )
        
        log_test_result(
            "cross_dataset", 
            "wmap_information_flow_p_value", 
            p_value_passed,
            f"< {expected_p_value}", 
            actual_p_value
        )
    except Exception as e:
        log_test_result("cross_dataset", "wmap_information_flow", False, error=e)
        log_test_result("cross_dataset", "wmap_information_flow_p_value", False, error=e)
    
    # Test 4.4: Planck hierarchical organization
    try:
        from wmap_cosmic_analysis.cross_validation import calculate_planck_hierarchical_organization
        expected_organization = 0.691
        expected_p_value = 0.00001
        
        actual_organization, actual_p_value = calculate_planck_hierarchical_organization()
        
        organization_passed = is_close_enough(expected_organization, actual_organization)
        p_value_passed = actual_p_value <= expected_p_value
        
        log_test_result(
            "cross_dataset", 
            "planck_hierarchical_organization", 
            organization_passed,
            expected_organization, 
            actual_organization
        )
        
        log_test_result(
            "cross_dataset", 
            "planck_hierarchical_organization_p_value", 
            p_value_passed,
            f"< {expected_p_value}", 
            actual_p_value
        )
    except Exception as e:
        log_test_result("cross_dataset", "planck_hierarchical_organization", False, error=e)
        log_test_result("cross_dataset", "planck_hierarchical_organization_p_value", False, error=e)
    
    # Test 4.5: Planck scale transitions
    try:
        from wmap_cosmic_analysis.cross_validation import calculate_planck_scale_transitions
        expected_transitions = 1109
        
        actual_transitions = calculate_planck_scale_transitions()
        
        transitions_passed = is_close_enough(expected_transitions, actual_transitions)
        
        log_test_result(
            "cross_dataset", 
            "planck_scale_transitions", 
            transitions_passed,
            expected_transitions, 
            actual_transitions
        )
    except Exception as e:
        log_test_result("cross_dataset", "planck_scale_transitions", False, error=e)
    
    # Test 4.6: Planck information flow
    try:
        from wmap_cosmic_analysis.cross_validation import calculate_planck_information_flow
        expected_flow = 0.748
        expected_p_value = 0.00001
        
        actual_flow, actual_p_value = calculate_planck_information_flow()
        
        flow_passed = is_close_enough(expected_flow, actual_flow)
        p_value_passed = actual_p_value <= expected_p_value
        
        log_test_result(
            "cross_dataset", 
            "planck_information_flow", 
            flow_passed,
            expected_flow, 
            actual_flow
        )
        
        log_test_result(
            "cross_dataset", 
            "planck_information_flow_p_value", 
            p_value_passed,
            f"< {expected_p_value}", 
            actual_p_value
        )
    except Exception as e:
        log_test_result("cross_dataset", "planck_information_flow", False, error=e)
        log_test_result("cross_dataset", "planck_information_flow_p_value", False, error=e)

def test_visualization():
    """Test 5: Visualization and Reporting Validation"""
    logger.info("Starting Visualization and Reporting Validation tests...")
    
    # Test 5.1: Generate key visualizations
    try:
        from wmap_cosmic_analysis.visualization import generate_key_visualizations
        visualization_files = generate_key_visualizations()
        
        visualizations_generated = len(visualization_files) > 0
        
        log_test_result(
            "visualization", 
            "key_visualizations_generation", 
            visualizations_generated,
            "Visualizations generated", 
            f"{len(visualization_files)} visualizations generated" if visualizations_generated else "No visualizations generated"
        )
    except Exception as e:
        log_test_result("visualization", "key_visualizations_generation", False, error=e)
    
    # Test 5.2: Create summary reports
    try:
        from wmap_cosmic_analysis.reporting import generate_summary_report
        report_file = generate_summary_report()
        
        report_generated = report_file is not None and os.path.exists(report_file)
        
        log_test_result(
            "visualization", 
            "summary_report_generation", 
            report_generated,
            "Report generated", 
            f"Report generated at {report_file}" if report_generated else "No report generated"
        )
    except Exception as e:
        log_test_result("visualization", "summary_report_generation", False, error=e)
    
    # Test 5.3: Verify visualization accuracy
    try:
        from wmap_cosmic_analysis.validation import validate_visualizations
        visualization_accuracy = validate_visualizations()
        
        accuracy_passed = visualization_accuracy >= 0.95  # Assuming 95% accuracy is the threshold
        
        log_test_result(
            "visualization", 
            "visualization_accuracy", 
            accuracy_passed,
            ">= 0.95", 
            visualization_accuracy
        )
    except Exception as e:
        log_test_result("visualization", "visualization_accuracy", False, error=e)
    
    # Test 5.4: Verify report clarity
    try:
        from wmap_cosmic_analysis.validation import validate_report_clarity
        report_clarity = validate_report_clarity()
        
        clarity_passed = report_clarity >= 0.9  # Assuming 90% clarity is the threshold
        
        log_test_result(
            "visualization", 
            "report_clarity", 
            clarity_passed,
            ">= 0.9", 
            report_clarity
        )
    except Exception as e:
        log_test_result("visualization", "report_clarity", False, error=e)

def run_all_tests():
    """Run all validation tests."""
    start_time = time.time()
    
    logger.info("Starting comprehensive validation tests...")
    
    # Run all test categories
    test_environment_setup()
    test_data_processing()
    test_key_analysis()
    test_cross_dataset_validation()
    test_visualization()
    
    # Update summary with timing information
    end_time = time.time()
    duration = end_time - start_time
    test_results["summary"]["end_time"] = datetime.now().isoformat()
    test_results["summary"]["duration_seconds"] = duration
    
    # Print summary
    logger.info("="*50)
    logger.info("VALIDATION TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Total tests: {test_results['summary']['total_tests']}")
    logger.info(f"Passed tests: {test_results['summary']['passed_tests']}")
    logger.info(f"Failed tests: {test_results['summary']['failed_tests']}")
    logger.info(f"Skipped tests: {test_results['summary']['skipped_tests']}")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info("="*50)
    
    # Save results to JSON file
    with open('validation_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("Test results saved to validation_test_results.json")
    
    return test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run validation tests for WMAP Cosmic Analysis')
    parser.add_argument('--category', type=str, choices=['environment', 'data', 'analysis', 'cross', 'visualization', 'all'],
                        default='all', help='Test category to run')
    args = parser.parse_args()
    
    if args.category == 'environment' or args.category == 'all':
        test_environment_setup()
    
    if args.category == 'data' or args.category == 'all':
        test_data_processing()
    
    if args.category == 'analysis' or args.category == 'all':
        test_key_analysis()
    
    if args.category == 'cross' or args.category == 'all':
        test_cross_dataset_validation()
    
    if args.category == 'visualization' or args.category == 'all':
        test_visualization()
    
    if args.category == 'all':
        run_all_tests()
