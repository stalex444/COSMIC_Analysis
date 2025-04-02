#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the cross-validation framework.
This script creates mock test results and runs the cross-validation framework on them.
"""

import os
import numpy as np
import json
import sys
# Import cross-validation framework without importing matplotlib first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.cross_validation.cross_validation_framework import cross_validate_findings

def create_mock_test_results():
    """Create mock test results for testing the cross-validation framework."""
    # Constants to test
    constants = ['phi', 'sqrt2', 'sqrt3', 'ln2', 'e', 'pi']
    
    # Create mock phase alignment results
    phase_alignment = {
        'WMAP': {},
        'Planck': {},
        'constant_performance': {
            'WMAP': {},
            'Planck': {}
        }
    }
    
    # Create mock information architecture results
    info_arch = {
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
    
    # Generate random results for phase alignment test
    for dataset in ['WMAP', 'Planck']:
        for constant in constants:
            # Phase alignment results
            coherence = np.random.uniform(0.3, 0.7)
            p_value = np.random.uniform(0, 0.1) if constant in ['phi', 'sqrt2', 'sqrt3'] else np.random.uniform(0.1, 1.0)
            z_score = (0.5 - p_value) * 10
            
            phase_alignment[dataset][constant] = {
                'Traditional_p_value': p_value,
                'Traditional_z_score': z_score,
                'Traditional_coherence': coherence,
                'Traditional_significant': p_value < 0.05,
                'PLV_p_value': p_value * 0.9,  # Slightly different for PLV
                'PLV_z_score': z_score * 1.1,
                'PLV_coherence': coherence * 1.05,
                'PLV_significant': p_value < 0.05
            }
            
            # Performance score
            performance = coherence * (1.0 - min(p_value, 0.99))
            phase_alignment['constant_performance'][dataset][constant] = performance
            
            # Information architecture results
            score = np.random.uniform(0.3, 0.8)
            p_value = np.random.uniform(0, 0.1) if constant in ['phi', 'sqrt2', 'ln2'] else np.random.uniform(0.1, 1.0)
            
            info_arch[dataset][constant] = {
                'score': score,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # Performance score
            performance = score * (1.0 - min(p_value, 0.99))
            info_arch['constant_performance'][dataset][constant] = performance
    
    # Add layer specialization
    for dataset in ['WMAP', 'Planck']:
        info_arch['layer_specialization'][dataset] = {
            'low_scales': {'phi': 0.7, 'sqrt2': 0.6},
            'mid_scales': {'sqrt3': 0.65, 'ln2': 0.55},
            'high_scales': {'e': 0.4, 'pi': 0.35}
        }
    
    # Combine all results
    all_results = {
        'phase_alignment': phase_alignment,
        'information_architecture': info_arch
    }
    
    return all_results

def test_cross_validation():
    """Test the cross-validation framework with mock data."""
    print("Creating mock test results...")
    mock_results = create_mock_test_results()
    
    print("Running cross-validation...")
    try:
        # Skip visualizations to avoid matplotlib issues
        cross_val_results = cross_validate_findings(mock_results, create_visualizations=False)
        
        # Create output directory if it doesn't exist
        output_dir = "../results/test_cross_validation"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save cross-validation results
        results_file = os.path.join(output_dir, "cross_validation_test_results.json")
        
        # Make results serializable
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
        
        # Extract non-visualization results
        serializable_results = {}
        for key, value in cross_val_results.items():
            if key != 'visualizations':
                serializable_results[key] = make_serializable(value)
        
        # Save results
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print("Cross-validation test completed.")
        print("Results saved to %s" % results_file)
        
        # Print key findings
        if 'summary' in cross_val_results:
            print("\nKey Cross-Validation Findings:")
            summary = cross_val_results['summary']
            
            if 'scale_dependence' in summary:
                print("\nScale Dependence:")
                scale_dep = summary['scale_dependence']
                if 'scale_dependent_constants' in scale_dep:
                    print("  Scale-dependent constants: %s" % ', '.join(scale_dep['scale_dependent_constants']))
            
            if 'constant_specialization' in summary:
                print("\nConstant Specialization:")
                const_spec = summary['constant_specialization']
                if 'specialized_constants' in const_spec:
                    # Check if specialized_constants is a list of strings or a list of dicts
                    if const_spec['specialized_constants'] and isinstance(const_spec['specialized_constants'][0], dict):
                        # Extract constant names from the list of dicts
                        const_names = [c.get('constant', str(c)) for c in const_spec['specialized_constants']]
                        print("  Specialized constants: %s" % ', '.join(const_names))
                    else:
                        print("  Specialized constants: %s" % ', '.join(const_spec['specialized_constants']))
            
            if 'cross_dataset_consistency' in summary:
                print("\nCross-Dataset Consistency:")
                consistency = summary['cross_dataset_consistency']
                if 'consistency_score' in consistency:
                    print("  Overall consistency score: %.2f" % consistency['consistency_score'])
                if 'most_consistent_constants' in consistency:
                    print("  Most consistent constants: %s" % ', '.join(consistency['most_consistent_constants']))
            
            if 'multi_dimensional_validation' in summary:
                print("\nMulti-Dimensional Validation:")
                multi_val = summary['multi_dimensional_validation']
                if 'high_confidence_constants' in multi_val:
                    print("  High confidence constants: %s" % ', '.join(multi_val['high_confidence_constants']))
    
    except Exception as e:
        print("Error during cross-validation: %s" % e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cross_validation()
