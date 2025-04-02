import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def cross_validate_findings(test_results, datasets=['WMAP', 'Planck'], 
                           constants=['phi', 'pi', 'e', 'sqrt2', 'sqrt3', 'ln2'],
                           create_visualizations=True):
    """
    Comprehensive cross-validation framework for CMB analysis results.
    
    Parameters:
    - test_results: Dictionary mapping test names to results dictionaries
    - datasets: List of dataset names
    - constants: List of mathematical constants analyzed
    - create_visualizations: Whether to create visualizations (may cause errors in some environments)
    
    Returns:
    - Dictionary with cross-validation metrics and visualizations
    """
    # Initialize cross-validation results
    cross_val = {}
    
    # 1. Analyze scale dependence
    print("Analyzing scale dependence...")
    cross_val['scale_dependence'] = analyze_scale_dependence(test_results, datasets)
    
    # 2. Analyze constant specialization
    print("Analyzing constant specialization...")
    cross_val['constant_specialization'] = analyze_constant_specialization(test_results, datasets, constants)
    
    # 3. Calculate test correlations
    print("Calculating test correlations...")
    cross_val['test_correlations'] = calculate_test_correlations(test_results)
    
    # 4. Assess cross-dataset consistency
    print("Assessing cross-dataset consistency...")
    cross_val['cross_dataset_consistency'] = assess_cross_dataset_consistency(test_results, datasets)
    
    # 5. Perform multi-dimensional validation
    print("Performing multi-dimensional validation...")
    cross_val['multi_dimensional_validation'] = perform_multi_dimensional_validation(test_results, datasets, constants)
    
    # Generate summary
    cross_val['summary'] = generate_cross_validation_summary(cross_val)
    
    # Create visualizations (optional)
    if create_visualizations:
        try:
            cross_val['visualizations'] = create_cross_validation_visualizations(cross_val)
        except Exception as e:
            print("Warning: Could not create visualizations - %s" % e)
            cross_val['visualizations'] = {}
    
    return cross_val

def analyze_scale_dependence(test_results, datasets):
    """
    Analyze how results vary across different scales and datasets.
    
    This function specifically looks for patterns that support scale covariance
    rather than scale invariance.
    """
    scale_dependence = {}
    
    # Compare results across datasets (typically representing different scales)
    for test_name, results in test_results.items():
        scale_dependence[test_name] = {}
        
        # Extract metrics that can be compared across datasets
        comparable_metrics = find_comparable_metrics(results, datasets)
        
        for metric in comparable_metrics:
            # Calculate scale covariance: how metrics change across scales
            scale_values = {dataset: extract_metric_value(results, dataset, metric) 
                           for dataset in datasets if dataset in results}
            
            if len(scale_values) > 1:
                # Quantify scale dependence
                variation = np.std(list(scale_values.values())) / np.mean(list(scale_values.values()))
                scale_dependence[test_name][metric] = {
                    'values': scale_values,
                    'variation_coefficient': variation,
                    'scale_dependent': variation > 0.1  # Threshold for meaningful variation
                }
    
    return scale_dependence

def analyze_constant_specialization(test_results, datasets, constants):
    """
    Analyze how different mathematical constants specialize in different
    aspects of cosmic organization.
    """
    specialization = {constant: {'dominant_tests': [], 'dominant_scales': []} 
                     for constant in constants}
    
    # For each test, identify which constant performs best
    for test_name, results in test_results.items():
        if 'constant_performance' not in results:
            continue
            
        for dataset in datasets:
            if dataset not in results:
                continue
                
            # Find the dominant constant for this test and dataset
            constant_scores = results['constant_performance'][dataset]
            if constant_scores:
                dominant_constant = max(constant_scores.items(), key=lambda x: x[1])[0]
                specialization[dominant_constant]['dominant_tests'].append(
                    {'test': test_name, 'dataset': dataset})
    
    # For Information Architecture test, look at layer specialization
    if 'information_architecture' in test_results:
        arch_results = test_results['information_architecture']
        
        for dataset in datasets:
            if dataset in arch_results and 'layer_specialization' in arch_results[dataset]:
                layer_spec = arch_results[dataset]['layer_specialization']
                
                for scale, constant in layer_spec.items():
                    specialization[constant]['dominant_scales'].append(
                        {'scale': scale, 'dataset': dataset})
    
    return specialization

def calculate_test_correlations(test_results):
    """
    Calculate correlations between different test results to determine
    if they're measuring independent aspects of organization.
    """
    # Identify metrics that can be correlated across tests
    test_names = list(test_results.keys())
    correlation_matrix = pd.DataFrame(index=test_names, columns=test_names)
    p_value_matrix = pd.DataFrame(index=test_names, columns=test_names)
    
    for test1 in test_names:
        for test2 in test_names:
            if test1 == test2:
                correlation_matrix.loc[test1, test2] = 1.0
                p_value_matrix.loc[test1, test2] = 0.0
                continue
                
            # Try to find comparable metrics between tests
            metric_pairs = find_correlatable_metrics(test_results[test1], test_results[test2])
            
            if metric_pairs:
                # Use the first available pair of metrics
                metric1, metric2 = metric_pairs[0]
                values1 = extract_metric_values(test_results[test1], metric1)
                values2 = extract_metric_values(test_results[test2], metric2)
                
                # Ensure equal length
                min_length = min(len(values1), len(values2))
                if min_length > 1:
                    values1 = values1[:min_length]
                    values2 = values2[:min_length]
                    
                    # Calculate correlation
                    corr, p_value = stats.pearsonr(values1, values2)
                    correlation_matrix.loc[test1, test2] = corr
                    p_value_matrix.loc[test1, test2] = p_value
                else:
                    correlation_matrix.loc[test1, test2] = None
                    p_value_matrix.loc[test1, test2] = None
            else:
                correlation_matrix.loc[test1, test2] = None
                p_value_matrix.loc[test1, test2] = None
    
    return {
        'correlation_matrix': correlation_matrix,
        'p_value_matrix': p_value_matrix,
        'test_independence': assess_test_independence(correlation_matrix)
    }

def assess_cross_dataset_consistency(test_results, datasets):
    """
    Assess consistency of findings across independent datasets.
    Strong consistency would indicate intrinsic properties rather than artifacts.
    """
    consistency = {}
    
    for test_name, results in test_results.items():
        consistency[test_name] = {}
        
        # Extract key metrics for this test
        key_metrics = identify_key_metrics(results)
        
        for metric in key_metrics:
            metric_values = {dataset: extract_metric_value(results, dataset, metric)
                            for dataset in datasets if dataset in results}
            
            if len(metric_values) > 1:
                # Calculate consistency metrics
                consistency[test_name][metric] = {
                    'values': metric_values,
                    'direction_consistent': is_direction_consistent(metric_values),
                    'magnitude_ratio': calculate_magnitude_ratio(metric_values),
                    'significance_consistent': is_significance_consistent(
                        results, datasets, metric)
                }
    
    return consistency

def perform_multi_dimensional_validation(test_results, datasets, constants):
    """
    Perform multi-dimensional validation by integrating results across
    multiple tests and datasets.
    """
    # Count how many tests show significant results for each constant
    constant_significance = {constant: 0 for constant in constants}
    
    for test_name, results in test_results.items():
        for constant in constants:
            # Check if this constant shows significance in this test
            if is_constant_significant(results, constant):
                constant_significance[constant] += 1
    
    # Calculate overall confidence in findings
    overall_confidence = {}
    
    for constant in constants:
        # Higher confidence if multiple tests show significance
        test_confidence = constant_significance[constant] / len(test_results)
        
        # Check cross-dataset validation
        dataset_validation = assess_constant_across_datasets(
            test_results, datasets, constant)
        
        overall_confidence[constant] = {
            'test_confidence': test_confidence,
            'dataset_validation': dataset_validation,
            'overall_score': 0.7 * test_confidence + 0.3 * dataset_validation
        }
    
    return {
        'constant_significance': constant_significance,
        'overall_confidence': overall_confidence,
        'most_validated_constant': max(overall_confidence.items(), 
                                    key=lambda x: x[1]['overall_score'])[0]
    }

def generate_cross_validation_summary(cross_val):
    """Generate a comprehensive summary of cross-validation results."""
    summary = {
        'scale_dependence': summarize_scale_dependence(cross_val['scale_dependence']),
        'constant_specialization': summarize_constant_specialization(
            cross_val['constant_specialization']),
        'test_independence': summarize_test_independence(
            cross_val['test_correlations']),
        'cross_dataset_consistency': summarize_consistency(
            cross_val['cross_dataset_consistency']),
        'multi_dimensional_validation': summarize_multi_dimensional_validation(
            cross_val['multi_dimensional_validation'])
    }
    return summary

def create_cross_validation_visualizations(cross_val):
    """
    Create visualizations of cross-validation results.
    
    Parameters:
    - cross_val: Cross-validation results dictionary
    
    Returns:
    - Dictionary of matplotlib plot objects
    """
    visualizations = {}
    
    # 1. Create correlation heatmap for test independence
    if 'test_correlations' in cross_val and 'correlation_matrix' in cross_val['test_correlations']:
        try:
            corr_matrix = cross_val['test_correlations']['correlation_matrix']
            
            # Convert to numpy array if it's a pandas DataFrame
            if hasattr(corr_matrix, 'values'):
                corr_matrix = corr_matrix.values
                
            # Ensure matrix is numeric
            if not np.issubdtype(corr_matrix.dtype, np.number):
                print("Warning: Correlation matrix contains non-numeric values, skipping visualization")
            else:
                plt.figure(figsize=(10, 8))
                plt.title('Test Correlation Heatmap')
                plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar(label='Pearson Correlation')
                
                # Get test names
                if hasattr(cross_val['test_correlations']['correlation_matrix'], 'columns'):
                    test_names = cross_val['test_correlations']['correlation_matrix'].columns
                else:
                    test_names = ['Test %d' % i for i in range(len(corr_matrix))]
                    
                plt.xticks(range(len(test_names)), test_names, rotation=90)
                plt.yticks(range(len(test_names)), test_names)
                visualizations['correlation_heatmap'] = plt
        except Exception as e:
            print("Warning: Could not create correlation heatmap - %s" % e)
    
    # 2. Create constant specialization visualization
    if 'constant_specialization' in cross_val:
        const_spec = cross_val['constant_specialization']
        
        # Count dominant tests per constant
        dominant_counts = {const: len(info['dominant_tests']) 
                          for const, info in const_spec.items()}
        
        plt.figure(figsize=(10, 6))
        plt.bar(dominant_counts.keys(), dominant_counts.values())
        plt.title('Constant Specialization: Dominant Tests Count')
        plt.xlabel('Mathematical Constant')
        plt.ylabel('Number of Tests/Datasets Where Dominant')
        visualizations['constant_specialization'] = plt
    
    # 3. Multi-dimensional validation visualization
    if 'multi_dimensional_validation' in cross_val:
        multi_val = cross_val['multi_dimensional_validation']
        
        if 'overall_confidence' in multi_val:
            overall_conf = multi_val['overall_confidence']
            
            constants = list(overall_conf.keys())
            overall_scores = [info['overall_score'] for info in overall_conf.values()]
            
            plt.figure(figsize=(10, 6))
            plt.bar(constants, overall_scores)
            plt.title('Overall Confidence by Mathematical Constant')
            plt.xlabel('Mathematical Constant')
            plt.ylabel('Overall Confidence Score (0-1)')
            visualizations['overall_confidence'] = plt
    
    return visualizations

# Helper functions

def find_comparable_metrics(results, datasets):
    """Find metrics that can be compared across datasets."""
    comparable_metrics = []
    
    # Look for metrics that exist for multiple datasets
    if isinstance(results, dict):
        # First, collect all metrics from all datasets
        all_metrics = set()
        dataset_metrics = {}
        
        for dataset in datasets:
            if dataset in results:
                dataset_data = results[dataset]
                if isinstance(dataset_data, dict):
                    dataset_metrics[dataset] = set(dataset_data.keys())
                    all_metrics.update(dataset_metrics[dataset])
        
        # Find metrics that exist in multiple datasets
        for metric in all_metrics:
            count = sum(1 for dataset in datasets 
                      if dataset in dataset_metrics and metric in dataset_metrics[dataset])
            if count > 1:
                comparable_metrics.append(metric)
    
    return comparable_metrics

def extract_metric_value(results, dataset, metric):
    """Extract a specific metric value from test results."""
    try:
        # Handle different result structures
        if dataset in results and metric in results[dataset]:
            value = results[dataset][metric]
            
            # Convert to float if possible
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return 0.0
            elif isinstance(value, bool):
                return 1.0 if value else 0.0
            elif isinstance(value, dict) and 'value' in value:
                return float(value['value'])
        
        # Special case for constants
        if 'results' in results and dataset in results:
            for constant_data in results['results'].values():
                if metric in constant_data:
                    return float(constant_data[metric])
        
        return 0.0
    except:
        return 0.0

def find_correlatable_metrics(test1_results, test2_results):
    """Find pairs of metrics that can be correlated between tests."""
    correlatable_pairs = []
    
    # Define metrics that are conceptually similar across tests
    similarity_groups = [
        ['p_value', 'p-value', 'pvalue', 'significance'],
        ['coherence', 'mean_coherence', 'score', 'value'],
        ['z_score', 'z-score', 'zscore', 'standardized_score']
    ]
    
    # Extract all metrics from both tests
    test1_metrics = extract_all_metrics(test1_results)
    test2_metrics = extract_all_metrics(test2_results)
    
    # Find pairs of metrics that belong to the same similarity group
    for group in similarity_groups:
        test1_group_metrics = [m for m in test1_metrics if any(g in m.lower() for g in group)]
        test2_group_metrics = [m for m in test2_metrics if any(g in m.lower() for g in group)]
        
        for m1 in test1_group_metrics:
            for m2 in test2_group_metrics:
                correlatable_pairs.append((m1, m2))
    
    return correlatable_pairs

def extract_all_metrics(test_results):
    """Extract all metric names from test results."""
    metrics = set()
    
    def extract_from_dict(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                extract_from_dict(v, prefix + k + '.')
            elif isinstance(v, (int, float, bool, str)):
                metrics.add(prefix + k)
    
    if isinstance(test_results, dict):
        extract_from_dict(test_results)
    
    return list(metrics)

def extract_metric_values(test_results, metric):
    """Extract a list of values for a specific metric."""
    values = []
    
    def extract_from_dict(d):
        if metric in d:
            value = d[metric]
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, str):
                try:
                    values.append(float(value))
                except ValueError:
                    pass
        
        for k, v in d.items():
            if isinstance(v, dict):
                extract_from_dict(v)
    
    if isinstance(test_results, dict):
        extract_from_dict(test_results)
    
    return values

def assess_test_independence(correlation_matrix):
    """Assess how independent the different tests are."""
    independence = {}
    
    # Calculate average absolute correlation for each test
    for test in correlation_matrix.index:
        correlations = correlation_matrix.loc[test].dropna()
        # Remove self-correlation
        correlations = correlations[correlations.index != test]
        
        if not correlations.empty:
            avg_abs_corr = correlations.abs().mean()
            independence[test] = {
                'average_absolute_correlation': avg_abs_corr,
                'is_independent': avg_abs_corr < 0.5  # Threshold for independence
            }
    
    # Calculate overall independence score
    if independence:
        avg_scores = [info['average_absolute_correlation'] for info in independence.values()]
        independence['overall'] = {
            'average_absolute_correlation': sum(avg_scores) / len(avg_scores),
            'independent_test_count': sum(1 for info in independence.values() 
                                       if info.get('is_independent', False))
        }
    
    return independence

def identify_key_metrics(results):
    """Identify key metrics for a test."""
    key_metrics = []
    
    # Look for common metric names
    common_metrics = [
        'p_value', 'p-value', 'pvalue', 
        'coherence', 'mean_coherence', 'score',
        'z_score', 'z-score', 'zscore',
        'significance', 'significant'
    ]
    
    def search_dict(d, prefix=''):
        for k, v in d.items():
            if any(metric.lower() in k.lower() for metric in common_metrics):
                key_metrics.append(prefix + k)
            
            if isinstance(v, dict):
                search_dict(v, prefix + k + '.')
    
    if isinstance(results, dict):
        search_dict(results)
    
    return key_metrics

def is_direction_consistent(metric_values):
    """Check if metric values are consistent in direction across datasets."""
    if len(metric_values) < 2:
        return True
    
    values = list(metric_values.values())
    
    # Check if all values are positive or all are negative
    all_positive = all(v > 0 for v in values)
    all_negative = all(v < 0 for v in values)
    
    return all_positive or all_negative

def calculate_magnitude_ratio(metric_values):
    """Calculate ratio of magnitudes across datasets."""
    if len(metric_values) < 2:
        return 1.0
    
    values = list(metric_values.values())
    max_val = max(abs(v) for v in values)
    min_val = min(abs(v) for v in values)
    
    if min_val == 0:
        return float('inf')
    
    return max_val / min_val

def is_significance_consistent(results, datasets, metric):
    """Check if significance is consistent across datasets."""
    # Look for p-values or explicit significance indicators
    significance_indicators = {}
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        # Try to find p-value or significance indicator
        p_value = None
        
        # Direct p-value in dataset results
        if 'p_value' in results[dataset]:
            p_value = results[dataset]['p_value']
        elif 'p-value' in results[dataset]:
            p_value = results[dataset]['p-value']
        
        # Look in constant-specific results
        if 'results' in results and dataset in results['results']:
            for constant_data in results['results'][dataset].values():
                if 'p_value' in constant_data:
                    p_value = constant_data['p_value']
                elif 'p-value' in constant_data:
                    p_value = constant_data['p-value']
        
        if p_value is not None:
            try:
                p_value = float(p_value)
                significance_indicators[dataset] = p_value < 0.05
            except (ValueError, TypeError):
                pass
    
    # Check if significance is consistent
    if len(significance_indicators) >= 2:
        return all(significance_indicators.values()) or not any(significance_indicators.values())
    
    return True

def is_constant_significant(results, constant):
    """Check if a constant shows significance in test results."""
    # Look for the constant in results
    if 'results' in results:
        for dataset_results in results['results'].values():
            if constant in dataset_results:
                constant_data = dataset_results[constant]
                
                # Check for p-value
                if 'p_value' in constant_data:
                    try:
                        p_value = float(constant_data['p_value'])
                        if p_value < 0.05:
                            return True
                    except (ValueError, TypeError):
                        pass
                
                # Check for explicit significance indicator
                if 'significant' in constant_data:
                    return constant_data['significant']
    
    # Check for constant-specific results
    for dataset, dataset_results in results.items():
        if isinstance(dataset_results, dict) and constant in dataset_results:
            constant_data = dataset_results[constant]
            
            # Check for p-value
            if 'p_value' in constant_data:
                try:
                    p_value = float(constant_data['p_value'])
                    if p_value < 0.05:
                        return True
                except (ValueError, TypeError):
                    pass
            
            # Check for explicit significance indicator
            if 'significant' in constant_data:
                return constant_data['significant']
    
    return False

def assess_constant_across_datasets(test_results, datasets, constant):
    """Assess how consistently a constant performs across datasets."""
    dataset_count = 0
    significant_count = 0
    
    for test_name, results in test_results.items():
        for dataset in datasets:
            # Check if this dataset exists in results
            if dataset not in results and 'results' not in results:
                continue
            
            dataset_count += 1
            
            # Check if constant is significant in this dataset
            if is_constant_significant_in_dataset(results, dataset, constant):
                significant_count += 1
    
    if dataset_count == 0:
        return 0.0
    
    return significant_count / dataset_count

def is_constant_significant_in_dataset(results, dataset, constant):
    """Check if a constant is significant in a specific dataset."""
    # Check in dataset-specific results
    if dataset in results:
        dataset_results = results[dataset]
        
        if isinstance(dataset_results, dict):
            # Direct constant entry
            if constant in dataset_results:
                constant_data = dataset_results[constant]
                
                # Check p-value
                if 'p_value' in constant_data:
                    try:
                        p_value = float(constant_data['p_value'])
                        if p_value < 0.05:
                            return True
                    except (ValueError, TypeError):
                        pass
                
                # Check explicit significance
                if 'significant' in constant_data:
                    return constant_data['significant']
    
    # Check in overall results structure
    if 'results' in results and dataset in results['results']:
        dataset_results = results['results'][dataset]
        
        if isinstance(dataset_results, dict) and constant in dataset_results:
            constant_data = dataset_results[constant]
            
            # Check p-value
            if 'p_value' in constant_data:
                try:
                    p_value = float(constant_data['p_value'])
                    if p_value < 0.05:
                        return True
                except (ValueError, TypeError):
                    pass
            
            # Check explicit significance
            if 'significant' in constant_data:
                return constant_data['significant']
    
    return False

def summarize_scale_dependence(scale_dependence):
    """Summarize scale dependence results."""
    summary = {
        'scale_dependent_tests': [],
        'scale_invariant_tests': [],
        'overall_scale_dependence': 0.0
    }
    
    scale_dependent_count = 0
    total_tests = 0
    
    for test_name, metrics in scale_dependence.items():
        has_scale_dependence = False
        
        for metric, info in metrics.items():
            if info.get('scale_dependent', False):
                has_scale_dependence = True
                break
        
        if has_scale_dependence:
            scale_dependent_count += 1
            summary['scale_dependent_tests'].append(test_name)
        else:
            summary['scale_invariant_tests'].append(test_name)
        
        total_tests += 1
    
    if total_tests > 0:
        summary['overall_scale_dependence'] = scale_dependent_count / total_tests
    
    return summary

def summarize_constant_specialization(constant_specialization):
    """Summarize constant specialization results."""
    summary = {
        'specialized_constants': [],
        'general_constants': []
    }
    
    for constant, info in constant_specialization.items():
        dominant_test_count = len(info['dominant_tests'])
        dominant_scale_count = len(info['dominant_scales'])
        
        if dominant_test_count > 0 or dominant_scale_count > 0:
            summary['specialized_constants'].append({
                'constant': constant,
                'dominant_test_count': dominant_test_count,
                'dominant_scale_count': dominant_scale_count
            })
        else:
            summary['general_constants'].append(constant)
    
    # Sort specialized constants by dominance
    summary['specialized_constants'].sort(
        key=lambda x: x['dominant_test_count'] + x['dominant_scale_count'],
        reverse=True
    )
    
    if summary['specialized_constants']:
        summary['most_specialized_constant'] = summary['specialized_constants'][0]['constant']
    
    return summary

def summarize_test_independence(test_correlations):
    """Summarize test independence results."""
    summary = {}
    
    if 'test_independence' in test_correlations:
        independence = test_correlations['test_independence']
        
        if 'overall' in independence:
            summary['overall_independence'] = independence['overall']['average_absolute_correlation']
            summary['independent_test_count'] = independence['overall']['independent_test_count']
        
        # Identify most and least independent tests
        tests = [test for test in independence.keys() if test != 'overall']
        
        if tests:
            most_independent = min(tests, 
                                key=lambda t: independence[t]['average_absolute_correlation'])
            least_independent = max(tests, 
                                 key=lambda t: independence[t]['average_absolute_correlation'])
            
            summary['most_independent_test'] = most_independent
            summary['least_independent_test'] = least_independent
    
    return summary

def summarize_consistency(consistency_metrics):
    """Summarize cross-dataset consistency results."""
    summary = {
        'consistent_metrics': [],
        'inconsistent_metrics': [],
        'overall_consistency': 0.0
    }
    
    consistent_count = 0
    total_metrics = 0
    
    for test_name, metrics in consistency_metrics.items():
        for metric, info in metrics.items():
            is_consistent = (
                info.get('direction_consistent', False) and
                info.get('significance_consistent', False) and
                info.get('magnitude_ratio', float('inf')) < 5.0  # Threshold for consistency
            )
            
            metric_summary = {
                'test': test_name,
                'metric': metric,
                'direction_consistent': info.get('direction_consistent', False),
                'significance_consistent': info.get('significance_consistent', False),
                'magnitude_ratio': info.get('magnitude_ratio', float('inf'))
            }
            
            if is_consistent:
                consistent_count += 1
                summary['consistent_metrics'].append(metric_summary)
            else:
                summary['inconsistent_metrics'].append(metric_summary)
            
            total_metrics += 1
    
    if total_metrics > 0:
        summary['overall_consistency'] = consistent_count / total_metrics
    
    return summary

def summarize_multi_dimensional_validation(multi_dim_validation):
    """Summarize multi-dimensional validation results."""
    summary = {}
    
    if 'most_validated_constant' in multi_dim_validation:
        summary['most_validated_constant'] = multi_dim_validation['most_validated_constant']
    
    if 'overall_confidence' in multi_dim_validation:
        overall_conf = multi_dim_validation['overall_confidence']
        
        # Calculate average confidence
        confidence_scores = [info['overall_score'] for info in overall_conf.values()]
        if confidence_scores:
            summary['average_confidence'] = sum(confidence_scores) / len(confidence_scores)
        
        # Identify high and low confidence constants
        constants = list(overall_conf.keys())
        if constants:
            high_confidence = [c for c in constants if overall_conf[c]['overall_score'] > 0.7]
            low_confidence = [c for c in constants if overall_conf[c]['overall_score'] < 0.3]
            
            summary['high_confidence_constants'] = high_confidence
            summary['low_confidence_constants'] = low_confidence
    
    return summary
