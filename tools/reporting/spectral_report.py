#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Report generator for the Fractal Analysis (Spectral) Test.

This module provides functions for generating detailed reports from
fractal analysis test results, including visualization and interpretation
of Hurst exponent values and their relationship to the Golden Ratio.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json
from typing import Dict, Any, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SpectralReport")

# Mathematical constants
PHI = 1.618033988749895  # Golden ratio


def generate_spectral_report(results: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate a comprehensive report for the Fractal Analysis (Spectral) Test.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Dictionary containing fractal analysis test results
    
    Returns:
    --------
    Dict[str, str]
        Dictionary with report sections in Markdown format
    """
    logger.info("Generating Spectral Test report")
    
    # Initialize report sections
    report = {
        'title': 'Fractal Analysis (Spectral) Test Report',
        'summary': _generate_summary_section(results),
        'methodology': _generate_methodology_section(results),
        'results': _generate_results_section(results),
        'interpretation': _generate_interpretation_section(results),
        'comparison': _generate_comparison_section(results) if 'comparison' in results else '',
        'conclusion': _generate_conclusion_section(results)
    }
    
    return report


def _generate_summary_section(results: Dict[str, Any]) -> str:
    """Generate the summary section of the report."""
    summary = f"""
## Summary

This report presents the results of the Fractal Analysis (Spectral) Test performed on Cosmic Microwave Background (CMB) data. The test evaluates the fractal behavior in the CMB power spectrum using the Hurst exponent.

**Key Findings:**

- Hurst Exponent: {results.get('actual_hurst', 'N/A'):.6f}
- P-value: {results.get('p_value', 'N/A'):.6f}
- Statistical Significance: {"Yes" if results.get('p_value', 1) < 0.05 else "No"}
- Golden Ratio Optimality: {results.get('phi_optimality', 'N/A'):.6f}
- Analyzed Dataset: {results.get('name', 'Unknown').upper()}
- Number of Simulations: {results.get('n_simulations', 'N/A')}

The Hurst exponent provides insight into the long-range correlations and fractal structure of the CMB power spectrum. Of particular interest is its relationship to the Golden Ratio (φ), where a Hurst exponent of H = φ-1 ≈ 0.618 represents optimal fractal behavior.
"""
    return summary


def _generate_methodology_section(results: Dict[str, Any]) -> str:
    """Generate the methodology section of the report."""
    methodology = f"""
## Methodology

The Fractal Analysis Test employs the following methodology:

1. **Data Preprocessing**:
   - Normalization of CMB power spectrum data
   - Optional smoothing and detrending (not applied in this analysis)

2. **Hurst Exponent Calculation**:
   - Rescaled Range (R/S) Analysis with maximum lag of {results.get('max_lag', 20)}
   - Log-log linear regression to estimate Hurst exponent

3. **Statistical Significance Assessment**:
   - Monte Carlo simulations ({results.get('n_simulations', 10000)} iterations)
   - Generation of surrogate data with phase randomization
   - Two-tailed p-value calculation relative to Golden Ratio optimality

4. **Golden Ratio Optimality**:
   - Comparison of observed Hurst exponent to φ-1 ≈ 0.618
   - Calculation of optimality score normalized by random baseline

The Hurst exponent (H) provides key insights into the data's fractal structure:
- H = 0.5: Random walk (Brownian motion)
- 0 < H < 0.5: Anti-persistent series
- 0.5 < H < 1: Persistent series (fractal behavior)
- H ≈ φ-1: Optimal fractal behavior aligned with the Golden Ratio
"""
    return methodology


def _generate_results_section(results: Dict[str, Any]) -> str:
    """Generate the results section of the report."""
    # Extract key results
    hurst = results.get('actual_hurst', 0)
    p_value = results.get('p_value', 1)
    phi_opt = results.get('phi_optimality', 0)
    
    results_section = f"""
## Results

### Hurst Exponent Analysis

The analysis of the {results.get('name', '').upper()} CMB power spectrum yielded a Hurst exponent of **H = {hurst:.6f}**. This value indicates:

- {"Persistent behavior (H > 0.5), suggesting long-range correlations and fractal structure" if hurst > 0.5 else 
  "Anti-persistent behavior (H < 0.5), suggesting a tendency to revert to the mean" if hurst < 0.5 else 
  "Random walk behavior (H = 0.5), suggesting no long-range correlations"}

- The exponent differs from the Golden Ratio optimum (φ-1 ≈ 0.618) by {abs(hurst - (PHI-1)):.6f}

### Statistical Significance

The Monte Carlo analysis with {results.get('n_simulations', 10000)} simulations resulted in a p-value of **{p_value:.6f}**, which is {"statistically significant (p < 0.05)" if p_value < 0.05 else "not statistically significant (p ≥ 0.05)"}.

### Golden Ratio Optimality

The calculated Golden Ratio optimality score is **{phi_opt:.6f}**, which is {"above" if phi_opt > 1 else "below"} the random baseline of 1.0. This indicates that the observed fractal behavior {"shows stronger" if phi_opt > 1 else "does not show stronger"} alignment with the Golden Ratio than would be expected by chance.

"""
    return results_section


def _generate_interpretation_section(results: Dict[str, Any]) -> str:
    """Generate the interpretation section of the report."""
    # Extract key results
    hurst = results.get('actual_hurst', 0)
    p_value = results.get('p_value', 1)
    phi_opt = results.get('phi_optimality', 0)
    
    interpretation = f"""
## Interpretation

### Fractal Structure Implications

The observed Hurst exponent of {hurst:.6f} {"is significantly different from" if p_value < 0.05 else "cannot be distinguished from"} what would be expected by chance. This suggests:

"""
    
    # Add specific interpretations based on the Hurst exponent
    if hurst > 0.5 and p_value < 0.05:
        interpretation += """
- The CMB power spectrum exhibits persistent behavior, indicating long-term memory effects
- Scale-invariant patterns are present across different frequencies
- The power spectrum has self-similar (fractal) properties
- The structure shows more predictability than would be expected from random noise
"""
    elif hurst < 0.5 and p_value < 0.05:
        interpretation += """
- The CMB power spectrum exhibits anti-persistent behavior
- The data tends to revert to the mean more frequently than random noise
- This may suggest a "restoring force" in the underlying physical processes
- The pattern is more erratic but still contains structure different from pure randomness
"""
    else:
        interpretation += """
- The CMB power spectrum behavior is consistent with random processes
- No strong evidence for long-range correlations was detected
- The observed patterns could be explained by standard random processes
"""
    
    # Add specific interpretations for Golden Ratio alignment
    interpretation += f"""
### Golden Ratio Alignment

The proximity of the Hurst exponent to φ-1 (≈ 0.618) provides insights into the optimality of the fractal structure:

"""
    
    if abs(hurst - (PHI-1)) < 0.05:
        interpretation += """
- The Hurst exponent is very close to the Golden Ratio optimum
- This suggests the CMB exhibits a form of "optimal complexity"
- Such structures balance order and chaos in a way that maximizes information content
- Golden Ratio-related structures have been observed in many complex natural systems
"""
    else:
        if hurst > (PHI-1):
            interpretation += """
- The Hurst exponent is higher than the Golden Ratio optimum
- This indicates stronger persistence than would be optimal according to the φ-1 criterion
- The system may exhibit more order and less randomness than an optimally complex system
"""
        else:
            interpretation += """
- The Hurst exponent is lower than the Golden Ratio optimum
- This indicates less persistence than would be optimal according to the φ-1 criterion
- The system may exhibit more randomness and less order than an optimally complex system
"""
    
    return interpretation


def _generate_comparison_section(results: Dict[str, Any]) -> str:
    """Generate the comparison section of the report if comparison data exists."""
    if 'comparison' not in results:
        return ""
    
    comparison = results.get('comparison', {})
    
    section = f"""
## WMAP vs. Planck Comparison

### Hurst Exponent Comparison

- WMAP Hurst exponent: {comparison.get('wmap_hurst', 0):.6f}
- Planck Hurst exponent: {comparison.get('planck_hurst', 0):.6f}
- Absolute difference: {comparison.get('hurst_difference', 0):.6f}
- Ratio (max/min): {comparison.get('hurst_ratio', 0):.6f}

{"The Hurst exponents for WMAP and Planck are very similar, suggesting consistent fractal behavior across both datasets." 
if comparison.get('hurst_difference', 1) < 0.05 else
"There is a notable difference between the Hurst exponents for WMAP and Planck, suggesting different fractal structures."}

### Statistical Significance Comparison

- WMAP p-value: {comparison.get('wmap_p_value', 1):.6f} ({'Significant' if comparison.get('wmap_p_value', 1) < 0.05 else 'Not significant'})
- Planck p-value: {comparison.get('planck_p_value', 1):.6f} ({'Significant' if comparison.get('planck_p_value', 1) < 0.05 else 'Not significant'})

"""
    
    # Add interpretation based on significance patterns
    wmap_sig = comparison.get('wmap_p_value', 1) < 0.05
    planck_sig = comparison.get('planck_p_value', 1) < 0.05
    
    if wmap_sig and planck_sig:
        section += """Both datasets show statistically significant fractal behavior, strongly supporting the presence of scale-invariant patterns in the CMB."""
    elif wmap_sig:
        section += """Only the WMAP data shows statistically significant fractal behavior, suggesting the Planck observations may contain more instrumental effects or noise."""
    elif planck_sig:
        section += """Only the Planck data shows statistically significant fractal behavior, suggesting the higher resolution of Planck may better capture the fractal structure."""
    else:
        section += """Neither dataset shows statistically significant fractal behavior, suggesting the observed patterns could be due to chance or noise."""
    
    section += f"""

### Golden Ratio Optimality Comparison

- WMAP φ-optimality: {comparison.get('wmap_phi_optimality', 0):.6f}
- Planck φ-optimality: {comparison.get('planck_phi_optimality', 0):.6f}
- Absolute difference: {comparison.get('phi_difference', 0):.6f}
- Ratio (max/min): {comparison.get('phi_ratio', 0):.6f}

"""
    
    # Add interpretation based on phi-optimality patterns
    wmap_phi = comparison.get('wmap_phi_optimality', 0) > 1.1
    planck_phi = comparison.get('planck_phi_optimality', 0) > 1.1
    phi_diff = comparison.get('phi_difference', 1)
    
    if wmap_phi and planck_phi:
        section += """Both datasets show above-random alignment with Golden Ratio optimality, """
        if phi_diff < 0.1:
            section += """with very similar levels of alignment across datasets."""
        elif comparison.get('wmap_phi_optimality', 0) > comparison.get('planck_phi_optimality', 0):
            section += """with WMAP showing stronger alignment than Planck."""
        else:
            section += """with Planck showing stronger alignment than WMAP."""
    elif wmap_phi:
        section += """Only the WMAP data shows above-random alignment with Golden Ratio optimality."""
    elif planck_phi:
        section += """Only the Planck data shows above-random alignment with Golden Ratio optimality."""
    else:
        section += """Neither dataset shows strong alignment with Golden Ratio optimality."""
    
    return section


def _generate_conclusion_section(results: Dict[str, Any]) -> str:
    """Generate the conclusion section of the report."""
    # Extract key results
    hurst = results.get('actual_hurst', 0)
    p_value = results.get('p_value', 1)
    phi_opt = results.get('phi_optimality', 0)
    
    conclusion = f"""
## Conclusion

The Fractal Analysis (Spectral) Test performed on the {results.get('name', '').upper()} CMB power spectrum has revealed:

1. **Fractal Behavior**: The observed Hurst exponent of {hurst:.6f} indicates {"persistent" if hurst > 0.5 else "anti-persistent" if hurst < 0.5 else "random"} behavior, which {"is" if p_value < 0.05 else "is not"} statistically significant.

2. **Golden Ratio Alignment**: The fractal structure {"shows" if phi_opt > 1 else "does not show"} alignment with the Golden Ratio optimality criterion (φ-1 ≈ 0.618).

3. **Scale Invariance**: {"Evidence of" if p_value < 0.05 and hurst != 0.5 else "No strong evidence for"} scale-invariant patterns across different frequencies.

These findings {"support" if p_value < 0.05 and phi_opt > 1 else "do not strongly support"} the hypothesis that the CMB exhibits fractal behavior with specific mathematical properties linked to universal constants. The analysis suggests that {"there are underlying organizational principles in the cosmic structure that transcend simple randomness" if p_value < 0.05 else "the observed patterns could be explained by standard random processes"}.

Future work could expand this analysis to:
- Test different preprocessing techniques
- Apply multifractal analysis methods
- Compare with alternative cosmological models
- Investigate correlations with other CMB features
"""
    return conclusion
