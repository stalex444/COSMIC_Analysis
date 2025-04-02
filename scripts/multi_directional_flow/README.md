# Multi-Scale Directional Flow Analysis

This module implements a comprehensive Multi-Scale Directional Flow Analysis test for Cosmic Microwave Background (CMB) data. It analyzes directional tendencies in information flow across different scales, helping resolve contradictions found in previous analyses regarding bidirectional flow and negative transfer entropy results.

## Overview

The Multi-Scale Directional Flow Analysis provides insights into:

1. **Directional Information Flow**: Identifies predominant direction of information flow (forward, reverse, or balanced) across different scales
2. **Scale-Specific Patterns**: Detects scales that consistently act as information sources or sinks
3. **Statistical Significance**: Validates results against surrogate data to determine statistical significance
4. **Cross-Dataset Comparison**: Compares flow patterns between WMAP and Planck datasets

## Installation

No special installation is required beyond the Python dependencies:

- numpy
- matplotlib
- scipy
- argparse
- multiprocessing

## Usage

Run the analysis using the following command:

```bash
python multi_directional_analysis.py [options]
```

### Options

- `--wmap-only`: Run analysis only on WMAP data
- `--planck-only`: Run analysis only on Planck data
- `--num-scales`: Number of scales to analyze (default: 10)
- `--num-surrogates`: Number of surrogate datasets for statistical validation (default: 100)
- `--bins`: Number of bins for discretization (default: 10)
- `--delay`: Time delay for information transfer (default: 1)
- `--max-points`: Maximum number of points to use for transfer entropy calculation (default: 500)
- `--output-dir`: Output directory for results (default: results/multi_scale_flow_TIMESTAMP)

### Examples

Run analysis on both WMAP and Planck data with default parameters:
```bash
python multi_directional_analysis.py
```

Run analysis only on Planck data with increased number of scales:
```bash
python multi_directional_analysis.py --planck-only --num-scales 15
```

Run a more thorough analysis with more surrogate datasets:
```bash
python multi_directional_analysis.py --num-surrogates 1000
```

## Output

The analysis generates several output files and visualizations in the specified output directory:

1. **Information Flow Matrix**: Visualization of transfer entropy between scales
2. **Scale-Specific Flow Patterns**: Bar charts showing net directional ratio for each scale
3. **Statistical Significance**: Highlights scales with statistically significant flow patterns
4. **Dataset Comparison**: Comparative visualization of WMAP and Planck results

### Output Structure

```
output_dir/
├── wmap/
│   ├── flow_matrix.png
│   ├── scale_patterns.png
│   └── flow_results.json
├── planck/
│   ├── flow_matrix.png
│   ├── scale_patterns.png
│   └── flow_results.json
├── comparison/
│   ├── flow_pattern_comparison.png
│   └── flow_comparison.txt
└── multi_scale_flow_summary.txt
```

## Interpretation

The analysis results help interpret:

1. Whether information flows predominantly from larger to smaller scales (forward) or smaller to larger scales (reverse)
2. If certain scales consistently act as information sources or sinks
3. Whether flow patterns are consistent across different CMB datasets (WMAP vs. Planck)
4. How these patterns align with theoretical predictions from Consciousness Field Theory

## Relation to Previous Tests

This analysis complements the Information Architecture Test by:

1. Focusing on directionality rather than just hierarchical organization
2. Providing explicit characterization of flow regimes across scales
3. Highlighting differences between datasets that may explain contradictions in earlier results

## References

For theoretical background and methodology details, refer to:
- "Information Architecture in Cosmic Microwave Background Data" (Previous work)
- "Multi-Scale Information Flow in Complex Systems" (Methodology)
