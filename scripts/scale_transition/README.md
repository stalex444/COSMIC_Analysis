# Scale Transition Analysis

Scripts for analyzing transitions between different scale regimes in CMB data.

## Overview

The Scale Transition Test analyzes scale boundaries where organizational principles change in the CMB power spectrum. It identifies transitions between different organizational regimes and analyzes their alignment with mathematical constants, particularly the golden ratio (φ).

This implementation is designed to run with 10,000 simulations for statistical robustness and includes special analysis of window size 55, which has previously shown strong Square Root of 2 specialization patterns in both WMAP and Planck data.

## Available Scripts

- `scale_transition_test.py`: Main script for running the Scale Transition Test on WMAP and Planck data
- `scale_transition_utils.py`: Utility functions for the Scale Transition Test

## Usage

Run the Scale Transition Test with the following command:

```bash
python scale_transition_test.py --wmap-file /path/to/wmap_data.txt --planck-file /path/to/planck_data.txt [options]
```

### Command Line Arguments

- `--wmap-file`: Path to WMAP power spectrum file (required)
- `--planck-file`: Path to Planck power spectrum file (required)
- `--output-dir`: Directory to save results (default: 'scale_transition_<timestamp>')
- `--n-simulations`: Number of Monte Carlo simulations (default: 10000)
- `--window-size`: Window size for complexity calculation (default: 10)
- `--n-clusters`: Number of clusters for transition detection (default: 3)
- `--timeout`: Maximum time in seconds to spend on simulations (default: 3600)
- `--num-processes`: Number of processes for parallel computation (default: all available cores)
- `--no-parallel`: Disable parallel processing

## Features

- Robust detection of scale transitions in CMB power spectrum data
- Efficient parallel processing optimized for 10,000 simulations
- Golden ratio alignment analysis to detect mathematical patterns
- Multiple window size analysis, including window size 55 which has shown strong sqrt(2) specialization
- Comprehensive visualization of results
- Detailed comparison between WMAP and Planck data sets
- JSON and text report generation for further analysis

## Output

The test generates the following outputs in the specified output directory:

- PNG visualizations of scale transitions and complexity
- Detailed text reports and JSON files with analysis results
- Comparative analysis between WMAP and Planck data
- P-values and phi-optimality scores for statistical analysis
