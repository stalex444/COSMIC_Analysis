# Transfer Entropy Tests

This directory contains scripts for running transfer entropy analysis on CMB data. Transfer entropy 
measures the directed information flow from one time series to another, quantifying the statistical 
coherence between CMB scales.

## Available Scripts

- `test_transfer_entropy.py`: Original implementation
- `test_transfer_entropy_optimized.py`: Memory-optimized implementation for 10,000+ simulations

## Usage

Example:
```
python test_transfer_entropy_optimized.py --data_file /path/to/data --simulations 10000 --batch_size 100
```
