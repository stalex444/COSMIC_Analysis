#!/bin/bash
# Definitive runner script for CMB Information Architecture Test
# This script runs the test on both WMAP and Planck data with 10,000 simulations
# Author: Stephanie Alexander
# Last Updated: 2025-03-31

# Exit on error
set -e

# Set up environment
echo "Setting up environment..."
if [ -d "cmb_env" ]; then
    echo "Using existing environment..."
    source cmb_env/bin/activate
else
    echo "Creating virtual environment..."
    python -m venv cmb_env
    source cmb_env/bin/activate
    
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Create output directories
mkdir -p results/information_architecture/wmap
mkdir -p results/information_architecture/planck
mkdir -p logs

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run test on WMAP data
echo "Running test on WMAP data..."
echo "======================================================" 

python src/cmb_info_architecture.py \
    --data_file data/wmap_tt_spectrum_9yr_v5.txt \
    --data_type power_spectrum \
    --n_simulations 10000 \
    --scale_method conventional \
    --output_dir results/information_architecture/wmap \
    2>&1 | tee logs/wmap_info_architecture_${TIMESTAMP}.log

# Check exit code
if [ $? -ne 0 ]; then
    echo "Error running WMAP test!"
    exit 1
fi

# Run test on Planck data
echo "Running test on Planck data..."
echo "======================================================" 

python src/cmb_info_architecture.py \
    --data_file data/planck_tt_spectrum_2018.txt \
    --data_type power_spectrum \
    --n_simulations 10000 \
    --scale_method conventional \
    --output_dir results/information_architecture/planck \
    2>&1 | tee logs/planck_info_architecture_${TIMESTAMP}.log

# Check exit code
if [ $? -ne 0 ]; then
    echo "Error running Planck test!"
    exit 1
fi

echo "Tests completed. Results stored in:"
echo "results/information_architecture/wmap"
echo "results/information_architecture/planck"
echo "Log files stored in logs/ directory"
