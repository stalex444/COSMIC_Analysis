#!/bin/bash

# Run the definitive Information Architecture Test with 10,000 simulations
# This script runs the test on both WMAP and Planck data

echo "======================================================"
echo "Running definitive Information Architecture Test"
echo "10,000 simulations with full statistical analysis"
echo "======================================================"

# Create output directories
mkdir -p /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/definitive_results/wmap
mkdir -p /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/definitive_results/planck
mkdir -p /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/definitive_logs

# Set up Python environment with required dependencies
echo "Setting up Python environment..."
cd /Users/stephaniealexander/CascadeProjects
if [ -d "ia_test_env" ]; then
    echo "Using existing environment..."
    source ia_test_env/bin/activate
else
    echo "Creating environment..."
    python -m venv ia_test_env
    source ia_test_env/bin/activate
    pip install -r ia_test_requirements.txt
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/definitive_logs/ia_test_${TIMESTAMP}.log"

# Check if the new implementation exists
if [ ! -f "/Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/src/cmb_info_architecture.py" ]; then
    echo "ERROR: Implementation file not found. Make sure src/cmb_info_architecture.py exists."
    exit 1
fi

echo "======================================================"
echo "Starting test with timestamp: ${TIMESTAMP}"
echo "Log file: ${LOG_FILE}"
echo "======================================================"

# WMAP data test
echo "Running test on WMAP data..." | tee -a ${LOG_FILE}
python /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/src/cmb_info_architecture.py \
  --data_file /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/data/wmap_tt_spectrum_9yr_v5.txt \
  --data_type power_spectrum \
  --n_simulations 10000 \
  --scale_method conventional \
  --output_dir /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/definitive_results/wmap 2>&1 | tee -a ${LOG_FILE}

# Check if WMAP test was successful
if [ $? -ne 0 ]; then
    echo "ERROR: WMAP test failed. Check the log file for details." | tee -a ${LOG_FILE}
    echo "Exiting without running Planck test." | tee -a ${LOG_FILE}
    exit 1
fi

# Planck data test
echo "Running test on Planck data..." | tee -a ${LOG_FILE}
python /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/src/cmb_info_architecture.py \
  --data_file /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/data/planck_tt_spectrum_2018.txt \
  --data_type power_spectrum \
  --n_simulations 10000 \
  --scale_method conventional \
  --output_dir /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/definitive_results/planck 2>&1 | tee -a ${LOG_FILE}

# Check if Planck test was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Planck test failed. Check the log file for details." | tee -a ${LOG_FILE}
    exit 1
fi

echo "======================================================"
echo "Tests completed successfully!"
echo "Results stored in:"
echo "- /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/definitive_results/wmap"
echo "- /Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis/definitive_results/planck"
echo "Log file: ${LOG_FILE}"
echo "======================================================"
