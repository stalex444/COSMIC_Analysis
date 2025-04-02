#!/usr/bin/env python
"""
Full Transfer Entropy Test Runner with 10k Simulations

This script runs the transfer entropy test with 10,000 simulations and no early stopping,
providing the statistical power needed for publication-quality results.

The transfer entropy test measures information flow between different scales in the
CMB power spectrum, with particular focus on identifying golden ratio patterns.
"""

import os
import sys
import time
import datetime
import subprocess

def main():
    """Run the full transfer entropy test with 10k simulations."""
    start_time = time.time()
    
    # Set up paths
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create a timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(repo_root, f"results/transfer_entropy/te_10k_{timestamp}")
    
    # Construct the command to run the transfer entropy test
    cmd = [
        "python", "-m", "scripts.transfer_entropy.test_transfer_entropy",
        "--n-simulations", "10000",
        "--scales", "10",       # Analyze 10 scales for better coverage
        "--bins", "15",         # Increase bin count for better resolution
        "--parallel",           # Use parallel processing
        "--timeout", "0",       # No timeout (0 = run until completion)
        "--visualize",          # Generate visualizations
        "--output-dir", output_dir
    ]
    
    # Print run information
    print("\n" + "="*80)
    print(f"Starting Full Transfer Entropy Test ({timestamp})")
    print("="*80)
    print(f"- Simulations: 10,000")
    print(f"- Scales: 10")
    print(f"- Bins: 15")
    print(f"- Parallel: Yes")
    print(f"- Timeout: None (will run until completion)")
    print(f"- Output directory: {output_dir}")
    print("="*80 + "\n")
    
    try:
        # Run the command from the repository root directory
        subprocess.run(cmd, cwd=repo_root, check=True)
        
        # Print completion message
        elapsed_time = (time.time() - start_time) / 60  # Convert to minutes
        print("\n" + "="*80)
        print(f"Transfer Entropy Test completed in {elapsed_time:.2f} minutes")
        print(f"Results saved to: {output_dir}")
        print("="*80 + "\n")
        
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running transfer entropy test: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        print(f"Partial results may be available in: {output_dir}")
        return 130

if __name__ == "__main__":
    sys.exit(main())
