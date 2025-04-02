#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rerun all WMAP data tests with validated data files.
This script identifies and reruns all tests that depend on WMAP data,
particularly focusing on those that might have used the corrupted ILC map.
"""

from __future__ import print_function
import os
import sys
import time
import subprocess
import datetime
import json

# Define the tests to rerun
TESTS_TO_RUN = [
    # Core tests that use WMAP data
    {
        "name": "Correlation Analysis (10k simulations)",
        "command": "python test_correlation_analysis.py --output-dir correlation_results_10k_validated --surrogates 10000 --gr-levels 10 --use-real-data",
        "description": "Rerun correlation analysis with 10,000 Monte Carlo simulations using validated WMAP data"
    },
    {
        "name": "Hierarchical Organization Test",
        "command": "python test_hierarchical_organization.py --output-dir hierarchy_results_validated",
        "description": "Test hierarchical organization in WMAP and Planck data with validated files"
    },
    {
        "name": "GR-Specific Coherence Test",
        "command": "python test_gr_specific_coherence.py --output-dir coherence_results_validated --n-simulations 10000",
        "description": "Test for golden ratio specific coherence in WMAP data with 10,000 simulations"
    },
    {
        "name": "Information Integration Test",
        "command": "python test_information_integration.py --output-dir info_integration_results_validated --n-simulations 10000",
        "description": "Test information integration in WMAP and Planck data with 10,000 simulations"
    },
    {
        "name": "Orthogonality Test (1k simulations)",
        "command": "python test_orthogonality.py --output-dir orthogonality_results_1k --n-simulations 1000",
        "description": "Test whether golden ratio patterns are orthogonal to other mathematical constants with 1,000 simulations"
    },
    {
        "name": "Golden Ratio Cascade Test (10k simulations)",
        "command": "python test_golden_ratio_cascade.py --output-dir results/golden_ratio_cascade_10k --num-simulations 10000",
        "description": "Test whether the Golden Ratio creates unique resonance cascade patterns across multiple scales in the CMB with 10,000 simulations"
    }
]

def run_test(test):
    """
    Run a single test and capture its output.
    
    Parameters:
    -----------
    test : dict
        Test configuration
        
    Returns:
    --------
    dict
        Test results
    """
    print("\n" + "="*80)
    print("Running test: %s" % test["name"])
    print(test["description"])
    print("Command: %s" % test["command"])
    print("="*80)
    
    start_time = time.time()
    
    # Create output directory if specified in command
    if "--output-dir" in test["command"]:
        output_dir = test["command"].split("--output-dir")[1].strip().split()[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Run the test
    process = subprocess.Popen(
        test["command"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Capture output in real-time
    stdout_lines = []
    stderr_lines = []
    
    # Process stdout
    for line in iter(process.stdout.readline, ""):
        stdout_lines.append(line.strip())
        print(line.strip())
        sys.stdout.flush()
    
    # Process stderr
    for line in iter(process.stderr.readline, ""):
        stderr_lines.append(line.strip())
        print("ERROR: %s" % line.strip())
        sys.stdout.flush()
    
    # Wait for process to complete
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    
    result = {
        "name": test["name"],
        "command": test["command"],
        "description": test["description"],
        "exit_code": process.returncode,
        "duration": duration,
        "stdout": stdout_lines,
        "stderr": stderr_lines,
        "start_time": datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if process.returncode == 0:
        print("\nTest completed successfully in %.2f seconds" % duration)
    else:
        print("\nTest failed with exit code %d after %.2f seconds" % (process.returncode, duration))
    
    return result

def validate_wmap_data():
    """
    Validate WMAP data files before running tests.
    
    Returns:
    --------
    bool
        True if all required files are valid, False otherwise
    """
    print("\n" + "="*80)
    print("Validating WMAP data files")
    print("="*80)
    
    # Run validation script
    process = subprocess.Popen(
        "python validate_wmap_data.py",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Capture output
    stdout, stderr = process.communicate()
    
    print(stdout)
    if stderr:
        print("ERROR: %s" % stderr)
    
    # Check if validation was successful
    if process.returncode == 0 or "All required WMAP data files are valid" in stdout or "Total files:" in stdout and "Invalid files: 0" in stdout:
        print("\nWMAP data validation successful")
        return True
    else:
        print("\nWMAP data validation failed")
        return False

def run_all_tests():
    """
    Run all tests and generate a summary report.
    
    Returns:
    --------
    dict
        Summary of test results
    """
    # First validate WMAP data
    if not validate_wmap_data():
        print("\nError: WMAP data validation failed. Please run validate_wmap_data.py --fix-issues to fix the issues.")
        return None
    
    # Run all tests
    results = []
    for test in TESTS_TO_RUN:
        result = run_test(test)
        results.append(result)
    
    # Generate summary
    summary = {
        "total_tests": len(results),
        "successful_tests": sum(1 for r in results if r["exit_code"] == 0),
        "failed_tests": sum(1 for r in results if r["exit_code"] != 0),
        "total_duration": sum(r["duration"] for r in results),
        "start_time": results[0]["start_time"] if results else None,
        "end_time": results[-1]["end_time"] if results else None,
        "results": results
    }
    
    # Save summary to file
    with open("wmap_tests_rerun_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("WMAP Tests Rerun Summary")
    print("="*80)
    print("Total tests: %d" % summary["total_tests"])
    print("Successful tests: %d" % summary["successful_tests"])
    print("Failed tests: %d" % summary["failed_tests"])
    print("Total duration: %.2f seconds (%.2f minutes)" % (summary["total_duration"], summary["total_duration"]/60))
    print("Start time: %s" % summary["start_time"])
    print("End time: %s" % summary["end_time"])
    print("\nDetailed results saved to: wmap_tests_rerun_summary.json")
    
    return summary

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rerun all WMAP data tests with validated data files")
    parser.add_argument("--test", type=int, help="Run a specific test by index (0-%d)" % (len(TESTS_TO_RUN)-1))
    
    args = parser.parse_args()
    
    if args.test is not None:
        if args.test < 0 or args.test >= len(TESTS_TO_RUN):
            print("Error: Invalid test index. Must be between 0 and %d" % (len(TESTS_TO_RUN)-1))
            return 1
        
        # Run a specific test
        test = TESTS_TO_RUN[args.test]
        result = run_test(test)
        return 0 if result["exit_code"] == 0 else 1
    else:
        # Run all tests
        summary = run_all_tests()
        return 0 if summary and summary["failed_tests"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
