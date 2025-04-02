#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run all tests in the WMAP_Cosmic_Analysis repository with comprehensive reporting.
This script implements the testing plan to:
1. Run each test individually to identify specific failures
2. Focus on core tests related to GR coherence and cosmic consciousness analysis
3. Validate data provenance and validation tools
4. Cross-validate results between WMAP and Planck data
5. Generate comprehensive reports for each test category
"""

from __future__ import print_function
import os
import sys
import glob
import time
import datetime
import subprocess
import argparse
import json
import multiprocessing
from collections import OrderedDict

# Define test categories
TEST_CATEGORIES = {
    "gr_coherence": [
        "test_gr_specific_coherence.py",
        "tests/test_gr_coherence.py"
    ],
    "information_theory": [
        "test_transfer_entropy.py",
        "test_information_integration.py"
    ],
    "correlation_analysis": [
        "test_correlation_analysis.py"
    ],
    "structural_analysis": [
        "test_fractal_analysis.py",
        "test_hierarchical_organization.py",
        "test_laminarity.py"
    ],
    "coherence_analysis": [
        "test_coherence_analysis.py",
        "test_meta_coherence.py",
        "test_resonance_analysis.py",
        "test_scale_transition.py"
    ],
    "data_validation": [
        "test_wmap_data.py",
        "test_wmap_analysis.py",
        "tests/test_data_provenance.py",
        "tests/test_data_validation.py"
    ],
    "framework": [
        "tests/test_cosmic_framework.py",
        "tests/test_optimized_tests.py"
    ]
}

def find_test_files(directory="."):
    """
    Find all test files in the repository.
    
    Args:
        directory (str): Directory to search in
        
    Returns:
        list: List of test files
    """
    test_files = []
    
    # Find files matching test_*.py pattern
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    
    return sorted(test_files)

def categorize_test_files(test_files):
    """
    Categorize test files according to TEST_CATEGORIES.
    
    Args:
        test_files (list): List of test files
        
    Returns:
        dict: Dictionary of categorized test files
    """
    categorized = {}
    uncategorized = []
    
    # Initialize categories
    for category in TEST_CATEGORIES:
        categorized[category] = []
    
    # Categorize test files
    for test_file in test_files:
        test_file_rel = os.path.relpath(test_file)
        categorized_flag = False
        
        for category, patterns in TEST_CATEGORIES.items():
            for pattern in patterns:
                if test_file_rel.endswith(pattern):
                    categorized[category].append(test_file)
                    categorized_flag = True
                    break
            
            if categorized_flag:
                break
        
        if not categorized_flag:
            uncategorized.append(test_file)
    
    # Add uncategorized tests
    if uncategorized:
        categorized["uncategorized"] = uncategorized
    
    return categorized

def run_test(test_file, output_dir, n_simulations=10000, timeout=None):
    """
    Run a single test and capture its output.
    
    Args:
        test_file (str): Path to test file
        output_dir (str): Directory to save output
        n_simulations (int): Number of Monte Carlo simulations
        timeout (int): Timeout in seconds
        
    Returns:
        dict: Test results
    """
    start_time = time.time()
    test_name = os.path.basename(test_file)
    
    # Create output directory for this test
    test_output_dir = os.path.join(output_dir, test_name.replace(".py", ""))
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    
    # Prepare command
    cmd = [
        sys.executable,
        test_file,
        "--output-dir", test_output_dir,
        "--n-simulations", str(n_simulations)
    ]
    
    print("Running test: {}".format(test_name))
    print("Command: {}".format(" ".join(cmd)))
    
    # Run the test
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        returncode = process.returncode
        status = "success" if returncode == 0 else "failure"
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        returncode = -1
        status = "timeout"
    except Exception as e:
        stdout = ""
        stderr = str(e)
        returncode = -2
        status = "error"
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Save output
    with open(os.path.join(test_output_dir, "stdout.txt"), "w") as f:
        f.write(stdout)
    
    with open(os.path.join(test_output_dir, "stderr.txt"), "w") as f:
        f.write(stderr)
    
    # Prepare results
    results = {
        "test_file": test_file,
        "test_name": test_name,
        "status": status,
        "returncode": returncode,
        "duration": duration,
        "output_dir": test_output_dir,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Save results
    with open(os.path.join(test_output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print("Test completed: {} (status: {}, duration: {:.2f}s)".format(
        test_name, status, duration))
    
    return results

def run_test_category(category, test_files, output_dir, n_simulations=10000, timeout=None):
    """
    Run all tests in a category.
    
    Args:
        category (str): Category name
        test_files (list): List of test files in this category
        output_dir (str): Directory to save output
        n_simulations (int): Number of Monte Carlo simulations
        timeout (int): Timeout in seconds
        
    Returns:
        dict: Category results
    """
    category_start_time = time.time()
    
    # Create output directory for this category
    category_output_dir = os.path.join(output_dir, category)
    if not os.path.exists(category_output_dir):
        os.makedirs(category_output_dir)
    
    print("\n=== Running {} tests ({} files) ===\n".format(category, len(test_files)))
    
    # Run each test
    results = []
    for test_file in test_files:
        test_result = run_test(
            test_file,
            category_output_dir,
            n_simulations=n_simulations,
            timeout=timeout
        )
        results.append(test_result)
    
    category_end_time = time.time()
    category_duration = category_end_time - category_start_time
    
    # Calculate summary statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    failure_count = sum(1 for r in results if r["status"] == "failure")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    # Prepare category results
    category_results = {
        "category": category,
        "test_count": len(test_files),
        "success_count": success_count,
        "failure_count": failure_count,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "duration": category_duration,
        "test_results": results,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Save category results
    with open(os.path.join(category_output_dir, "category_results.json"), "w") as f:
        json.dump(category_results, f, indent=2)
    
    print("\n=== {} tests completed ===".format(category))
    print("Success: {}/{} ({:.1f}%)".format(
        success_count, len(test_files), 100 * success_count / len(test_files)))
    print("Failure: {}/{} ({:.1f}%)".format(
        failure_count, len(test_files), 100 * failure_count / len(test_files)))
    print("Timeout: {}/{} ({:.1f}%)".format(
        timeout_count, len(test_files), 100 * timeout_count / len(test_files)))
    print("Error: {}/{} ({:.1f}%)".format(
        error_count, len(test_files), 100 * error_count / len(test_files)))
    print("Total duration: {:.2f}s".format(category_duration))
    
    return category_results

def run_all_tests(output_dir, n_simulations=10000, timeout=None, categories=None, parallel=False):
    """
    Run all tests in the repository.
    
    Args:
        output_dir (str): Directory to save output
        n_simulations (int): Number of Monte Carlo simulations
        timeout (int): Timeout in seconds
        categories (list): List of categories to run (None for all)
        parallel (bool): Run categories in parallel
        
    Returns:
        dict: Overall results
    """
    overall_start_time = time.time()
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find test files
    test_files = find_test_files()
    print("Found {} test files".format(len(test_files)))
    
    # Categorize test files
    categorized_tests = categorize_test_files(test_files)
    
    # Filter categories if specified
    if categories:
        categorized_tests = {k: v for k, v in categorized_tests.items() if k in categories}
    
    # Save test inventory
    inventory = {
        "total_tests": len(test_files),
        "categories": {k: len(v) for k, v in categorized_tests.items()}
    }
    with open(os.path.join(output_dir, "test_inventory.json"), "w") as f:
        json.dump(inventory, f, indent=2)
    
    # Run tests by category
    category_results = {}
    
    if parallel:
        # Run categories in parallel
        pool = multiprocessing.Pool(processes=min(len(categorized_tests), multiprocessing.cpu_count()))
        tasks = []
        
        for category, category_test_files in categorized_tests.items():
            tasks.append((
                category,
                category_test_files,
                output_dir,
                n_simulations,
                timeout
            ))
        
        results = pool.starmap(run_test_category, tasks)
        pool.close()
        pool.join()
        
        # Collect results
        for result in results:
            category_results[result["category"]] = result
    else:
        # Run categories sequentially
        for category, category_test_files in categorized_tests.items():
            category_result = run_test_category(
                category,
                category_test_files,
                output_dir,
                n_simulations=n_simulations,
                timeout=timeout
            )
            category_results[category] = category_result
    
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    
    # Calculate overall statistics
    total_tests = sum(len(v) for v in categorized_tests.values())
    total_success = sum(r["success_count"] for r in category_results.values())
    total_failure = sum(r["failure_count"] for r in category_results.values())
    total_timeout = sum(r["timeout_count"] for r in category_results.values())
    total_error = sum(r["error_count"] for r in category_results.values())
    
    # Prepare overall results
    overall_results = {
        "total_tests": total_tests,
        "total_success": total_success,
        "total_failure": total_failure,
        "total_timeout": total_timeout,
        "total_error": total_error,
        "success_rate": 100 * total_success / total_tests if total_tests > 0 else 0,
        "duration": overall_duration,
        "category_results": category_results,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Save overall results
    with open(os.path.join(output_dir, "overall_results.json"), "w") as f:
        json.dump(overall_results, f, indent=2)
    
    # Generate summary report
    generate_summary_report(overall_results, output_dir)
    
    print("\n=== All tests completed ===")
    print("Total tests: {}".format(total_tests))
    print("Success: {}/{} ({:.1f}%)".format(
        total_success, total_tests, 100 * total_success / total_tests if total_tests > 0 else 0))
    print("Failure: {}/{} ({:.1f}%)".format(
        total_failure, total_tests, 100 * total_failure / total_tests if total_tests > 0 else 0))
    print("Timeout: {}/{} ({:.1f}%)".format(
        total_timeout, total_tests, 100 * total_timeout / total_tests if total_tests > 0 else 0))
    print("Error: {}/{} ({:.1f}%)".format(
        total_error, total_tests, 100 * total_error / total_tests if total_tests > 0 else 0))
    print("Total duration: {:.2f}s".format(overall_duration))
    
    return overall_results

def generate_summary_report(overall_results, output_dir):
    """
    Generate a summary report of test results.
    
    Args:
        overall_results (dict): Overall test results
        output_dir (str): Directory to save report
    """
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, "w") as f:
        f.write("=== WMAP Cosmic Analysis Test Summary ===\n")
        f.write("Date: {}\n".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("\n")
        
        f.write("=== Overall Results ===\n")
        f.write("Total tests: {}\n".format(overall_results["total_tests"]))
        f.write("Success: {}/{} ({:.1f}%)\n".format(
            overall_results["total_success"],
            overall_results["total_tests"],
            overall_results["success_rate"]))
        f.write("Failure: {}/{} ({:.1f}%)\n".format(
            overall_results["total_failure"],
            overall_results["total_tests"],
            100 * overall_results["total_failure"] / overall_results["total_tests"] 
            if overall_results["total_tests"] > 0 else 0))
        f.write("Timeout: {}/{} ({:.1f}%)\n".format(
            overall_results["total_timeout"],
            overall_results["total_tests"],
            100 * overall_results["total_timeout"] / overall_results["total_tests"]
            if overall_results["total_tests"] > 0 else 0))
        f.write("Error: {}/{} ({:.1f}%)\n".format(
            overall_results["total_error"],
            overall_results["total_tests"],
            100 * overall_results["total_error"] / overall_results["total_tests"]
            if overall_results["total_tests"] > 0 else 0))
        f.write("Total duration: {:.2f}s\n".format(overall_results["duration"]))
        f.write("\n")
        
        f.write("=== Results by Category ===\n")
        for category, results in overall_results["category_results"].items():
            f.write("--- {} ---\n".format(category))
            f.write("Tests: {}\n".format(results["test_count"]))
            f.write("Success: {}/{} ({:.1f}%)\n".format(
                results["success_count"],
                results["test_count"],
                100 * results["success_count"] / results["test_count"]
                if results["test_count"] > 0 else 0))
            f.write("Failure: {}/{} ({:.1f}%)\n".format(
                results["failure_count"],
                results["test_count"],
                100 * results["failure_count"] / results["test_count"]
                if results["test_count"] > 0 else 0))
            f.write("Duration: {:.2f}s\n".format(results["duration"]))
            f.write("\n")
        
        f.write("=== Failed Tests ===\n")
        failed_tests = []
        for category, results in overall_results["category_results"].items():
            for test_result in results["test_results"]:
                if test_result["status"] != "success":
                    failed_tests.append((category, test_result))
        
        if failed_tests:
            for category, test_result in failed_tests:
                f.write("- {} ({}): {}\n".format(
                    test_result["test_name"],
                    category,
                    test_result["status"]))
                f.write("  Output directory: {}\n".format(test_result["output_dir"]))
        else:
            f.write("No failed tests.\n")
    
    print("Summary report generated: {}".format(report_path))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run all tests in the WMAP_Cosmic_Analysis repository")
    parser.add_argument("--output-dir", default="test_results",
                        help="Directory to save test results")
    parser.add_argument("--n-simulations", type=int, default=10000,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout for each test in seconds")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=list(TEST_CATEGORIES.keys()) + ["all"],
                        help="Categories to run (default: all)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run categories in parallel")
    
    args = parser.parse_args()
    
    # Handle 'all' category
    if args.categories and "all" in args.categories:
        args.categories = None
    
    # Add timestamp to output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    
    # Run tests
    run_all_tests(
        output_dir,
        n_simulations=args.n_simulations,
        timeout=args.timeout,
        categories=args.categories,
        parallel=args.parallel
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
