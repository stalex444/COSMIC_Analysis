#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate comprehensive reports from test results.
This script processes the output of run_all_tests.py to create detailed reports
with visualizations for each test category.
"""

from __future__ import print_function
import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_test_results(results_dir):
    """
    Load test results from the specified directory.
    
    Args:
        results_dir (str): Directory containing test results
        
    Returns:
        dict: Test results
    """
    overall_results_path = os.path.join(results_dir, "overall_results.json")
    
    if not os.path.exists(overall_results_path):
        print("Error: Overall results file not found: {}".format(overall_results_path))
        return None
    
    with open(overall_results_path, "r") as f:
        overall_results = json.load(f)
    
    return overall_results

def generate_overview_report(overall_results, output_dir):
    """
    Generate an overview report of all test results.
    
    Args:
        overall_results (dict): Overall test results
        output_dir (str): Directory to save report
    """
    report_path = os.path.join(output_dir, "overview_report.html")
    
    # Create HTML report
    with open(report_path, "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n")
        f.write("<head>\n")
        f.write("  <title>WMAP Cosmic Analysis Test Report</title>\n")
        f.write("  <style>\n")
        f.write("    body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write("    h1, h2, h3 { color: #333366; }\n")
        f.write("    table { border-collapse: collapse; width: 100%; }\n")
        f.write("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
        f.write("    th { background-color: #f2f2f2; }\n")
        f.write("    tr:nth-child(even) { background-color: #f9f9f9; }\n")
        f.write("    .success { color: green; }\n")
        f.write("    .failure { color: red; }\n")
        f.write("    .timeout { color: orange; }\n")
        f.write("    .error { color: darkred; }\n")
        f.write("  </style>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        
        # Header
        f.write("  <h1>WMAP Cosmic Analysis Test Report</h1>\n")
        f.write("  <p>Generated on: {}</p>\n".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        # Overall summary
        f.write("  <h2>Overall Summary</h2>\n")
        f.write("  <table>\n")
        f.write("    <tr><th>Metric</th><th>Value</th></tr>\n")
        f.write("    <tr><td>Total Tests</td><td>{}</td></tr>\n".format(
            overall_results["total_tests"]))
        f.write("    <tr><td>Success</td><td class='success'>{} ({:.1f}%)</td></tr>\n".format(
            overall_results["total_success"],
            overall_results["success_rate"]))
        f.write("    <tr><td>Failure</td><td class='failure'>{} ({:.1f}%)</td></tr>\n".format(
            overall_results["total_failure"],
            100 * overall_results["total_failure"] / overall_results["total_tests"]
            if overall_results["total_tests"] > 0 else 0))
        f.write("    <tr><td>Timeout</td><td class='timeout'>{} ({:.1f}%)</td></tr>\n".format(
            overall_results["total_timeout"],
            100 * overall_results["total_timeout"] / overall_results["total_tests"]
            if overall_results["total_tests"] > 0 else 0))
        f.write("    <tr><td>Error</td><td class='error'>{} ({:.1f}%)</td></tr>\n".format(
            overall_results["total_error"],
            100 * overall_results["total_error"] / overall_results["total_tests"]
            if overall_results["total_tests"] > 0 else 0))
        f.write("    <tr><td>Total Duration</td><td>{:.2f}s</td></tr>\n".format(
            overall_results["duration"]))
        f.write("  </table>\n")
        
        # Category summary
        f.write("  <h2>Category Summary</h2>\n")
        f.write("  <table>\n")
        f.write("    <tr><th>Category</th><th>Tests</th><th>Success</th><th>Failure</th>"
                "<th>Timeout</th><th>Error</th><th>Duration</th></tr>\n")
        
        for category, results in overall_results["category_results"].items():
            f.write("    <tr>\n")
            f.write("      <td>{}</td>\n".format(category))
            f.write("      <td>{}</td>\n".format(results["test_count"]))
            f.write("      <td class='success'>{} ({:.1f}%)</td>\n".format(
                results["success_count"],
                100 * results["success_count"] / results["test_count"]
                if results["test_count"] > 0 else 0))
            f.write("      <td class='failure'>{} ({:.1f}%)</td>\n".format(
                results["failure_count"],
                100 * results["failure_count"] / results["test_count"]
                if results["test_count"] > 0 else 0))
            f.write("      <td class='timeout'>{} ({:.1f}%)</td>\n".format(
                results["timeout_count"],
                100 * results["timeout_count"] / results["test_count"]
                if results["test_count"] > 0 else 0))
            f.write("      <td class='error'>{} ({:.1f}%)</td>\n".format(
                results["error_count"],
                100 * results["error_count"] / results["test_count"]
                if results["test_count"] > 0 else 0))
            f.write("      <td>{:.2f}s</td>\n".format(results["duration"]))
            f.write("    </tr>\n")
        
        f.write("  </table>\n")
        
        # Failed tests
        f.write("  <h2>Failed Tests</h2>\n")
        
        failed_tests = []
        for category, results in overall_results["category_results"].items():
            for test_result in results["test_results"]:
                if test_result["status"] != "success":
                    failed_tests.append((category, test_result))
        
        if failed_tests:
            f.write("  <table>\n")
            f.write("    <tr><th>Test</th><th>Category</th><th>Status</th>"
                    "<th>Duration</th><th>Output Directory</th></tr>\n")
            
            for category, test_result in failed_tests:
                status_class = {
                    "failure": "failure",
                    "timeout": "timeout",
                    "error": "error"
                }.get(test_result["status"], "")
                
                f.write("    <tr>\n")
                f.write("      <td>{}</td>\n".format(test_result["test_name"]))
                f.write("      <td>{}</td>\n".format(category))
                f.write("      <td class='{}'>{}</td>\n".format(
                    status_class, test_result["status"]))
                f.write("      <td>{:.2f}s</td>\n".format(test_result["duration"]))
                f.write("      <td>{}</td>\n".format(test_result["output_dir"]))
                f.write("    </tr>\n")
            
            f.write("  </table>\n")
        else:
            f.write("  <p>No failed tests.</p>\n")
        
        # Category links
        f.write("  <h2>Category Reports</h2>\n")
        f.write("  <ul>\n")
        
        for category in overall_results["category_results"]:
            category_report_path = os.path.join(
                "category_reports", "{}_report.html".format(category))
            f.write("    <li><a href='{}'>{}</a></li>\n".format(
                category_report_path, category))
        
        f.write("  </ul>\n")
        
        f.write("</body>\n")
        f.write("</html>\n")
    
    print("Overview report generated: {}".format(report_path))

def generate_category_report(category, results, output_dir):
    """
    Generate a detailed report for a test category.
    
    Args:
        category (str): Category name
        results (dict): Category results
        output_dir (str): Directory to save report
    """
    # Create category reports directory
    category_reports_dir = os.path.join(output_dir, "category_reports")
    if not os.path.exists(category_reports_dir):
        os.makedirs(category_reports_dir)
    
    report_path = os.path.join(category_reports_dir, "{}_report.html".format(category))
    
    # Create HTML report
    with open(report_path, "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n")
        f.write("<head>\n")
        f.write("  <title>{} Test Report</title>\n".format(category))
        f.write("  <style>\n")
        f.write("    body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write("    h1, h2, h3 { color: #333366; }\n")
        f.write("    table { border-collapse: collapse; width: 100%; }\n")
        f.write("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
        f.write("    th { background-color: #f2f2f2; }\n")
        f.write("    tr:nth-child(even) { background-color: #f9f9f9; }\n")
        f.write("    .success { color: green; }\n")
        f.write("    .failure { color: red; }\n")
        f.write("    .timeout { color: orange; }\n")
        f.write("    .error { color: darkred; }\n")
        f.write("    pre { background-color: #f5f5f5; padding: 10px; overflow: auto; }\n")
        f.write("  </style>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        
        # Header
        f.write("  <h1>{} Test Report</h1>\n".format(category))
        f.write("  <p><a href='../overview_report.html'>Back to Overview</a></p>\n")
        
        # Category summary
        f.write("  <h2>Summary</h2>\n")
        f.write("  <table>\n")
        f.write("    <tr><th>Metric</th><th>Value</th></tr>\n")
        f.write("    <tr><td>Tests</td><td>{}</td></tr>\n".format(
            results["test_count"]))
        f.write("    <tr><td>Success</td><td class='success'>{} ({:.1f}%)</td></tr>\n".format(
            results["success_count"],
            100 * results["success_count"] / results["test_count"]
            if results["test_count"] > 0 else 0))
        f.write("    <tr><td>Failure</td><td class='failure'>{} ({:.1f}%)</td></tr>\n".format(
            results["failure_count"],
            100 * results["failure_count"] / results["test_count"]
            if results["test_count"] > 0 else 0))
        f.write("    <tr><td>Timeout</td><td class='timeout'>{} ({:.1f}%)</td></tr>\n".format(
            results["timeout_count"],
            100 * results["timeout_count"] / results["test_count"]
            if results["test_count"] > 0 else 0))
        f.write("    <tr><td>Error</td><td class='error'>{} ({:.1f}%)</td></tr>\n".format(
            results["error_count"],
            100 * results["error_count"] / results["test_count"]
            if results["test_count"] > 0 else 0))
        f.write("    <tr><td>Duration</td><td>{:.2f}s</td></tr>\n".format(
            results["duration"]))
        f.write("  </table>\n")
        
        # Test details
        f.write("  <h2>Test Details</h2>\n")
        f.write("  <table>\n")
        f.write("    <tr><th>Test</th><th>Status</th><th>Duration</th>"
                "<th>Output Directory</th></tr>\n")
        
        for test_result in results["test_results"]:
            status_class = {
                "success": "success",
                "failure": "failure",
                "timeout": "timeout",
                "error": "error"
            }.get(test_result["status"], "")
            
            f.write("    <tr>\n")
            f.write("      <td><a href='#{}'>".format(test_result["test_name"]))
            f.write(test_result["test_name"])
            f.write("</a></td>\n")
            f.write("      <td class='{}'>{}</td>\n".format(
                status_class, test_result["status"]))
            f.write("      <td>{:.2f}s</td>\n".format(test_result["duration"]))
            f.write("      <td>{}</td>\n".format(test_result["output_dir"]))
            f.write("    </tr>\n")
        
        f.write("  </table>\n")
        
        # Individual test details
        for test_result in results["test_results"]:
            f.write("  <h2 id='{}'>{}</h2>\n".format(
                test_result["test_name"], test_result["test_name"]))
            
            status_class = {
                "success": "success",
                "failure": "failure",
                "timeout": "timeout",
                "error": "error"
            }.get(test_result["status"], "")
            
            f.write("  <p>Status: <span class='{}'>{}</span></p>\n".format(
                status_class, test_result["status"]))
            f.write("  <p>Duration: {:.2f}s</p>\n".format(test_result["duration"]))
            f.write("  <p>Output Directory: {}</p>\n".format(test_result["output_dir"]))
            
            # Show stdout and stderr
            stdout_path = os.path.join(test_result["output_dir"], "stdout.txt")
            stderr_path = os.path.join(test_result["output_dir"], "stderr.txt")
            
            if os.path.exists(stdout_path):
                with open(stdout_path, "r") as stdout_file:
                    stdout = stdout_file.read()
                
                f.write("  <h3>Standard Output</h3>\n")
                f.write("  <pre>{}</pre>\n".format(stdout))
            
            if os.path.exists(stderr_path):
                with open(stderr_path, "r") as stderr_file:
                    stderr = stderr_file.read()
                
                if stderr.strip():
                    f.write("  <h3>Standard Error</h3>\n")
                    f.write("  <pre>{}</pre>\n".format(stderr))
        
        f.write("</body>\n")
        f.write("</html>\n")
    
    print("Category report generated: {}".format(report_path))

def generate_visualizations(overall_results, output_dir):
    """
    Generate visualizations of test results.
    
    Args:
        overall_results (dict): Overall test results
        output_dir (str): Directory to save visualizations
    """
    # Create visualizations directory
    visualizations_dir = os.path.join(output_dir, "visualizations")
    if not os.path.exists(visualizations_dir):
        os.makedirs(visualizations_dir)
    
    # Category success rates
    plt.figure(figsize=(12, 8))
    
    categories = []
    success_rates = []
    
    for category, results in overall_results["category_results"].items():
        categories.append(category)
        success_rate = 100 * results["success_count"] / results["test_count"] if results["test_count"] > 0 else 0
        success_rates.append(success_rate)
    
    # Sort by success rate
    sorted_indices = np.argsort(success_rates)
    categories = [categories[i] for i in sorted_indices]
    success_rates = [success_rates[i] for i in sorted_indices]
    
    plt.barh(categories, success_rates, color='skyblue')
    plt.xlabel('Success Rate (%)')
    plt.title('Test Success Rate by Category')
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(success_rates):
        plt.text(v + 1, i, f'{v:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'category_success_rates.png'), dpi=300)
    plt.close()
    
    # Test duration by category
    plt.figure(figsize=(12, 8))
    
    categories = []
    durations = []
    
    for category, results in overall_results["category_results"].items():
        categories.append(category)
        durations.append(results["duration"])
    
    # Sort by duration
    sorted_indices = np.argsort(durations)
    categories = [categories[i] for i in sorted_indices]
    durations = [durations[i] for i in sorted_indices]
    
    plt.barh(categories, durations, color='lightgreen')
    plt.xlabel('Duration (seconds)')
    plt.title('Test Duration by Category')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(durations):
        plt.text(v + 1, i, f'{v:.1f}s', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'category_durations.png'), dpi=300)
    plt.close()
    
    # Overall status distribution
    plt.figure(figsize=(10, 10))
    
    labels = ['Success', 'Failure', 'Timeout', 'Error']
    sizes = [
        overall_results["total_success"],
        overall_results["total_failure"],
        overall_results["total_timeout"],
        overall_results["total_error"]
    ]
    colors = ['#4CAF50', '#F44336', '#FF9800', '#9C27B0']
    explode = (0.1, 0, 0, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('Test Status Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'status_distribution.png'), dpi=300)
    plt.close()
    
    print("Visualizations generated in: {}".format(visualizations_dir))

def generate_reports(results_dir, output_dir=None):
    """
    Generate comprehensive reports from test results.
    
    Args:
        results_dir (str): Directory containing test results
        output_dir (str): Directory to save reports (default: results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    # Load test results
    overall_results = load_test_results(results_dir)
    
    if overall_results is None:
        return 1
    
    # Generate overview report
    generate_overview_report(overall_results, output_dir)
    
    # Generate category reports
    for category, results in overall_results["category_results"].items():
        generate_category_report(category, results, output_dir)
    
    # Generate visualizations
    generate_visualizations(overall_results, output_dir)
    
    print("Reports generated in: {}".format(output_dir))
    return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive reports from test results")
    parser.add_argument("results_dir", help="Directory containing test results")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save reports (default: results_dir)")
    
    args = parser.parse_args()
    
    return generate_reports(args.results_dir, args.output_dir)

if __name__ == "__main__":
    sys.exit(main())
