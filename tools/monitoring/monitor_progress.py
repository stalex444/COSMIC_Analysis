#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monitor Progress of Information Architecture Test
This script checks the progress of the test by monitoring process activity
and file changes.
"""

import os
import sys
import time
import psutil
import glob
from datetime import datetime, timedelta

def find_test_processes():
    """Find all processes related to the Information Architecture Test"""
    test_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('run_ia_test_10k.py' in arg for arg in cmdline):
                    test_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return test_processes

def get_process_info(processes):
    """Get information about the test processes"""
    if not processes:
        return "No test processes found."
    
    main_process = processes[0]
    worker_processes = processes[1:] if len(processes) > 1 else []
    
    # Calculate runtime
    create_time = datetime.fromtimestamp(main_process.create_time())
    runtime = datetime.now() - create_time
    
    # Calculate CPU and memory usage
    total_cpu = sum(p.cpu_percent(interval=0.1) for p in processes)
    total_memory = sum(p.memory_info().rss for p in processes) / (1024 * 1024)  # MB
    
    info = []
    info.append("Test Process Information:")
    info.append("  Main PID: %d" % main_process.pid)
    info.append("  Worker Processes: %d" % len(worker_processes))
    info.append("  Running Time: %s" % str(runtime).split('.')[0])
    info.append("  Total CPU Usage: %.1f%%" % total_cpu)
    info.append("  Total Memory Usage: %.1f MB" % total_memory)
    
    return "\n".join(info)

def check_output_files():
    """Check the output files for the test"""
    results_dir = os.path.join(os.getcwd(), "results", "information_architecture_10k")
    
    if not os.path.exists(results_dir):
        return "No results directory found."
    
    info = []
    info.append("Output Files Information:")
    
    # Check WMAP results
    wmap_dir = os.path.join(results_dir, "wmap")
    if os.path.exists(wmap_dir):
        info.append("\nWMAP Dataset:")
        constant_dirs = [d for d in os.listdir(wmap_dir) if os.path.isdir(os.path.join(wmap_dir, d))]
        
        for constant in constant_dirs:
            constant_dir = os.path.join(wmap_dir, constant)
            info.append("  Constant: %s" % constant)
            
            # Check progress files
            progress_files = glob.glob(os.path.join(constant_dir, "progress*.txt"))
            for progress_file in progress_files:
                if os.path.exists(progress_file):
                    # Get the last line to determine progress
                    try:
                        with open(progress_file, 'r') as f:
                            lines = f.readlines()
                            
                        if len(lines) > 3:  # Header is 3 lines
                            # Count number of simulations
                            simulation_lines = [l for l in lines if l.strip() and not l.startswith('#')]
                            num_simulations = len(simulation_lines)
                            
                            # Get the last simulation result
                            if simulation_lines:
                                last_line = simulation_lines[-1].strip()
                                parts = last_line.split(',')
                                if len(parts) >= 3:
                                    sim_num, score, p_value = parts
                                    info.append("    Progress: %s/%s simulations (%.1f%%)" % 
                                              (sim_num, "10000", float(sim_num) / 100.0))
                                    info.append("    Latest p-value: %s" % p_value)
                            else:
                                info.append("    Progress: Starting simulations...")
                        else:
                            info.append("    Progress: Initializing...")
                    except Exception as e:
                        info.append("    Error reading progress file: %s" % str(e))
            
            # Check for completed results
            result_files = glob.glob(os.path.join(constant_dir, "results*.txt"))
            if result_files:
                info.append("    Status: Completed")
            else:
                info.append("    Status: In progress")
    
    # Check Planck results
    planck_dir = os.path.join(results_dir, "planck")
    if os.path.exists(planck_dir):
        info.append("\nPlanck Dataset:")
        constant_dirs = [d for d in os.listdir(planck_dir) if os.path.isdir(os.path.join(planck_dir, d))]
        
        for constant in constant_dirs:
            constant_dir = os.path.join(planck_dir, constant)
            info.append("  Constant: %s" % constant)
            
            # Check progress files
            progress_files = glob.glob(os.path.join(constant_dir, "progress*.txt"))
            for progress_file in progress_files:
                if os.path.exists(progress_file):
                    # Get the last line to determine progress
                    try:
                        with open(progress_file, 'r') as f:
                            lines = f.readlines()
                            
                        if len(lines) > 3:  # Header is 3 lines
                            # Count number of simulations
                            simulation_lines = [l for l in lines if l.strip() and not l.startswith('#')]
                            num_simulations = len(simulation_lines)
                            
                            # Get the last simulation result
                            if simulation_lines:
                                last_line = simulation_lines[-1].strip()
                                parts = last_line.split(',')
                                if len(parts) >= 3:
                                    sim_num, score, p_value = parts
                                    info.append("    Progress: %s/%s simulations (%.1f%%)" % 
                                              (sim_num, "10000", float(sim_num) / 100.0))
                                    info.append("    Latest p-value: %s" % p_value)
                            else:
                                info.append("    Progress: Starting simulations...")
                        else:
                            info.append("    Progress: Initializing...")
                    except Exception as e:
                        info.append("    Error reading progress file: %s" % str(e))
            
            # Check for completed results
            result_files = glob.glob(os.path.join(constant_dir, "results*.txt"))
            if result_files:
                info.append("    Status: Completed")
            else:
                info.append("    Status: In progress")
    
    if len(info) == 1:
        info.append("  No output files found.")
    
    return "\n".join(info)

def estimate_overall_progress():
    """Estimate the overall progress of the test"""
    results_dir = os.path.join(os.getcwd(), "results", "information_architecture_10k")
    
    if not os.path.exists(results_dir):
        return "No results directory found."
    
    # Define expected constants and datasets
    expected_constants = ['phi', 'sqrt2', 'sqrt3', 'ln2', 'e', 'pi']
    expected_datasets = ['wmap', 'planck']
    
    total_tasks = len(expected_constants) * len(expected_datasets)
    completed_tasks = 0
    in_progress_task = None
    in_progress_percent = 0
    
    # Check each dataset and constant
    for dataset in expected_datasets:
        dataset_dir = os.path.join(results_dir, dataset)
        if not os.path.exists(dataset_dir):
            continue
        
        for constant in expected_constants:
            constant_dir = os.path.join(dataset_dir, constant)
            if not os.path.exists(constant_dir):
                continue
            
            # Check if this constant is completed
            result_files = glob.glob(os.path.join(constant_dir, "results*.txt"))
            if result_files:
                completed_tasks += 1
            else:
                # This constant is in progress
                progress_files = glob.glob(os.path.join(constant_dir, "progress*.txt"))
                for progress_file in progress_files:
                    if os.path.exists(progress_file):
                        try:
                            with open(progress_file, 'r') as f:
                                lines = f.readlines()
                            
                            if len(lines) > 3:  # Header is 3 lines
                                # Count number of simulations
                                simulation_lines = [l for l in lines if l.strip() and not l.startswith('#')]
                                num_simulations = len(simulation_lines)
                                
                                if simulation_lines:
                                    last_line = simulation_lines[-1].strip()
                                    parts = last_line.split(',')
                                    if len(parts) >= 3:
                                        sim_num = int(parts[0])
                                        in_progress_percent = float(sim_num) / 100.0
                                        in_progress_task = "%s/%s" % (dataset, constant)
                        except:
                            pass
    
    # Calculate overall progress
    if total_tasks == 0:
        overall_percent = 0
    else:
        if in_progress_task:
            overall_percent = (completed_tasks * 100.0 + in_progress_percent) / total_tasks
        else:
            overall_percent = (completed_tasks * 100.0) / total_tasks
    
    info = []
    info.append("Overall Progress: %.1f%%" % overall_percent)
    info.append("Completed Tasks: %d/%d" % (completed_tasks, total_tasks))
    
    if in_progress_task:
        info.append("Current Task: %s (%.1f%%)" % (in_progress_task, in_progress_percent))
    
    return "\n".join(info)

def main():
    """Main function to monitor test progress"""
    print("\n" + "="*50)
    print("Information Architecture Test Progress Monitor")
    print("="*50)
    
    # Find test processes
    processes = find_test_processes()
    print("\n" + get_process_info(processes))
    
    # Check output files
    print("\n" + check_output_files())
    
    # Estimate overall progress
    print("\n" + estimate_overall_progress())
    
    print("\n" + "="*50)
    print("Last updated: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*50)

if __name__ == "__main__":
    main()
