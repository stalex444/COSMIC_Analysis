#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check the progress of the Information Architecture Test
This script checks the status of the running test by examining the process memory
"""

import os
import sys
import psutil
import time

def find_python_processes():
    """Find all Python processes running on the system"""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('run_ia_test_efficient.py' in arg for arg in cmdline):
                    python_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return python_processes

def main():
    """Main function to check test progress"""
    print("Checking Information Architecture Test progress...")
    
    # Find Python processes
    processes = find_python_processes()
    
    if not processes:
        print("No Information Architecture Test processes found.")
        return
    
    print("Found %d test processes" % len(processes))
    
    # Get the main process (the one that's not a worker)
    main_process = None
    for proc in processes:
        try:
            if proc.parent() is None or proc.parent().pid == 1:
                main_process = proc
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if not main_process:
        main_process = processes[0]
    
    # Print process information
    print("\nProcess Information:")
    print("PID: %d" % main_process.pid)
    print("Running time: %s" % str(time.time() - main_process.create_time()))
    print("CPU usage: %.1f%%" % main_process.cpu_percent(interval=1.0))
    print("Memory usage: %.1f MB" % (main_process.memory_info().rss / 1024 / 1024))
    
    # Check if the process is responding
    print("\nProcess Status: %s" % main_process.status())
    
    # Check for any output files
    output_dir = "../results/information_architecture_10k_efficient"
    print("\nChecking output directory: %s" % output_dir)
    
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                file_mtime = os.path.getmtime(file_path)
                print("File: %s, Size: %d bytes, Modified: %s" % 
                      (file_path, file_size, time.ctime(file_mtime)))
    else:
        print("Output directory not found.")
    
    print("\nTest appears to be running, but progress reporting may be stuck.")
    print("You can try one of the following:")
    print("1. Wait longer - the test may be taking more time than expected")
    print("2. Restart the test with improved progress reporting")
    print("3. Check system resources to ensure there are no bottlenecks")

if __name__ == "__main__":
    main()
