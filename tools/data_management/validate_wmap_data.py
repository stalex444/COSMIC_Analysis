#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate WMAP data integrity.

This script checks all WMAP data files for integrity and validity,
ensuring they are not corrupted and can be properly loaded for analysis.
It builds on the download_wmap_data_lambda.py script to provide comprehensive
validation of all WMAP data files used in the analysis.
"""

from __future__ import print_function
import os
import sys
import glob
import json
import argparse
import hashlib
import numpy as np
from astropy.io import fits

# Import download script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_wmap_data_lambda

# Define expected WMAP data files and their properties
EXPECTED_FILES = {
    "power_spectrum": {
        "filename": "wmap_tt_spectrum_9yr_v5.txt",
        "min_lines": 1000,
        "contains": ["multipole", "power", "error"],
        "type": "txt",
        "data_type": "POWER_SPECTRUM",
        "required": True
    },
    "binned_power_spectrum": {
        "filename": "wmap_binned_tt_spectrum_9yr_v5.txt",
        "min_lines": 50,
        "contains": ["multipole", "power", "error"],
        "type": "txt",
        "data_type": "BINNED_POWER_SPECTRUM",
        "required": True
    },
    "ilc_map": {
        "filename": "wmap_ilc_9yr_v5.fits",
        "contains_header_keys": ["SIMPLE", "BITPIX", "NAXIS"],
        "type": "fits",
        "data_type": "ILC_MAP",
        "required": True
    },
    "likelihood_data": {
        "filename": "wmap_likelihood_data_9yr_v5.tar.gz",
        "min_size_mb": 5,
        "type": "tar.gz",
        "data_type": None,  # Not available in the direct URLs
        "required": False   # Making this optional since it's not critical for our current analysis
    }
}

def md5sum(filename):
    """
    Calculate MD5 hash of a file.
    
    Args:
        filename (str): Path to file
        
    Returns:
        str: MD5 hash
    """
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def validate_txt_file(filepath, min_lines=None, contains=None):
    """
    Validate a text file.
    
    Args:
        filepath (str): Path to file
        min_lines (int): Minimum number of lines expected
        contains (list): List of strings that should be in the file
        
    Returns:
        tuple: (is_valid, issues)
    """
    issues = []
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            return False, ["File does not exist"]
        
        # Check file size
        if os.path.getsize(filepath) == 0:
            issues.append("File is empty")
        
        # Read file
        with open(filepath, "r") as f:
            content = f.read()
            lines = content.split("\n")
        
        # Check number of lines
        if min_lines is not None and len(lines) < min_lines:
            issues.append("File has fewer than {} lines: {}".format(min_lines, len(lines)))
        
        # Check content
        if contains is not None:
            for keyword in contains:
                if keyword.lower() not in content.lower():
                    issues.append("File does not contain keyword: {}".format(keyword))
        
        # Check for HTML content (which might indicate an error page)
        if "<html" in content.lower() or "<body" in content.lower():
            issues.append("File appears to be an HTML document, not a data file")
        
        # Check for valid data structure (assuming space-separated values)
        data_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        if data_lines:
            try:
                # Try to parse the first data line
                values = data_lines[0].split()
                for value in values:
                    float(value)
            except ValueError:
                issues.append("First data line cannot be parsed as numbers")
        else:
            issues.append("No data lines found")
    
    except Exception as e:
        return False, ["Error reading file: {}".format(str(e))]
    
    return len(issues) == 0, issues

def validate_fits_file(filepath, contains_header_keys=None):
    """
    Validate a FITS file.
    
    Args:
        filepath (str): Path to file
        contains_header_keys (list): List of header keys that should be in the file
        
    Returns:
        tuple: (is_valid, issues)
    """
    issues = []
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            return False, ["File does not exist"]
        
        # Check file size
        if os.path.getsize(filepath) == 0:
            issues.append("File is empty")
        
        # Try to open with astropy
        try:
            with fits.open(filepath) as hdul:
                # Check if file has at least one HDU
                if len(hdul) == 0:
                    issues.append("FITS file has no HDUs")
                
                # Check primary HDU
                primary_hdu = hdul[0]
                
                # Check header keys
                if contains_header_keys is not None:
                    for key in contains_header_keys:
                        if key not in primary_hdu.header:
                            issues.append("FITS header missing key: {}".format(key))
                
                # Check for END card in header
                if 'END' not in str(primary_hdu.header):
                    issues.append("FITS header missing END card")
                
                # Check data
                if hasattr(primary_hdu, 'data') and primary_hdu.data is not None:
                    # Check if data is accessible
                    try:
                        data_shape = primary_hdu.data.shape
                        data_size = primary_hdu.data.size
                        
                        if data_size == 0:
                            issues.append("FITS data is empty")
                    except Exception as e:
                        issues.append("Error accessing FITS data: {}".format(str(e)))
                else:
                    issues.append("FITS file has no data in primary HDU")
        
        except Exception as e:
            issues.append("Error opening FITS file: {}".format(str(e)))
    
    except Exception as e:
        return False, ["Error validating FITS file: {}".format(str(e))]
    
    return len(issues) == 0, issues

def validate_archive_file(filepath, min_size_mb=None):
    """
    Validate an archive file (tar.gz, zip, etc.).
    
    Args:
        filepath (str): Path to file
        min_size_mb (float): Minimum size in MB
        
    Returns:
        tuple: (is_valid, issues)
    """
    issues = []
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            return False, ["File does not exist"]
        
        # Check file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb == 0:
            issues.append("File is empty")
        
        if min_size_mb is not None and file_size_mb < min_size_mb:
            issues.append("File size ({:.2f} MB) is less than expected ({} MB)".format(
                file_size_mb, min_size_mb))
        
        # Check file integrity
        # For tar.gz, we would need to use the tarfile module to check integrity
        # This is a simplified check
        if filepath.endswith(".tar.gz"):
            import tarfile
            try:
                with tarfile.open(filepath, "r:gz") as tar:
                    # Try to list contents
                    members = tar.getmembers()
                    if len(members) == 0:
                        issues.append("Archive is empty")
            except Exception as e:
                issues.append("Error opening archive: {}".format(str(e)))
    
    except Exception as e:
        return False, ["Error validating archive file: {}".format(str(e))]
    
    return len(issues) == 0, issues

def validate_wmap_file(filepath, file_info):
    """
    Validate a WMAP data file based on its type and expected properties.
    
    Args:
        filepath (str): Path to file
        file_info (dict): File information
        
    Returns:
        tuple: (is_valid, issues)
    """
    file_type = file_info.get("type", "")
    
    if file_type == "txt":
        return validate_txt_file(
            filepath,
            min_lines=file_info.get("min_lines"),
            contains=file_info.get("contains")
        )
    elif file_type == "fits":
        return validate_fits_file(
            filepath,
            contains_header_keys=file_info.get("contains_header_keys")
        )
    elif file_type in ["tar.gz", "zip"]:
        return validate_archive_file(
            filepath,
            min_size_mb=file_info.get("min_size_mb")
        )
    else:
        return False, ["Unknown file type: {}".format(file_type)]

def find_wmap_files(data_dir):
    """
    Find WMAP data files in the specified directory.
    
    Args:
        data_dir (str): Directory to search
        
    Returns:
        dict: Dictionary of found files
    """
    found_files = {}
    
    for file_key, file_info in EXPECTED_FILES.items():
        filename = file_info["filename"]
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            found_files[file_key] = filepath
        else:
            # Try to find the file with a glob pattern
            pattern = os.path.join(data_dir, "*" + filename)
            matches = glob.glob(pattern)
            
            if matches:
                found_files[file_key] = matches[0]
    
    return found_files

def validate_wmap_data(data_dir, report_file=None, fix_issues=False):
    """
    Validate all WMAP data files in the specified directory.
    
    Args:
        data_dir (str): Directory containing WMAP data
        report_file (str): Path to save validation report
        fix_issues (bool): Whether to attempt to fix issues
        
    Returns:
        bool: True if all files are valid
    """
    print("Validating WMAP data in: {}".format(data_dir))
    
    # Find WMAP files
    found_files = find_wmap_files(data_dir)
    
    # Check for missing files
    missing_files = set(EXPECTED_FILES.keys()) - set(found_files.keys())
    
    # Validate found files
    validation_results = {}
    all_valid = len(missing_files) == 0
    
    for file_key, filepath in found_files.items():
        file_info = EXPECTED_FILES[file_key]
        print("Validating {}: {}".format(file_key, os.path.basename(filepath)))
        
        is_valid, issues = validate_wmap_file(filepath, file_info)
        
        validation_results[file_key] = {
            "filepath": filepath,
            "is_valid": is_valid,
            "issues": issues,
            "md5": md5sum(filepath)
        }
        
        if not is_valid:
            all_valid = False
            print("  INVALID: {}".format(", ".join(issues)))
            
            if fix_issues:
                print("  Attempting to fix issues...")
                # Use the download function from the imported module
                data_type = file_info.get("data_type")
                if data_type:
                    try:
                        # Force re-download
                        download_wmap_data_lambda.download_wmap_data(
                            data_types=[data_type],
                            output_dir=data_dir,
                            force=True
                        )
                        
                        # Re-validate
                        is_valid, issues = validate_wmap_file(filepath, file_info)
                        validation_results[file_key]["is_valid"] = is_valid
                        validation_results[file_key]["issues"] = issues
                        validation_results[file_key]["md5"] = md5sum(filepath)
                        
                        if is_valid:
                            print("  Fixed successfully")
                        else:
                            print("  Failed to fix: {}".format(", ".join(issues)))
                    except Exception as e:
                        print("  Error fixing file: {}".format(str(e)))
                else:
                    print("  Cannot fix file: No download URL available")
        else:
            print("  VALID")
    
    # Report missing files
    if missing_files:
        print("\nMissing files:")
        for file_key in missing_files:
            file_info = EXPECTED_FILES[file_key]
            if file_info.get("required", True):
                print("  - {}".format(file_info["filename"]))
                all_valid = False
            else:
                print("  - {} (optional)".format(file_info["filename"]))
        
        if fix_issues:
            print("\nAttempting to download missing files...")
            
            for file_key in missing_files:
                file_info = EXPECTED_FILES[file_key]
                if file_info.get("required", True):
                    data_type = file_info.get("data_type")
                    if data_type:
                        print("  Downloading {}...".format(file_info["filename"]))
                        
                        try:
                            download_wmap_data_lambda.download_wmap_data(
                                data_types=[data_type],
                                output_dir=data_dir,
                                force=True
                            )
                            
                            # Check if file was downloaded
                            filepath = os.path.join(data_dir, file_info["filename"])
                            if os.path.exists(filepath):
                                # Validate
                                is_valid, issues = validate_wmap_file(filepath, file_info)
                                validation_results[file_key] = {
                                    "filepath": filepath,
                                    "is_valid": is_valid,
                                    "issues": issues,
                                    "md5": md5sum(filepath)
                                }
                                
                                if is_valid:
                                    print("  Downloaded and validated successfully")
                                else:
                                    print("  Downloaded but invalid: {}".format(", ".join(issues)))
                            else:
                                print("  Failed to download")
                        except Exception as e:
                            print("  Error downloading file: {}".format(str(e)))
                    else:
                        print("  Cannot download {}: No download URL available".format(
                            file_info["filename"]))
    
    # Generate validation report
    report = {
        "data_directory": data_dir,
        "all_valid": all_valid,
        "missing_files": list(missing_files),
        "validation_results": validation_results,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    if report_file:
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print("\nValidation report saved to: {}".format(report_file))
    
    # Print summary
    print("\nValidation Summary:")
    print("  Total files: {}".format(len(EXPECTED_FILES)))
    print("  Found files: {}".format(len(found_files)))
    print("  Missing files: {}".format(len(missing_files)))
    print("  Valid files: {}".format(sum(1 for r in validation_results.values() if r["is_valid"])))
    print("  Invalid files: {}".format(sum(1 for r in validation_results.values() if not r["is_valid"])))
    
    return all_valid

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Validate WMAP data integrity")
    parser.add_argument("--data-dir", default=".",
                        help="Directory containing WMAP data")
    parser.add_argument("--report-file", default="wmap_data_validation_report.json",
                        help="Path to save validation report")
    parser.add_argument("--fix-issues", action="store_true",
                        help="Attempt to fix issues by re-downloading files")
    
    args = parser.parse_args()
    
    # Validate WMAP data
    all_valid = validate_wmap_data(
        args.data_dir,
        args.report_file,
        args.fix_issues
    )
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    import datetime  # Import here to avoid conflict with function parameter
    sys.exit(main())
