#!/usr/bin/env python
"""
Import Path Fixer for COSMIC_Analysis Repository

This script scans Python files in the migrated repository and updates:
1. Import statements to reflect the new package structure
2. File path references to use the new directory layout
3. Adds necessary __init__.py files to make packages importable

Usage:
    python fix_imports.py [--test] [--apply]

Options:
    --test   Dry run mode, shows what would be changed without making changes
    --apply  Actually apply the changes (default is to just report)
"""

import os
import re
import sys
from pathlib import Path
import argparse
import glob

# Root of the repository
REPO_ROOT = Path('/Users/stephaniealexander/CascadeProjects/COSMIC_Analysis')

# Set of modules we know need to be updated
KNOWN_MODULES = {
    'utils': 'scripts.common.utils',
    'wmap_data': 'scripts.common.data_handlers',
    'config.config_loader': 'config.config_loader',
    'test_transfer_entropy': 'scripts.transfer_entropy.test_transfer_entropy',
    'test_transfer_entropy_optimized': 'scripts.transfer_entropy.test_transfer_entropy_optimized',
    'cmb_info_architecture': 'scripts.info_architecture.cmb_info_architecture',
    'information_architecture_test': 'scripts.info_architecture.archive.information_architecture_test',
    'cross_validation_framework': 'scripts.cross_validation.cross_validation_framework',
    'test_golden_ratio_cascade': 'scripts.golden_ratio.test_golden_ratio_cascade',
    'test_coherence_analysis': 'scripts.coherence.test_coherence_analysis',
    'phase_alignment_test': 'scripts.phase_alignment.phase_alignment_test',
    'cosmic_test_integration': 'scripts.integration.cosmic_test_integration',
}

# File path patterns to update
PATH_PATTERNS = {
    r'(?:\'|\")(?:\.{0,2}/)?data/([^\'\"]+)(?:\'|\")': r'"../data/\1"',  # Data paths
    r'(?:\'|\")(?:\.{0,2}/)?results/([^\'\"]+)(?:\'|\")': r'"../results/\1"',  # Results paths
    r'os\.path\.join\(os\.path\.dirname\(__file__\),\s*[\'"]\.{0,2}/data/([^\'"]+)[\'"]\)': r'os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "../data/\1")',  # os.path.join patterns
}

# Files to process or skip
def should_process_file(file_path):
    """Determine if a file should be processed."""
    # Skip __pycache__, _venv, etc.
    if any(part.startswith('_') or part.startswith('.') for part in file_path.parts):
        return False
    
    # Only process Python files
    if file_path.suffix != '.py':
        return False
    
    return True

def create_init_files():
    """Create __init__.py files in all directories to make them importable."""
    dirs_needing_init = []
    
    for root, dirs, files in os.walk(REPO_ROOT):
        root_path = Path(root)
        
        # Skip git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        if root_path != REPO_ROOT and '__init__.py' not in files:
            init_file = root_path / '__init__.py'
            dirs_needing_init.append(init_file)
    
    return dirs_needing_init

def fix_imports_in_file(file_path, apply_changes=False):
    """Fix imports in a single file."""
    changes = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Handle import statements
    import_pattern = re.compile(r'^(from|import)\s+([\w\.]+)(.*)$', re.MULTILINE)
    
    def import_replacer(match):
        import_type, module, rest = match.groups()
        
        # Only replace known modules
        if module in KNOWN_MODULES:
            new_module = KNOWN_MODULES[module]
            changes.append(f"{file_path}: {import_type} {module}{rest} -> {import_type} {new_module}{rest}")
            return f"{import_type} {new_module}{rest}"
        
        # Handle potential relative imports within modules
        parts = module.split('.')
        if parts[0] in KNOWN_MODULES:
            new_module = KNOWN_MODULES[parts[0]] + '.' + '.'.join(parts[1:])
            changes.append(f"{file_path}: {import_type} {module}{rest} -> {import_type} {new_module}{rest}")
            return f"{import_type} {new_module}{rest}"
            
        return match.group(0)
    
    new_content = import_pattern.sub(import_replacer, content)
    
    # Handle file paths
    for pattern, replacement in PATH_PATTERNS.items():
        path_pattern = re.compile(pattern)
        
        def path_replacer(match):
            old_text = match.group(0)
            # Keep the capture group from the original pattern
            new_text = path_pattern.sub(replacement, old_text)
            if old_text != new_text:
                changes.append(f"{file_path}: {old_text} -> {new_text}")
            return new_text
        
        new_content = path_pattern.sub(path_replacer, new_content)
    
    if apply_changes and content != new_content:
        with open(file_path, 'w') as f:
            f.write(new_content)
    
    return changes

def main():
    parser = argparse.ArgumentParser(description="Fix imports in the COSMIC_Analysis repository")
    parser.add_argument('--test', action='store_true', help="Run in test mode without making changes")
    parser.add_argument('--apply', action='store_true', help="Apply the changes")
    args = parser.parse_args()
    
    apply_changes = args.apply and not args.test
    
    print(f"Running in {'test' if not apply_changes else 'apply'} mode\n")
    
    # Create __init__.py files
    init_files = create_init_files()
    if apply_changes:
        for init_file in init_files:
            print(f"Creating {init_file}")
            init_file.touch()
    else:
        if init_files:
            print(f"Would create {len(init_files)} __init__.py files")
            for init_file in init_files[:5]:  # Show a sample
                print(f"  {init_file}")
            if len(init_files) > 5:
                print(f"  ... and {len(init_files) - 5} more")
    
    # Find all Python files
    all_changes = []
    for py_file in glob.glob(str(REPO_ROOT) + '/**/*.py', recursive=True):
        file_path = Path(py_file)
        if should_process_file(file_path):
            changes = fix_imports_in_file(file_path, apply_changes=apply_changes)
            all_changes.extend(changes)
    
    # Report changes
    if all_changes:
        print(f"\nFound {len(all_changes)} potential import/path changes:")
        for change in all_changes[:20]:  # Limit to 20 to prevent overwhelming output
            print(f"  {change}")
        if len(all_changes) > 20:
            print(f"  ... and {len(all_changes) - 20} more")
        
        if not apply_changes:
            print("\nRun with --apply to make these changes")
    else:
        print("No changes needed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
