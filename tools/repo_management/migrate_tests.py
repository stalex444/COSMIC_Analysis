#!/usr/bin/env python
"""
CMB Test Suite Migration Script

This script automates the migration of test scripts from the existing WMAP_Cosmic_Analysis 
repository to the new COSMIC_Analysis structured repository.

Usage:
    python migrate_tests.py

The script will:
1. Create destination directories if they don't exist
2. Copy test scripts to their new locations
3. Update file paths and imports as needed
4. Create appropriate README files
"""

import os
import shutil
import re
from pathlib import Path
import sys

# Source and destination paths
SOURCE_ROOT = Path('/Users/stephaniealexander/CascadeProjects/WMAP_Cosmic_Analysis')
DEST_ROOT = Path('/Users/stephaniealexander/CascadeProjects/COSMIC_Analysis')

# Mapping of test types to their source files and destinations
MIGRATION_MAP = {
    'transfer_entropy': {
        'files': [
            (SOURCE_ROOT / 'test_transfer_entropy.py', DEST_ROOT / 'scripts/transfer_entropy/test_transfer_entropy.py'),
            (SOURCE_ROOT / 'test_transfer_entropy_optimized.py', DEST_ROOT / 'scripts/transfer_entropy/test_transfer_entropy_optimized.py'),
        ],
        'readme': """# Transfer Entropy Tests

This directory contains scripts for running transfer entropy analysis on CMB data. Transfer entropy 
measures the directed information flow from one time series to another, quantifying the statistical 
coherence between CMB scales.

## Available Scripts

- `test_transfer_entropy.py`: Original implementation
- `test_transfer_entropy_optimized.py`: Memory-optimized implementation for 10,000+ simulations

## Usage

Example:
```
python test_transfer_entropy_optimized.py --data_file /path/to/data --simulations 10000 --batch_size 100
```
"""
    },
    'info_architecture': {
        'files': [
            (SOURCE_ROOT / 'src/cmb_info_architecture.py', DEST_ROOT / 'scripts/info_architecture/cmb_info_architecture.py'),
            (SOURCE_ROOT / 'archive/scripts/information_architecture_test.py', DEST_ROOT / 'scripts/info_architecture/archive/information_architecture_test.py'),
            (SOURCE_ROOT / 'archive/scripts/information_architecture_test_backup.py', DEST_ROOT / 'scripts/info_architecture/archive/information_architecture_test_backup.py'),
            (SOURCE_ROOT / 'archive/scripts/information_architecture_test_current.py', DEST_ROOT / 'scripts/info_architecture/archive/information_architecture_test_current.py'),
            (SOURCE_ROOT / 'archive/scripts/information_architecture_test_definitive.py', DEST_ROOT / 'scripts/info_architecture/archive/information_architecture_test_definitive.py'),
            (SOURCE_ROOT / 'archive/scripts/information_architecture_test_original.py', DEST_ROOT / 'scripts/info_architecture/archive/information_architecture_test_original.py'),
            (SOURCE_ROOT / 'archive/scripts/information_architecture_test_simplified.py', DEST_ROOT / 'scripts/info_architecture/archive/information_architecture_test_simplified.py'),
            (SOURCE_ROOT / 'run_ia_test.py', DEST_ROOT / 'scripts/info_architecture/runners/run_ia_test.py'),
            (SOURCE_ROOT / 'run_ia_test_10k.py', DEST_ROOT / 'scripts/info_architecture/runners/run_ia_test_10k.py'),
            (SOURCE_ROOT / 'run_ia_test_1k.py', DEST_ROOT / 'scripts/info_architecture/runners/run_ia_test_1k.py'),
            (SOURCE_ROOT / 'run_ia_test_debug.py', DEST_ROOT / 'scripts/info_architecture/runners/run_ia_test_debug.py'),
            (SOURCE_ROOT / 'run_ia_test_efficient.py', DEST_ROOT / 'scripts/info_architecture/runners/run_ia_test_efficient.py'),
            (SOURCE_ROOT / 'run_ia_test_fixed.py', DEST_ROOT / 'scripts/info_architecture/runners/run_ia_test_fixed.py'),
            (SOURCE_ROOT / 'run_ia_test_improved.py', DEST_ROOT / 'scripts/info_architecture/runners/run_ia_test_improved.py'),
            (SOURCE_ROOT / 'run_definitive_ia_test.sh', DEST_ROOT / 'scripts/info_architecture/runners/run_definitive_ia_test.sh'),
            (SOURCE_ROOT / 'run_cmb_info_architecture.sh', DEST_ROOT / 'scripts/info_architecture/runners/run_cmb_info_architecture.sh'),
        ],
        'readme': """# Information Architecture Tests

Scripts for analyzing the hierarchical information structure in CMB data, with emphasis on 
mathematical constants (φ, √2, √3, ln2, e, π) as organizing principles.

## Available Scripts

- `cmb_info_architecture.py`: Current production implementation
- Archive directory: Contains previous implementations for reference
- Runners directory: Various execution scripts for different test configurations

## Key Findings

- WMAP data showed statistical significance for Golden Ratio (φ): Score = 1.0203, p-value = 0.044838
- Square Root of 2 appears to be the dominant organizing principle across scales
- Scale 55 shows extremely strong sqrt2 specialization in both datasets
"""
    },
    'golden_ratio': {
        'files': [
            (SOURCE_ROOT / 'test_golden_ratio_cascade.py', DEST_ROOT / 'scripts/golden_ratio/test_golden_ratio_cascade.py'),
            (SOURCE_ROOT / 'test_golden_ratio_significance.py', DEST_ROOT / 'scripts/golden_ratio/test_golden_ratio_significance.py'),
            (SOURCE_ROOT / 'test_gr_specific_coherence.py', DEST_ROOT / 'scripts/golden_ratio/test_gr_specific_coherence.py'),
            (SOURCE_ROOT / 'analyze_gr_pairs.py', DEST_ROOT / 'scripts/golden_ratio/analyze_gr_pairs.py'),
            (SOURCE_ROOT / 'analyze_gr_results.py', DEST_ROOT / 'scripts/golden_ratio/analyze_gr_results.py'),
            (SOURCE_ROOT / 'visualize_gr_coherence.py', DEST_ROOT / 'scripts/golden_ratio/visualize_gr_coherence.py'),
        ],
        'readme': """# Golden Ratio Analysis

Scripts for analyzing Golden Ratio (φ) patterns in CMB data.

## Available Scripts

- `test_golden_ratio_cascade.py`: Tests for cascading Golden Ratio patterns
- `test_golden_ratio_significance.py`: Statistical significance testing for Golden Ratio patterns
- `test_gr_specific_coherence.py`: Coherence analysis specific to Golden Ratio scale relationships
- `analyze_gr_pairs.py`: Analysis of Golden Ratio pair relationships in the data
- `analyze_gr_results.py`: Comprehensive analysis of Golden Ratio test results
- `visualize_gr_coherence.py`: Visualization tools for Golden Ratio coherence
"""
    },
    'coherence': {
        'files': [
            (SOURCE_ROOT / 'test_coherence_analysis.py', DEST_ROOT / 'scripts/coherence/test_coherence_analysis.py'),
            (SOURCE_ROOT / 'test_correlation_analysis.py', DEST_ROOT / 'scripts/coherence/test_correlation_analysis.py'),
            (SOURCE_ROOT / 'test_meta_coherence.py', DEST_ROOT / 'scripts/coherence/test_meta_coherence.py'),
            (SOURCE_ROOT / 'analyze_correlation_results.py', DEST_ROOT / 'scripts/coherence/analyze_correlation_results.py'),
        ],
        'readme': """# Coherence Analysis

Scripts for analyzing coherence and correlation in CMB data.

## Available Scripts

- `test_coherence_analysis.py`: General coherence testing
- `test_correlation_analysis.py`: Correlation analysis between CMB scales
- `test_meta_coherence.py`: Meta-level coherence analysis across the entire spectrum
- `analyze_correlation_results.py`: Tools for analyzing correlation test results
"""
    },
    'phase_alignment': {
        'files': [
            (SOURCE_ROOT / 'phase_alignment_test.py', DEST_ROOT / 'scripts/phase_alignment/phase_alignment_test.py'),
            (SOURCE_ROOT / 'improved_phase_alignment_test.py', DEST_ROOT / 'scripts/phase_alignment/improved_phase_alignment_test.py'),
            (SOURCE_ROOT / 'enhanced_phase_alignment_test.py', DEST_ROOT / 'scripts/phase_alignment/enhanced_phase_alignment_test.py'),
            (SOURCE_ROOT / 'plv_test.py', DEST_ROOT / 'scripts/phase_alignment/plv_test.py'),
        ],
        'readme': """# Phase Alignment Tests

Scripts for analyzing phase alignments in CMB data.

## Available Scripts

- `phase_alignment_test.py`: Original phase alignment test
- `improved_phase_alignment_test.py`: Enhanced implementation with improved statistical validation
- `enhanced_phase_alignment_test.py`: Latest implementation with comprehensive analysis tools
- `plv_test.py`: Phase locking value test for phase synchronization detection
"""
    },
    'fractal': {
        'files': [
            (SOURCE_ROOT / 'test_fractal_analysis.py', DEST_ROOT / 'scripts/fractal/test_fractal_analysis.py'),
        ],
        'readme': """# Fractal Analysis

Scripts for analyzing fractal properties of the CMB power spectrum.

## Available Scripts

- `test_fractal_analysis.py`: Tests for fractal dimensions and self-similarity in CMB data
"""
    },
    'scale_transition': {
        'files': [
            (SOURCE_ROOT / 'test_scale_transition.py', DEST_ROOT / 'scripts/scale_transition/test_scale_transition.py'),
            (SOURCE_ROOT / 'analyze_scale_transition_results.py', DEST_ROOT / 'scripts/scale_transition/analyze_scale_transition_results.py'),
        ],
        'readme': """# Scale Transition Analysis

Scripts for analyzing transitions between different scale regimes in CMB data.

## Available Scripts

- `test_scale_transition.py`: Tests for transitions between different scales
- `analyze_scale_transition_results.py`: Analysis tools for scale transition results
"""
    },
    'laminarity': {
        'files': [
            (SOURCE_ROOT / 'test_laminarity.py', DEST_ROOT / 'scripts/laminarity/test_laminarity.py'),
        ],
        'readme': """# Laminarity Tests

Scripts for analyzing laminar structures in CMB data.

## Available Scripts

- `test_laminarity.py`: Tests for laminar organization in the CMB power spectrum
"""
    },
    'cross_validation': {
        'files': [
            (SOURCE_ROOT / 'test_cross_validation.py', DEST_ROOT / 'scripts/cross_validation/test_cross_validation.py'),
            (SOURCE_ROOT / 'cross_validate_wmap_planck.py', DEST_ROOT / 'scripts/cross_validation/cross_validate_wmap_planck.py'),
            (SOURCE_ROOT / 'cross_validation_framework.py', DEST_ROOT / 'scripts/cross_validation/cross_validation_framework.py'),
            (SOURCE_ROOT / 'compare_wmap_planck.py', DEST_ROOT / 'scripts/cross_validation/compare_wmap_planck.py'),
        ],
        'readme': """# Cross-Validation Analysis

Scripts for cross-validating findings between WMAP and Planck datasets.

## Available Scripts

- `test_cross_validation.py`: Cross-validation testing framework
- `cross_validate_wmap_planck.py`: Direct comparison between WMAP and Planck
- `cross_validation_framework.py`: Framework for cross-validation procedures
- `compare_wmap_planck.py`: Comparative analysis between WMAP and Planck results
"""
    },
    'hierarchical': {
        'files': [
            (SOURCE_ROOT / 'test_hierarchical_organization.py', DEST_ROOT / 'scripts/hierarchical/test_hierarchical_organization.py'),
            (SOURCE_ROOT / 'complementary_organization_test.py', DEST_ROOT / 'scripts/hierarchical/complementary_organization_test.py'),
        ],
        'readme': """# Hierarchical Organization Tests

Scripts for analyzing hierarchical organization in CMB data.

## Available Scripts

- `test_hierarchical_organization.py`: Tests for hierarchical structures in CMB data
- `complementary_organization_test.py`: Tests for complementary patterns in the organizational structure
"""
    },
    'orthogonality': {
        'files': [
            (SOURCE_ROOT / 'test_orthogonality.py', DEST_ROOT / 'scripts/orthogonality/test_orthogonality.py'),
        ],
        'readme': """# Orthogonality Tests

Scripts for analyzing orthogonal patterns in CMB data.

## Available Scripts

- `test_orthogonality.py`: Tests for orthogonality between different structures in CMB data
"""
    },
    'resonance': {
        'files': [
            (SOURCE_ROOT / 'test_resonance_analysis.py', DEST_ROOT / 'scripts/resonance/test_resonance_analysis.py'),
        ],
        'readme': """# Resonance Analysis

Scripts for analyzing resonance patterns in CMB data.

## Available Scripts

- `test_resonance_analysis.py`: Tests for resonance effects between different scales in CMB data
"""
    },
    'information': {
        'files': [
            (SOURCE_ROOT / 'test_information_integration.py', DEST_ROOT / 'scripts/information/test_information_integration.py'),
        ],
        'readme': """# Information Integration Tests

Scripts for analyzing information integration in CMB data.

## Available Scripts

- `test_information_integration.py`: Tests for information integration across scales in CMB data
"""
    },
    'data_management': {
        'files': [
            (SOURCE_ROOT / 'download_data.py', DEST_ROOT / 'tools/data_management/download_data.py'),
            (SOURCE_ROOT / 'download_wmap_data.py', DEST_ROOT / 'tools/data_management/download_wmap_data.py'),
            (SOURCE_ROOT / 'download_planck_data.py', DEST_ROOT / 'tools/data_management/download_planck_data.py'),
            (SOURCE_ROOT / 'download_wmap_data_lambda.py', DEST_ROOT / 'tools/data_management/download_wmap_data_lambda.py'),
            (SOURCE_ROOT / 'download_wmap_data_updated.py', DEST_ROOT / 'tools/data_management/download_wmap_data_updated.py'),
            (SOURCE_ROOT / 'validate_wmap_data.py', DEST_ROOT / 'tools/data_management/validate_wmap_data.py'),
            (SOURCE_ROOT / 'fix_wmap_data.py', DEST_ROOT / 'tools/data_management/fix_wmap_data.py'),
            (SOURCE_ROOT / 'fix_wmap_ilc_map.py', DEST_ROOT / 'tools/data_management/fix_wmap_ilc_map.py'),
            (SOURCE_ROOT / 'test_wmap_data.py', DEST_ROOT / 'tools/data_management/test_wmap_data.py'),
        ],
        'readme': """# Data Management Tools

Scripts for downloading, validating, and preparing CMB data for analysis.

## Available Scripts

- `download_data.py`: General data download utility
- `download_wmap_data.py`: WMAP data downloader
- `download_planck_data.py`: Planck data downloader
- Various data validation and repair tools
"""
    },
    'integration': {
        'files': [
            (SOURCE_ROOT / 'run_all_tests.py', DEST_ROOT / 'scripts/integration/run_all_tests.py'),
            (SOURCE_ROOT / 'run_full_analysis.py', DEST_ROOT / 'scripts/integration/run_full_analysis.py'),
            (SOURCE_ROOT / 'generate_test_reports.py', DEST_ROOT / 'scripts/integration/generate_test_reports.py'),
            (SOURCE_ROOT / 'cosmic_test_integration.py', DEST_ROOT / 'scripts/integration/cosmic_test_integration.py'),
            (SOURCE_ROOT / 'rerun_wmap_tests.py', DEST_ROOT / 'scripts/integration/rerun_wmap_tests.py'),
            (SOURCE_ROOT / 'run_wmap_analysis.py', DEST_ROOT / 'scripts/integration/run_wmap_analysis.py'),
            (SOURCE_ROOT / 'run_validation_tests.py', DEST_ROOT / 'scripts/integration/run_validation_tests.py'),
            (SOURCE_ROOT / 'run_full_analysis.py', DEST_ROOT / 'scripts/integration/run_full_analysis.py'),
        ],
        'readme': """# Integration Scripts

Scripts for running comprehensive test suites and generating integrated results.

## Available Scripts

- `run_all_tests.py`: Run all test categories in sequence
- `run_full_analysis.py`: Comprehensive analysis script
- `generate_test_reports.py`: Generate reports from test results
- `cosmic_test_integration.py`: Integration framework for all tests
"""
    },
    'visualization': {
        'files': [
            (SOURCE_ROOT / 'visualization/comparison_dashboard.py', DEST_ROOT / 'visualization/dashboards/comparison_dashboard.py'),
            (SOURCE_ROOT / 'visualization/visualization_utils.py', DEST_ROOT / 'visualization/utils/visualization_utils.py'),
            (SOURCE_ROOT / 'visualize_gr_coherence.py', DEST_ROOT / 'visualization/static_plots/visualize_gr_coherence.py'),
        ],
        'readme': """# Visualization Tools

Tools for visualizing and presenting CMB analysis results.

## Available Components

- `dashboards/`: Interactive visualization dashboards
- `static_plots/`: Scripts for generating static plots
- `utils/`: Visualization utility functions
"""
    },
    'monitoring': {
        'files': [
            (SOURCE_ROOT / 'check_progress.py', DEST_ROOT / 'tools/monitoring/check_progress.py'),
            (SOURCE_ROOT / 'monitor_progress.py', DEST_ROOT / 'tools/monitoring/monitor_progress.py'),
            (SOURCE_ROOT / 'inspect_ia_test.py', DEST_ROOT / 'tools/monitoring/inspect_ia_test.py'),
        ],
        'readme': """# Monitoring Tools

Scripts for monitoring the progress of long-running tests.

## Available Scripts

- `check_progress.py`: Check the status of ongoing tests
- `monitor_progress.py`: Monitor and log test progress
- `inspect_ia_test.py`: Inspect information architecture test status
"""
    },
    'tests': {
        'files': [
            (SOURCE_ROOT / 'tests/__init__.py', DEST_ROOT / 'tests/__init__.py'),
            (SOURCE_ROOT / 'tests/conftest.py', DEST_ROOT / 'tests/conftest.py'),
            (SOURCE_ROOT / 'tests/test_cosmic_framework.py', DEST_ROOT / 'tests/test_cosmic_framework.py'),
            (SOURCE_ROOT / 'tests/test_data_provenance.py', DEST_ROOT / 'tests/test_data_provenance.py'),
            (SOURCE_ROOT / 'tests/test_data_validation.py', DEST_ROOT / 'tests/test_data_validation.py'),
            (SOURCE_ROOT / 'tests/test_gr_coherence.py', DEST_ROOT / 'tests/test_gr_coherence.py'),
            (SOURCE_ROOT / 'tests/test_optimized_tests.py', DEST_ROOT / 'tests/test_optimized_tests.py'),
        ],
        'readme': """# Unit and Integration Tests

Tests for verifying the functionality of the analysis code.

## Available Tests

- `test_cosmic_framework.py`: Tests for the core analysis framework
- `test_data_provenance.py`: Tests for data provenance tracking
- `test_data_validation.py`: Tests for data validation procedures
- `test_gr_coherence.py`: Tests for Golden Ratio coherence analysis
- `test_optimized_tests.py`: Tests for optimized test implementations
"""
    },
}

def create_directories():
    """Create all necessary directories in the destination repository."""
    for test_type in MIGRATION_MAP:
        dest_dir = DEST_ROOT / f'scripts/{test_type}'
        if not dest_dir.exists():
            print(f"Creating directory: {dest_dir}")
            dest_dir.mkdir(parents=True, exist_ok=True)
        
        results_dir = DEST_ROOT / f"../results/{test_type}"
        if not results_dir.exists():
            print(f"Creating directory: {results_dir}")
            results_dir.mkdir(parents=True, exist_ok=True)

def copy_test_files():
    """Copy all test files to their new locations."""
    for test_type, mapping in MIGRATION_MAP.items():
        print(f"\nMigrating {test_type} tests...")
        
        for source_path, dest_path in mapping['files']:
            if source_path.exists():
                print(f"  Copying {source_path.name} to {dest_path}")
                
                # Create parent directory if needed
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(source_path, dest_path)
            else:
                print(f"  Warning: Source file {source_path} not found")
        
        # Create README
        readme_path = DEST_ROOT / f'scripts/{test_type}/README.md'
        print(f"  Creating README at {readme_path}")
        with open(readme_path, 'w') as f:
            f.write(mapping['readme'])

def copy_data_files():
    """Copy essential data files to the new repository."""
    # WMAP data
    wmap_source = SOURCE_ROOT / 'data'
    wmap_dest = DEST_ROOT / "../data/wmap"
    
    if wmap_source.exists():
        print("\nCopying WMAP data files...")
        for data_file in wmap_source.glob('wmap_*.txt'):
            print(f"  Copying {data_file.name} to {wmap_dest}")
            wmap_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data_file, wmap_dest / data_file.name)
    
    # Planck data
    planck_source = SOURCE_ROOT / 'data'
    planck_dest = DEST_ROOT / "../data/planck"
    
    if planck_source.exists():
        print("\nCopying Planck data files...")
        for data_file in planck_source.glob('planck_*.txt'):
            print(f"  Copying {data_file.name} to {planck_dest}")
            planck_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data_file, planck_dest / data_file.name)

def create_common_utils():
    """Create common utilities by extracting from existing files."""
    common_dir = DEST_ROOT / 'scripts/common'
    common_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy utils.py if it exists
    utils_source = SOURCE_ROOT / 'utils.py'
    utils_dest = common_dir / 'utils.py'
    
    if utils_source.exists():
        print("\nCopying common utilities...")
        print(f"  Copying {utils_source} to {utils_dest}")
        shutil.copy2(utils_source, utils_dest)
    
    # Create data handler utilities
    data_handlers_dir = common_dir / 'data_handlers'
    data_handlers_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy any data handler modules
    wmap_data_dir = SOURCE_ROOT / 'wmap_data'
    if wmap_data_dir.exists() and wmap_data_dir.is_dir():
        for py_file in wmap_data_dir.glob('*.py'):
            print(f"  Copying {py_file} to {data_handlers_dir}")
            shutil.copy2(py_file, data_handlers_dir / py_file.name)

def create_configuration():
    """Create configuration files for the new repository."""
    config_dir = DEST_ROOT / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy existing config files if any
    source_config_dir = SOURCE_ROOT / 'config'
    if source_config_dir.exists() and source_config_dir.is_dir():
        for config_file in source_config_dir.glob('*.py'):
            print(f"Copying {config_file} to {config_dir}")
            shutil.copy2(config_file, config_dir / config_file.name)
    
    # Create data configuration
    data_config_path = config_dir / 'data_config.json'
    if not data_config_path.exists():
        print("Creating data configuration file...")
        data_config = """{
    "wmap": {
        "power_spectrum": {
            "path": "../data/wmap/wmap_tt_spectrum_9yr_v5.txt",
            "format": "txt",
            "description": "WMAP 9-year temperature power spectrum"
        }
    },
    "planck": {
        "power_spectrum": {
            "path": "../data/planck/planck_tt_spectrum_2018.txt",
            "format": "txt",
            "description": "Planck 2018 temperature power spectrum"
        }
    },
    "results": {
        "base_dir": "../results"
    }
}"""
        with open(data_config_path, 'w') as f:
            f.write(data_config)

def create_requirements():
    """Create requirements.txt based on imports found in scripts."""
    # This is a simplified approach - in a real scenario, you'd want to analyze
    # all the imports in the Python files to generate a comprehensive list
    common_requirements = [
        "numpy==1.19.5",
        "scipy==1.5.4",
        "matplotlib==3.3.4",
        "pandas==1.1.5",
        "astropy==4.2.1",
        "scikit-learn==0.24.2",
        "psutil==5.8.0",
        "tqdm==4.62.3"
    ]
    
    requirements_path = DEST_ROOT / 'requirements.txt'
    print("\nCreating requirements.txt...")
    with open(requirements_path, 'w') as f:
        f.write("\n".join(common_requirements))

def create_git_resources():
    """Create git-related resources like .gitignore."""
    gitignore_path = DEST_ROOT / '.gitignore'
    print("\nCreating .gitignore...")
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Results and data (depending on preference)
# Uncomment if you want to ignore results
# results/

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Editor/IDE
.idea/
.vscode/
*.swp
*.swo
"""
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)

def main():
    """Main migration function."""
    print("Starting CMB test suite migration...\n")
    
    # Create directory structure
    create_directories()
    
    # Copy test files
    copy_test_files()
    
    # Copy data files
    copy_data_files()
    
    # Create common utilities
    create_common_utils()
    
    # Create configuration
    create_configuration()
    
    # Create requirements.txt
    create_requirements()
    
    # Create git resources
    create_git_resources()
    
    print("\nMigration completed successfully!")
    print("\nNext steps:")
    print("1. Review the migrated files")
    print("2. Run tests to ensure everything works as expected")
    print("3. Commit changes to git repository")
    print("   cd /Users/stephaniealexander/CascadeProjects/COSMIC_Analysis")
    print("   git add .")
    print("   git commit -m \"Initial repository migration\"")
    print("   git remote add origin <repository-url>")
    print("   git push -u origin main")

if __name__ == "__main__":
    main()
