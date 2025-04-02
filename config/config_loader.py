#!/usr/bin/env python
"""
Configuration Loader for WMAP Cosmic Analysis

This module loads and validates configuration settings from YAML files,
with support for overriding defaults and merging multiple configuration sources.
"""

import os
import sys
import yaml
import logging
from copy import deepcopy

# Python 2.7 compatibility
if sys.version_info[0] < 3:
    from io import open  # For encoding support in Python 2.7

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""
    pass

def deep_update(original, update):
    """
    Recursively update a nested dictionary.
    
    Parameters:
    -----------
    original : dict
        Original dictionary to update
    update : dict
        Dictionary with updates to apply
        
    Returns:
    --------
    dict
        Updated dictionary
    """
    result = deepcopy(original)
    
    for key, value in update.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result

def load_config(config_path=None, override_dict=None):
    """
    Load configuration from a YAML file with optional overrides.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the configuration YAML file. If None, uses default config.
    override_dict : dict, optional
        Dictionary with values to override in the configuration.
        
    Returns:
    --------
    dict
        Configuration dictionary
    
    Raises:
    -------
    ConfigurationError
        If the configuration file cannot be loaded or is invalid.
    """
    # Determine the default config path relative to this file
    module_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(module_dir, 'default_config.yaml')
    
    # Load the default configuration
    try:
        with open(default_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
    except Exception as e:
        raise ConfigurationError(f"Failed to load default configuration: {str(e)}")
    
    # Load the user-specified configuration if provided
    if config_path is not None:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config is not None:
                    config = deep_update(config, user_config)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {str(e)}")
    
    # Apply override dictionary if provided
    if override_dict is not None:
        config = deep_update(config, override_dict)
    
    # Validate the configuration
    validate_config(config)
    
    return config

def validate_config(config):
    """
    Validate the configuration dictionary.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary to validate
        
    Raises:
    -------
    ConfigurationError
        If the configuration is invalid.
    """
    # Check for required sections
    required_sections = ['data', 'analysis', 'visualization', 'output']
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required configuration section: {section}")
    
    # Validate specific settings
    try:
        # Validate data paths
        if 'data' in config:
            data_config = config['data']
            if 'wmap_data_path' in data_config and not os.path.isabs(data_config['wmap_data_path']):
                # Convert relative path to absolute path based on project root
                module_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(module_dir)
                data_config['wmap_data_path'] = os.path.join(
                    project_root, data_config['wmap_data_path'])
        
        # Validate analysis parameters
        if 'analysis' in config:
            analysis_config = config['analysis']
            if 'num_simulations' in analysis_config and analysis_config['num_simulations'] < 1:
                raise ConfigurationError("num_simulations must be at least 1")
            
            if 'timeout_seconds' in analysis_config and analysis_config['timeout_seconds'] < 1:
                raise ConfigurationError("timeout_seconds must be at least 1")
        
        # Validate output directory
        if 'output' in config and 'results_dir' in config['output']:
            results_dir = config['output']['results_dir']
            if not os.path.isabs(results_dir):
                # Convert relative path to absolute path based on project root
                module_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(module_dir)
                config['output']['results_dir'] = os.path.join(project_root, results_dir)
            
            # Ensure the results directory exists
            os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {str(e)}")

def config_from_args(args):
    """
    Create a configuration dictionary from command line arguments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Configuration dictionary with overrides from command line arguments
    """
    # Start with an empty override dictionary
    override_dict = {}
    
    # Map command line arguments to configuration paths
    arg_to_config_map = {
        'data_file': 'data.wmap_data_path',
        'seed': 'analysis.random_seed',
        'phi_bias': 'analysis.golden_ratio.phi_bias',
        'visualize': 'visualization.enabled',
        'report': 'output.report.enabled',
        'parallel': 'performance.parallel',
        'n_jobs': 'performance.n_jobs',
    }
    
    # Process each argument
    for arg_name, config_path in arg_to_config_map.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            # Split the config path and set the value in the override dictionary
            parts = config_path.split('.')
            current = override_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = getattr(args, arg_name)
    
    # Handle boolean flags with 'no_' prefix
    for arg_name in ['visualize', 'report', 'parallel']:
        no_arg = f'no_{arg_name}'
        if hasattr(args, no_arg) and getattr(args, no_arg):
            # Split the config path and set the value to False
            config_path = arg_to_config_map[arg_name]
            parts = config_path.split('.')
            current = override_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = False
    
    # Handle special cases
    if hasattr(args, 'data_dir') and args.data_dir:
        override_dict.setdefault('output', {})['results_dir'] = args.data_dir
    
    if hasattr(args, 'simulated'):
        override_dict.setdefault('data', {})['use_simulated'] = args.simulated
    
    # Handle test-specific settings
    test_flags = [
        'golden_ratio', 'coherence_analysis', 'gr_specific_coherence',
        'hierarchical_organization', 'information_integration', 'scale_transition',
        'resonance_analysis', 'fractal_analysis', 'meta_coherence', 'transfer_entropy'
    ]
    
    # If any specific test is enabled, disable all tests by default
    if any(hasattr(args, flag) and getattr(args, flag) for flag in test_flags):
        override_dict.setdefault('analysis', {})['enabled_tests'] = []
    
    # Enable specific tests
    for flag in test_flags:
        if hasattr(args, flag) and getattr(args, flag):
            override_dict.setdefault('analysis', {}).setdefault('enabled_tests', []).append(flag)
    
    # If 'all' flag is set, enable all tests
    if hasattr(args, 'all') and args.all:
        override_dict.setdefault('analysis', {})['enabled_tests'] = 'all'
    
    return override_dict

def get_config(args=None, config_path=None):
    """
    Get configuration from command line arguments and/or config file.
    
    Parameters:
    -----------
    args : argparse.Namespace, optional
        Command line arguments
    config_path : str, optional
        Path to the configuration YAML file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    # Get override dictionary from command line arguments if provided
    override_dict = config_from_args(args) if args is not None else None
    
    # Load configuration with overrides
    return load_config(config_path, override_dict)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test configuration loading")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    try:
        config = get_config(args, args.config if hasattr(args, 'config') else None)
        print("Configuration loaded successfully:")
        print(yaml.dump(config, default_flow_style=False))
    except ConfigurationError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
