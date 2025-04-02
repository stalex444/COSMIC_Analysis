#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for predictive power tests.
"""

import os
import numpy as np

def load_power_spectrum(filename):
    """
    Load power spectrum data from file.
    
    Args:
        filename (str): Path to the power spectrum file
        
    Returns:
        tuple: (ell, power) arrays
    """
    try:
        data = np.loadtxt(filename)
        if data.shape[1] >= 2:
            ell = data[:, 0]
            power = data[:, 1]
            return ell, power
        else:
            raise ValueError("Invalid data format: expected at least 2 columns")
    except Exception as e:
        raise Exception("Error loading power spectrum: {}".format(str(e)))
