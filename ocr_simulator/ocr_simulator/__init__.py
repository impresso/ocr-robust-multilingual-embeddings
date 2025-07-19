"""
OCR Simulator Library
====================

A comprehensive library for simulating OCR process with various conditions:
- Simple (clean text)
- Blackletter font
- Distorted text
- Salt and pepper noise

The library supports multiple languages and various configuration options.
"""

from .core import OCRSimulator
from .languages import LANGUAGE_CONFIGS
from .effects import apply_effects, apply_salt_and_pepper, apply_distortions

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Expose main components
__all__ = [
    'OCRSimulator',
    'LANGUAGE_CONFIGS',
    'apply_effects',
    'apply_salt_and_pepper',
    'apply_distortions'
]

# Supported conditions
SUPPORTED_CONDITIONS = [
    'simple',
    'blackletter',
    'distorted',
    'noisy'
]

# Supported languages
SUPPORTED_LANGUAGES = list(LANGUAGE_CONFIGS.keys())


def get_version():
    """Return the current version of the library."""
    return __version__


def get_supported_conditions():
    """Return list of supported conditions."""
    return SUPPORTED_CONDITIONS


def get_supported_languages():
    """Return list of supported languages."""
    return SUPPORTED_LANGUAGES


def get_default_config(condition):
    """
    Get default configuration for a specific condition.
    
    Args:
        condition (str): One of the supported conditions
        
    Returns:
        dict: Default configuration for the specified condition
    
    Raises:
        ValueError: If condition is not supported
    """
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"Condition must be one of {SUPPORTED_CONDITIONS}")

    if condition == 'simple':
        return {}

    elif condition == 'blackletter':
        return {
            'font_multiplier': 1.2
        }

    elif condition == 'distorted':
        return {
            'skew_range': (-0.06, 0.06),
            'incomplete_prob': 0.15,
            'gap_range': (1, 3),
            'text_noise_range': (-30, 30),
            'bg_noise_prob': 0.05,
            'bg_noise_range': (-10, 10)
        }

    else:  # noisy
        return {
            'dot_density': 0.0045
        }
