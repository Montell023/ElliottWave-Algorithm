# algostack/core/__init__.py
"""
Core components for Elliott Wave algorithmic trading system.

This module provides the foundational classes for data management,
real-time peak detection, and Fibonacci calculations used by
Elliott Wave pattern detection algorithms.
"""

from .data_manager import DataManager
from .real_time_peak_detector import RealTimePeakDetector
from .fibonacci_calculator import FibonacciCalculator

__all__ = [
    'DataManager',
    'RealTimePeakDetector', 
    'FibonacciCalculator',
]

__version__ = '1.0.0'

# Package metadata
__author__ = 'Montell Greef'
__description__ = 'Core components for Elliott Wave algorithmic trading system'