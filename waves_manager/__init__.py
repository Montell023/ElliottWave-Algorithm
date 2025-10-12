# algostack/waves_manager/__init__.py
"""
Waves Manager for Elliott Wave algorithmic trading system.

This module provides the WavesManager class that orchestrates all 
Elliott Wave subdivision detection algorithms including:
- motive_abc: Mountain-style up-down-up patterns
- corrective_abc: Down-up-down corrective patterns  
- motive_five: Five-wave impulse structures
- corrective_five: Five-wave corrective structures
"""

from .waves_manager import WavesManager

__all__ = [
    'WavesManager',
]

__version__ = '1.0.0'

# Package metadata
__author__ = 'Montell Greef'
__description__ = 'Waves Manager for Elliott Wave subdivision detection'