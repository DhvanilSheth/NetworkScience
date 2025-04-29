"""
Enhanced Community Detection

This package implements an enhanced community detection algorithm
that improves upon standard Greedy Modularity Optimization.
"""

__version__ = '0.1.0'

from .enhanced_community_detection import EnhancedCommunityDetection
from .data_utils import load_network, generate_synthetic_network, save_network

__all__ = [
    'EnhancedCommunityDetection',
    'load_network',
    'generate_synthetic_network',
    'save_network'
]
