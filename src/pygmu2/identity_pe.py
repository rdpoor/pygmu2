"""
IdentityPE - outputs the sample index as the sample value.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.source_pe import SourcePE
from pygmu2.snippet import Snippet


class IdentityPE(SourcePE):
    """
    A SourcePE that outputs the sample index as the sample value.
    
    At sample index n, the output value is n (as a float).
    Useful for testing time-based operations like delays.
    
    The extent is infinite.
    
    Args:
        channels: Number of output channels (default: 1)
    
    Example:
        # Create identity signal
        identity_stream = IdentityPE()
        snippet = identity_stream.render(0, 5)
        # snippet.data = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        
        # Test delay
        identity_stream = IdentityPE()
        delayed_stream = DelayPE(identity_stream, delay=100)
        snippet = delayed_stream.render(100, 3)
        # snippet.data = [[0.0], [1.0], [2.0]]  # Values from source at 0, 1, 2
    """
    
    def __init__(self, channels: int = 1):
        self._channels = channels
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Generate sample indices as values.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet where each sample value equals its index
        """
        # Create array of indices
        indices = np.arange(start, start + duration, dtype=np.float32)
        
        # Expand to all channels (same value per channel)
        data = np.tile(indices.reshape(-1, 1), (1, self._channels))
        
        return Snippet(start, data)
    
    def channel_count(self) -> int:
        """Return the number of output channels."""
        return self._channels
    
    def __repr__(self) -> str:
        return f"IdentityPE(channels={self._channels})"
