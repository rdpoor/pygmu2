"""
DiracPE - outputs a unit impulse (1.0 at sample 0, 0.0 elsewhere).

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.processing_element import SourcePE
from pygmu2.snippet import Snippet


class DiracPE(SourcePE):
    """
    A SourcePE that outputs a unit impulse (Dirac delta in discrete time).
    
    Outputs 1.0 at sample index 0 and 0.0 everywhere else.
    Useful for impulse response testing and analysis.
    
    The extent is infinite (though only sample 0 is non-zero).
    
    Args:
        channels: Number of output channels (default: 1)
    
    Example:
        # Create unit impulse
        impulse = DiracPE()
        snippet = impulse.render(-2, 5)
        # snippet.data = [[0.0], [0.0], [1.0], [0.0], [0.0]]
        #                  ^-2    ^-1    ^0     ^1     ^2
        
        # Test filter impulse response
        impulse = DiracPE()
        filtered = SomeFilterPE(impulse)
        response = filtered.render(0, 1000)  # Get impulse response
        
        # Delayed impulse
        impulse = DiracPE()
        delayed = DelayPE(impulse, delay=100)
        # Now impulse is at sample 100
    """
    
    def __init__(self, channels: int = 1):
        self._channels = channels
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Generate unit impulse.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate
        
        Returns:
            Snippet with 1.0 at index 0, 0.0 elsewhere
        """
        # Initialize with zeros
        data = np.zeros((duration, self._channels), dtype=np.float32)
        
        # Set impulse at sample 0 if it falls within the requested range
        if start <= 0 < start + duration:
            impulse_index = -start  # Position in the output array
            data[impulse_index, :] = 1.0
        
        return Snippet(start, data)
    
    def channel_count(self) -> int:
        """Return the number of output channels."""
        return self._channels
    
    def __repr__(self) -> str:
        return f"DiracPE(channels={self._channels})"
