"""
ConstantPE - a source that outputs a constant value.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.processing_element import SourcePE
from pygmu2.snippet import Snippet


class ConstantPE(SourcePE):
    """
    A SourcePE that outputs a constant value.
    
    Useful for:
    - DC offset signals
    - Constant parameters for modulation
    - Testing and debugging
    
    The extent is infinite - it generates samples for any requested range.
    
    Args:
        value: The constant value to output
        channels: Number of output channels (default: 1)
    
    Example:
        # DC offset of 0.5
        dc_stream = ConstantPE(0.5)
        
        # Stereo constant
        stereo_dc_stream = ConstantPE(0.5, channels=2)
        
        # Use as modulation input
        base_freq_stream = ConstantPE(440.0)
        sine_stream = SinePE(frequency=base_freq_stream)
    """
    
    def __init__(self, value: float, channels: int = 1):
        self._value = value
        self._channels = channels
    
    @property
    def value(self) -> float:
        """The constant output value."""
        return self._value
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Generate constant samples for the given range.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet containing constant data
        """
        data = np.full((duration, self._channels), self._value, dtype=np.float32)
        return Snippet(start, data)
    
    # extent() uses default infinite from ProcessingElement
    
    def channel_count(self) -> int:
        """Return the number of output channels."""
        return self._channels
    
    def __repr__(self) -> str:
        return f"ConstantPE(value={self._value}, channels={self._channels})"
