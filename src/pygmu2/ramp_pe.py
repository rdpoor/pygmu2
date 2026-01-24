"""
RampPE - a source that outputs a linear ramp.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.processing_element import SourcePE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class RampPE(SourcePE):
    """
    A SourcePE that outputs a linear ramp from start_value to end_value.
    
    The ramp spans a fixed number of samples. Requests outside this
    range return zeros.
    
    Useful for:
    - Envelope generation (attack/decay ramps)
    - Parameter automation
    - Crossfades
    - Testing
    
    Args:
        start_value: Value at the beginning of the ramp
        end_value: Value at the end of the ramp
        duration: Length of the ramp in samples
        channels: Number of output channels (default: 1)
    
    Example:
        # Fade in over 1 second at 44100 Hz
        fade_in = RampPE(0.0, 1.0, duration=44100)
        
        # Fade out
        fade_out = RampPE(1.0, 0.0, duration=44100)
        
        # Frequency sweep from 220 Hz to 880 Hz
        sweep = RampPE(220.0, 880.0, duration=44100)
        sine = SinePE(frequency=sweep)
    """
    
    def __init__(
        self,
        start_value: float,
        end_value: float,
        duration: int,
        channels: int = 1,
    ):
        self._start_value = start_value
        self._end_value = end_value
        self._duration = duration
        self._channels = channels
        
        # Pre-compute the full ramp for efficiency
        t = np.linspace(0, 1, duration, dtype=np.float32)
        self._ramp = self._start_value + (self._end_value - self._start_value) * t
    
    @property
    def start_value(self) -> float:
        """Value at the beginning of the ramp."""
        return self._start_value
    
    @property
    def end_value(self) -> float:
        """Value at the end of the ramp."""
        return self._end_value
    
    @property
    def ramp_duration(self) -> int:
        """Length of the ramp in samples."""
        return self._duration
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Generate ramp samples for the given range.
        
        Samples outside the ramp extent are zero-filled.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet containing ramp data (zeros outside extent)
        """
        # Initialize output with zeros
        data = np.zeros((duration, self._channels), dtype=np.float32)
        
        # Calculate overlap with ramp extent
        ramp_start = 0
        ramp_end = self._duration
        
        # Request range
        req_start = start
        req_end = start + duration
        
        # Find intersection
        overlap_start = max(ramp_start, req_start)
        overlap_end = min(ramp_end, req_end)
        
        if overlap_start < overlap_end:
            # There is overlap - copy ramp values
            ramp_slice_start = overlap_start - ramp_start
            ramp_slice_end = overlap_end - ramp_start
            
            output_slice_start = overlap_start - req_start
            output_slice_end = overlap_end - req_start
            
            ramp_values = self._ramp[ramp_slice_start:ramp_slice_end]
            data[output_slice_start:output_slice_end, :] = ramp_values.reshape(-1, 1)
        
        return Snippet(start, data)
    
    def _compute_extent(self) -> Extent:
        """Return the extent of the ramp (0 to duration)."""
        return Extent(0, self._duration)
    
    def channel_count(self) -> int:
        """Return the number of output channels."""
        return self._channels
    
    def __repr__(self) -> str:
        return (
            f"RampPE(start_value={self._start_value}, end_value={self._end_value}, "
            f"duration={self._duration}, channels={self._channels})"
        )
