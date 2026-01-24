"""
ArrayPE - outputs data from a numpy array.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
from numpy.typing import ArrayLike

from pygmu2.extent import Extent
from pygmu2.processing_element import SourcePE
from pygmu2.snippet import Snippet


class ArrayPE(SourcePE):
    """
    A SourcePE that outputs values from a provided array.
    
    This is useful for:
    - Testing with deterministic data
    - Creating control signals from pre-computed tables
    - One-shot playback of raw sample data
    
    Output is zero for time indices outside the array bounds.
    The array is assumed to be at the system sample rate.
    
    Args:
        data: Array of audio data. Can be 1D (mono) or 2D (samples, channels).
    
    Example:
        # Mono ramp
        pe = ArrayPE([0.0, 0.5, 1.0, 0.5, 0.0])
        
        # Stereo impulse
        pe = ArrayPE([[1.0, -1.0], [0.0, 0.0]])
    """
    
    def __init__(self, data: ArrayLike):
        self._data = np.asarray(data, dtype=np.float32)
        
        # Ensure 2D shape (samples, channels)
        if self._data.ndim == 1:
            self._data = self._data.reshape(-1, 1)
        elif self._data.ndim > 2:
            raise ValueError(f"ArrayPE data must be 1D or 2D, got {self._data.ndim}D")
            
        self._channels = self._data.shape[1]
        self._length = self._data.shape[0]
        
        if self._length == 0:
            raise ValueError("ArrayPE data cannot be empty")
    
    @property
    def data(self) -> np.ndarray:
        """The underlying data array."""
        return self._data
    
    def channel_count(self) -> int:
        """Return the number of output channels."""
        return self._channels
    
    def _compute_extent(self) -> Extent:
        """Return the extent of the array (0 to length)."""
        return Extent(0, self._length)
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Generate samples from the array.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet with array data (zeros outside extent)
        """
        # Calculate overlap with array extent
        data_start = 0
        data_end = self._length
        
        # Request range
        req_start = start
        req_end = start + duration
        
        # Initialize output with zeros
        out = np.zeros((duration, self._channels), dtype=np.float32)
        
        # Find intersection
        overlap_start = max(data_start, req_start)
        overlap_end = min(data_end, req_end)
        
        if overlap_start < overlap_end:
            # There is overlap - copy array values
            
            # Indices into the source array
            src_start = overlap_start
            src_end = overlap_end
            
            # Indices into the output buffer
            dst_start = overlap_start - req_start
            dst_end = overlap_end - req_start
            
            out[dst_start:dst_end] = self._data[src_start:src_end]
        
        return Snippet(start, out)
    
    def __repr__(self) -> str:
        return f"ArrayPE(shape={self._data.shape})"
