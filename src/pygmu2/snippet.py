"""
Snippet class for containing audio sample data.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class Snippet:
    """
    A thin wrapper around a numpy array containing audio samples.
    
    Data layout: shape (samples, channels)
    - Mono: shape (N, 1)
    - Stereo: shape (N, 2)
    - Multi-channel: shape (N, C)
    
    Samples are floating point values, typically in range [-1.0, 1.0].
    """
    
    def __init__(self, start: int, data: NDArray[np.floating]):
        """
        Create a Snippet.
        
        Args:
            start: Starting sample index for this snippet
            data: Numpy array of shape (samples, channels) containing audio data
        
        Raises:
            ValueError: If data is not 2D or has invalid shape
        """
        if data.ndim == 1:
            # Convert mono 1D array to 2D (samples, 1)
            data = data.reshape(-1, 1)
        elif data.ndim != 2:
            raise ValueError(f"data must be 1D or 2D, got {data.ndim}D")
        
        # Normalize dtype to float32 for consistent audio buffers.
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)

        # Zero-length snippets are allowed (duration=0)
        self._start = start
        self._data = data
    
    @property
    def start(self) -> int:
        """Starting sample index of this snippet."""
        return self._start
    
    @property
    def end(self) -> int:
        """Ending sample index (exclusive) of this snippet."""
        return self._start + self._data.shape[0]
    
    @property
    def duration(self) -> int:
        """Number of samples in this snippet."""
        return self._data.shape[0]
    
    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._data.shape[1]
    
    @property
    def data(self) -> NDArray[np.floating]:
        """
        The underlying numpy array of shape (samples, channels).
        
        Note: Returns the actual array, not a copy. Treat as immutable.
        """
        return self._data
    
    @classmethod
    def from_zeros(cls, start: int, duration: int, channels: int = 1) -> Snippet:
        """
        Create a snippet filled with zeros (silence).
        
        Args:
            start: Starting sample index
            duration: Number of samples
            channels: Number of audio channels (default: 1)
        
        Returns:
            A new Snippet filled with zeros
        """
        data = np.zeros((duration, channels), dtype=np.float32)
        return cls(start, data)
    
    def __repr__(self) -> str:
        return (
            f"Snippet(start={self._start}, duration={self.duration}, "
            f"channels={self.channels})"
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Snippet):
            return NotImplemented
        return (
            self._start == other._start
            and self._data.shape == other._data.shape
            and np.allclose(self._data, other._data)
        )
