"""
TransformPE - apply an arbitrary function to audio samples.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from typing import Callable, Optional

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class TransformPE(ProcessingElement):
    """
    Apply an arbitrary transformation function to audio samples.
    
    The function is applied element-wise to the source data. It should
    accept a numpy array and return an array of the same shape.
    
    Args:
        source: Input audio PE
        func: Transformation function. Should accept and return numpy arrays.
              Common examples: np.abs, np.square, np.sqrt, lambda x: x**2
        name: Optional name for the transform (used in repr)
    
    Example:
        # Absolute value (full-wave rectification)
        rectified = TransformPE(source, func=np.abs)
        
        # Convert amplitude to dB
        from pygmu2.conversions import ratio_to_db
        db_signal = TransformPE(envelope, func=ratio_to_db)
        
        # Convert MIDI pitch to frequency
        from pygmu2.conversions import pitch_to_freq
        freq_signal = TransformPE(pitch_pe, func=pitch_to_freq)
        
        # Custom transform
        squared = TransformPE(source, func=lambda x: x ** 2)
        
        # Soft clipping
        soft_clip = TransformPE(source, func=np.tanh, name="tanh")
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        func: Callable[[np.ndarray], np.ndarray],
        name: Optional[str] = None,
    ):
        self._source = source
        self._func = func
        self._name = name or getattr(func, '__name__', 'transform')
    
    @property
    def source(self) -> ProcessingElement:
        """The input audio PE."""
        return self._source
    
    @property
    def func(self) -> Callable[[np.ndarray], np.ndarray]:
        """The transformation function."""
        return self._func
    
    @property
    def name(self) -> str:
        """Name of the transform."""
        return self._name
    
    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs."""
        return [self._source]
    
    def is_pure(self) -> bool:
        """
        TransformPE is pure - the function is applied statelessly.
        """
        return True
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """Return the extent of this PE (matches source)."""
        return self._source.extent()
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render transformed audio.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
        Returns:
            Snippet containing transformed audio
        """
        # Get source data
        source_snippet = self._source.render(start, duration)
        
        # Apply transformation
        transformed = self._func(source_snippet.data.astype(np.float64))
        
        return Snippet(start, transformed.astype(np.float32))
    
    def __repr__(self) -> str:
        return (
            f"TransformPE(source={self._source.__class__.__name__}, "
            f"func={self._name})"
        )
