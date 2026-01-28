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
from pygmu2.logger import get_logger

logger = get_logger(__name__)


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
        rectified_stream = TransformPE(source_stream, func=np.abs)
        
        # Convert amplitude to dB
        from pygmu2.conversions import ratio_to_db
        db_stream = TransformPE(envelope_stream, func=ratio_to_db)
        
        # Convert MIDI pitch to frequency
        from pygmu2.conversions import pitch_to_freq
        freq_stream = TransformPE(pitch_stream, func=pitch_to_freq)
        
        # Custom transform
        squared_stream = TransformPE(source_stream, func=lambda x: x ** 2)
        
        # Soft clipping
        soft_clip_stream = TransformPE(source_stream, func=np.tanh, name="tanh")
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
        source_data = source_snippet.data.astype(np.float64)
        
        # Debug logging for pitch_to_freq transforms
        if self._name == "pitch_to_freq" and duration > 0:
            first_input = source_data[0, 0] if source_data.shape[0] > 0 else 0
            mid_idx = duration // 2
            mid_input = source_data[mid_idx, 0] if mid_idx < source_data.shape[0] else 0
            last_input = source_data[-1, 0] if source_data.shape[0] > 0 else 0
            logger.debug(
                f"TransformPE ({self._name}): start={start}, duration={duration}, "
                f"input_shape={source_data.shape}, "
                f"first_input={first_input:.2f}, mid_input={mid_input:.2f}, last_input={last_input:.2f}"
            )
        
        # Apply transformation
        transformed = self._func(source_data)
        
        # Debug logging for output
        if self._name == "pitch_to_freq" and duration > 0:
            first_output = transformed[0] if transformed.ndim == 1 else transformed[0, 0]
            mid_idx = duration // 2
            mid_output = transformed[mid_idx] if transformed.ndim == 1 else transformed[mid_idx, 0]
            last_output = transformed[-1] if transformed.ndim == 1 else transformed[-1, 0]
            logger.debug(
                f"TransformPE ({self._name}): output_shape={transformed.shape}, "
                f"first_output={first_output:.2f}, mid_output={mid_output:.2f}, last_output={last_output:.2f}"
            )
        
        # Ensure output shape matches input shape
        # Some functions (like pitch_to_freq) may return 1D arrays even for 2D input
        if source_data.ndim == 2 and transformed.ndim == 1:
            transformed = transformed.reshape(-1, source_data.shape[1])
        elif source_data.ndim == 2 and transformed.ndim == 2:
            # Ensure same number of channels
            if transformed.shape[1] != source_data.shape[1]:
                # If channels don't match, take first channel or broadcast
                if transformed.shape[1] == 1:
                    transformed = np.broadcast_to(transformed, source_data.shape)
                else:
                    transformed = transformed[:, :source_data.shape[1]]
        
        return Snippet(start, transformed.astype(np.float32))
    
    def __repr__(self) -> str:
        return (
            f"TransformPE(source={self._source.__class__.__name__}, "
            f"func={self._name})"
        )
