"""
CropPE - limits audio to a specified time window.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
from typing import Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class CropPE(ProcessingElement):
    """
    A ProcessingElement that limits its input to a specified extent.
    
    Samples inside the crop extent are passed through from the source.
    Samples outside this extent are zero.
    
    The crop extent can have None for either bound:
    - start=None: No lower bound (pass through from beginning)
    - end=None: No upper bound (pass through to end)
    - Both None: Pass through everything (identity operation)
    
    The output extent is the intersection of the crop extent and source extent.
    
    Args:
        source: Input ProcessingElement
        extent: The extent to crop to (supports None for open bounds)
    
    Example:
        # Crop to samples 44100-88200 (second 1-2 at 44.1kHz)
        reader = WavReaderPE("audio.wav")
        cropped = CropPE(reader, Extent(44100, 88200))
        
        # Trim the beginning (start at sample 1000)
        trimmed = CropPE(source, Extent(1000, None))
        
        # Trim the end (stop at sample 50000)
        trimmed = CropPE(source, Extent(None, 50000))
        
        # Crop to match another PE's extent
        reference = WavReaderPE("reference.wav")
        cropped = CropPE(source, reference.extent())
        
        # Extract a window from an infinite source
        sine = SinePE(frequency=440.0)
        burst = CropPE(sine, Extent(0, 44100))  # 1 second burst
    """
    
    def __init__(self, source: ProcessingElement, extent: Extent):
        self._source = source
        self._extent = extent
    
    @property
    def source(self) -> ProcessingElement:
        """The input ProcessingElement."""
        return self._source
    
    @property
    def crop_extent(self) -> Extent:
        """The extent to crop to."""
        return self._extent
    
    @property
    def start(self) -> Optional[int]:
        """First sample to include (inclusive), or None for no lower bound."""
        return self._extent.start
    
    @property
    def end(self) -> Optional[int]:
        """First sample to exclude (exclusive), or None for no upper bound."""
        return self._extent.end
    
    def inputs(self) -> list[ProcessingElement]:
        """Return the input PE."""
        return [self._source]
    
    def is_pure(self) -> bool:
        """CropPE is pure - it's a stateless window operation."""
        return True
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Render audio, zeroing samples outside the crop extent.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render
        
        Returns:
            Snippet with zeros outside the crop extent
        """
        end = start + duration
        crop_start = self._extent.start
        crop_end = self._extent.end
        
        # Calculate overlap between request and crop window
        # Handle None (infinite) bounds
        overlap_start = start if crop_start is None else max(start, crop_start)
        overlap_end = end if crop_end is None else min(end, crop_end)
        
        if overlap_start >= overlap_end:
            # No overlap - return zeros
            channels = self._source.channel_count()
            if channels is None:
                # Try to determine from source's inputs
                source_inputs = self._source.inputs()
                if source_inputs:
                    channels = source_inputs[0].channel_count()
                if channels is None:
                    channels = 1  # Default fallback
            data = np.zeros((duration, channels), dtype=np.float32)
            return Snippet(start, data)
        
        # Get data from source for the overlapping region
        source_snippet = self._source.render(overlap_start, overlap_end - overlap_start)
        
        # Build output with zeros for non-overlapping parts
        channels = source_snippet.channels
        data = np.zeros((duration, channels), dtype=np.float32)
        
        # Copy overlapping data into correct position
        output_start = overlap_start - start
        output_end = output_start + (overlap_end - overlap_start)
        data[output_start:output_end, :] = source_snippet.data
        
        return Snippet(start, data)
    
    def _compute_extent(self) -> Extent:
        """
        Return the intersection of crop extent and source extent.
        """
        source_extent = self._source.extent()
        
        result = self._extent.intersection(source_extent)
        if result is None:
            # No intersection - return crop extent (will render zeros)
            return self._extent
        return result
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def __repr__(self) -> str:
        start_str = str(self._extent.start) if self._extent.start is not None else "None"
        end_str = str(self._extent.end) if self._extent.end is not None else "None"
        return f"CropPE(source={self._source.__class__.__name__}, extent=Extent({start_str}, {end_str}))"
