"""
CropPE - limits audio to a specified time window.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np
from typing import Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent, ExtendMode
from pygmu2.snippet import Snippet


class CropPE(ProcessingElement):
    """
    A ProcessingElement that limits its input to a specified extent.
    
    Samples inside the crop extent are passed through from the source.
    Behavior outside this extent is controlled by extend_mode.
    
    The crop extent can have None for either bound:
    - start=None: No lower bound (pass through from beginning)
    - end=None: No upper bound (pass through to end)
    - Both None: Pass through everything (identity operation)
    
    The output extent is the intersection of the crop extent and source extent.
    
    Args:
        source: Input ProcessingElement
        extent: The extent to crop to (supports None for open bounds)
        extend_mode: Behavior outside crop extent (default: ZERO)
                     - ZERO: Output zeros outside crop
                     - HOLD_FIRST: Hold first sample value before crop
                     - HOLD_LAST: Hold last sample value after crop
                     - HOLD_BOTH: Hold first before, last after
    
    Example:
        # Crop to samples 44100-88200 (second 1-2 at 44.1kHz)
        reader_stream = WavReaderPE("audio.wav")
        cropped_stream = CropPE(reader_stream, Extent(44100, 88200))
        
        # Sustain last value after crop ends
        from pygmu2 import ExtendMode
        sustained = CropPE(source, Extent(0, 1000), extend_mode=ExtendMode.HOLD_LAST)
        
        # Trim the beginning (start at sample 1000)
        trimmed_stream = CropPE(source_stream, Extent(1000, None))
        
        # Extract a window from an infinite source
        sine = SinePE(frequency=440.0)
        burst = CropPE(sine, Extent(0, 44100))  # 1 second burst
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        extent: Extent,
        extend_mode: ExtendMode = ExtendMode.ZERO,
    ):
        self._source = source
        self._extent = extent
        self._extend_mode = extend_mode
        # Cache first/last values when needed
        self._first_value: Optional[np.ndarray] = None
        self._last_value: Optional[np.ndarray] = None
    
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
    
    @property
    def extend_mode(self) -> ExtendMode:
        """Behavior for samples outside the crop extent."""
        return self._extend_mode
    
    def inputs(self) -> list[ProcessingElement]:
        """Return the input PE."""
        return [self._source]
    
    def is_pure(self) -> bool:
        """CropPE is pure - it's a stateless window operation."""
        return True
    
    def _get_first_value(self) -> Optional[np.ndarray]:
        """Get the first sample value from the crop window."""
        if self._first_value is not None:
            return self._first_value
        
        crop_start = self._extent.start
        if crop_start is not None:
            try:
                snippet = self._source.render(crop_start, 1)
                self._first_value = snippet.data[0:1, :].copy()
                return self._first_value
            except Exception:
                return None
        return None
    
    def _get_last_value(self) -> Optional[np.ndarray]:
        """Get the last sample value from the crop window."""
        if self._last_value is not None:
            return self._last_value
        
        crop_end = self._extent.end
        if crop_end is not None and crop_end > 0:
            try:
                snippet = self._source.render(crop_end - 1, 1)
                self._last_value = snippet.data[0:1, :].copy()
                return self._last_value
            except Exception:
                return None
        return None
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render audio, zeroing samples outside the crop extent.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
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
        
        # Determine channels
        channels = self._source.channel_count()
        if channels is None:
            # Try to determine from source's inputs
            source_inputs = self._source.inputs()
            if source_inputs:
                channels = source_inputs[0].channel_count()
            if channels is None:
                channels = 1  # Default fallback
        
        # Special case: if crop_start is None, crop extends infinitely backward
        # So requests starting before crop_end should pass through source values
        if crop_start is None and crop_end is not None and start < crop_end:
            # Request starts before crop_end - pass through source values
            # If request is entirely before crop_end, pass through completely
            if end <= crop_end:
                source_snippet = self._source.render(start, duration)
                return source_snippet
            # Otherwise, let the normal path handle it (it will get the overlapping part)
        
        # Check for no overlap: request is entirely before or after crop window
        # This happens when:
        # 1. overlap_start >= overlap_end (standard case - no overlap)
        # 2. Request ends before or at crop starts (end <= crop_start, when crop_start is not None)
        #    Note: end == crop_start means request ends exactly where crop starts, so no overlap
        # 3. Request starts at or after crop ends (start >= crop_end, when crop_end is not None)
        #    Note: start == crop_end means request starts exactly where crop ends, so no overlap
        # 4. overlap_end < overlap_start (can happen when request is entirely before crop)
        if (overlap_start >= overlap_end or
            (crop_start is not None and end <= crop_start) or
            (crop_end is not None and start >= crop_end)):
            # No overlap - handle based on extend_mode
            data = np.zeros((duration, channels), dtype=np.float32)
            
            # Check if we need to hold values
            if crop_start is not None and end <= crop_start:
                # Request is entirely before crop
                if self._extend_mode in (ExtendMode.HOLD_FIRST, ExtendMode.HOLD_BOTH):
                    first_val = self._get_first_value()
                    if first_val is not None:
                        data[:, :] = first_val
            elif crop_end is not None and start >= crop_end:
                # Request is entirely after crop
                if self._extend_mode in (ExtendMode.HOLD_LAST, ExtendMode.HOLD_BOTH):
                    last_val = self._get_last_value()
                    if last_val is not None:
                        data[:, :] = last_val
            
            return Snippet(start, data)
        
        # Get data from source for the overlapping region
        source_snippet = self._source.render(overlap_start, overlap_end - overlap_start)
        
        # Build output with initial values based on extend_mode
        channels = source_snippet.channels
        data = np.zeros((duration, channels), dtype=np.float32)
        
        # Handle before crop (if request starts before crop)
        if crop_start is not None and start < crop_start:
            if self._extend_mode in (ExtendMode.HOLD_FIRST, ExtendMode.HOLD_BOTH):
                first_val = self._get_first_value()
                if first_val is not None:
                    before_count = crop_start - start
                    data[:before_count, :] = first_val
        
        # Copy overlapping data into correct position
        output_start = overlap_start - start
        output_end = output_start + (overlap_end - overlap_start)
        data[output_start:output_end, :] = source_snippet.data
        
        # Handle after crop (if request extends after crop)
        if crop_end is not None and end > crop_end:
            if self._extend_mode in (ExtendMode.HOLD_LAST, ExtendMode.HOLD_BOTH):
                last_val = self._get_last_value()
                if last_val is not None:
                    after_start = crop_end - start
                    if after_start < duration:
                        data[after_start:, :] = last_val
        
        return Snippet(start, data)
    
    def _compute_extent(self) -> Extent:
        """
        Return the intersection of crop extent and source extent.
        """
        source_extent = self._source.extent()
        
        result = self._extent.intersection(source_extent)
        return result
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def __repr__(self) -> str:
        start_str = str(self._extent.start) if self._extent.start is not None else "None"
        end_str = str(self._extent.end) if self._extent.end is not None else "None"
        extend_str = f", extend_mode={self._extend_mode.value}" if self._extend_mode != ExtendMode.ZERO else ""
        return f"CropPE(source={self._source.__class__.__name__}, extent=Extent({start_str}, {end_str}){extend_str})"
