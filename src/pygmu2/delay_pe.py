"""
DelayPE - delays audio by a specified amount (fixed or variable).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""


import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.wavetable_pe import InterpolationMode
from pygmu2.interpolated_lookup import interpolated_lookup


class DelayPE(ProcessingElement):
    """
    A ProcessingElement that delays its input by a specified amount.
    
    Supports three modes based on the delay argument type:
    
    1. **Integer delay**: Fast path with no interpolation. The extent is
       shifted by the delay amount.
    
    2. **Float delay**: Constant fractional delay with interpolation.
       Useful for sub-sample phase alignment and precise timing.
    
    3. **PE delay**: Variable delay from another ProcessingElement,
       with interpolation. Enables effects like vibrato, chorus, and flanging.
    
    Sign convention: Positive delay values push audio later in time
    (look into the past). This is consistent across all modes.
    
    Args:
        source: Input ProcessingElement
        delay: Delay amount - int (samples), float (fractional samples),
               or ProcessingElement (variable delay)
        interpolation: Interpolation method for float/PE delays
                      (default: LINEAR). Ignored for integer delays.
    
    Example:
        # Integer delay (fast path, no interpolation)
        delayed_stream = DelayPE(source_stream, delay=44100)  # 1 second at 44.1kHz
        
        # Fractional delay (sub-sample precision)
        aligned_stream = DelayPE(source_stream, delay=10.5)
        
        # Variable delay (vibrato effect)
        lfo_stream = SinePE(frequency=5.0, amplitude=50.0)
        delay_mod_stream = MixPE(ConstantPE(100.0), lfo_stream)  # 100 ± 50 samples
        vibrato = DelayPE(source, delay=delay_mod)
        
        # Simple echo effect
        delayed = DelayPE(source, delay=22050)  # 0.5 second delay
        echo = MixPE(source, GainPE(delayed, 0.5))
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        delay: int | float | ProcessingElement,
        interpolation: InterpolationMode = InterpolationMode.LINEAR,
    ):
        self._source = source
        self._delay = delay
        self._interpolation = interpolation
        
        # Determine delay mode
        if isinstance(delay, ProcessingElement):
            self._mode = "pe"
        elif isinstance(delay, float) and not delay.is_integer():
            self._mode = "float"
        else:
            # int or float that is a whole number
            self._mode = "int"
            self._delay = int(delay)
    
    @property
    def source(self) -> ProcessingElement:
        """The input ProcessingElement."""
        return self._source
    
    @property
    def delay(self) -> int | float | ProcessingElement:
        """The delay value or PE."""
        return self._delay
    
    @property
    def interpolation(self) -> InterpolationMode:
        """The interpolation method (relevant for float/PE delays)."""
        return self._interpolation
    
    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs (source, and delay if it's a PE)."""
        if self._mode == "pe":
            return [self._source, self._delay]
        return [self._source]
    
    def is_pure(self) -> bool:
        """DelayPE is pure - it's a stateless operation."""
        return True
    
    def channel_count(self) -> int | None:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """
        Return the delayed extent.
        
        For integer/float delays: extent is shifted by the delay amount.
        For PE delays: extent matches the delay PE's extent.
        """
        if self._mode == "pe":
            # Variable delay: output exists where both source and delay control exist
            return self._source.extent().intersection(self._delay.extent())
        else:
            # Fixed delay: shift source extent
            source_extent = self._source.extent()
            delay_amount = int(self._delay) if self._mode == "int" else self._delay
            
            new_start = None if source_extent.start is None else source_extent.start + delay_amount
            new_end = None if source_extent.end is None else source_extent.end + delay_amount
            
            # For float delay, round to int for extent (extent is always integer-based)
            if self._mode == "float":
                new_start = None if new_start is None else int(np.floor(new_start))
                new_end = None if new_end is None else int(np.ceil(new_end))
            
            return Extent(new_start, new_end)
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render delayed audio.
        
        Args:
            start: Starting sample index (in output time)
            duration: Number of samples to render (> 0)
        
        Returns:
            Snippet containing the delayed audio
        """
        if self._mode == "int":
            return self._render_int(start, duration)
        elif self._mode == "float":
            return self._render_float(start, duration)
        else:  # pe
            return self._render_pe(start, duration)
    
    def _render_int(self, start: int, duration: int) -> Snippet:
        """Fast path for integer delay - no interpolation."""
        # Delay "looks into the past": at output time t, we read source time (t - delay)
        # This means we can request negative source times, which the source will handle
        # appropriately (e.g., IdentityPE returns negative values, ArrayPE returns zeros)
        source_start = start - self._delay
        source_snippet = self._source.render(source_start, duration)
        return Snippet(start, source_snippet.data)
    
    def _render_float(self, start: int, duration: int) -> Snippet:
        """Render with constant fractional delay using interpolation."""
        # Compute lookup indices
        t = np.arange(start, start + duration, dtype=np.float64)
        indices = t - self._delay
        
        return self._render_interpolated(start, duration, indices)
    
    def _render_pe(self, start: int, duration: int) -> Snippet:
        """Render with variable delay from PE using interpolation."""
        # Get delay values (assumed mono)
        delay_snippet = self._delay.render(start, duration)
        delay_values = delay_snippet.data[:, 0].astype(np.float64)
        
        # Compute lookup indices: t - delay[t]
        t = np.arange(start, start + duration, dtype=np.float64)
        indices = t - delay_values
        
        return self._render_interpolated(start, duration, indices)
    
    def _render_interpolated(
        self, start: int, duration: int, indices: np.ndarray
    ) -> Snippet:
        """
        Render using interpolated lookup at the given indices.
        
        Args:
            start: Output start position
            duration: Number of samples
            indices: Fractional source indices to look up
        
        Returns:
            Interpolated snippet
        """
        # Get source extent for out-of-bounds detection
        source_extent = self._source.extent()
        src_start = source_extent.start
        src_end = source_extent.end
        
        has_finite_extent = src_start is not None and src_end is not None
        
        # Track out-of-bounds indices
        if has_finite_extent:
            oob_mask = (indices < src_start) | (indices >= src_end)
        else:
            oob_mask = None
        
        return interpolated_lookup(
            self._source,
            start,
            indices,
            self._interpolation,
            out_of_bounds_mask=oob_mask,
            out_dtype=np.float32,
        )
    
    def __repr__(self) -> str:
        if self._mode == "pe":
            delay_str = f"{self._delay.__class__.__name__}(...)"
            return (
                f"DelayPE(source={self._source.__class__.__name__}, "
                f"delay={delay_str}, interpolation={self._interpolation.value})"
            )
        elif self._mode == "float":
            return (
                f"DelayPE(source={self._source.__class__.__name__}, "
                f"delay={self._delay}, interpolation={self._interpolation.value})"
            )
        else:
            return f"DelayPE(source={self._source.__class__.__name__}, delay={self._delay})"
