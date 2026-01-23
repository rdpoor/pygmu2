"""
DelayPE - delays audio by a specified amount (fixed or variable).

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from typing import Optional, Union

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.wavetable_pe import InterpolationMode


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
        delayed = DelayPE(source, delay=44100)  # 1 second at 44.1kHz
        
        # Fractional delay (sub-sample precision)
        aligned = DelayPE(source, delay=10.5)
        
        # Variable delay (vibrato effect)
        lfo = SinePE(frequency=5.0, amplitude=50.0)
        delay_mod = MixPE(ConstantPE(100.0), lfo)  # 100 ± 50 samples
        vibrato = DelayPE(source, delay=delay_mod)
        
        # Simple echo effect
        delayed = DelayPE(source, delay=22050)  # 0.5 second delay
        echo = MixPE(source, GainPE(delayed, 0.5))
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        delay: Union[int, float, ProcessingElement],
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
    def delay(self) -> Union[int, float, ProcessingElement]:
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
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """
        Return the delayed extent.
        
        For integer/float delays: extent is shifted by the delay amount.
        For PE delays: extent matches the delay PE's extent.
        """
        if self._mode == "pe":
            # Variable delay: extent comes from delay PE
            return self._delay.extent()
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
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Render delayed audio.
        
        Args:
            start: Starting sample index (in output time)
            duration: Number of samples to render
        
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
        source_snippet = self._source.render(start - self._delay, duration)
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
        
        # Determine output channel count
        channels = self._source.channel_count()
        if channels is None:
            channels = 1
        
        # Determine samples needed from source
        if self._interpolation == InterpolationMode.CUBIC:
            margin = 2
        else:
            margin = 1
        
        idx_min = np.min(indices)
        idx_max = np.max(indices)
        needed_min = int(np.floor(idx_min)) - (margin - 1)
        needed_max = int(np.ceil(idx_max)) + margin
        needed_duration = needed_max - needed_min
        
        # Render source for the needed range
        source_snippet = self._source.render(needed_min, needed_duration)
        source_data = source_snippet.data
        
        # Perform interpolation
        if self._interpolation == InterpolationMode.CUBIC:
            result = self._cubic_interp(indices, source_data, needed_min)
        else:
            result = self._linear_interp(indices, source_data, needed_min)
        
        # Apply out-of-bounds mask (zero for out-of-bounds)
        if oob_mask is not None and np.any(oob_mask):
            result[oob_mask] = 0.0
        
        return Snippet(start, result.astype(np.float32))
    
    def _linear_interp(
        self,
        indices: np.ndarray,
        source_data: np.ndarray,
        source_data_start: int,
    ) -> np.ndarray:
        """Perform linear interpolation."""
        idx_floor = np.floor(indices).astype(np.int64)
        frac = (indices - idx_floor).reshape(-1, 1)
        
        local_floor = idx_floor - source_data_start
        local_ceil = local_floor + 1
        
        local_floor_clipped = np.clip(local_floor, 0, len(source_data) - 1)
        local_ceil_clipped = np.clip(local_ceil, 0, len(source_data) - 1)
        
        val_floor = source_data[local_floor_clipped]
        val_ceil = source_data[local_ceil_clipped]
        
        return (1.0 - frac) * val_floor + frac * val_ceil
    
    def _cubic_interp(
        self,
        indices: np.ndarray,
        source_data: np.ndarray,
        source_data_start: int,
    ) -> np.ndarray:
        """Perform cubic (Catmull-Rom) interpolation."""
        idx_floor = np.floor(indices).astype(np.int64)
        t = (indices - idx_floor).reshape(-1, 1)
        
        local_p1 = idx_floor - source_data_start
        local_p0 = local_p1 - 1
        local_p2 = local_p1 + 1
        local_p3 = local_p1 + 2
        
        max_idx = len(source_data) - 1
        local_p0_clipped = np.clip(local_p0, 0, max_idx)
        local_p1_clipped = np.clip(local_p1, 0, max_idx)
        local_p2_clipped = np.clip(local_p2, 0, max_idx)
        local_p3_clipped = np.clip(local_p3, 0, max_idx)
        
        p0 = source_data[local_p0_clipped]
        p1 = source_data[local_p1_clipped]
        p2 = source_data[local_p2_clipped]
        p3 = source_data[local_p3_clipped]
        
        t2 = t * t
        t3 = t2 * t
        
        return 0.5 * (
            (2.0 * p1) +
            (-p0 + p2) * t +
            (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
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
