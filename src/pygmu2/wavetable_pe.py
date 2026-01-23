"""
WavetablePE - wavetable lookup synthesis with interpolation.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from enum import Enum
from typing import Optional

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class InterpolationMode(Enum):
    """Interpolation method for fractional wavetable indices."""
    LINEAR = "linear"
    CUBIC = "cubic"


class OutOfBoundsMode(Enum):
    """Behavior when wavetable index is outside the wavetable's extent."""
    ZERO = "zero"    # Output silence (0.0)
    CLAMP = "clamp"  # Clamp to nearest valid index
    WRAP = "wrap"    # Wrap around (modulo wavetable length)


class WavetablePE(ProcessingElement):
    """
    Wavetable lookup synthesis with interpolation.
    
    Produces output according to the formula:
        s_out[t] = s_wavetable[s_indexer[t]]
    
    where s_indexer[t] can be fractional, requiring interpolation.
    
    Args:
        wavetable: PE providing the wavetable data (multi-channel supported)
        indexer: PE providing the index values (mono, can be fractional)
        interpolation: Interpolation method (default: LINEAR)
        out_of_bounds: Behavior for indices outside wavetable extent (default: ZERO)
    
    Example:
        # Classic wavetable synthesis with phase accumulator
        wavetable = WavReaderPE("sine_cycle.wav")  # One cycle of a waveform
        
        # Phase accumulator that wraps at wavetable length
        # (This would need a custom PE or combination of PEs)
        phase = PhaseAccumulatorPE(frequency=440.0, table_length=1024)
        
        # Output: interpolated wavetable lookup
        output = WavetablePE(wavetable, phase, out_of_bounds=OutOfBoundsMode.WRAP)
        
        # Granular-style random access
        random_indices = RandomPE(0, 1024)  # Hypothetical
        grains = WavetablePE(wavetable, random_indices)
    """
    
    def __init__(
        self,
        wavetable: ProcessingElement,
        indexer: ProcessingElement,
        interpolation: InterpolationMode = InterpolationMode.LINEAR,
        out_of_bounds: OutOfBoundsMode = OutOfBoundsMode.ZERO,
    ):
        self._wavetable = wavetable
        self._indexer = indexer
        self._interpolation = interpolation
        self._out_of_bounds = out_of_bounds
    
    @property
    def wavetable(self) -> ProcessingElement:
        """The wavetable PE providing sample data."""
        return self._wavetable
    
    @property
    def indexer(self) -> ProcessingElement:
        """The indexer PE providing lookup indices."""
        return self._indexer
    
    @property
    def interpolation(self) -> InterpolationMode:
        """The interpolation method used."""
        return self._interpolation
    
    @property
    def out_of_bounds(self) -> OutOfBoundsMode:
        """The out-of-bounds handling mode."""
        return self._out_of_bounds
    
    def inputs(self) -> list[ProcessingElement]:
        """Return both wavetable and indexer as inputs."""
        return [self._wavetable, self._indexer]
    
    def is_pure(self) -> bool:
        """WavetablePE is pure - interpolated lookup is stateless."""
        return True
    
    def channel_count(self) -> Optional[int]:
        """Output channels match the wavetable's channel count."""
        return self._wavetable.channel_count()
    
    def _compute_extent(self) -> Extent:
        """
        The extent matches the indexer's extent.
        
        We produce output wherever the indexer produces values.
        Whether those outputs are meaningful depends on whether
        the indices fall within the wavetable's extent.
        """
        return self._indexer.extent()
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Render wavetable output via interpolated lookup.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render
        
        Returns:
            Snippet with interpolated wavetable values
        """
        # Get index values from indexer (assumed mono)
        indexer_snippet = self._indexer.render(start, duration)
        raw_indices = indexer_snippet.data[:, 0].astype(np.float64)
        
        # Get wavetable extent
        wt_extent = self._wavetable.extent()
        wt_start = wt_extent.start
        wt_end = wt_extent.end
        
        # Check if wavetable has finite extent (needed for clamp/wrap)
        has_finite_extent = wt_start is not None and wt_end is not None
        wt_duration = (wt_end - wt_start) if has_finite_extent else None
        
        # Determine output channel count
        channels = self._wavetable.channel_count()
        if channels is None:
            channels = 1
        
        # Process indices based on out_of_bounds mode
        if self._out_of_bounds == OutOfBoundsMode.WRAP and has_finite_extent:
            indices = ((raw_indices - wt_start) % wt_duration) + wt_start
            oob_mask = None
        elif self._out_of_bounds == OutOfBoundsMode.CLAMP and has_finite_extent:
            indices = np.clip(raw_indices, wt_start, wt_end - 1)
            oob_mask = None
        else:  # ZERO mode or infinite wavetable
            indices = raw_indices
            # Track out-of-bounds indices for zeroing
            if has_finite_extent:
                oob_mask = (raw_indices < wt_start) | (raw_indices >= wt_end)
            else:
                oob_mask = None
        
        # Determine samples needed from wavetable
        if self._interpolation == InterpolationMode.CUBIC:
            margin = 2  # Need idx-1, idx, idx+1, idx+2
        else:  # LINEAR
            margin = 1  # Need idx, idx+1
        
        idx_min = np.min(indices)
        idx_max = np.max(indices)
        needed_min = int(np.floor(idx_min)) - (margin - 1)
        needed_max = int(np.ceil(idx_max)) + margin
        needed_duration = needed_max - needed_min
        
        # Render wavetable for the needed range
        # (The wavetable's render() handles zero-padding outside its extent)
        wt_snippet = self._wavetable.render(needed_min, needed_duration)
        wt_data = wt_snippet.data
        
        # Perform interpolation
        if self._interpolation == InterpolationMode.CUBIC:
            result = self._cubic_interp(indices, wt_data, needed_min, channels)
        else:
            result = self._linear_interp(indices, wt_data, needed_min, channels)
        
        # Apply out-of-bounds mask for ZERO mode
        if oob_mask is not None and np.any(oob_mask):
            result[oob_mask] = 0.0
        
        return Snippet(start, result.astype(np.float32))
    
    def _linear_interp(
        self,
        indices: np.ndarray,
        wt_data: np.ndarray,
        wt_data_start: int,
        channels: int,
    ) -> np.ndarray:
        """
        Perform linear interpolation.
        
        Args:
            indices: Fractional indices into wavetable (shape: (duration,))
            wt_data: Rendered wavetable data (shape: (wt_duration, channels))
            wt_data_start: Starting sample index of wt_data
            channels: Number of output channels
        
        Returns:
            Interpolated values (shape: (duration, channels))
        """
        duration = len(indices)
        
        # Floor indices and fractional parts
        idx_floor = np.floor(indices).astype(np.int64)
        frac = (indices - idx_floor).reshape(-1, 1)  # Shape: (duration, 1)
        
        # Convert to local indices within wt_data
        local_floor = idx_floor - wt_data_start
        local_ceil = local_floor + 1
        
        # Clip to valid array indices (wt_data already zero-padded by wavetable)
        local_floor_clipped = np.clip(local_floor, 0, len(wt_data) - 1)
        local_ceil_clipped = np.clip(local_ceil, 0, len(wt_data) - 1)
        
        # Look up values
        val_floor = wt_data[local_floor_clipped]
        val_ceil = wt_data[local_ceil_clipped]
        
        # Linear interpolation: (1-t)*v0 + t*v1
        result = (1.0 - frac) * val_floor + frac * val_ceil
        
        return result
    
    def _cubic_interp(
        self,
        indices: np.ndarray,
        wt_data: np.ndarray,
        wt_data_start: int,
        channels: int,
    ) -> np.ndarray:
        """
        Perform cubic (Catmull-Rom) interpolation.
        
        Uses 4 points: p0=floor-1, p1=floor, p2=floor+1, p3=floor+2
        Interpolates between p1 and p2.
        
        Args:
            indices: Fractional indices into wavetable (shape: (duration,))
            wt_data: Rendered wavetable data (shape: (wt_duration, channels))
            wt_data_start: Starting sample index of wt_data
            channels: Number of output channels
        
        Returns:
            Interpolated values (shape: (duration, channels))
        """
        duration = len(indices)
        
        # Floor indices and fractional parts
        idx_floor = np.floor(indices).astype(np.int64)
        t = (indices - idx_floor).reshape(-1, 1)  # Shape: (duration, 1)
        
        # Convert to local indices within wt_data
        local_p1 = idx_floor - wt_data_start
        local_p0 = local_p1 - 1
        local_p2 = local_p1 + 1
        local_p3 = local_p1 + 2
        
        # Clip to valid array indices
        max_idx = len(wt_data) - 1
        local_p0_clipped = np.clip(local_p0, 0, max_idx)
        local_p1_clipped = np.clip(local_p1, 0, max_idx)
        local_p2_clipped = np.clip(local_p2, 0, max_idx)
        local_p3_clipped = np.clip(local_p3, 0, max_idx)
        
        # Look up values
        p0 = wt_data[local_p0_clipped]
        p1 = wt_data[local_p1_clipped]
        p2 = wt_data[local_p2_clipped]
        p3 = wt_data[local_p3_clipped]
        
        # Catmull-Rom spline interpolation
        # result = 0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t^2 + (-p0+3*p1-3*p2+p3)*t^3)
        t2 = t * t
        t3 = t2 * t
        
        result = 0.5 * (
            (2.0 * p1) +
            (-p0 + p2) * t +
            (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        
        return result
    
    def __repr__(self) -> str:
        return (
            f"WavetablePE("
            f"wavetable={self._wavetable.__class__.__name__}, "
            f"indexer={self._indexer.__class__.__name__}, "
            f"interpolation={self._interpolation.value}, "
            f"out_of_bounds={self._out_of_bounds.value})"
        )
