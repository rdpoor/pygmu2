"""
WavetablePE - wavetable lookup synthesis with interpolation.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from enum import Enum

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.interpolated_lookup import interpolated_lookup


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
        wavetable_stream = WavReaderPE("sine_cycle.wav")  # One cycle of a waveform
        
        # Phase accumulator that wraps at wavetable length
        # (This would need a custom PE or combination of PEs)
        phase_stream = PhaseAccumulatorPE(frequency=440.0, table_length=1024)
        
        # Output: interpolated wavetable lookup
        output_stream = WavetablePE(wavetable_stream, phase_stream, out_of_bounds=OutOfBoundsMode.WRAP)
        
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
    
    def channel_count(self) -> int | None:
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
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render wavetable output via interpolated lookup.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
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
        return interpolated_lookup(
            self._wavetable,
            start,
            indices,
            self._interpolation,
            out_of_bounds_mask=oob_mask,
            out_dtype=np.float32,
        )
    
    def __repr__(self) -> str:
        return (
            f"WavetablePE("
            f"wavetable={self._wavetable.__class__.__name__}, "
            f"indexer={self._indexer.__class__.__name__}, "
            f"interpolation={self._interpolation.value}, "
            f"out_of_bounds={self._out_of_bounds.value})"
        )
