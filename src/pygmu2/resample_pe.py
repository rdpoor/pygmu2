"""
ResamplePE - pure constant-rate resampling (pitch/speed shifting).

For each output sample at absolute position t, reads the source at t * rate
using interpolation. This enables pitch shifting by playing a source at a
different speed:

  rate = 1.0  →  original pitch and speed
  rate = 2.0  →  double speed, one octave up
  rate = 0.5  →  half speed, one octave down
  rate = 2^(n/12) →  n semitones up

Unlike TimeWarpPE, ResamplePE is pure: it depends only on position, not on
any accumulated state. This means it can be rendered at any position in any
order, making it safe to use in parallel note chains (e.g. NotesPE).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.wavetable_pe import InterpolationMode
from pygmu2.interpolated_lookup import interpolated_lookup


class ResamplePE(ProcessingElement):
    """
    Pure constant-rate resampling of a source PE.

    For each output sample at absolute position t, reads the source at t * rate.
    Equivalent to playing a tape at a fixed speed ratio: faster = higher pitch.

    This PE is pure (is_pure() == True), meaning it can be rendered at any
    position in any order. The source PE should also be pure for correct results.

    Args:
        source: Input ProcessingElement to resample
        rate: Playback rate as a ratio (source samples per output sample).
              Must be > 0. Default: 1.0 (no change)
        interpolation: Sample interpolation method (default: LINEAR)

    Example:
        # Pitch-shift a WAV file up by 7 semitones (a perfect fifth)
        import math
        source = WavReaderPE("piano.wav")
        rate = math.pow(2, 7 / 12)
        higher = ResamplePE(source, rate=rate)

        # Half speed (one octave down)
        lower = ResamplePE(source, rate=0.5)
    """

    def __init__(
        self,
        source: ProcessingElement,
        rate: float = 1.0,
        interpolation: InterpolationMode = InterpolationMode.LINEAR,
    ):
        if rate <= 0.0:
            raise ValueError(f"ResamplePE rate must be > 0, got {rate}")
        self._source = source
        self._rate = float(rate)
        self._interpolation = interpolation

    @property
    def source(self) -> ProcessingElement:
        return self._source

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def interpolation(self) -> InterpolationMode:
        return self._interpolation

    def inputs(self) -> list[ProcessingElement]:
        return [self._source]

    def is_pure(self) -> bool:
        return True

    def channel_count(self) -> int | None:
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        """
        Map source extent through the rate factor.

        If source spans [s, e] in source-sample space, then the output spans
        [floor(s / rate), ceil(e / rate)] in output-sample space.
        """
        src = self._source.extent()
        start = None if src.start is None else int(np.floor(src.start / self._rate))
        end = None if src.end is None else int(np.ceil(src.end / self._rate))
        return Extent(start, end)

    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render by interpolated lookup at fractional source indices.

        Output sample n maps to source index (start + n) * rate.
        """
        indices = np.arange(start, start + duration, dtype=np.float64) * self._rate

        # Build out-of-bounds mask against source extent
        src_extent = self._source.extent()
        oob_mask: np.ndarray | None = None
        if src_extent.start is not None or src_extent.end is not None:
            oob = np.zeros(duration, dtype=bool)
            if src_extent.start is not None:
                oob |= indices < float(src_extent.start)
            if src_extent.end is not None:
                oob |= indices >= float(src_extent.end)
            if np.any(oob):
                oob_mask = oob

        return interpolated_lookup(
            self._source,
            start,
            indices,
            self._interpolation,
            out_of_bounds_mask=oob_mask,
            out_dtype=np.float32,
        )

    def __repr__(self) -> str:
        return (
            f"ResamplePE(source={self._source.__class__.__name__}, "
            f"rate={self._rate:.6g}, interpolation={self._interpolation.value})"
        )
