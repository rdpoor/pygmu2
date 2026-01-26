"""
TimeWarpPE - variable-speed resampling ("tape head") of a source signal.

TimeWarpPE treats its source like tape and advances a fractional read head
according to a rate input:

    pos[n+1] = pos[n] + rate[n]

where `rate` is measured in (source samples) / (output sample). For example:
- rate = 1.0: normal speed
- rate = 2.0: double speed
- rate = 0.5: half speed
- rate = -1.0: reverse (from the initial position)

The PE is stateful: the read head position is preserved across render calls,
and reset to 0.0 on on_start()/on_stop()/reset_state().

Out-of-bounds behavior: when the read head falls outside the source extent,
the output is forced to 0.0 for those samples.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.wavetable_pe import InterpolationMode
from pygmu2.interpolated_lookup import interpolated_lookup


class TimeWarpPE(ProcessingElement):
    """
    Resample a source at a time-varying rate.

    Args:
        source: Input signal to be time-warped
        rate: Scalar or PE. Interpreted as (source-samples)/(output-sample).
              Rate is assumed mono if provided as a PE (channel 0 is used).
        interpolation: Linear or cubic interpolation (default: LINEAR)
    """

    def __init__(
        self,
        source: ProcessingElement,
        rate: Union[float, int, ProcessingElement] = 1.0,
        interpolation: InterpolationMode = InterpolationMode.LINEAR,
    ):
        self._source = source
        self._rate = rate
        self._rate_is_pe = isinstance(rate, ProcessingElement)
        self._interpolation = interpolation

        # Stateful tape head
        self._pos: float = 0.0
        self._last_render_end: Optional[int] = None

    @property
    def source(self) -> ProcessingElement:
        return self._source

    @property
    def rate(self) -> Union[float, int, ProcessingElement]:
        return self._rate

    @property
    def interpolation(self) -> InterpolationMode:
        return self._interpolation

    def inputs(self) -> list[ProcessingElement]:
        if self._rate_is_pe:
            return [self._source, self._rate]
        return [self._source]

    def is_pure(self) -> bool:
        return False

    def channel_count(self) -> Optional[int]:
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        """
        Extent of TimeWarpPE.

        - If rate is a PE: extent matches rate extent (we produce output wherever
          the rate produces values).
        - If rate is a constant and source has finite extent, compute a finite
          output extent that covers the region where the tape head is within
          the source bounds (starting from pos=0).
        - Otherwise: infinite extent.
        """
        if self._rate_is_pe:
            return self._rate.extent()

        src = self._source.extent()
        if src.start is None or src.end is None:
            return Extent(None, None)

        src_start = float(src.start)
        src_end = float(src.end)
        r = float(self._rate)
        p0 = 0.0

        if r == 0.0:
            # Head doesn't move. Output is constant if p0 is in-bounds, else silence.
            if src_start <= p0 < src_end:
                return Extent(None, None)
            return Extent(0, 0)

        if r > 0.0:
            # n_start = smallest n where p0 + n*r >= src_start
            n_start = int(np.ceil((src_start - p0) / r)) if src_start > p0 else 0
            # n_end = smallest n where p0 + n*r >= src_end
            n_end = int(np.ceil((src_end - p0) / r))
            n_start = max(0, n_start)
            n_end = max(n_start, n_end)
            return Extent(n_start, n_end)

        # r < 0.0
        # Need n such that src_start <= p0 + n*r < src_end
        # => n > (src_end - p0) / r  (note r<0 flips inequality)
        # => n <= (src_start - p0) / r
        lower = (src_end - p0) / r
        upper = (src_start - p0) / r
        n_start = max(0, int(np.floor(lower)) + 1)
        n_end = int(np.floor(upper)) + 1
        if n_end < n_start:
            n_end = n_start
        return Extent(n_start, n_end)

    def on_start(self) -> None:
        self._reset_state()

    def on_stop(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._pos = 0.0
        self._last_render_end = None

    def _render(self, start: int, duration: int) -> Snippet:
        # Rate is mono control (channel 0). We intentionally do not attempt
        # to handle per-channel rate in this PE.
        rate_values = self._scalar_or_pe_values(self._rate, start, duration, dtype=np.float64)

        # Compute read indices for each output sample:
        # indices[0] = pos
        # indices[n] = pos + sum(rate[0:n])
        if duration == 1:
            indices = np.array([self._pos], dtype=np.float64)
        else:
            prefix = np.concatenate(([0.0], np.cumsum(rate_values[:-1], dtype=np.float64)))
            indices = self._pos + prefix

        # Update state for next render
        self._pos = float(self._pos + np.sum(rate_values, dtype=np.float64))
        self._last_render_end = start + duration

        # Out-of-bounds mask against source extent (force zeros when outside)
        src_extent = self._source.extent()
        oob_mask: Optional[np.ndarray] = None
        if src_extent.start is not None or src_extent.end is not None:
            oob = np.zeros((duration,), dtype=bool)
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
        rate_str = (
            f"{self._rate.__class__.__name__}(...)"
            if isinstance(self._rate, ProcessingElement)
            else str(self._rate)
        )
        return (
            f"TimeWarpPE(source={self._source.__class__.__name__}, "
            f"rate={rate_str}, interpolation={self._interpolation.value})"
        )

