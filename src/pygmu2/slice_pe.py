"""
SlicePE - extract a region from a source and optionally taper edges.

SlicePE is a convenience PE for the common workflow:
- extract a short region from a longer source (e.g., a snare hit)
- shift the extracted region so it starts at time 0
- optionally apply a short fade-in and/or fade-out to avoid clicks

Conceptually, SlicePE is equivalent to (in samples):
    CropPE(source, start, duration) -> DelayPE(..., -start) -> GainPE(..., envelope)

where the envelope is 1.0 in the middle and ramps at the edges.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations


import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.crop_pe import CropPE
from pygmu2.delay_pe import DelayPE
from pygmu2.gain_pe import GainPE
from pygmu2.array_pe import ArrayPE


class SlicePE(ProcessingElement):
    """
    Extract a region from a source and shift it to start at time 0.

    Args:
        source: input audio PE
        start: start time (in samples) within source to extract (inclusive)
        duration: number of samples to extract
        fade_in_seconds: fade-in length in seconds (optional)
        fade_out_seconds: fade-out length in seconds (optional)
    """

    def __init__(
        self,
        source: ProcessingElement,
        start: int,
        duration: int,
        *,
        fade_in_seconds: float | None = None,
        fade_out_seconds: float | None = None,
    ):
        self._source = source
        self._start = int(start)
        self._duration = int(duration)
        self._fade_in_seconds = fade_in_seconds
        self._fade_out_seconds = fade_out_seconds

        if self._duration < 0:
            raise ValueError(f"duration must be >= 0, got {duration}")

        crop = CropPE(self._source, self._start, self._duration)
        self._base = DelayPE(crop, delay=-self._start)

        self._fade_in = (
            int(round(self._fade_in_seconds * self.sample_rate))
            if self._fade_in_seconds is not None
            else 0
        )
        self._fade_out = (
            int(round(self._fade_out_seconds * self.sample_rate))
            if self._fade_out_seconds is not None
            else 0
        )

        # Build envelope if needed
        if self._duration > 0 and (self._fade_in > 0 or self._fade_out > 0):
            env = np.ones((self._duration,), dtype=np.float32)

            fi = min(self._fade_in, self._duration)
            fo = min(self._fade_out, self._duration)

            if fi > 0:
                if fi == 1:
                    ramp = np.array([1.0], dtype=np.float32)
                else:
                    # Raised-cosine fade avoids the sharp slope change of a linear ramp.
                    phase = np.linspace(0.0, np.pi, fi, dtype=np.float32)
                    ramp = 0.5 - 0.5 * np.cos(phase)
                env[:fi] = np.minimum(env[:fi], ramp)

            if fo > 0:
                if fo == 1:
                    ramp = np.array([0.0], dtype=np.float32)
                else:
                    # Mirror the fade-in shape so fade-out begins at unity with zero slope.
                    phase = np.linspace(0.0, np.pi, fo, dtype=np.float32)
                    ramp = 0.5 + 0.5 * np.cos(phase)
                env[-fo:] = np.minimum(env[-fo:], ramp)

            env_pe = ArrayPE(env)
            self._out = GainPE(self._base, gain=env_pe)
        else:
            self._out = self._base

        # Extent may change depending on graph composition
        self._cached_extent = None

    @property
    def source(self) -> ProcessingElement:
        return self._source

    @property
    def start(self) -> int:
        return self._start

    @property
    def duration(self) -> int:
        return self._duration

    @property
    def fade_in_samples(self) -> int:
        return self._fade_in

    @property
    def fade_out_samples(self) -> int:
        return self._fade_out

    def inputs(self) -> list[ProcessingElement]:
        # Delegate to the composed output graph so configure() reaches all internals.
        return [self._out]

    @property
    def stateful(self) -> bool:  # type: ignore[override]
        return self._out.stateful

    def channel_count(self):
        return self._out.channel_count()

    def _compute_extent(self) -> Extent:
        return self._out.extent()

    def _render(self, start: int, duration: int):
        return self._out.render(start, duration)

    def __repr__(self) -> str:
        return (
            f"SlicePE(source={self._source.__class__.__name__}, "
            f"start={self._start}, duration={self._duration}, "
            f"fade_in_seconds={self._fade_in_seconds}, fade_out_seconds={self._fade_out_seconds})"
        )
