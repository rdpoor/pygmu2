"""
SlicePE - extract a region from a source and optionally taper edges.

SlicePE is a convenience PE for the common workflow:
- extract a short region from a longer source (e.g., a snare hit)
- shift the extracted region so it starts at time 0
- optionally apply a short fade-in and/or fade-out to avoid clicks

Conceptually, SlicePE is equivalent to (in samples):
    CropPE(source, Extent(start, start + duration)) -> DelayPE(..., -start) -> GainPE(..., envelope)

where the envelope is 1.0 in the middle and ramps at the edges.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from typing import Optional

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
        fade_in_samples: fade-in length in samples (optional)
        fade_in_seconds: fade-in length in seconds (optional)
        fade_out_samples: fade-out length in samples (optional)
        fade_out_seconds: fade-out length in seconds (optional)
    """

    def __init__(
        self,
        source: ProcessingElement,
        start: int,
        duration: int,
        *,
        fade_in_samples: Optional[int] = None,
        fade_in_seconds: Optional[float] = None,
        fade_out_samples: Optional[int] = None,
        fade_out_seconds: Optional[float] = None,
    ):
        self._source = source
        self._start = int(start)
        self._duration = int(duration)
        self._fade_in_samples = fade_in_samples
        self._fade_in_seconds = fade_in_seconds
        self._fade_out_samples = fade_out_samples
        self._fade_out_seconds = fade_out_seconds

        if self._duration < 0:
            raise ValueError(f"duration must be >= 0, got {duration}")

        crop = CropPE(self._source, Extent(self._start, self._start + self._duration))
        self._base = DelayPE(crop, delay=-self._start)

        # Resolved after configure()
        self._fade_in: int = 0
        self._fade_out: int = 0
        self._out: ProcessingElement = self._base

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

    def configure(self, sample_rate: int) -> None:
        """
        Configure and resolve fade durations (seconds -> samples if provided).
        """
        # `_time_to_samples()` consults `self.sample_rate`, so set it early.
        self._sample_rate = sample_rate

        # Resolve fades (None/None -> 0 handled by _time_to_samples).
        self._fade_in = self._time_to_samples(
            samples=self._fade_in_samples,
            seconds=self._fade_in_seconds,
            name="fade_in",
        )
        self._fade_out = self._time_to_samples(
            samples=self._fade_out_samples,
            seconds=self._fade_out_seconds,
            name="fade_out",
        )

        # Build envelope if needed
        if self._duration > 0 and (self._fade_in > 0 or self._fade_out > 0):
            env = np.ones((self._duration,), dtype=np.float32)

            fi = min(self._fade_in, self._duration)
            fo = min(self._fade_out, self._duration)

            if fi > 0:
                ramp = (np.arange(fi, dtype=np.float32) + 1.0) / float(fi)
                env[:fi] = np.minimum(env[:fi], ramp)

            if fo > 0:
                ramp = 1.0 - (np.arange(fo, dtype=np.float32) + 1.0) / float(fo)
                env[-fo:] = np.minimum(env[-fo:], ramp)

            env_pe = ArrayPE(env)
            self._out = GainPE(self._base, gain=env_pe)
        else:
            self._out = self._base

        # Configure the composed graph via the standard traversal.
        super().configure(sample_rate)

        # Extent may change depending on graph composition
        self._cached_extent = None

    def inputs(self) -> list[ProcessingElement]:
        # Delegate to the composed output graph so configure() reaches all internals.
        return [self._out]

    def is_pure(self) -> bool:
        return self._out.is_pure()

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
            f"fade_in_samples={self._fade_in}, fade_out_samples={self._fade_out})"
        )

