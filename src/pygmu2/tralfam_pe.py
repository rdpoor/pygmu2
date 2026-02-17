"""
TralfamPE - spread a finite source's spectrum randomly across its time span.

Tralfamadorians exist in all times simultaneously. This PE takes a finite
source, FFTs the whole extent, keeps magnitudes but randomizes phases, then
IFFTs. The result is cached; output extent matches the source extent.

Requires finite extent (start and end not None). Large extents will use
significant memory (full buffer in memory for FFT).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pygmu2.extent import Extent
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet


class TralfamPE(ProcessingElement):
    """
    PE that spreads a finite source's spectrum randomly across its time span.

    Renders the full source extent, FFTs (per channel), keeps magnitudes,
    replaces phases with random [0, 2π), IFFTs, and caches the result.
    Subsequent render requests return slices (or zero-padded slices) of that
    cached buffer.

    The source must have finite extent (extent().start and extent().end
    not None). Memory use is O(extent.duration * channels).

    Args:
        source: Input PE with finite extent.
        seed: Optional RNG seed for reproducible random phases (default: None).
    """

    def __init__(
        self,
        source: ProcessingElement,
        seed: Optional[int] = None,
    ):
        self._source = source
        self._seed = seed
        self._mogrified: Optional[np.ndarray] = None  # (samples, channels), float32

    def inputs(self) -> list[ProcessingElement]:
        return [self._source]

    def _compute_extent(self) -> Extent:
        return self._source.extent()

    def channel_count(self) -> Optional[int]:
        return self._source.channel_count()

    def is_pure(self) -> bool:
        return True

    def _mogrify(self) -> np.ndarray:
        """Render full source, FFT → random phases → IFFT; cache and return (samples, channels)."""
        if self._mogrified is not None:
            return self._mogrified

        ext = self.extent()
        if ext.start is None or ext.end is None:
            raise ValueError(
                f"{self.__class__.__name__} requires finite source extent; "
                f"got start={ext.start}, end={ext.end}"
            )
        n_frames = ext.duration
        if n_frames is None or n_frames <= 0:
            raise ValueError(
                f"{self.__class__.__name__} requires positive extent duration; "
                f"got duration={n_frames}"
            )

        snippet = self._source.render(ext.start, n_frames)
        frames = snippet.data  # (samples, channels), float32

        # FFT along time axis (axis=0)
        analysis = np.fft.fft(frames, axis=0)
        magnitudes = np.abs(analysis)

        rng = np.random.default_rng(self._seed)
        mangled_phases = rng.random(frames.shape) * 2.0 * np.pi
        mangled_analysis = magnitudes * np.exp(1j * mangled_phases)
        self._mogrified = np.real(np.fft.ifft(mangled_analysis, axis=0)).astype(
            np.float32
        )
        return self._mogrified

    def _render(self, start: int, duration: int) -> Snippet:
        ext = self.extent()
        if ext.start is None or ext.end is None:
            return Snippet.from_zeros(
                start, duration, self.channel_count() or 1
            )

        mogrified = self._mogrify()
        channels = mogrified.shape[1]
        req_end = start + duration

        # No overlap with extent
        if req_end <= ext.start or start >= ext.end:
            return Snippet.from_zeros(start, duration, channels)

        # Request fully inside extent: return slice
        if ext.spans(start, duration):
            local_start = start - ext.start
            slice_data = mogrified[local_start : local_start + duration].copy()
            return Snippet(start, slice_data)

        # Partial overlap: build output with zeros and mogrified slice
        out = np.zeros((duration, channels), dtype=np.float32)
        overlap_start = max(start, ext.start)
        overlap_end = min(req_end, ext.end)
        if overlap_end <= overlap_start:
            return Snippet(start, out)

        mog_start = overlap_start - ext.start
        mog_end = overlap_end - ext.start
        out_start = overlap_start - start
        out_end = overlap_end - start
        out[out_start:out_end, :] = mogrified[mog_start:mog_end, :]
        return Snippet(start, out)

    def __repr__(self) -> str:
        seed_str = f", seed={self._seed}" if self._seed is not None else ""
        return f"TralfamPE(source={self._source.__class__.__name__}{seed_str})"
