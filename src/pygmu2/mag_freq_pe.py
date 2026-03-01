"""
MagFreqPE — FFT-domain magnitude and phase processor.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from pygmu2.extent import Extent
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet


class MagFreqPE(ProcessingElement):
    """
    PE that modifies the magnitude and phase of a source in the frequency
    domain before returning it to the time domain.

    Renders the full source extent, FFTs (per channel), modifies the magnitude
    and phase with a user-supplied function, IFFTs, and caches the result.
    Subsequent render requests return slices (or zero-padded slices) of that
    cached buffer.

    The source must have finite extent (extent().start and extent().end
    not None). Memory use is O(extent.duration * channels).

    Args:
        source:         Input PE with finite extent.
        mangler:        Callable ``(magnitudes, phases) -> (magnitudes, phases)``
                        where both arrays have shape ``(fft_bins, channels)``.
        normalize_peak: Optional linear amplitude (e.g. 0.5) to scale peak to;
                        None = no normalization.
    """

    def __init__(
        self,
        source: ProcessingElement,
        mangler: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        normalize_peak: float | None = None,
    ):
        self._source = source
        self._mangler = mangler
        if normalize_peak is not None and (
            normalize_peak <= 0 or not np.isfinite(normalize_peak)
        ):
            raise ValueError(
                f"normalize_peak must be a positive finite number, got {normalize_peak!r}"
            )
        self._normalize_peak = normalize_peak
        self._mogrified: np.ndarray | None = None  # (samples, channels), float32

    def inputs(self) -> list[ProcessingElement]:
        return [self._source]

    def _compute_extent(self) -> Extent:
        return self._source.extent()

    def channel_count(self) -> int | None:
        return self._source.channel_count()

    def is_pure(self) -> bool:
        return True

    def _mogrify(self) -> np.ndarray:
        """Render full source, FFT → mangler → IFFT; cache and return (samples, channels)."""
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
        phases = np.angle(analysis)

        # Apply user mangler
        m_magnitudes, m_phases = self._mangler(magnitudes, phases)
        m_analysis = m_magnitudes * np.exp(1j * m_phases)

        # IFFT back to time domain
        self._mogrified = np.real(np.fft.ifft(m_analysis, axis=0)).astype(np.float32)

        # Optionally normalise peak level
        if self._normalize_peak is not None:
            peak = np.max(np.abs(self._mogrified))
            if peak > 0:
                self._mogrified *= self._normalize_peak / peak

        return self._mogrified

    def _render(self, start: int, duration: int) -> Snippet:
        ext = self.extent()
        if ext.start is None or ext.end is None:
            return Snippet.from_zeros(start, duration, self.channel_count() or 1)

        mogrified = self._mogrify()
        channels = mogrified.shape[1]
        req_end = start + duration

        # No overlap with extent
        if req_end <= ext.start or start >= ext.end:
            return Snippet.from_zeros(start, duration, channels)

        # Request fully inside extent: return slice
        if ext.spans(start, duration):
            local_start = start - ext.start
            return Snippet(start, mogrified[local_start : local_start + duration].copy())

        # Partial overlap: build output with zeros and fill mogrified slice
        out = np.zeros((duration, channels), dtype=np.float32)
        overlap_start = max(start, ext.start)
        overlap_end = min(req_end, ext.end)
        if overlap_end > overlap_start:
            mog_start = overlap_start - ext.start
            mog_end = overlap_end - ext.start
            out_start = overlap_start - start
            out_end = overlap_end - start
            out[out_start:out_end, :] = mogrified[mog_start:mog_end, :]
        return Snippet(start, out)

    def __repr__(self) -> str:
        parts = [f"source={self._source.__class__.__name__}"]
        if self._normalize_peak is not None:
            parts.append(f"normalize_peak={self._normalize_peak}")
        return f"MagFreqPE({', '.join(parts)})"
