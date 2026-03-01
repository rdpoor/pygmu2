"""
TralfamPE - spread a finite source's spectrum randomly across its time span.

Tralfamadorians exist in all times simultaneously. This PE takes a finite
source, FFTs the whole extent, keeps magnitudes but randomizes phases, then
IFFTs. The result is cached; output extent matches the (padded) source extent.

Requires finite extent (start and end not None). Large extents will use
significant memory (full buffer in memory for FFT).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.mag_freq_pe import MagFreqPE
from pygmu2.processing_element import ProcessingElement
from pygmu2.set_extent_pe import SetExtentPE


class TralfamPE(MagFreqPE):
    """
    PE that spreads a finite source's spectrum randomly across its time span.

    Keeps FFT magnitudes intact but replaces every phase with an independent
    uniform random value in [0, 2π). All rendering and caching behaviour is
    inherited from MagFreqPE.

    Args:
        source:         Input PE with finite extent.
        seed:           Optional RNG seed for reproducible random phases.
        normalize_peak: Optional linear amplitude (e.g. 0.5) to scale peak to;
                        None = no normalization.
        padded_length:  If given, the source is zero-padded (or truncated) to
                        this many samples before the FFT, anchored at the
                        source's own start sample.  The output extent grows or
                        shrinks to match.
    """

    def __init__(
        self,
        source: ProcessingElement,
        seed: int | None = None,
        normalize_peak: float | None = None,
        padded_length: int | None = None,
    ):
        self._seed = seed
        self._padded_length = padded_length

        if padded_length is not None:
            src_start = source.extent().start
            source = SetExtentPE(source, start=src_start, duration=padded_length)

        rng = np.random.default_rng(seed)

        def _tralfam_mangler(magnitudes, phases):
            return magnitudes, rng.random(phases.shape) * 2.0 * np.pi

        super().__init__(source, _tralfam_mangler, normalize_peak=normalize_peak)

    def __repr__(self) -> str:
        parts = [f"source={self._source.__class__.__name__}"]
        if self._seed is not None:
            parts.append(f"seed={self._seed}")
        if self._normalize_peak is not None:
            parts.append(f"normalize_peak={self._normalize_peak}")
        if self._padded_length is not None:
            parts.append(f"padded_length={self._padded_length}")
        return f"TralfamPE({', '.join(parts)})"
