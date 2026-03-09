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
        pitch_shift:    Shift in semitones applied to the magnitude spectrum
                        before phase randomization (e.g. 12 = octave up,
                        -12 = octave down, 7 = perfect fifth up).  Default 0.
    """

    def __init__(
        self,
        source: ProcessingElement,
        seed: int | None = None,
        normalize_peak: float | None = None,
        padded_length: int | None = None,
        pitch_shift: float = 0.0,
    ):
        self._seed = seed
        self._padded_length = padded_length
        self._pitch_shift = pitch_shift

        if padded_length is not None:
            src_start = source.extent().start
            source = SetExtentPE(source, start=src_start, duration=padded_length)

        rng = np.random.default_rng(seed)
        pitch_ratio = 2.0 ** (pitch_shift / 12.0)

        def _tralfam_mangler(magnitudes, phases):
            mags = magnitudes
            if pitch_ratio != 1.0:
                n = mags.shape[0]
                half = n // 2 + 1
                pos_mags = mags[:half].copy()
                idx = np.arange(half, dtype=np.float64)
                src = idx / pitch_ratio
                shifted = np.zeros((half, mags.shape[1]), dtype=mags.dtype)
                for ch in range(mags.shape[1]):
                    shifted[:, ch] = np.interp(
                        src, idx, pos_mags[:, ch], left=0.0, right=0.0
                    )
                mags[:half] = shifted
                mags[half:] = shifted[-2:0:-1] if n % 2 == 0 else shifted[-1:0:-1]
            return mags, rng.random(phases.shape) * 2.0 * np.pi

        super().__init__(source, _tralfam_mangler, normalize_peak=normalize_peak)

    def __repr__(self) -> str:
        parts = [f"source={self._source.__class__.__name__}"]
        if self._seed is not None:
            parts.append(f"seed={self._seed}")
        if self._normalize_peak is not None:
            parts.append(f"normalize_peak={self._normalize_peak}")
        if self._padded_length is not None:
            parts.append(f"padded_length={self._padded_length}")
        if self._pitch_shift != 0.0:
            parts.append(f"pitch_shift={self._pitch_shift}")
        return f"TralfamPE({', '.join(parts)})"
