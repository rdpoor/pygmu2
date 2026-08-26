"""
MovingAveragePE - pure box-filter low-pass using a sliding window mean.

A moving average of window N replaces each output sample with the mean of the
N source samples ending at that position:

    y[t] = mean(x[t-N+1], x[t-N+2], ..., x[t])

This is a linear-phase FIR low-pass filter (a "box" or "rectangular" filter).
It is **pure**: the output at any position depends only on a fixed window of
source samples around that position, so renders can happen in any order.

Cutoff / window relationship (−3 dB point of the box filter):

    f_cutoff ≈ 0.443 * sample_rate / N
    N        ≈ 0.443 * sample_rate / f_cutoff

``window_for_cutoff(cutoff_hz, sample_rate)`` does the conversion and accepts
either a scalar frequency or a ProcessingElement for time-varying cutoff.

Rendering uses the prefix-sum (cumsum) trick for O(duration + N) cost per
block regardless of window size, even when N varies per sample.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet

# Hard ceiling on the window size accepted from a PE, in samples.
# Prevents runaway memory allocation if a PE outputs an enormous value.
_MAX_WINDOW = 1_000_000


def window_for_cutoff(
    cutoff_hz: float | ProcessingElement,
    sample_rate: float,
) -> int | ProcessingElement:
    """
    Convert a cutoff frequency to a moving-average window length.

    The −3 dB point of a box filter of length N is:

        f_cutoff ≈ 0.443 * sample_rate / N

    Args:
        cutoff_hz:   Cutoff frequency in Hz, **or** a ProcessingElement whose
                     output is a time-varying frequency signal.
        sample_rate: Sample rate in Hz.

    Returns:
        ``int`` when *cutoff_hz* is a scalar.
        ``ProcessingElement`` when *cutoff_hz* is a PE — the returned PE
        outputs per-sample window lengths (as float32) ready to pass directly
        to ``MovingAveragePE(source, window=...)``.

    Example:
        # Fixed cutoff
        filt = MovingAveragePE(source, window=window_for_cutoff(1000, 48000))

        # Time-varying cutoff driven by a ramp PE
        freq_ramp = PiecewisePE([(0, 200.0), (sr * 4, 4000.0)])
        filt = MovingAveragePE(source, window=window_for_cutoff(freq_ramp, sr))
    """
    if isinstance(cutoff_hz, ProcessingElement):
        from pygmu2.transform_pe import TransformPE

        _sr = float(sample_rate)
        scale = 0.443 * _sr

        def _freq_to_window(freq_data: np.ndarray) -> np.ndarray:
            # freq_data shape: (duration, channels) — use channel 0, broadcast
            hz = np.maximum(freq_data[:, 0:1], 1e-6)  # avoid div-by-zero
            w = np.round(scale / hz)
            return np.clip(w, 1.0, float(_MAX_WINDOW)).astype(np.float32)

        return TransformPE(cutoff_hz, func=_freq_to_window, name="window_for_cutoff")

    # Scalar path
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be > 0, got {cutoff_hz}")
    return max(1, round(0.443 * sample_rate / cutoff_hz))


class MovingAveragePE(ProcessingElement):
    """
    Pure box-filter low-pass via a sliding window mean.

    Args:
        source: Input ProcessingElement.
        window: Window length — one of:

                * **int** — fixed number of samples to average (>= 1).
                * **ProcessingElement** — time-varying window lengths in
                  samples, typically the output of ``window_for_cutoff(pe, sr)``.
                  Values are rounded to the nearest integer and clamped to
                  [1, 1 000 000] per block.

    Example:
        # Fixed cutoff (~500 Hz at 48 kHz)
        filt = MovingAveragePE(source, window=window_for_cutoff(500, 48000))

        # Time-varying cutoff: ramp from 200 Hz to 4 kHz over 4 seconds
        sr = 48000
        freq_ramp = PiecewisePE([(0, 200.0), (sr * 4, 4000.0)])
        filt = MovingAveragePE(source, window=window_for_cutoff(freq_ramp, sr))

        # Specify directly by sample count (fixed)
        filt = MovingAveragePE(source, window=64)
    """

    def __init__(
        self,
        source: ProcessingElement,
        window: int | ProcessingElement = 16,
    ):
        self._source = source
        self._window_is_pe = isinstance(window, ProcessingElement)

        if self._window_is_pe:
            self._window = window
        else:
            w = int(window)
            if w < 1:
                raise ValueError(f"window must be >= 1, got {w}")
            self._window = w

    @property
    def source(self) -> ProcessingElement:
        return self._source

    @property
    def window(self) -> int | ProcessingElement:
        return self._window

    def inputs(self) -> list[ProcessingElement]:
        if self._window_is_pe:
            return [self._source, self._window]
        return [self._source]

    def is_pure(self) -> bool:
        return True

    def channel_count(self) -> int | None:
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        return self._source.extent()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, start: int, duration: int) -> Snippet:
        if self._window_is_pe:
            return self._render_variable(start, duration)
        return self._render_fixed(start, duration)

    def _render_fixed(self, start: int, duration: int) -> Snippet:
        """
        Fixed-window path: classic prefix-sum, O(duration + N).

        y[i] = (padded[i+N] - padded[i]) / N
        """
        N = self._window
        src = self._source.render(start - N + 1, duration + N - 1).data
        channels = src.shape[1]
        zeros = np.zeros((1, channels), dtype=np.float64)
        padded = np.concatenate(
            [zeros, np.cumsum(src.astype(np.float64, copy=False), axis=0)]
        )
        result = (padded[N : N + duration] - padded[0:duration]) * (1.0 / N)
        return Snippet(start, result.astype(np.float32, copy=False))

    def _render_variable(self, start: int, duration: int) -> Snippet:
        """
        Variable-window path using numpy fancy indexing, O(duration + max_N).

        For block index i (output at absolute position start+i):

            y[i] = mean( source[ start+i - N[i]+1 .. start+i ] )
                 = (padded[ i + max_N ] - padded[ i + max_N - N[i] ]) / N[i]

        where padded is the prefix sum of the source fetch
        source[ start - max_N + 1 .. start + duration - 1 ].
        """
        # --- resolve per-sample window lengths ---
        win_vals = self._scalar_or_pe_values(
            self._window, start, duration, dtype=np.float64
        )
        N_vals = np.clip(np.round(win_vals), 1, _MAX_WINDOW).astype(np.int64)
        max_N = int(N_vals.max())

        # --- fetch source (enough for the widest window in this block) ---
        src = self._source.render(start - max_N + 1, duration + max_N - 1).data
        channels = src.shape[1]

        # --- prefix sum with leading zero row ---
        zeros = np.zeros((1, channels), dtype=np.float64)
        padded = np.concatenate(
            [zeros, np.cumsum(src.astype(np.float64, copy=False), axis=0)]
        )
        # padded shape: (duration + max_N, channels)

        # --- fancy-index the prefix sum ---
        i = np.arange(duration, dtype=np.int64)
        high = i + max_N  # end of each window (exclusive) in padded
        low = i + max_N - N_vals  # start of each window in padded

        result = (padded[high] - padded[low]) / N_vals.reshape(-1, 1)
        return Snippet(start, result.astype(np.float32, copy=False))

    def __repr__(self) -> str:
        win_str = (
            self._window.__class__.__name__ + "(...)"
            if self._window_is_pe
            else str(self._window)
        )
        return (
            f"MovingAveragePE(source={self._source.__class__.__name__}, "
            f"window={win_str})"
        )
