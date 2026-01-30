"""
KarplusStrongPE - Plucked string synthesis using the Karplus-Strong algorithm.

Minimal implementation: delay line (one period) filled with white noise,
feedback through a two-point average with gain rho. Delay length sets pitch;
rho (0 < rho <= 1) sets decay (lower = faster decay).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from pygmu2.processing_element import SourcePE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class KarplusStrongPE(SourcePE):
    """
    Plucked string using the classic Karplus-Strong algorithm.

    Algorithm:
      1. Delay line of length N = round(sample_rate / frequency) samples.
      2. Fill with one period of white noise (excitation).
      3. For each sample: out = rho * (buf[r] + buf[r+1]) / 2; write out to buf[r]; advance r.

    Args:
        frequency: Fundamental frequency in Hz.
        duration: Length of the pluck in seconds (total output length).
        rho: Feedback gain in (0, 1]. Lower = faster decay (duller); 1 = longest ring.
        amplitude: Scale of the initial noise (default 0.3).
        seed: Optional random seed for reproducible excitation.
        channels: Output channel count (default 1).

    Example:
        pluck = KarplusStrongPE(frequency=440.0, duration=1.0, rho=0.996)
    """

    def __init__(
        self,
        frequency: float,
        duration: float,
        rho: float = 0.996,
        amplitude: float = 0.3,
        seed: Optional[int] = None,
        channels: int = 1,
    ):
        if frequency <= 0:
            raise ValueError(f"frequency must be positive, got {frequency}")
        if duration < 0:
            raise ValueError(f"duration must be non-negative, got {duration}")
        if not (0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {amplitude}")

        self._frequency = float(frequency)
        self._duration_sec = float(duration)
        self._rho = float(rho)
        self._amplitude = float(amplitude)
        self._seed = seed
        self._channels = channels

        self._cached_output: Optional[np.ndarray] = None
        self._cached_total_samples: Optional[int] = None

    def configure(self, sample_rate: int) -> None:
        super().configure(sample_rate)
        self._cached_output = None
        self._cached_total_samples = None

    def _build_ks_buffer(self) -> None:
        sr = self.sample_rate
        total_samples = max(0, int(round(self._duration_sec * sr)))

        if total_samples == 0:
            self._cached_output = np.zeros((0, self._channels), dtype=np.float32)
            self._cached_total_samples = 0
            return

        # Delay line: one period of the fundamental
        delay_len = max(2, int(round(sr / self._frequency)))

        rng = np.random.default_rng(self._seed)
        noise = rng.standard_normal(delay_len).astype(np.float32)
        noise *= self._amplitude / (np.max(np.abs(noise)) + 1e-9)

        buf = noise.copy()
        out = np.zeros(total_samples, dtype=np.float32)
        r = 0

        for i in range(total_samples):
            r_next = (r + 1) % delay_len
            out[i] = self._rho * (buf[r] + buf[r_next]) * 0.5
            buf[r] = out[i]
            r = r_next

        self._cached_output = np.broadcast_to(
            out[:, np.newaxis], (total_samples, self._channels)
        ).copy()
        self._cached_total_samples = total_samples

    def _total_samples(self) -> int:
        sr = self.sample_rate
        return max(0, int(round(self._duration_sec * sr)))

    def _compute_extent(self) -> Extent:
        return Extent(0, self._total_samples())

    def _render(self, start: int, duration: int) -> Snippet:
        if duration <= 0:
            return Snippet.from_zeros(start, 0, self._channels)

        total = self._total_samples()
        if self._cached_output is None or self._cached_total_samples != total:
            self._build_ks_buffer()

        data = np.zeros((duration, self._channels), dtype=np.float32)
        end = start + duration
        copy_start = max(0, start)
        copy_end = min(total, end)
        if copy_start < copy_end:
            src_start = copy_start - start
            src_len = copy_end - copy_start
            data[src_start : src_start + src_len] = self._cached_output[
                copy_start:copy_end
            ]

        return Snippet(start, data)

    def channel_count(self) -> int:
        return self._channels

    def __repr__(self) -> str:
        return f"KarplusStrongPE(frequency={self._frequency}, duration={self._duration_sec}, rho={self._rho})"
