"""
DecayingSinePE - Exponentially decaying sine tone.

Generates  x[n] = A · rho^n · sin(2π·f·n / sr)  directly from the closed
form (vectorized), so the PE is stateless and can be rendered at any
position in any order.  (An earlier version used a two-sample recurrence
in a Python loop — measured slower than the vectorized closed form, and
it forced stateful/contiguous rendering.)

Extent is finite: (0, crop_samples), where crop_samples is the sample index
at which the envelope reaches db_floor.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations
import math
import numpy as np
from pygmu2.source_pe import SourcePE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet

# ---------------------------------------------------------------------------
# PE
# ---------------------------------------------------------------------------


class DecayingSinePE(SourcePE):
    """
    Exponentially decaying sine tone.

    Output is A · rho^n · sin(n·ω) for all n ≥ 0 (ω = 2π·f / sr),
    computed in closed form — stateless, random-access.

    Extent is finite: the stream is cropped at the sample where the envelope
    reaches db_floor, computed as:
        crop_samples = ceil(−tau · sr · ln(10) · db_floor / 20)

    Args:
        frequency:  Fundamental frequency in Hz.
        amplitude:  Peak amplitude (default 0.3).
        tau:        Exponential time constant in seconds.  The envelope is
                    A · exp(−t / tau), so after one tau the amplitude has
                    fallen to A / e (≈ −8.7 dB).
        channels:   Output channel count (default 1).
        db_floor:   Level at which the stream is cropped, in dB (default −60).
                    Must be negative.

    Example::
        # 440 Hz, time constant 0.289 s (≈ −60 dB in 2 s), crop at −60 dB
        tone = DecayingSinePE(frequency=440.0, tau=0.289)

        # Same decay rate, but crop earlier at −40 dB
        tone2 = DecayingSinePE(frequency=440.0, tau=0.289, db_floor=-40.0)
    """

    def __init__(
        self,
        frequency: float,
        amplitude: float = 0.3,
        tau: float = 1.0,
        channels: int = 1,
        db_floor: float = -60.0,
    ):
        super().__init__()
        if frequency <= 0:
            raise ValueError(f"frequency must be positive, got {frequency}")
        if amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {amplitude}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if db_floor >= 0:
            raise ValueError(f"db_floor must be negative, got {db_floor}")

        self._frequency = float(frequency)
        self._amplitude = float(amplitude)
        self._tau = float(tau)
        self._channels = channels
        self._db_floor = float(db_floor)

        self._crop_samples = math.ceil(
            -self._tau * self.sample_rate * (self._db_floor / 20.0) * math.log(10)
        )

    # ------------------------------------------------------------------
    # SourcePE interface
    # ------------------------------------------------------------------

    def _compute_extent(self) -> Extent:
        # Always finite — crop_samples is known at construction time.
        return Extent(0, self._crop_samples)

    def _render(self, start: int, duration: int) -> Snippet:
        if duration <= 0:
            return Snippet.from_zeros(start, 0, self._channels)

        data = np.zeros((duration, self._channels), dtype=np.float32)

        ks_start = max(0, start)
        ks_end = min(start + duration, self._crop_samples)
        if ks_end <= ks_start:
            return Snippet(start, data)

        sr = self.sample_rate
        n = np.arange(ks_start, ks_end, dtype=np.float64)
        omega = 2.0 * math.pi * self._frequency / sr
        env = self._amplitude * np.exp(-n / (self._tau * sr))
        out = (env * np.sin(n * omega)).astype(np.float32)

        offset = ks_start - start
        data[offset : offset + len(out)] = out[:, np.newaxis]
        return Snippet(start, data)

    def channel_count(self) -> int:
        return self._channels

    def __repr__(self) -> str:
        floor = f", db_floor={self._db_floor}" if self._db_floor != -60.0 else ""
        return (
            f"DecayingSinePE(frequency={self._frequency}, "
            f"amplitude={self._amplitude}, "
            f"tau={self._tau}{floor})"
        )
