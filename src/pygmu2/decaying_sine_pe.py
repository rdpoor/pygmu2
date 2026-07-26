"""
DecayingSinePE - Exponentially decaying sine tone using a recurrence formula.

Generates  x[n] = A · rho^n · sin(2π·f·n / sr)  via the two-sample recurrence

    x[n] = 2·rho·cos(ω)·x[n−1] − rho²·x[n−2]          (ω = 2π·f / sr)

which requires only two multiplies and one add per sample in the inner loop.
Startup: x[-1]=0, x[0] = amplitude · rho · sin(ω), so the first true peak
falls at n=1 and the envelope is exactly amplitude · rho^n for all n ≥ 0.

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

    Algorithm (inner loop):
        x[n] = 2·rho·cos(ω)·x[n−1] − rho²·x[n−2]     ω = 2π·f / sr

    Output matches A · rho^n · sin(n·ω) for all n ≥ 0.

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
        # Recurrence state (initialised on first render)
        self._x_prev: float = 0.0
        self._x_prev2: float = 0.0
        self._coeff: float = 0.0
        self._rho2: float = 0.0
        self._ready: bool = False

    # ------------------------------------------------------------------
    # SourcePE interface
    # ------------------------------------------------------------------

    def _compute_extent(self) -> Extent:
        # Always finite — crop_samples is known at construction time.
        return Extent(0, self._crop_samples)

    def _reset_state(self) -> None:
        self._x_prev = 0.0
        self._x_prev2 = 0.0
        self._coeff = 0.0
        self._rho2 = 0.0
        self._ready = False

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._reset_state()

    def _render(self, start: int, duration: int) -> Snippet:
        if duration <= 0:
            return Snippet.from_zeros(start, 0, self._channels)

        data = np.zeros((duration, self._channels), dtype=np.float32)

        ks_start = max(0, start)
        ks_end = max(0, start + duration)
        need = ks_end - ks_start
        if need <= 0:
            return Snippet(start, data)

        if not self._ready:
            sr = self.sample_rate
            rho = math.exp(-1.0 / (self._tau * sr))
            omega = 2.0 * math.pi * self._frequency / sr

            self._coeff = np.float32(2.0 * rho * math.cos(omega))
            self._rho2 = np.float32(rho * rho)
            self._x_prev = np.float32(0.0)
            self._x_prev2 = np.float32(-self._amplitude * math.sin(omega) / rho)

            self._crop_samples = math.ceil(
                -self._tau * sr * (self._db_floor / 20.0) * math.log(10)
            )
            self._ready = True

        # No output past the crop point.
        ks_end = min(ks_end, self._crop_samples)
        need = max(0, ks_end - ks_start)
        if need == 0:
            return Snippet(start, data)

        coeff = self._coeff
        rho2 = self._rho2
        x_prev = self._x_prev
        x_prev2 = self._x_prev2

        # ----------------------------------------------------------------
        # Inner loop — recurrence relation
        # ----------------------------------------------------------------
        out = np.empty(need, dtype=np.float32)
        for i in range(need):
            out[i] = x_prev
            x_new = coeff * x_prev - rho2 * x_prev2
            x_prev2 = x_prev
            x_prev = x_new

        self._x_prev = x_prev
        self._x_prev2 = x_prev2

        offset = ks_start - start
        data[offset : offset + need] = np.broadcast_to(
            out[:, np.newaxis], (need, self._channels)
        )
        return Snippet(start, data)

    def channel_count(self) -> int:
        return self._channels

    def is_pure(self) -> bool:
        return False

    def __repr__(self) -> str:
        floor = f", db_floor={self._db_floor}" if self._db_floor != -60.0 else ""
        return (
            f"DecayingSinePE(frequency={self._frequency}, "
            f"amplitude={self._amplitude}, "
            f"tau={self._tau}{floor})"
        )
