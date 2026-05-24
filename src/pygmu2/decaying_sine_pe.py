"""
DecayingSinePE - Exponentially decaying sine tone using a recurrence formula.

Generates  x[n] = A · rho^n · sin(2π·f·n / sr)  via the two-sample recurrence

    x[n] = 2·rho·cos(ω)·x[n−1] − rho²·x[n−2]          (ω = 2π·f / sr)

which requires only two multiplies and one add per sample in the inner loop.
Startup: x[-1]=0, x[0] = amplitude · rho · sin(ω), so the first true peak
falls at n=1 and the envelope is exactly amplitude · rho^n for all n ≥ 0.

Extent is infinite (0, None); higher-level code crops to desired duration.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""
from __future__ import annotations

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

    Output matches A · rho^n · sin(n·ω) for all n ≥ 0: the first sample is
    exactly zero and the envelope rises naturally from there.

    Extent is (0, None) — infinite. Use CropPE (or similar) to limit duration.

    Args:
        frequency:  Fundamental frequency in Hz.
        amplitude:  Peak amplitude (default 0.3).
        duration:   Seconds to −60 dB.  Exactly one of `duration` or `rho`
                    must be supplied; the other is computed automatically.
        rho:        Feedback gain in (0, 1).  If given, `duration` is ignored
                    for state purposes but may still be provided for documentation.
        channels:   Output channel count (default 1).  The sine is broadcast
                    identically across all channels.

    Example::

        # 440 Hz, decays to -60 dB in 2 seconds
        tone = DecayingSinePE(frequency=440.0, duration=2.0)
        two_sec = CropPE(tone, 0, 44100 * 2)

        # Same thing, specifying rho directly
        rho = DecayingSinePE.rho_for_decay_db(2.0, 44100)
        tone2 = DecayingSinePE(frequency=440.0, rho=rho)
    """

    # --------------------------------------------------------------------------
    # Helper
    # --------------------------------------------------------------------------

    @staticmethod
    def rho_for_decay_db(
        seconds: float,
        sample_rate: int,
        db: float = -60.0,
    ) -> float:
        """
        Feedback gain rho such that amplitude decays by |db| dB over `seconds`.

        For DecayingSinePE the envelope is  A · rho^n, so after  T = seconds·sr
        samples the amplitude is  A · rho^T.  Solving for rho:

            rho^T = 10^(db/20)
            rho   = 10^(db / (20 · seconds · sample_rate))

        Args:
            seconds:     Time in seconds over which the decay occurs.
            sample_rate: Sample rate in Hz.
            db:          Target decay in dB (negative, e.g. -60). Default -60.

        Returns:
            rho in (0, 1).
        """
        samples = seconds * sample_rate
        if samples <= 0:
            raise ValueError("seconds * sample_rate must be positive")
        rho = float(10 ** (db / (20.0 * samples)))
        return min(1.0, max(rho, 1e-9))


    def __init__(
        self,
        frequency: float,
        amplitude: float = 0.3,
        duration: float | None = None,
        rho: float | None = None,
        channels: int = 1,
    ):
        if frequency <= 0:
            raise ValueError(f"frequency must be positive, got {frequency}")
        if amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {amplitude}")
        if duration is None and rho is None:
            raise ValueError("supply exactly one of 'duration' (seconds to -60 dB) or 'rho'")
        if duration is not None and rho is not None:
            raise ValueError(
                "supply exactly one of 'duration' or 'rho', not both; "
                "use DecayingSinePE.rho_for_decay_db() to convert manually if needed"
            )
        if duration is not None:
            if duration <= 0:
                raise ValueError(f"duration must be positive, got {duration}")
            # rho is resolved in _render once sample_rate is known; cache seconds
            self._duration_seconds: float | None = float(duration)
            self._rho_param: float | None = None
        else:
            assert rho is not None
            if not (0 < rho < 1.0):
                raise ValueError(f"rho must be in (0, 1), got {rho}")
            self._duration_seconds = None
            self._rho_param = float(rho)

        self._frequency = float(frequency)
        self._amplitude = float(amplitude)
        self._channels = channels

        # Recurrence state (initialised on first render)
        self._x_prev: float = 0.0   # x[n-1]
        self._x_prev2: float = 0.0  # x[n-2]
        self._coeff: float = 0.0    # 2·rho·cos(ω)
        self._rho2: float = 0.0     # rho²
        self._ready: bool = False

    # ------------------------------------------------------------------
    # SourcePE interface
    # ------------------------------------------------------------------

    def _compute_extent(self) -> Extent:
        return Extent(0, None)

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

        # We only produce non-zero output for t >= 0
        ks_start = max(0, start)
        ks_end = max(0, start + duration)
        need = ks_end - ks_start
        if need <= 0:
            return Snippet(start, data)

        sr = self.sample_rate

        if not self._ready:
            # First call: resolve all constants that depend on sample_rate.
            # These are fixed for the lifetime of the object, so we compute
            # them once here and cache them — no trig on subsequent calls.
            if self._rho_param is not None:
                rho = self._rho_param
            else:
                assert self._duration_seconds is not None
                rho = DecayingSinePE.rho_for_decay_db(self._duration_seconds, sr)
            omega = 2.0 * np.pi * self._frequency / sr
            # Startup for out[n] = A·rho^n·sin(n·omega), n = 0, 1, 2, ...
            #
            # The loop emits x_prev then computes x_new = coeff·x_prev - rho²·x_prev2.
            # We want:
            #   out[0] = 0            → x_prev  = 0
            #   out[1] = A·rho·sin(ω) → x_new after step 0 must equal A·rho·sin(ω)
            #
            # From the recurrence at step 0:
            #   x_new = coeff·0 - rho²·x_prev2 = -rho²·x_prev2
            # Setting that equal to A·rho·sin(ω):
            #   x_prev2 = -A·sin(ω) / rho
            self._coeff = 2.0 * rho * np.cos(omega)
            self._rho2 = rho * rho
            self._x_prev = 0.0
            self._x_prev2 = -self._amplitude * np.sin(omega) / rho
            self._ready = True

        coeff = np.float32(self._coeff)
        rho2 = np.float32(self._rho2)
        x_prev = np.float32(self._x_prev)
        x_prev2 = np.float32(self._x_prev2)

        # ----------------------------------------------------------------
        # Inner loop — recurrence relation
        # ----------------------------------------------------------------
        out = np.empty(need, dtype=np.float32)
        for i in range(need):
            out[i] = x_prev                         # emit the "current" sample
            x_new = coeff * x_prev - rho2 * x_prev2
            x_prev2 = x_prev
            x_prev = x_new

        # Persist state for the next call
        self._x_prev = float(x_prev)
        self._x_prev2 = float(x_prev2)

        # Place generated samples into the output array
        offset = ks_start - start
        data[offset : offset + need] = np.broadcast_to(
            out[:, np.newaxis], (need, self._channels)
        )
        return Snippet(start, data)

    def channel_count(self) -> int:
        return self._channels

    def is_pure(self) -> bool:
        """DecayingSinePE maintains recurrence state; requires contiguous requests."""
        return False

    def __repr__(self) -> str:
        if self._duration_seconds is not None:
            return (
                f"DecayingSinePE(frequency={self._frequency}, "
                f"amplitude={self._amplitude}, "
                f"duration={self._duration_seconds})"
            )
        return (
            f"DecayingSinePE(frequency={self._frequency}, "
            f"amplitude={self._amplitude}, "
            f"rho={self._rho_param})"
        )
