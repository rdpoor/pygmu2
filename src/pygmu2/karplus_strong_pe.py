"""
KarplusStrongPE - Plucked string synthesis using the Karplus-Strong algorithm.

Delay line (one period) + first-order allpass for fractional-sample delay
(accurate intonation). Feedback: two-point average with gain rho. Extent is
infinite (0, None); higher-level code crops to desired duration.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.source_pe import SourcePE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


def rho_for_decay_db(
    seconds: float,
    frequency: float,
    sample_rate: int,
    db: float = -60.0,
) -> float:
    """
    Feedback gain rho such that amplitude decays by |db| dB over `seconds`.

    In Karplus-Strong each delay-line cell is updated once per period (N =
    sample_rate/frequency samples) with out = rho * (buf[r] + buf[r+1])/2.
    The two-point average has magnitude cos(pi/N) at the fundamental, so
    effective gain per period is rho * cos(pi/N), not rho alone. Thus:

        (rho * cos(pi/N))^(seconds * frequency) = 10^(db/20)
        rho = 10^(db / (20 * seconds * frequency)) / cos(pi/N)

    Result is clamped to 1.0 (rho > 1 would be needed for db=0 due to averaging loss).

    Args:
        seconds: Time in seconds over which decay occurs.
        frequency: Fundamental frequency in Hz (sets delay-line length).
        sample_rate: Sample rate in Hz (to compute N).
        db: Target decay in dB (negative, e.g. -60 for 60 dB down). Default -60.

    Returns:
        rho in (0, 1].
    """
    periods = seconds * frequency
    if periods <= 0:
        raise ValueError("seconds * frequency must be positive")
    delay_len = max(2, int(np.floor(sample_rate / frequency)))
    avg_gain = np.cos(np.pi / delay_len)
    if avg_gain <= 0:
        return 1.0
    rho = float(10 ** (db / (20.0 * periods)) / avg_gain)
    return min(1.0, max(rho, 1e-9))


class KarplusStrongPE(SourcePE):
    """
    Plucked string using the classic Karplus-Strong algorithm.

    Algorithm:
      1. Total delay = sample_rate / frequency (integer + fractional).
         Integer part = delay line length N; fractional part = first-order allpass.
      2. Fill delay line with one period of white noise (excitation).
      3. For each sample: avg = (buf[r] + buf[r+1])/2; out_val = rho * avg;
         ap_out = allpass(out_val); write ap_out to buf[r] and output; advance r.

    Extent is (0, None) — infinite. Use CropPE (or similar) to limit duration.

    Optional two-phase decay: if both duration and rho_damping are provided,
    use rho for the first duration samples (sustain), then rho_damping
    (faster fade-out). If either is omitted, use a single rho throughout.

    Args:
        frequency: Fundamental frequency in Hz.
        rho: Feedback gain in (0, 1]. Sustain decay rate (or sole rate if no two-phase).
        duration: Optional sample count after which to switch to rho_damping.
        rho_damping: Optional feedback gain after duration (0 < rho_damping <= 1).
        amplitude: Scale of the initial noise (default 0.3).
        seed: Optional random seed for reproducible excitation.
        channels: Output channel count (default 1).

    Example:
        pluck = KarplusStrongPE(frequency=440.0, rho=0.996)
        one_sec = CropPE(pluck, 0, (44100) - (0))
        # Sustain 1 sec then fade faster:
        pluck2 = KarplusStrongPE(440.0, rho=0.996, duration=44100, rho_damping=0.95)
    """

    def __init__(
        self,
        frequency: float,
        rho: float = 0.996,
        duration: int | None = None,
        rho_damping: float | None = None,
        amplitude: float = 0.3,
        seed: int | None = None,
        channels: int = 1,
    ):
        if frequency <= 0:
            raise ValueError(f"frequency must be positive, got {frequency}")
        if not (0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {amplitude}")
        if duration is not None and rho_damping is not None:
            if duration < 0:
                raise ValueError(f"duration must be >= 0, got {duration}")
            if not (0 < rho_damping <= 1.0):
                raise ValueError(f"rho_damping must be in (0, 1], got {rho_damping}")

        self._frequency = float(frequency)
        self._rho = float(rho)
        self._duration_param: int | None = duration if (duration is not None and rho_damping is not None) else None
        self._rho_damping: float | None = float(rho_damping) if (duration is not None and rho_damping is not None) else None
        self._amplitude = float(amplitude)
        self._seed = seed
        self._channels = channels

        # KS state: delay line (one period, circular) + allpass state. No output cache
        # (impure PE receives contiguous requests only, so we stream).
        self._ks_buf: np.ndarray | None = None
        self._ks_r: int = 0
        self._ks_ap_in_prev: float = 0.0
        self._ks_ap_out_prev: float = 0.0
        self._delay_len: int = 0
        self._allpass_c: float = 0.0

    def _compute_extent(self) -> Extent:
        return Extent(0, None)

    def _reset_state(self) -> None:
        self._ks_buf = None

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._reset_state()

    def _render(self, start: int, duration: int) -> Snippet:
        if duration <= 0:
            return Snippet.from_zeros(start, 0, self._channels)

        data = np.zeros((duration, self._channels), dtype=np.float32)
        end = start + duration
        # Overlap with [0, inf): generate KS samples for this range only
        ks_start = max(0, start)
        ks_end = max(0, end)
        need = ks_end - ks_start
        if need <= 0:
            return Snippet(start, data)

        sr = self.sample_rate
        delay_float = sr / self._frequency
        delay_len = max(2, int(np.floor(delay_float)))
        frac_d = max(0.0, min(1.0, delay_float - delay_len))
        allpass_c = (1.0 - frac_d) / (1.0 + frac_d) if frac_d <= 1.0 else 0.0

        if self._ks_buf is None:
            # First call: initialize delay line with noise, run loop for need samples
            rng = np.random.default_rng(self._seed)
            noise = rng.standard_normal(delay_len).astype(np.float32)
            noise *= self._amplitude / (np.max(np.abs(noise)) + 1e-9)
            buf = noise.copy()
            ap_in_prev = np.float32(0.0)
            ap_out_prev = np.float32(0.0)
            r = 0
        else:
            buf = self._ks_buf
            r = self._ks_r
            ap_in_prev = self._ks_ap_in_prev
            ap_out_prev = self._ks_ap_out_prev

        out = np.zeros(need, dtype=np.float32)
        for i in range(need):
            # Two-phase decay: use rho_damping after duration samples (global index ks_start + i)
            rho_eff = self._rho
            if self._duration_param is not None and self._rho_damping is not None:
                if (ks_start + i) >= self._duration_param:
                    rho_eff = self._rho_damping
            r_next = (r + 1) % delay_len
            out_val = rho_eff * (buf[r] + buf[r_next]) * 0.5
            ap_out = allpass_c * out_val + ap_in_prev - allpass_c * ap_out_prev
            ap_in_prev = out_val
            ap_out_prev = ap_out
            out[i] = ap_out
            buf[r] = ap_out
            r = r_next

        self._ks_buf = buf
        self._ks_r = r
        self._ks_ap_in_prev = ap_in_prev
        self._ks_ap_out_prev = ap_out_prev
        self._delay_len = delay_len
        self._allpass_c = allpass_c

        # Place generated samples into snippet: data[0:duration] maps to time [start, end)
        # KS output out[0:need] maps to time [ks_start, ks_end)
        offset = ks_start - start  # index in data where KS output starts
        data[offset : offset + need] = np.broadcast_to(
            out[:, np.newaxis], (need, self._channels)
        )
        return Snippet(start, data)

    def channel_count(self) -> int:
        return self._channels

    def is_pure(self) -> bool:
        """KarplusStrongPE maintains delay-line state; requires contiguous requests."""
        return False

    def __repr__(self) -> str:
        if self._duration_param is not None and self._rho_damping is not None:
            return f"KarplusStrongPE(frequency={self._frequency}, rho={self._rho}, duration={self._duration_param}, rho_damping={self._rho_damping})"
        return f"KarplusStrongPE(frequency={self._frequency}, rho={self._rho})"
