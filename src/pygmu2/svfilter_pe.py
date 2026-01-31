"""
SVFilterPE - second-order state variable filter with same API as BiquadPE.

Uses the trapezoidal-integration (bilinear) state variable filter design from
Andrew Simper (Cytomic), as described in the Google Music Synthesizer for Android
notebook "Second order sections in matrix form". Offers better stability and
precision under modulation than a biquad, especially for time-varying cutoff
and resonance.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy import signal

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.config import handle_error

from pygmu2.biquad_pe import BiquadMode

# Try to import numba for JIT compilation (optional optimization)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def _svf_constant_numba(
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """
    Numba-accelerated SVF with constant (A, B, C).
    Updates state in place. Returns output array (duration, channels).
    """
    duration, ch = x.shape
    y_out = np.empty_like(x)
    for n in range(duration):
        for c in range(ch):
            xn = x[n, c]
            y0, y1 = state[0, c], state[1, c]
            y_out[n, c] = C[0] * xn + C[1] * y0 + C[2] * y1
            state[0, c] = B[0] * xn + A[0, 0] * y0 + A[0, 1] * y1
            state[1, c] = B[1] * xn + A[1, 0] * y0 + A[1, 1] * y1
    return y_out


@jit(nopython=True, cache=True)
def _svf_varying_numba(
    x: np.ndarray,
    A_arr: np.ndarray,
    B_arr: np.ndarray,
    C_arr: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """
    Numba-accelerated SVF with per-sample (A, B, C).
    A_arr (duration, 2, 2), B_arr (duration, 2), C_arr (duration, 3).
    Updates state in place. Returns output array (duration, channels).
    """
    duration, ch = x.shape
    y_out = np.empty_like(x)
    for n in range(duration):
        a00, a01, a10, a11 = A_arr[n, 0, 0], A_arr[n, 0, 1], A_arr[n, 1, 0], A_arr[n, 1, 1]
        b0, b1 = B_arr[n, 0], B_arr[n, 1]
        c0, c1, c2 = C_arr[n, 0], C_arr[n, 1], C_arr[n, 2]
        for c in range(ch):
            xn = x[n, c]
            y0, y1 = state[0, c], state[1, c]
            y_out[n, c] = c0 * xn + c1 * y0 + c2 * y1
            state[0, c] = b0 * xn + a00 * y0 + a01 * y1
            state[1, c] = b1 * xn + a10 * y0 + a11 * y1
    return y_out


# Mode integer for Numba (no Enum in nopython). 0=LOWPASS, 1=HIGHPASS, 2=BANDPASS,
# 3=NOTCH, 4=PEAKING, 5=LOWSHELF, 6=HIGHSHELF. ALLPASS not used.
_MODE_TO_INT = {
    BiquadMode.LOWPASS: 0,
    BiquadMode.HIGHPASS: 1,
    BiquadMode.BANDPASS: 2,
    BiquadMode.NOTCH: 3,
    BiquadMode.PEAKING: 4,
    BiquadMode.LOWSHELF: 5,
    BiquadMode.HIGHSHELF: 6,
}


@jit(nopython=True, cache=True)
def _svf_coefficients_batch_numba(
    freq_values: np.ndarray,
    q_values: np.ndarray,
    mode_int: int,
    gain_db: float,
    sample_rate: int,
    A_arr: np.ndarray,
    B_arr: np.ndarray,
    C_arr: np.ndarray,
) -> None:
    """
    Fill A_arr (duration, 2, 2), B_arr (duration, 2), C_arr (duration, 3)
    from per-sample freq (Hz) and q. mode_int: 0=lowpass, 1=highpass, 2=bandpass,
    3=notch, 4=peaking, 5=lowshelf, 6=highshelf. Writes in place; returns None.
    """
    duration = freq_values.shape[0]
    fs = float(sample_rate)
    for n in range(duration):
        freq = freq_values[n]
        q = q_values[n]
        # Normalized frequency (cycles per sample); g = tan(pi*f_norm) matches Biquad
        f_norm = freq / fs
        if f_norm < 1e-6:
            f_norm = 1e-6
        if f_norm > 0.5:
            f_norm = 0.5
        # Resonance from Q (or PEAKING-specific)
        if mode_int == 4:  # PEAKING
            A_lin = 10.0 ** (gain_db / 40.0)
            q_clip = q
            if q_clip < 0.01:
                q_clip = 0.01
            if q_clip > 100.0:
                q_clip = 100.0
            k_bell = 1.0 / (q_clip * A_lin)
            res = 1.0 - 0.5 * k_bell
            if res < 0.0:
                res = 0.0
            if res > 0.999:
                res = 0.999
        else:
            if q < 0.01:
                q = 0.01
            if q > 100.0:
                q = 100.0
            res = 1.0 - 0.5 / q
            if res < 0.0:
                res = 0.0
            if res > 0.999:
                res = 0.999
        k = 2.0 - 2.0 * res
        g = math.tan(math.pi * f_norm)
        shelf_a = 1.0
        if mode_int == 5:  # LOWSHELF
            A_lin = 10.0 ** (gain_db / 40.0)
            shelf_a = 1.0 / math.sqrt(A_lin)
        elif mode_int == 6:  # HIGHSHELF
            A_lin = 10.0 ** (gain_db / 40.0)
            shelf_a = math.sqrt(A_lin)
        g = g * shelf_a
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        A_arr[n, 0, 0] = 2.0 * a1 - 1.0
        A_arr[n, 0, 1] = -2.0 * a2
        A_arr[n, 1, 0] = 2.0 * a2
        A_arr[n, 1, 1] = 1.0 - 2.0 * a3
        B_arr[n, 0] = 2.0 * a2
        B_arr[n, 1] = 2.0 * a3
        # Output mix m = [m0, m1, m2]; C = m0*C_v0 + m1*C_v1 + m2*C_v2
        # C_v0=[1,0,0], C_v1=[a2,a1,-a2], C_v2=[a3,a2,1-a3]
        if mode_int == 0:  # LOWPASS
            m0, m1, m2 = 0.0, 0.0, 1.0
        elif mode_int == 1:  # HIGHPASS
            m0, m1, m2 = 1.0, -k, -1.0
        elif mode_int == 2:  # BANDPASS
            m0, m1, m2 = 0.0, 1.0, 0.0
        elif mode_int == 3:  # NOTCH
            m0, m1, m2 = 1.0, -k, 0.0
        elif mode_int == 4:  # PEAKING
            A_lin = 10.0 ** (gain_db / 40.0)
            m0, m1, m2 = 1.0, k * (A_lin * A_lin - 1.0), 0.0
        elif mode_int == 5:  # LOWSHELF
            A_lin = 10.0 ** (gain_db / 40.0)
            m0, m1, m2 = 1.0, k * (A_lin - 1.0), A_lin * A_lin - 1.0
        elif mode_int == 6:  # HIGHSHELF
            A_lin = 10.0 ** (gain_db / 40.0)
            A2 = A_lin * A_lin
            m0, m1, m2 = A2, k * (A_lin - A2), 1.0 - A2
        else:
            m0, m1, m2 = 0.0, 0.0, 1.0
        C_arr[n, 0] = m0 * 1.0 + m1 * a2 + m2 * a3
        C_arr[n, 1] = m1 * a1 + m2 * a2
        C_arr[n, 2] = -m1 * a2 + m2 * (1.0 - a3)


def _svf_coefficients(
    f_norm: float,
    res: float,
    mode: BiquadMode,
    gain_db: float,
    sample_rate: int,
    q: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute state variable filter (A, B, C) for normalized frequency and resonance.

    f_norm: normalized frequency (cycles per sample), 0.5 = Nyquist.
    res: resonance in [0, 1); higher = more resonance. Related to Q by res = 1 - 0.5/Q.
    mode: filter type (LOWPASS, HIGHPASS, BANDPASS, NOTCH, PEAKING, LOWSHELF, HIGHSHELF).
    gain_db: gain in dB for peaking/shelf modes.
    q: Q factor (used for PEAKING/bell mode). If None, uses res only.

    Returns:
        (A, B, C) where A is 2x2, B is length 2, C is length 3.
        out = C @ [x, y0, y1],  y_new = B*x + A @ y
    """
    if mode == BiquadMode.ALLPASS:
        handle_error(
            "SVFilterPE: ALLPASS mode is not supported by the state variable filter.",
            fatal=True,
            exception_class=ValueError,
        )

    # PEAKING (bell): res and k from Q and A: k = 1/(Q*A), res = 1 - 0.5*k
    if mode == BiquadMode.PEAKING and q is not None and q > 0:
        A_lin = 10.0 ** (gain_db / 40.0)
        q_clip = max(0.01, min(100.0, q))
        k_bell = 1.0 / (q_clip * A_lin)
        res = 1.0 - 0.5 * k_bell
        res = max(0.0, min(0.999, res))

    k = 2.0 - 2.0 * res
    g = math.tan(math.pi * f_norm)

    shelf_a = 1.0
    if mode == BiquadMode.LOWSHELF:
        A_lin = 10.0 ** (gain_db / 40.0)
        shelf_a = 1.0 / math.sqrt(A_lin)
    elif mode == BiquadMode.HIGHSHELF:
        A_lin = 10.0 ** (gain_db / 40.0)
        shelf_a = math.sqrt(A_lin)

    g = g * shelf_a
    a1 = 1.0 / (1.0 + g * (g + k))
    a2 = g * a1
    a3 = g * a2

    A = np.array([
        [2.0 * a1 - 1.0, -2.0 * a2],
        [2.0 * a2, 1.0 - 2.0 * a3],
    ], dtype=np.float64)
    B = np.array([2.0 * a2, 2.0 * a3], dtype=np.float64)

    C_v0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    C_v1 = np.array([a2, a1, -a2], dtype=np.float64)
    C_v2 = np.array([a3, a2, 1.0 - a3], dtype=np.float64)

    if mode == BiquadMode.LOWPASS:
        m = np.array([0.0, 0.0, 1.0])
    elif mode == BiquadMode.BANDPASS:
        m = np.array([0.0, 1.0, 0.0])
    elif mode == BiquadMode.HIGHPASS:
        m = np.array([1.0, -k, -1.0])
    elif mode == BiquadMode.NOTCH:
        m = np.array([1.0, -k, 0.0])
    elif mode == BiquadMode.PEAKING:
        # Bell: m = [1, k*(A^2-1), 0] (k already set from Q and A above)
        A_lin = 10.0 ** (gain_db / 40.0)
        m = np.array([1.0, k * (A_lin * A_lin - 1.0), 0.0])
    elif mode == BiquadMode.LOWSHELF:
        A_lin = 10.0 ** (gain_db / 40.0)
        m = np.array([1.0, k * (A_lin - 1.0), A_lin * A_lin - 1.0])
    elif mode == BiquadMode.HIGHSHELF:
        A_lin = 10.0 ** (gain_db / 40.0)
        A2 = A_lin * A_lin
        m = np.array([A2, k * (A_lin - A2), 1.0 - A2])
    else:
        m = np.array([0.0, 0.0, 1.0])

    C = m[0] * C_v0 + m[1] * C_v1 + m[2] * C_v2
    return A, B, C


class SVFilterPE(ProcessingElement):
    """
    Second-order state variable filter with the same API as BiquadPE.

    Based on the trapezoidal-integration SVF design (Cytomic/Simper). Better
    stability and precision under modulation than a biquad. Supports the same
    filter types as BiquadPE except ALLPASS (not supported).

    State update: y_{n+1} = B*x_n + A*y_n,  out_n = C @ [x_n, y_0, y_1].

    This filter maintains internal state; is_pure() returns False.
    State is reset on on_start() and on_stop().

    Args:
        source: Input audio PE
        frequency: Cutoff/center frequency in Hz (float or PE)
        q: Quality factor (float or PE). Higher Q = sharper resonance.
        mode: Filter type (BiquadMode). ALLPASS is not supported.
        gain_db: Gain in dB for peaking/shelf filters (default: 0.0)
    """

    def __init__(
        self,
        source: ProcessingElement,
        frequency: Union[float, ProcessingElement],
        q: Union[float, ProcessingElement],
        mode: BiquadMode = BiquadMode.LOWPASS,
        gain_db: float = 0.0,
    ):
        if mode == BiquadMode.ALLPASS:
            raise ValueError(
                "SVFilterPE does not support ALLPASS mode. "
                "Use BiquadPE for allpass, or another mode."
            )
        self._source = source
        self._frequency = frequency
        self._q = q
        self._mode = mode
        self._gain_db = gain_db

        self._freq_is_pe = isinstance(frequency, ProcessingElement)
        self._q_is_pe = isinstance(q, ProcessingElement)

        # State: y vector (2, channels). Initialized in on_start().
        self._state: Optional[np.ndarray] = None

    @property
    def source(self) -> ProcessingElement:
        return self._source

    @property
    def frequency(self) -> Union[float, ProcessingElement]:
        return self._frequency

    @property
    def q(self) -> Union[float, ProcessingElement]:
        return self._q

    @property
    def mode(self) -> BiquadMode:
        return self._mode

    @property
    def gain_db(self) -> float:
        return self._gain_db

    def inputs(self) -> list[ProcessingElement]:
        result = [self._source]
        if self._freq_is_pe:
            result.append(self._frequency)
        if self._q_is_pe:
            result.append(self._q)
        return result

    def is_pure(self) -> bool:
        return False

    def channel_count(self) -> Optional[int]:
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        extent = self._source.extent()
        if self._freq_is_pe:
            freq_extent = self._frequency.extent()
            extent = extent.intersection(freq_extent) or extent
        if self._q_is_pe:
            q_extent = self._q.extent()
            extent = extent.intersection(q_extent) or extent
        return extent

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._state = None

    def _reset_state(self) -> None:
        channels = self._source.channel_count() or 1
        self._state = np.zeros((2, channels), dtype=np.float64)

    @staticmethod
    def _freq_to_normalized(freq_hz: float, sample_rate: int) -> float:
        """Convert Hz to normalized frequency (cycles per sample). 0.5 = Nyquist.
        SVF uses g = tan(pi * f_norm) so cutoff matches Biquad (omega = 2*pi*f/Fs)."""
        f = freq_hz / float(sample_rate)
        return float(np.clip(f, 1e-6, 0.5))

    @staticmethod
    def _q_to_res(q: float) -> float:
        """Convert Q to SVF resonance parameter res = 1 - 0.5/Q."""
        q = np.clip(q, 0.01, 100.0)
        return float(np.clip(1.0 - 0.5 / q, 0.0, 0.999))

    def _render(self, start: int, duration: int) -> Snippet:
        if self._state is None:
            self._reset_state()

        source_snippet = self._source.render(start, duration)
        x = source_snippet.data.astype(np.float64)
        channels = x.shape[1]

        if self._state.shape[1] != channels:
            self._state = np.zeros((2, channels), dtype=np.float64)

        freq_values = self._scalar_or_pe_values(
            self._frequency, start, duration, dtype=np.float64
        )
        q_values = self._scalar_or_pe_values(
            self._q, start, duration, dtype=np.float64
        )

        sr = self.sample_rate
        if not self._freq_is_pe and not self._q_is_pe:
            f_norm = self._freq_to_normalized(float(self._frequency), sr)
            res = self._q_to_res(float(self._q))
            q_arg = float(self._q) if self._mode == BiquadMode.PEAKING else None
            A, B, C = _svf_coefficients(
                f_norm, res, self._mode, self._gain_db, sr, q=q_arg
            )
            y = self._filter_constant(x, A, B, C)
        else:
            y = self._filter_varying(x, freq_values, q_values, sr)

        return Snippet(start, y.astype(np.float32))

    def _filter_constant(
        self,
        x: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
    ) -> np.ndarray:
        """Process with constant (A, B, C). Uses Numba when available, else scipy.signal.dlsim."""
        if NUMBA_AVAILABLE:
            return _svf_constant_numba(x, A, B, C, self._state)
        # Fallback: scipy.signal.dlsim (vectorized per channel)
        duration, ch = x.shape
        y_out = np.zeros_like(x)
        B_ss = B.reshape(2, 1).astype(np.float64)
        C_ss = np.array([[C[1], C[2]]], dtype=np.float64)
        D_ss = np.array([[C[0]]], dtype=np.float64)
        sys = signal.StateSpace(A, B_ss, C_ss, D_ss, dt=1.0)
        for c in range(ch):
            u = np.asarray(x[:, c], dtype=np.float64)
            x0 = np.asarray(self._state[:, c], dtype=np.float64)
            _tout, yout, xout = signal.dlsim(sys, u, x0=x0)
            y_out[:, c] = yout.ravel()
            self._state[:, c] = xout[-1]
        return y_out

    def _filter_varying(
        self,
        x: np.ndarray,
        freq_values: np.ndarray,
        q_values: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Process with per-sample (A, B, C). Uses Numba when available."""
        duration, ch = x.shape
        A_arr = np.zeros((duration, 2, 2), dtype=np.float64)
        B_arr = np.zeros((duration, 2), dtype=np.float64)
        C_arr = np.zeros((duration, 3), dtype=np.float64)
        if NUMBA_AVAILABLE:
            mode_int = _MODE_TO_INT[self._mode]
            _svf_coefficients_batch_numba(
                freq_values, q_values, mode_int, self._gain_db, sample_rate,
                A_arr, B_arr, C_arr,
            )
            return _svf_varying_numba(x, A_arr, B_arr, C_arr, self._state)
        # Fallback: Python loop over samples
        for n in range(duration):
            f_norm = self._freq_to_normalized(float(freq_values[n]), sample_rate)
            res = self._q_to_res(float(q_values[n]))
            q_arg = float(q_values[n]) if self._mode == BiquadMode.PEAKING else None
            A, B, C = _svf_coefficients(
                f_norm, res, self._mode, self._gain_db, sample_rate, q=q_arg
            )
            A_arr[n] = A
            B_arr[n] = B
            C_arr[n] = C
        y_out = np.zeros_like(x)
        y_state = self._state.copy()
        for n in range(duration):
            A, B, C = A_arr[n], B_arr[n], C_arr[n]
            for c in range(ch):
                xn = x[n, c]
                y0, y1 = y_state[0, c], y_state[1, c]
                y_out[n, c] = C[0] * xn + C[1] * y0 + C[2] * y1
                y_state[0, c] = B[0] * xn + A[0, 0] * y0 + A[0, 1] * y1
                y_state[1, c] = B[1] * xn + A[1, 0] * y0 + A[1, 1] * y1
        self._state = y_state
        return y_out

    def __repr__(self) -> str:
        freq_str = (
            f"{self._frequency.__class__.__name__}(...)"
            if self._freq_is_pe else str(self._frequency)
        )
        q_str = (
            f"{self._q.__class__.__name__}(...)"
            if self._q_is_pe else str(self._q)
        )
        return (
            f"SVFilterPE(source={self._source.__class__.__name__}, "
            f"frequency={freq_str}, q={q_str}, mode={self._mode.value})"
        )
