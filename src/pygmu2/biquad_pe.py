"""
BiquadPE - second-order IIR filter with multiple filter types.

Coefficients calculated using Robert Bristow-Johnson's Audio EQ Cookbook.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from enum import Enum

import numpy as np
from scipy import signal

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.config import handle_error

# Try to import numba for JIT compilation (optional optimization)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator when numba isn't available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# JIT-compiled filter function for time-varying coefficients
@jit(nopython=True, cache=True)
def _biquad_varying_numba(
    x: np.ndarray,
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
    a1: np.ndarray, a2: np.ndarray,
    x1: np.ndarray, x2: np.ndarray,
    y1: np.ndarray, y2: np.ndarray,
) -> tuple:
    """
    Numba-accelerated biquad filter with time-varying coefficients.
    
    Returns (output, x1_final, x2_final, y1_final, y2_final).
    """
    duration = x.shape[0]
    channels = x.shape[1]
    y = np.zeros((duration, channels), dtype=np.float64)
    
    for n in range(duration):
        for c in range(channels):
            y0 = (b0[n] * x[n, c] + b1[n] * x1[c] + b2[n] * x2[c]
                  - a1[n] * y1[c] - a2[n] * y2[c])
            y[n, c] = y0
            x2[c] = x1[c]
            x1[c] = x[n, c]
            y2[c] = y1[c]
            y1[c] = y0
    
    return y, x1, x2, y1, y2


class BiquadMode(Enum):
    """Biquad filter types."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"
    ALLPASS = "allpass"
    PEAKING = "peaking"
    LOWSHELF = "lowshelf"
    HIGHSHELF = "highshelf"


class BiquadPE(ProcessingElement):
    """
    Second-order IIR (biquad) filter.
    
    Implements the standard biquad difference equation:
        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    
    Coefficients are calculated using Robert Bristow-Johnson's Audio EQ
    Cookbook formulas.
    
    This filter maintains internal state, so is_pure() returns False.
    The state is reset on on_start() and on_stop().
    
    Args:
        source: Input audio PE
        frequency: Cutoff/center frequency in Hz (float or PE)
        q: Quality factor (float or PE). Higher Q = sharper resonance.
        mode: Filter type (BiquadMode)
        gain_db: Gain in dB for peaking/shelf filters (default: 0.0)
    
    Example:
        # Simple lowpass filter at 1kHz
        filtered_stream = BiquadPE(source_stream, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS)
        
        # Auto-wah effect (envelope-controlled frequency)
        envelope_stream = EnvelopeFollowerPE(source_stream)  # Hypothetical
        freq_mod_stream = MixPE(ConstantPE(500.0), GainPE(envelope_stream, 2000.0))  # 500-2500 Hz
        autowah_stream = BiquadPE(source_stream, frequency=freq_mod_stream, q=5.0, mode=BiquadMode.BANDPASS)
        
        # Filter sweep
        freq_sweep = PiecewisePE([(0, 100.0), (44100*2, 5000.0)])  # 2 second sweep
        sweep = BiquadPE(source, frequency=freq_sweep, q=2.0, mode=BiquadMode.LOWPASS)
        
        # Parametric EQ boost at 1kHz
        eq = BiquadPE(source, frequency=1000.0, q=2.0, mode=BiquadMode.PEAKING, gain_db=6.0)
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        frequency: float | ProcessingElement,
        q: float | ProcessingElement,
        mode: BiquadMode = BiquadMode.LOWPASS,
        gain_db: float = 0.0,
    ):
        self._source = source
        self._frequency = frequency
        self._q = q
        self._mode = mode
        self._gain_db = gain_db
        
        # Track if parameters are PEs
        self._freq_is_pe = isinstance(frequency, ProcessingElement)
        self._q_is_pe = isinstance(q, ProcessingElement)
        
        # Filter state: [x[n-1], x[n-2], y[n-1], y[n-2]] per channel
        # Initialized in on_start()
        self._state: np.ndarray | None = None
        # State for scipy.signal.lfilter (constant coefficient path)
        self._lfilter_state: np.ndarray | None = None
    
    @property
    def source(self) -> ProcessingElement:
        """The input audio PE."""
        return self._source
    
    @property
    def frequency(self) -> float | ProcessingElement:
        """The cutoff/center frequency."""
        return self._frequency
    
    @property
    def q(self) -> float | ProcessingElement:
        """The Q (quality factor)."""
        return self._q
    
    @property
    def mode(self) -> BiquadMode:
        """The filter type."""
        return self._mode
    
    @property
    def gain_db(self) -> float:
        """The gain in dB (for peaking/shelf filters)."""
        return self._gain_db
    
    def inputs(self) -> list[ProcessingElement]:
        """Return all input PEs."""
        result = [self._source]
        if self._freq_is_pe:
            result.append(self._frequency)
        if self._q_is_pe:
            result.append(self._q)
        return result
    
    def is_pure(self) -> bool:
        """
        BiquadPE is NOT pure - it maintains internal filter state.
        """
        return False
    
    def channel_count(self) -> int | None:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """
        Return the extent of this PE.
        
        Matches source extent, intersected with parameter extents if they are PEs.
        """
        extent = self._source.extent()
        
        if self._freq_is_pe:
            freq_extent = self._frequency.extent()
            extent = extent.intersection(freq_extent) or extent
        
        if self._q_is_pe:
            q_extent = self._q.extent()
            extent = extent.intersection(q_extent) or extent
        
        return extent
    
    def _on_start(self) -> None:
        """Reset filter state at start of rendering."""
        self._reset_state()

    def _on_stop(self) -> None:
        """Clear filter state at end of rendering."""
        self._state = None
        self._lfilter_state = None
    
    def _reset_state(self) -> None:
        """Initialize/reset the filter state to zeros."""
        channels = self._source.channel_count() or 1
        # State for scipy.signal.lfilter: shape (2, channels) for biquad
        # Also keep legacy state for time-varying coefficient path
        self._state = np.zeros((4, channels), dtype=np.float64)
        self._lfilter_state = np.zeros((2, channels), dtype=np.float64)
    
    def _compute_coefficients(
        self,
        freq: np.ndarray,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute biquad coefficients for given frequency and Q values.
        
        Args:
            freq: Array of frequencies (shape: (n,) or scalar)
            q: Array of Q values (shape: (n,) or scalar)
        
        Returns:
            Tuple of (b0, b1, b2, a1, a2) coefficient arrays
        """
        # Ensure arrays
        freq = np.atleast_1d(freq).astype(np.float64)
        q = np.atleast_1d(q).astype(np.float64)
        
        # Clamp frequency to valid range (avoid aliasing and division issues)
        nyquist = self.sample_rate / 2.0
        freq = np.clip(freq, 1.0, nyquist * 0.99)
        
        # Clamp Q to avoid division by zero
        q = np.clip(q, 0.01, 100.0)
        
        # Intermediate values
        omega = 2.0 * np.pi * freq / self.sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2.0 * q)
        
        # For peaking/shelf filters
        A = 10.0 ** (self._gain_db / 40.0)  # sqrt of amplitude
        
        # Compute coefficients based on filter type
        if self._mode == BiquadMode.LOWPASS:
            b0 = (1.0 - cos_omega) / 2.0
            b1 = 1.0 - cos_omega
            b2 = (1.0 - cos_omega) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha
            
        elif self._mode == BiquadMode.HIGHPASS:
            b0 = (1.0 + cos_omega) / 2.0
            b1 = -(1.0 + cos_omega)
            b2 = (1.0 + cos_omega) / 2.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha
            
        elif self._mode == BiquadMode.BANDPASS:
            # Constant 0 dB peak gain
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha
            
        elif self._mode == BiquadMode.NOTCH:
            b0 = 1.0
            b1 = -2.0 * cos_omega
            b2 = 1.0
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha
            
        elif self._mode == BiquadMode.ALLPASS:
            b0 = 1.0 - alpha
            b1 = -2.0 * cos_omega
            b2 = 1.0 + alpha
            a0 = 1.0 + alpha
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha
            
        elif self._mode == BiquadMode.PEAKING:
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_omega
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_omega
            a2 = 1.0 - alpha / A
            
        elif self._mode == BiquadMode.LOWSHELF:
            sqrt_A = np.sqrt(A)
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_omega + 2.0 * sqrt_A * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_omega)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_omega - 2.0 * sqrt_A * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_omega + 2.0 * sqrt_A * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_omega)
            a2 = (A + 1.0) + (A - 1.0) * cos_omega - 2.0 * sqrt_A * alpha
            
        elif self._mode == BiquadMode.HIGHSHELF:
            sqrt_A = np.sqrt(A)
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_omega + 2.0 * sqrt_A * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_omega)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_omega - 2.0 * sqrt_A * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_omega + 2.0 * sqrt_A * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_omega)
            a2 = (A + 1.0) - (A - 1.0) * cos_omega - 2.0 * sqrt_A * alpha
        
        else:
            handle_error(
                f"Unknown filter mode: {self._mode}",
                fatal=True,
                exception_class=ValueError,
            )
            return (0.0, 0.0, 0.0, 0.0, 0.0)  # Unreachable, satisfies type checker
        
        # Normalize by a0
        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0
        
        return b0, b1, b2, a1, a2
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render filtered audio.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
        Returns:
            Snippet containing filtered audio
        """
        # Ensure state is initialized
        if self._state is None or self._lfilter_state is None:
            self._reset_state()
        
        # Get source audio
        source_snippet = self._source.render(start, duration)
        x = source_snippet.data.astype(np.float64)
        channels = x.shape[1]
        
        # Ensure state has correct channel count
        if self._state.shape[1] != channels:
            self._state = np.zeros((4, channels), dtype=np.float64)
            self._lfilter_state = np.zeros((2, channels), dtype=np.float64)
        
        # Get parameter values
        freq_values = self._scalar_or_pe_values(self._frequency, start, duration, dtype=np.float64)
        q_values = self._scalar_or_pe_values(self._q, start, duration, dtype=np.float64)
        
        # Check if we can use optimized constant-coefficient path
        if not self._freq_is_pe and not self._q_is_pe:
            # Constant coefficients - faster path
            b0, b1, b2, a1, a2 = self._compute_coefficients(
                np.array([self._frequency]),
                np.array([self._q])
            )
            y = self._filter_constant_coeffs(
                x, b0[0], b1[0], b2[0], a1[0], a2[0]
            )
        else:
            # Time-varying coefficients - per-sample calculation
            b0, b1, b2, a1, a2 = self._compute_coefficients(freq_values, q_values)
            y = self._filter_varying_coeffs(x, b0, b1, b2, a1, a2)
        
        return Snippet(start, y.astype(np.float32))
    
    def _filter_constant_coeffs(
        self,
        x: np.ndarray,
        b0: float, b1: float, b2: float,
        a1: float, a2: float,
    ) -> np.ndarray:
        """
        Apply biquad filter with constant coefficients.
        
        Uses scipy.signal.lfilter for optimized C implementation.
        """
        # Coefficient arrays for lfilter
        b = np.array([b0, b1, b2], dtype=np.float64)
        a = np.array([1.0, a1, a2], dtype=np.float64)
        
        # Use scipy.signal.lfilter with state preservation
        # lfilter processes along axis=0 by default
        y, self._lfilter_state = signal.lfilter(
            b, a, x, axis=0, zi=self._lfilter_state
        )
        
        return y
    
    def _filter_varying_coeffs(
        self,
        x: np.ndarray,
        b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
        a1: np.ndarray, a2: np.ndarray,
    ) -> np.ndarray:
        """
        Apply biquad filter with time-varying coefficients.
        
        Uses Numba JIT when available, otherwise optimized numpy with
        vectorized feedforward computation.
        """
        duration, channels = x.shape
        
        # Extract state as contiguous arrays
        x1 = self._state[0, :].copy()
        x2 = self._state[1, :].copy()
        y1 = self._state[2, :].copy()
        y2 = self._state[3, :].copy()
        
        if NUMBA_AVAILABLE:
            # Use Numba-accelerated version
            y, x1, x2, y1, y2 = _biquad_varying_numba(
                x, b0, b1, b2, a1, a2, x1, x2, y1, y2
            )
        else:
            # Optimized numpy: vectorize feedforward, sequential feedback
            # Build padded input for shifted access
            # x_padded[0] = x2 (state), x_padded[1] = x1 (state), x_padded[2:] = x
            x_padded = np.vstack([x2.reshape(1, -1), x1.reshape(1, -1), x])
            
            # Vectorized feedforward computation
            # ff[n] = b0[n]*x[n] + b1[n]*x[n-1] + b2[n]*x[n-2]
            ff = (b0[:, np.newaxis] * x +
                  b1[:, np.newaxis] * x_padded[1:-1, :] +
                  b2[:, np.newaxis] * x_padded[:-2, :])
            
            # Sequential feedback loop (minimal work per iteration)
            y = np.zeros_like(x)
            for n in range(duration):
                y[n, :] = ff[n, :] - a1[n] * y1 - a2[n] * y2
                y2 = y1
                y1 = y[n, :]
            
            # Update x state from end of input
            x1 = x[-1, :].copy()
            x2 = x[-2, :].copy() if duration > 1 else x2
        
        # Save state
        self._state[0, :] = x1
        self._state[1, :] = x2
        self._state[2, :] = y1
        self._state[3, :] = y2
        
        return y
    
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
            f"BiquadPE(source={self._source.__class__.__name__}, "
            f"frequency={freq_str}, q={q_str}, mode={self._mode.value})"
        )
