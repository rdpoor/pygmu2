"""
BlitSawPE - Band-limited sawtooth oscillator using BLIT synthesis.

Uses the Stilson/Smith BLIT (Band-Limited Impulse Train) algorithm with
leaky integration to generate alias-free sawtooth waves.

Reference: "Alias-Free Digital Synthesis of Classic Analog Waveforms"
           Tim Stilson & Julius Smith, CCRMA, Stanford University

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet

from pygmu2.logger import get_logger
logger = get_logger(__name__)
logger.setLevel("DEBUG")

class BlitSawPE(ProcessingElement):
    """
    Band-limited sawtooth oscillator using BLIT synthesis.
    
    Generates alias-free sawtooth waves by integrating a band-limited
    impulse train (Dirichlet kernel). Supports time-varying frequency
    and harmonic count modulation.
    
    The sawtooth output ranges from approximately -1 to +1.
    
    Args:
        frequency: Frequency in Hz, or PE providing frequency values
        amplitude: Peak amplitude, or PE providing amplitude values (default: 1.0)
        initial_phase: Initial phase, internally set to initial_state % 1.0
        m: Number of harmonics. If None (default), automatically computed
           to keep all harmonics below Nyquist. Can be int or PE for
           fixed/modulated harmonic content.
        leak: Leaky integrator coefficient (default: 0.999). Controls
              DC rejection vs low-frequency response tradeoff.
        channels: Number of output channels (default: 1)
    
    Example:
        # Simple 440 Hz sawtooth (auto M, alias-free)
        saw_stream = BlitSawPE(frequency=440.0)
        
        # Fixed harmonics for consistent timbre during pitch sweep
        saw_stream = BlitSawPE(frequency=lfo_stream, m=20)
        
        # Frequency modulation (FM bass)
        mod_stream = SinePE(frequency=100.0, amplitude=50.0)
        bass_stream = BlitSawPE(frequency=GainPE(ConstantPE(100.0) + mod_stream, gain=1.0))
    
    Notes:
        - When m=None, M is computed per-sample as the largest odd integer
          such that M*frequency < Nyquist. This ensures no aliasing but
          may cause subtle timbral changes during pitch sweeps.
        - For consistent timbre across pitch changes, specify a fixed m
          value, but ensure m*max_frequency < sample_rate/2 to avoid aliasing.
        - The leak parameter trades off DC rejection (lower values) against
          low-frequency response (higher values). 0.999 works well for most cases.
    """
    
    def __init__(
        self,
        frequency: float | ProcessingElement,
        amplitude: float | ProcessingElement = 1.0,
        initial_phase: float = 0.0,
        m: int | ProcessingElement | None = None,
        leak: float = 0.999,
        channels: int = 1,
    ):
        self._frequency = frequency
        self._amplitude = amplitude
        self._initial_phase = initial_phase % 1.0
        self._m = m
        self._leak = leak
        self._channels = channels
        
        self._reset_state()
    
    @property
    def frequency(self) -> float | ProcessingElement:
        """Frequency in Hz (constant or PE)."""
        return self._frequency
    
    @property
    def amplitude(self) -> float | ProcessingElement:
        """Peak amplitude (constant or PE)."""
        return self._amplitude
    
    @property
    def m(self) -> int | ProcessingElement | None:
        """Number of harmonics (None for auto, or constant/PE)."""
        return self._m
    
    @property
    def leak(self) -> float:
        """Leaky integrator coefficient."""
        return self._leak
    
    @property
    def initial_phase(self) -> float:
        return self._initial_phase

    # @initial_phase.setter
    # def initial_phase(self, value:float) -> None:
    #     """takes effect on the next call to _reset_state()"""
    #     self._initial_phase = value % 1.0

    def inputs(self) -> list[ProcessingElement]:
        """Return list of PE inputs."""
        result = []
        if isinstance(self._frequency, ProcessingElement):
            result.append(self._frequency)
        if isinstance(self._amplitude, ProcessingElement):
            result.append(self._amplitude)
        if isinstance(self._m, ProcessingElement):
            result.append(self._m)
        return result
    
    def is_pure(self) -> bool:
        """
        BlitSawPE is never pure due to integrator state.
        """
        return False
    
    def channel_count(self) -> int:
        """Return the number of output channels."""
        return self._channels
    
    def _reset_state(self) -> None:
        """Reset oscillator state (phase and integrator)."""
        self._phase = self._initial_phase
        self._integrator = 0.0
        self._last_render_end = None
        logger.debug(f"_reset_state: initial_phase = {self._initial_phase}")
    
    def _on_start(self) -> None:
        """Reset state on start."""
        self._reset_state()

    def _on_stop(self) -> None:
        """Reset state on stop."""
        self._reset_state()
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Generate band-limited sawtooth samples.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet containing sawtooth wave data
        """
        # Get parameter values (1D control vectors)
        freq = self._scalar_or_pe_values(self._frequency, start, duration, dtype=np.float64)
        amp = self._scalar_or_pe_values(self._amplitude, start, duration, dtype=np.float64)
        
        # Handle M: auto-compute or use provided value
        if self._m is None:
            # Auto: largest odd M keeping harmonics below Nyquist
            # M * freq < sample_rate / 2  =>  M < sample_rate / (2 * freq)
            m_float = self.sample_rate / (2.0 * np.maximum(freq, 1.0))
            m = np.floor(m_float).astype(np.int32)
            # Make odd (if even, subtract 1)
            m = m - (1 - m % 2)
            m = np.maximum(m, 1)  # At least 1 harmonic
        else:
            m_vals = self._scalar_or_pe_values(self._m, start, duration, dtype=np.float64)
            m = np.maximum(m_vals.astype(np.int32), 1).astype(np.float64)
        
        # Ensure arrays for vectorized computation (already 1D)
        m = np.atleast_1d(m).astype(np.float64, copy=False)
        
        # Handle discontinuous rendering
        if self._last_render_end is None or start != self._last_render_end:
            logger.debug("discontinuous rendering, resetting...")
            self._phase = self._initial_phase
            self._integrator = 0.0
        
        # Compute phase increment per sample (0 to 1 per period)
        phase_inc = freq / self.sample_rate
        
        # Accumulate phase using cumsum
        phase = self._phase + np.cumsum(phase_inc)
        
        # Wrap phase to [0, 1)
        phase = np.mod(phase, 1.0)
        
        # Period in samples (for normalization)
        P = self.sample_rate / np.maximum(freq, 1.0)
        
        # BLIT computation using Dirichlet kernel
        # blit = sin(π * M * phase) / (P * sin(π * phase))
        # At phase ≈ 0: use limit M / P
        theta = np.pi * phase
        m_theta = m * theta
        
        sin_num = np.sin(m_theta)
        sin_den = np.sin(theta)
        
        # Handle singularity near phase = 0 or 1
        eps = 1e-9
        near_zero = np.abs(sin_den) < eps
        
        # BLIT value (normalized by period)
        blit = np.where(near_zero, m / P, sin_num / (P * sin_den))
        
        # Remove DC component: average value of BLIT is 1/P
        # (one impulse of area 1 per period P)
        blit_ac = blit - 1.0 / P
        
        # Leaky integration to produce sawtooth
        # y[n] = x[n] + leak * y[n-1]
        # This is a one-pole IIR filter: lfilter(b=[1], a=[1, -leak], x)
        leak = self._leak
        
        try:
            from scipy.signal import lfilter
            # Transfer function: H(z) = 1 / (1 - leak*z^-1)
            # b = [1.0], a = [1.0, -leak]
            b = np.array([1.0])
            a = np.array([1.0, -leak])
            
            # Initial condition for lfilter: zi = [leak * y_initial]
            # This ensures continuity from previous render
            zi = np.array([leak * self._integrator])
            saw, zf = lfilter(b, a, blit_ac, zi=zi)
            
            # Update integrator state from final filter state
            y = zf[0] / leak if leak != 0 else saw[-1]
        except ImportError:
            # Fallback to Python loop if scipy not available
            saw = np.zeros(duration, dtype=np.float64)
            y = self._integrator
            for i in range(duration):
                y = blit_ac[i] + leak * y
                saw[i] = y
        
        # Update state
        self._phase = phase[-1]
        self._integrator = saw[-1]  # Use actual output value for state
        self._last_render_end = start + duration
        
        # Scale output to approximately ±1
        # The integrated BLIT produces a sawtooth with amplitude ≈ 0.5
        # Scale by 2 to get ±1 range
        saw = saw * 2.0
        
        # Apply amplitude
        samples = saw * amp
        
        # Reshape to (duration, channels)
        samples = samples.reshape(-1, 1).astype(np.float32)
        if self._channels > 1:
            samples = np.tile(samples, (1, self._channels))
        
        return Snippet(start, samples)
    
    def _compute_extent(self) -> Extent:
        """
        Compute extent from PE inputs.
        
        If all inputs are constants: infinite extent.
        If any input is a PE: intersection of input extents.
        """
        result = Extent(None, None)
        for pe_input in self.inputs():
            input_extent = pe_input.extent()
            result = result.intersection(input_extent)
        return result
    
    def __repr__(self) -> str:
        freq_str = (
            f"{self._frequency.__class__.__name__}"
            if isinstance(self._frequency, ProcessingElement)
            else str(self._frequency)
        )
        amp_str = (
            f"{self._amplitude.__class__.__name__}"
            if isinstance(self._amplitude, ProcessingElement)
            else str(self._amplitude)
        )
        m_str = (
            "auto" if self._m is None
            else f"{self._m.__class__.__name__}"
            if isinstance(self._m, ProcessingElement)
            else str(self._m)
        )
        return (
            f"BlitSawPE(frequency={freq_str}, amplitude={amp_str}, "
            f"m={m_str}, leak={self._leak}, channels={self._channels})"
        )
