"""
EnvelopePE - causal envelope follower with optional lookahead.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from enum import Enum
from typing import Optional

import numpy as np
import numba as nb

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class DetectionMode(Enum):
    """Envelope detection modes."""
    PEAK = "peak"  # Absolute value (rectified)
    RMS = "rms"    # Root mean square over small window


class EnvelopePE(ProcessingElement):
    """
    Causal envelope follower with attack/release dynamics and optional lookahead.
    
    Tracks the amplitude envelope of an input signal using classic attack/release
    timing. The optional lookahead allows the envelope to start rising before
    a transient arrives, eliminating the lag inherent in real-time systems.
    
    The envelope rises toward the input level with attack time constant, and
    falls with release time constant. With lookahead, the detection point is
    shifted forward in time, so the envelope anticipates upcoming peaks.
    
    Args:
        source: Input audio PE
        attack: Attack time in seconds (how fast envelope rises)
        release: Release time in seconds (how fast envelope falls)
        lookahead: Lookahead time in seconds (clamped to attack time)
        mode: Detection mode (PEAK or RMS)
    
    Example:
        # Basic envelope follower
        env_stream = EnvelopePE(source_stream, attack=0.01, release=0.1)
        
        # With lookahead for zero-latency attack
        env_stream = EnvelopePE(source_stream, attack=0.01, release=0.1, lookahead=0.01)
        
        # RMS-based detection (smoother)
        env_stream = EnvelopePE(source_stream, attack=0.01, release=0.1, mode=DetectionMode.RMS)
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        attack: float = 0.01,
        release: float = 0.1,
        lookahead: float = 0.0,
        mode: DetectionMode = DetectionMode.PEAK,
    ):
        self._source = source
        self._attack = max(0.0, attack)
        self._release = max(0.0, release)
        # Clamp lookahead to attack time
        self._lookahead = max(0.0, min(lookahead, self._attack))
        self._mode = mode
        
        # Envelope state (per channel)
        self._envelope: Optional[np.ndarray] = None
    
    @property
    def source(self) -> ProcessingElement:
        """The input audio PE."""
        return self._source
    
    @property
    def attack(self) -> float:
        """Attack time in seconds."""
        return self._attack
    
    @property
    def release(self) -> float:
        """Release time in seconds."""
        return self._release
    
    @property
    def lookahead(self) -> float:
        """Lookahead time in seconds."""
        return self._lookahead
    
    @property
    def mode(self) -> DetectionMode:
        """Detection mode (PEAK or RMS)."""
        return self._mode
    
    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs."""
        return [self._source]
    
    def is_pure(self) -> bool:
        """
        EnvelopePE is NOT pure - it maintains envelope state.
        """
        return False
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """Return the extent of this PE (matches source)."""
        return self._source.extent()
    
    def _reset_state(self) -> None:
        """Reset envelope state."""
        self._envelope = None
    
    def _on_start(self) -> None:
        """Reset envelope state at start of rendering."""
        self._reset_state()

    def _on_stop(self) -> None:
        """Clear envelope state at end of rendering."""
        self._reset_state()
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render the envelope.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
        Returns:
            Snippet containing envelope values (always positive)
        """
        sample_rate = self.sample_rate
        
        # Calculate lookahead in samples
        lookahead_samples = int(self._lookahead * sample_rate)
        
        # Get source audio (shifted forward by lookahead)
        source_snippet = self._source.render(start + lookahead_samples, duration)
        x = np.abs(source_snippet.data.astype(np.float64))
        
        channels = x.shape[1]
        
        # For RMS mode, compute windowed RMS
        if self._mode == DetectionMode.RMS:
            # Use a small window (about 10ms or attack time, whichever is smaller)
            rms_window = max(1, int(min(0.01, self._attack) * sample_rate))
            x = self._compute_rms(x, rms_window)
        
        # Initialize envelope state if needed
        if self._envelope is None:
            self._envelope = np.zeros(channels, dtype=np.float64)
        
        # Ensure envelope has correct channel count
        if len(self._envelope) != channels:
            self._envelope = np.zeros(channels, dtype=np.float64)
        
        # Calculate attack/release coefficients
        # Using exponential smoothing: coeff = 1 - exp(-1 / (time * sample_rate))
        if self._attack > 0:
            attack_coeff = 1.0 - np.exp(-1.0 / (self._attack * sample_rate))
        else:
            attack_coeff = 1.0  # Instant attack
        
        if self._release > 0:
            release_coeff = 1.0 - np.exp(-1.0 / (self._release * sample_rate))
        else:
            release_coeff = 1.0  # Instant release
        
        # Process envelope
        output = np.zeros((duration, channels), dtype=np.float64)
        env = self._envelope.copy()
        
        # Optimization: if attack == release, use scipy.signal.lfilter (much faster)
        if self._attack == self._release and attack_coeff < 1.0:
            try:
                from scipy.signal import lfilter
                # One-pole lowpass: y[n] = (1-coeff)*y[n-1] + coeff*x[n]
                # Transfer function: b = [coeff], a = [1, -(1-coeff)]
                b = np.array([attack_coeff])
                a = np.array([1.0, -(1.0 - attack_coeff)])
                
                for ch in range(channels):
                    # zi is the initial condition (scaled for lfilter's convention)
                    zi = np.array([env[ch] * (1.0 - attack_coeff)])
                    output[:, ch], zf = lfilter(b, a, x[:, ch], zi=zi)
                    env[ch] = output[-1, ch]
                
                self._envelope = env
                return Snippet(start, output.astype(np.float32))
            except ImportError:
                pass  # Fall through to manual implementation
        
        # For asymmetric attack/release, use simple loop
        _envelope_ar_numba(x, attack_coeff, release_coeff, env, output)
        
        # Save state
        self._envelope = env
        
        return Snippet(start, output.astype(np.float32))

    def _compute_rms(self, x: np.ndarray, window: int) -> np.ndarray:
        """
        Compute windowed RMS of the signal.
        
        Vectorized implementation using cumulative sums.
        """
        duration, channels = x.shape
        half_window = window // 2
        
        # Square the signal
        x_squared = x ** 2
        
        result = np.zeros_like(x)
        
        # Try scipy.ndimage.uniform_filter1d for optimal performance
        try:
            from scipy.ndimage import uniform_filter1d
            for ch in range(channels):
                # uniform_filter1d computes the mean, so we apply to squared signal
                mean_squared = uniform_filter1d(x_squared[:, ch], size=window, mode='nearest')
                result[:, ch] = np.sqrt(mean_squared)
            return result
        except ImportError:
            pass
        
        # Fallback: vectorized cumulative sum approach
        for ch in range(channels):
            cumsum = np.cumsum(np.concatenate([[0], x_squared[:, ch]]))
            
            # Compute start and end indices for all positions
            # Handle boundary conditions
            n = np.arange(duration)
            start_idx = np.maximum(0, n - half_window)
            end_idx = np.minimum(duration, n + half_window + 1)
            
            # Vectorized window sums
            window_sum = cumsum[end_idx] - cumsum[start_idx]
            window_len = end_idx - start_idx
            
            result[:, ch] = np.sqrt(window_sum / window_len)
        
        return result
    
    def __repr__(self) -> str:
        return (
            f"EnvelopePE(source={self._source.__class__.__name__}, "
            f"attack={self._attack}, release={self._release}, "
            f"lookahead={self._lookahead}, mode={self._mode.value})"
        )


@nb.njit(cache=True)
def _envelope_ar_numba(x, attack_coeff, release_coeff, env, output):
    n_samples, channels = x.shape
    for n in range(n_samples):
        for ch in range(channels):
            target = x[n, ch]
            e = env[ch]
            if target > e:
                e = e + attack_coeff * (target - e)
            else:
                e = e + release_coeff * (target - e)
            env[ch] = e
            output[n, ch] = e
