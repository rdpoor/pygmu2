"""
RampPE - a source that outputs a ramp (linear, exponential, or sigmoid).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.processing_element import SourcePE
from pygmu2.extent import Extent, ExtendMode
from pygmu2.snippet import Snippet


class RampType:
    """Curve shape for RampPE."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"
    CONSTANT_POWER = "constant_power"


class RampPE(SourcePE):
    """
    A SourcePE that outputs a ramp from start_value to end_value.

    The ramp spans a fixed number of samples. Behavior outside this range
    is controlled by extend_mode. The curve shape is controlled by ramp_type.

    Useful for:
    - Envelope generation (attack/decay ramps)
    - Parameter automation
    - Crossfades (exponential or sigmoid often sound smoother than linear)
    - Testing
    - Portamento effects (with HOLD_BOTH)

    Args:
        start_value: Value at the beginning of the ramp
        end_value: Value at the end of the ramp
        duration: Length of the ramp in samples
        channels: Number of output channels (default: 1)
        extend_mode: Behavior outside ramp range (default: ZERO)
                     - ZERO: Output zeros outside ramp
                     - HOLD_FIRST: Hold start_value before ramp
                     - HOLD_LAST: Hold end_value after ramp
                     - HOLD_BOTH: Hold start_value before, end_value after
        ramp_type: Curve shape (default: LINEAR)
                   - LINEAR: Linear interpolation
                   - EXPONENTIAL: Exponential curve (start*(end/start)^t when both > 0)
                   - SIGMOID: S-curve (slow at ends, fast in middle)
                   - CONSTANT_POWER: sin(π/2·t) so fade-in and fade-out sum to constant
                     power (sin²+cos²=1) when used as a crossfade pair

    Example:
        # Fade in over 1 second at 44100 Hz
        fade_in_stream = RampPE(0.0, 1.0, duration=44100)

        # Exponential crossfade (often better perceived balance)
        fade = RampPE(0.0, 1.0, duration=44100, ramp_type=RampType.EXPONENTIAL)

        # Sigmoid for smooth parameter sweep
        sweep = RampPE(220.0, 880.0, duration=44100, ramp_type=RampType.SIGMOID)

        # Ramp that holds values outside its range (useful for portamento)
        from pygmu2 import ExtendMode
        portamento_stream = RampPE(440.0, 880.0, duration=1000, extend_mode=ExtendMode.HOLD_BOTH)
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        duration: int,
        *,
        channels: int = 1,
        extend_mode: ExtendMode = ExtendMode.ZERO,
        ramp_type: str = RampType.LINEAR,
    ):
        self._start_value = float(start_value)
        self._end_value = float(end_value)
        self._duration = int(duration)
        self._channels = int(channels)
        self._extend_mode = extend_mode
        self._ramp_type = str(ramp_type).lower() if ramp_type else RampType.LINEAR

        if self._duration < 1:
            raise ValueError(f"RampPE duration must be >= 1, got {duration}")

        # Pre-compute the full ramp for efficiency
        t = np.linspace(0, 1, self._duration, dtype=np.float32)
        self._ramp = self._compute_curve(t).astype(np.float32)

    def _compute_curve(self, t: np.ndarray) -> np.ndarray:
        """Map t in [0, 1] to ramp values based on ramp_type."""
        if self._ramp_type == RampType.LINEAR:
            return self._start_value + (self._end_value - self._start_value) * t

        if self._ramp_type == RampType.EXPONENTIAL:
            # start * (end/start)^t; use linear if start or end is 0 or sign change
            start, end = self._start_value, self._end_value
            if start <= 0 or end <= 0 or (start > 0) != (end > 0):
                return start + (end - start) * t
            ratio = end / start
            return start * (ratio ** t)

        if self._ramp_type == RampType.SIGMOID:
            # S-curve: logistic over t in [0,1], steepness 6
            k = 6.0
            x = k * (2.0 * t - 1.0)
            # 1/(1+exp(-x)) maps (-inf, inf) -> (0, 1); clip for numerical safety
            sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
            return self._start_value + (self._end_value - self._start_value) * sig

        if self._ramp_type == RampType.CONSTANT_POWER:
            # Fade-in: sin(π/2·t). Fade-out: multiplier 1-cos(π/2·t) so value = cos(π/2·t); sin²+cos²=1.
            if self._end_value >= self._start_value:
                curve = np.sin(0.5 * np.pi * t)
            else:
                curve = 1.0 - np.cos(0.5 * np.pi * t)
            return self._start_value + (self._end_value - self._start_value) * curve

        # Unknown ramp_type: fall back to linear
        return self._start_value + (self._end_value - self._start_value) * t
    
    @property
    def start_value(self) -> float:
        """Value at the beginning of the ramp."""
        return self._start_value
    
    @property
    def end_value(self) -> float:
        """Value at the end of the ramp."""
        return self._end_value
    
    @property
    def ramp_duration(self) -> int:
        """Length of the ramp in samples."""
        return self._duration
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Generate ramp samples for the given range.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet containing ramp data. Behavior outside ramp extent depends
            on extend_mode.
        """
        # Initialize with zeros
        data = np.zeros((duration, self._channels), dtype=np.float32)
        
        # Calculate overlap with ramp extent
        ramp_start = 0
        ramp_end = self._duration
        
        # Request range
        req_start = start
        req_end = start + duration
        
        # Find intersection
        overlap_start = max(ramp_start, req_start)
        overlap_end = min(ramp_end, req_end)
        
        if overlap_start < overlap_end:
            # There is overlap - copy ramp values
            ramp_slice_start = overlap_start - ramp_start
            ramp_slice_end = overlap_end - ramp_start
            
            output_slice_start = overlap_start - req_start
            output_slice_end = overlap_end - req_start
            
            ramp_values = self._ramp[ramp_slice_start:ramp_slice_end]
            data[output_slice_start:output_slice_end, :] = ramp_values.reshape(-1, 1)
        
        # Handle held values outside ramp extent
        if self._extend_mode in (ExtendMode.HOLD_FIRST, ExtendMode.HOLD_BOTH):
            # Before ramp: hold start_value
            if req_start < ramp_start:
                before_count = min(duration, ramp_start - req_start)
                data[:before_count, :] = self._start_value
        
        if self._extend_mode in (ExtendMode.HOLD_LAST, ExtendMode.HOLD_BOTH):
            # After ramp: hold end_value
            if req_end > ramp_end:
                after_start = max(0, ramp_end - req_start)
                if after_start < duration:
                    data[after_start:, :] = self._end_value
        
        return Snippet(start, data)
    
    def _compute_extent(self) -> Extent:
        """
        Return the extent of the ramp.
        
        If extend_mode holds values (HOLD_FIRST, HOLD_LAST, or HOLD_BOTH),
        the extent is infinite. Otherwise (ZERO), the extent is finite.
        """
        if self._extend_mode != ExtendMode.ZERO:
            return Extent(None, None)  # Infinite extent (holds values)
        return Extent(0, self._duration)  # Finite extent (zeros outside)
    
    def channel_count(self) -> int:
        """Return the number of output channels."""
        return self._channels
    
    @property
    def extend_mode(self) -> ExtendMode:
        """Behavior for samples outside the ramp range."""
        return self._extend_mode

    @property
    def ramp_type(self) -> str:
        """Curve shape: LINEAR, EXPONENTIAL, or SIGMOID."""
        return self._ramp_type

    def __repr__(self) -> str:
        parts = [
            f"start_value={self._start_value}",
            f"end_value={self._end_value}",
            f"duration={self._duration}",
            f"channels={self._channels}",
        ]
        if self._extend_mode != ExtendMode.ZERO:
            parts.append(f"extend_mode={self._extend_mode.value}")
        if self._ramp_type != RampType.LINEAR:
            parts.append(f"ramp_type={self._ramp_type!r}")
        return f"RampPE({', '.join(parts)})"
