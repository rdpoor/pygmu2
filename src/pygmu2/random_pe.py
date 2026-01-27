"""
RandomPE - Musically useful random value generator.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

from enum import Enum
from typing import Optional, Union
import numpy as np

from pygmu2.processing_element import ProcessingElement, SourcePE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class RandomMode(Enum):
    """Interpolation modes for random value generation."""
    
    SAMPLE_HOLD = "sample_hold"  # Jump to new value, hold until next
    LINEAR = "linear"            # Linear ramp between random values
    SMOOTH = "smooth"            # Smooth (cosine) interpolation
    WALK = "walk"                # Random walk with bounded drift


class RandomPE(SourcePE):
    """
    Musically useful random value generator.
    
    Generates random values with various interpolation modes, suitable for
    modulating other PE parameters. Can be rate-based (Hz) or triggered
    by an external signal.
    
    Args:
        rate: How often new random values are generated (Hz). Ignored if
            trigger is provided.
        min_value: Minimum output value (default: 0.0)
        max_value: Maximum output value (default: 1.0)
        mode: Interpolation mode (default: SAMPLE_HOLD)
        seed: Random seed for reproducibility. If None, uses system entropy.
        slew: Maximum change per sample in WALK mode, as fraction of range.
            Value of 1.0 means full range in one sample. (default: 0.01)
        trigger: Optional PE that triggers new values. When provided, a new
            random value is generated on each rising edge (signal crosses
            from <= 0 to > 0). The rate parameter is ignored when trigger
            is provided.
    
    Modes:
        SAMPLE_HOLD: Classic sample-and-hold. Jumps to new random value,
            holds until next trigger/rate tick.
        LINEAR: Linear interpolation (ramp) between consecutive random values.
        SMOOTH: Smooth cosine interpolation for organic movement.
        WALK: Random walk - each step adds/subtracts a small random amount,
            bounded by min_value/max_value.
    
    Example:
        # Smooth random LFO for vibrato depth
        depth_stream = RandomPE(rate=0.5, min_value=5, max_value=20, mode=RandomMode.SMOOTH)
        
        # Triggered sample-and-hold for random notes
        trigger_stream = SomeTriggerPE(...)
        random_note_stream = RandomPE(min_value=48, max_value=72, trigger=trigger_stream)
        
        # Random walk for gentle drift
        drift_stream = RandomPE(rate=10, min_value=-0.1, max_value=0.1, 
                        mode=RandomMode.WALK, slew=0.001)
    """
    
    def __init__(
        self,
        rate: float = 1.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        mode: RandomMode = RandomMode.SAMPLE_HOLD,
        seed: Optional[int] = None,
        slew: float = 0.01,
        trigger: Optional[ProcessingElement] = None,
    ):
        self._rate = rate
        self._min_value = min_value
        self._max_value = max_value
        self._mode = mode
        self._seed = seed
        self._slew = slew
        self._trigger = trigger
        
        # State (initialized in on_start)
        self._rng: Optional[np.random.Generator] = None
        self._sample_rate: Optional[int] = None
        self._current_value: float = 0.0
        self._next_value: float = 0.0
        self._phase: float = 0.0  # 0-1, progress between values
        self._samples_per_period: int = 0
        self._sample_counter: int = 0
        self._last_trigger: float = 0.0  # For edge detection
    
    @property
    def rate(self) -> float:
        """Rate of random value changes in Hz."""
        return self._rate
    
    @property
    def min_value(self) -> float:
        """Minimum output value."""
        return self._min_value
    
    @property
    def max_value(self) -> float:
        """Maximum output value."""
        return self._max_value
    
    @property
    def mode(self) -> RandomMode:
        """Interpolation mode."""
        return self._mode
    
    @property
    def seed(self) -> Optional[int]:
        """Random seed, or None if using system entropy."""
        return self._seed
    
    @property
    def slew(self) -> float:
        """Maximum change per sample in WALK mode."""
        return self._slew
    
    @property
    def trigger(self) -> Optional[ProcessingElement]:
        """Trigger PE, or None if rate-based."""
        return self._trigger
    
    def inputs(self) -> list[ProcessingElement]:
        """Return list of input PEs."""
        if self._trigger is not None:
            return [self._trigger]
        return []
    
    def is_pure(self) -> bool:
        """RandomPE is stateful (maintains RNG and interpolation state)."""
        return False
    
    def channel_count(self) -> Optional[int]:
        """Output is mono."""
        return 1
    
    def _compute_extent(self) -> Extent:
        """Random generator runs forever."""
        return Extent(None, None)
    
    def _reset_state(self) -> None:
        """Reset random generator state and current values."""
        if self._rng is None:
            # Not initialized yet - initialize now
            self._rng = np.random.default_rng(self._seed)
            self._sample_rate = self.sample_rate
            if self._rate > 0:
                self._samples_per_period = max(1, int(self._sample_rate / self._rate))
            else:
                self._samples_per_period = self._sample_rate  # Default to 1 Hz
        else:
            # Re-seed the RNG to reset its state
            self._rng = np.random.default_rng(self._seed)
        
        # Reset current values
        self._current_value = self._random_value()
        self._next_value = self._random_value()
        self._phase = 0.0
        self._sample_counter = 0
        self._last_trigger = 0.0
    
    def on_start(self) -> None:
        """Initialize RNG and state."""
        self._reset_state()
    
    def on_stop(self) -> None:
        """Clean up state."""
        self._rng = None
    
    def _random_value(self) -> float:
        """Generate a random value in [min_value, max_value]."""
        return self._rng.uniform(self._min_value, self._max_value)
    
    def _random_step(self) -> float:
        """Generate a random step for WALK mode."""
        range_size = self._max_value - self._min_value
        max_step = range_size * self._slew
        return self._rng.uniform(-max_step, max_step)
    
    def _interpolate(self, phase: float) -> float:
        """Interpolate between current and next value based on mode."""
        if self._mode == RandomMode.SAMPLE_HOLD:
            return self._current_value
        elif self._mode == RandomMode.LINEAR:
            return self._current_value + phase * (self._next_value - self._current_value)
        elif self._mode == RandomMode.SMOOTH:
            # Cosine interpolation for smooth transitions
            t = (1.0 - np.cos(phase * np.pi)) / 2.0
            return self._current_value + t * (self._next_value - self._current_value)
        else:  # WALK mode handled separately
            return self._current_value
    
    def _advance_to_next(self) -> None:
        """Move to next random value."""
        self._current_value = self._next_value
        self._next_value = self._random_value()
        self._phase = 0.0
        self._sample_counter = 0
    
    def _walk_step(self) -> float:
        """Take one step in random walk mode."""
        step = self._random_step()
        new_value = self._current_value + step
        # Clamp to bounds
        new_value = max(self._min_value, min(self._max_value, new_value))
        self._current_value = new_value
        return new_value
    
    def _render(self, start: int, duration: int) -> Snippet:
        """Generate random values."""
        
        output = np.zeros((duration, 1), dtype=np.float64)
        
        # Get trigger signal if provided
        trigger_data = None
        if self._trigger is not None:
            trigger_snippet = self._trigger.render(start, duration)
            trigger_data = trigger_snippet.data[:, 0]  # Use first channel
        
        if self._mode == RandomMode.WALK:
            # Walk mode: each sample takes a random step
            for i in range(duration):
                # Check for trigger
                if trigger_data is not None:
                    current_trigger = trigger_data[i]
                    if current_trigger > 0 and self._last_trigger <= 0:
                        # Rising edge - reset to new random value
                        self._current_value = self._random_value()
                    self._last_trigger = current_trigger
                    output[i, 0] = self._current_value
                else:
                    output[i, 0] = self._walk_step()
        else:
            # Other modes: interpolate between random values
            for i in range(duration):
                if trigger_data is not None:
                    # Trigger-based: advance on rising edge
                    current_trigger = trigger_data[i]
                    if current_trigger > 0 and self._last_trigger <= 0:
                        # Rising edge detected
                        self._advance_to_next()
                    self._last_trigger = current_trigger
                    
                    # For triggered mode, phase doesn't advance automatically
                    # Use sample_hold-like behavior between triggers
                    output[i, 0] = self._current_value
                else:
                    # Rate-based: advance based on samples_per_period
                    output[i, 0] = self._interpolate(self._phase)
                    
                    self._sample_counter += 1
                    self._phase = self._sample_counter / self._samples_per_period
                    
                    if self._sample_counter >= self._samples_per_period:
                        self._advance_to_next()
        
        return Snippet(start, output.astype(np.float32))
    
    def __repr__(self) -> str:
        trigger_str = ", triggered" if self._trigger else f", rate={self._rate}"
        return (f"RandomPE(mode={self._mode.value}, "
                f"range=[{self._min_value}, {self._max_value}]{trigger_str})")
