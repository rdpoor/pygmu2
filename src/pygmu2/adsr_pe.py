"""
AdsrPE - Attack-Decay-Sustain-Release envelope generator.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from enum import Enum
from typing import Optional

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.logger import get_logger

logger = get_logger(__name__)


DEFAULT_ATTACK_SECONDS = 0.01
DEFAULT_DECAY_SECONDS = 0.10
DEFAULT_RELEASE_SECONDS = 0.20


class AdsrState(Enum):
    """Internal ADSR state."""
    IDLE = "idle"          # Output is 0, waiting for gate
    ATTACK = "attack"       # Ramping from current value to 1.0
    DECAY = "decay"         # Ramping from 1.0 to sustain_level
    SUSTAIN = "sustain"     # Holding at sustain_level
    RELEASE = "release"     # Ramping from current value to 0.0


class AdsrPE(ProcessingElement):
    """
    [Attack - Decay - Sustain - Release] control signal source.

    Implements the classic ADSR control module, popular in audio synthesizers.

    When the gate input transitions to a positive value, the output ramps from
    0.0 to 1.0 over the given attack time.  It then decays from 1.0 to the 
    sustain value and stays at that value until the gate signal transitions to
    a non-positive value, at which point the output ramps from the sustain 
    value down to zero over the release time.

    Some edge cases: 

    If the gate signal transitions to a non-positive value before the output 
    reaches the sustain value, the output will ramp down from its current value
    to 0.0 over the given release time.

    If the ADSR is re-triggered before reaching a zero value output, the output
    will start ramping up from its current value at a rate of 1.0/attack_samples
    until it reaches a value of 1.0, at which point it enters the decay state.

    This processing element maintains internal state so is_pure() returns False.
    Any internal state is reset on on_start() and on_stop().
    
    Args:
        gate: gate input PE
        attack_samples: attack time in samples (optional)
        attack_seconds: attack time in seconds (optional)
        decay_samples: decay time in samples (optional)
        decay_seconds: decay time in seconds (optional)
        sustain_level: output value to hold until gate signal goes non-positive
        release_samples: release time in samples (optional)
        release_seconds: release time in seconds (optional)
    """
    
    def __init__(
        self,
        gate: ProcessingElement,
        attack_samples: Optional[int] = None,
        attack_seconds: Optional[float] = None,
        decay_samples: Optional[int] = None,
        decay_seconds: Optional[float] = None,
        sustain_level: float = 0.7,
        release_samples: Optional[int] = None,
        release_seconds: Optional[float] = None,
    ):
        self._gate = gate
        self._attack_samples = attack_samples
        self._attack_seconds = attack_seconds
        self._decay_samples = decay_samples
        self._decay_seconds = decay_seconds
        self._sustain_level = max(0.0, min(1.0, sustain_level))
        self._release_samples = release_samples
        self._release_seconds = release_seconds

        # Resolved times (in samples). These are set in configure() once the
        # sample rate is known (needed for *_seconds defaults/conversion).
        self._attack = 1
        self._decay = 1
        self._release = 1
        
        # State (will be initialized in on_start)
        self._state: Optional[AdsrState] = None
        self._value: float = 0.0  # Current output value
        self._phase_start_time: int = 0  # Sample time when current phase started
        self._phase_start_value: float = 0.0  # Value at start of current phase
        self._prev_gate: float = 0.0  # Previous gate value (for transition detection)
    
    @property
    def gate(self) -> ProcessingElement:
        """The gate input PE."""
        return self._gate
    
    @property
    def attack(self) -> int:
        """Attack time in samples."""
        return self._attack
    
    @property
    def decay(self) -> int:
        """Decay time in samples."""
        return self._decay
    
    @property
    def sustain_level(self) -> float:
        """Sustain level (0.0 to 1.0)."""
        return self._sustain_level
    
    @property
    def release(self) -> int:
        """Release time in samples."""
        return self._release
    
    def inputs(self) -> list[ProcessingElement]:
        """Return the gate input."""
        return [self._gate]
    
    def is_pure(self) -> bool:
        """AdsrPE is not pure - it maintains internal state."""
        return False
    
    def channel_count(self) -> Optional[int]:
        """Return channel count from gate (assumed mono for control signal)."""
        # ADSR typically outputs mono control signal
        gate_channels = self._gate.channel_count()
        if gate_channels is not None:
            # Use first channel of gate for state machine
            return 1
        return 1  # Default to mono
    
    def _compute_extent(self) -> Extent:
        """Return infinite extent (gate-driven, no fixed duration)."""
        return Extent(None, None)
    
    def _reset_state(self) -> None:
        """Reset ADSR state machine."""
        self._state = AdsrState.IDLE
        self._value = 0.0
        self._phase_start_time = 0
        self._phase_start_value = 0.0
        self._prev_gate = 0.0
    
    def on_start(self) -> None:
        """Reset state at start of rendering."""
        self._reset_state()
    
    def on_stop(self) -> None:
        """Reset state at end of rendering."""
        self._reset_state()

    def configure(self, sample_rate: int) -> None:
        """
        Configure with sample rate and resolve optional *_seconds parameters.
        """
        super().configure(sample_rate)

        def _resolve_time(
            *,
            samples: Optional[int],
            seconds: Optional[float],
            default_seconds: float,
            name: str,
        ) -> int:
            # Defaults are specified in seconds so they are independent of the
            # configured sample rate.
            if samples is None and seconds is None:
                seconds = default_seconds
            return max(1, int(self._time_to_samples(samples=samples, seconds=seconds, name=name)))

        self._attack = _resolve_time(
            samples=self._attack_samples,
            seconds=self._attack_seconds,
            default_seconds=DEFAULT_ATTACK_SECONDS,
            name="attack",
        )
        self._decay = _resolve_time(
            samples=self._decay_samples,
            seconds=self._decay_seconds,
            default_seconds=DEFAULT_DECAY_SECONDS,
            name="decay",
        )
        self._release = _resolve_time(
            samples=self._release_samples,
            seconds=self._release_seconds,
            default_seconds=DEFAULT_RELEASE_SECONDS,
            name="release",
        )
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render ADSR envelope based on gate signal.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet containing the ADSR envelope values
        """
        # Get gate signal (use first channel)
        gate_snippet = self._gate.render(start, duration)
        gate_signal = gate_snippet.data[:, 0] if gate_snippet.data.shape[1] > 0 else np.zeros(duration)
        
        # Initialize output
        output = np.zeros((duration, 1), dtype=np.float32)
        
        # Process sample by sample (state machine)
        for i in range(duration):
            current_time = start + i
            gate_value = gate_signal[i]
            
            # Log gate value changes
            if gate_value != self._prev_gate:
                logger.debug(f"sample={current_time}: gate={gate_value:.3f} (was {self._prev_gate:.3f})")
            
            gate_rising = gate_value > 0.0 and self._prev_gate <= 0.0
            gate_falling = gate_value <= 0.0 and self._prev_gate > 0.0
            
            old_state = self._state
            old_value = self._value
            
            # State transitions
            if gate_rising:
                # Gate just went high - start attack (or restart if already active)
                if self._state == AdsrState.RELEASE:
                    # Re-triggered during release: start attack from current value
                    logger.debug(f"sample={current_time}: RELEASE -> ATTACK (re-trigger, value={self._value:.3f})")
                    self._state = AdsrState.ATTACK
                    self._phase_start_time = current_time
                    self._phase_start_value = self._value
                elif self._state == AdsrState.IDLE:
                    # Starting from idle - ensure we start from 0.0
                    logger.debug(f"sample={current_time}: IDLE -> ATTACK (gate rising, value=0.000)")
                    self._state = AdsrState.ATTACK
                    self._phase_start_time = current_time
                    self._phase_start_value = 0.0
                    self._value = 0.0  # Explicitly reset value when starting from IDLE
                # If already in ATTACK, DECAY, or SUSTAIN, continue (ignore re-trigger)
            
            elif gate_falling:
                # Gate just went low - enter release
                if self._state in (AdsrState.ATTACK, AdsrState.DECAY, AdsrState.SUSTAIN):
                    logger.debug(f"sample={current_time}: {old_state.value} -> RELEASE (gate falling, value={self._value:.3f})")
                    self._state = AdsrState.RELEASE
                    self._phase_start_time = current_time
                    self._phase_start_value = self._value
                # If already in RELEASE or IDLE, continue
            
            # State updates (compute current value based on phase)
            phase_elapsed = current_time - self._phase_start_time
            
            if self._state == AdsrState.IDLE:
                self._value = 0.0
            
            elif self._state == AdsrState.ATTACK:
                # Ramp from phase_start_value to 1.0
                if phase_elapsed >= self._attack:
                    # Attack complete - enter decay
                    self._value = 1.0
                    self._state = AdsrState.DECAY
                    self._phase_start_time = current_time
                    self._phase_start_value = 1.0
                    logger.debug(f"sample={current_time}: ATTACK -> DECAY (value={self._value:.3f})")
                else:
                    # Linear ramp: start + (end - start) * progress
                    progress = phase_elapsed / self._attack
                    self._value = self._phase_start_value + (1.0 - self._phase_start_value) * progress
            
            elif self._state == AdsrState.DECAY:
                # Ramp from 1.0 to sustain_level
                if phase_elapsed >= self._decay:
                    # Decay complete - enter sustain
                    self._value = self._sustain_level
                    self._state = AdsrState.SUSTAIN
                    logger.debug(f"sample={current_time}: DECAY -> SUSTAIN (value={self._value:.3f})")
                else:
                    # Linear ramp: 1.0 -> sustain_level
                    progress = phase_elapsed / self._decay
                    self._value = 1.0 + (self._sustain_level - 1.0) * progress
            
            elif self._state == AdsrState.SUSTAIN:
                # Hold at sustain_level (until gate falls)
                self._value = self._sustain_level
            
            elif self._state == AdsrState.RELEASE:
                # Ramp from phase_start_value to 0.0
                if phase_elapsed >= self._release:
                    # Release complete - enter idle
                    self._value = 0.0
                    self._state = AdsrState.IDLE
                    logger.debug(f"sample={current_time}: RELEASE -> IDLE (value={self._value:.3f})")
                else:
                    # Linear ramp: phase_start_value -> 0.0
                    progress = phase_elapsed / self._release
                    self._value = self._phase_start_value * (1.0 - progress)
            
            output[i, 0] = self._value
            self._prev_gate = gate_value
        
        return Snippet(start, output)
    
    def __repr__(self) -> str:
        return (
            f"AdsrPE(gate={self._gate.__class__.__name__}, "
            f"attack_samples={self._attack_samples}, attack_seconds={self._attack_seconds}, "
            f"decay_samples={self._decay_samples}, decay_seconds={self._decay_seconds}, "
            f"sustain={self._sustain_level}, "
            f"release_samples={self._release_samples}, release_seconds={self._release_seconds})"
        )
