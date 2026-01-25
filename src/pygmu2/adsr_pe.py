"""
AdsrPE - Attack-Decay-Sustain-Release envelope generator.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from enum import Enum
from typing import Optional

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


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
        attack: number of samples to ramp from 0.0 to 1.0
        decay: number of samples to ramp from 1.0 to sustain_level.
        sustain_level: output value to hold until gate signal goes non-positive.
        release: number of samples to ramp from current value to 0.0
    """
    
    def __init__(
        self,
        gate: ProcessingElement,
        attack: int = 441,
        decay: int = 4410,
        sustain_level: float = 0.7,
        release: int = 8820,
    ):
        self._gate = gate
        self._attack = max(1, int(attack))
        self._decay = max(1, int(decay))
        self._sustain_level = max(0.0, min(1.0, sustain_level))
        self._release = max(1, int(release))
        
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
            gate_rising = gate_value > 0.0 and self._prev_gate <= 0.0
            gate_falling = gate_value <= 0.0 and self._prev_gate > 0.0
            
            old_state = self._state
            old_value = self._value
            
            # State transitions
            if gate_rising:
                # Gate just went high - start attack (or restart if already active)
                if self._state == AdsrState.RELEASE:
                    # Re-triggered during release: start attack from current value
                    self._state = AdsrState.ATTACK
                    self._phase_start_time = current_time
                    self._phase_start_value = self._value
                elif self._state == AdsrState.IDLE:
                    # Starting from idle - ensure we start from 0.0
                    self._state = AdsrState.ATTACK
                    self._phase_start_time = current_time
                    self._phase_start_value = 0.0
                    self._value = 0.0  # Explicitly reset value when starting from IDLE
                # If already in ATTACK, DECAY, or SUSTAIN, continue (ignore re-trigger)
            
            elif gate_falling:
                # Gate just went low - enter release
                if self._state in (AdsrState.ATTACK, AdsrState.DECAY, AdsrState.SUSTAIN):
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
            f"attack={self._attack}, decay={self._decay}, "
            f"sustain={self._sustain_level}, release={self._release})"
        )
