"""
SinePE - a sine wave generator with optional modulation inputs.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.source_pe import SourcePE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class SinePE(ProcessingElement):
    """
    A ProcessingElement that generates a sine wave.
    
    Frequency, amplitude, and phase can be either constant floats or
    ProcessingElement inputs for modulation (FM/AM/PM synthesis).
    
    When all parameters are constants, this PE is pure (stateless).
    When any parameter is a PE, this PE becomes non-pure and maintains
    internal phase state for accurate frequency integration.
    
    Args:
        frequency: Frequency in Hz, or PE providing frequency values (default: 440.0)
        amplitude: Peak amplitude, or PE providing amplitude values (default: 1.0)
        phase: Phase offset in radians, or PE providing phase modulation (default: 0.0)
        channels: Number of output channels (default: 1)
    
    Example:
        # Simple 440 Hz sine wave (pure) - uses default frequency
        sine_stream = SinePE()
        
        # FM synthesis: carrier modulated by another sine (non-pure)
        modulator_stream = SinePE(frequency=5.0, amplitude=100.0)  # 5 Hz LFO, ±100 Hz
        carrier_stream = SinePE(frequency=440.0 + modulator_stream)  # Oops, need different approach
        
        # Better: use a dedicated modulation PE or constant offset
        lfo_stream = SinePE(frequency=5.0, amplitude=50.0)  # ±50 Hz deviation
        # Then combine in a wrapper or use phase modulation
        
        # AM synthesis: amplitude modulated
        envelope_stream = SinePE(frequency=2.0, amplitude=0.5)  # Tremolo
        tone_stream = SinePE(frequency=440.0, amplitude=envelope_stream)
    """
    
    def __init__(
        self,
        frequency: float | ProcessingElement = 440.0,
        amplitude: float | ProcessingElement = 1.0,
        phase: float | ProcessingElement = 0.0,
        channels: int = 1,
    ):
        self._frequency = frequency
        self._amplitude = amplitude
        self._phase = phase
        self._channels = channels
        
        # Phase accumulator for non-pure operation (FM synthesis)
        # Stores the accumulated phase at the END of the last rendered chunk
        self._accumulated_phase: float = 0.0
        self._phase_initialized: bool = False
    
    @property
    def frequency(self) -> float | ProcessingElement:
        """Frequency in Hz (constant or PE)."""
        return self._frequency
    
    @property
    def amplitude(self) -> float | ProcessingElement:
        """Peak amplitude (constant or PE)."""
        return self._amplitude
    
    @property
    def initial_phase(self) -> float | ProcessingElement:
        """Phase offset/modulation in radians (constant or PE)."""
        return self._phase
    
    def _has_pe_inputs(self) -> bool:
        """Check if any input is a ProcessingElement."""
        return (
            isinstance(self._frequency, ProcessingElement) or
            isinstance(self._amplitude, ProcessingElement) or
            isinstance(self._phase, ProcessingElement)
        )
    
    def inputs(self) -> list[ProcessingElement]:
        """Return list of PE inputs."""
        result = []
        if isinstance(self._frequency, ProcessingElement):
            result.append(self._frequency)
        if isinstance(self._amplitude, ProcessingElement):
            result.append(self._amplitude)
        if isinstance(self._phase, ProcessingElement):
            result.append(self._phase)
        return result
    
    def is_pure(self) -> bool:
        """
        Pure if all parameters are constants.
        Non-pure if any parameter is a PE (requires state for FM).
        """
        return not self._has_pe_inputs()
    
    def _on_start(self) -> None:
        """Reset phase accumulator on start."""
        self._accumulated_phase = 0.0
        self._phase_initialized = False

    def _on_stop(self) -> None:
        """Reset phase accumulator on stop."""
        self._accumulated_phase = 0.0
        self._phase_initialized = False
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Generate sine wave samples for the given range.
        
        For constant parameters: computes phase directly (pure).
        For PE parameters: integrates frequency and uses phase state (non-pure).
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet containing sine wave data
        """
        # Get parameter values (either constant or from PEs)
        freq_values = self._scalar_or_pe_values(self._frequency, start, duration, dtype=np.float64)
        amp_values = self._scalar_or_pe_values(self._amplitude, start, duration, dtype=np.float64).reshape(-1, 1)
        phase_mod = self._scalar_or_pe_values(self._phase, start, duration, dtype=np.float64)
        
        # Calculate phase
        if self._has_pe_inputs():
            # Non-pure: integrate frequency, use state
            phase = self._compute_phase_stateful(freq_values, phase_mod, start, duration)
        else:
            # Pure: calculate phase directly
            phase = self._compute_phase_pure(freq_values, phase_mod, start, duration)
        
        # Generate sine wave
        samples = amp_values * np.sin(phase)
        
        # Ensure correct shape (duration, channels)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        
        # Replicate to channels if needed
        if samples.shape[1] == 1 and self._channels > 1:
            samples = np.tile(samples, (1, self._channels))
        
        return Snippet(start, samples.astype(np.float32))
    
    def _compute_phase_pure(
        self,
        frequency: float | np.ndarray,
        phase_offset: float | np.ndarray,
        start: int,
        duration: int,
    ) -> np.ndarray:
        """
        Compute phase for pure operation (all constants).
        
        Phase is calculated directly from sample index:
        phase = phase_offset + 2π * frequency * (sample_index / sample_rate)
        """
        sample_indices = np.arange(start, start + duration, dtype=np.float64)
        time = sample_indices / self.sample_rate
        phase = phase_offset + 2.0 * np.pi * frequency * time
        return phase.reshape(-1, 1)
    
    def _compute_phase_stateful(
        self,
        frequency: float | np.ndarray,
        phase_mod: float | np.ndarray,
        start: int,
        duration: int,
    ) -> np.ndarray:
        """
        Compute phase for non-pure operation (PE inputs).
        
        Integrates instantaneous frequency over time to accumulate phase.
        Maintains state across render() calls for continuity.
        
        For FM synthesis accuracy, phase is computed as:
        phase[i] = phase[i-1] + 2π * freq[i] / sample_rate + phase_mod[i]
        """
        # Ensure frequency is an array
        if not isinstance(frequency, np.ndarray):
            frequency = np.full((duration, 1), frequency, dtype=np.float64)
        elif frequency.ndim == 1:
            frequency = frequency.reshape(-1, 1)
        
        # Calculate phase increments from frequency
        # phase_increment = 2π * f / sample_rate (radians per sample)
        phase_increment = 2.0 * np.pi * frequency / self.sample_rate
        
        # Determine starting phase
        # The base class enforces contiguous rendering for non-pure PEs,
        # so we only need to detect the first render to set initial phase.
        if not self._phase_initialized:
            if isinstance(self._phase, (int, float)):
                initial_phase = float(self._phase)
            else:
                initial_phase = 0.0
            self._phase_initialized = True
        else:
            # Contiguous render: continue from accumulated phase
            initial_phase = self._accumulated_phase
        
        # Cumulative sum of phase increments
        cumulative_phase = np.cumsum(phase_increment, axis=0) + initial_phase
        
        # Add phase modulation if it's from a PE
        if isinstance(phase_mod, np.ndarray):
            if phase_mod.ndim == 1:
                phase_mod = phase_mod.reshape(-1, 1)
            cumulative_phase = cumulative_phase + phase_mod
        elif phase_mod != 0.0 and not isinstance(self._phase, ProcessingElement):
            # Constant phase offset (already included in initial_phase for pure case,
            # but for stateful case with PE freq, we add it here if it's a constant)
            pass  # Already handled in initial_phase
        
        # Update accumulated phase for next render
        self._accumulated_phase = float(cumulative_phase[-1, 0])
        
        return cumulative_phase
    
    def _compute_extent(self) -> Extent:
        """
        Compute the extent of this sine wave.
        
        If all inputs are constants: infinite extent (default).
        If any input is a PE: intersection of input extents.
        """
        result = Extent(None, None)
        for pe_input in self.inputs():
            input_extent = pe_input.extent()
            result = result.intersection(input_extent)
        return result
    
    def channel_count(self) -> int:
        """Return the number of output channels."""
        return self._channels
    
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
        phase_str = (
            f"{self._phase.__class__.__name__}"
            if isinstance(self._phase, ProcessingElement)
            else str(self._phase)
        )
        return (
            f"SinePE(frequency={freq_str}, amplitude={amp_str}, "
            f"phase={phase_str}, channels={self._channels})"
        )
