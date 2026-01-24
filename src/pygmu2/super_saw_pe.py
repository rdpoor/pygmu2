"""
SuperSawPE - detuned unison sawtooth oscillator for warm, rich sounds.

Inspired by the Roland JP-8000's "Supersaw" waveform, this PE creates
multiple slightly-detuned sawtooth oscillators mixed together.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
from typing import Union, Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.blit_saw_pe import BlitSawPE
from pygmu2.gain_pe import GainPE


class SuperSawPE(ProcessingElement):
    """
    Detuned unison sawtooth oscillator for warm, analog-like sounds.
    
    Creates multiple BlitSawPE oscillators spread symmetrically around
    the center frequency. The result is a thick, chorused sound that's
    a staple of trance, EDM, and modern synth pads.
    
    The detune spread follows a symmetric pattern: for N voices, one is
    at the center frequency, and the rest are spread evenly above and
    below by up to ±detune_cents.
    
    Args:
        frequency: Center frequency in Hz, or PE for modulation
        amplitude: Overall amplitude, or PE for modulation (default: 1.0)
        voices: Number of oscillators (default: 7, must be odd for symmetry)
        detune_cents: Maximum detune in cents (default: 20.0, 100 cents = 1 semitone)
        mix_mode: Voice mixing mode (default: 'center_heavy')
            - 'equal': All voices at equal amplitude
            - 'center_heavy': Center voice louder, outer voices quieter
            - 'linear': Linear falloff from center to edges
        channels: Number of output channels (default: 1)
    
    Example:
        # Classic supersaw lead
        lead = SuperSawPE(frequency=440.0, detune_cents=15.0)
        
        # Thick pad with more voices and detune
        pad = SuperSawPE(frequency=220.0, voices=9, detune_cents=30.0)
        
        # With pitch modulation (vibrato)
        vibrato = SinePE(frequency=5.0, amplitude=10.0)  # ±10 Hz
        lead = SuperSawPE(frequency=vibrato + ConstantPE(440.0))
        
        # With filter for classic trance sound
        saw = SuperSawPE(frequency=440.0, detune_cents=20.0)
        filtered = BiquadPE(saw, mode=LOWPASS, frequency=2000.0, q=0.7)
    
    Notes:
        - Odd voice counts work best for symmetry (center + equal spread)
        - Even voice counts will have no center voice (all detuned)
        - detune_cents=0 collapses to a single BlitSawPE
        - Higher voice counts increase CPU load linearly
        - The mix is normalized to prevent clipping
    """
    
    # Mix mode constants
    MIX_EQUAL = 'equal'
    MIX_CENTER_HEAVY = 'center_heavy'
    MIX_LINEAR = 'linear'
    
    def __init__(
        self,
        frequency: Union[float, ProcessingElement],
        amplitude: Union[float, ProcessingElement] = 1.0,
        voices: int = 7,
        detune_cents: float = 20.0,
        mix_mode: str = 'center_heavy',
        channels: int = 1,
    ):
        if voices < 1:
            voices = 1
        
        self._frequency = frequency
        self._amplitude = amplitude
        self._voices = voices
        self._detune_cents = detune_cents
        self._mix_mode = mix_mode
        self._channels = channels
        
        # Compute detune ratios and mix gains for each voice
        self._detune_ratios = self._compute_detune_ratios()
        self._mix_gains = self._compute_mix_gains()
        
        # Create internal oscillators (created fresh on each configure)
        self._oscillators: list[BlitSawPE] = []
    
    @property
    def frequency(self) -> Union[float, ProcessingElement]:
        """Center frequency in Hz (constant or PE)."""
        return self._frequency
    
    @property
    def amplitude(self) -> Union[float, ProcessingElement]:
        """Overall amplitude (constant or PE)."""
        return self._amplitude
    
    @property
    def voices(self) -> int:
        """Number of oscillators."""
        return self._voices
    
    @property
    def detune_cents(self) -> float:
        """Maximum detune in cents."""
        return self._detune_cents
    
    @property
    def mix_mode(self) -> str:
        """Voice mixing mode."""
        return self._mix_mode
    
    def _compute_detune_ratios(self) -> np.ndarray:
        """
        Compute frequency ratios for each voice.
        
        Returns array of ratios to multiply with center frequency.
        Center voice = 1.0, others spread symmetrically.
        """
        if self._voices == 1 or self._detune_cents == 0:
            return np.array([1.0])
        
        # Convert cents to ratio: ratio = 2^(cents/1200)
        max_ratio = 2.0 ** (self._detune_cents / 1200.0)
        min_ratio = 2.0 ** (-self._detune_cents / 1200.0)
        
        # Spread voices evenly from min_ratio to max_ratio
        ratios = np.linspace(min_ratio, max_ratio, self._voices)
        
        return ratios
    
    def _compute_mix_gains(self) -> np.ndarray:
        """
        Compute amplitude gain for each voice based on mix mode.
        
        Returns array of gains, normalized so total power is reasonable.
        """
        n = self._voices
        
        if n == 1:
            return np.array([1.0])
        
        if self._mix_mode == self.MIX_EQUAL:
            # Equal amplitude, normalize by sqrt(n) for constant power
            gains = np.ones(n) / np.sqrt(n)
        
        elif self._mix_mode == self.MIX_CENTER_HEAVY:
            # Center voice at 1.0, outer voices at 0.5, normalize
            # This mimics the JP-8000 character
            gains = np.ones(n) * 0.5
            center_idx = n // 2
            gains[center_idx] = 1.0
            # Normalize
            gains = gains / np.sqrt(np.sum(gains ** 2))
        
        elif self._mix_mode == self.MIX_LINEAR:
            # Linear falloff from center
            center_idx = n // 2
            distances = np.abs(np.arange(n) - center_idx)
            max_dist = np.max(distances) if np.max(distances) > 0 else 1
            gains = 1.0 - 0.5 * (distances / max_dist)  # 1.0 at center, 0.5 at edges
            # Normalize
            gains = gains / np.sqrt(np.sum(gains ** 2))
        
        else:
            # Default to equal
            gains = np.ones(n) / np.sqrt(n)
        
        return gains
    
    def _create_oscillators(self) -> list[BlitSawPE]:
        """Create the internal BlitSawPE oscillators."""
        oscillators = []
        
        for i, ratio in enumerate(self._detune_ratios):
            gain = self._mix_gains[i]
            
            # Create frequency source for this voice
            if isinstance(self._frequency, ProcessingElement):
                # For PE frequency, apply detune ratio via GainPE
                # GainPE multiplies the input by the gain value
                freq = GainPE(self._frequency, gain=ratio)
            else:
                freq = self._frequency * ratio
            
            osc = BlitSawPE(
                frequency=freq,
                amplitude=gain,  # Individual voice gain
                channels=1,  # Mix to mono first, then expand
            )
            oscillators.append(osc)
        
        return oscillators
    
    def configure(self, sample_rate: int) -> None:
        """Configure this PE and create internal oscillators."""
        super().configure(sample_rate)
        
        # Create oscillators after we have sample rate
        self._oscillators = self._create_oscillators()
        
        # Configure all internal oscillators
        for osc in self._oscillators:
            osc.configure(sample_rate)
    
    def inputs(self) -> list[ProcessingElement]:
        """Return list of PE inputs (frequency and amplitude if PEs)."""
        result = []
        if isinstance(self._frequency, ProcessingElement):
            result.append(self._frequency)
        if isinstance(self._amplitude, ProcessingElement):
            result.append(self._amplitude)
        return result
    
    def is_pure(self) -> bool:
        """
        SuperSawPE is not pure due to internal oscillator state.
        """
        return False
    
    def channel_count(self) -> int:
        """Return the number of output channels."""
        return self._channels
    
    def on_start(self) -> None:
        """Start all internal oscillators."""
        for osc in self._oscillators:
            osc.on_start()
    
    def on_stop(self) -> None:
        """Stop all internal oscillators."""
        for osc in self._oscillators:
            osc.on_stop()
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render the supersaw by mixing all voices.
        
        Args:
            start: Starting sample index
            duration: Number of samples to generate (> 0)
        
        Returns:
            Snippet containing mixed sawtooth data
        """
        # Get amplitude values
        if isinstance(self._amplitude, ProcessingElement):
            amp_snippet = self._amplitude.render(start, duration)
            amp = amp_snippet.data[:, 0].flatten()
        else:
            amp = self._amplitude
        
        # Render all oscillators and mix
        # Each oscillator already has the correct detuned frequency
        # (either constant * ratio, or GainPE(freq_pe, ratio))
        result = np.zeros(duration, dtype=np.float64)
        
        for osc in self._oscillators:
            snippet = osc.render(start, duration)
            result += snippet.data[:, 0]
        
        # Apply overall amplitude
        if isinstance(amp, np.ndarray):
            result = result * amp
        else:
            result = result * amp
        
        # Reshape to (duration, channels)
        samples = result.reshape(-1, 1).astype(np.float32)
        if self._channels > 1:
            samples = np.tile(samples, (1, self._channels))
        
        return Snippet(start, samples)
    
    def _compute_extent(self) -> Extent:
        """
        Compute extent from PE inputs.
        
        If all inputs are constants: infinite extent.
        If any input is a PE: intersection of input extents.
        """
        if not self.inputs():
            return Extent(None, None)
        
        result = Extent(None, None)
        for pe_input in self.inputs():
            input_extent = pe_input.extent()
            intersection = result.intersection(input_extent)
            if intersection is None:
                return Extent(0, 0)
            result = intersection
        return result
    
    def __repr__(self) -> str:
        freq_str = (
            f"{self._frequency.__class__.__name__}"
            if isinstance(self._frequency, ProcessingElement)
            else str(self._frequency)
        )
        return (
            f"SuperSawPE(frequency={freq_str}, voices={self._voices}, "
            f"detune_cents={self._detune_cents}, mix_mode={self._mix_mode!r})"
        )
