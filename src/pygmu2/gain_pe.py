"""
GainPE - applies gain (amplitude scaling) to audio.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
from typing import Union, Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class GainPE(ProcessingElement):
    """
    A ProcessingElement that applies gain (amplitude scaling) to its input.
    
    The gain can be either:
    - A constant float value (fixed gain)
    - A ProcessingElement (time-varying gain, e.g., for tremolo, ducking, envelopes)
    
    When gain is a PE, the source samples are multiplied element-wise by the
    gain PE's output. The gain PE should output mono (single channel) values
    that are broadcast across all channels of the source.
    
    Args:
        source: Input ProcessingElement to apply gain to
        gain: Gain multiplier - either a float or a PE (default: 1.0)
    
    Example:
        # Fixed gain (attenuate by half)
        quiet = GainPE(source, gain=0.5)
        
        # Fixed gain (amplify by 2x)
        loud = GainPE(source, gain=2.0)
        
        # Invert phase
        inverted = GainPE(source, gain=-1.0)
        
        # Tremolo effect (amplitude modulation)
        lfo = SinePE(frequency=5.0, amplitude=0.3)  # ±0.3 at 5Hz
        tremolo_gain = MixPE(ConstantPE(0.7), lfo)  # 0.7 ± 0.3
        tremolo = GainPE(source, gain=tremolo_gain)
        
        # Fade in envelope
        fade_in = RampPE(0.0, 1.0, duration=44100)  # 1 second fade
        faded = GainPE(source, gain=fade_in)
        
        # Ducking (reduce gain when another signal is present)
        sidechain = WavReaderPE("kick.wav")
        envelope = EnvelopeFollowerPE(sidechain)  # Hypothetical PE
        duck_amount = GainPE(envelope, gain=-0.5)  # Invert and scale
        duck_gain = MixPE(ConstantPE(1.0), duck_amount)
        ducked = GainPE(source, gain=duck_gain)
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        gain: Union[float, ProcessingElement] = 1.0,
    ):
        self._source = source
        self._gain = gain
        
        # Track whether gain is a PE for inputs() and extent calculation
        self._gain_is_pe = isinstance(gain, ProcessingElement)
    
    @property
    def source(self) -> ProcessingElement:
        """The input ProcessingElement."""
        return self._source
    
    @property
    def gain(self) -> Union[float, ProcessingElement]:
        """The gain value or PE."""
        return self._gain
    
    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs (source, and gain if it's a PE)."""
        if self._gain_is_pe:
            return [self._source, self._gain]
        return [self._source]
    
    def is_pure(self) -> bool:
        """
        GainPE is pure - multiplication is stateless.
        """
        return True
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render gain-adjusted audio.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
        Returns:
            Snippet with gain applied
        """
        # Get source audio
        source_snippet = self._source.render(start, duration)
        
        if self._gain_is_pe:
            # Get gain values from PE (allow multi-channel for per-channel gain)
            gain_data = self._scalar_or_pe_values(
                self._gain,
                start,
                duration,
                dtype=np.float32,
                allow_multichannel=True,
            )
            
            # Broadcast gain across channels if needed
            # gain_snippet is typically mono, source may be stereo
            if gain_data.shape[1] == 1 and source_snippet.channels > 1:
                gain_data = np.tile(gain_data, (1, source_snippet.channels))
            
            # Apply gain
            result = source_snippet.data * gain_data
        else:
            # Constant gain
            result = source_snippet.data * self._gain
        
        return Snippet(start, result.astype(np.float32))
    
    def _compute_extent(self) -> Extent:
        """
        Return the extent of this PE.
        
        If gain is constant: same as source extent.
        If gain is a PE: intersection of source and gain extents.
        """
        source_extent = self._source.extent()
        
        if not self._gain_is_pe:
            return source_extent
        
        # Intersection of source and gain extents
        gain_extent = self._gain.extent()
        result = source_extent.intersection(gain_extent)
        return result
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def __repr__(self) -> str:
        if self._gain_is_pe:
            gain_str = f"{self._gain.__class__.__name__}(...)"
        else:
            gain_str = str(self._gain)
        return f"GainPE(source={self._source.__class__.__name__}, gain={gain_str})"
