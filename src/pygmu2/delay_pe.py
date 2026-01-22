"""
DelayPE - delays audio by a specified number of samples.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from typing import Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class DelayPE(ProcessingElement):
    """
    A ProcessingElement that delays its input by a specified number of samples.
    
    This shifts the audio forward in time. Positive delay values push the
    audio later; negative values would shift it earlier (though typically
    delay is non-negative).
    
    The extent is shifted by the delay amount.
    
    Args:
        source: Input ProcessingElement
        delay: Number of samples to delay (positive = later in time)
    
    Example:
        # Delay a WAV file by 1 second (at 44.1kHz)
        reader = WavReaderPE("vocals.wav")
        delayed = DelayPE(reader, delay=44100)
        
        # Create a simple echo effect by mixing original with delayed
        reader = WavReaderPE("sound.wav")
        delayed = DelayPE(reader, delay=22050)  # 0.5 second delay
        echo = MixPE(reader, GainPE(delayed, 0.5))  # Mix at 50% level
        
        # Negative delay (shift earlier - use with caution)
        early = DelayPE(source, delay=-1000)
    """
    
    def __init__(self, source: ProcessingElement, delay: int):
        self._source = source
        self._delay = delay
    
    @property
    def source(self) -> ProcessingElement:
        """The input ProcessingElement."""
        return self._source
    
    @property
    def delay(self) -> int:
        """The delay in samples."""
        return self._delay
    
    def inputs(self) -> list[ProcessingElement]:
        """Return the input PE."""
        return [self._source]
    
    def is_pure(self) -> bool:
        """
        DelayPE is pure - it's a stateless time shift.
        """
        return True
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Render delayed audio.
        
        Requests samples from the source at (start - delay), effectively
        shifting the output forward in time.
        
        Args:
            start: Starting sample index (in output time)
            duration: Number of samples to render
        
        Returns:
            Snippet containing the delayed audio
        """
        # Get samples from source at shifted position
        source_snippet = self._source.render(start - self._delay, duration)
        
        # Return with the requested start position
        return Snippet(start, source_snippet.data)
    
    def _compute_extent(self) -> Extent:
        """
        Return the delayed extent.
        
        The extent is shifted forward by the delay amount.
        """
        source_extent = self._source.extent()
        
        # Shift the extent by delay
        new_start = None if source_extent.start is None else source_extent.start + self._delay
        new_end = None if source_extent.end is None else source_extent.end + self._delay
        
        return Extent(new_start, new_end)
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def __repr__(self) -> str:
        return f"DelayPE(source={self._source.__class__.__name__}, delay={self._delay})"
