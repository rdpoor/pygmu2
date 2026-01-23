"""
LoopPE - repeat a segment of audio.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from typing import Optional

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class LoopPE(ProcessingElement):
    """
    Repeat a segment of audio from the source.
    
    Loops the source audio between loop_start and loop_end, either
    indefinitely or for a specified number of repetitions.
    
    Args:
        source: Input audio PE
        loop_start: Start frame of loop region (default: source extent start)
        loop_end: End frame of loop region (default: source extent end)
        count: Number of repetitions, or None for infinite looping
        crossfade: Duration in seconds for crossfade at loop points (0 = no crossfade)
    
    Example:
        # Loop entire file forever
        looped = LoopPE(WavReaderPE("drum.wav"))
        
        # Loop a specific region 4 times
        looped = LoopPE(source, loop_start=1000, loop_end=5000, count=4)
        
        # Seamless looping with crossfade
        looped = LoopPE(source, crossfade=0.01)  # 10ms crossfade
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        loop_start: Optional[int] = None,
        loop_end: Optional[int] = None,
        count: Optional[int] = None,
        crossfade: float = 0.0,
    ):
        self._source = source
        self._loop_start = loop_start
        self._loop_end = loop_end
        self._count = count
        self._crossfade = max(0.0, crossfade)
        
        # These will be resolved when configured
        self._resolved_start: Optional[int] = None
        self._resolved_end: Optional[int] = None
        self._loop_length: Optional[int] = None
        self._crossfade_samples: int = 0
    
    @property
    def source(self) -> ProcessingElement:
        """The input audio PE."""
        return self._source
    
    @property
    def loop_start(self) -> Optional[int]:
        """Start frame of the loop region."""
        return self._loop_start
    
    @property
    def loop_end(self) -> Optional[int]:
        """End frame of the loop region."""
        return self._loop_end
    
    @property
    def count(self) -> Optional[int]:
        """Number of loop repetitions (None = infinite)."""
        return self._count
    
    @property
    def crossfade(self) -> float:
        """Crossfade duration in seconds."""
        return self._crossfade
    
    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs."""
        return [self._source]
    
    def is_pure(self) -> bool:
        """LoopPE is pure - just index remapping."""
        return True
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def configure(self, sample_rate: float) -> None:
        """Configure with sample rate and resolve loop boundaries."""
        super().configure(sample_rate)
        
        # Resolve loop boundaries from source extent
        source_extent = self._source.extent()
        
        if self._loop_start is not None:
            self._resolved_start = self._loop_start
        elif source_extent.start is not None:
            self._resolved_start = source_extent.start
        else:
            self._resolved_start = 0
        
        if self._loop_end is not None:
            self._resolved_end = self._loop_end
        elif source_extent.end is not None:
            self._resolved_end = source_extent.end
        else:
            # Can't loop infinite source without explicit end
            raise ValueError("Cannot loop source with infinite extent without explicit loop_end")
        
        self._loop_length = self._resolved_end - self._resolved_start
        if self._loop_length <= 0:
            raise ValueError(f"Loop length must be positive, got {self._loop_length}")
        
        # Calculate crossfade samples
        self._crossfade_samples = int(self._crossfade * sample_rate)
        # Crossfade can't be more than half the loop length
        self._crossfade_samples = min(self._crossfade_samples, self._loop_length // 2)
    
    def _compute_extent(self) -> Extent:
        """Return the extent of this PE."""
        if self._loop_length is None:
            # Not configured yet, return infinite
            return Extent(0, None)
        
        if self._count is None:
            # Infinite looping
            return Extent(0, None)
        else:
            # Finite number of loops
            total_length = self._count * self._loop_length
            return Extent(0, total_length)
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Render looped audio.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render
        
        Returns:
            Snippet containing looped audio
        """
        if self._loop_length is None or self._resolved_start is None:
            # Not configured, return silence
            channels = self._source.channel_count() or 1
            return Snippet(start, np.zeros((duration, channels), dtype=np.float32))
        
        channels = self._source.channel_count() or 1
        output = np.zeros((duration, channels), dtype=np.float32)
        
        # Check if we're past the end (for finite loops)
        if self._count is not None:
            total_length = self._count * self._loop_length
            if start >= total_length:
                return Snippet(start, output)
        
        # For each output sample, calculate the source index
        for i in range(duration):
            out_idx = start + i
            
            # Check bounds for finite loops
            if self._count is not None:
                total_length = self._count * self._loop_length
                if out_idx >= total_length:
                    break
            
            # Map to position within loop
            loop_pos = out_idx % self._loop_length
            source_idx = self._resolved_start + loop_pos
            
            # Get sample from source
            snippet = self._source.render(source_idx, 1)
            sample = snippet.data[0, :]
            
            # Apply crossfade if enabled
            if self._crossfade_samples > 0:
                # Near end of loop - fade out and blend with start
                if loop_pos >= self._loop_length - self._crossfade_samples:
                    fade_pos = loop_pos - (self._loop_length - self._crossfade_samples)
                    fade_out = 1.0 - (fade_pos / self._crossfade_samples)
                    fade_in = fade_pos / self._crossfade_samples
                    
                    # Get corresponding sample from start of loop
                    blend_idx = self._resolved_start + fade_pos
                    blend_snippet = self._source.render(blend_idx, 1)
                    blend_sample = blend_snippet.data[0, :]
                    
                    sample = sample * fade_out + blend_sample * fade_in
            
            output[i, :] = sample
        
        return Snippet(start, output.astype(np.float32))
    
    def __repr__(self) -> str:
        count_str = f", count={self._count}" if self._count is not None else ""
        xfade_str = f", crossfade={self._crossfade}" if self._crossfade > 0 else ""
        return (
            f"LoopPE(source={self._source.__class__.__name__}, "
            f"loop_start={self._loop_start}, loop_end={self._loop_end}"
            f"{count_str}{xfade_str})"
        )
