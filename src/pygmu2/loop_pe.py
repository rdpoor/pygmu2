"""
LoopPE - repeat a segment of audio.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

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
        crossfade_seconds: Duration in seconds for crossfade at loop points (optional)
        crossfade_samples: Duration in samples for crossfade at loop points (optional)
    
    Example:
        # Loop entire file forever
        looped_stream = LoopPE(WavReaderPE("drum.wav"))
        
        # Loop a specific region 4 times
        looped_stream = LoopPE(source_stream, loop_start=1000, loop_end=5000, count=4)
        
        # Seamless looping with crossfade
        looped_stream = LoopPE(source_stream, crossfade_seconds=0.01)  # 10ms crossfade
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        loop_start: Optional[int] = None,
        loop_end: Optional[int] = None,
        count: Optional[int] = None,
        crossfade_seconds: Optional[float] = None,
        crossfade_samples: Optional[int] = None,
    ):
        self._source = source
        self._loop_start = loop_start
        self._loop_end = loop_end
        self._count = count
        self._crossfade_seconds = crossfade_seconds
        self._crossfade_samples = crossfade_samples
        
        # These will be resolved when configured
        self._resolved_start: Optional[int] = None
        self._resolved_end: Optional[int] = None
        self._loop_length: Optional[int] = None
        self._crossfade: int = 0
    
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
    def crossfade_seconds(self) -> float:
        """Crossfade duration in seconds (requested)."""
        return float(self._crossfade_seconds or 0.0)

    @property
    def crossfade_samples(self) -> int:
        """Crossfade duration in samples (resolved after configure)."""
        return int(self._crossfade)
    
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
        
        # Resolve crossfade samples (seconds -> samples if provided)
        self._crossfade = self._time_to_samples(
            samples=self._crossfade_samples,
            seconds=self._crossfade_seconds,
            name="crossfade",
        )
        # Crossfade can't be more than half the loop length
        self._crossfade = min(self._crossfade, self._loop_length // 2)
    
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
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render looped audio.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
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
            # Clamp duration to not exceed total length
            valid_duration = min(duration, total_length - start)
        else:
            valid_duration = duration
        
        if valid_duration <= 0:
            return Snippet(start, output)
        
        # Fetch the entire loop region from source once
        loop_data = self._source.render(self._resolved_start, self._loop_length).data
        
        # Create array of output sample indices
        out_indices = np.arange(start, start + valid_duration)
        
        # Map to positions within the loop (vectorized modulo)
        loop_positions = out_indices % self._loop_length
        
        # Use fancy indexing to get all samples at once
        output[:valid_duration, :] = loop_data[loop_positions, :]
        
        # Apply crossfade if enabled
        if self._crossfade > 0:
            # Find samples in the crossfade region (near end of loop)
            xfade_threshold = self._loop_length - self._crossfade
            in_xfade = loop_positions >= xfade_threshold
            
            if np.any(in_xfade):
                # Get indices of samples needing crossfade
                xfade_indices = np.where(in_xfade)[0]
                xfade_loop_pos = loop_positions[in_xfade]
                
                # Calculate fade position (0 to crossfade_samples)
                fade_pos = xfade_loop_pos - xfade_threshold
                
                # Calculate fade coefficients (vectorized)
                fade_out = 1.0 - (fade_pos / self._crossfade)
                fade_in = fade_pos / self._crossfade
                
                # Reshape for broadcasting with channels
                fade_out = fade_out[:, np.newaxis]
                fade_in = fade_in[:, np.newaxis]
                
                # Get blend samples from start of loop (vectorized lookup)
                blend_positions = fade_pos.astype(int)
                blend_samples = loop_data[blend_positions, :]
                
                # Apply crossfade: out = current * fade_out + blend * fade_in
                output[xfade_indices, :] = (
                    output[xfade_indices, :] * fade_out +
                    blend_samples * fade_in
                )
        
        return Snippet(start, output.astype(np.float32))
    
    def __repr__(self) -> str:
        count_str = f", count={self._count}" if self._count is not None else ""
        xfade_str = f", crossfade_samples={self.crossfade_samples}" if self.crossfade_samples > 0 else ""
        return (
            f"LoopPE(source={self._source.__class__.__name__}, "
            f"loop_start={self._loop_start}, loop_end={self._loop_end}"
            f"{count_str}{xfade_str})"
        )
