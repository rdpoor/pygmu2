"""
AudioRenderer - plays audio through the system sound output.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
import sounddevice as sd
from typing import Optional

from pygmu2.renderer import Renderer
from pygmu2.snippet import Snippet
from pygmu2.logger import get_logger

logger = get_logger(__name__)


class AudioRenderer(Renderer):
    """
    A Renderer that plays audio through the system's audio output (DAC).
    
    Uses sounddevice (PortAudio) for cross-platform audio playback.
    
    Two modes of operation:
    1. **Blocking playback**: Call render() to play audio synchronously
    2. **Streaming playback**: Call play_extent() to stream a range
    
    Args:
        sample_rate: Audio sample rate in Hz (default: 44100)
        device: Audio device index or name (default: None = system default)
        blocksize: Samples per block for streaming (default: 1024)
        latency: Latency setting ('low', 'high', or seconds, default: 'low')
    
    Example:
        # Play a 1-second sine wave
        sine = SinePE(frequency=440.0)
        cropped = CropPE(sine, Extent(0, 44100))
        
        renderer = AudioRenderer(sample_rate=44100)
        renderer.set_source(cropped)
        renderer.start()
        renderer.play_extent()  # Play the entire extent
        renderer.stop()
        
        # Or use context manager
        with AudioRenderer(sample_rate=44100) as renderer:
            renderer.set_source(cropped)
            renderer.start()
            renderer.play_extent()
        
        # Play a specific range
        renderer.play_range(start=0, duration=44100)
        
        # Non-blocking streaming (advanced)
        renderer.stream_start()
        # ... audio plays in background ...
        renderer.stream_stop()
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        device: Optional[int | str] = None,
        blocksize: int = 1024,
        latency: str | float = 'low',
    ):
        super().__init__(sample_rate=sample_rate)
        self._device = device
        self._blocksize = blocksize
        self._latency = latency
        
        # Streaming state
        self._stream: Optional[sd.OutputStream] = None
        self._stream_position: int = 0
        self._stream_end: Optional[int] = None
    
    @property
    def device(self) -> Optional[int | str]:
        """The audio output device."""
        return self._device
    
    @property
    def blocksize(self) -> int:
        """Samples per block for streaming."""
        return self._blocksize
    
    def _output(self, snippet: Snippet) -> None:
        """
        Play a snippet through the audio output (blocking).
        
        This is called by render() and plays synchronously.
        Uses OutputStream.write() to avoid CFFI callback bugs in sd.play().
        
        Args:
            snippet: The audio data to play
        """
        channels = snippet.channels
        
        # Use OutputStream with blocking write to avoid callback issues
        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=channels,
            dtype='float32',
            device=self._device,
        ) as stream:
            stream.write(snippet.data)
    
    def play_range(self, start: int, duration: int) -> None:
        """
        Play a specific range of samples (blocking).
        
        Convenience method that renders and plays in one call.
        
        Args:
            start: Starting sample index
            duration: Number of samples to play
        """
        self.render(start, duration)
    
    def play_extent(self, chunk_size: Optional[int] = None) -> None:
        """
        Play the entire extent of the source (blocking).
        
        For finite extents, plays the whole thing using a single continuous
        stream to avoid gaps between chunks.
        For infinite extents, raises an error.
        
        Args:
            chunk_size: Samples per chunk for rendering (default: blocksize * 16)
        
        Raises:
            RuntimeError: If source has infinite extent
        """
        if self._source is None:
            raise RuntimeError("No source set. Call set_source() first.")
        
        extent = self._source.extent()
        
        if extent.start is None or extent.end is None:
            raise RuntimeError(
                "Cannot play_extent() on infinite source. "
                "Use CropPE to limit the extent, or use play_range()."
            )
        
        if chunk_size is None:
            chunk_size = self._blocksize * 16
        
        channels = self._channel_count or 1
        
        # Use a single continuous stream for gapless playback
        with sd.OutputStream(
            samplerate=self._sample_rate,
            channels=channels,
            dtype='float32',
            device=self._device,
        ) as stream:
            position = extent.start
            while position < extent.end:
                remaining = extent.end - position
                this_chunk = min(chunk_size, remaining)
                snippet = self._source.render(position, this_chunk)
                stream.write(snippet.data)
                position += this_chunk
        
        logger.info(f"Played {extent.end - extent.start} samples")
    
    def stream_start(self, start: int = 0, end: Optional[int] = None) -> None:
        """
        Start non-blocking audio streaming.
        
        Audio plays in the background via a callback. Use stream_stop()
        to stop playback.
        
        Args:
            start: Starting sample index
            end: Ending sample index (None = play until stopped)
        
        Raises:
            RuntimeError: If already streaming or not started
        """
        if not self._started:
            raise RuntimeError("Not started. Call start() first.")
        
        if self._stream is not None:
            raise RuntimeError("Already streaming. Call stream_stop() first.")
        
        if self._source is None:
            raise RuntimeError("No source set.")
        
        channels = self._channel_count or 1
        
        self._stream_position = start
        self._stream_end = end
        
        def callback(outdata, frames, time_info, status):
            if status:
                logger.warning(f"Stream status: {status}")
            
            # Check if we've reached the end
            if self._stream_end is not None:
                remaining = self._stream_end - self._stream_position
                if remaining <= 0:
                    outdata.fill(0)
                    raise sd.CallbackStop()
                frames = min(frames, remaining)
            
            # Render audio
            snippet = self._source.render(self._stream_position, frames)
            
            # Copy to output buffer
            if snippet.duration < len(outdata):
                outdata[:snippet.duration] = snippet.data
                outdata[snippet.duration:] = 0
            else:
                outdata[:] = snippet.data[:len(outdata)]
            
            self._stream_position += frames
        
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=channels,
            dtype='float32',
            device=self._device,
            blocksize=self._blocksize,
            latency=self._latency,
            callback=callback,
        )
        self._stream.start()
        logger.info(f"Streaming started at position {start}")
    
    def stream_stop(self) -> None:
        """
        Stop non-blocking audio streaming.
        
        Safe to call even if not streaming.
        """
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info(f"Streaming stopped at position {self._stream_position}")
    
    def stream_wait(self) -> None:
        """
        Wait for streaming to complete.
        
        Only useful when streaming with a finite end position.
        """
        if self._stream is not None:
            while self._stream.active:
                sd.sleep(10)
    
    @property
    def stream_position(self) -> int:
        """Current position in the stream (samples)."""
        return self._stream_position
    
    @property
    def is_streaming(self) -> bool:
        """True if currently streaming audio."""
        return self._stream is not None and self._stream.active
    
    def stop(self) -> None:
        """
        Stop the renderer and any active streams.
        """
        self.stream_stop()
        super().stop()
    
    @staticmethod
    def list_devices() -> None:
        """Print available audio devices."""
        print(sd.query_devices())
    
    @staticmethod
    def get_default_device() -> dict:
        """Get information about the default output device."""
        return sd.query_devices(kind='output')
    
    def __repr__(self) -> str:
        return (
            f"AudioRenderer(sample_rate={self._sample_rate}, "
            f"device={self._device}, blocksize={self._blocksize})"
        )
