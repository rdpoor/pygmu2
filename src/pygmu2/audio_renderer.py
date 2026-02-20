"""
AudioRenderer - plays audio through the system sound output.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np
import sounddevice as sd

from pygmu2.renderer import Renderer
from pygmu2.snippet import Snippet
from pygmu2.config import handle_error
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
        cropped = CropPE(sine, 0, (44100) - (0))
        
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
        device: int | str | None = None,
        blocksize: int = 1024,
        latency: str | float = 'low',
    ):
        super().__init__(sample_rate=sample_rate)
        self._device = device
        self._blocksize = blocksize
        self._latency = latency
        
        # Streaming state (for stream_start/stream_stop callback mode)
        self._stream: sd.OutputStream | None = None
        self._stream_position: int = 0
        self._stream_end: int | None = None
        # Single long-lived stream for blocking render() loop (avoids open/close per block)
        self._blocking_stream: sd.OutputStream | None = None
    
    @property
    def device(self) -> int | str | None:
        """The audio output device."""
        return self._device
    
    @property
    def blocksize(self) -> int:
        """Samples per block for streaming."""
        return self._blocksize
    
    def _output(self, snippet: Snippet) -> None:
        """
        Play a snippet through the audio output (blocking).

        This is called by render() and plays synchronously. Uses a single
        long-lived OutputStream (opened on first write, closed in stop())
        to avoid the cost of opening/closing the stream every block.

        Args:
            snippet: The audio data to play
        """
        channels = snippet.channels
        if self._blocking_stream is None:
            self._blocking_stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=channels,
                dtype='float32',
                device=self._device,
                blocksize=self._blocksize,
                latency=self._latency,
            )
            self._blocking_stream.start()
        data = snippet.data
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        self._blocking_stream.write(data)
    
    def play_range(self, start: int, duration: int) -> None:
        """
        Play a specific range of samples (blocking).
        
        Convenience method that renders and plays in one call.
        
        Args:
            start: Starting sample index
            duration: Number of samples to play
        """
        self.render(start, duration)
    
    def play_extent(self, chunk_size: int | None = None) -> None:
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
            handle_error("No source set. Call set_source() first.", fatal=True)
            return
        
        extent = self._source.extent()
        
        if extent.start is None or extent.end is None:
            handle_error(
                "Cannot play_extent() on infinite source. "
                "Use CropPE to limit the extent, or use play_range().",
                fatal=True,
            )
            return
        
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
                data = snippet.data
                if data.dtype != np.float32:
                    data = data.astype(np.float32, copy=False)
                stream.write(data)
                position += this_chunk
        
        logger.info(f"Played {extent.end - extent.start} samples")
    
    def stream_start(self, start: int = 0, end: int | None = None) -> None:
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
            handle_error("Not started. Call start() first.", fatal=True)
            return
        
        if self._stream is not None:
            handle_error("Already streaming. Call stream_stop() first.", fatal=True)
            return
        
        if self._source is None:
            handle_error("No source set.", fatal=True)
            return
        
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
        if self._blocking_stream is not None:
            try:
                self._blocking_stream.stop()
                self._blocking_stream.close()
            except Exception:
                pass
            self._blocking_stream = None
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
