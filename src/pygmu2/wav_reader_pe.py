"""
WavReaderPE - reads audio samples from a WAV file.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np
import soundfile as sf
from typing import Optional

from pygmu2.processing_element import SourcePE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.logger import get_logger

logger = get_logger(__name__)


class WavReaderPE(SourcePE):
    """
    A SourcePE that reads audio samples from a WAV file.
    
    The file is opened on on_start() and closed on on_stop().
    Samples are read on demand via render().
    
    The extent is finite (0 to frame_count), based on the file's length.
    Requests outside the file's extent return zeros.
    
    To shift the audio in time, use DelayPE.
    
    Args:
        path: Path to the WAV file
    
    Example:
        # Read a WAV file
        reader_stream = WavReaderPE("drums.wav")
        
        # Use in a graph
        reader_stream = WavReaderPE("input.wav")
        processed_stream = SomeEffectPE(reader_stream)
        renderer.set_source(processed_stream)
        
        # Delay audio by 1 second (use DelayPE)
        reader = WavReaderPE("vocals.wav")
        delayed = DelayPE(reader, delay=44100)
    """
    
    def __init__(self, path: str):
        self._path = path
        
        # File info (populated lazily on first access)
        self._frame_count: Optional[int] = None
        self._channels: Optional[int] = None
        self._file_sample_rate: Optional[int] = None
    
    @property
    def path(self) -> str:
        """Path to the WAV file."""
        return self._path
    
    @property
    def file_sample_rate(self) -> Optional[int]:
        """Sample rate of the WAV file (reads file metadata if needed)."""
        self._ensure_file_info()
        return self._file_sample_rate

    @property
    def sample_rate(self) -> Optional[int]:
        """
        The sample rate in Hz, if known.

        For WavReaderPE, this is the file's sample rate, even before configuration.
        """
        if self._sample_rate is not None:
            return self._sample_rate
        return self.file_sample_rate
    
    def _ensure_file_info(self) -> None:
        """Read file metadata if not already loaded."""
        if self._frame_count is None:
            with sf.SoundFile(self._path) as f:
                self._frame_count = f.frames
                self._channels = f.channels
                self._file_sample_rate = f.samplerate
    
    def _on_start(self) -> None:
        """Ensure file metadata is loaded at start."""
        self._ensure_file_info()
        logger.info(
            f"Opened {self._path}: {self._frame_count} frames, "
            f"{self._channels} channels, {self._file_sample_rate} Hz"
        )
    
    def _on_stop(self) -> None:
        """Log file close (no file handle to release)."""
        logger.info(f"Stopped reading {self._path}")
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Read audio samples from the WAV file.
        
        Samples outside the file's extent (0 to frame_count) are zero-filled.
        
        Args:
            start: Starting sample index
            duration: Number of samples to read (> 0)
        
        Returns:
            Snippet containing the audio data
        """
        self._ensure_file_info()
        
        # Initialize output with zeros
        data = np.zeros((duration, self._channels), dtype=np.float32)
        
        # Calculate overlap with file extent (0 to frame_count)
        overlap_start = max(start, 0)
        overlap_end = min(start + duration, self._frame_count)
        
        if overlap_start < overlap_end:
            # Always use the stateless sf.read() with explicit start/stop.
            # WavReaderPE is pure (multiple sinks allowed), so interleaved
            # render() calls from different consumers must not interfere.
            # The seek+read pattern on a shared SoundFile handle is NOT safe
            # for this because seek() mutates the file position.
            file_data, _ = sf.read(
                self._path,
                start=overlap_start,
                stop=overlap_end,
                dtype='float32',
            )
            
            # Handle mono files (soundfile returns 1D for mono)
            if file_data.ndim == 1:
                file_data = file_data.reshape(-1, 1)
            
            # Copy into output buffer
            output_start = overlap_start - start
            output_end = output_start + read_count
            data[output_start:output_end, :] = file_data
        
        return Snippet(start, data)
    
    def _compute_extent(self) -> Extent:
        """Return the extent of the WAV file (0 to frame_count)."""
        self._ensure_file_info()
        return Extent(0, self._frame_count)
    
    def channel_count(self) -> int:
        """Return the number of channels in the WAV file."""
        self._ensure_file_info()
        return self._channels
    
    def __repr__(self) -> str:
        return f"WavReaderPE(path={self._path!r})"
