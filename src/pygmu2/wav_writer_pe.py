"""
WavWriterPE - writes audio samples to a WAV file as a side effect.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
import soundfile as sf
from typing import Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.config import handle_error
from pygmu2.logger import get_logger

logger = get_logger(__name__)


class WavWriterPE(ProcessingElement):
    """
    A ProcessingElement that writes audio to a WAV file as a side effect.
    
    This PE passes audio through unchanged while writing it to a file.
    It can be inserted anywhere in a processing chain.
    
    The file is opened on on_start() and closed on on_stop().
    Samples are written on each render() call.
    
    Note: This PE is NOT pure - it has side effects (file I/O).
    It should not be used with multiple sinks.
    
    Args:
        source: Input ProcessingElement to read from
        path: Path to the output WAV file
        sample_rate: Sample rate for the output file (default: uses Renderer's rate)
        subtype: Audio subtype (default: 'PCM_16' for 16-bit PCM)
    
    Example:
        # Write processing output to a file
        sine = SinePE(frequency=440.0)
        writer = WavWriterPE(sine, "output.wav")
        
        with NullRenderer(sample_rate=44100) as renderer:
            renderer.set_source(writer)
            renderer.start()
            renderer.render(0, 44100)  # Write 1 second
        # File is closed automatically
        
        # Specify format
        writer = WavWriterPE(source, "output.wav", subtype='FLOAT')
    """
    
    # Common subtypes:
    # 'PCM_16' - 16-bit signed integer (CD quality)
    # 'PCM_24' - 24-bit signed integer (professional)
    # 'PCM_32' - 32-bit signed integer
    # 'FLOAT'  - 32-bit float
    # 'DOUBLE' - 64-bit float
    
    def __init__(
        self,
        source: ProcessingElement,
        path: str,
        sample_rate: Optional[int] = None,
        subtype: str = 'PCM_16',
    ):
        self._source = source
        self._path = path
        self._output_sample_rate = sample_rate
        self._subtype = subtype
        
        # File handle (opened on on_start)
        self._file: Optional[sf.SoundFile] = None
        self._frames_written: int = 0
    
    @property
    def path(self) -> str:
        """Path to the output WAV file."""
        return self._path
    
    @property
    def frames_written(self) -> int:
        """Number of frames written so far."""
        return self._frames_written
    
    def inputs(self) -> list[ProcessingElement]:
        """Return the input PE."""
        return [self._source]
    
    def is_pure(self) -> bool:
        """
        WavWriterPE is NOT pure - it has file I/O side effects.
        """
        return False
    
    def on_start(self) -> None:
        """Open the WAV file for writing."""
        # Determine sample rate
        rate = self._output_sample_rate or self.sample_rate
        
        # Get channel count from source
        # Note: channel_count might be None for pass-through PEs,
        # so we need to resolve it
        channels = self._source.channel_count()
        if channels is None:
            # Try to get from source's inputs
            source_inputs = self._source.inputs()
            if source_inputs:
                channels = source_inputs[0].channel_count()
        
        if channels is None:
            handle_error(
                f"Cannot determine channel count for WavWriterPE. "
                f"Source {self._source.__class__.__name__} returns None for channel_count().",
                fatal=True,
            )
            return
        
        self._file = sf.SoundFile(
            self._path,
            mode='w',
            samplerate=rate,
            channels=channels,
            subtype=self._subtype,
        )
        self._frames_written = 0
        logger.info(
            f"Opened {self._path} for writing: "
            f"{channels} channels, {rate} Hz, {self._subtype}"
        )
    
    def on_stop(self) -> None:
        """Close the WAV file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info(f"Closed {self._path}: {self._frames_written} frames written")
    
    def render(self, start: int, duration: int) -> Snippet:
        """
        Render from source, write to file, and pass through.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render
        
        Returns:
            Snippet from source (unchanged)
        """
        # Render from source
        snippet = self._source.render(start, duration)
        
        # Write to file if open
        if self._file is not None:
            self._file.write(snippet.data)
            self._frames_written += snippet.duration
        
        # Pass through
        return snippet
    
    def _compute_extent(self) -> Extent:
        """Extent matches the source's extent."""
        return self._source.extent()
    
    def __repr__(self) -> str:
        return (
            f"WavWriterPE(source={self._source.__class__.__name__}, "
            f"path={self._path!r}, subtype={self._subtype!r})"
        )
