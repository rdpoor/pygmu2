"""
CropPE - limits audio to a specified time window.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from typing import Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent, ExtendMode
from pygmu2.extent_window_pe import _ExtentWindowPE


class CropPE(_ExtentWindowPE):
    """
    A ProcessingElement that limits its input to a specified range.

    Samples inside the crop range are passed through from the source.
    Behavior outside this extent is controlled by extend_mode.

    The crop range can be open-ended by passing duration=None:
    - duration=None: No upper bound (pass through to end)

    The output extent is the intersection of the crop range and source extent.
    
    Args:
        source: Input ProcessingElement
        start: First sample to include (inclusive)
        duration: Number of samples to include. If None, no upper bound.
        extend_mode: Behavior outside crop extent (default: ZERO)
                     - ZERO: Output zeros outside crop
                     - HOLD_FIRST: Hold first sample value before crop
                     - HOLD_LAST: Hold last sample value after crop
                     - HOLD_BOTH: Hold first before, last after
    
    Example:
        # Crop to samples 44100-88200 (second 1-2 at 44.1kHz)
        reader_stream = WavReaderPE("audio.wav")
        cropped_stream = CropPE(reader_stream, 44100, 44100)
        
        # Sustain last value after crop ends
        from pygmu2 import ExtendMode
        sustained = CropPE(source, 0, 1000, extend_mode=ExtendMode.HOLD_LAST)
        
        # Trim the beginning (start at sample 1000)
        trimmed_stream = CropPE(source_stream, 1000, None)
        
        # Extract a window from an infinite source
        sine = SinePE(frequency=440.0)
        burst = CropPE(sine, 0, 44100)  # 1 second burst
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        start: int,
        duration: Optional[int],
        extend_mode: ExtendMode = ExtendMode.ZERO,
    ):
        if duration is not None and duration < 0:
            raise ValueError(f"duration must be >= 0, got {duration}")

        self._start = int(start)
        self._duration = int(duration) if duration is not None else None
        end = None if self._duration is None else self._start + self._duration
        extent = Extent(self._start, end)
        super().__init__(source, extent, extend_mode)
    
    @property
    def crop_extent(self) -> Extent:
        """The extent to crop to."""
        return self._extent
    
    @property
    def start(self) -> int:
        """First sample to include (inclusive)."""
        return self._start

    @property
    def duration(self) -> Optional[int]:
        """Number of samples to include, or None for no upper bound."""
        return self._duration
    
    @property
    def end(self) -> Optional[int]:
        """First sample to exclude (exclusive), or None for no upper bound."""
        return self._extent.end
    
    def __repr__(self) -> str:
        end_str = str(self._extent.end) if self._extent.end is not None else "None"
        extend_str = f", extend_mode={self._extend_mode.value}" if self._extend_mode != ExtendMode.ZERO else ""
        return (
            f"CropPE(source={self._source.__class__.__name__}, "
            f"start={self._start}, end={end_str}{extend_str})"
        )
