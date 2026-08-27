"""
CropPE - impose a time window on a source.

Absorbs the former SetExtentPE: the two rendered byte-identically and
differed only in the advertised extent (clip=True intersects the window
with the source's extent; clip=False advertises the window verbatim, so
the output can pad past the source — zero-padding, appended silence).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent, ExtendMode
from pygmu2.snippet import Snippet


class CropPE(ProcessingElement):
    """
    A ProcessingElement that imposes a time window on its input.

    Samples inside the window pass through from the source. Behavior
    outside is controlled by extend_mode. The window can be open-ended by
    passing duration=None (no upper bound).

    Args:
        source: Input ProcessingElement
        start: First sample to include (inclusive)
        duration: Number of samples to include. If None, no upper bound.
        extend_mode: Behavior outside the window (default: ZERO)
                     - ZERO: Output zeros outside the window
                     - HOLD_FIRST: Hold first sample value before the window
                     - HOLD_LAST: Hold last sample value after the window
                     - HOLD_BOTH: Hold first before, last after
        clip: Extent policy (keyword-only, default True).
              - True: extent() is the window intersected with the source's
                extent — the window can only shrink what the source offers.
              - False: extent() is the window verbatim — the window may
                extend past the source, and the excess is filled per
                extend_mode (zero-padding by default). This is the former
                SetExtentPE behavior.

    Note on extend_mode and extent(): the hold/zero regions lie OUTSIDE
    the advertised extent, so they are only observable when a consumer
    (e.g. a MixPE sibling with a wider extent) renders past this PE's
    extent — render() itself never clamps to extent().

    Example:
        # Crop to samples 44100-88200 (second 1-2 at 44.1kHz)
        cropped_stream = CropPE(WavReaderPE("audio.wav"), 44100, 44100)

        # One-second burst from an infinite source
        burst = CropPE(SinePE(frequency=440.0), 0, 44100)

        # Zero-pad a short source out to a fixed length
        padded = CropPE(short_stream, 0, 88200, clip=False)

        # Sustain the last value after the window ends
        sustained = CropPE(source, 0, 1000, extend_mode=ExtendMode.HOLD_LAST)
    """

    def __init__(
        self,
        source: ProcessingElement,
        start: int,
        duration: int | None,
        extend_mode: ExtendMode = ExtendMode.ZERO,
        *,
        clip: bool = True,
    ):
        if duration is not None and duration < 0:
            raise ValueError(f"duration must be >= 0, got {duration}")

        self._source = source
        self._start = int(start)
        self._duration = int(duration) if duration is not None else None
        end = None if self._duration is None else self._start + self._duration
        self._window = Extent(self._start, end)
        self._extend_mode = extend_mode
        self._clip = bool(clip)
        self._first_value: np.ndarray | None = None
        self._last_value: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def source(self) -> ProcessingElement:
        return self._source

    @property
    def crop_extent(self) -> Extent:
        """The window extent (before any clipping to the source)."""
        return self._window

    @property
    def extend_mode(self) -> ExtendMode:
        return self._extend_mode

    @property
    def clip(self) -> bool:
        return self._clip

    @property
    def start(self) -> int:
        """First sample to include (inclusive)."""
        return self._start

    @property
    def duration(self) -> int | None:
        """Number of samples to include, or None for no upper bound."""
        return self._duration

    @property
    def end(self) -> int | None:
        """First sample to exclude (exclusive), or None for no upper bound."""
        return self._window.end

    # ------------------------------------------------------------------
    # ProcessingElement interface
    # ------------------------------------------------------------------

    def inputs(self) -> list[ProcessingElement]:
        return [self._source]

    def channel_count(self) -> int | None:
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        if self._clip:
            return self._window.intersection(self._source.extent())
        return self._window

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _get_first_value(self) -> np.ndarray:
        if self._first_value is None:
            snippet = self._source.render(self._window.start, 1)
            self._first_value = snippet.data[0:1, :].copy()
        return self._first_value

    def _get_last_value(self) -> np.ndarray:
        if self._last_value is None:
            snippet = self._source.render(self._window.end - 1, 1)
            self._last_value = snippet.data[0:1, :].copy()
        return self._last_value

    def _render(self, start: int, duration: int) -> Snippet:
        end = start + duration
        win_start = self._window.start
        win_end = self._window.end

        # Overlap between the request and the window
        overlap_start = max(start, win_start)
        overlap_end = end if win_end is None else min(end, win_end)

        # No overlap: fill per extend_mode
        if overlap_start >= overlap_end:
            channels = self._source.channel_count() or 1
            data = np.zeros((duration, channels), dtype=np.float32)
            if end <= win_start:
                if self._extend_mode in (ExtendMode.HOLD_FIRST, ExtendMode.HOLD_BOTH):
                    data[:, :] = self._get_first_value()
            elif win_end is not None and start >= win_end:
                if self._extend_mode in (ExtendMode.HOLD_LAST, ExtendMode.HOLD_BOTH):
                    data[:, :] = self._get_last_value()
            return Snippet(start, data)

        # Full containment: pass the source snippet straight through
        if overlap_start == start and overlap_end == end:
            return self._source.render(start, duration)

        source_snippet = self._source.render(overlap_start, overlap_end - overlap_start)
        channels = source_snippet.channels
        data = np.zeros((duration, channels), dtype=np.float32)

        # Before the window
        if start < win_start:
            if self._extend_mode in (ExtendMode.HOLD_FIRST, ExtendMode.HOLD_BOTH):
                data[: win_start - start, :] = self._get_first_value()

        # Copy the overlap
        out_lo = overlap_start - start
        data[out_lo : out_lo + (overlap_end - overlap_start), :] = source_snippet.data

        # After the window
        if win_end is not None and end > win_end:
            if self._extend_mode in (ExtendMode.HOLD_LAST, ExtendMode.HOLD_BOTH):
                after_start = win_end - start
                if after_start < duration:
                    data[after_start:, :] = self._get_last_value()

        return Snippet(start, data)

    def __repr__(self) -> str:
        end_str = str(self._window.end) if self._window.end is not None else "None"
        extend_str = (
            f", extend_mode={self._extend_mode.value}"
            if self._extend_mode != ExtendMode.ZERO
            else ""
        )
        clip_str = "" if self._clip else ", clip=False"
        return (
            f"CropPE(source={self._source.__class__.__name__}, "
            f"start={self._start}, end={end_str}{extend_str}{clip_str})"
        )
