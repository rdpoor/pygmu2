"""
SetExtentPE - force a PE to a specified extent (may extend or truncate).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent, ExtendMode
from pygmu2.snippet import Snippet


class SetExtentPE(ProcessingElement):
    """
    Force a PE to a specified extent, padding or truncating as needed.

    Samples inside the extent are passed through from the source. Samples
    outside the extent are handled according to extend_mode.

    The extent can be open-ended by passing None:
    - start=None: No lower bound (extends infinitely backward)
    - duration=None: No upper bound (extends infinitely forward)
    - Both None: Pass through everything (identity operation)

    Args:
        source: Input ProcessingElement
        start: First sample to include (inclusive), or None for no lower bound
        duration: Number of samples to include, or None for no upper bound
        extend_mode: Behavior outside extent (default: ZERO)
                     - ZERO: Output zeros outside extent
                     - HOLD_FIRST: Hold first sample value before extent
                     - HOLD_LAST: Hold last sample value after extent
                     - HOLD_BOTH: Hold first before, last after
    """

    def __init__(
        self,
        source: ProcessingElement,
        start: Optional[int],
        duration: Optional[int],
        extend_mode: ExtendMode = ExtendMode.ZERO,
    ):
        if duration is not None and duration < 0:
            raise ValueError(f"duration must be >= 0, got {duration}")

        self._source = source
        self._start = int(start) if start is not None else None
        self._duration = int(duration) if duration is not None else None
        end = None
        if self._duration is not None:
            end = self._duration if self._start is None else self._start + self._duration
        self._extent = Extent(self._start, end)
        self._extend_mode = extend_mode
        # Cache first/last values when needed
        self._first_value: Optional[np.ndarray] = None
        self._last_value: Optional[np.ndarray] = None

    @property
    def source(self) -> ProcessingElement:
        return self._source

    @property
    def start(self) -> Optional[int]:
        return self._start

    @property
    def duration(self) -> Optional[int]:
        return self._duration

    @property
    def end(self) -> Optional[int]:
        return self._extent.end

    @property
    def extend_mode(self) -> ExtendMode:
        return self._extend_mode

    def inputs(self) -> list[ProcessingElement]:
        return [self._source]

    def is_pure(self) -> bool:
        return True

    def _get_first_value(self) -> Optional[np.ndarray]:
        if self._first_value is not None:
            return self._first_value
        crop_start = self._extent.start
        if crop_start is not None:
            try:
                snippet = self._source.render(crop_start, 1)
                self._first_value = snippet.data[0:1, :].copy()
                return self._first_value
            except Exception:
                return None
        return None

    def _get_last_value(self) -> Optional[np.ndarray]:
        if self._last_value is not None:
            return self._last_value
        crop_end = self._extent.end
        if crop_end is not None and crop_end > 0:
            try:
                snippet = self._source.render(crop_end - 1, 1)
                self._last_value = snippet.data[0:1, :].copy()
                return self._last_value
            except Exception:
                return None
        return None

    def _render(self, start: int, duration: int) -> Snippet:
        end = start + duration
        crop_start = self._extent.start
        crop_end = self._extent.end

        # Calculate overlap between request and crop window
        overlap_start = start if crop_start is None else max(start, crop_start)
        overlap_end = end if crop_end is None else min(end, crop_end)

        # Determine channels
        channels = self._source.channel_count()
        if channels is None:
            source_inputs = self._source.inputs()
            if source_inputs:
                channels = source_inputs[0].channel_count()
            if channels is None:
                channels = 1

        # Special case: no lower bound, finite end
        if crop_start is None and crop_end is not None and start < crop_end:
            if end <= crop_end:
                source_snippet = self._source.render(start, duration)
                return source_snippet

        # No overlap
        if (overlap_start >= overlap_end or
            (crop_start is not None and end <= crop_start) or
            (crop_end is not None and start >= crop_end)):
            data = np.zeros((duration, channels), dtype=np.float32)

            if crop_start is not None and end <= crop_start:
                if self._extend_mode in (ExtendMode.HOLD_FIRST, ExtendMode.HOLD_BOTH):
                    first_val = self._get_first_value()
                    if first_val is not None:
                        data[:, :] = first_val
            elif crop_end is not None and start >= crop_end:
                if self._extend_mode in (ExtendMode.HOLD_LAST, ExtendMode.HOLD_BOTH):
                    last_val = self._get_last_value()
                    if last_val is not None:
                        data[:, :] = last_val

            return Snippet(start, data)

        source_snippet = self._source.render(overlap_start, overlap_end - overlap_start)
        channels = source_snippet.channels
        data = np.zeros((duration, channels), dtype=np.float32)

        # Before crop
        if crop_start is not None and start < crop_start:
            if self._extend_mode in (ExtendMode.HOLD_FIRST, ExtendMode.HOLD_BOTH):
                first_val = self._get_first_value()
                if first_val is not None:
                    before_count = crop_start - start
                    data[:before_count, :] = first_val

        # Copy overlap
        output_start = overlap_start - start
        output_end = output_start + (overlap_end - overlap_start)
        data[output_start:output_end, :] = source_snippet.data

        # After crop
        if crop_end is not None and end > crop_end:
            if self._extend_mode in (ExtendMode.HOLD_LAST, ExtendMode.HOLD_BOTH):
                last_val = self._get_last_value()
                if last_val is not None:
                    after_start = crop_end - start
                    if after_start < duration:
                        data[after_start:, :] = last_val

        return Snippet(start, data)

    def _compute_extent(self) -> Extent:
        source_extent = self._source.extent()
        return self._extent.intersection(source_extent)

    def channel_count(self) -> Optional[int]:
        return self._source.channel_count()

    def __repr__(self) -> str:
        start_str = str(self._extent.start) if self._extent.start is not None else "None"
        end_str = str(self._extent.end) if self._extent.end is not None else "None"
        extend_str = f", extend_mode={self._extend_mode.value}" if self._extend_mode != ExtendMode.ZERO else ""
        return f"SetExtentPE(source={self._source.__class__.__name__}, extent=Extent({start_str}, {end_str}){extend_str})"
