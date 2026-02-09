"""
SetExtentPE - force a PE to a specified extent (may extend or truncate).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from typing import Optional

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent, ExtendMode
from pygmu2.extent_window_pe import _ExtentWindowPE


class SetExtentPE(_ExtentWindowPE):
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

        self._start = int(start) if start is not None else None
        self._duration = int(duration) if duration is not None else None
        end = None
        if self._duration is not None:
            end = self._duration if self._start is None else self._start + self._duration
        extent = Extent(self._start, end)
        super().__init__(source, extent, extend_mode)

    @property
    def start(self) -> Optional[int]:
        return self._start

    @property
    def duration(self) -> Optional[int]:
        return self._duration

    @property
    def end(self) -> Optional[int]:
        return self._extent.end

    def _compute_extent(self) -> Extent:
        return self._extent

    def __repr__(self) -> str:
        start_str = str(self._extent.start) if self._extent.start is not None else "None"
        end_str = str(self._extent.end) if self._extent.end is not None else "None"
        extend_str = f", extend_mode={self._extend_mode.value}" if self._extend_mode != ExtendMode.ZERO else ""
        return f"SetExtentPE(source={self._source.__class__.__name__}, extent=Extent({start_str}, {end_str}){extend_str})"
