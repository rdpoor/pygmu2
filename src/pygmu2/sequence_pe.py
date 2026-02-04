"""
SequencePE - schedule PEs at specific start times.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.crop_pe import CropPE
from pygmu2.delay_pe import DelayPE
from pygmu2.mix_pe import MixPE


class SequenceMode(Enum):
    """Sequencer overlap behavior."""
    OVERLAP = "overlap"        # Mix overlapping segments
    NON_OVERLAP = "non_overlap"  # Stop each segment when the next begins


class SequencePE(ProcessingElement):
    """
    Schedule PEs at specific start times.

    Args:
        *input_start_pairs: (pe, start) pairs, where start is in samples.
            You may also pass a single list/tuple of pairs.
        mode: SequenceMode or "overlap"/"non_overlap" (default: OVERLAP)

    Notes:
        - Each PE is rendered starting from its own t=0 at the given start time.
        - OVERLAP mode mixes overlapping segments.
        - NON_OVERLAP mode crops each segment to end at the next start time.
    """

    def __init__(
        self,
        *input_start_pairs: Tuple[ProcessingElement, int],
        mode: SequenceMode | str = SequenceMode.OVERLAP,
    ):
        if len(input_start_pairs) == 1 and isinstance(input_start_pairs[0], (list, tuple)):
            pairs = list(input_start_pairs[0])
        else:
            pairs = list(input_start_pairs)

        if not pairs:
            raise ValueError("SequencePE requires at least one (pe, start) pair")

        normalized: list[tuple[ProcessingElement, int]] = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each input must be a (pe, start) pair")
            pe, start = pair
            normalized.append((pe, int(start)))

        if isinstance(mode, str):
            mode = SequenceMode(mode.lower())
        self._mode = mode

        # Sort by start time to define CUT boundaries deterministically.
        normalized.sort(key=lambda p: p[1])
        self._pairs = normalized

        # Build composed graph.
        scheduled: list[ProcessingElement] = []
        for idx, (pe, start) in enumerate(self._pairs):
            delayed = DelayPE(pe, delay=start)

            if self._mode == SequenceMode.NON_OVERLAP and idx + 1 < len(self._pairs):
                next_start = self._pairs[idx + 1][1]
                cropped = CropPE(delayed, Extent(start, next_start))
                scheduled.append(cropped)
            else:
                scheduled.append(delayed)

        if len(scheduled) == 1:
            self._out = scheduled[0]
        else:
            self._out = MixPE(*scheduled)

    @property
    def mode(self) -> SequenceMode:
        return self._mode

    def inputs(self) -> list[ProcessingElement]:
        return [self._out]

    def is_pure(self) -> bool:
        return self._out.is_pure()

    def channel_count(self) -> Optional[int]:
        return self._out.channel_count()

    def _compute_extent(self) -> Extent:
        return self._out.extent()

    def _render(self, start: int, duration: int):
        return self._out.render(start, duration)

    def __repr__(self) -> str:
        return f"SequencePE(pairs={len(self._pairs)}, mode={self._mode.value})"
