"""
SampleHoldPE - latch a source value on each trigger event.

On each rising-edge trigger the current output of `source` is latched and
held until the next event.  Between events the last-latched value is output.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.trigger_signal import TriggerSignal
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class SampleHoldPE(ProcessingElement):
    """
    Sample-and-Hold processing element.

    Latches the current output of `source` on each positive trigger event
    and holds that value until the next event.

    Args:
        source:        Mono control or audio PE whose output will be latched.
        trigger:       TriggerSignal that fires latch events (positive values).
        initial_value: Output before the first trigger event (default 0.0).

    Notes:
        - is_pure() is False; state (held value) persists between renders.
        - source is rendered for the full buffer on every render call so that
          impure sources (e.g. NoisePE) advance their internal state correctly.
        - Only channel 0 of a multi-channel source is used.
    """

    def __init__(
        self,
        source: ProcessingElement,
        trigger: TriggerSignal,
        initial_value: float = 0.0,
    ):
        self._source = source
        self._trigger = trigger
        self._initial_value = float(initial_value)
        self._held_value = self._initial_value

    @property
    def initial_value(self) -> float:
        return self._initial_value

    def inputs(self) -> list[ProcessingElement]:
        return [self._source, self._trigger]

    def is_pure(self) -> bool:
        return False

    def channel_count(self) -> int:
        return 1

    def _compute_extent(self) -> Extent:
        return Extent(None, None)

    def _reset_state(self) -> None:
        self._held_value = self._initial_value

    def _on_start(self) -> None:
        self._reset_state()

    def _render(self, start: int, duration: int) -> Snippet:
        trig = self._trigger.render(start, duration).data[:, 0]
        src = self._source.render(start, duration).data[:, 0]

        out = np.empty(duration, dtype=np.float32)
        held = self._held_value
        for i in range(duration):
            if trig[i] > 0:
                held = float(src[i])
            out[i] = held
        self._held_value = held

        return Snippet(start, out.reshape(-1, 1))

    def __repr__(self) -> str:
        return (
            f"SampleHoldPE(source={self._source.__class__.__name__}, "
            f"trigger={self._trigger.__class__.__name__}, "
            f"initial_value={self._initial_value})"
        )
