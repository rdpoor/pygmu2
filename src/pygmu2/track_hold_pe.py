"""
TrackHoldPE - follow a source while a gate is open; hold while closed.

While gate=1 the output tracks the source sample-for-sample.
While gate=0 the last tracked value is held.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.gate_signal import GateSignal
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class TrackHoldPE(ProcessingElement):
    """
    Track-and-Hold processing element.

    Follows `source` sample-for-sample while `gate` is 1; freezes and holds
    the last tracked value while `gate` is 0.

    Args:
        source:        Mono control or audio PE to track.
        gate:          GateSignal controlling track (1) / hold (0) behaviour.
        initial_value: Output before the first gate-open period (default 0.0).

    Notes:
        - is_pure() is False; the held value persists between renders.
        - Both source and gate are rendered for the full block each call so
          that impure sources advance their internal state correctly.
        - Only channel 0 of a multi-channel source is used.
    """

    def __init__(
        self,
        source: ProcessingElement,
        gate: GateSignal,
        initial_value: float = 0.0,
    ):
        self._source = source
        self._gate = gate
        self._initial_value = float(initial_value)
        self._held_value = self._initial_value

    @property
    def initial_value(self) -> float:
        return self._initial_value

    def inputs(self) -> list[ProcessingElement]:
        return [self._source, self._gate]

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
        gate = self._gate.render(start, duration).data[:, 0]
        src = self._source.render(start, duration).data[:, 0]

        out = np.empty(duration, dtype=np.float32)
        held = self._held_value
        for i in range(duration):
            if gate[i] > 0.5:
                held = float(src[i])
            out[i] = held
        self._held_value = held

        return Snippet(start, out.reshape(-1, 1))

    def __repr__(self) -> str:
        return (
            f"TrackHoldPE(source={self._source.__class__.__name__}, "
            f"gate={self._gate.__class__.__name__}, "
            f"initial_value={self._initial_value})"
        )
