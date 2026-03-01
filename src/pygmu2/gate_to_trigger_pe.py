"""
GateToTriggerPE - rising-edge detector: GateSignal → TriggerSignal.

Emits +1 at each 0→1 rising edge of the input GateSignal.
All other samples are 0.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.extent import Extent
from pygmu2.gate_signal import GateSignal
from pygmu2.snippet import Snippet
from pygmu2.trigger_signal import TriggerSignal


class GateToTriggerPE(TriggerSignal):
    """
    Converts a GateSignal to a TriggerSignal by emitting +1 at each
    rising edge (0 → 1 transition). All other samples are 0.

    Args:
        gate: A GateSignal whose 0→1 transitions become trigger events.
    """

    def __init__(self, gate: GateSignal):
        self._gate = gate
        self._prev = 0.0

    def inputs(self) -> list:
        return [self._gate]

    def is_pure(self) -> bool:
        return False

    def _compute_extent(self) -> Extent:
        return self._gate.extent()

    def _on_start(self) -> None:
        self._prev = 0.0

    def _render_trigger(self, start: int, duration: int) -> Snippet:
        gate_data = self._gate.render(start, duration).data[:, 0]
        out = np.zeros((duration, 1), dtype=np.float32)
        prev = self._prev
        for i in range(duration):
            g = gate_data[i]
            if g > 0.0 and prev == 0.0:
                out[i, 0] = 1.0
            prev = g
        self._prev = prev
        return Snippet(start, out)

    def __repr__(self) -> str:
        return f"GateToTriggerPE(gate={self._gate!r})"
