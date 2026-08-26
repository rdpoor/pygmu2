"""
SignalToGatePE - Schmitt-trigger analog-to-gate converter.

Converts a continuous analog signal to a binary gate signal (0 or 1) using
Schmitt trigger logic with configurable thresholds, hysteresis, and holdoff.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.extent import Extent
from pygmu2.gate_signal import GateSignal
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet


class SignalToGatePE(GateSignal):
    """
    Schmitt-trigger gate: converts an analog signal to a gate signal.

    Gate opens  when signal rises above ``high_threshold + hysteresis``.
    Gate closes when signal falls below ``low_threshold  - hysteresis``.

    After any transition a ``holdoff_time``-second lockout prevents further
    transitions, eliminating chatter near threshold crossings.

    Args:
        source:         Analog input PE (first channel used if multi-channel).
        low_threshold:  Gate closes below this level (minus hysteresis). Default 0.0.
        high_threshold: Gate opens above this level (plus hysteresis).   Default 1.0.
        hysteresis:     Extra dead-band margin applied to both thresholds. Default 0.0.
        holdoff_time:   Minimum seconds between transitions.              Default 0.0.
    """

    def __init__(
        self,
        source: ProcessingElement,
        low_threshold: float = 0.0,
        high_threshold: float = 1.0,
        hysteresis: float = 0.0,
        holdoff_time: float = 0.0,
    ):
        self._source = source
        self._low_threshold = float(low_threshold)
        self._high_threshold = float(high_threshold)
        self._hysteresis = float(hysteresis)
        self._holdoff_time = float(holdoff_time)
        self._holdoff_count = 0
        self._gate_open = False
        self._holdoff_remaining = 0

    def inputs(self) -> list[ProcessingElement]:
        return [self._source]

    def is_pure(self) -> bool:
        return False

    def _compute_extent(self) -> Extent:
        return self._source.extent()

    def _on_start(self) -> None:
        self._holdoff_count = int(round(self._holdoff_time * self.sample_rate))
        self._gate_open = False
        self._holdoff_remaining = 0

    def _render_gate(self, start: int, duration: int) -> Snippet:
        src_data = self._source.render(start, duration).data[:, 0]

        open_thr = self._high_threshold + self._hysteresis
        close_thr = self._low_threshold - self._hysteresis

        out = np.zeros((duration, 1), dtype=np.float32)
        gate = self._gate_open
        holdoff = self._holdoff_remaining
        holdoff_n = self._holdoff_count

        for i in range(duration):
            if holdoff > 0:
                holdoff -= 1
            elif not gate and src_data[i] > open_thr:
                gate = True
                holdoff = holdoff_n
            elif gate and src_data[i] < close_thr:
                gate = False
                holdoff = holdoff_n
            out[i, 0] = 1.0 if gate else 0.0

        self._gate_open = gate
        self._holdoff_remaining = holdoff
        return Snippet(start, out)

    def __repr__(self) -> str:
        return (
            f"SignalToGatePE(source={self._source!r}, "
            f"low_threshold={self._low_threshold}, "
            f"high_threshold={self._high_threshold}, "
            f"hysteresis={self._hysteresis}, "
            f"holdoff_time={self._holdoff_time})"
        )
