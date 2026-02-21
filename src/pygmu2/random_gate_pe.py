"""
RandomGatePE - Poisson-process toggle gate.

Outputs a gate signal (values 0 or 1) that toggles at each randomly timed
jump event.  Jump timing follows a Poisson process: at each sample there is an
independent probability p = rate / sr of toggling, giving exponentially
distributed hold durations with mean sr/rate samples (1/rate seconds).

At rate r, the gate is high for an expected 1/(2r) seconds then low for
1/(2r) seconds, alternating at random intervals.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.extent import Extent
from pygmu2.gate_signal import GateSignal
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet


class RandomGatePE(GateSignal):
    """
    Poisson-process toggle gate.

    Outputs 0 or 1, toggling at each randomly timed event.  At every sample an
    independent Bernoulli trial with probability ``p = rate / sr`` decides
    whether to toggle.  The gate starts in state 0 (low) and alternates
    between 0 and 1 at Poisson-distributed intervals.

    Args:
        rate: Mean toggle rate in events/second.  Accepts ``float`` or
              ``ProcessingElement`` for real-time modulation.
              Default 10.0 → ~100 ms mean hold time at 44100 Hz.
        seed: Optional RNG seed for reproducible gate sequences.
        initial_state: Starting gate value, 0 (default) or 1.
    """

    def __init__(
        self,
        rate: float | ProcessingElement = 10.0,
        seed: int | None = None,
        initial_state: int = 0,
    ):
        self._rate = rate
        self._seed = seed
        self._initial_state = int(bool(initial_state))
        self._rng: np.random.Generator | None = None
        self._gate_state: int = self._initial_state

    def inputs(self) -> list[ProcessingElement]:
        if isinstance(self._rate, ProcessingElement):
            return [self._rate]
        return []

    def _on_start(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._gate_state = self._initial_state

    def _on_stop(self) -> None:
        self._rng = None

    def is_pure(self) -> bool:
        return False

    def _compute_extent(self) -> Extent:
        return Extent(None, None)

    def _render_gate(self, start: int, duration: int) -> Snippet:
        rate_data = self._scalar_or_pe_values(
            self._rate, start, duration, dtype=np.float64
        )
        sr = float(self._sample_rate)
        out = np.zeros((duration, 1), dtype=np.float32)
        rng = self._rng
        gate = self._gate_state

        for i in range(duration):
            p = min(float(rate_data[i]) / sr, 1.0)
            if rng.random() < p:
                gate = 1 - gate
            out[i, 0] = float(gate)

        self._gate_state = gate
        return Snippet(start, out)

    def __repr__(self) -> str:
        return (
            f"RandomGatePE(rate={self._rate!r}, seed={self._seed!r}, "
            f"initial_state={self._initial_state!r})"
        )
