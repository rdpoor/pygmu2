"""
HoldPE - follow the source while the control is high; hold while it is low.

One PE covers both classic behaviours, selected by what you plug into
`control`:

- TriggerSignal (one-sample events)  -> sample-and-hold: the source value
  is latched at each event and held until the next.
- GateSignal (sustained 0/1 level)   -> track-and-hold: the output follows
  the source while the gate is high and freezes when it drops.

(Replaces the former SampleHoldPE and TrackHoldPE, whose implementations
differed only in the comparison threshold.)

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class HoldPE(ProcessingElement):
    """
    Follow `source` while `control` > 0; hold the last value while it is 0.

    Args:
        source:        Mono control or audio PE to sample from.
        control:       Gate or trigger signal (mono, 0 or positive).
        initial_value: Output before the first high control sample
                       (default 0.0).

    Notes:
        - stateful = True; the held value persists between renders.
        - source is rendered for the full buffer on every render call so
          that stateful sources (e.g. NoisePE) advance correctly.
        - Only channel 0 of a multi-channel source is used.
        - Extent is the source's extent (the held value is only meaningful
          where the source has data).

    Example:
        # Sample-and-hold: stepped random values, 8 steps/second
        stepped = HoldPE(NoisePE(seed=1), GateToTriggerPE(PeriodicGatePE(8.0)))

        # Track-and-hold: follow while the gate is high, freeze while low
        frozen = HoldPE(lfo, PeriodicGatePE(frequency=2.0, duty_cycle=0.5))
    """

    def __init__(
        self,
        source: ProcessingElement,
        control: ProcessingElement,
        initial_value: float = 0.0,
    ):
        self._source = source
        self._control = control
        self._initial_value = float(initial_value)
        self._held_value = self._initial_value

    @property
    def initial_value(self) -> float:
        return self._initial_value

    def inputs(self) -> list[ProcessingElement]:
        return [self._source, self._control]

    stateful = True

    def channel_count(self) -> int:
        return 1

    def _compute_extent(self) -> Extent:
        return self._source.extent()

    def _reset_state(self) -> None:
        self._held_value = self._initial_value

    def _on_start(self) -> None:
        self._reset_state()

    def _render(self, start: int, duration: int) -> Snippet:
        ctrl = self._control.render(start, duration).data[:, 0]
        src = self._source.render(start, duration).data[:, 0]

        out = np.empty(duration, dtype=np.float32)
        held = self._held_value
        for i in range(duration):
            if ctrl[i] > 0:
                held = float(src[i])
            out[i] = held
        self._held_value = held

        return Snippet(start, out.reshape(-1, 1))

    def __repr__(self) -> str:
        return (
            f"HoldPE(source={self._source.__class__.__name__}, "
            f"control={self._control.__class__.__name__}, "
            f"initial_value={self._initial_value})"
        )
