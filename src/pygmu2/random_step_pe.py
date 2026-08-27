"""
RandomStepPE - Poisson sample-and-hold random generator.

Outputs a piecewise-constant (stepped) random signal in [0, 1].  At each
sample the output jumps to a new uniform random value with probability
p = rate / sr, giving exponentially distributed hold times with mean sr/rate
samples (1/rate seconds).

This is the "stepped random voltages" complement to RandomValuePE (continuous
Ornstein-Uhlenbeck wandering) — analogous to the Buchla 266 Source of
Uncertainty stepped vs. fluctuating outputs.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.extent import Extent
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet


class RandomStepPE(ProcessingElement):
    """
    Poisson sample-and-hold random generator.

    Outputs a piecewise-constant signal in [0, 1] that jumps to a new uniform
    random value at Poisson-distributed random instants.  At each sample there
    is an independent probability ``p = rate / sr`` of triggering a jump, so
    hold times are geometrically (approximately exponentially) distributed with
    mean ``sr / rate`` samples (``1 / rate`` seconds).

    Args:
        rate: Jump-rate parameter.  Identical semantics to RandomValuePE:
              ``rate`` = 10.0 gives ~100 ms mean hold time at 44100 Hz.
              Accepts ``float`` or ``ProcessingElement`` for real-time modulation.
        seed: Optional RNG seed for reproducible sequences.
    """

    def __init__(
        self,
        rate: float | ProcessingElement = 10.0,
        seed: int | None = None,
    ):
        self._rate = rate
        self._seed = seed
        self._rng: np.random.Generator | None = None
        self._held_value: float = 0.5

    def inputs(self) -> list[ProcessingElement]:
        """Expose PE-valued rate so the Renderer manages its lifecycle."""
        if isinstance(self._rate, ProcessingElement):
            return [self._rate]
        return []

    def _on_start(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._held_value = float(self._rng.random())

    def _on_stop(self) -> None:
        self._rng = None

    stateful = True

    def channel_count(self) -> int:
        return 1

    def _compute_extent(self) -> Extent:
        return Extent(None, None)

    def _render(self, start: int, duration: int) -> Snippet:
        rate_data = self._scalar_or_pe_values(
            self._rate, start, duration, dtype=np.float64
        )
        sr = float(self._sample_rate)
        out = np.empty(duration, dtype=np.float32)
        held = self._held_value
        rng = self._rng

        for i in range(duration):
            p = min(float(rate_data[i]) / sr, 1.0)
            if rng.random() < p:
                held = float(rng.random())
            out[i] = held

        self._held_value = held
        return Snippet(start, out.reshape(-1, 1))

    def __repr__(self) -> str:
        return f"RandomStepPE(rate={self._rate!r}, seed={self._seed!r})"
