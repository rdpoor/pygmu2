"""
RandomTriggerPE - Poisson-process trigger generator.

Emits a +1 trigger impulse at each randomly timed jump event.  Jump timing
follows a Poisson process: at each sample there is an independent probability
p = rate / sr of firing, giving exponentially distributed inter-event intervals
with mean sr/rate samples (1/rate seconds).

This is the trigger-output analogue of RandomStepPE (which also samples a new
random value at each jump).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.extent import Extent
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet
from pygmu2.trigger_signal import TriggerSignal


class RandomTriggerPE(TriggerSignal):
    """
    Poisson-process trigger generator.

    Emits a ``+1`` impulse at each randomly timed jump event.  At every sample
    an independent Bernoulli trial with probability ``p = rate / sr`` decides
    whether to fire.  Inter-event intervals are geometrically (approximately
    exponentially) distributed with mean ``sr / rate`` samples.

    Args:
        rate: Mean event rate in events/second.  Accepts ``float`` or
              ``ProcessingElement`` for real-time modulation.
              Default 10.0 → ~100 ms mean inter-event interval at 44100 Hz.
        seed: Optional RNG seed for reproducible event streams.
    """

    def __init__(
        self,
        rate: float | ProcessingElement = 10.0,
        seed: int | None = None,
    ):
        self._rate = rate
        self._seed = seed
        self._rng: np.random.Generator | None = None

    def inputs(self) -> list[ProcessingElement]:
        if isinstance(self._rate, ProcessingElement):
            return [self._rate]
        return []

    def _on_start(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def _on_stop(self) -> None:
        self._rng = None

    def is_pure(self) -> bool:
        return False

    def _compute_extent(self) -> Extent:
        return Extent(None, None)

    def _render_trigger(self, start: int, duration: int) -> Snippet:
        rate_data = self._scalar_or_pe_values(
            self._rate, start, duration, dtype=np.float64
        )
        sr = float(self._sample_rate)
        out = np.zeros((duration, 1), dtype=np.float32)
        rng = self._rng

        for i in range(duration):
            p = min(float(rate_data[i]) / sr, 1.0)
            if rng.random() < p:
                out[i, 0] = 1.0

        return Snippet(start, out)

    def __repr__(self) -> str:
        return f"RandomTriggerPE(rate={self._rate!r}, seed={self._seed!r})"
