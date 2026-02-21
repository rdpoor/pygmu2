"""
RandomValuePE - random value generator inspired by the Buchla 266
"Source of Uncertainty" (Fluctuating Random Voltages section).

Implements an Ornstein-Uhlenbeck process:

    y[n] = y[n-1] + α * (noise[n] - y[n-1])

where α = rate / sr.  This is identical to SlewLimiterPE(EXPONENTIAL), so the
implementation is a pure composition:

    NoisePE(0..1, seed) ──► SlewLimiterPE(rate, EXPONENTIAL) ──► output

Output is bounded in [0, 1] because each y[n] is a convex combination of
y[n-1] ∈ [0,1] and noise[n] ∈ [0,1].

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

from pygmu2.extent import Extent
from pygmu2.noise_pe import NoisePE
from pygmu2.processing_element import ProcessingElement
from pygmu2.slew_limiter_pe import SlewLimiterPE, SlewMode
from pygmu2.snippet import Snippet


class RandomValuePE(ProcessingElement):
    """
    Random value generator using an Ornstein-Uhlenbeck process.

    Produces a mono control signal in [0.0, 1.0] that wanders continuously.
    Higher `rate` values mean faster, more erratic variation; lower values
    produce slow, smooth drift.

    Args:
        rate: O-U coefficient × sr.  Accepts float or ProcessingElement.
              Default 10.0 gives a ~100 ms time constant at 44100 Hz.
        seed: Optional RNG seed for reproducible sequences.
    """

    def __init__(
        self,
        rate: float | ProcessingElement = 10.0,
        seed: int | None = None,
    ):
        self._rate = rate
        self._seed = seed

        self._noise = NoisePE(min_value=0.0, max_value=1.0, seed=seed)
        self._output = SlewLimiterPE(self._noise, rate=rate, mode=SlewMode.EXPONENTIAL)

    def inputs(self) -> list[ProcessingElement]:
        """Expose the tip of the internal chain so the Renderer manages lifecycle."""
        return [self._output]

    def _on_start(self) -> None:
        self._noise.on_start()
        self._output.on_start()

    def is_pure(self) -> bool:
        return False

    def channel_count(self) -> int:
        return 1

    def _compute_extent(self) -> Extent:
        return Extent(None, None)

    def _render(self, start: int, duration: int) -> Snippet:
        return self._output.render(start, duration)

    def __repr__(self) -> str:
        return f"RandomValuePE(rate={self._rate!r}, seed={self._seed!r})"
