"""
RandomValuePE - continuously wandering random voltage generator inspired by
the Buchla 266 "Source of Uncertainty" (Fluctuating Random Voltages section).

Algorithm
---------
At each sample a Bernoulli trial with probability p = rate/sr decides whether
to draw a new target value from Uniform[0, 1].  The output then exponentially
chases the current target with the same coefficient:

    if rng.random() < p:
        target = rng.random()          # new Poisson-rate target
    current += p * (target - current)  # one-pole RC approach
    out[n] = current

With p = rate/sr the time constant equals the mean jump interval (1/rate s),
so the output typically reaches ~63 % of each target before the next one
arrives.  Equilibrium std ≈ 0.20, mean = 0.5; output is bounded in [0, 1].

Why not white-noise + RC filter?
---------------------------------
Filtering audio-rate white noise with a very narrow LFO-rate RC filter
(α = rate/sr ≪ 1) rejects almost all noise power and leaves a near-constant
output at the noise mean (0.5) with standard deviation ≈ sqrt(α/2) * σ_noise.
At rate=10, sr=44100 that is std ≈ 0.003 — indistinguishable from silence.
The Poisson-target approach sidesteps this by sampling the noise source at the
modulation rate rather than the audio rate.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import numpy as np

from pygmu2.extent import Extent
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet


class RandomValuePE(ProcessingElement):
    """
    Continuously wandering random voltage in [0, 1].

    Jumps to a new random target at Poisson-distributed instants (mean
    interval = 1/rate seconds) and exponentially approaches each target with
    the same time constant, giving smooth continuous wandering across the full
    output range.

    Args:
        rate: Mean jump rate in Hz; also sets the approach time constant
              (τ = 1/rate seconds).  Accepts ``float`` or ``ProcessingElement``.
              Default 10.0 → ~100 ms mean hold / approach time.
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
        self._target: float = 0.5
        self._current: float = 0.5

    def inputs(self) -> list[ProcessingElement]:
        if isinstance(self._rate, ProcessingElement):
            return [self._rate]
        return []

    def _on_start(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._target = float(self._rng.random())
        self._current = self._target  # start at target — no initial transient

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
        rng = self._rng
        current = self._current
        target = self._target

        for i in range(duration):
            p = min(float(rate_data[i]) / sr, 1.0)
            if rng.random() < p:  # Poisson jump: pick new target
                target = float(rng.random())
            current += p * (target - current)  # exponential approach
            out[i] = current

        self._current = current
        self._target = target
        return Snippet(start, out.reshape(-1, 1))

    def __repr__(self) -> str:
        return f"RandomValuePE(rate={self._rate!r}, seed={self._seed!r})"
