"""
SlewLimiterPE - limit the rate of change of a control signal.

The output "chases" the source at no more than `rate` units/second
(symmetric: same limit for rise and fall).

Two modes are available:
  LINEAR      - output changes at a constant rate toward target
  EXPONENTIAL - output changes proportionally to remaining error (RC-filter
                style); rate sets the per-sample coefficient so that at full
                error the initial velocity equals the linear mode velocity.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class SlewMode(Enum):
    """Rate-limiting shape."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class SlewLimiterPE(ProcessingElement):
    """
    Slew-rate limiter for control signals.

    Args:
        source: Mono control PE to limit.
        rate:   Maximum slew velocity in units/second (float or ProcessingElement).
                Applied symmetrically for both rise and fall.
        mode:   SlewMode.LINEAR (default) or SlewMode.EXPONENTIAL.

    Notes:
        - is_pure() is False; the current output value is state.
        - In LINEAR mode the output moves toward the source at a constant rate
          of at most `rate` units/second.
        - In EXPONENTIAL mode the per-sample coefficient is derived from rate
          so that the initial velocity (at maximum error) matches the linear
          mode.  The output asymptotically approaches the target.
    """

    def __init__(
        self,
        source: ProcessingElement,
        rate: float | ProcessingElement,
        mode: SlewMode = SlewMode.LINEAR,
    ):
        self._source = source
        if isinstance(rate, ProcessingElement):
            self._rate = rate
            self._rate_is_pe = True
        else:
            if rate <= 0:
                raise ValueError("rate must be > 0")
            self._rate = float(rate)
            self._rate_is_pe = False
        self._mode = mode
        self._current: float = 0.0

    @property
    def rate(self) -> float | ProcessingElement:
        return self._rate

    @property
    def mode(self) -> SlewMode:
        return self._mode

    def inputs(self) -> list[ProcessingElement]:
        if self._rate_is_pe:
            return [self._source, self._rate]
        return [self._source]

    def is_pure(self) -> bool:
        return False

    def channel_count(self) -> int:
        return 1

    def _compute_extent(self) -> Extent:
        return Extent(None, None)

    def _reset_state(self) -> None:
        self._current = 0.0

    def _on_start(self) -> None:
        self._reset_state()

    def _render(self, start: int, duration: int) -> Snippet:
        src = self._source.render(start, duration).data[:, 0]
        rate_data = self._scalar_or_pe_values(self._rate, start, duration, dtype=np.float32)

        sr = float(self._sample_rate)
        out = np.empty(duration, dtype=np.float32)
        current = self._current

        if self._mode == SlewMode.LINEAR:
            for i in range(duration):
                dt = float(rate_data[i]) / sr
                delta = float(src[i]) - current
                if delta > dt:
                    delta = dt
                elif delta < -dt:
                    delta = -dt
                current += delta
                out[i] = current
        else:  # EXPONENTIAL
            for i in range(duration):
                dt = float(rate_data[i]) / sr
                k = min(dt, 1.0)
                error = float(src[i]) - current
                current += k * error
                out[i] = current

        self._current = current
        return Snippet(start, out.reshape(-1, 1))

    def __repr__(self) -> str:
        return (
            f"SlewLimiterPE(rate={self._rate!r}, mode={self._mode.value})"
        )
