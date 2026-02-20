"""
SlewLimiterPE - limit the rate of change of a control signal.

The output "chases" the source at no more than `rise_rate` (units/second
upward) and `fall_rate` (units/second downward).

Two modes are available:
  LINEAR      - output changes at a constant rate toward target
  EXPONENTIAL - output changes proportionally to remaining error (RC-filter
                style); rise_rate / fall_rate set the per-sample coefficient
                so that at full error the initial velocity equals the linear
                mode velocity.

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
        source:    Mono control PE to limit.
        rise_rate: Maximum upward velocity in units/second.
        fall_rate: Maximum downward velocity in units/second.
                   Defaults to `rise_rate` (symmetric) if None.
        mode:      SlewMode.LINEAR (default) or SlewMode.EXPONENTIAL.

    Notes:
        - is_pure() is False; the current output value is state.
        - In LINEAR mode the output moves toward the source at a constant rate
          of at most `rise_rate` or `fall_rate` units/second.
        - In EXPONENTIAL mode the per-sample coefficients are derived from the
          rates so that the initial velocity (at maximum error) matches the
          linear mode.  The output asymptotically approaches the target.
    """

    def __init__(
        self,
        source: ProcessingElement,
        rise_rate: float,
        fall_rate: float | None = None,
        mode: SlewMode = SlewMode.LINEAR,
    ):
        if rise_rate <= 0:
            raise ValueError("rise_rate must be > 0")
        self._source = source
        self._rise_rate = float(rise_rate)
        self._fall_rate = float(fall_rate) if fall_rate is not None else self._rise_rate
        if self._fall_rate <= 0:
            raise ValueError("fall_rate must be > 0")
        self._mode = mode
        self._current: float = 0.0

    @property
    def rise_rate(self) -> float:
        return self._rise_rate

    @property
    def fall_rate(self) -> float:
        return self._fall_rate

    @property
    def mode(self) -> SlewMode:
        return self._mode

    def inputs(self) -> list[ProcessingElement]:
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

        sr = float(self._sample_rate)
        rise_dt = self._rise_rate / sr  # max upward change per sample
        fall_dt = self._fall_rate / sr  # max downward change per sample
        # Exponential coefficients: k = rate / sr, clamped to (0, 1)
        rise_k = min(rise_dt, 1.0)
        fall_k = min(fall_dt, 1.0)

        out = np.empty(duration, dtype=np.float32)
        current = self._current

        if self._mode == SlewMode.LINEAR:
            for i in range(duration):
                delta = float(src[i]) - current
                if delta > rise_dt:
                    delta = rise_dt
                elif delta < -fall_dt:
                    delta = -fall_dt
                current += delta
                out[i] = current
        else:  # EXPONENTIAL
            for i in range(duration):
                error = float(src[i]) - current
                if error > 0:
                    current += rise_k * error
                else:
                    current += fall_k * error
                out[i] = current

        self._current = current
        return Snippet(start, out.reshape(-1, 1))

    def __repr__(self) -> str:
        return (
            f"SlewLimiterPE(rise_rate={self._rise_rate}, "
            f"fall_rate={self._fall_rate}, mode={self._mode.value})"
        )
