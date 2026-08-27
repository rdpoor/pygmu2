"""
SlewLimiterPE - limit the rate of change of a control signal.

The output "chases" the source at no more than `rate` units/second
(symmetric: same limit for rise and fall).

Three modes are available:
  LINEAR      - output changes at a constant rate toward target (units/s)
  EXPONENTIAL - output changes proportionally to remaining error (RC-filter
                style); rate sets the per-sample coefficient so that at full
                error the initial velocity equals the linear mode velocity.
  LOGARITHMIC - slewing is performed in log₂ (octave) space, so `rate` is
                measured in octaves/second.  The output signal stays in the
                original (e.g. Hz) domain.  Useful for frequency glides where
                a musically constant glide speed is desired regardless of
                register: e.g. rate=2 takes 0.5 s to move one octave at any
                base frequency.

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
    LOGARITHMIC = "logarithmic"  # rate in octaves/second; signal stays in Hz


class SlewLimiterPE(ProcessingElement):
    """
    Slew-rate limiter for control signals.

    Args:
        source: Mono control PE to limit.
        rate:   Maximum slew velocity (float or ProcessingElement).
                LINEAR/EXPONENTIAL: units/second.
                LOGARITHMIC: octaves/second (signal remains in Hz).
                Applied symmetrically for both rise and fall.
        mode:   SlewMode.LINEAR (default), SlewMode.EXPONENTIAL, or
                SlewMode.LOGARITHMIC.

    Notes:
        - stateful = True; the current output value is state.
        - LINEAR: output moves toward source at ≤ rate units/second.
        - EXPONENTIAL: per-sample coefficient derived so that the initial
          velocity at maximum error matches LINEAR.  Asymptotically approaches
          target.
        - LOGARITHMIC: slewing computed in log₂ space so `rate` octaves/second
          is constant regardless of register.  If the current value is ≤ 0 the
          output jumps immediately to the target (avoids log(0)).
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

    stateful = True

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
        rate_data = self._scalar_or_pe_values(
            self._rate, start, duration, dtype=np.float32
        )

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
        elif self._mode == SlewMode.EXPONENTIAL:
            for i in range(duration):
                dt = float(rate_data[i]) / sr
                k = min(dt, 1.0)
                error = float(src[i]) - current
                current += k * error
                out[i] = current
        else:  # LOGARITHMIC — slew in octave space, signal stays in Hz
            import math

            for i in range(duration):
                target = float(src[i])
                if current <= 0.0 or target <= 0.0:
                    # Can't take log of non-positive; jump immediately.
                    current = target if target > 0.0 else current
                else:
                    max_oct = float(rate_data[i]) / sr  # octaves per sample
                    delta_oct = math.log2(target) - math.log2(current)
                    if delta_oct > max_oct:
                        delta_oct = max_oct
                    elif delta_oct < -max_oct:
                        delta_oct = -max_oct
                    current *= math.pow(2.0, delta_oct)
                out[i] = current

        self._current = current
        return Snippet(start, out.reshape(-1, 1))

    def __repr__(self) -> str:
        return f"SlewLimiterPE(rate={self._rate!r}, mode={self._mode.value})"
