import math
from collections.abc import MutableSequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygmu2.meltysynth.synth.synthesizer import Synthesizer


class BiQuadFilter:
    def __init__(self, synthesizer: "Synthesizer") -> None:
        self._synthesizer = synthesizer
        self._resonance_peak_offset = 1.0 - 1.0 / math.sqrt(2.0)

    def clear_buffer(self) -> None:
        self._x1 = 0
        self._x2 = 0
        self._y1 = 0
        self._y2 = 0

    def set_low_pass_filter(
        self, cutoff_frequency: float, resonance: float
    ) -> None:
        if cutoff_frequency < 0.499 * self._synthesizer.sample_rate:
            self._active = True

            q = resonance - self._resonance_peak_offset / (
                1 + 6 * (resonance - 1)
            )

            w = (
                2
                * math.pi
                * cutoff_frequency
                / self._synthesizer.sample_rate
            )
            cosw = math.cos(w)
            alpha = math.sin(w) / (2 * q)

            b0 = (1 - cosw) / 2
            b1 = 1 - cosw
            b2 = (1 - cosw) / 2
            a0 = 1 + alpha
            a1 = -2 * cosw
            a2 = 1 - alpha

            self.set_coefficients(a0, a1, a2, b0, b1, b2)
        else:
            self._active = False

    def process(self, block: MutableSequence[float]) -> None:
        if self._active:
            for t in range(len(block)):
                input_val = block[t]
                output = (
                    self._a0 * input_val
                    + self._a1 * self._x1
                    + self._a2 * self._x2
                    - self._a3 * self._y1
                    - self._a4 * self._y2
                )

                self._x2 = self._x1
                self._x1 = input_val
                self._y2 = self._y1
                self._y1 = output

                block[t] = output
        else:
            self._x2 = block[len(block) - 2]
            self._x1 = block[len(block) - 1]
            self._y2 = self._x2
            self._y1 = self._x1

    def set_coefficients(
        self,
        a0: float,
        a1: float,
        a2: float,
        b0: float,
        b1: float,
        b2: float,
    ) -> None:
        self._a0 = b0 / a0
        self._a1 = b1 / a0
        self._a2 = b2 / a0
        self._a3 = a1 / a0
        self._a4 = a2 / a0
