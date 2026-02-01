from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygmu2.meltysynth.synth.synthesizer import Synthesizer


class Lfo:
    def __init__(self, synthesizer: "Synthesizer") -> None:
        self._synthesizer = synthesizer

    def start(self, delay: float, frequency: float) -> None:
        if frequency > 1.0e-3:
            self._active = True
            self._delay = delay
            self._period = 1.0 / frequency
            self._processed_sample_count = 0
            self._value = 0
        else:
            self._active = False
            self._value = 0

    def process(self) -> None:
        if not self._active:
            return

        self._processed_sample_count += self._synthesizer.block_size

        current_time = (
            float(self._processed_sample_count)
            / self._synthesizer.sample_rate
        )

        if current_time < self._delay:
            self._value = 0
        else:
            phase = ((current_time - self._delay) % self._period) / self._period
            if phase < 0.25:
                self._value = 4 * phase
            elif phase < 0.75:
                self._value = 4 * (0.5 - phase)
            else:
                self._value = 4 * (phase - 1.0)

    @property
    def value(self) -> float:
        return self._value
