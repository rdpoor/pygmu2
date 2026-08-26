"""
etude #12: a funny sounding instrument

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pygmu2 as pg
from pathlib import Path
import numpy as np

SRATE = 44100
pg.set_sample_rate(SRATE)


def sigmoid(x, L=1, k=1, x0=0.5):
    return L / (1 + np.exp(-k * (x - x0)))


class WarglePE(pg.ProcessingElement):
    """
    Warble + Gargle: Stochastic additive synthesizer whose harmonic amplitudes
    wander independently over time.

    A set of sine oscillators (warglets) is built, one per partial.  Each
    warglet's amplitude is driven by a RandomValuePE passed through a sigmoid
    shaper, so every partial continuously fades in and out at a random rate.
    The result is summed and scaled by the master amplitude.

    Args:
        frequency: Fundamental frequency in Hz, or a PE providing per-sample
            values.
        amplitude: Master output gain (scalar or PE).
        rate: Rate (Hz) at which the per-partial amplitudes change.  Higher
            values produce more rapid timbral fluctuation.
        partials: If an int, use the first N integer partials (1, 2, ..., N).
            If a list of floats, use those as exact frequency multipliers
            (e.g. [1, 2.01, 3.02] for slightly detuned partials).
        intensity: Sigmoid steepness controlling how sharply amplitudes gate
            on and off.  Low values (≈1) give smooth, gradual fades; high
            values (≈10+) make each partial snap between silent and full.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        frequency: float | pg.ProcessingElement,
        amplitude: float | pg.ProcessingElement,
        rate: float | pg.ProcessingElement = 10.0,
        partials: int | list[float] = 3,
        intensity: float = 2.0,
        seed: int | None = None,
    ):
        self._frequency = frequency
        self._amplitude = amplitude
        self._rate = rate
        self._partials = partials
        self._intensity = intensity
        self._seed = seed

        warglets = []

        if isinstance(self._partials, int):
            for i in range(self._partials):
                warglets.append(self.make_warglet(i + 1))
        else:
            for partial in self._partials:
                warglets.append(self.make_warglet(partial))

        # Use nested GainPE since self._amplitude might be a PE, not a constant
        self._mix = pg.GainPE(
            pg.GainPE(pg.MixPE(*warglets), 1.0 / len(warglets)), self._amplitude
        )

    def inputs(self) -> list[pg.ProcessingElement]:
        candidates = [self._frequency, self._amplitude, self._rate]
        inputs = [e for e in candidates if isinstance(e, pg.ProcessingElement)]
        # Be sure to expose mix as an "input" so tree walking will find it
        return inputs + [self._mix]

    def _on_start(self) -> None:
        for pe in self.inputs():
            pe.on_start()

    def _on_stop(self) -> None:
        for pe in self.inputs():
            pe.on_stop()

    def is_pure(self) -> bool:
        return False

    def channel_count(self) -> int:
        return 1

    def _compute_extent(self):
        return pg.Extent(None, None)

    def _render(self, start: int, duration: int) -> pg.Snippet:
        return self._mix.render(start, duration)

    def make_warglet(self, partial: int) -> pg.ProcessingElement:
        amp = pg.RandomValuePE(rate=self._rate, seed=self._seed)
        amp = pg.TransformPE(amp, func=lambda x: sigmoid(x, k=self._intensity))
        if isinstance(self._frequency, int):
            freq = self._frequency * partial
        elif partial == 1:
            freq = self._frequency
        else:
            freq = pg.TransformPE(self._frequency, func=lambda x: partial * x)
        return pg.SinePE(frequency=freq, amplitude=amp)


if __name__ == "__main__":

    def demo_wargle(
        rate: float = 10.0,
        partials: int | list[float] = 3,
        intensity: float = 2.0,
    ):
        print(f"rate={rate}, partials={partials}, intensity={intensity}")
        vox = WarglePE(
            frequency=110,
            amplitude=0.5,
            rate=rate,
            partials=partials,
            intensity=intensity,
        )
        pg.play_offline(pg.CropPE(vox, 0, 10 * SRATE))

    print("===== for example")
    demo_wargle(rate=20, partials=[2, 3, 4, 5, 6], intensity=10.0)
    print("===== stepping rate")
    for rate in [3, 10, 30]:
        demo_wargle(rate=rate, partials=5, intensity=4.0)
    print("===== stepping fixed partials")
    for partials in [2, 5, 8]:
        demo_wargle(rate=10, partials=partials, intensity=2.0)
    print("===== explicit partials")
    for partials in [
        [1, 3, 5],
        [1, 2, 4, 6],
        [5, 6, 7, 9],
        [1, 2.01, 2.99, 4.02, 4.98, 6.03],
    ]:
        demo_wargle(rate=10, partials=partials, intensity=2.0)
    print("===== stepping intensity")
    for intensity in [1, 3, 10, 30]:
        demo_wargle(rate=10, partials=6, intensity=intensity)
