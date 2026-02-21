"""
PeriodicTrigger - a TriggerSignal that generates periodic trigger impulses.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np
from pygmu2.extent import Extent
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet
from pygmu2.trigger_signal import TriggerSignal
from pygmu2.config import get_sample_rate

class PeriodicTrigger(TriggerSignal):
    """
    A TriggerSignal that emits +1 impulses periodically.

    Args:
        hz: trigger rate in events/second (float or ProcessingElement)
        phase: initial phase in [0, 1) cycles
        amplitude: value emitted at event sample (typically 1)

    When hz is a float, rendering is index-based (pure, stateless).
    When hz is a ProcessingElement, a phase accumulator tracks the
    instantaneous frequency (impure, stateful).
    """
    def __init__(self, hz: float | ProcessingElement = 1.0, phase: float = 0.0, amplitude: int = 1):
        self._phase_init = float(phase) % 1.0
        self._amp = int(amplitude)

        if isinstance(hz, ProcessingElement):
            self._hz = hz
            self._hz_is_pe = True
            self._phase_accum = self._phase_init
        else:
            if hz <= 0:
                raise ValueError("PeriodicTrigger hz must be > 0")
            self._hz = float(hz)
            self._hz_is_pe = False

            self._period = int(round(get_sample_rate() / self._hz))
            if self._period <= 0:
                raise ValueError("PeriodicTrigger computed period <= 0; check sample rate / hz")

            # Convert phase (cycles) to an offset in samples
            self._phase_samples = int(round(self._phase_init * self._period))

    def inputs(self) -> list[ProcessingElement]:
        if self._hz_is_pe:
            return [self._hz]
        return []

    def is_pure(self) -> bool:
        return not self._hz_is_pe

    def _compute_extent(self) -> Extent:
        return Extent(None, None)

    def _reset_state(self) -> None:
        if self._hz_is_pe:
            self._phase_accum = self._phase_init

    def _on_start(self) -> None:
        self._reset_state()

    def _render_trigger(self, start: int, duration: int) -> Snippet:
        out = np.zeros((duration, 1), dtype=np.float32)

        if self._hz_is_pe:
            hz_data = self._scalar_or_pe_values(self._hz, start, duration, dtype=np.float64)
            sr = float(self._sample_rate)
            phase = self._phase_accum
            for i in range(duration):
                phase += hz_data[i] / sr
                if phase >= 1.0 - 1e-9:
                    phase -= 1.0
                    out[i, 0] = float(self._amp)
            self._phase_accum = phase
        else:
            # Static float mode: index-based (pure)
            abs_idx = np.arange(start, start + duration, dtype=np.int64)
            hits = ((abs_idx + self._phase_samples) % self._period) == 0
            out[hits, 0] = float(self._amp)

        return Snippet(start, out)
