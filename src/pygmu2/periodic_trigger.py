# src/pygmu2/periodic_trigger.py
#
# A TriggerSignal that generates periodic trigger impulses

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
        hz: trigger rate in events/second
        phase: initial phase in [0, 1) cycles
        amplitude: value emitted at event sample (typically 1)
    """
    def __init__(self, hz: float, phase: float = 0.0, amplitude: int = 1):
        if hz <= 0:
            raise ValueError("PeriodicTrigger hz must be > 0")
        self._hz = float(hz)
        self._phase = float(phase) % 1.0
        self._amp = int(amplitude)

        self._period = int(round(get_sample_rate() / self._hz))
        if self._period <= 0:
            raise ValueError("PeriodicTrigger computed period <= 0; check sample rate / hz")

        # Convert phase (cycles) to an offset in samples
        self._phase_samples = int(round(self._phase * self._period))

    def inputs(self) -> list[ProcessingElement]:
        return []

    def is_pure(self) -> bool:
        # Pure: deterministic function of time.
        return True

    def _compute_extent(self) -> Extent:
        return Extent(None, None)

    def _render_trigger(self, start: int, duration: int) -> Snippet:
        out = np.zeros((duration, 1), dtype=np.float32)

        # Emit +amp when (absolute_sample + phase_offset) % period == 0
        # Note: if you want "phase=0 means event at t=0", keep this as-is.
        abs_idx = np.arange(start, start + duration, dtype=np.int64)
        hits = ((abs_idx + self._phase_samples) % self._period) == 0
        out[hits, 0] = float(self._amp)

        return Snippet(start, out)
