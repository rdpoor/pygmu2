"""
PeriodicGatePE - a GateSignal that generates a periodic gate.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.gate_signal import GateSignal
from pygmu2.analog_osc_pe import AnalogOscPE


class PeriodicGatePE(GateSignal):
    """
    A GateSignal that emits a periodic rectangular gate (0/1), with fixed or
    dynamic frequency, duty cycle, and phase.

    Args:
        frequency: Hz, or PE providing per-sample frequency values.
        duty_cycle: fraction high in [0.0, 1.0), or PE providing per-sample duty.
        phase: cycles in [0.0, 1.0), or PE providing per-sample phase offset.
        channels: number of channels (default 1). (GateSignal is mono; this must be 1.)
    """

    def __init__(
        self,
        frequency: float | ProcessingElement = 1.0,
        duty_cycle: float | ProcessingElement = 0.5,
        phase: float | ProcessingElement = 0.0,
    ):
        # GateSignal is mono by definition; enforce here.
        self._fg = AnalogOscPE(
            frequency=frequency,
            duty_cycle=duty_cycle,
            phase=phase,
            waveform=AnalogOscPE.WAVE_RECTANGLE,
            antialias=False,  # a gate needs exact hard edges, not BLEP ramps
            channels=1,
        )

    def inputs(self) -> list[ProcessingElement]:
        # Expose the internal oscillator so the Renderer (and any graph
        # walk) manages its lifecycle — the gate itself is a stateless map
        # of its output; the generator declares its own state.
        return [self._fg]

    def _compute_extent(self) -> Extent:
        return self._fg.extent()

    def _render_gate(self, start: int, duration: int) -> Snippet:
        # FunctionGen rectangle yields exactly -1 or +1, so mapping yields exactly 0 or 1.
        wave = self._fg.render(start, duration).data[:, 0]
        gate = ((wave + 1.0) * 0.5).astype(np.float32).reshape(-1, 1)
        return Snippet(start, gate)
