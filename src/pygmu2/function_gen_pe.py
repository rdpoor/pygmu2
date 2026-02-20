"""
FunctionGenPE - simple "DSP-like" function generator (naive rectangle + saw/triangle morph).

This PE intentionally does NOT implement anti-aliasing (no BLIT/BLEP).
It is useful as a low-level building block and for “raw” DSP experiments.

Waveforms:
- "rectangle": naive PWM pulse (+1 for phase < duty, else -1)
- "sawtooth": duty-controlled morph where:
    duty=0.0   -> ascending ramp (saw up)
    duty=0.5   -> triangle
    duty=1.0   -> descending ramp (saw down)

Notes:
- Because the saw/triangle morph uses a piecewise-linear definition with a
  controllable peak location, duty values very close to 0 or 1 can become
  numerically ill-conditioned (division by tiny numbers). We treat the exact
  endpoints (0 and 1) as the saw up/down limits and clamp only for the
  interior case.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations


import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.config import get_sample_rate

class FunctionGenPE(ProcessingElement):
    """
    Naive function generator (no anti-aliasing).

    Args:
        frequency: Frequency in Hz, or PE providing per-sample frequency values.
        duty_cycle: Duty cycle in [0, 1], or PE providing per-sample duty values.
        waveform: "rectangle" or "sawtooth"
        channels: Number of output channels (default: 1)
    """

    WAVE_RECTANGLE = "rectangle"
    WAVE_SAWTOOTH = "sawtooth"

    def __init__(
        self,
        frequency: float | ProcessingElement = 1.0,
        duty_cycle: float | ProcessingElement = 0.5,
        phase: float | ProcessingElement = 0.0,
        waveform: str = "rectangle",
        channels: int = 1,
    ):
        self._frequency = frequency
        self._duty_cycle = duty_cycle
        self._phase_in = phase                                 # NEW
        self._waveform = str(waveform).lower()
        self._channels = int(channels)

        if self._waveform not in (self.WAVE_RECTANGLE, self.WAVE_SAWTOOTH):
            raise ValueError(f"waveform must be 'rectangle' or 'sawtooth', got {waveform!r}")
        if self._channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        # Stateful path for PE-driven params
        self._phase: float = 0.0  # [0,1)
        self._last_render_end: int | None = None

    @property
    def frequency(self) -> float | ProcessingElement:
        return self._frequency

    @property
    def duty_cycle(self) -> float | ProcessingElement:
        return self._duty_cycle

    @property
    def phase(self) -> float | ProcessingElement:         # NEW
        return self._phase_in

    @property
    def waveform(self) -> str:
        return self._waveform

    def inputs(self) -> list[ProcessingElement]:
        result: list[ProcessingElement] = []
        if isinstance(self._frequency, ProcessingElement):
            result.append(self._frequency)
        if isinstance(self._duty_cycle, ProcessingElement):
            result.append(self._duty_cycle)
        if isinstance(self._phase_in, ProcessingElement):       # NEW
            result.append(self._phase_in)
        return result

    def is_pure(self) -> bool:
        return not self.inputs()

    def channel_count(self) -> int:
        return self._channels

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._phase = 0.0
        self._last_render_end = None

    def _compute_extent(self) -> Extent:
        result = Extent(None, None)
        for pe_input in self.inputs():
            result = result.intersection(pe_input.extent())
        return result

    @staticmethod
    def _piecewise_linear(phase: np.ndarray, duty: np.ndarray) -> np.ndarray:
        """
        Duty-controlled saw/triangle morph:
        duty=0 -> rising saw, duty=0.5 -> triangle, duty=1 -> falling saw.
        """
        duty = np.clip(duty, 0.0, 1.0)

        # Peak location a = 1 - duty
        a = 1.0 - duty

        # Handle endpoints explicitly (avoid division by zero):
        # duty==0 -> rising saw: 2*phase-1
        # duty==1 -> falling saw: 1-2*phase
        eps = 1e-12
        m_up = duty <= eps
        m_down = duty >= 1.0 - eps
        m_mid = ~(m_up | m_down)

        y = np.empty_like(phase, dtype=np.float64)
        y[m_up] = 2.0 * phase[m_up] - 1.0
        y[m_down] = 1.0 - 2.0 * phase[m_down]

        if np.any(m_mid):
            a_mid = np.clip(a[m_mid], eps, 1.0 - eps)
            p = phase[m_mid]
            rise = p < a_mid
            y_mid = np.empty_like(p, dtype=np.float64)
            # -1 -> +1 over [0,a)
            y_mid[rise] = -1.0 + 2.0 * (p[rise] / a_mid[rise])
            # +1 -> -1 over [a,1)
            y_mid[~rise] = 1.0 - 2.0 * ((p[~rise] - a_mid[~rise]) / (1.0 - a_mid[~rise]))
            y[m_mid] = y_mid

        return y

    def _render(self, start: int, duration: int) -> Snippet:
        freq = self._scalar_or_pe_values(self._frequency, start, duration, dtype=np.float64)
        duty = self._scalar_or_pe_values(self._duty_cycle, start, duration, dtype=np.float64)
        ph_in = self._scalar_or_pe_values(self._phase_in, start, duration, dtype=np.float64)  # NEW


        # Phase increment per sample (cycles/sample)
        dt = freq / float(get_sample_rate())   # or your existing sr variable

        if self.is_pure():
            idx = np.arange(start, start + duration, dtype=np.float64)
            base_phase = np.mod(idx * float(dt[0]), 1.0)
        else:
            if self._last_render_end is None or start != self._last_render_end:
                self._phase = 0.0

            inc = np.concatenate(([0.0], np.cumsum(dt[:-1], dtype=np.float64)))
            base_phase = np.mod(self._phase + inc, 1.0)

            self._phase = float(np.mod(self._phase + float(np.sum(dt)), 1.0))
            self._last_render_end = start + duration

        # Apply phase offset (scalar or per-sample), in cycles
        phase = np.mod(base_phase + ph_in, 1.0)


        # Naive waveform
        duty = np.clip(duty, 0.0, 1.0)
        if self._waveform == self.WAVE_RECTANGLE:
            y = np.where(phase < duty, 1.0, -1.0).astype(np.float64)
        else:
            y = self._piecewise_linear(phase.astype(np.float64), duty.astype(np.float64))

        data = y.reshape(-1, 1)
        if self._channels > 1:
            data = np.tile(data, (1, self._channels))
        return Snippet(start, data.astype(np.float32))

    def __repr__(self) -> str:
        freq_str = (
            f"{self._frequency.__class__.__name__}"
            if isinstance(self._frequency, ProcessingElement)
            else str(self._frequency)
        )
        duty_str = (
            f"{self._duty_cycle.__class__.__name__}"
            if isinstance(self._duty_cycle, ProcessingElement)
            else str(self._duty_cycle)
        )
        return (
            f"FunctionGenPE(frequency={freq_str}, duty_cycle={duty_str}, "
            f"waveform={self._waveform!r}, channels={self._channels})"
        )

