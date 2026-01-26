"""
AnalogOscPE - bandlimited analog-style oscillator (PWM + morphing saw/triangle).

This PE is intended to feel familiar to analog synth users:
- "rectangle": a pulse wave with duty-cycle (PWM) control
- "sawtooth": a duty-controlled morph where:
    duty=0.0   -> ascending ramp (saw up)
    duty=0.5   -> triangle
    duty=1.0   -> descending ramp (saw down)

The output is bandlimited using polyBLEP-style discontinuity correction.

Notes:
- No explicit "bandwidth" parameter is provided; patch a filter PE (e.g. LadderPE)
  for classic subtractive synth tone shaping.
- duty_cycle is clamped away from 0 and 1 (and away from the polyBLEP window
  around discontinuities) to avoid degeneracies under modulation.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class AnalogOscPE(ProcessingElement):
    """
    Bandlimited analog-style oscillator (PWM rectangle + duty-controlled saw/triangle morph).

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
        frequency: Union[float, ProcessingElement] = 440.0,
        duty_cycle: Union[float, ProcessingElement] = 0.5,
        waveform: str = "rectangle",
        channels: int = 1,
    ):
        self._frequency = frequency
        self._duty_cycle = duty_cycle
        self._waveform = str(waveform).lower()
        self._channels = int(channels)

        if self._waveform not in (self.WAVE_RECTANGLE, self.WAVE_SAWTOOTH):
            raise ValueError(f"waveform must be 'rectangle' or 'sawtooth', got {waveform!r}")
        if self._channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        # Stateful path: phase + (for saw/triangle morph) current output value
        self._phase: float = 0.0  # [0,1)
        self._saw_value: float = -1.0
        self._last_render_end: Optional[int] = None

    @property
    def frequency(self) -> Union[float, ProcessingElement]:
        return self._frequency

    @property
    def duty_cycle(self) -> Union[float, ProcessingElement]:
        return self._duty_cycle

    @property
    def waveform(self) -> str:
        return self._waveform

    def inputs(self) -> list[ProcessingElement]:
        result: list[ProcessingElement] = []
        if isinstance(self._frequency, ProcessingElement):
            result.append(self._frequency)
        if isinstance(self._duty_cycle, ProcessingElement):
            result.append(self._duty_cycle)
        return result

    def is_pure(self) -> bool:
        return not self.inputs()

    def channel_count(self) -> int:
        return self._channels

    def on_start(self) -> None:
        self._reset_state()

    def on_stop(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._phase = 0.0
        self._saw_value = -1.0
        self._last_render_end = None

    def _compute_extent(self) -> Extent:
        """
        If all inputs are constants: infinite extent.
        If any input is a PE: intersection of input extents.
        """
        result = Extent(None, None)
        for pe_input in self.inputs():
            result = result.intersection(pe_input.extent())
        return result

    @staticmethod
    def _blep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """
        4-point polyBLEP residual for step discontinuities.

        t: phase in [0,1)
        dt: phase increment per sample (> 0), same shape as t
        """
        y = np.zeros_like(t, dtype=np.float64)

        # Work only where dt > 0
        dt = np.maximum(dt, 1e-12)

        m = t < (2.0 * dt)
        if np.any(m):
            x = np.zeros_like(t, dtype=np.float64)
            x[m] = t[m] / dt[m]

            u = 2.0 - x
            y[m] += (u[m] ** 4)

            m2 = t < dt
            if np.any(m2):
                v = 1.0 - x
                y[m2] -= 4.0 * (v[m2] ** 4)

        return y / 12.0

    @classmethod
    def _blep_residual(cls, t: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """
        Double-sided residual around a discontinuity at phase 0.
        """
        t = np.mod(t, 1.0)
        return cls._blep(t, dt) - cls._blep(1.0 - t, dt)

    @staticmethod
    def _phase_pure(start: int, duration: int, dt: float) -> np.ndarray:
        idx = np.arange(start, start + duration, dtype=np.float64)
        return np.mod(idx * dt, 1.0)

    def _phase_stateful(self, start: int, dt: np.ndarray) -> np.ndarray:
        if self._last_render_end is None or start != self._last_render_end:
            # Non-contiguous: restart
            self._phase = 0.0
            self._saw_value = -1.0

        increments = np.concatenate(([0.0], np.cumsum(dt[:-1], dtype=np.float64)))
        phase = np.mod(self._phase + increments, 1.0)

        self._phase = float(np.mod(self._phase + float(np.sum(dt)), 1.0))
        self._last_render_end = start + len(dt)
        return phase

    @staticmethod
    def _piecewise_linear_value(phase0: float, a: float) -> float:
        """
        Value of the naive piecewise-linear saw/triangle morph at phase0, with peak at a.
        """
        if phase0 < a:
            return -1.0 + 2.0 * (phase0 / a)
        return 1.0 - 2.0 * ((phase0 - a) / (1.0 - a))

    def _render(self, start: int, duration: int) -> Snippet:
        # Parameter streams
        freq = self._scalar_or_pe_values(self._frequency, start, duration, dtype=np.float64)
        duty = self._scalar_or_pe_values(self._duty_cycle, start, duration, dtype=np.float64)

        # Phase increment per sample (can be negative for negative freq)
        dt = freq / float(self.sample_rate)
        dt_blep = np.clip(np.abs(dt), 1e-12, 0.5)

        # Clamp duty away from endpoints and away from BLEP windows
        # (prevents overlapping correction regions at high frequencies)
        edge = np.maximum(1e-5, 2.0 * dt_blep)
        duty = np.clip(duty, edge, 1.0 - edge)

        # Phase per sample
        if self.is_pure():
            phase = self._phase_pure(start, duration, float(dt[0]))
        else:
            phase = self._phase_stateful(start, dt)

        if self._waveform == self.WAVE_RECTANGLE:
            base = np.where(phase < duty, 1.0, -1.0).astype(np.float64)

            # Discontinuities: +2 at phase=0 wrap, -2 at phase=duty
            r0 = self._blep_residual(phase, dt_blep)
            r1 = self._blep_residual(phase - duty, dt_blep)
            # polyBLEP residual here is normalized for a ±1 step (height=2),
            # so scale by step_height/2.
            y = base + 1.0 * r0 - 1.0 * r1

        else:
            # "sawtooth" mode: piecewise-linear wave with peak at a=1-duty
            a = 1.0 - duty

            # Derivative w.r.t phase (integrate u * dphase to get y)
            u1 = 2.0 / a
            u2 = -2.0 / (1.0 - a)
            u = np.where(phase < a, u1, u2).astype(np.float64)

            # Correct derivative discontinuities with BLEP residuals, then integrate.
            # Step at phase=a: u jumps from u1 -> u2 (delta = u2-u1)
            delta = (u2 - u1).astype(np.float64)
            u_corr = (
                u
                # polyBLEP residual is normalized for a ±1 step (height=2),
                # so scale by step_height/2.
                + (-0.5 * delta) * self._blep_residual(phase, dt_blep)          # wrap at 0
                + (0.5 * delta) * self._blep_residual(phase - a, dt_blep)       # corner at a
            )

            dy = u_corr * dt

            if self.is_pure():
                # Deterministic start value from phase[0]
                a0 = float(a[0])
                y0 = self._piecewise_linear_value(float(phase[0]), a0)
            else:
                # Stateful continuity
                y0 = float(self._saw_value)

            increments = np.concatenate(([0.0], np.cumsum(dy[:-1], dtype=np.float64)))
            y = y0 + increments

            if not self.is_pure():
                self._saw_value = float(y0 + float(np.sum(dy)))

        # Shape and dtype
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
            f"AnalogOscPE(frequency={freq_str}, duty_cycle={duty_str}, "
            f"waveform={self._waveform!r}, channels={self._channels})"
        )

