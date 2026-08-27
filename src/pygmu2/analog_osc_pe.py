"""
AnalogOscPE - analog-style oscillator (PWM rectangle + morphing saw/triangle),
bandlimited or deliberately naive.

Waveforms:
- "rectangle": a pulse wave with duty-cycle (PWM) control
- "sawtooth": a duty-controlled morph where:
    duty=0.0   -> ascending ramp (saw up)
    duty=0.5   -> triangle
    duty=1.0   -> descending ramp (saw down)

The `antialias` flag selects the rendering math (this class absorbs the
former FunctionGenPE, which was the naive variant):

- antialias=True (default): polyBLEP-corrected, alias-free audio waveforms.
  duty_cycle is clamped away from 0/1 (and away from the polyBLEP window),
  so the exact saw endpoints are unreachable in this mode.
- antialias=False: exact naive waveforms with hard edges — aliased at audio
  rates, but exactly what LFOs, gates, and "raw DSP" experiments need.
  duty=0 and duty=1 produce exact rising/falling saws.

Notes:
- No explicit "bandwidth" parameter; patch a filter PE (e.g. LadderPE) for
  classic subtractive tone shaping.
- `phase` offsets the oscillator in cycles. Per-sample (PE-valued) phase
  modulation is exact with antialias=False; with antialias=True the BLEP
  correction assumes the offset changes slowly relative to the pitch.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations


import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class AnalogOscPE(ProcessingElement):
    """
    Analog-style oscillator: PWM rectangle + duty-controlled saw/triangle
    morph, bandlimited (polyBLEP) or naive per the `antialias` flag.

    Args:
        frequency: Frequency in Hz, or PE providing per-sample values.
        duty_cycle: Duty cycle in [0, 1], or PE providing per-sample values.
        phase: Phase offset in cycles [0, 1), or PE providing per-sample
               values (default 0.0).
        waveform: "rectangle" or "sawtooth"
        antialias: True (default) for polyBLEP bandlimiting; False for the
               exact naive waveform (the former FunctionGenPE).
        channels: Number of output channels (default: 1)
    """

    WAVE_RECTANGLE = "rectangle"
    WAVE_SAWTOOTH = "sawtooth"

    def __init__(
        self,
        frequency: float | ProcessingElement = 440.0,
        duty_cycle: float | ProcessingElement = 0.5,
        phase: float | ProcessingElement = 0.0,
        waveform: str = "rectangle",
        antialias: bool = True,
        channels: int = 1,
    ):
        self._frequency = frequency
        self._duty_cycle = duty_cycle
        self._phase_in = phase
        self._waveform = str(waveform).lower()
        self._antialias = bool(antialias)
        self._channels = int(channels)

        if self._waveform not in (self.WAVE_RECTANGLE, self.WAVE_SAWTOOTH):
            raise ValueError(
                f"waveform must be 'rectangle' or 'sawtooth', got {waveform!r}"
            )
        if self._channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        # Stateful path: phase + (for the bandlimited saw morph) current value
        self._phase: float = 0.0  # [0,1)
        self._saw_value: float = -1.0
        self._last_render_end: int | None = None

    @property
    def frequency(self) -> float | ProcessingElement:
        return self._frequency

    @property
    def duty_cycle(self) -> float | ProcessingElement:
        return self._duty_cycle

    @property
    def phase(self) -> float | ProcessingElement:
        return self._phase_in

    @property
    def waveform(self) -> str:
        return self._waveform

    @property
    def antialias(self) -> bool:
        return self._antialias

    def inputs(self) -> list[ProcessingElement]:
        result: list[ProcessingElement] = []
        if isinstance(self._frequency, ProcessingElement):
            result.append(self._frequency)
        if isinstance(self._duty_cycle, ProcessingElement):
            result.append(self._duty_cycle)
        if isinstance(self._phase_in, ProcessingElement):
            result.append(self._phase_in)
        return result

    @property
    def stateful(self) -> bool:  # type: ignore[override]
        # PE-driven parameters need a phase accumulator; constants are closed-form.
        return bool(self.inputs())

    def channel_count(self) -> int:
        return self._channels

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
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

    # ------------------------------------------------------------------
    # polyBLEP machinery (antialias=True)
    # ------------------------------------------------------------------

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
            y[m] += u[m] ** 4

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

    # ------------------------------------------------------------------
    # Naive waveform math (antialias=False; the former FunctionGenPE)
    # ------------------------------------------------------------------

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
            y_mid[~rise] = 1.0 - 2.0 * (
                (p[~rise] - a_mid[~rise]) / (1.0 - a_mid[~rise])
            )
            y[m_mid] = y_mid

        return y

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _base_phase(self, start: int, duration: int, dt: np.ndarray) -> np.ndarray:
        """Un-offset oscillator phase, closed-form or accumulated."""
        if not self.stateful:
            idx = np.arange(start, start + duration, dtype=np.float64)
            return np.mod(idx * float(dt[0]), 1.0)

        if self._last_render_end is None:
            # First render after start/reset. Non-contiguous renders never
            # reach here (the base class raises; stateful when PE-driven).
            self._phase = 0.0
            self._saw_value = -1.0

        increments = np.concatenate(([0.0], np.cumsum(dt[:-1], dtype=np.float64)))
        phase = np.mod(self._phase + increments, 1.0)

        self._phase = float(np.mod(self._phase + float(np.sum(dt)), 1.0))
        self._last_render_end = start + duration
        return phase

    def _render(self, start: int, duration: int) -> Snippet:
        # Parameter streams
        freq = self._scalar_or_pe_values(
            self._frequency, start, duration, dtype=np.float64
        )
        duty = self._scalar_or_pe_values(
            self._duty_cycle, start, duration, dtype=np.float64
        )
        ph_in = self._scalar_or_pe_values(
            self._phase_in, start, duration, dtype=np.float64
        )

        # Phase increment per sample (can be negative for negative freq)
        dt = freq / float(self.sample_rate)

        # Final phase: offset applied BEFORE any BLEP residual computation,
        # so corrections align with the actual output discontinuities.
        phase = np.mod(self._base_phase(start, duration, dt) + ph_in, 1.0)

        if not self._antialias:
            duty = np.clip(duty, 0.0, 1.0)
            if self._waveform == self.WAVE_RECTANGLE:
                y = np.where(phase < duty, 1.0, -1.0).astype(np.float64)
            else:
                y = self._piecewise_linear(phase, duty)
            return self._shape(start, y)

        # --- antialias=True: polyBLEP path ---
        dt_blep = np.clip(np.abs(dt), 1e-12, 0.5)

        # Clamp duty away from endpoints and away from BLEP windows
        # (prevents overlapping correction regions at high frequencies).
        # Consequence: exact saw endpoints need antialias=False.
        edge = np.maximum(1e-5, 2.0 * dt_blep)
        duty = np.clip(duty, edge, 1.0 - edge)

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
                + (-0.5 * delta) * self._blep_residual(phase, dt_blep)  # wrap at 0
                + (0.5 * delta) * self._blep_residual(phase - a, dt_blep)  # corner at a
            )

            dy = u_corr * dt

            if not self.stateful:
                # Deterministic start value from phase[0]
                y0 = float(self._piecewise_linear(phase[0:1], duty[0:1])[0])
            else:
                # Stateful continuity
                y0 = float(self._saw_value)

            increments = np.concatenate(([0.0], np.cumsum(dy[:-1], dtype=np.float64)))
            y = y0 + increments

            if self.stateful:
                self._saw_value = float(y0 + float(np.sum(dy)))

        return self._shape(start, y)

    def _shape(self, start: int, y: np.ndarray) -> Snippet:
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
        anti_str = "" if self._antialias else ", antialias=False"
        return (
            f"AnalogOscPE(frequency={freq_str}, duty_cycle={duty_str}, "
            f"waveform={self._waveform!r}{anti_str}, channels={self._channels})"
        )
