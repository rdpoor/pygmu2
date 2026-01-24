"""
LadderPE - Moog-style ladder filter with multiple response modes.

Ported from DaisySP ladder filter implementation and msynth.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import numpy as np

# Try to import numba for JIT compilation (optional optimization)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator when numba isn't available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# JIT-compiled ladder filter core for per-sample parameter modulation
@jit(nopython=True, cache=True)
def _ladder_process_numba(
    x: np.ndarray,
    freq: np.ndarray,
    resonance: np.ndarray,
    drive: np.ndarray,
    z0: np.ndarray,
    z1: np.ndarray,
    old_input: np.ndarray,
    sample_rate: float,
    passband_gain: float,
    oversample: int,
    mode_index: int,
    state_decay: float,
    input_threshold: float,
    resonance_multiplier: float,
) -> tuple:
    duration = x.shape[0]
    channels = x.shape[1]
    y = np.zeros((duration, channels), dtype=np.float64)

    oversample_recip = 1.0 / oversample
    min_cutoff = 5.0
    nyquist = sample_rate / 2.0
    max_cutoff = nyquist * 0.85
    if max_cutoff > nyquist - 1.0:
        max_cutoff = nyquist - 1.0

    for n in range(duration):
        cutoff = freq[n]
        if cutoff < min_cutoff:
            cutoff = min_cutoff
        if cutoff > max_cutoff:
            cutoff = max_cutoff

        wc = cutoff * (2.0 * np.pi) / (sample_rate * oversample)
        wc2 = wc * wc
        wc3 = wc2 * wc
        wc4 = wc3 * wc

        alpha = 0.9892 * wc - 0.4324 * wc2 + 0.1381 * wc3 - 0.0202 * wc4
        q_adjust = 1.006 + 0.0536 * wc - 0.095 * wc2 - 0.05 * wc4

        res = resonance[n]
        if res < 0.0:
            res = 0.0
        if res > 1.0:
            res = 1.0
        k = 4.0 * res * resonance_multiplier

        drv = drive[n]
        if drv < 0.0:
            drv = 0.0
        if drv > 1.0:
            if drv > 4.0:
                drv = 4.0
            drive_scaled = 1.0 + (drv - 1.0) * (1.0 - passband_gain)
        else:
            drive_scaled = drv

        for c in range(channels):
            input_sample = x[n, c] * drive_scaled
            input_abs = input_sample if input_sample >= 0.0 else -input_sample
            if input_abs < input_threshold:
                for s in range(4):
                    z0[c, s] *= state_decay
                    z1[c, s] *= state_decay
                old_input[c] *= state_decay

            total = 0.0
            interp = 0.0
            for _ in range(oversample):
                in_interp = interp * old_input[c] + (1.0 - interp) * input_sample
                u = np.tanh(in_interp - (z1[c, 3] - passband_gain * in_interp) * k * q_adjust)

                ft = u * 0.76923077 + 0.23076923 * z0[c, 0] - z1[c, 0]
                ft = ft * alpha + z1[c, 0]
                z1[c, 0] = ft
                z0[c, 0] = u
                stage1 = ft

                ft = stage1 * 0.76923077 + 0.23076923 * z0[c, 1] - z1[c, 1]
                ft = ft * alpha + z1[c, 1]
                z1[c, 1] = ft
                z0[c, 1] = stage1
                stage2 = ft

                ft = stage2 * 0.76923077 + 0.23076923 * z0[c, 2] - z1[c, 2]
                ft = ft * alpha + z1[c, 2]
                z1[c, 2] = ft
                z0[c, 2] = stage2
                stage3 = ft

                ft = stage3 * 0.76923077 + 0.23076923 * z0[c, 3] - z1[c, 3]
                ft = ft * alpha + z1[c, 3]
                z1[c, 3] = ft
                z0[c, 3] = stage3
                stage4 = ft

                if mode_index == 0:  # LP24
                    weighted = stage4
                elif mode_index == 1:  # LP12
                    weighted = stage2
                elif mode_index == 2:  # BP24
                    weighted = (stage2 + stage4) * 4.0 - stage3 * 8.0
                elif mode_index == 3:  # BP12
                    weighted = (stage1 - stage2) * 2.0
                elif mode_index == 4:  # HP24
                    weighted = u + stage4 - (stage1 + stage3) * 4.0 + stage2 * 6.0
                else:  # HP12
                    weighted = u + stage2 - stage1 * 2.0

                total += weighted * oversample_recip
                interp += oversample_recip

            old_input[c] = input_sample
            y[n, c] = total

    return y, z0, z1, old_input

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class LadderMode(Enum):
    """Ladder filter response modes."""
    LP24 = "lp24"
    LP12 = "lp12"
    BP24 = "bp24"
    BP12 = "bp12"
    HP24 = "hp24"
    HP12 = "hp12"


class LadderPE(ProcessingElement):
    """
    Moog-style ladder filter with non-linear saturation.

    This implementation uses 4 cascaded one-pole stages with a tanh
    nonlinearity in the feedback path, and supports multiple responses
    (LP/HP/BP at 12dB or 24dB slopes).

    The resonance parameter expects a 0..1 range and is mapped to the
    original ladder resonance range internally.

    Args:
        source: Input audio PE
        frequency: Cutoff frequency in Hz (float or PE)
        resonance: Resonance amount (0..1, float or PE)
        mode: Ladder response mode
        drive: Input drive (float or PE). Values >1 increase saturation.
        passband_gain: Passband gain (0..0.5) used in feedback path
                       (default: 0.5, as in msynth)
        oversample: Oversampling factor (>=1). Higher is more accurate
                    but more CPU intensive (default: 2).
    """

    _DEFAULT_OVERSAMPLE = 2
    _RESONANCE_MULTIPLIER = 1.8
    _MIN_CUTOFF_FREQ = 5.0
    _STATE_DECAY = 0.95
    _INPUT_THRESHOLD = 1e-5

    def __init__(
        self,
        source: ProcessingElement,
        frequency: Union[float, ProcessingElement],
        resonance: Union[float, ProcessingElement] = 0.0,
        mode: LadderMode = LadderMode.LP24,
        drive: Union[float, ProcessingElement] = 1.0,
        passband_gain: float = 0.5,
        oversample: int = _DEFAULT_OVERSAMPLE,
    ):
        self._source = source
        self._frequency = frequency
        self._resonance = resonance
        self._mode = mode
        self._drive = drive
        self._passband_gain = float(np.clip(passband_gain, 0.0, 0.5))
        self._oversample = max(1, int(oversample))
        self._oversample_recip = 1.0 / self._oversample

        self._freq_is_pe = isinstance(frequency, ProcessingElement)
        self._res_is_pe = isinstance(resonance, ProcessingElement)
        self._drive_is_pe = isinstance(drive, ProcessingElement)

        self._z0: Optional[np.ndarray] = None  # shape (channels, 4)
        self._z1: Optional[np.ndarray] = None  # shape (channels, 4)
        self._old_input: Optional[np.ndarray] = None  # shape (channels,)

    @property
    def source(self) -> ProcessingElement:
        """The input audio PE."""
        return self._source

    @property
    def frequency(self) -> Union[float, ProcessingElement]:
        """Cutoff frequency in Hz."""
        return self._frequency

    @property
    def resonance(self) -> Union[float, ProcessingElement]:
        """Resonance amount (0..1)."""
        return self._resonance

    @property
    def mode(self) -> LadderMode:
        """Ladder response mode."""
        return self._mode

    @property
    def drive(self) -> Union[float, ProcessingElement]:
        """Input drive amount."""
        return self._drive

    @property
    def passband_gain(self) -> float:
        """Passband gain (0..0.5)."""
        return self._passband_gain

    @property
    def oversample(self) -> int:
        """Oversampling factor (>=1)."""
        return self._oversample

    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs (source and any parameter PEs)."""
        inputs = [self._source]
        if self._freq_is_pe:
            inputs.append(self._frequency)
        if self._res_is_pe:
            inputs.append(self._resonance)
        if self._drive_is_pe:
            inputs.append(self._drive)
        return inputs

    def is_pure(self) -> bool:
        """LadderPE maintains internal state and is not pure."""
        return False

    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()

    def _compute_extent(self) -> Extent:
        """Intersect source extent with parameter extents."""
        extent = self._source.extent()

        if self._freq_is_pe:
            freq_extent = self._frequency.extent()
            extent = extent.intersection(freq_extent) or extent

        if self._res_is_pe:
            res_extent = self._resonance.extent()
            extent = extent.intersection(res_extent) or extent

        if self._drive_is_pe:
            drive_extent = self._drive.extent()
            extent = extent.intersection(drive_extent) or extent

        return extent

    def on_start(self) -> None:
        """Reset filter state."""
        self._reset_state()

    def on_stop(self) -> None:
        """Clear filter state."""
        self._z0 = None
        self._z1 = None
        self._old_input = None

    def _reset_state(self, channels: Optional[int] = None) -> None:
        """Initialize filter state for current channel count."""
        if channels is None:
            channels = self._source.channel_count() or 1
        self._z0 = np.zeros((channels, 4), dtype=np.float64)
        self._z1 = np.zeros((channels, 4), dtype=np.float64)
        self._old_input = np.zeros((channels,), dtype=np.float64)

    def _ensure_state(self, channels: int) -> None:
        """Ensure state arrays are allocated for current channel count."""
        if (
            self._z0 is None
            or self._z1 is None
            or self._old_input is None
            or self._z0.shape[0] != channels
        ):
            self._reset_state(channels)

    def _compute_coeffs(self, freq: float) -> tuple[float, float]:
        """
        Compute alpha and qAdjust for the given cutoff frequency.
        """
        nyquist = self.sample_rate / 2.0
        max_cutoff = min(nyquist * 0.85, nyquist - 1.0)
        cutoff = float(np.clip(freq, self._MIN_CUTOFF_FREQ, max_cutoff))

        wc = cutoff * (2.0 * np.pi) / (self.sample_rate * self._oversample)
        wc2 = wc * wc
        wc3 = wc2 * wc
        wc4 = wc3 * wc

        alpha = 0.9892 * wc - 0.4324 * wc2 + 0.1381 * wc3 - 0.0202 * wc4
        q_adjust = 1.006 + 0.0536 * wc - 0.095 * wc2 - 0.05 * wc4

        return alpha, q_adjust

    def _drive_scaled(self, drive: float) -> float:
        """Scale input drive based on passband gain."""
        drive = max(drive, 0.0)
        if drive > 1.0:
            drive = min(drive, 4.0)
            return 1.0 + (drive - 1.0) * (1.0 - self._passband_gain)
        return drive

    def _weighted_sum(
        self,
        stage0: float,
        stage1: float,
        stage2: float,
        stage3: float,
        stage4: float,
    ) -> float:
        """Weighted sum to achieve selected response."""
        if self._mode == LadderMode.LP24:
            return stage4
        if self._mode == LadderMode.LP12:
            return stage2
        if self._mode == LadderMode.BP24:
            return (stage2 + stage4) * 4.0 - stage3 * 8.0
        if self._mode == LadderMode.BP12:
            return (stage1 - stage2) * 2.0
        if self._mode == LadderMode.HP24:
            return stage0 + stage4 - (stage1 + stage3) * 4.0 + stage2 * 6.0
        if self._mode == LadderMode.HP12:
            return stage0 + stage2 - stage1 * 2.0
        return 0.0

    def _lpf(self, s: float, channel: int, stage: int, alpha: float) -> float:
        """Single ladder low-pass stage."""
        z0 = self._z0
        z1 = self._z1
        if z0 is None or z1 is None:
            return 0.0
        ft = s * 0.76923077 + 0.23076923 * z0[channel, stage] - z1[channel, stage]
        ft = ft * alpha + z1[channel, stage]
        z1[channel, stage] = ft
        z0[channel, stage] = s
        return ft

    def render(self, start: int, duration: int) -> Snippet:
        """
        Render filtered audio.
        """
        source_snippet = self._source.render(start, duration)
        data = source_snippet.data.astype(np.float64)
        samples, channels = data.shape
        self._ensure_state(channels)

        if self._freq_is_pe:
            freq_values = self._frequency.render(start, duration).data[:, 0].astype(np.float64)
        else:
            freq_values = np.full(samples, float(self._frequency), dtype=np.float64)

        if self._res_is_pe:
            res_values = self._resonance.render(start, duration).data[:, 0].astype(np.float64)
        else:
            res_values = np.full(samples, float(self._resonance), dtype=np.float64)

        if self._drive_is_pe:
            drive_values = self._drive.render(start, duration).data[:, 0].astype(np.float64)
        else:
            drive_values = np.full(samples, float(self._drive), dtype=np.float64)

        if NUMBA_AVAILABLE:
            mode_index = {
                LadderMode.LP24: 0,
                LadderMode.LP12: 1,
                LadderMode.BP24: 2,
                LadderMode.BP12: 3,
                LadderMode.HP24: 4,
                LadderMode.HP12: 5,
            }[self._mode]

            output, self._z0, self._z1, self._old_input = _ladder_process_numba(
                data,
                freq_values,
                res_values,
                drive_values,
                self._z0,
                self._z1,
                self._old_input,
                float(self.sample_rate),
                float(self._passband_gain),
                int(self._oversample),
                int(mode_index),
                float(self._STATE_DECAY),
                float(self._INPUT_THRESHOLD),
                float(self._RESONANCE_MULTIPLIER),
            )
            return Snippet(start, output.astype(np.float32))

        output = np.zeros_like(data, dtype=np.float64)
        old_input = self._old_input
        if old_input is None:
            old_input = np.zeros((channels,), dtype=np.float64)
            self._old_input = old_input

        for i in range(samples):
            alpha, q_adjust = self._compute_coeffs(freq_values[i])
            resonance = float(np.clip(res_values[i], 0.0, 1.0))
            k = 4.0 * resonance * self._RESONANCE_MULTIPLIER
            drive_scaled = self._drive_scaled(float(drive_values[i]))

            for ch in range(channels):
                input_sample = data[i, ch] * drive_scaled

                if abs(input_sample) < self._INPUT_THRESHOLD:
                    if self._z0 is not None and self._z1 is not None:
                        self._z0[ch, :] *= self._STATE_DECAY
                        self._z1[ch, :] *= self._STATE_DECAY
                    old_input[ch] *= self._STATE_DECAY

                total = 0.0
                interp = 0.0
                for _ in range(self._oversample):
                    in_interp = interp * old_input[ch] + (1.0 - interp) * input_sample
                    u = np.tanh(in_interp - (self._z1[ch, 3] - self._passband_gain * in_interp) * k * q_adjust)
                    stage1 = self._lpf(u, ch, 0, alpha)
                    stage2 = self._lpf(stage1, ch, 1, alpha)
                    stage3 = self._lpf(stage2, ch, 2, alpha)
                    stage4 = self._lpf(stage3, ch, 3, alpha)
                    total += (
                        self._weighted_sum(u, stage1, stage2, stage3, stage4)
                        * self._oversample_recip
                    )
                    interp += self._oversample_recip

                old_input[ch] = input_sample
                output[i, ch] = total

        return Snippet(start, output.astype(np.float32))

    def __repr__(self) -> str:
        freq_repr = self._frequency if not self._freq_is_pe else f"{self._frequency.__class__.__name__}(...)"
        res_repr = self._resonance if not self._res_is_pe else f"{self._resonance.__class__.__name__}(...)"
        drive_repr = self._drive if not self._drive_is_pe else f"{self._drive.__class__.__name__}(...)"
        return (
            f"LadderPE(source={self._source.__class__.__name__}, "
            f"frequency={freq_repr}, resonance={res_repr}, "
            f"mode={self._mode.value}, drive={drive_repr}, "
            f"oversample={self._oversample})"
        )
