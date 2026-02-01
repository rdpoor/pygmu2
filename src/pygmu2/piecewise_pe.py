"""
PiecewisePE - piecewise (sample_index, value) curve with configurable transitions.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from pygmu2.extent import Extent, ExtendMode
from pygmu2.processing_element import SourcePE
from pygmu2.snippet import Snippet


class TransitionType(Enum):
    """How to transition between points."""

    STEP = "step"  # Hold segment start value until next point
    LINEAR = "linear"  # Linear interpolation
    EXPONENTIAL = "exponential"  # Exponential curve
    SIGMOID = "sigmoid"  # S-curve
    CONSTANT_POWER = "constant_power"  # sin(π/2·t) or 1-cos(π/2·t); crossfade pairs sum to constant power


def _parse_points(
    points: Sequence[Tuple[int, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse and sort (sample_index, value) pairs; return (times, values) arrays."""
    if not points:
        raise ValueError("PiecewisePE requires at least one point")
    arr = np.array(points, dtype=np.float64)
    times = arr[:, 0].astype(np.int64)
    values = arr[:, 1].astype(np.float64)
    # Sort by time
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    return times, values


def _segment_curve(
    t: np.ndarray,
    v0: float,
    v1: float,
    mode: TransitionType,
) -> np.ndarray:
    """Map t in [0, 1] to value between v0 and v1."""
    if mode == TransitionType.STEP:
        return np.full_like(t, v0)
    if mode == TransitionType.LINEAR:
        return v0 + (v1 - v0) * t
    if mode == TransitionType.EXPONENTIAL:
        if v0 <= 0 or v1 <= 0 or (v0 > 0) != (v1 > 0):
            return v0 + (v1 - v0) * t
        ratio = v1 / v0
        return v0 * (ratio ** t)
    if mode == TransitionType.SIGMOID:
        k = 6.0
        x = k * (2.0 * t - 1.0)
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
        return v0 + (v1 - v0) * sig
    if mode == TransitionType.CONSTANT_POWER:
        # sin(π/2·t) for fade-in; 1-cos(π/2·t) for fade-out. Pairs give sin²+cos²=1.
        if v1 >= v0:
            curve = np.sin(0.5 * np.pi * t)
        else:
            curve = 1.0 - np.cos(0.5 * np.pi * t)
        return v0 + (v1 - v0) * curve
    return v0 + (v1 - v0) * t


class PiecewisePE(SourcePE):
    """
    A SourcePE that outputs a piecewise curve defined by (sample_index, value) points.

    `points` determine where transitions occur.     
    `transition_type` controls how values change between points (step, linear, 
    exponential, sigmoid, or constant_power). 
    `extend_mode` controls behavior before the first point and after the last point.

    A ramp from 0 to 1 over 100 samples is
    PiecewisePE([(0, 0.0), (100, 1.0)], transition_type=TransitionType.LINEAR).

    Useful for:
    - Envelope automation (multi-stage curves)
    - Parameter sweeps with multiple segments
    - Step sequences (STEP), smooth curves (LINEAR, SIGMOID), or crossfades (CONSTANT_POWER)
    - Testing and control signals

    Args:
        points: Sequence of (sample_index, value) pairs. Sorted by sample_index internally.
                At least one point required. Duplicate times: later value wins.
        transition_type: How to interpolate between points (default: LINEAR).
        extend_mode: Behavior before first / after last point (default: ZERO).
        channels: Number of output channels (default: 1).

    Example:
        # Linear ramp from 0 to 1 over 100 samples, then hold
        pw = PiecewisePE([(0, 0.0), (100, 1.0)], transition_type=TransitionType.LINEAR)

        # Step envelope
        pw = PiecewisePE([(0, 0), (100, 1), (200, 0)], transition_type=TransitionType.STEP)

        # Multi-segment with hold after last point
        pw = PiecewisePE(
            [(0, 0), (100, 0.5), (200, 1.0)],
            transition_type=TransitionType.SIGMOID,
            extend_mode=ExtendMode.HOLD_LAST,
        )
    """

    def __init__(
        self,
        points: Sequence[Tuple[int, float]],
        transition_type: Union[TransitionType, str] = TransitionType.LINEAR,
        extend_mode: ExtendMode = ExtendMode.ZERO,
        channels: int = 1,
    ):
        self._times, self._values = _parse_points(points)
        self._n = len(self._times)
        if isinstance(transition_type, str):
            try:
                transition_type = TransitionType(transition_type.lower())
            except ValueError:
                transition_type = TransitionType.LINEAR
        self._transition_type = transition_type
        self._extend_mode = extend_mode
        self._channels = int(channels)
        if self._channels < 1:
            raise ValueError(f"channels must be >= 1, got {self._channels}")

    @property
    def points(self) -> List[Tuple[int, float]]:
        """Points as (sample_index, value) list."""
        return list(zip(self._times.tolist(), self._values.tolist()))

    @property
    def transition_type(self) -> TransitionType:
        return self._transition_type

    @property
    def extend_mode(self) -> ExtendMode:
        return self._extend_mode

    def _compute_extent(self) -> Extent:
        if self._extend_mode != ExtendMode.ZERO:
            return Extent(None, None)
        t0 = int(self._times[0])
        t_last = int(self._times[-1])
        # Single point: one sample of output at t0
        if self._n == 1:
            return Extent(t0, t0 + 1)
        return Extent(t0, t_last)

    def channel_count(self) -> int:
        return self._channels

    def _render(self, start: int, duration: int) -> Snippet:
        data = np.zeros((duration, self._channels), dtype=np.float32)
        t0 = int(self._times[0])
        t_last = int(self._times[-1])
        req_end = start + duration

        # Before first point
        if req_end <= t0:
            if self._extend_mode in (ExtendMode.HOLD_FIRST, ExtendMode.HOLD_BOTH):
                data[:] = self._values[0]
            return Snippet(start, data)
        if start < t0:
            n_before = t0 - start
            if self._extend_mode in (ExtendMode.HOLD_FIRST, ExtendMode.HOLD_BOTH):
                data[:n_before] = self._values[0]
            # rest filled below or by segment logic
        # After last point
        if start >= t_last:
            if self._extend_mode in (ExtendMode.HOLD_LAST, ExtendMode.HOLD_BOTH):
                data[:] = self._values[-1]
            return Snippet(start, data)
        if req_end > t_last:
            n_after = req_end - t_last
            after_start = duration - n_after
            if self._extend_mode in (ExtendMode.HOLD_LAST, ExtendMode.HOLD_BOTH):
                data[after_start:] = self._values[-1]

        # Single point: one sample at t0
        if self._n == 1:
            lo = t0 - start
            if 0 <= lo < duration:
                data[lo, :] = self._values[0]
            return Snippet(start, data)

        # Segments: for each output index s = start + i, find segment and compute value
        for i in range(duration):
            s = start + i
            if s < t0 or s >= t_last:
                continue
            # Find segment j: times[j] <= s < times[j+1]
            j = np.searchsorted(self._times, s, side="right") - 1
            if j < 0:
                j = 0
            if j >= self._n - 1:
                data[i, :] = self._values[-1]
                continue
            seg_start = int(self._times[j])
            seg_end = int(self._times[j + 1])
            v0 = float(self._values[j])
            v1 = float(self._values[j + 1])
            if seg_end <= seg_start:
                data[i, :] = v0
                continue
            t = (s - seg_start) / (seg_end - seg_start)
            t_arr = np.array([t], dtype=np.float64)
            val = _segment_curve(t_arr, v0, v1, self._transition_type)[0]
            data[i, :] = val

        return Snippet(start, data)

    def __repr__(self) -> str:
        return (
            f"PiecewisePE(points={self.points!r}, transition_type={self._transition_type.value}, "
            f"extend_mode={self._extend_mode.value}, channels={self._channels})"
        )
