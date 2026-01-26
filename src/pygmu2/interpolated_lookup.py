"""
Shared interpolated lookup helpers.

This module centralizes the "render a source window, then interpolate at
fractional indices" kernel that is used by multiple PEs (e.g., DelayPE and
WavetablePE).
"""

from __future__ import annotations

from typing import Optional, Any

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet


def _interp_mode_value(interpolation: Any) -> str:
    """
    Normalize interpolation mode to a string.

    Accepts either an Enum (with .value) or a string.
    """
    v = getattr(interpolation, "value", interpolation)
    return str(v).lower()


def _linear_interp(
    indices: np.ndarray,
    source_data: np.ndarray,
    source_data_start: int,
) -> np.ndarray:
    """Perform linear interpolation."""
    idx_floor = np.floor(indices).astype(np.int64)
    frac = (indices - idx_floor).reshape(-1, 1)

    local_floor = idx_floor - source_data_start
    local_ceil = local_floor + 1

    local_floor_clipped = np.clip(local_floor, 0, len(source_data) - 1)
    local_ceil_clipped = np.clip(local_ceil, 0, len(source_data) - 1)

    val_floor = source_data[local_floor_clipped]
    val_ceil = source_data[local_ceil_clipped]

    return (1.0 - frac) * val_floor + frac * val_ceil


def _cubic_interp(
    indices: np.ndarray,
    source_data: np.ndarray,
    source_data_start: int,
) -> np.ndarray:
    """Perform cubic (Catmull-Rom) interpolation."""
    idx_floor = np.floor(indices).astype(np.int64)
    t = (indices - idx_floor).reshape(-1, 1)

    local_p1 = idx_floor - source_data_start
    local_p0 = local_p1 - 1
    local_p2 = local_p1 + 1
    local_p3 = local_p1 + 2

    max_idx = len(source_data) - 1
    local_p0_clipped = np.clip(local_p0, 0, max_idx)
    local_p1_clipped = np.clip(local_p1, 0, max_idx)
    local_p2_clipped = np.clip(local_p2, 0, max_idx)
    local_p3_clipped = np.clip(local_p3, 0, max_idx)

    p0 = source_data[local_p0_clipped]
    p1 = source_data[local_p1_clipped]
    p2 = source_data[local_p2_clipped]
    p3 = source_data[local_p3_clipped]

    t2 = t * t
    t3 = t2 * t

    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def interpolated_lookup(
    source: ProcessingElement,
    out_start: int,
    indices: np.ndarray,
    interpolation: Any,
    *,
    out_of_bounds_mask: Optional[np.ndarray] = None,
    out_dtype: Any = np.float32,
) -> Snippet:
    """
    Render `source` by interpolated lookup at fractional `indices`.

    Args:
        source: source PE to sample from
        out_start: start sample index for the returned snippet
        indices: 1D array of fractional source indices to sample (len == duration)
        interpolation: interpolation mode (Enum with .value or string: "linear"/"cubic")
        out_of_bounds_mask: optional boolean mask (len == duration) indicating
            indices that should be forced to 0.0 in the output.
        out_dtype: output dtype (default: float32)

    Returns:
        Snippet: shape (duration, channels)
    """
    # Ensure 1D float indices
    indices = np.asarray(indices, dtype=np.float64).reshape(-1)
    duration = len(indices)
    if duration <= 0:
        # Determine channels conservatively
        ch = source.channel_count() or 1
        return Snippet.from_zeros(out_start, 0, ch)

    mode = _interp_mode_value(interpolation)
    cubic = mode == "cubic"
    margin = 2 if cubic else 1

    idx_min = float(np.min(indices))
    idx_max = float(np.max(indices))
    needed_min = int(np.floor(idx_min)) - (margin - 1)
    needed_max = int(np.ceil(idx_max)) + margin
    needed_duration = needed_max - needed_min

    source_snippet = source.render(needed_min, needed_duration)
    source_data = source_snippet.data

    if cubic:
        result = _cubic_interp(indices, source_data, needed_min)
    else:
        result = _linear_interp(indices, source_data, needed_min)

    if out_of_bounds_mask is not None and np.any(out_of_bounds_mask):
        result = result.copy()
        result[out_of_bounds_mask] = 0.0

    return Snippet(out_start, result.astype(out_dtype, copy=False))

