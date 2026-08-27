"""
Shared scaffolding for semantic control signals (GateSignal, TriggerSignal).

Both signal families are mono ProcessingElements with a validated value
domain; this module holds the one copy of the env-flag plumbing and the
probe/validate machinery that was previously duplicated in each.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from abc import ABC
import os
from typing import Final

import numpy as np

from .processing_element import ProcessingElement


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


class SemanticSignal(ProcessingElement, ABC):
    """
    Base for mono control signals with a validated value domain.

    Validation knobs (shared by all signal classes):
        PYGMU_VALIDATE_SIGNALS       enable validation (default on)
        PYGMU_VALIDATE_SIGNALS_FULL  check every sample instead of probing
    """

    VALIDATE: bool = _env_flag("PYGMU_VALIDATE_SIGNALS", "1")
    VALIDATE_FULL: bool = _env_flag("PYGMU_VALIDATE_SIGNALS_FULL", "0")
    VALIDATE_PROBE_SAMPLES: Final[int] = 64

    def channel_count(self) -> int:
        return 1

    @classmethod
    def _check_shape(cls, arr: np.ndarray, kind: str) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{kind} must render a numpy array, got {type(arr)}")
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"{kind} must be mono with shape (N,1); got {arr.shape}")
        if arr.dtype.kind not in ("f", "i", "u"):
            raise TypeError(f"{kind} must render numeric dtype; got {arr.dtype}")

    @classmethod
    def _probe(cls, arr: np.ndarray) -> np.ndarray:
        """Evenly-spaced probe of the buffer (or all of it when small or
        VALIDATE_FULL)."""
        if cls.VALIDATE_FULL or arr.shape[0] <= cls.VALIDATE_PROBE_SAMPLES:
            return arr[:, 0]
        n = arr.shape[0]
        k = min(cls.VALIDATE_PROBE_SAMPLES, n)
        idx = np.linspace(0, n - 1, num=k, dtype=int)
        return arr[idx, 0]
