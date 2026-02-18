# src/pygmu2/trigger_signal.py
#
# Semantic trigger signal base class for pygmu2.
#
# TriggerSignal: mono signal with values in {..., -2, -1, 0, +1, +2, ...}
#   where sign encodes edge direction (+ rising, - falling) and magnitude
#   encodes multiplicity (optional but useful).
#
# This is a *ProcessingElement* subclass, so it participates in the same
# purity/contiguity/diagnostics machinery.

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Final

import numpy as np

from .processing_element import ProcessingElement
from .snippet import Snippet
from .extent import Extent


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


class TriggerSignal(ProcessingElement, ABC):
    """
    A semantic ProcessingElement whose rendered output is a *trigger/event stream*:
      - mono (N, 1)
      - integer-valued samples
      - sign encodes edge direction: + rising, - falling
      - magnitude encodes multiplicity (e.g., +2 means two rising events at that sample)

    Subclasses implement _render_trigger().
    TriggerSignal implements _render() so ProcessingElement.render() stays in control.
    """

    # Validation knobs
    VALIDATE: bool = _env_flag("PYGMU_VALIDATE_SIGNALS", "1")
    VALIDATE_FULL: bool = _env_flag("PYGMU_VALIDATE_SIGNALS_FULL", "0")
    VALIDATE_PROBE_SAMPLES: Final[int] = 64

    # If False, restrict values to {-1, 0, +1}. If True, allow any integers.
    ALLOW_MULTIPLE_EVENTS: bool = _env_flag("PYGMU_TRIGGER_ALLOW_MULTIPLE", "1")

    def channel_count(self) -> int:
        return 1

    @abstractmethod
    def _render_trigger(self, start: int, duration: int) -> Snippet:
        """Subclasses must render a mono (N,1) snippet with integer-valued samples."""
        raise NotImplementedError

    def _render(self, start: int, duration: int) -> Snippet:
        snip = self._render_trigger(start, duration)
        if self.VALIDATE:
            self._validate_trigger_snippet(snip)
        return snip

    @classmethod
    def _validate_trigger_snippet(cls, snip: Snippet) -> None:
        arr = snip.data
        cls._validate_trigger_array(arr)

    @classmethod
    def _validate_trigger_array(cls, arr: np.ndarray) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"TriggerSignal must render a numpy array, got {type(arr)}")

        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"TriggerSignal must be mono with shape (N,1); got {arr.shape}")

        if arr.dtype.kind not in ("f", "i", "u"):
            raise TypeError(f"TriggerSignal must render numeric dtype; got {arr.dtype}")

        # Probe indices (cheap) or validate full buffer (strict)
        if cls.VALIDATE_FULL or arr.shape[0] <= cls.VALIDATE_PROBE_SAMPLES:
            probe = arr[:, 0]
        else:
            n = arr.shape[0]
            k = min(cls.VALIDATE_PROBE_SAMPLES, n)
            idx = np.linspace(0, n - 1, num=k, dtype=int)
            probe = arr[idx, 0]

        # Accept integer dtype directly; for float, require exact integer values
        if probe.dtype.kind in ("i", "u"):
            vals = probe
        else:
            # "exact integer" check (not tolerance-based by design)
            if not np.all(np.equal(probe, np.round(probe))):
                bad = probe[np.not_equal(probe, np.round(probe))]
                mn = float(np.min(bad)) if bad.size else float("nan")
                mx = float(np.max(bad)) if bad.size else float("nan")
                raise ValueError(
                    "TriggerSignal values must be integers "
                    f"(found non-integers in probe; min={mn}, max={mx})."
                )
            vals = probe.astype(np.int64)

        if cls.ALLOW_MULTIPLE_EVENTS:
            # Any integers OK (including 0). You may optionally disallow magnitude 0? no.
            return

        # Restrict to {-1, 0, +1}
        ok = np.logical_or.reduce((vals == -1, vals == 0, vals == 1))
        if not np.all(ok):
            bad = vals[~ok]
            mn = int(np.min(bad)) if bad.size else 0
            mx = int(np.max(bad)) if bad.size else 0
            raise ValueError(
                "TriggerSignal values must be in {-1, 0, +1} "
                f"(found out-of-domain values in probe; min={mn}, max={mx}). "
                "Set PYGMU_TRIGGER_ALLOW_MULTIPLE=1 to allow multiplicity."
            )
