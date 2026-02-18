# src/pygmu2/gate_signal.py
#
# Semantic gate signal base class for pygmu2.
#
# GateSignal:  mono signal with values exactly {0, 1}
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


class GateSignal(ProcessingElement, ABC):
    """
    A semantic ProcessingElement whose rendered output is a *gate*:
      - mono (N, 1)
      - values exactly 0.0 or 1.0

    Subclasses implement _render_gate().
    GateSignal implements _render() (used by ProcessingElement.render()) so the
    rest of the framework behavior remains intact.
    """

    # Validation knobs
    VALIDATE: bool = _env_flag("PYGMU_VALIDATE_SIGNALS", "1")
    # If True, check all samples. If False, do a quick probe.
    VALIDATE_FULL: bool = _env_flag("PYGMU_VALIDATE_SIGNALS_FULL", "0")
    # How many samples to probe when not validating full buffers.
    VALIDATE_PROBE_SAMPLES: Final[int] = 64

    def channel_count(self) -> int:
        return 1

    @abstractmethod
    def _render_gate(self, start: int, duration: int) -> Snippet:
        """Subclasses must render a mono (N,1) snippet with values 0/1."""
        raise NotImplementedError

    def _render(self, start: int, duration: int) -> Snippet:
        snip = self._render_gate(start, duration)
        if self.VALIDATE:
            self._validate_gate_snippet(snip)
        return snip

    @classmethod
    def _validate_gate_snippet(cls, snip: Snippet) -> None:
        arr = snip.data
        cls._validate_gate_array(arr)

    @classmethod
    def _validate_gate_array(cls, arr: np.ndarray) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"GateSignal must render a numpy array, got {type(arr)}")

        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"GateSignal must be mono with shape (N,1); got {arr.shape}")

        if arr.dtype.kind not in ("f", "i", "u"):
            raise TypeError(f"GateSignal must render numeric dtype; got {arr.dtype}")

        # Probe indices (cheap) or validate full buffer (strict)
        if cls.VALIDATE_FULL or arr.shape[0] <= cls.VALIDATE_PROBE_SAMPLES:
            probe = arr[:, 0]
        else:
            n = arr.shape[0]
            k = min(cls.VALIDATE_PROBE_SAMPLES, n)
            # evenly spaced probe points, includes first/last
            idx = np.linspace(0, n - 1, num=k, dtype=int)
            probe = arr[idx, 0]

        ok = np.logical_or(probe == 0.0, probe == 1.0)
        if not np.all(ok):
            # Provide a compact diagnostic without dumping huge arrays
            bad = probe[~ok]
            mn = float(np.min(bad)) if bad.size else float("nan")
            mx = float(np.max(bad)) if bad.size else float("nan")
            raise ValueError(
                "GateSignal values must be exactly 0 or 1 "
                f"(found out-of-domain values in probe; min={mn}, max={mx}). "
                "If you meant to threshold a control/audio signal, wrap it with ToGateSignal."
            )
