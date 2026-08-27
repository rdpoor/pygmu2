"""
TriggerSignal - semantic trigger signal base class for pygmu2.

Contract: a trigger is a stream of ISOLATED ONE-SAMPLE EVENTS. It is
mono, its values are exactly 0.0 (no event) or 1.0 (event), and a run of
two or more consecutive non-zero samples is a contract violation — that
would be a gate. Contrast GateSignal, whose value is a sustained level.

(Earlier revisions documented negative values as falling edges and
magnitudes as event multiplicity; no producer ever emitted them and every
consumer tested `> 0`, so those clauses are deleted rather than kept as
unenforced prose.)

Triggers are DERIVED signals: the canonical way to make one is
GateToTriggerPE over a gate generator.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .semantic_signal import SemanticSignal
from .snippet import Snippet


class TriggerSignal(SemanticSignal, ABC):
    """
    A semantic ProcessingElement whose rendered output is a *trigger/event
    stream*:
      - mono (N, 1)
      - values exactly 0.0 or 1.0
      - non-zero samples are isolated: consecutive events must be separated
        by at least one zero sample (checked when PYGMU_VALIDATE_SIGNALS_FULL
        is enabled; the default probe checks the value domain only)

    Subclasses implement _render_trigger().
    """

    @abstractmethod
    def _render_trigger(self, start: int, duration: int) -> Snippet:
        """Subclasses must render a mono (N,1) snippet of isolated 0/1 events."""
        raise NotImplementedError

    def _render(self, start: int, duration: int) -> Snippet:
        snip = self._render_trigger(start, duration)
        if self.VALIDATE:
            self._validate_trigger_array(snip.data)
        return snip

    @classmethod
    def _validate_trigger_array(cls, arr: np.ndarray) -> None:
        cls._check_shape(arr, "TriggerSignal")
        probe = cls._probe(arr)
        ok = np.logical_or(probe == 0.0, probe == 1.0)
        if not np.all(ok):
            bad = probe[~ok]
            mn = float(np.min(bad)) if bad.size else float("nan")
            mx = float(np.max(bad)) if bad.size else float("nan")
            raise ValueError(
                "TriggerSignal values must be exactly 0 or 1 "
                f"(found out-of-domain values in probe; min={mn}, max={mx})."
            )
        if cls.VALIDATE_FULL:
            vals = arr[:, 0]
            if np.any((vals[1:] > 0) & (vals[:-1] > 0)):
                raise ValueError(
                    "TriggerSignal events must be isolated one-sample pulses; "
                    "found consecutive non-zero samples (that is a gate — "
                    "use a GateSignal, or derive events with GateToTriggerPE)."
                )
