"""
GateSignal - semantic gate signal base class for pygmu2.

Contract: a gate is a SUSTAINED LEVEL. It is mono, its values are exactly
0.0 or 1.0, transitions are meaningful (a rising edge is an onset, a
falling edge a release), and its duration is meaningful (how long the
gate is held). Contrast TriggerSignal, whose non-zero samples are
isolated instantaneous events.

Gates are the primitive of the family: a trigger is derived from a gate
with GateToTriggerPE (rising edges -> one-sample events).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .semantic_signal import SemanticSignal
from .snippet import Snippet


class GateSignal(SemanticSignal, ABC):
    """
    A semantic ProcessingElement whose rendered output is a *gate*:
      - mono (N, 1)
      - values exactly 0.0 or 1.0
      - a sustained level: how LONG it is high is meaningful

    Subclasses implement _render_gate().
    """

    @abstractmethod
    def _render_gate(self, start: int, duration: int) -> Snippet:
        """Subclasses must render a mono (N,1) snippet with values 0/1."""
        raise NotImplementedError

    def _render(self, start: int, duration: int) -> Snippet:
        snip = self._render_gate(start, duration)
        if self.VALIDATE:
            self._validate_gate_array(snip.data)
        return snip

    @classmethod
    def _validate_gate_array(cls, arr: np.ndarray) -> None:
        cls._check_shape(arr, "GateSignal")
        probe = cls._probe(arr)
        ok = np.logical_or(probe == 0.0, probe == 1.0)
        if not np.all(ok):
            bad = probe[~ok]
            mn = float(np.min(bad)) if bad.size else float("nan")
            mx = float(np.max(bad)) if bad.size else float("nan")
            raise ValueError(
                "GateSignal values must be exactly 0 or 1 "
                f"(found out-of-domain values in probe; min={mn}, max={mx}). "
                "If you meant to threshold a control/audio signal, wrap it with SignalToGatePE."
            )
