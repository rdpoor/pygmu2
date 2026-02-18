# src/pygmu2/trigger_restart_pe.py

from __future__ import annotations

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.trigger_signal import TriggerSignal

class TriggerRestartPE(ProcessingElement):
    """
    Trigger-controlled restart/time-remap.

    Semantics:
      - `trigger` is a TriggerSignal (mono, integer-valued).
      - On each sample where trigger > 0 (rising-edge event), we:
          1) call `src.reset_state()`
          2) set local time origin so src renders from t=0 at that instant
      - Between triggers, we keep rendering the same src with increasing local time.
      - Before the first trigger, output is silence.

    Notes:
      - Impure (stateful), so must be rendered contiguously.
      - If trigger has multiplicity (e.g. +2), this treats it as "an event occurred"
        and restarts once at that sample (last-event-wins behavior).
    """

    def __init__(self, trigger: TriggerSignal, src: ProcessingElement):
        self._trigger = trigger
        self._src = src

        # Absolute sample index where the current run started (trigger instant).
        self._t0_abs: int | None = None

    def inputs(self) -> list[ProcessingElement]:
        return [self._trigger, self._src]

    def is_pure(self) -> bool:
        return False

    def channel_count(self) -> int | None:
        return self._src.channel_count()

    def resolve_channel_count(self, input_channel_counts: list[int]) -> int:
        # input_channel_counts: [trigger_cc, src_cc]
        if len(input_channel_counts) != 2:
            raise ValueError("TriggerRestartPE expects exactly two inputs")
        return input_channel_counts[1]

    def _compute_extent(self) -> Extent:
        # Driven by trigger timing; conservative choice is trigger extent.
        return self._trigger.extent()

    def _reset_state(self) -> None:
        self._t0_abs = None

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._reset_state()


    def _render(self, start: int, duration: int) -> Snippet:
        n = duration
        ch = self.channel_count() or 1
        out = np.zeros((n, ch), dtype=np.float32)

        trig = self._trigger.render(start, duration).data[:, 0]
        event_idxs = np.nonzero(trig > 0)[0]  # ascending

        # 1) Prefix: continue previous run up to first event (or whole buffer if no events)
        prefix_end = int(event_idxs[0]) if event_idxs.size else n
        if prefix_end > 0 and self._t0_abs is not None:
            local_start = start - self._t0_abs  # guaranteed >= 0 if state is sane
            out[0:prefix_end, :] = self._src.render(local_start, prefix_end).data

        # 2) For each event, restart and render until next event (or end)
        for i, k in enumerate(event_idxs.tolist()):
            k_end = int(event_idxs[i + 1]) if (i + 1) < event_idxs.size else n
            if k_end <= k:
                continue

            self._src.reset_state()
            self._t0_abs = start + k

            seg_len = k_end - k
            out[k:k_end, :] = self._src.render(0, seg_len).data

        return Snippet(start, out)
