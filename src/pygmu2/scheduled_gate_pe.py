"""
ScheduledGatePE - gate signal from a list of (start, duration) notes.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.extent import Extent
from pygmu2.gate_signal import GateSignal
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet


class ScheduledGatePE(GateSignal):
    """
    Convert note (start, duration) pairs into a gate signal: 1.0 while a
    note sounds, 0.0 between notes. Typically feeds AdsrGatedPE, or
    GateToTriggerPE when per-note onset events are needed.

    Args:
        notes: list of (start, duration) pairs, in samples.
        merge_overlaps: If True (default), overlapping/abutting notes fuse
            into one sustained gate span — legato. If False, each new note
            that begins while the gate is already high punches a one-sample
            0 immediately before its onset, so every note produces its own
            rising edge and GateToTriggerPE sees one event per note
            (otherwise legato passages silently lose onsets).
    """

    def __init__(
        self,
        notes: list[tuple[int, int]],  # start, duration pairs
        merge_overlaps: bool = True,
    ):
        # Merge overlapping note intervals once at construction time.
        notes_sorted = sorted(notes)
        self._merge_overlaps = bool(merge_overlaps)
        self._merged = []
        if notes_sorted:
            a, b = notes_sorted[0][0], notes_sorted[0][0] + notes_sorted[0][1]
            for note_start, note_dur in notes_sorted[1:]:
                note_end = note_start + note_dur
                if note_start < b:  # overlap: extend the current interval
                    b = max(b, note_end)
                else:  # gap: close current interval, start a new one
                    self._merged.append((a, b))
                    a, b = note_start, note_end
            self._merged.append((a, b))

        # Retrigger notches: one-sample gaps before onsets that fall
        # strictly inside a merged span (merge_overlaps=False only).
        self._notches: list[int] = []
        if not self._merge_overlaps:
            for note_start, _ in notes_sorted:
                for a, b in self._merged:
                    if a < note_start < b:
                        self._notches.append(note_start - 1)
                        break

        # Keep the original sorted notes for extent calculation.
        self._notes = notes_sorted

    def inputs(self) -> list[ProcessingElement]:
        return []

    def _compute_extent(self) -> Extent:
        if len(self._notes) == 0:
            return Extent(None, None)
        else:
            # Extent starts with the onset of the first note (start) and ends
            # with the end of the last note (start + duration)
            first = self._notes[0]
            last = self._notes[-1]
            return Extent(first[0], last[0] + last[1])

    def _render_gate(self, start: int, duration: int) -> Snippet:
        out = np.zeros((duration, 1), dtype=np.float32)

        # Fill the output buffer for any merged interval that overlaps [start, start+duration).
        buf_end = start + duration
        for a, b in self._merged:
            lo = max(a, start) - start
            hi = min(b, buf_end) - start
            if lo < hi:
                out[lo:hi, 0] = 1.0

        # Retrigger notches (merge_overlaps=False): force a one-sample 0
        # before each swallowed onset so every note has a rising edge.
        for n in self._notches:
            if start <= n < buf_end:
                out[n - start, 0] = 0.0

        return Snippet(start, out)
