import numpy as np

from pygmu2.extent import Extent
from pygmu2.gate_signal import GateSignal
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet


class ScheduledGatePE(GateSignal):
    """
    Convert note durations into gate signals, specifically for feeding into an
    AdsrGatePE.

    notes is a list of (start, duration) pairs (in samples).  The _render
    method will generate gate signals: 1.0 at the onset of a note, 0.0 when
    the duration of that note has elapsed unless an new note has started
    before the previous note ends.
    """

    def __init__(self, notes: list[tuple[int, int]]):  # start, duration pairs
        # Merge overlapping note intervals once at construction time.
        notes_sorted = sorted(notes)
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

        return Snippet(start, out)
