"""
Tests for ScheduledGatePE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np

import pygmu2 as pg
from pygmu2.scheduled_gate_pe import ScheduledGatePE


def gate(notes, start=0, duration=None):
    """Render a ScheduledGatePE and return the flat 1-D output array."""
    sg = ScheduledGatePE(notes)
    sg.on_start()
    if duration is None:
        ext = sg.extent()
        duration = ext.end - ext.start
        start = ext.start
    return sg.render(start, duration).data[:, 0]


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


class TestScheduledGatePEConstruction:

    def test_inputs_is_empty(self):
        assert ScheduledGatePE([]).inputs() == []

    def test_stateful(self):
        assert not ScheduledGatePE([]).stateful

    def test_channel_count(self):
        assert ScheduledGatePE([]).channel_count() == 1

    def test_unsorted_input_is_sorted(self):
        """Notes given out of order must be sorted by onset."""
        sg = ScheduledGatePE([(20, 5), (0, 5), (10, 5)])
        assert sg._notes[0][0] == 0
        assert sg._notes[1][0] == 10
        assert sg._notes[2][0] == 20


# ---------------------------------------------------------------------------
# Extent
# ---------------------------------------------------------------------------


class TestScheduledGatePEExtent:

    def test_empty_notes_infinite_extent(self):
        ext = ScheduledGatePE([]).extent()
        assert ext.start is None
        assert ext.end is None

    def test_single_note_extent(self):
        ext = ScheduledGatePE([(10, 5)]).extent()
        assert ext.start == 10
        assert ext.end == 15

    def test_multiple_notes_extent(self):
        """Extent spans from first onset to end of last note."""
        ext = ScheduledGatePE([(10, 5), (30, 10), (50, 3)]).extent()
        assert ext.start == 10
        assert ext.end == 53

    def test_unsorted_notes_extent(self):
        """Extent is correct even when notes are given unsorted."""
        ext = ScheduledGatePE([(50, 3), (10, 5), (30, 10)]).extent()
        assert ext.start == 10
        assert ext.end == 53


# ---------------------------------------------------------------------------
# Output shape and GateSignal contract
# ---------------------------------------------------------------------------


class TestScheduledGatePEOutput:

    def test_shape_is_N_by_1(self):
        sg = ScheduledGatePE([(0, 10)])
        sg.on_start()
        assert sg.render(0, 20).data.shape == (20, 1)

    def test_dtype_is_float32(self):
        sg = ScheduledGatePE([(0, 10)])
        sg.on_start()
        assert sg.render(0, 20).data.dtype == np.float32

    def test_values_are_exactly_zero_or_one(self):
        """GateSignal validation must pass (values must be exactly 0 or 1)."""
        sg = ScheduledGatePE([(5, 3), (15, 7), (30, 4)])
        sg.on_start()
        out = sg.render(0, 40).data[:, 0]
        assert set(out.tolist()).issubset({0.0, 1.0})

    def test_empty_notes_all_zeros(self):
        sg = ScheduledGatePE([])
        sg.on_start()
        out = sg.render(0, 20).data[:, 0]
        np.testing.assert_array_equal(out, 0.0)


# ---------------------------------------------------------------------------
# Gate logic — separated notes
# ---------------------------------------------------------------------------


class TestScheduledGatePEGateLogic:

    def test_before_first_note_is_zero(self):
        out = gate([(10, 5)], start=0, duration=10)
        np.testing.assert_array_equal(out, 0.0)

    def test_during_note_is_one(self):
        out = gate([(10, 5)], start=10, duration=5)
        np.testing.assert_array_equal(out, 1.0)

    def test_after_note_is_zero(self):
        out = gate([(10, 5)], start=15, duration=10)
        np.testing.assert_array_equal(out, 0.0)

    def test_single_note_exact_span(self):
        """Gate is 1 exactly for samples [start, start+duration)."""
        out = gate([(10, 5)], start=0, duration=20)
        np.testing.assert_array_equal(out[:10], 0.0)
        np.testing.assert_array_equal(out[10:15], 1.0)
        np.testing.assert_array_equal(out[15:], 0.0)

    def test_two_separated_notes(self):
        """Gap between notes produces 0 in between."""
        out = gate([(5, 3), (15, 4)], start=0, duration=25)
        np.testing.assert_array_equal(out[:5], 0.0)
        np.testing.assert_array_equal(out[5:8], 1.0)
        np.testing.assert_array_equal(out[8:15], 0.0)
        np.testing.assert_array_equal(out[15:19], 1.0)
        np.testing.assert_array_equal(out[19:], 0.0)

    def test_note_at_sample_zero(self):
        out = gate([(0, 5)], start=0, duration=10)
        np.testing.assert_array_equal(out[:5], 1.0)
        np.testing.assert_array_equal(out[5:], 0.0)

    def test_single_sample_note(self):
        """A note of duration 1 produces exactly one high sample."""
        out = gate([(10, 1)], start=0, duration=20)
        assert out[10] == 1.0
        assert out[9] == 0.0
        assert out[11] == 0.0

    def test_adjacent_notes_have_gap(self):
        """Notes that are adjacent (next_start == prev_end) produce a 1-sample gap."""
        out = gate([(5, 5), (10, 5)], start=0, duration=20)
        # note 1: [5,10), note 2: [10,15) — next_start == prev_end → NOT merged
        np.testing.assert_array_equal(out[5:10], 1.0)
        # sample 10: gate falls for one sample, then next note starts
        # (adjacent means they abut but the gate drops briefly since next_start == b,
        # which fails the strict overlap condition note_start < b)
        np.testing.assert_array_equal(out[10:15], 1.0)


# ---------------------------------------------------------------------------
# Note merging
# ---------------------------------------------------------------------------


class TestScheduledGatePEMerging:

    def test_overlapping_notes_merged(self):
        """Overlapping notes keep gate high continuously."""
        out = gate([(5, 10), (10, 10)], start=0, duration=30)
        # [5,15) and [10,20) → merged [5,20)
        np.testing.assert_array_equal(out[:5], 0.0)
        np.testing.assert_array_equal(out[5:20], 1.0)
        np.testing.assert_array_equal(out[20:], 0.0)

    def test_contained_note_does_not_shorten_interval(self):
        """A note wholly inside another doesn't shorten the outer note."""
        out = gate([(5, 20), (8, 3)], start=0, duration=30)
        # [5,25) contains [8,11) → merged [5,25)
        np.testing.assert_array_equal(out[5:25], 1.0)
        np.testing.assert_array_equal(out[25:], 0.0)

    def test_three_chained_overlapping_notes(self):
        """Three notes that each overlap the next merge into one span."""
        # [0,5), [3,8), [6,10) → merged [0,10)
        out = gate([(0, 5), (3, 5), (6, 4)], start=0, duration=15)
        np.testing.assert_array_equal(out[:10], 1.0)
        np.testing.assert_array_equal(out[10:], 0.0)

    def test_unsorted_input_same_as_sorted(self):
        """Unsorted notes produce identical output to sorted notes."""
        notes_sorted = [(0, 5), (3, 5), (15, 4)]
        notes_unsorted = [(15, 4), (0, 5), (3, 5)]
        out_s = gate(notes_sorted, start=0, duration=25)
        out_u = gate(notes_unsorted, start=0, duration=25)
        np.testing.assert_array_equal(out_s, out_u)

    def test_merged_stored_at_construction(self):
        """_merged is computed in __init__, not in _render_gate."""
        sg = ScheduledGatePE([(0, 5), (3, 5), (15, 4)])
        assert sg._merged == [(0, 8), (15, 19)]

    def test_no_notes_merged_is_empty(self):
        assert ScheduledGatePE([])._merged == []

    def test_single_note_merged(self):
        assert ScheduledGatePE([(10, 5)])._merged == [(10, 15)]


# ---------------------------------------------------------------------------
# Buffer boundary conditions
# ---------------------------------------------------------------------------


class TestScheduledGatePEBufferBoundaries:

    def test_render_before_any_note(self):
        out = gate([(100, 10)], start=0, duration=50)
        np.testing.assert_array_equal(out, 0.0)

    def test_render_after_all_notes(self):
        out = gate([(0, 10)], start=50, duration=20)
        np.testing.assert_array_equal(out, 0.0)

    def test_render_overlapping_note_start(self):
        """Buffer starts before note onset and ends during it."""
        out = gate([(10, 10)], start=5, duration=10)
        # buffer covers [5,15): samples 5-9 are 0, samples 10-14 are 1
        np.testing.assert_array_equal(out[:5], 0.0)
        np.testing.assert_array_equal(out[5:], 1.0)

    def test_render_overlapping_note_end(self):
        """Buffer starts during a note and ends after it."""
        out = gate([(5, 10)], start=10, duration=10)
        # buffer covers [10,20): note ends at 15, so [10,15) → 1, [15,20) → 0
        np.testing.assert_array_equal(out[:5], 1.0)
        np.testing.assert_array_equal(out[5:], 0.0)

    def test_render_entirely_inside_note(self):
        """Buffer falls entirely within a note."""
        out = gate([(0, 100)], start=20, duration=10)
        np.testing.assert_array_equal(out, 1.0)

    def test_multiple_render_calls_consistent(self):
        """Pure PE: same render request always returns the same data."""
        sg = ScheduledGatePE([(10, 5), (25, 8)])
        sg.on_start()
        out1 = sg.render(0, 40).data[:, 0].copy()
        out2 = sg.render(0, 40).data[:, 0]
        np.testing.assert_array_equal(out1, out2)

    def test_split_render_matches_full_render(self):
        """Two consecutive render calls reconstruct the same output as one call."""
        notes = [(5, 8), (20, 6), (35, 4)]
        sg = ScheduledGatePE(notes)
        sg.on_start()
        full = sg.render(0, 50).data[:, 0].copy()
        part_a = sg.render(0, 25).data[:, 0]
        part_b = sg.render(25, 25).data[:, 0]
        np.testing.assert_array_equal(np.concatenate([part_a, part_b]), full)


class TestMergeOverlaps:
    """merge_overlaps=False punches a one-sample 0 before each onset that
    would otherwise be swallowed by legato, so GateToTriggerPE sees one
    rising edge per note."""

    def test_legato_default_merges_onsets(self):
        gate = pg.ScheduledGatePE([(0, 100), (50, 100)])  # overlapping notes
        trig = pg.GateToTriggerPE(gate)
        r = pg.NullRenderer(sample_rate=44100)
        r.set_source(trig)
        r.start()
        events = trig.render(0, 200).data[:, 0]
        assert events.sum() == 1.0  # one fused span -> one onset

    def test_merge_overlaps_false_preserves_onsets(self):
        gate = pg.ScheduledGatePE([(0, 100), (50, 100)], merge_overlaps=False)
        trig = pg.GateToTriggerPE(gate)
        r = pg.NullRenderer(sample_rate=44100)
        r.set_source(trig)
        r.start()
        events = trig.render(0, 200).data[:, 0]
        assert events.sum() == 2.0  # both notes fire
        assert events[0] == 1.0 and events[50] == 1.0
