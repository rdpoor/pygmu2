"""
Tests for TrackHoldPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.gate_signal import GateSignal
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.track_hold_pe import TrackHoldPE


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _ArrayGate(GateSignal):
    """Minimal GateSignal backed by a fixed array (values must be 0 or 1)."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32).reshape(-1, 1)

    def inputs(self):
        return []

    def is_pure(self):
        return True

    def _compute_extent(self):
        return Extent(0, len(self._data))

    def _render_gate(self, start: int, duration: int) -> Snippet:
        out = np.zeros((duration, 1), dtype=np.float32)
        for i in range(duration):
            idx = start + i
            if 0 <= idx < len(self._data):
                out[i, 0] = self._data[idx, 0]
        return Snippet(start, out)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestTrackHoldPEConstruction:

    def test_default_initial_value(self):
        src = pg.ConstantPE(1.0)
        gate = _ArrayGate([1, 1])
        th = TrackHoldPE(src, gate)
        assert th.initial_value == 0.0

    def test_custom_initial_value(self):
        src = pg.ConstantPE(1.0)
        gate = _ArrayGate([0, 0])
        th = TrackHoldPE(src, gate, initial_value=0.75)
        assert th.initial_value == 0.75

    def test_inputs_exposes_source_and_gate(self):
        src = pg.ConstantPE(1.0)
        gate = _ArrayGate([1])
        th = TrackHoldPE(src, gate)
        assert th.inputs() == [src, gate]

    def test_is_not_pure(self):
        th = TrackHoldPE(pg.ConstantPE(1.0), _ArrayGate([1]))
        assert th.is_pure() is False

    def test_channel_count_is_one(self):
        th = TrackHoldPE(pg.ConstantPE(1.0), _ArrayGate([1]))
        assert th.channel_count() == 1

    def test_repr(self):
        src = pg.ConstantPE(1.0)
        gate = _ArrayGate([1])
        th = TrackHoldPE(src, gate, initial_value=0.5)
        r = repr(th)
        assert "TrackHoldPE" in r
        assert "0.5" in r


# ---------------------------------------------------------------------------
# Rendering tests
# ---------------------------------------------------------------------------

class TestTrackHoldPERender:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(10)
        yield
        pg.set_sample_rate(44100)

    def test_tracks_source_when_gate_open(self):
        """While gate=1 output mirrors source sample-for-sample."""
        src = pg.ArrayPE([0.1, 0.2, 0.3, 0.4])
        gate = _ArrayGate([1, 1, 1, 1])
        th = TrackHoldPE(src, gate)

        out = th.render(0, 4).data[:, 0]
        np.testing.assert_allclose(out, [0.1, 0.2, 0.3, 0.4], atol=1e-6)

    def test_holds_when_gate_closed(self):
        """While gate=0 output holds the value from before the gate closed."""
        src = pg.ArrayPE([0.5, 0.6, 0.7, 0.8])
        gate = _ArrayGate([1, 0, 0, 0])  # open for 1 sample then close
        th = TrackHoldPE(src, gate)

        out = th.render(0, 4).data[:, 0]
        # sample 0: gate open → track 0.5
        # samples 1-3: gate closed → hold 0.5
        expected = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_initial_value_before_first_open(self):
        """Before gate first opens, initial_value is output."""
        src = pg.ConstantPE(1.0)
        gate = _ArrayGate([0, 0, 1, 0])
        th = TrackHoldPE(src, gate, initial_value=0.25)

        out = th.render(0, 4).data[:, 0]
        # samples 0-1: closed → 0.25
        # sample 2: open → track 1.0
        # sample 3: closed → hold 1.0
        expected = np.array([0.25, 0.25, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_track_hold_alternating_gate(self):
        """Gate alternating opens and closes produce track/hold segments."""
        src = pg.ArrayPE([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        gate = _ArrayGate([1, 0, 1, 0, 1, 0])
        th = TrackHoldPE(src, gate)

        out = th.render(0, 6).data[:, 0]
        # 0: open → 0.1
        # 1: close → hold 0.1
        # 2: open → 0.3
        # 3: close → hold 0.3
        # 4: open → 0.5
        # 5: close → hold 0.5
        expected = np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_state_persists_across_render_calls(self):
        """Held value carries into the next render call."""
        src = pg.ArrayPE([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        gate = _ArrayGate([1, 1, 0, 0, 0, 0])  # close after sample 1

        th = TrackHoldPE(src, gate)

        out1 = th.render(0, 3).data[:, 0]
        # 0: track 0.4, 1: track 0.5, 2: hold 0.5
        np.testing.assert_allclose(out1, [0.4, 0.5, 0.5], atol=1e-6)

        out2 = th.render(3, 3).data[:, 0]
        # gate still 0 → hold 0.5
        np.testing.assert_allclose(out2, [0.5, 0.5, 0.5], atol=1e-6)

    def test_on_start_resets_to_initial_value(self):
        """on_start() resets held value to initial_value."""
        src = pg.ConstantPE(1.0)
        gate = _ArrayGate([1, 0, 0, 0])
        th = TrackHoldPE(src, gate, initial_value=0.0)

        th.render(0, 4)
        assert th._held_value == pytest.approx(1.0)

        th.on_start()
        assert th._held_value == pytest.approx(0.0)

    def test_output_shape_is_mono(self):
        """Output must be (duration, 1)."""
        src = pg.ConstantPE(1.0)
        gate = _ArrayGate([1, 1, 1])
        th = TrackHoldPE(src, gate)

        snip = th.render(0, 3)
        assert snip.data.shape == (3, 1)

    def test_zero_duration(self):
        """Zero-duration render returns empty snippet."""
        src = pg.ConstantPE(1.0)
        gate = _ArrayGate([])
        th = TrackHoldPE(src, gate)

        snip = th.render(0, 0)
        assert snip.data.shape[0] == 0

    def test_gate_fully_closed_uses_initial_value(self):
        """With gate always 0, output is always initial_value."""
        src = pg.ConstantPE(0.9)
        gate = _ArrayGate([0, 0, 0, 0, 0])
        th = TrackHoldPE(src, gate, initial_value=0.3)

        out = th.render(0, 5).data[:, 0]
        np.testing.assert_allclose(out, [0.3, 0.3, 0.3, 0.3, 0.3], atol=1e-6)


# ---------------------------------------------------------------------------
# Integration: with PeriodicGate
# ---------------------------------------------------------------------------

class TestTrackHoldPEIntegration:

    def test_with_periodic_gate_tracks_and_holds(self):
        """With PeriodicGate, output tracks during open and holds during closed."""
        sr = 100
        pg.set_sample_rate(sr)

        # Ramp source: value = sample_index / sr
        src = pg.ArrayPE(
            np.arange(100, dtype=np.float32) / 100.0,
            extend_mode=pg.ExtendMode.HOLD_LAST,
        )

        # Gate: 5 Hz at 50% duty → 10-sample period, 5 open + 5 closed
        gate = pg.PeriodicGate(frequency=5.0, duty_cycle=0.5)

        th = TrackHoldPE(src, gate)
        out = th.render(0, 100).data[:, 0]

        # During hold periods (gate=0) consecutive samples should be equal
        gate_data = gate.render(0, 100).data[:, 0]
        for i in range(1, 100):
            if gate_data[i] == 0 and gate_data[i - 1] == 0:
                assert out[i] == pytest.approx(out[i - 1], abs=1e-6), (
                    f"Hold broken at sample {i}: {out[i-1]} → {out[i]}"
                )

        pg.set_sample_rate(44100)
