"""
Tests for GateToTriggerPE (rising-edge detector: GateSignal → TriggerSignal).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.gate_to_trigger_pe import GateToTriggerPE
from pygmu2.signal_to_gate_pe import SignalToGatePE


def make_gate(samples):
    """
    Build a GateToTriggerPE from a list of gate values (0.0 or 1.0).
    Uses SignalToGatePE with threshold 0.5 to convert the raw values cleanly.
    """
    data = np.array(samples, dtype=np.float32)
    source = pg.ArrayPE(data)
    gate = SignalToGatePE(source, low_threshold=0.25, high_threshold=0.75)
    return gate


def render_trigger(gate_values, sr=1000):
    """Render GateToTriggerPE and return the 1-D output array."""
    pg.set_sample_rate(sr)
    gate = make_gate(gate_values)
    trigger = GateToTriggerPE(gate)
    trigger.on_start()
    n = len(gate_values)
    return trigger.render(0, n).data[:, 0]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGateToTriggerPEConstruction:

    def test_is_not_pure(self):
        gate = SignalToGatePE(pg.ConstantPE(0.0))
        pe = GateToTriggerPE(gate)
        assert pe.is_pure() is False

    def test_channel_count(self):
        gate = SignalToGatePE(pg.ConstantPE(0.0))
        pe = GateToTriggerPE(gate)
        assert pe.channel_count() == 1

    def test_inputs_contains_gate(self):
        gate = SignalToGatePE(pg.ConstantPE(0.5))
        pe = GateToTriggerPE(gate)
        assert gate in pe.inputs()
        assert len(pe.inputs()) == 1

    def test_repr(self):
        gate = SignalToGatePE(pg.ConstantPE(0.5))
        pe = GateToTriggerPE(gate)
        assert "GateToTriggerPE" in repr(pe)

    def test_is_trigger_signal(self):
        from pygmu2.trigger_signal import TriggerSignal

        gate = SignalToGatePE(pg.ConstantPE(0.0))
        pe = GateToTriggerPE(gate)
        assert isinstance(pe, TriggerSignal)


# ---------------------------------------------------------------------------
# Output shape and value constraints
# ---------------------------------------------------------------------------


class TestGateToTriggerPEShape:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(1000)
        yield
        pg.set_sample_rate(44100)

    def test_output_shape_mono(self):
        gate = make_gate([0.0, 1.0, 1.0, 0.0])
        trigger = GateToTriggerPE(gate)
        trigger.on_start()
        snip = trigger.render(0, 4)
        assert snip.data.shape == (4, 1)

    def test_values_are_zero_or_one(self):
        """All output samples must be exactly 0.0 or 1.0."""
        out = render_trigger([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        assert set(out.tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Rising-edge detection
# ---------------------------------------------------------------------------


class TestGateToTriggerPERisingEdge:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(1000)
        yield
        pg.set_sample_rate(44100)

    def test_single_rising_edge_emits_one(self):
        """0→1 transition emits exactly +1 at the first high sample."""
        out = render_trigger([0.0, 0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_no_trigger_when_gate_stays_low(self):
        """Gate held at 0 produces all-zero trigger output."""
        out = render_trigger([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 0.0])

    def test_no_trigger_when_gate_stays_high(self):
        """Gate held at 1 after initial open: only one trigger at the start."""
        out = render_trigger([1.0, 1.0, 1.0, 1.0])
        # First sample is a rising edge from initial _prev=0
        np.testing.assert_array_equal(out, [1.0, 0.0, 0.0, 0.0])

    def test_falling_edge_produces_no_trigger(self):
        """1→0 transition must NOT produce a trigger event."""
        out = render_trigger([1.0, 1.0, 0.0, 0.0])
        # Rising edge at i=0 (prev=0 → 1); falling edge at i=2 produces nothing
        np.testing.assert_array_equal(out, [1.0, 0.0, 0.0, 0.0])

    def test_multiple_cycles_emit_multiple_triggers(self):
        """Each 0→1 transition emits a separate +1."""
        out = render_trigger([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(out, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    def test_trigger_only_on_first_high_sample(self):
        """With sustained gate, trigger fires once then stays 0 until next edge."""
        out = render_trigger([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(out, [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# State persistence and lifecycle
# ---------------------------------------------------------------------------


class TestGateToTriggerPEState:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(1000)
        yield
        pg.set_sample_rate(44100)

    def test_prev_state_persists_across_render_chunks(self):
        """
        When gate is high at end of chunk 1, no new trigger in chunk 2
        while gate remains high.
        """
        gate_vals = [0.0, 1.0, 1.0, 1.0]
        gate = make_gate(gate_vals)
        trigger = GateToTriggerPE(gate)
        trigger.on_start()

        chunk1 = trigger.render(0, 2).data[:, 0]  # [0→1 edge at i=1]
        chunk2 = trigger.render(2, 2).data[:, 0]  # gate still high → no trigger

        np.testing.assert_array_equal(chunk1, [0.0, 1.0])
        np.testing.assert_array_equal(chunk2, [0.0, 0.0])

    def test_rising_edge_across_chunk_boundary(self):
        """
        Gate is 0 at end of chunk 1, 1 at start of chunk 2 → trigger fires
        at first sample of chunk 2.
        """
        gate_vals = [1.0, 0.0, 1.0, 1.0]
        gate = make_gate(gate_vals)
        trigger = GateToTriggerPE(gate)
        trigger.on_start()

        chunk1 = trigger.render(0, 2).data[:, 0]  # [edge at i=0], gate=0 at end
        chunk2 = trigger.render(2, 2).data[:, 0]  # rising edge at i=2

        np.testing.assert_array_equal(chunk1, [1.0, 0.0])
        np.testing.assert_array_equal(chunk2, [1.0, 0.0])

    def test_on_start_resets_prev_to_zero(self):
        """
        After on_start(), _prev is 0, so a gate=1 input immediately triggers.
        """
        gate_vals = [1.0, 1.0]
        gate = make_gate(gate_vals)
        trigger = GateToTriggerPE(gate)
        trigger.on_start()

        out = trigger.render(0, 2).data[:, 0]
        # _prev starts at 0 → rising edge at i=0
        assert out[0] == 1.0
        assert out[1] == 0.0


# ---------------------------------------------------------------------------
# Extent
# ---------------------------------------------------------------------------


class TestGateToTriggerPEExtent:

    def test_extent_matches_gate(self):
        source = pg.CropPE(pg.ConstantPE(1.0), start=100, duration=200)
        gate = SignalToGatePE(source)
        trigger = GateToTriggerPE(gate)
        assert trigger.extent() == pg.Extent(100, 300)

    def test_infinite_gate_gives_infinite_extent(self):
        gate = SignalToGatePE(pg.ConstantPE(0.5))
        trigger = GateToTriggerPE(gate)
        ext = trigger.extent()
        assert ext.start is None
        assert ext.end is None


# ---------------------------------------------------------------------------
# Integration: SignalToGatePE → GateToTriggerPE
# ---------------------------------------------------------------------------


class TestGateToTriggerPEIntegration:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(1000)
        yield
        pg.set_sample_rate(44100)

    def test_pulse_train_from_threshold_crossing(self):
        """
        A signal that crosses a threshold multiple times should produce one
        trigger per crossing, detected via the full SignalToGatePE pipeline.
        """
        # 4 pulses of high (1.5 > 1.0) then low (0.0 < 0.0 doesn't trigger,
        # but signal 0.0 < low=0.0 does not close; use negative to close)
        signal = [1.5, 1.5, -0.5, -0.5] * 4
        source = pg.ArrayPE(np.array(signal, dtype=np.float32))
        gate = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0)
        trigger = GateToTriggerPE(gate)
        trigger.on_start()

        out = trigger.render(0, 16).data[:, 0]

        # Triggers fire at each 0→1 transition (i=0, 4, 8, 12)
        expected = np.zeros(16, dtype=np.float32)
        expected[[0, 4, 8, 12]] = 1.0
        np.testing.assert_array_equal(out, expected)

    def test_accessible_via_pg_namespace(self):
        """GateToTriggerPE must be importable as pg.GateToTriggerPE."""
        assert hasattr(pg, "GateToTriggerPE")
        assert pg.GateToTriggerPE is GateToTriggerPE
