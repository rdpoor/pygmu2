"""
Tests for HoldPE (the merged SampleHoldPE + TrackHoldPE).

The control signal's shape selects the behaviour: one-sample trigger
events give sample-and-hold; sustained gates give track-and-hold.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

from pygmu2 import (
    ArrayPE,
    ConstantPE,
    GateToTriggerPE,
    HoldPE,
    IdentityPE,
    NullRenderer,
    PeriodicGatePE,
)


def _started(pe):
    renderer = NullRenderer(sample_rate=44100)
    renderer.set_source(pe)
    renderer.start()
    return pe


class TestHoldPEBasics:
    def test_construct_and_properties(self):
        hold = HoldPE(
            ConstantPE(1.0),
            GateToTriggerPE(PeriodicGatePE(frequency=10.0)),
            initial_value=0.5,
        )
        assert hold.initial_value == 0.5
        assert hold.stateful
        assert hold.channel_count() == 1

    def test_inputs(self):
        src = ConstantPE(1.0)
        ctrl = GateToTriggerPE(PeriodicGatePE(frequency=10.0))
        assert HoldPE(src, ctrl).inputs() == [src, ctrl]

    def test_extent_is_source_extent(self):
        """The held value is only meaningful where the source has data."""
        from pygmu2 import CropPE

        src = CropPE(ConstantPE(1.0), 0, 1000)
        hold = HoldPE(src, GateToTriggerPE(PeriodicGatePE(frequency=10.0)))
        assert hold.extent() == src.extent()

    def test_repr(self):
        r = repr(
            HoldPE(ConstantPE(1.0), GateToTriggerPE(PeriodicGatePE(frequency=10.0)))
        )
        assert "HoldPE" in r and "ConstantPE" in r


class TestSampleAndHold:
    """A TriggerSignal control latches the source at each event."""

    def test_initial_value_before_first_event(self):
        # trigger fires at sample 0 only if phase aligns; use a schedule
        # via ArrayPE-like ramp: IdentityPE tells us WHEN we latched.
        hold = _started(
            HoldPE(
                IdentityPE(),
                GateToTriggerPE(PeriodicGatePE(frequency=44100.0 / 100.0)),
                initial_value=-1.0,
            )
        )
        out = hold.render(0, 100).data[:, 0]
        # gate rising edge at sample 0 -> trigger latches identity(0)=0
        assert out[0] == 0.0
        assert np.all(out[:100] == 0.0)

    def test_latches_source_value_at_events(self):
        # period 50: events at 0, 50, 100, ...
        hold = _started(
            HoldPE(
                IdentityPE(), GateToTriggerPE(PeriodicGatePE(frequency=44100.0 / 50.0))
            )
        )
        out = hold.render(0, 150).data[:, 0]
        assert np.all(out[0:50] == 0.0)  # latched identity(0)
        assert np.all(out[50:100] == 50.0)  # latched identity(50)
        assert np.all(out[100:150] == 100.0)

    def test_state_persists_across_blocks(self):
        hold = _started(
            HoldPE(
                IdentityPE(), GateToTriggerPE(PeriodicGatePE(frequency=44100.0 / 100.0))
            )
        )
        a = hold.render(0, 60).data[:, 0]
        b = hold.render(60, 60).data[:, 0]
        assert np.all(a == 0.0)
        assert np.all(b[:40] == 0.0)  # still holding across the block seam
        assert np.all(b[40:] == 100.0)  # event at 100

    def test_reset_restores_initial_value(self):
        hold = _started(
            HoldPE(
                IdentityPE(),
                GateToTriggerPE(PeriodicGatePE(frequency=44100.0 / 50.0)),
                initial_value=7.0,
            )
        )
        hold.render(0, 100)
        # Seeking requires resetting the WHOLE graph: the derived trigger
        # (GateToTriggerPE) is stateful too.
        seen = set()

        def graph_reset(pe):
            if id(pe) in seen:
                return
            seen.add(id(pe))
            for inp in pe.inputs():
                graph_reset(inp)
            pe.reset_state()

        graph_reset(hold)
        # After a reset the edge detector's memory is gone, so rendering
        # where the gate is HIGH would fire a synthetic edge. Render where
        # the gate is LOW instead: period 50, duty 0.5 -> low on [75, 100).
        out = hold.render(75, 15).data[:, 0]
        assert np.all(out == 7.0)


class TestTrackAndHold:
    """A GateSignal control follows the source while high, holds while low."""

    def test_follows_while_high_holds_while_low(self):
        # gate: period 100, duty 0.5 -> high on [0,50), low on [50,100)
        gate = PeriodicGatePE(frequency=44100.0 / 100.0, duty_cycle=0.5)
        hold = _started(HoldPE(IdentityPE(), gate))
        out = hold.render(0, 100).data[:, 0]
        np.testing.assert_array_equal(out[:50], np.arange(50, dtype=np.float32))
        assert np.all(out[50:] == 49.0)  # frozen at the last tracked value

    def test_trigger_from_gate_is_sample_and_hold(self):
        """The derivation path: GateToTriggerPE(gate) turns tracking into
        latching — one PE, two behaviours, chosen by the control signal."""
        gate = PeriodicGatePE(frequency=44100.0 / 100.0, duty_cycle=0.5)
        hold = _started(HoldPE(IdentityPE(), GateToTriggerPE(gate)))
        out = hold.render(0, 100).data[:, 0]
        assert np.all(out == 0.0)  # single latch at the rising edge


class TestHoldPEMultichannelSource:
    def test_uses_channel_zero(self):
        stereoish = ConstantPE(0.25, channels=2)
        hold = _started(
            HoldPE(stereoish, GateToTriggerPE(PeriodicGatePE(frequency=100.0)))
        )
        snippet = hold.render(0, 64)
        assert snippet.channels == 1
        assert np.all(snippet.data == 0.25)
