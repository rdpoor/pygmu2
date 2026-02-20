"""
Tests for SampleHoldPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.trigger_signal import TriggerSignal
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.sample_hold_pe import SampleHoldPE


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _ArrayTrigger(TriggerSignal):
    """Minimal TriggerSignal backed by a fixed array for testing."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.float32).reshape(-1, 1)

    def inputs(self):
        return []

    def is_pure(self):
        return True

    def _compute_extent(self):
        return Extent(0, len(self._data))

    def _render_trigger(self, start: int, duration: int) -> Snippet:
        out = np.zeros((duration, 1), dtype=np.float32)
        for i in range(duration):
            idx = start + i
            if 0 <= idx < len(self._data):
                out[i, 0] = self._data[idx, 0]
        return Snippet(start, out)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestSampleHoldPEConstruction:

    def test_default_initial_value(self):
        src = pg.ConstantPE(1.0)
        trig = _ArrayTrigger([0, 0, 0, 0])
        sh = SampleHoldPE(src, trig)
        assert sh.initial_value == 0.0

    def test_custom_initial_value(self):
        src = pg.ConstantPE(1.0)
        trig = _ArrayTrigger([0, 0])
        sh = SampleHoldPE(src, trig, initial_value=0.5)
        assert sh.initial_value == 0.5

    def test_inputs_exposes_source_and_trigger(self):
        src = pg.ConstantPE(1.0)
        trig = _ArrayTrigger([0])
        sh = SampleHoldPE(src, trig)
        assert sh.inputs() == [src, trig]

    def test_is_not_pure(self):
        sh = SampleHoldPE(pg.ConstantPE(1.0), _ArrayTrigger([0]))
        assert sh.is_pure() is False

    def test_channel_count_is_one(self):
        sh = SampleHoldPE(pg.ConstantPE(1.0), _ArrayTrigger([0]))
        assert sh.channel_count() == 1

    def test_repr(self):
        src = pg.ConstantPE(1.0)
        trig = _ArrayTrigger([0])
        sh = SampleHoldPE(src, trig, initial_value=0.25)
        r = repr(sh)
        assert "SampleHoldPE" in r
        assert "0.25" in r


# ---------------------------------------------------------------------------
# Rendering tests
# ---------------------------------------------------------------------------

class TestSampleHoldPERender:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(10)
        yield
        pg.set_sample_rate(44100)

    def test_output_before_first_trigger_is_initial_value(self):
        """No trigger fired → output equals initial_value throughout."""
        src = pg.ConstantPE(1.0)
        trig = _ArrayTrigger([0, 0, 0, 0])
        sh = SampleHoldPE(src, trig, initial_value=0.5)

        out = sh.render(0, 4).data[:, 0]
        np.testing.assert_array_equal(out, [0.5, 0.5, 0.5, 0.5])

    def test_trigger_latches_source_value(self):
        """Trigger at sample 2 latches the source value at that sample."""
        # Source ramps: sample i = i*0.1 at t=0,1,2,3,...
        src = pg.ArrayPE([0.0, 0.1, 0.2, 0.3, 0.4], extend_mode=pg.ExtendMode.HOLD_LAST)
        trig = _ArrayTrigger([0, 0, 1, 0, 0])  # trigger at sample 2
        sh = SampleHoldPE(src, trig, initial_value=-1.0)

        out = sh.render(0, 5).data[:, 0]
        # Samples 0-1: initial_value = -1.0
        # Samples 2-4: latched value = 0.2
        expected = np.array([-1.0, -1.0, 0.2, 0.2, 0.2], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_multiple_triggers_update_held_value(self):
        """Each trigger latches a new value from the source."""
        src = pg.ArrayPE([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        trig = _ArrayTrigger([1, 0, 1, 0, 0, 1])  # triggers at 0, 2, 5
        sh = SampleHoldPE(src, trig, initial_value=-1.0)

        out = sh.render(0, 6).data[:, 0]
        # sample 0: trigger → latch 0.0 → output 0.0
        # sample 1: no trigger → hold 0.0
        # sample 2: trigger → latch 0.2 → output 0.2
        # sample 3: no trigger → hold 0.2
        # sample 4: no trigger → hold 0.2
        # sample 5: trigger → latch 0.5 → output 0.5
        expected = np.array([0.0, 0.0, 0.2, 0.2, 0.2, 0.5], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_held_value_persists_across_render_calls(self):
        """State carries over from one render call to the next."""
        src = pg.ArrayPE([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        trig = _ArrayTrigger([0, 0, 1, 0, 0, 0, 0, 0])  # trigger at sample 2

        sh = SampleHoldPE(src, trig, initial_value=-1.0)

        out1 = sh.render(0, 4).data[:, 0]
        # samples 0-1: -1.0; sample 2: latch 0.2; sample 3: hold 0.2
        expected1 = np.array([-1.0, -1.0, 0.2, 0.2], dtype=np.float32)
        np.testing.assert_allclose(out1, expected1, atol=1e-6)

        out2 = sh.render(4, 4).data[:, 0]
        # no more triggers → hold 0.2 through samples 4-7
        expected2 = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        np.testing.assert_allclose(out2, expected2, atol=1e-6)

    def test_on_start_resets_to_initial_value(self):
        """on_start() should reset held value to initial_value."""
        src = pg.ConstantPE(1.0)
        trig = _ArrayTrigger([1, 0, 0, 0])

        sh = SampleHoldPE(src, trig, initial_value=0.0)

        # Render once to latch 1.0
        sh.render(0, 4)
        assert sh._held_value == pytest.approx(1.0)

        # Reset via on_start
        sh.on_start()
        assert sh._held_value == pytest.approx(0.0)

    def test_output_shape_is_mono(self):
        """Output must be (duration, 1)."""
        src = pg.ConstantPE(1.0)
        trig = _ArrayTrigger([0, 0, 0])
        sh = SampleHoldPE(src, trig)

        snip = sh.render(0, 3)
        assert snip.data.shape == (3, 1)

    def test_zero_duration(self):
        """Zero-duration render should return empty snippet."""
        src = pg.ConstantPE(1.0)
        trig = _ArrayTrigger([])
        sh = SampleHoldPE(src, trig)

        snip = sh.render(0, 0)
        assert snip.data.shape[0] == 0


# ---------------------------------------------------------------------------
# Integration: with PeriodicTrigger and NoisePE
# ---------------------------------------------------------------------------

class TestSampleHoldPEIntegration:

    def test_with_periodic_trigger_produces_steps(self):
        """Output changes only at trigger events, forming a stepped signal."""
        sr = 100
        pg.set_sample_rate(sr)

        src = pg.NoisePE(min_value=0.0, max_value=1.0, seed=42)
        trig = pg.PeriodicTrigger(hz=10.0)  # trigger every 10 samples at sr=100
        sh = SampleHoldPE(src, trig)

        # NoisePE needs on_start() to initialize its RNG
        src.on_start()
        out = sh.render(0, 50).data[:, 0]

        # Samples within each 10-sample window should be identical
        # Window 0-9: latched at sample 0
        # Window 10-19: latched at sample 10, etc.
        for window_start in range(0, 50, 10):
            window = out[window_start:window_start + 10]
            assert np.all(window == window[0]), (
                f"Window at {window_start} not constant: {window}"
            )

        pg.set_sample_rate(44100)

    def test_constant_source_latches_constant(self):
        """With a constant source, any trigger still latches the same value."""
        pg.set_sample_rate(100)
        src = pg.ConstantPE(0.7)
        trig = pg.PeriodicTrigger(hz=10.0)
        sh = SampleHoldPE(src, trig)

        out = sh.render(0, 50).data[:, 0]
        np.testing.assert_allclose(out, 0.7, atol=1e-6)
        pg.set_sample_rate(44100)
