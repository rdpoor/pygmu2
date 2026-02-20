"""
Tests for SlewLimiterPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.slew_limiter_pe import SlewLimiterPE, SlewMode


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestSlewLimiterPEConstruction:

    def test_symmetric_rates(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=2.0)
        assert sl.rise_rate == pytest.approx(2.0)
        assert sl.fall_rate == pytest.approx(2.0)

    def test_asymmetric_rates(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=2.0, fall_rate=4.0)
        assert sl.rise_rate == pytest.approx(2.0)
        assert sl.fall_rate == pytest.approx(4.0)

    def test_default_mode_is_linear(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=1.0)
        assert sl.mode == SlewMode.LINEAR

    def test_exponential_mode(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=1.0, mode=SlewMode.EXPONENTIAL)
        assert sl.mode == SlewMode.EXPONENTIAL

    def test_invalid_rise_rate_raises(self):
        src = pg.ConstantPE(1.0)
        with pytest.raises(ValueError, match="rise_rate"):
            SlewLimiterPE(src, rise_rate=0.0)

    def test_invalid_fall_rate_raises(self):
        src = pg.ConstantPE(1.0)
        with pytest.raises(ValueError, match="fall_rate"):
            SlewLimiterPE(src, rise_rate=1.0, fall_rate=-1.0)

    def test_is_not_pure(self):
        sl = SlewLimiterPE(pg.ConstantPE(1.0), rise_rate=1.0)
        assert sl.is_pure() is False

    def test_channel_count_is_one(self):
        sl = SlewLimiterPE(pg.ConstantPE(1.0), rise_rate=1.0)
        assert sl.channel_count() == 1

    def test_inputs_exposes_source(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=1.0)
        assert sl.inputs() == [src]

    def test_repr(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=2.0, fall_rate=3.0)
        r = repr(sl)
        assert "SlewLimiterPE" in r
        assert "2.0" in r


# ---------------------------------------------------------------------------
# Linear mode rendering
# ---------------------------------------------------------------------------

class TestSlewLimiterPELinear:

    @pytest.fixture(autouse=True)
    def _sr(self):
        # Use sr=10 so rise_rate=1.0/s → 0.1 units/sample (easy arithmetic)
        pg.set_sample_rate(10)
        yield
        pg.set_sample_rate(44100)

    def test_step_up_at_limited_rate(self):
        """Output ramps up to a step target at rise_rate."""
        # rise_rate=1 unit/s at sr=10 → 0.1 unit/sample
        # Source immediately jumps to 1.0; output should ramp
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=1.0)

        out = sl.render(0, 6).data[:, 0]
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_step_down_at_limited_rate(self):
        """Output ramps down from 1.0 to 0.0 at fall_rate."""
        # Start current at 1.0 by pre-rendering up, then switch source to 0.0
        src = pg.ConstantPE(0.0)
        sl = SlewLimiterPE(src, rise_rate=1.0, fall_rate=1.0)
        sl._current = 1.0  # manually prime state

        out = sl.render(0, 6).data[:, 0]
        expected = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_asymmetric_rates(self):
        """rise_rate and fall_rate are applied independently."""
        # rise: 1 u/s (0.1/sample), fall: 2 u/s (0.2/sample)
        # Alternate high/low source to exercise both
        sl = SlewLimiterPE(pg.ConstantPE(1.0), rise_rate=1.0, fall_rate=2.0)

        # Rise to 1.0: 0.1/sample
        out_up = sl.render(0, 5).data[:, 0]
        np.testing.assert_allclose(out_up, [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-5)

        # Now switch source to 0.0: fall at 0.2/sample from 0.5
        sl._source = pg.ConstantPE(0.0)
        out_down = sl.render(5, 3).data[:, 0]
        np.testing.assert_allclose(out_down, [0.3, 0.1, 0.0], atol=1e-5)

    def test_reaches_target_and_stays(self):
        """Output stays at target once it gets there."""
        # rise_rate=10 u/s at sr=10 → 1 unit/sample; reaches 1.0 in one step
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=10.0)

        out = sl.render(0, 5).data[:, 0]
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 1.0, 1.0], atol=1e-6)

    def test_state_persists_across_renders(self):
        """Current value carries over from one render call to the next."""
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=1.0)  # 0.1/sample

        out1 = sl.render(0, 3).data[:, 0]
        np.testing.assert_allclose(out1, [0.1, 0.2, 0.3], atol=1e-5)

        out2 = sl.render(3, 3).data[:, 0]
        np.testing.assert_allclose(out2, [0.4, 0.5, 0.6], atol=1e-5)

    def test_on_start_resets_current(self):
        """on_start resets internal current to 0."""
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=1.0)

        sl.render(0, 5)
        assert sl._current == pytest.approx(0.5, abs=1e-5)

        sl.on_start()
        assert sl._current == pytest.approx(0.0)

    def test_zero_duration(self):
        """Zero-duration render returns empty snippet."""
        sl = SlewLimiterPE(pg.ConstantPE(1.0), rise_rate=1.0)
        snip = sl.render(0, 0)
        assert snip.data.shape[0] == 0

    def test_output_shape_is_mono(self):
        """Output must be (duration, 1)."""
        sl = SlewLimiterPE(pg.ConstantPE(1.0), rise_rate=1.0)
        snip = sl.render(0, 4)
        assert snip.data.shape == (4, 1)


# ---------------------------------------------------------------------------
# Exponential mode rendering
# ---------------------------------------------------------------------------

class TestSlewLimiterPEExponential:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(10)
        yield
        pg.set_sample_rate(44100)

    def test_rises_toward_target(self):
        """In EXPONENTIAL mode output increases monotonically toward target."""
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=2.0, mode=SlewMode.EXPONENTIAL)

        out = sl.render(0, 10).data[:, 0]
        # Monotonically increasing
        assert np.all(np.diff(out) >= 0)
        # Never exceeds target
        assert np.all(out <= 1.0 + 1e-6)

    def test_approaches_but_never_exceeds_target(self):
        """EXPONENTIAL output asymptotically approaches target from below."""
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rise_rate=5.0, mode=SlewMode.EXPONENTIAL)

        out = sl.render(0, 50).data[:, 0]
        assert out[-1] > 0.9, "Should get close to target after 50 samples"
        assert np.all(out <= 1.0 + 1e-6)

    def test_falls_toward_zero(self):
        """EXPONENTIAL mode tracks downward as well."""
        src = pg.ConstantPE(0.0)
        sl = SlewLimiterPE(src, rise_rate=2.0, mode=SlewMode.EXPONENTIAL)
        sl._current = 1.0  # prime state

        out = sl.render(0, 10).data[:, 0]
        # Monotonically decreasing
        assert np.all(np.diff(out) <= 0)
        # Never goes below target
        assert np.all(out >= -1e-6)


# ---------------------------------------------------------------------------
# Integration: composed stepped random LFO
# ---------------------------------------------------------------------------

class TestSlewLimiterPEComposed:

    def test_slew_smooths_stepped_signal(self):
        """SlewLimiter on a stepped S&H source produces a smooth ramp."""
        from pygmu2.sample_hold_pe import SampleHoldPE

        pg.set_sample_rate(100)

        # Stepped source: NoisePE with fixed seed triggered at 10 Hz
        src = pg.NoisePE(min_value=0.0, max_value=1.0, seed=7)
        trig = pg.PeriodicTrigger(hz=10.0)  # every 10 samples
        stepped = SampleHoldPE(src, trig)

        # Slew-limit the steps (rise_rate = 5 units/s → 0.05/sample)
        slewed = SlewLimiterPE(stepped, rise_rate=5.0)

        src.on_start()
        out = slewed.render(0, 100).data[:, 0]

        # The output should be smooth: max difference between consecutive samples
        # is at most 5/100 = 0.05 (the slew limit)
        diffs = np.abs(np.diff(out.astype(np.float64)))
        assert np.all(diffs <= 0.05 + 1e-6), (
            f"Max diff {diffs.max():.6f} exceeds slew limit 0.05"
        )

        pg.set_sample_rate(44100)
