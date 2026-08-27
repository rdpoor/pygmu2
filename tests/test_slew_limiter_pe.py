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

    def test_rate_property(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rate=2.0)
        assert sl.rate == pytest.approx(2.0)

    def test_default_mode_is_linear(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rate=1.0)
        assert sl.mode == SlewMode.LINEAR

    def test_exponential_mode(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rate=1.0, mode=SlewMode.EXPONENTIAL)
        assert sl.mode == SlewMode.EXPONENTIAL

    def test_invalid_rate_zero_raises(self):
        src = pg.ConstantPE(1.0)
        with pytest.raises(ValueError, match="rate"):
            SlewLimiterPE(src, rate=0.0)

    def test_invalid_rate_negative_raises(self):
        src = pg.ConstantPE(1.0)
        with pytest.raises(ValueError, match="rate"):
            SlewLimiterPE(src, rate=-1.0)

    def test_is_not_pure(self):
        sl = SlewLimiterPE(pg.ConstantPE(1.0), rate=1.0)
        assert sl.stateful

    def test_channel_count_is_one(self):
        sl = SlewLimiterPE(pg.ConstantPE(1.0), rate=1.0)
        assert sl.channel_count() == 1

    def test_inputs_exposes_source_only_for_scalar_rate(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rate=1.0)
        assert sl.inputs() == [src]

    def test_inputs_exposes_source_and_rate_pe(self):
        src = pg.ConstantPE(1.0)
        rate_pe = pg.ConstantPE(2.0)
        sl = SlewLimiterPE(src, rate=rate_pe)
        assert sl.inputs() == [src, rate_pe]

    def test_repr(self):
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rate=2.0)
        r = repr(sl)
        assert "SlewLimiterPE" in r
        assert "2.0" in r


# ---------------------------------------------------------------------------
# Linear mode rendering
# ---------------------------------------------------------------------------


class TestSlewLimiterPELinear:

    @pytest.fixture(autouse=True)
    def _sr(self):
        # Use sr=10 so rate=1.0/s → 0.1 units/sample (easy arithmetic)
        pg.set_sample_rate(10)
        yield
        pg.set_sample_rate(44100)

    def test_step_up_at_limited_rate(self):
        """Output ramps up to a step target at rate."""
        # rate=1 unit/s at sr=10 → 0.1 unit/sample
        # Source immediately jumps to 1.0; output should ramp
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rate=1.0)

        out = sl.render(0, 6).data[:, 0]
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_step_down_at_limited_rate(self):
        """Output ramps down from 1.0 to 0.0 at rate."""
        src = pg.ConstantPE(0.0)
        sl = SlewLimiterPE(src, rate=1.0)
        sl._current = 1.0  # manually prime state

        out = sl.render(0, 6).data[:, 0]
        expected = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_symmetric_rise_and_fall(self):
        """Rise and fall occur at the same rate (symmetric)."""
        # rate=1 u/s (0.1/sample) at sr=10
        sl = SlewLimiterPE(pg.ConstantPE(1.0), rate=1.0)

        # Rise from 0 toward 1.0: 0.1/sample
        out_up = sl.render(0, 5).data[:, 0]
        np.testing.assert_allclose(out_up, [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-5)

        # Now fall toward 0.0: 0.1/sample from 0.5
        sl._source = pg.ConstantPE(0.0)
        out_down = sl.render(5, 5).data[:, 0]
        np.testing.assert_allclose(out_down, [0.4, 0.3, 0.2, 0.1, 0.0], atol=1e-5)

    def test_reaches_target_and_stays(self):
        """Output stays at target once it gets there."""
        # rate=10 u/s at sr=10 → 1 unit/sample; reaches 1.0 in one step
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rate=10.0)

        out = sl.render(0, 5).data[:, 0]
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 1.0, 1.0], atol=1e-6)

    def test_state_persists_across_renders(self):
        """Current value carries over from one render call to the next."""
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rate=1.0)  # 0.1/sample

        out1 = sl.render(0, 3).data[:, 0]
        np.testing.assert_allclose(out1, [0.1, 0.2, 0.3], atol=1e-5)

        out2 = sl.render(3, 3).data[:, 0]
        np.testing.assert_allclose(out2, [0.4, 0.5, 0.6], atol=1e-5)

    def test_on_start_resets_current(self):
        """on_start resets internal current to 0."""
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rate=1.0)

        sl.render(0, 5)
        assert sl._current == pytest.approx(0.5, abs=1e-5)

        sl.on_start()
        assert sl._current == pytest.approx(0.0)

    def test_zero_duration(self):
        """Zero-duration render returns empty snippet."""
        sl = SlewLimiterPE(pg.ConstantPE(1.0), rate=1.0)
        snip = sl.render(0, 0)
        assert snip.data.shape[0] == 0

    def test_output_shape_is_mono(self):
        """Output must be (duration, 1)."""
        sl = SlewLimiterPE(pg.ConstantPE(1.0), rate=1.0)
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
        sl = SlewLimiterPE(src, rate=2.0, mode=SlewMode.EXPONENTIAL)

        out = sl.render(0, 10).data[:, 0]
        # Monotonically increasing
        assert np.all(np.diff(out) >= 0)
        # Never exceeds target
        assert np.all(out <= 1.0 + 1e-6)

    def test_approaches_but_never_exceeds_target(self):
        """EXPONENTIAL output asymptotically approaches target from below."""
        src = pg.ConstantPE(1.0)
        sl = SlewLimiterPE(src, rate=5.0, mode=SlewMode.EXPONENTIAL)

        out = sl.render(0, 50).data[:, 0]
        assert out[-1] > 0.9, "Should get close to target after 50 samples"
        assert np.all(out <= 1.0 + 1e-6)

    def test_falls_toward_zero(self):
        """EXPONENTIAL mode tracks downward as well."""
        src = pg.ConstantPE(0.0)
        sl = SlewLimiterPE(src, rate=2.0, mode=SlewMode.EXPONENTIAL)
        sl._current = 1.0  # prime state

        out = sl.render(0, 10).data[:, 0]
        # Monotonically decreasing
        assert np.all(np.diff(out) <= 0)
        # Never goes below target
        assert np.all(out >= -1e-6)


# ---------------------------------------------------------------------------
# PE rate input tests
# ---------------------------------------------------------------------------


class TestSlewLimiterPEWithPERate:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(10)
        yield
        pg.set_sample_rate(44100)

    def test_constant_pe_rate_matches_scalar_rate(self):
        """ConstantPE(rate) produces the same output as scalar rate."""
        src = pg.ConstantPE(1.0)
        sl_scalar = SlewLimiterPE(src, rate=1.0)
        sl_pe = SlewLimiterPE(src, rate=pg.ConstantPE(1.0))

        out_scalar = sl_scalar.render(0, 6).data[:, 0]
        out_pe = sl_pe.render(0, 6).data[:, 0]
        np.testing.assert_allclose(out_pe, out_scalar, atol=1e-5)

    def test_pe_rate_inputs_includes_rate_pe(self):
        """inputs() must include the rate PE when rate is a PE."""
        src = pg.ConstantPE(1.0)
        rate_pe = pg.ConstantPE(2.0)
        sl = SlewLimiterPE(src, rate=rate_pe)
        assert rate_pe in sl.inputs()
        assert src in sl.inputs()

    def test_dynamic_rate_changes_tracking_speed(self):
        """A faster rate PE causes the output to track the source more quickly."""
        from pygmu2.piecewise_pe import PiecewisePE, TransitionType
        from pygmu2.extent import ExtendMode

        src = pg.ConstantPE(1.0)

        # Slow rate for first 5 samples, then fast rate for next 5
        # At sr=10: slow=1 u/s → 0.1/sample, fast=10 u/s → 1.0/sample
        rate_curve = PiecewisePE(
            [(0, 1.0), (5, 10.0)],
            transition_type=TransitionType.STEP,
            extend_mode=ExtendMode.HOLD_LAST,
        )
        sl = SlewLimiterPE(src, rate=rate_curve)

        out = sl.render(0, 10).data[:, 0]

        # First 5 samples: slow ramp
        np.testing.assert_allclose(out[:5], [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-5)
        # Next 5 samples: fast ramp — should reach 1.0 immediately
        np.testing.assert_allclose(out[5:], [1.0, 1.0, 1.0, 1.0, 1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Integration: composed stepped random LFO
# ---------------------------------------------------------------------------


class TestSlewLimiterPEComposed:

    def test_slew_smooths_stepped_signal(self):
        """SlewLimiter on a stepped S&H source produces a smooth ramp."""
        from pygmu2.hold_pe import HoldPE

        pg.set_sample_rate(100)

        # Stepped source: NoisePE with fixed seed triggered at 10 Hz
        src = pg.NoisePE(min_value=0.0, max_value=1.0, seed=7)
        trig = pg.GateToTriggerPE(pg.PeriodicGatePE(frequency=10.0))  # every 10 samples
        stepped = HoldPE(src, trig)

        # Slew-limit the steps (rate = 5 units/s → 0.05/sample)
        slewed = SlewLimiterPE(stepped, rate=5.0)

        # Start the whole graph (the derived trigger is stateful and
        # needs lifecycle, unlike the stateless trigger it replaced)
        renderer = pg.NullRenderer(sample_rate=100)
        renderer.set_source(slewed)
        renderer.start()
        out = slewed.render(0, 100).data[:, 0]

        # The output should be smooth: max difference between consecutive samples
        # is at most 5/100 = 0.05 (the slew limit)
        diffs = np.abs(np.diff(out.astype(np.float64)))
        assert np.all(
            diffs <= 0.05 + 1e-6
        ), f"Max diff {diffs.max():.6f} exceeds slew limit 0.05"

        pg.set_sample_rate(44100)
