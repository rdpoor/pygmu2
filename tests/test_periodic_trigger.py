"""
Tests for PeriodicTrigger.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.periodic_trigger import PeriodicTrigger


# ---------------------------------------------------------------------------
# Construction tests (static float hz)
# ---------------------------------------------------------------------------

class TestPeriodicTriggerConstruction:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(10)
        yield
        pg.set_sample_rate(44100)

    def test_default_hz(self):
        """Default hz=1.0 constructs without error."""
        pt = PeriodicTrigger()
        assert pt.is_pure() is True

    def test_zero_hz_raises(self):
        with pytest.raises(ValueError, match="hz"):
            PeriodicTrigger(hz=0.0)

    def test_negative_hz_raises(self):
        with pytest.raises(ValueError, match="hz"):
            PeriodicTrigger(hz=-1.0)

    def test_is_pure_for_float_hz(self):
        pt = PeriodicTrigger(hz=2.0)
        assert pt.is_pure() is True

    def test_inputs_empty_for_float_hz(self):
        pt = PeriodicTrigger(hz=2.0)
        assert pt.inputs() == []

    def test_is_not_pure_for_pe_hz(self):
        pt = PeriodicTrigger(hz=pg.ConstantPE(2.0))
        assert pt.is_pure() is False

    def test_inputs_contains_pe_hz(self):
        hz_pe = pg.ConstantPE(2.0)
        pt = PeriodicTrigger(hz=hz_pe)
        assert pt.inputs() == [hz_pe]

    def test_channel_count_is_one(self):
        pt = PeriodicTrigger(hz=2.0)
        assert pt.channel_count() == 1


# ---------------------------------------------------------------------------
# Static float hz rendering
# ---------------------------------------------------------------------------

class TestPeriodicTriggerStaticHz:

    @pytest.fixture(autouse=True)
    def _sr(self):
        # sr=10, hz=1 → period=10 samples (trigger every 10 samples)
        pg.set_sample_rate(10)
        yield
        pg.set_sample_rate(44100)

    def test_triggers_at_correct_period(self):
        """At sr=10, hz=1 → period=10; trigger at sample 0, 10, 20, ..."""
        pt = PeriodicTrigger(hz=1.0)
        out = pt.render(0, 30).data[:, 0]
        # Trigger at indices 0, 10, 20
        expected = np.zeros(30, dtype=np.float32)
        expected[[0, 10, 20]] = 1.0
        np.testing.assert_array_equal(out, expected)

    def test_triggers_at_5hz(self):
        """At sr=10, hz=5 → period=2; trigger every 2 samples."""
        pt = PeriodicTrigger(hz=5.0)
        out = pt.render(0, 10).data[:, 0]
        # Trigger at 0, 2, 4, 6, 8
        expected = np.zeros(10, dtype=np.float32)
        expected[[0, 2, 4, 6, 8]] = 1.0
        np.testing.assert_array_equal(out, expected)

    def test_output_shape_is_mono(self):
        pt = PeriodicTrigger(hz=2.0)
        snip = pt.render(0, 8)
        assert snip.data.shape == (8, 1)

    def test_zero_duration(self):
        pt = PeriodicTrigger(hz=1.0)
        snip = pt.render(0, 0)
        assert snip.data.shape == (0, 1)

    def test_amplitude_parameter(self):
        """amplitude controls the value emitted at trigger samples."""
        pt = PeriodicTrigger(hz=5.0, amplitude=3)
        out = pt.render(0, 4).data[:, 0]
        # Trigger at 0, 2
        expected = np.array([3.0, 0.0, 3.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_non_zero_start(self):
        """Rendering from a non-zero start still places triggers correctly."""
        pt = PeriodicTrigger(hz=1.0)  # period=10
        # Trigger at absolute sample 0, 10, 20; render window [8, 22)
        out = pt.render(8, 14).data[:, 0]
        expected = np.zeros(14, dtype=np.float32)
        expected[2] = 1.0   # absolute 10
        expected[12] = 1.0  # absolute 20
        np.testing.assert_array_equal(out, expected)

    def test_pure_renders_are_independent(self):
        """Two separate render calls with same start produce identical output."""
        pt = PeriodicTrigger(hz=2.0)
        out1 = pt.render(0, 10).data[:, 0]
        out2 = pt.render(0, 10).data[:, 0]
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# PE hz rendering (phase accumulator mode)
# ---------------------------------------------------------------------------

class TestPeriodicTriggerPEHz:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(10)
        yield
        pg.set_sample_rate(44100)

    def test_constant_pe_same_rate_as_float_hz(self):
        """ConstantPE(hz) produces triggers at the same rate as float hz.

        Note: the two modes differ by at most 1 sample in absolute phase due to
        the accumulator firing at cycle-completion rather than cycle-start.
        We verify trigger count and spacing, not exact sample positions.
        """
        pt_float = PeriodicTrigger(hz=5.0)
        pt_pe = PeriodicTrigger(hz=pg.ConstantPE(5.0))

        n = 20  # large enough window to see multiple periods
        out_float = pt_float.render(0, n).data[:, 0]
        out_pe = pt_pe.render(0, n).data[:, 0]

        # Same trigger count
        assert np.sum(out_float > 0) == np.sum(out_pe > 0)

        # PE triggers are evenly spaced (period = 2 samples)
        pe_positions = np.where(out_pe > 0)[0]
        assert len(pe_positions) > 1
        spacings = np.diff(pe_positions)
        assert np.all(spacings == spacings[0]), f"Uneven spacings: {spacings}"
        assert spacings[0] == 2  # period = sr/hz = 10/5 = 2

    def test_on_start_resets_phase_accumulator(self):
        """on_start causes PE-mode trigger to restart from initial phase."""
        pt = PeriodicTrigger(hz=pg.ConstantPE(5.0))  # period=2 at sr=10

        out1 = pt.render(0, 4).data[:, 0]
        pt.on_start()
        out2 = pt.render(0, 4).data[:, 0]
        np.testing.assert_array_equal(out1, out2)

    def test_state_accumulates_across_renders(self):
        """Phase accumulator state carries over between sequential renders."""
        pt = PeriodicTrigger(hz=pg.ConstantPE(5.0))  # trigger every 2 samples

        # Render first 4 samples, then next 4 samples
        out1 = pt.render(0, 4).data[:, 0]
        out2 = pt.render(4, 4).data[:, 0]

        # Combined, triggers should appear every 2 samples (period = sr/hz = 10/5 = 2)
        combined = np.concatenate([out1, out2])
        trigger_positions = np.where(combined > 0)[0]
        assert len(trigger_positions) >= 3, "Expected multiple triggers across 8 samples"
        spacings = np.diff(trigger_positions)
        assert np.all(spacings == 2), f"Expected spacing=2, got {spacings}"

    def test_dynamic_hz_changes_trigger_rate(self):
        """When PE hz increases, more triggers appear per unit time."""
        from pygmu2.piecewise_pe import PiecewisePE, TransitionType
        from pygmu2.extent import ExtendMode

        # First 10 samples: hz=1 → period=10, at most 1 trigger
        # Next 10 samples: hz=5 → period=2, up to 5 triggers
        hz_curve = PiecewisePE(
            [(0, 1.0), (10, 5.0)],
            transition_type=TransitionType.STEP,
            extend_mode=ExtendMode.HOLD_LAST,
        )
        pt = PeriodicTrigger(hz=hz_curve)

        out = pt.render(0, 20).data[:, 0]

        triggers_slow = int(np.sum(out[:10] > 0))
        triggers_fast = int(np.sum(out[10:] > 0))

        # Fast half must have more triggers than slow half
        assert triggers_fast > triggers_slow, (
            f"Expected more triggers in fast half; slow={triggers_slow}, fast={triggers_fast}"
        )
        # Fast section should have triggers evenly spaced at period=2
        fast_positions = np.where(out[10:] > 0)[0]
        if len(fast_positions) > 1:
            spacings = np.diff(fast_positions)
            assert np.all(spacings == 2), f"Expected spacing=2 in fast section, got {spacings}"

    def test_inputs_returns_hz_pe(self):
        hz_pe = pg.ConstantPE(2.0)
        pt = PeriodicTrigger(hz=hz_pe)
        assert pt.inputs() == [hz_pe]
