"""
Tests for RandomTriggerPE (Poisson-process trigger generator).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.random_trigger_pe import RandomTriggerPE

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRandomTriggerPEConstruction:

    def test_default_construction(self):
        rt = RandomTriggerPE()
        assert rt.stateful
        assert rt.channel_count() == 1

    def test_inputs_empty_for_scalar_rate(self):
        assert RandomTriggerPE(rate=5.0).inputs() == []

    def test_inputs_nonempty_for_pe_rate(self):
        rate_pe = pg.ConstantPE(5.0)
        rt = RandomTriggerPE(rate=rate_pe)
        assert rate_pe in rt.inputs()

    def test_seed_stored(self):
        assert RandomTriggerPE(seed=42)._seed == 42

    def test_repr_contains_rate(self):
        assert "7.0" in repr(RandomTriggerPE(rate=7.0))

    def test_repr_contains_seed(self):
        assert "42" in repr(RandomTriggerPE(seed=42))


# ---------------------------------------------------------------------------
# Output values and shape
# ---------------------------------------------------------------------------


class TestRandomTriggerPEOutput:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_output_shape_is_mono(self):
        rt = RandomTriggerPE()
        rt.on_start()
        assert rt.render(0, 50).data.shape == (50, 1)

    def test_zero_duration(self):
        rt = RandomTriggerPE()
        rt.on_start()
        assert rt.render(0, 0).data.shape == (0, 1)

    def test_values_are_zero_or_one(self):
        """Output must be exactly 0 or 1 (integer-valued trigger)."""
        rt = RandomTriggerPE(rate=10.0, seed=0)
        rt.on_start()
        out = rt.render(0, 500).data[:, 0]
        assert set(out.tolist()).issubset({0.0, 1.0})

    def test_triggers_occur(self):
        """At rate 10 in sr=100 there should be some +1 events."""
        rt = RandomTriggerPE(rate=10.0, seed=1)
        rt.on_start()
        out = rt.render(0, 500).data[:, 0]
        assert out.sum() > 0, "No trigger events fired"

    def test_zero_rate_no_triggers(self):
        """Rate 0 → p=0 → no events ever."""
        rt = RandomTriggerPE(rate=0.0, seed=2)
        rt.on_start()
        out = rt.render(0, 500).data[:, 0]
        assert out.sum() == 0.0

    def test_low_rate_fewer_events(self):
        """Low rate produces fewer events than high rate."""
        rt_low = RandomTriggerPE(rate=1.0, seed=3)
        rt_high = RandomTriggerPE(rate=50.0, seed=3)
        rt_low.on_start()
        rt_high.on_start()
        n = 2000
        low_count = int(rt_low.render(0, n).data[:, 0].sum())
        high_count = int(rt_high.render(0, n).data[:, 0].sum())
        assert low_count < high_count, f"low={low_count}, high={high_count}"

    def test_event_rate_approximately_correct(self):
        """Observed event rate should be close to rate/sr."""
        sr = 100
        rate = 10.0
        n = 10_000
        rt = RandomTriggerPE(rate=rate, seed=7)
        rt.on_start()
        count = int(rt.render(0, n).data[:, 0].sum())
        expected = n * rate / sr
        assert (
            expected * 0.5 < count < expected * 1.5
        ), f"count={count}, expected≈{expected:.0f}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestRandomTriggerPEReproducibility:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_same_seed_same_output(self):
        rt1 = RandomTriggerPE(rate=10.0, seed=99)
        rt2 = RandomTriggerPE(rate=10.0, seed=99)
        rt1.on_start()
        rt2.on_start()
        np.testing.assert_array_equal(rt1.render(0, 200).data, rt2.render(0, 200).data)

    def test_on_start_replays_sequence(self):
        rt = RandomTriggerPE(rate=10.0, seed=7)
        rt.on_start()
        out1 = rt.render(0, 200).data.copy()
        rt.on_start()
        out2 = rt.render(0, 200).data
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# PE rate
# ---------------------------------------------------------------------------


class TestRandomTriggerPEPERate:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_pe_rate_matches_scalar(self):
        rt_s = RandomTriggerPE(rate=5.0, seed=6)
        rt_p = RandomTriggerPE(rate=pg.ConstantPE(5.0), seed=6)
        rt_s.on_start()
        rt_p.on_start()
        np.testing.assert_array_equal(
            rt_s.render(0, 200).data, rt_p.render(0, 200).data
        )
