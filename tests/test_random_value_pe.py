"""
Tests for RandomValuePE (Poisson-target continuous wandering).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.random_value_pe import RandomValuePE

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRandomValuePEConstruction:

    def test_default_construction(self):
        rv = RandomValuePE()
        assert rv.stateful
        assert rv.channel_count() == 1

    def test_inputs_empty_for_scalar_rate(self):
        rv = RandomValuePE()
        assert rv.inputs() == []

    def test_inputs_nonempty_for_pe_rate(self):
        rate_pe = pg.ConstantPE(5.0)
        rv = RandomValuePE(rate=rate_pe)
        assert rate_pe in rv.inputs()

    def test_pe_rate_accepted(self):
        rv = RandomValuePE(rate=pg.ConstantPE(5.0))
        assert rv.channel_count() == 1

    def test_seed_stored(self):
        rv = RandomValuePE(seed=42)
        assert rv._seed == 42

    def test_repr_contains_rate(self):
        r = repr(RandomValuePE(rate=7.0))
        assert "7.0" in r

    def test_repr_contains_seed(self):
        r = repr(RandomValuePE(seed=42))
        assert "42" in r


# ---------------------------------------------------------------------------
# Output range
# ---------------------------------------------------------------------------


class TestRandomValuePERange:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_output_in_unit_range(self):
        rv = RandomValuePE(rate=10.0, seed=0)
        rv.on_start()
        out = rv.render(0, 200).data[:, 0]
        assert np.all(out >= 0.0 - 1e-5), f"min={out.min()}"
        assert np.all(out <= 1.0 + 1e-5), f"max={out.max()}"

    def test_output_shape_is_mono(self):
        rv = RandomValuePE()
        rv.on_start()
        snip = rv.render(0, 50)
        assert snip.data.shape == (50, 1)

    def test_zero_duration(self):
        rv = RandomValuePE()
        rv.on_start()
        snip = rv.render(0, 0)
        assert snip.data.shape == (0, 1)


# ---------------------------------------------------------------------------
# Wandering behaviour
# ---------------------------------------------------------------------------


class TestRandomValuePEWanderingBehaviour:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_output_varies(self):
        """Output must wander significantly — not stuck near 0.5."""
        rv = RandomValuePE(rate=10.0, seed=1)
        rv.on_start()
        out = rv.render(0, 200).data[:, 0]
        assert out.max() - out.min() > 0.01, "Output appears constant"

    def test_low_rate_slow_variation(self):
        """Low rate → small inter-sample deltas (slow wandering)."""
        rv = RandomValuePE(rate=0.5, seed=2)
        rv.on_start()
        out = rv.render(0, 200).data[:, 0]
        max_delta = float(np.abs(np.diff(out.astype(np.float64))).max())
        # p = 0.5/100 = 0.005; max per-sample step ≤ p * 1.0 = 0.005
        assert max_delta < 0.01, f"Low-rate max delta {max_delta:.6f} is too large"

    def test_high_rate_faster_variation(self):
        """Higher rate → larger inter-sample deltas than low rate."""
        rv_slow = RandomValuePE(rate=1.0, seed=3)
        rv_fast = RandomValuePE(rate=50.0, seed=3)
        rv_slow.on_start()
        rv_fast.on_start()

        out_slow = rv_slow.render(0, 500).data[:, 0]
        out_fast = rv_fast.render(0, 500).data[:, 0]

        max_delta_slow = float(np.abs(np.diff(out_slow.astype(np.float64))).max())
        max_delta_fast = float(np.abs(np.diff(out_fast.astype(np.float64))).max())
        assert (
            max_delta_fast > max_delta_slow
        ), f"Fast rate delta {max_delta_fast:.4f} not > slow rate delta {max_delta_slow:.4f}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestRandomValuePEReproducibility:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_same_seed_same_output(self):
        """Two instances with the same seed produce identical output."""
        rv1 = RandomValuePE(rate=10.0, seed=99)
        rv2 = RandomValuePE(rate=10.0, seed=99)
        rv1.on_start()
        rv2.on_start()
        out1 = rv1.render(0, 100).data[:, 0]
        out2 = rv2.render(0, 100).data[:, 0]
        np.testing.assert_array_equal(out1, out2)

    def test_on_start_replays_sequence(self):
        """on_start() resets the seeded instance to replay the same sequence."""
        rv = RandomValuePE(rate=10.0, seed=7)
        rv.on_start()
        out1 = rv.render(0, 100).data[:, 0]
        rv.on_start()
        out2 = rv.render(0, 100).data[:, 0]
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# PE rate
# ---------------------------------------------------------------------------


class TestRandomValuePEPERate:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_pe_rate_matches_scalar(self):
        """ConstantPE(v) rate produces identical output to scalar v."""
        rv_scalar = RandomValuePE(rate=5.0, seed=6)
        rv_pe = RandomValuePE(rate=pg.ConstantPE(5.0), seed=6)
        rv_scalar.on_start()
        rv_pe.on_start()
        out_s = rv_scalar.render(0, 100).data[:, 0]
        out_p = rv_pe.render(0, 100).data[:, 0]
        np.testing.assert_allclose(out_s, out_p, atol=1e-5)
