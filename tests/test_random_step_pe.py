"""
Tests for RandomStepPE (Poisson sample-and-hold).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.random_step_pe import RandomStepPE

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRandomStepPEConstruction:

    def test_default_construction(self):
        rs = RandomStepPE()
        assert rs.is_pure() is False
        assert rs.channel_count() == 1

    def test_inputs_empty_for_scalar_rate(self):
        rs = RandomStepPE(rate=10.0)
        assert rs.inputs() == []

    def test_pe_rate_accepted(self):
        rs = RandomStepPE(rate=pg.ConstantPE(5.0))
        assert rs.channel_count() == 1

    def test_inputs_nonempty_for_pe_rate(self):
        rate_pe = pg.ConstantPE(5.0)
        rs = RandomStepPE(rate=rate_pe)
        assert rate_pe in rs.inputs()

    def test_seed_stored(self):
        rs = RandomStepPE(seed=42)
        assert rs._seed == 42

    def test_repr_contains_rate(self):
        r = repr(RandomStepPE(rate=7.0))
        assert "7.0" in r

    def test_repr_contains_seed(self):
        r = repr(RandomStepPE(seed=42))
        assert "42" in r


# ---------------------------------------------------------------------------
# Output range and shape
# ---------------------------------------------------------------------------


class TestRandomStepPERange:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_output_in_unit_range(self):
        rs = RandomStepPE(rate=10.0, seed=0)
        rs.on_start()
        out = rs.render(0, 500).data[:, 0]
        assert np.all(out >= 0.0), f"min={out.min()}"
        assert np.all(out <= 1.0), f"max={out.max()}"

    def test_output_shape_is_mono(self):
        rs = RandomStepPE()
        rs.on_start()
        snip = rs.render(0, 50)
        assert snip.data.shape == (50, 1)

    def test_zero_duration(self):
        rs = RandomStepPE()
        rs.on_start()
        snip = rs.render(0, 0)
        assert snip.data.shape == (0, 1)


# ---------------------------------------------------------------------------
# Step (sample-and-hold) behaviour
# ---------------------------------------------------------------------------


class TestRandomStepPEBehaviour:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_output_holds_between_jumps(self):
        """Between jumps, all samples have identical values (piecewise-constant)."""
        rs = RandomStepPE(rate=10.0, seed=0)
        rs.on_start()
        out = rs.render(0, 200).data[:, 0]
        # Identify run lengths; within each constant run delta == 0
        diffs = np.diff(out.astype(np.float64))
        # There must be at least some zero-diffs (held samples)
        assert np.sum(diffs == 0.0) > 0, "No held samples found — output never constant"

    def test_output_varies(self):
        """Output is not globally constant over a long render."""
        rs = RandomStepPE(rate=10.0, seed=1)
        rs.on_start()
        out = rs.render(0, 500).data[:, 0]
        assert (
            out.max() - out.min() > 0.01
        ), "Output appears constant — no jumps occurred"

    def test_low_rate_fewer_jumps(self):
        """Low rate produces fewer value transitions than high rate."""
        rs_low = RandomStepPE(rate=1.0, seed=5)
        rs_high = RandomStepPE(rate=50.0, seed=5)
        rs_low.on_start()
        rs_high.on_start()

        n = 2000
        out_low = rs_low.render(0, n).data[:, 0]
        out_high = rs_high.render(0, n).data[:, 0]

        jumps_low = int(np.sum(np.diff(out_low.astype(np.float64)) != 0.0))
        jumps_high = int(np.sum(np.diff(out_high.astype(np.float64)) != 0.0))
        assert (
            jumps_low < jumps_high
        ), f"Low-rate jumps ({jumps_low}) not < high-rate jumps ({jumps_high})"

    def test_jump_distribution_is_poisson_like(self):
        """
        Mean jump rate should be approximately rate/sr.

        At sr=100, rate=10 → p=0.1 → expect ~10 jumps per 100 samples.
        We render 10 000 samples and check the observed rate is within
        ±50 % of expected (loose tolerance to avoid flakiness).
        """
        sr = 100
        rate = 10.0
        n = 10_000
        rs = RandomStepPE(rate=rate, seed=7)
        rs.on_start()
        out = rs.render(0, n).data[:, 0]

        jumps = int(np.sum(np.diff(out.astype(np.float64)) != 0.0))
        expected = n * rate / sr
        assert (
            expected * 0.5 < jumps < expected * 1.5
        ), f"Jump count {jumps} far from expected {expected:.1f}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestRandomStepPEReproducibility:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_same_seed_same_output(self):
        """Two instances with the same seed produce identical output."""
        rs1 = RandomStepPE(rate=10.0, seed=99)
        rs2 = RandomStepPE(rate=10.0, seed=99)
        rs1.on_start()
        rs2.on_start()
        out1 = rs1.render(0, 200).data[:, 0]
        out2 = rs2.render(0, 200).data[:, 0]
        np.testing.assert_array_equal(out1, out2)

    def test_on_start_replays_sequence(self):
        """on_start() resets the seeded instance to replay the same sequence."""
        rs = RandomStepPE(rate=10.0, seed=7)
        rs.on_start()
        out1 = rs.render(0, 200).data[:, 0]
        rs.on_start()
        out2 = rs.render(0, 200).data[:, 0]
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# PE rate
# ---------------------------------------------------------------------------


class TestRandomStepPEPERate:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(100)
        yield
        pg.set_sample_rate(44100)

    def test_pe_rate_matches_scalar(self):
        """ConstantPE(v) rate produces identical output to scalar v."""
        rs_scalar = RandomStepPE(rate=5.0, seed=6)
        rs_pe = RandomStepPE(rate=pg.ConstantPE(5.0), seed=6)
        rs_scalar.on_start()
        rs_pe.on_start()
        out_s = rs_scalar.render(0, 200).data[:, 0]
        out_p = rs_pe.render(0, 200).data[:, 0]
        np.testing.assert_array_equal(out_s, out_p)
