"""
Tests for SignalToGatePE (Schmitt-trigger analog-to-gate converter).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

import pygmu2 as pg
from pygmu2.signal_to_gate_pe import SignalToGatePE


def make_source(samples):
    """Build an ArrayPE from a list of float values."""
    data = np.array(samples, dtype=np.float32)
    return pg.ArrayPE(data)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSignalToGatePEConstruction:

    def test_is_not_pure(self):
        pe = SignalToGatePE(pg.ConstantPE(0.0))
        assert pe.is_pure() is False

    def test_channel_count(self):
        pe = SignalToGatePE(pg.ConstantPE(0.0))
        assert pe.channel_count() == 1

    def test_inputs_contains_source(self):
        source = pg.ConstantPE(0.5)
        pe = SignalToGatePE(source)
        assert source in pe.inputs()
        assert len(pe.inputs()) == 1

    def test_repr(self):
        source = pg.ConstantPE(0.5)
        pe = SignalToGatePE(source, low_threshold=0.1, high_threshold=0.9,
                            hysteresis=0.05, holdoff_time=0.01)
        r = repr(pe)
        assert "SignalToGatePE" in r
        assert "0.1" in r
        assert "0.9" in r
        assert "0.05" in r
        assert "0.01" in r


# ---------------------------------------------------------------------------
# Output shape and value constraints
# ---------------------------------------------------------------------------

class TestSignalToGatePEShape:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(1000)
        yield
        pg.set_sample_rate(44100)

    def test_output_shape_mono(self):
        pe = SignalToGatePE(pg.ConstantPE(0.0))
        pe.on_start()
        assert pe.render(0, 20).data.shape == (20, 1)

    def test_values_exactly_zero_or_one(self):
        """GateSignal constraint: values must be exactly 0.0 or 1.0."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 1000)).astype(np.float32)
        pe = SignalToGatePE(make_source(signal), low_threshold=-0.5,
                            high_threshold=0.5)
        pe.on_start()
        out = pe.render(0, 1000).data[:, 0]
        assert set(out.tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------

class TestSignalToGatePEThresholds:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(1000)
        yield
        pg.set_sample_rate(44100)

    def test_gate_opens_above_high_threshold(self):
        """Signal rising strictly above high_threshold opens the gate."""
        source = make_source([0.5, 0.5, 1.1, 1.1])
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0)
        pe.on_start()
        out = pe.render(0, 4).data[:, 0]
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 1.0])

    def test_gate_does_not_open_at_exact_threshold(self):
        """Strict comparison: signal == high_threshold does NOT open gate."""
        source = make_source([1.0, 1.0, 1.0])
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0)
        pe.on_start()
        out = pe.render(0, 3).data[:, 0]
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

    def test_gate_closes_below_low_threshold(self):
        """Signal falling strictly below low_threshold closes the gate."""
        # Open gate, then drop below 0.0 to close it.
        source = make_source([1.1, 1.1, -0.1, -0.1])
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0)
        pe.on_start()
        out = pe.render(0, 4).data[:, 0]
        np.testing.assert_array_equal(out, [1.0, 1.0, 0.0, 0.0])

    def test_gate_does_not_close_at_exact_low_threshold(self):
        """Strict comparison: signal == low_threshold does NOT close gate."""
        source = make_source([1.1, 0.0, 0.0])
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0)
        pe.on_start()
        out = pe.render(0, 3).data[:, 0]
        np.testing.assert_array_equal(out, [1.0, 1.0, 1.0])

    def test_gate_stays_closed_between_thresholds(self):
        """Signal in deadband [low, high] never opens a closed gate."""
        source = make_source([0.3, 0.5, 0.7, 0.9])
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0)
        pe.on_start()
        out = pe.render(0, 4).data[:, 0]
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 0.0])

    def test_gate_stays_open_between_thresholds(self):
        """Signal in deadband [low, high] does not close an open gate."""
        source = make_source([1.1, 0.3, 0.5, 0.7])
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0)
        pe.on_start()
        out = pe.render(0, 4).data[:, 0]
        np.testing.assert_array_equal(out, [1.0, 1.0, 1.0, 1.0])

    def test_multiple_open_close_cycles(self):
        """Gate can open and close repeatedly."""
        source = make_source([1.1, -0.1, 1.1, -0.1])
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0)
        pe.on_start()
        out = pe.render(0, 4).data[:, 0]
        np.testing.assert_array_equal(out, [1.0, 0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------

class TestSignalToGatePEHysteresis:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(1000)
        yield
        pg.set_sample_rate(44100)

    def test_hysteresis_raises_effective_open_threshold(self):
        """With hysteresis=0.1, effective open threshold = high_threshold + 0.1.
        Signal at high_threshold + 0.05 (below effective) should not open gate."""
        # high=0.5, hys=0.1 → effective open = 0.6
        source = make_source([0.55, 0.55, 0.65, 0.65])
        pe = SignalToGatePE(source, low_threshold=0.2, high_threshold=0.5,
                            hysteresis=0.1)
        pe.on_start()
        out = pe.render(0, 4).data[:, 0]
        # 0.55 < 0.6 → gate stays closed; 0.65 > 0.6 → gate opens
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0, 1.0])

    def test_hysteresis_lowers_effective_close_threshold(self):
        """With hysteresis=0.1, effective close threshold = low_threshold - 0.1.
        Signal between low and low-hysteresis should not close gate."""
        # low=0.2, hys=0.1 → effective close = 0.1
        # Once open, signal at 0.15 is above 0.1 → gate stays open
        # Signal at 0.05 is below 0.1 → gate closes
        source = make_source([0.65, 0.15, 0.15, 0.05, 0.05])
        pe = SignalToGatePE(source, low_threshold=0.2, high_threshold=0.5,
                            hysteresis=0.1)
        pe.on_start()
        out = pe.render(0, 5).data[:, 0]
        # i=0: 0.65 > 0.6 → opens gate
        # i=1,2: 0.15 not < 0.1 → stays open
        # i=3,4: 0.05 < 0.1 → closes gate
        np.testing.assert_array_equal(out, [1.0, 1.0, 1.0, 0.0, 0.0])

    def test_zero_hysteresis_is_default(self):
        """Default hysteresis=0 matches a plain two-threshold Schmitt trigger."""
        source = make_source([1.1, -0.1, 1.1])
        pe_hys = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0,
                                hysteresis=0.0)
        pe_hys.on_start()
        out = pe_hys.render(0, 3).data[:, 0]
        np.testing.assert_array_equal(out, [1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Holdoff
# ---------------------------------------------------------------------------

class TestSignalToGatePEHoldoff:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(1000)
        yield
        pg.set_sample_rate(44100)

    def test_holdoff_suppresses_immediate_close(self):
        """Signal drops below low_threshold within holdoff window; gate stays open.

        holdoff_time=0.005 s at 1000 Hz → holdoff_count = 5 samples.
        - i=0: 1.1 > 1.0 → gate opens, holdoff=5
        - i=1..5: holdoff counting (5→4→3→2→1→0), gate stays open
        - i=6: holdoff=0, -0.1 < 0.0 → gate closes
        """
        signal = [1.1] + [-0.1] * 20
        source = make_source(signal)
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0,
                            holdoff_time=0.005)
        pe.on_start()
        out = pe.render(0, 21).data[:, 0]

        assert out[0] == 1.0              # gate opened
        assert all(out[1:6] == 1.0)       # samples 1-5: holdoff active
        assert out[6] == 0.0              # sample 6: holdoff expired, gate closes

    def test_holdoff_suppresses_immediate_open(self):
        """After close transition, holdoff also prevents immediate re-open.

        - i=0: 1.1 > 1.0 → gate opens, holdoff=5
        - i=1..5: holdoff counts to 0
        - i=6: -0.1 < 0.0 → gate closes, holdoff=5
        - i=7..11: holdoff counts to 0, gate stays closed (even with 1.1)
        - i=12: holdoff=0, 1.1 > 1.0 → gate opens again
        """
        signal = ([1.1] +          # i=0: open
                  [-0.1] * 5 +     # i=1-5: holdoff counting
                  [-0.1] +         # i=6: close
                  [1.1] * 5 +      # i=7-11: holdoff counting, ignored
                  [1.1] * 5)       # i=12-16: holdoff expired, re-opens
        source = make_source(signal)
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0,
                            holdoff_time=0.005)
        pe.on_start()
        out = pe.render(0, 17).data[:, 0]

        assert out[0] == 1.0              # gate opened at i=0
        assert all(out[1:6] == 1.0)       # holdoff active (still open)
        assert out[6] == 0.0              # gate closes at i=6
        assert all(out[7:12] == 0.0)      # holdoff active (stays closed)
        assert out[12] == 1.0             # gate re-opens at i=12

    def test_zero_holdoff_allows_immediate_retrigger(self):
        """Default holdoff_time=0 → transitions happen every sample."""
        source = make_source([1.1, -0.1, 1.1, -0.1])
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0,
                            holdoff_time=0.0)
        pe.on_start()
        out = pe.render(0, 4).data[:, 0]
        np.testing.assert_array_equal(out, [1.0, 0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# State persistence and lifecycle
# ---------------------------------------------------------------------------

class TestSignalToGatePEState:

    @pytest.fixture(autouse=True)
    def _sr(self):
        pg.set_sample_rate(1000)
        yield
        pg.set_sample_rate(44100)

    def test_state_persists_across_renders(self):
        """Gate open in chunk 1 remains open in chunk 2 (signal in deadband)."""
        source = make_source([1.1, 1.1, 0.5, 0.5, 0.5, 0.5])
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0)
        pe.on_start()
        out1 = pe.render(0, 2).data[:, 0]  # opens gate
        out2 = pe.render(2, 4).data[:, 0]  # 0.5 in deadband → stays open
        np.testing.assert_array_equal(out1, [1.0, 1.0])
        np.testing.assert_array_equal(out2, [1.0, 1.0, 1.0, 1.0])

    def test_on_start_resets_gate_to_closed(self):
        """on_start() resets gate state to closed regardless of previous state."""
        source = make_source([1.1, 1.1])
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0)
        pe.on_start()
        pe.render(0, 2)  # opens gate

        # Re-start: gate should be back to closed
        source2 = make_source([0.5, 0.5])
        pe2 = SignalToGatePE(source2, high_threshold=1.0, low_threshold=0.0)
        pe2.on_start()
        out = pe2.render(0, 2).data[:, 0]
        np.testing.assert_array_equal(out, [0.0, 0.0])

    def test_on_start_resets_holdoff(self):
        """on_start() clears any in-progress holdoff from a previous run."""
        source = make_source([1.1] + [-0.1] * 3)
        pe = SignalToGatePE(source, high_threshold=1.0, low_threshold=0.0,
                            holdoff_time=0.010)
        pe.on_start()
        pe.render(0, 4)  # opens gate; holdoff_remaining > 0 after this chunk

        # Fresh start: gate is closed, holdoff_remaining is 0
        source2 = make_source([-0.5, -0.5])
        pe2 = SignalToGatePE(source2, high_threshold=1.0, low_threshold=0.0,
                             holdoff_time=0.010)
        pe2.on_start()
        out = pe2.render(0, 2).data[:, 0]
        np.testing.assert_array_equal(out, [0.0, 0.0])


# ---------------------------------------------------------------------------
# Extent
# ---------------------------------------------------------------------------

class TestSignalToGatePEExtent:

    def test_extent_matches_source(self):
        source = pg.CropPE(pg.ConstantPE(1.0), start=100, duration=100)
        pe = SignalToGatePE(source)
        assert pe.extent() == pg.Extent(100, 200)

    def test_infinite_source_gives_infinite_extent(self):
        pe = SignalToGatePE(pg.ConstantPE(0.5))
        ext = pe.extent()
        assert ext.start is None
        assert ext.end is None
