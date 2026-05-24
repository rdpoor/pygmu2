"""
Tests for DecayingSinePE.
Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""
import pytest
import numpy as np
from pygmu2 import (
    DecayingSinePE,
    NullRenderer,
    Extent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100
FREQUENCY = 440.0


def _renderer():
    return NullRenderer(sample_rate=SAMPLE_RATE)


def _analytical(n_samples, frequency, amplitude, rho, sample_rate):
    """Ground-truth: A · rho^n · sin(2π·f·n / sr) for n = 0, 1, ..., n_samples-1."""
    n = np.arange(n_samples, dtype=np.float64)
    omega = 2.0 * np.pi * frequency / sample_rate
    return (amplitude * rho ** n * np.sin(omega * n)).astype(np.float32)


# ---------------------------------------------------------------------------
# Basic construction and properties
# ---------------------------------------------------------------------------

class TestDecayingSinePEBasics:
    """Test basic DecayingSinePE creation and properties."""

    def test_create_with_duration(self):
        pe = DecayingSinePE(frequency=440.0, duration=2.0)
        assert pe._frequency == 440.0
        assert pe._amplitude == 0.3
        assert pe._duration_seconds == 2.0
        assert pe._rho_param is None
        assert pe.channel_count() == 1

    def test_create_with_rho(self):
        pe = DecayingSinePE(frequency=440.0, rho=0.9995)
        assert pe._frequency == 440.0
        assert pe._rho_param == pytest.approx(0.9995)
        assert pe._duration_seconds is None

    def test_create_with_all_params(self):
        pe = DecayingSinePE(
            frequency=220.0,
            amplitude=0.5,
            rho=0.999,
            channels=2,
        )
        assert pe._frequency == 220.0
        assert pe._amplitude == 0.5
        assert pe._rho_param == pytest.approx(0.999)
        assert pe.channel_count() == 2

    # --- invalid construction ---

    def test_invalid_frequency_zero(self):
        with pytest.raises(ValueError, match="frequency must be positive"):
            DecayingSinePE(frequency=0, duration=1.0)

    def test_invalid_frequency_negative(self):
        with pytest.raises(ValueError, match="frequency must be positive"):
            DecayingSinePE(frequency=-100, duration=1.0)

    def test_invalid_amplitude_zero(self):
        with pytest.raises(ValueError, match="amplitude must be positive"):
            DecayingSinePE(frequency=440, amplitude=0, duration=1.0)

    def test_invalid_amplitude_negative(self):
        with pytest.raises(ValueError, match="amplitude must be positive"):
            DecayingSinePE(frequency=440, amplitude=-0.3, duration=1.0)

    def test_invalid_neither_duration_nor_rho(self):
        with pytest.raises(ValueError, match="supply exactly one"):
            DecayingSinePE(frequency=440)

    def test_invalid_both_duration_and_rho(self):
        with pytest.raises(ValueError, match="supply exactly one"):
            DecayingSinePE(frequency=440, duration=1.0, rho=0.999)

    def test_invalid_rho_zero(self):
        with pytest.raises(ValueError, match="rho must be in"):
            DecayingSinePE(frequency=440, rho=0.0)

    def test_invalid_rho_one(self):
        """rho == 1.0 means the tone never decays; treat as out of range."""
        with pytest.raises(ValueError, match="rho must be in"):
            DecayingSinePE(frequency=440, rho=1.0)

    def test_invalid_rho_above_one(self):
        with pytest.raises(ValueError, match="rho must be in"):
            DecayingSinePE(frequency=440, rho=1.1)

    def test_invalid_duration_nonpositive(self):
        with pytest.raises(ValueError, match="duration must be positive"):
            DecayingSinePE(frequency=440, duration=0.0)
        with pytest.raises(ValueError, match="duration must be positive"):
            DecayingSinePE(frequency=440, duration=-1.0)

    # --- metadata ---

    def test_extent_infinite(self):
        pe = DecayingSinePE(frequency=440.0, duration=1.0)
        renderer = _renderer()
        renderer.set_source(pe)
        ext = pe.extent()
        assert ext.start == 0
        assert ext.end is None
        assert ext.duration is None

    def test_inputs_empty(self):
        pe = DecayingSinePE(frequency=440, duration=1.0)
        assert pe.inputs() == []

    def test_is_impure(self):
        """DecayingSinePE maintains recurrence state; requires contiguous requests."""
        pe = DecayingSinePE(frequency=440.0, duration=1.0)
        assert pe.is_pure() is False

    def test_channel_count_default(self):
        pe = DecayingSinePE(frequency=440.0, duration=1.0)
        assert pe.channel_count() == 1

    def test_channel_count_multichannel(self):
        pe = DecayingSinePE(frequency=440.0, duration=1.0, channels=4)
        assert pe.channel_count() == 4

    # --- repr ---

    def test_repr_duration(self):
        pe = DecayingSinePE(frequency=440.0, amplitude=0.3, duration=2.0)
        r = repr(pe)
        assert "DecayingSinePE" in r
        assert "440" in r
        assert "2.0" in r

    def test_repr_rho(self):
        pe = DecayingSinePE(frequency=440.0, rho=0.9995)
        r = repr(pe)
        assert "DecayingSinePE" in r
        assert "440" in r
        assert "0.9995" in r


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestDecayingSinePERender:
    """Test DecayingSinePE rendering."""

    def setup_method(self):
        self.renderer = _renderer()

    def test_render_returns_snippet(self):
        pe = DecayingSinePE(frequency=440.0, duration=1.0)
        self.renderer.set_source(pe)
        snippet = pe.render(0, 2000)
        assert snippet.start == 0
        assert snippet.duration == 2000
        assert snippet.channels == 1
        assert snippet.data.shape == (2000, 1)

    def test_render_dtype_float32(self):
        pe = DecayingSinePE(frequency=440.0, duration=1.0)
        self.renderer.set_source(pe)
        snippet = pe.render(0, 1000)
        assert snippet.data.dtype == np.float32

    def test_render_stereo_shape(self):
        pe = DecayingSinePE(frequency=440.0, duration=1.0, channels=2)
        self.renderer.set_source(pe)
        snippet = pe.render(0, 1000)
        assert snippet.data.shape == (1000, 2)

    def test_render_stereo_channels_identical(self):
        """Sine is broadcast identically across all channels."""
        pe = DecayingSinePE(frequency=440.0, duration=1.0, channels=3)
        self.renderer.set_source(pe)
        snippet = pe.render(0, 1000)
        np.testing.assert_array_equal(snippet.data[:, 0], snippet.data[:, 1])
        np.testing.assert_array_equal(snippet.data[:, 0], snippet.data[:, 2])

    def test_render_zero_duration(self):
        pe = DecayingSinePE(frequency=440.0, duration=1.0)
        self.renderer.set_source(pe)
        snippet = pe.render(0, 0)
        assert snippet.duration == 0

    def test_render_negative_start_zeros_before_origin(self):
        """Samples before t=0 must be zero; samples at/after t=0 non-zero."""
        pe = DecayingSinePE(frequency=440.0, duration=1.0)
        self.renderer.set_source(pe)
        snippet = pe.render(-200, 5000)
        assert snippet.start == -200
        assert snippet.duration == 5000
        np.testing.assert_array_almost_equal(snippet.data[:200], 0.0)
        assert np.any(np.abs(snippet.data[200:]) > 1e-4)

    def test_render_contiguous_extends_stream(self):
        """Second contiguous call continues the recurrence correctly."""
        pe = DecayingSinePE(frequency=440.0, duration=1.0)
        self.renderer.set_source(pe)
        pe.render(0, 1000)
        snippet = pe.render(1000, 2000)
        assert snippet.start == 1000
        assert snippet.duration == 2000
        assert np.any(np.abs(snippet.data) > 1e-6)

    def test_render_contiguous_matches_single_render(self):
        """Splitting a render into two contiguous calls gives the same output."""
        rho = DecayingSinePE.rho_for_decay_db(2.0, SAMPLE_RATE)
        pe_single = DecayingSinePE(frequency=FREQUENCY, rho=rho)
        pe_split = DecayingSinePE(frequency=FREQUENCY, rho=rho)
        self.renderer.set_source(pe_single)
        full = pe_single.render(0, 4000)
        self.renderer.set_source(pe_split)
        part1 = pe_split.render(0, 2000)
        part2 = pe_split.render(2000, 2000)
        np.testing.assert_array_almost_equal(
            full.data[:2000], part1.data, decimal=5
        )
        np.testing.assert_array_almost_equal(
            full.data[2000:4000], part2.data, decimal=5
        )

    # --- correctness of the recurrence ---

    def test_recurrence_matches_analytical_short(self):
        """Output matches A · rho^n · sin(n·ω) within float32 precision."""
        rho = 0.9995
        amplitude = 0.5
        n_samples = 500
        pe = DecayingSinePE(frequency=FREQUENCY, amplitude=amplitude, rho=rho)
        self.renderer.set_source(pe)
        snippet = pe.render(0, n_samples)
        expected = _analytical(n_samples, FREQUENCY, amplitude, rho, SAMPLE_RATE)
        np.testing.assert_allclose(
            snippet.data[:, 0], expected, atol=5e-5,
            err_msg="Recurrence output diverges from analytical formula"
        )

    def test_recurrence_matches_analytical_long(self):
        """Recurrence stays accurate over a longer window (float32 drift check)."""
        rho = DecayingSinePE.rho_for_decay_db(3.0, SAMPLE_RATE)
        n_samples = SAMPLE_RATE * 2  # 2 seconds
        pe = DecayingSinePE(frequency=FREQUENCY, rho=rho)
        self.renderer.set_source(pe)
        snippet = pe.render(0, n_samples)
        expected = _analytical(n_samples, FREQUENCY, 0.3, rho, SAMPLE_RATE)
        np.testing.assert_allclose(
            snippet.data[:, 0], expected, atol=1e-3,
            err_msg="Recurrence drifts from analytical over long window"
        )

    def test_zero_crossings_near_frequency(self):
        """Zero-crossing rate of the output matches the expected sine frequency."""
        rho = 0.9999  # very slow decay so amplitude stays roughly constant
        pe = DecayingSinePE(frequency=FREQUENCY, rho=rho)
        self.renderer.set_source(pe)
        n = SAMPLE_RATE  # 1 second
        snippet = pe.render(0, n)
        x = snippet.data[:, 0]
        sign_changes = np.sum(np.diff(np.sign(x)) != 0)
        expected_crossings = 2 * FREQUENCY  # two per cycle
        assert abs(sign_changes - expected_crossings) < 0.05 * expected_crossings, (
            f"Zero-crossing count {sign_changes} differs from expected {expected_crossings}"
        )

    def test_amplitude_scales_peak(self):
        """Peak of the output is proportional to the amplitude parameter."""
        rho = 0.9999
        pe1 = DecayingSinePE(frequency=FREQUENCY, amplitude=0.3, rho=rho)
        pe2 = DecayingSinePE(frequency=FREQUENCY, amplitude=0.6, rho=rho)
        self.renderer.set_source(pe1)
        s1 = pe1.render(0, 1000)
        self.renderer.set_source(pe2)
        s2 = pe2.render(0, 1000)
        ratio = np.max(np.abs(s2.data)) / (np.max(np.abs(s1.data)) + 1e-9)
        assert ratio == pytest.approx(2.0, abs=0.05)

    def test_high_rho_vs_low_rho_different_sustain(self):
        """Higher rho → signal sustains longer."""
        high_rho = DecayingSinePE(frequency=330.0, rho=0.9999)
        low_rho = DecayingSinePE(frequency=330.0, rho=0.998)
        self.renderer.set_source(high_rho)
        high_snippet = high_rho.render(0, SAMPLE_RATE)
        self.renderer.set_source(low_rho)
        low_snippet = low_rho.render(0, SAMPLE_RATE)
        start = SAMPLE_RATE // 2
        high_rms = np.sqrt(np.mean(high_snippet.data[start:] ** 2))
        low_rms = np.sqrt(np.mean(low_snippet.data[start:] ** 2))
        assert not np.allclose(high_snippet.data, low_snippet.data)
        assert high_rms > low_rms, "Higher rho must sustain longer"

    def test_different_frequencies_different_pitch(self):
        """Two instances at different frequencies produce distinguishable output."""
        pe_440 = DecayingSinePE(frequency=440.0, rho=0.9999)
        pe_880 = DecayingSinePE(frequency=880.0, rho=0.9999)
        self.renderer.set_source(pe_440)
        s_440 = pe_440.render(0, 2000)
        self.renderer.set_source(pe_880)
        s_880 = pe_880.render(0, 2000)
        assert not np.allclose(s_440.data, s_880.data)


# ---------------------------------------------------------------------------
# DecayingSinePE.rho_for_decay_db static method
# ---------------------------------------------------------------------------

class TestRhoForDecayDb:
    """Test the DecayingSinePE.rho_for_decay_db static method."""

    def test_formula_correctness(self):
        """rho = 10^(db / (20 · sr · seconds)), exact float equality."""
        sr = SAMPLE_RATE
        def expected(seconds, sr, db=-60.0):
            samples = seconds * sr
            return float(10 ** (db / (20.0 * samples)))

        for seconds in (0.5, 1.0, 2.0):
            assert DecayingSinePE.rho_for_decay_db(seconds, sr) == pytest.approx(
                expected(seconds, sr), rel=1e-10
            )

    def test_longer_duration_higher_rho(self):
        """Longer decay time → rho closer to 1 (slower per-sample decay)."""
        rho_short = DecayingSinePE.rho_for_decay_db(0.5, SAMPLE_RATE)
        rho_long = DecayingSinePE.rho_for_decay_db(2.0, SAMPLE_RATE)
        assert rho_long > rho_short

    def test_higher_sample_rate_higher_rho(self):
        """Higher sample rate → more samples → rho must be closer to 1."""
        rho_44k = DecayingSinePE.rho_for_decay_db(1.0, 44100)
        rho_96k = DecayingSinePE.rho_for_decay_db(1.0, 96000)
        assert rho_96k > rho_44k

    def test_db_parameter(self):
        """Shallower dB target → higher rho."""
        rho_60 = DecayingSinePE.rho_for_decay_db(1.0, SAMPLE_RATE, db=-60.0)
        rho_30 = DecayingSinePE.rho_for_decay_db(1.0, SAMPLE_RATE, db=-30.0)
        assert rho_30 > rho_60

    def test_result_less_than_one(self):
        """rho must always be < 1 for any valid input."""
        for seconds in (0.1, 0.5, 1.0, 5.0):
            rho = DecayingSinePE.rho_for_decay_db(seconds, SAMPLE_RATE)
            assert rho < 1.0, f"rho={rho} not < 1 for seconds={seconds}"

    def test_result_positive(self):
        rho = DecayingSinePE.rho_for_decay_db(1.0, SAMPLE_RATE)
        assert rho > 0.0

    def test_invalid_seconds_zero(self):
        with pytest.raises(ValueError, match="seconds \\* sample_rate must be positive"):
            DecayingSinePE.rho_for_decay_db(0, SAMPLE_RATE)

    def test_invalid_seconds_negative(self):
        with pytest.raises(ValueError, match="seconds \\* sample_rate must be positive"):
            DecayingSinePE.rho_for_decay_db(-1.0, SAMPLE_RATE)

    def test_empirical_decay_matches_target(self):
        """Signal rendered with rho_for_decay_db actually decays ~60 dB in `seconds`."""
        seconds = 1.0
        rho = DecayingSinePE.rho_for_decay_db(seconds, SAMPLE_RATE)
        pe = DecayingSinePE(frequency=FREQUENCY, rho=rho)
        renderer = _renderer()
        renderer.set_source(pe)
        n = int(seconds * SAMPLE_RATE)
        snippet = pe.render(0, n)
        x = snippet.data[:, 0]
        window = SAMPLE_RATE // 4
        rms_early = np.sqrt(np.mean(x[:window] ** 2)) + 1e-12
        rms_late = np.sqrt(np.mean(x[-window:] ** 2)) + 1e-12
        ratio_db = 20 * np.log10(rms_late / rms_early)
        assert ratio_db <= -40, (
            f"Late window should be well below early; got {ratio_db:.1f} dB"
        )
        assert ratio_db >= -80, (
            f"Decay should not far exceed 60 dB target; got {ratio_db:.1f} dB"
        )

    def test_duration_constructor_uses_rho_formula(self):
        """DecayingSinePE(duration=T) is equivalent to DecayingSinePE(rho=rho_for_decay_db(T))."""
        seconds = 1.5
        rho = DecayingSinePE.rho_for_decay_db(seconds, SAMPLE_RATE)
        pe_dur = DecayingSinePE(frequency=FREQUENCY, duration=seconds)
        pe_rho = DecayingSinePE(frequency=FREQUENCY, rho=rho)
        renderer = _renderer()
        renderer.set_source(pe_dur)
        s_dur = pe_dur.render(0, 4000)
        renderer.set_source(pe_rho)
        s_rho = pe_rho.render(0, 4000)
        np.testing.assert_array_almost_equal(s_dur.data, s_rho.data, decimal=6)
