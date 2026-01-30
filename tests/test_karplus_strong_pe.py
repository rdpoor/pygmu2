"""
Tests for KarplusStrongPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    KarplusStrongPE,
    NullRenderer,
    Extent,
    rho_for_decay_db,
)


class TestKarplusStrongPEBasics:
    """Test basic KarplusStrongPE creation and properties."""

    def test_create_defaults(self):
        ks = KarplusStrongPE(frequency=440.0)
        assert ks._frequency == 440.0
        assert ks._rho == 0.996
        assert ks._amplitude == 0.3
        assert ks.channel_count() == 1

    def test_create_with_all_params(self):
        ks = KarplusStrongPE(
            frequency=220.0,
            rho=0.99,
            amplitude=0.5,
            seed=42,
            channels=2,
        )
        assert ks._frequency == 220.0
        assert ks._rho == 0.99
        assert ks._amplitude == 0.5
        assert ks._seed == 42
        assert ks.channel_count() == 2

    def test_invalid_frequency(self):
        with pytest.raises(ValueError, match="frequency must be positive"):
            KarplusStrongPE(frequency=0)
        with pytest.raises(ValueError, match="frequency must be positive"):
            KarplusStrongPE(frequency=-100)

    def test_invalid_rho(self):
        with pytest.raises(ValueError, match="rho must be in"):
            KarplusStrongPE(frequency=440, rho=0)
        with pytest.raises(ValueError, match="rho must be in"):
            KarplusStrongPE(frequency=440, rho=1.5)

    def test_invalid_amplitude(self):
        with pytest.raises(ValueError, match="amplitude must be positive"):
            KarplusStrongPE(frequency=440, amplitude=0)

    def test_extent_infinite(self):
        ks = KarplusStrongPE(frequency=440.0)
        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(ks)
        ext = ks.extent()
        assert ext.start == 0
        assert ext.end is None
        assert ext.duration is None

    def test_inputs_empty(self):
        ks = KarplusStrongPE(frequency=440)
        assert ks.inputs() == []

    def test_repr(self):
        ks = KarplusStrongPE(frequency=440.0, rho=0.99)
        r = repr(ks)
        assert "KarplusStrongPE" in r
        assert "440" in r
        assert "0.99" in r

    def test_is_impure(self):
        """KarplusStrongPE is impure (delay-line/cache state, contiguous requests)."""
        ks = KarplusStrongPE(frequency=440.0)
        assert ks.is_pure() is False

    def test_two_phase_repr(self):
        """Repr includes duration and rho_damping when both provided."""
        ks = KarplusStrongPE(frequency=440.0, rho=0.996, duration=44100, rho_damping=0.95)
        r = repr(ks)
        assert "duration=44100" in r
        assert "rho_damping=0.95" in r


class TestKarplusStrongPERender:
    """Test KarplusStrongPE rendering."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_render_returns_snippet(self):
        ks = KarplusStrongPE(frequency=440.0, seed=123)
        self.renderer.set_source(ks)
        snippet = ks.render(0, 2000)
        assert snippet.start == 0
        assert snippet.duration == 2000
        assert snippet.channels == 1
        assert snippet.data.shape == (2000, 1)

    def test_render_stereo(self):
        ks = KarplusStrongPE(frequency=440.0, channels=2, seed=1)
        self.renderer.set_source(ks)
        snippet = ks.render(0, 1000)
        assert snippet.data.shape == (1000, 2)
        np.testing.assert_array_almost_equal(snippet.data[:, 0], snippet.data[:, 1])

    def test_render_contiguous_extends_stream(self):
        """Contiguous requests stream; second chunk continues from first."""
        ks = KarplusStrongPE(frequency=440.0, seed=99)
        self.renderer.set_source(ks)
        ks.render(0, 1000)
        # Impure PE requires contiguous requests; next start must be 1000
        snippet = ks.render(1000, 2000)
        assert snippet.start == 1000
        assert snippet.duration == 2000
        assert np.any(np.abs(snippet.data) > 0.01)

    def test_render_negative_start_zeros(self):
        ks = KarplusStrongPE(frequency=440.0, seed=7)
        self.renderer.set_source(ks)
        snippet = ks.render(-100, 5000)
        assert snippet.start == -100
        assert snippet.duration == 5000
        np.testing.assert_array_almost_equal(snippet.data[:100], 0.0)
        assert np.any(np.abs(snippet.data[100:2000]) > 0.01)

    def test_seed_reproducibility(self):
        ks1 = KarplusStrongPE(frequency=440.0, seed=42)
        ks2 = KarplusStrongPE(frequency=440.0, seed=42)
        self.renderer.set_source(ks1)
        s1 = ks1.render(0, 5000)
        self.renderer.set_source(ks2)
        s2 = ks2.render(0, 5000)
        np.testing.assert_array_almost_equal(s1.data, s2.data)

    def test_high_rho_vs_low_rho_different_sustain(self):
        """Higher rho = longer sustain."""
        high_rho = KarplusStrongPE(frequency=330.0, rho=0.999, seed=1)
        low_rho = KarplusStrongPE(frequency=330.0, rho=0.98, seed=1)
        self.renderer.set_source(high_rho)
        high_snippet = high_rho.render(0, 44100)
        self.renderer.set_source(low_rho)
        low_snippet = low_rho.render(0, 44100)
        start = int(0.3 * 44100)
        end = int(0.6 * 44100)
        high_rms = np.sqrt(np.mean(high_snippet.data[start:end] ** 2))
        low_rms = np.sqrt(np.mean(low_snippet.data[start:end] ** 2))
        assert not np.allclose(high_snippet.data, low_snippet.data)
        assert high_rms > low_rms, "Higher rho should sustain longer"

    def test_rho_for_decay_db_formula(self):
        """rho_for_decay_db(seconds, frequency, db) returns expected rho values."""
        # rho = 10^(db / (20 * seconds * frequency)); for -60 dB: 10^(-3/(s*f))
        assert rho_for_decay_db(1.0, 440.0, db=-60.0) == pytest.approx(
            10 ** (-3 / (1.0 * 440.0)), rel=1e-10
        )
        assert rho_for_decay_db(2.0, 440.0, db=-60.0) == pytest.approx(
            10 ** (-3 / (2.0 * 440.0)), rel=1e-10
        )
        assert rho_for_decay_db(0.25, 440.0, db=-60.0) == pytest.approx(
            10 ** (-3 / (0.25 * 440.0)), rel=1e-10
        )
        # Higher frequency => higher rho (more periods per second, so each period decays less)
        rho_440 = rho_for_decay_db(1.0, 440.0, db=-60.0)
        rho_220 = rho_for_decay_db(1.0, 220.0, db=-60.0)
        assert rho_440 > rho_220

    def test_rho_for_decay_db_empirical(self):
        """KS with rho from rho_for_decay_db decays ~60 dB over the given duration."""
        sample_rate = 44100
        frequency = 440.0
        seconds = 1.0
        rho = rho_for_decay_db(seconds, frequency, db=-60.0)
        ks = KarplusStrongPE(frequency=frequency, rho=rho, seed=42)
        renderer = NullRenderer(sample_rate=sample_rate)
        renderer.set_source(ks)
        n = int(seconds * sample_rate)
        snippet = ks.render(0, n)
        # Peak/RMS near start vs near end; expect ~60 dB drop
        early = snippet.data[: n // 10]
        late = snippet.data[9 * n // 10 :]
        rms_early = np.sqrt(np.mean(early ** 2)) + 1e-12
        rms_late = np.sqrt(np.mean(late ** 2)) + 1e-12
        ratio_db = 20 * np.log10(rms_late / rms_early)
        assert ratio_db <= -50, "Late window should be at least ~50 dB below early"
        assert ratio_db >= -75, "Decay should not be much more than 60 dB (tolerance)"

    def test_rho_for_decay_db_invalid(self):
        """rho_for_decay_db raises when seconds*frequency <= 0."""
        with pytest.raises(ValueError, match="seconds \\* frequency must be positive"):
            rho_for_decay_db(0, 440.0)
        with pytest.raises(ValueError, match="seconds \\* frequency must be positive"):
            rho_for_decay_db(1.0, 0)

    def test_two_phase_decay(self):
        """With duration and rho_damping, level drops faster after duration."""
        # Sustain 2000 samples with rho=0.996, then rho_damping=0.92
        ks = KarplusStrongPE(
            frequency=440.0, rho=0.996, duration=2000, rho_damping=0.92, seed=42
        )
        self.renderer.set_source(ks)
        snippet = ks.render(0, 8000)
        # RMS in late sustain (samples 1500-2000) vs well into damping (5000-6000)
        sustain_rms = np.sqrt(np.mean(snippet.data[1500:2000] ** 2))
        damping_rms = np.sqrt(np.mean(snippet.data[5000:6000] ** 2))
        assert damping_rms < sustain_rms, "After duration, rho_damping should decay faster"
