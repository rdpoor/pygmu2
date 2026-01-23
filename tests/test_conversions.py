"""
Tests for conversion utility functions.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    pitch_to_freq,
    freq_to_pitch,
    ratio_to_db,
    db_to_ratio,
    semitones_to_ratio,
    ratio_to_semitones,
    samples_to_seconds,
    seconds_to_samples,
)


class TestPitchFreqConversions:
    """Test pitch <-> frequency conversions."""
    
    def test_pitch_to_freq_a4(self):
        """A4 (MIDI 69) should be 440 Hz."""
        assert pitch_to_freq(69) == pytest.approx(440.0)
    
    def test_pitch_to_freq_middle_c(self):
        """Middle C (MIDI 60) should be ~261.63 Hz."""
        assert pitch_to_freq(60) == pytest.approx(261.6256, rel=1e-4)
    
    def test_pitch_to_freq_octave(self):
        """Octave should double frequency."""
        freq_60 = pitch_to_freq(60)
        freq_72 = pitch_to_freq(72)
        assert freq_72 == pytest.approx(freq_60 * 2)
    
    def test_pitch_to_freq_array(self):
        """Should work with arrays."""
        pitches = np.array([60, 64, 67])  # C major chord
        freqs = pitch_to_freq(pitches)
        
        assert len(freqs) == 3
        assert freqs[0] == pytest.approx(261.6256, rel=1e-4)
        assert freqs[1] == pytest.approx(329.6276, rel=1e-4)
        assert freqs[2] == pytest.approx(391.9954, rel=1e-4)
    
    def test_freq_to_pitch_440(self):
        """440 Hz should be MIDI 69 (A4)."""
        assert freq_to_pitch(440.0) == pytest.approx(69.0)
    
    def test_freq_to_pitch_middle_c(self):
        """~261.63 Hz should be MIDI 60 (Middle C)."""
        assert freq_to_pitch(261.6256) == pytest.approx(60.0, abs=0.01)
    
    def test_freq_to_pitch_array(self):
        """Should work with arrays."""
        freqs = np.array([261.626, 329.628, 391.995])
        pitches = freq_to_pitch(freqs)
        
        np.testing.assert_array_almost_equal(pitches, [60, 64, 67], decimal=0)
    
    def test_roundtrip_pitch_freq(self):
        """Converting pitch->freq->pitch should return original."""
        pitches = np.array([48, 60, 69, 72, 84])
        roundtrip = freq_to_pitch(pitch_to_freq(pitches))
        np.testing.assert_array_almost_equal(roundtrip, pitches, decimal=10)
    
    def test_roundtrip_freq_pitch(self):
        """Converting freq->pitch->freq should return original."""
        freqs = np.array([220, 440, 880, 1760])
        roundtrip = pitch_to_freq(freq_to_pitch(freqs))
        np.testing.assert_array_almost_equal(roundtrip, freqs, decimal=10)


class TestDbRatioConversions:
    """Test dB <-> ratio conversions."""
    
    def test_ratio_to_db_unity(self):
        """Ratio 1.0 should be 0 dB."""
        assert ratio_to_db(1.0) == pytest.approx(0.0)
    
    def test_ratio_to_db_double(self):
        """Ratio 2.0 should be ~6.02 dB."""
        assert ratio_to_db(2.0) == pytest.approx(6.0206, rel=1e-3)
    
    def test_ratio_to_db_half(self):
        """Ratio 0.5 should be ~-6.02 dB."""
        assert ratio_to_db(0.5) == pytest.approx(-6.0206, rel=1e-3)
    
    def test_ratio_to_db_ten(self):
        """Ratio 10.0 should be 20 dB."""
        assert ratio_to_db(10.0) == pytest.approx(20.0)
    
    def test_ratio_to_db_array(self):
        """Should work with arrays."""
        ratios = np.array([0.1, 1.0, 10.0])
        dbs = ratio_to_db(ratios)
        np.testing.assert_array_almost_equal(dbs, [-20, 0, 20], decimal=2)
    
    def test_db_to_ratio_zero(self):
        """0 dB should be ratio 1.0."""
        assert db_to_ratio(0.0) == pytest.approx(1.0)
    
    def test_db_to_ratio_6db(self):
        """6 dB should be ~2.0."""
        assert db_to_ratio(6.0) == pytest.approx(1.995, rel=1e-2)
    
    def test_db_to_ratio_neg6db(self):
        """-6 dB should be ~0.5."""
        assert db_to_ratio(-6.0) == pytest.approx(0.501, rel=1e-2)
    
    def test_db_to_ratio_20db(self):
        """20 dB should be 10.0."""
        assert db_to_ratio(20.0) == pytest.approx(10.0)
    
    def test_roundtrip_ratio_db(self):
        """Converting ratio->dB->ratio should return original."""
        ratios = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
        roundtrip = db_to_ratio(ratio_to_db(ratios))
        np.testing.assert_array_almost_equal(roundtrip, ratios, decimal=10)
    
    def test_roundtrip_db_ratio(self):
        """Converting dB->ratio->dB should return original."""
        dbs = np.array([-40, -20, 0, 20, 40])
        roundtrip = ratio_to_db(db_to_ratio(dbs))
        np.testing.assert_array_almost_equal(roundtrip, dbs, decimal=10)


class TestSemitonesRatioConversions:
    """Test semitones <-> ratio conversions."""
    
    def test_semitones_to_ratio_octave(self):
        """12 semitones should be ratio 2.0 (octave)."""
        assert semitones_to_ratio(12) == pytest.approx(2.0)
    
    def test_semitones_to_ratio_fifth(self):
        """7 semitones should be ~1.498 (perfect fifth)."""
        assert semitones_to_ratio(7) == pytest.approx(1.4983, rel=1e-3)
    
    def test_semitones_to_ratio_negative(self):
        """-12 semitones should be 0.5 (octave down)."""
        assert semitones_to_ratio(-12) == pytest.approx(0.5)
    
    def test_semitones_to_ratio_zero(self):
        """0 semitones should be ratio 1.0."""
        assert semitones_to_ratio(0) == pytest.approx(1.0)
    
    def test_ratio_to_semitones_octave(self):
        """Ratio 2.0 should be 12 semitones."""
        assert ratio_to_semitones(2.0) == pytest.approx(12.0)
    
    def test_ratio_to_semitones_fifth(self):
        """Ratio 1.5 should be ~7.02 semitones."""
        assert ratio_to_semitones(1.5) == pytest.approx(7.0195, rel=1e-3)
    
    def test_ratio_to_semitones_octave_down(self):
        """Ratio 0.5 should be -12 semitones."""
        assert ratio_to_semitones(0.5) == pytest.approx(-12.0)
    
    def test_roundtrip_semitones_ratio(self):
        """Converting semitones->ratio->semitones should return original."""
        semitones = np.array([-24, -12, 0, 7, 12, 24])
        roundtrip = ratio_to_semitones(semitones_to_ratio(semitones))
        np.testing.assert_array_almost_equal(roundtrip, semitones, decimal=10)


class TestSampleTimeConversions:
    """Test samples <-> seconds conversions."""
    
    def test_samples_to_seconds_one_second(self):
        """44100 samples at 44100 Hz should be 1 second."""
        assert samples_to_seconds(44100, 44100) == pytest.approx(1.0)
    
    def test_samples_to_seconds_half_second(self):
        """22050 samples at 44100 Hz should be 0.5 seconds."""
        assert samples_to_seconds(22050, 44100) == pytest.approx(0.5)
    
    def test_samples_to_seconds_array(self):
        """Should work with arrays."""
        samples = np.array([0, 22050, 44100, 88200])
        seconds = samples_to_seconds(samples, 44100)
        np.testing.assert_array_almost_equal(seconds, [0, 0.5, 1.0, 2.0])
    
    def test_seconds_to_samples_one_second(self):
        """1 second at 44100 Hz should be 44100 samples."""
        assert seconds_to_samples(1.0, 44100) == pytest.approx(44100.0)
    
    def test_seconds_to_samples_half_second(self):
        """0.5 seconds at 44100 Hz should be 22050 samples."""
        assert seconds_to_samples(0.5, 44100) == pytest.approx(22050.0)
    
    def test_roundtrip_samples_seconds(self):
        """Converting samples->seconds->samples should return original."""
        samples = np.array([0, 1000, 44100, 88200])
        roundtrip = seconds_to_samples(samples_to_seconds(samples, 44100), 44100)
        np.testing.assert_array_almost_equal(roundtrip, samples)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_freq_to_pitch_zero_protected(self):
        """freq_to_pitch should handle zero gracefully."""
        # Should not raise, returns very negative pitch
        result = freq_to_pitch(0.0)
        assert result < -100  # Very low pitch
    
    def test_ratio_to_db_zero_protected(self):
        """ratio_to_db should handle zero gracefully."""
        # Should not raise, returns very negative dB
        result = ratio_to_db(0.0)
        assert result < -100  # Very low dB
    
    def test_ratio_to_semitones_zero_protected(self):
        """ratio_to_semitones should handle zero gracefully."""
        result = ratio_to_semitones(0.0)
        assert result < -100  # Very negative
    
    def test_scalar_input(self):
        """Functions should work with Python scalars."""
        assert pitch_to_freq(69) == pytest.approx(440.0)
        assert freq_to_pitch(440) == pytest.approx(69.0)
        assert ratio_to_db(1) == pytest.approx(0.0)
        assert db_to_ratio(0) == pytest.approx(1.0)
