"""
Tests for temperament system.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np

from pygmu2.temperament import (
    Temperament,
    EqualTemperament,
    JustIntonation,
    PythagoreanTuning,
    CustomTemperament,
    set_temperament,
    get_temperament,
)


class TestEqualTemperament:
    """Tests for EqualTemperament class."""
    
    def test_12et_default(self):
        """12-ET is the default."""
        et12 = EqualTemperament()
        assert et12.divisions == 12
    
    def test_12et_pitch_to_freq_a4(self):
        """A4 (pitch 69) should be 440 Hz in 12-ET."""
        et12 = EqualTemperament(12)
        assert et12.pitch_to_freq(69) == pytest.approx(440.0)
    
    def test_12et_pitch_to_freq_middle_c(self):
        """Middle C (pitch 60) should be ~261.63 Hz in 12-ET."""
        et12 = EqualTemperament(12)
        assert et12.pitch_to_freq(60) == pytest.approx(261.6256, rel=1e-4)
    
    def test_12et_octave_doubles_frequency(self):
        """Octave (12 semitones) should double frequency in 12-ET."""
        et12 = EqualTemperament(12)
        freq_60 = et12.pitch_to_freq(60)
        freq_72 = et12.pitch_to_freq(72)
        assert freq_72 == pytest.approx(freq_60 * 2.0)
    
    def test_12et_freq_to_pitch_roundtrip(self):
        """freq_to_pitch should be inverse of pitch_to_freq."""
        et12 = EqualTemperament(12)
        pitches = np.array([60, 64, 67, 69, 72])
        freqs = et12.pitch_to_freq(pitches)
        roundtrip = et12.freq_to_pitch(freqs)
        np.testing.assert_array_almost_equal(roundtrip, pitches, decimal=10)
    
    def test_12et_interval_to_ratio_octave(self):
        """12 semitones should be ratio 2.0 in 12-ET."""
        et12 = EqualTemperament(12)
        assert et12.interval_to_ratio(12) == pytest.approx(2.0)
    
    def test_12et_interval_to_ratio_fifth(self):
        """7 semitones should be ~1.498 in 12-ET."""
        et12 = EqualTemperament(12)
        assert et12.interval_to_ratio(7) == pytest.approx(1.4983, rel=1e-3)
    
    def test_12et_ratio_to_interval_roundtrip(self):
        """ratio_to_interval should be inverse of interval_to_ratio."""
        et12 = EqualTemperament(12)
        intervals = np.array([0, 2, 4, 5, 7, 9, 11, 12])
        ratios = et12.interval_to_ratio(intervals)
        roundtrip = et12.ratio_to_interval(ratios)
        np.testing.assert_array_almost_equal(roundtrip, intervals, decimal=10)
    
    def test_19et_divisions(self):
        """19-ET should have 19 divisions."""
        et19 = EqualTemperament(19)
        assert et19.divisions == 19
    
    def test_19et_octave(self):
        """19 steps should be an octave in 19-ET."""
        et19 = EqualTemperament(19)
        assert et19.interval_to_ratio(19) == pytest.approx(2.0)
    
    def test_19et_pitch_to_freq(self):
        """Pitch 69 should still be 440 Hz (reference pitch)."""
        et19 = EqualTemperament(19)
        assert et19.pitch_to_freq(69) == pytest.approx(440.0)
    
    def test_24et_quarter_tone(self):
        """24-ET supports quarter tones."""
        et24 = EqualTemperament(24)
        # 24 steps = octave
        assert et24.interval_to_ratio(24) == pytest.approx(2.0)
        # 12 steps = half octave (tritone)
        assert et24.interval_to_ratio(12) == pytest.approx(1.4142, rel=1e-3)
    
    def test_invalid_divisions(self):
        """Division count must be positive."""
        with pytest.raises(ValueError):
            EqualTemperament(0)
        with pytest.raises(ValueError):
            EqualTemperament(-1)
    
    def test_name(self):
        """Temperament should have descriptive name."""
        et12 = EqualTemperament(12)
        assert "12" in et12.name()
        assert "Equal" in et12.name()
        
        et19 = EqualTemperament(19)
        assert "19" in et19.name()
    
    def test_reference_pitch_and_freq(self):
        """Custom reference pitch and frequency should work."""
        et12 = EqualTemperament(12)
        # Use C4 (pitch 60) as reference at 256 Hz
        freq = et12.pitch_to_freq(60, reference_pitch=60, reference_freq=256.0)
        assert freq == pytest.approx(256.0)
        
        # One octave up should be 512 Hz
        freq_72 = et12.pitch_to_freq(72, reference_pitch=60, reference_freq=256.0)
        assert freq_72 == pytest.approx(512.0)


class TestJustIntonation:
    """Tests for JustIntonation class."""
    
    def test_default_ratios(self):
        """Default should be 5-limit JI with 12 notes."""
        ji = JustIntonation()
        assert ji.num_notes == 12
        assert ji.ratios[0] == pytest.approx(1.0)  # Unison
    
    def test_pitch_to_freq_unison(self):
        """Pitch 60 (reference) should give reference frequency."""
        ji = JustIntonation(reference_pitch=60.0)
        # Default reference is A4=440, so we need to calculate C4
        freq = ji.pitch_to_freq(60, reference_pitch=69, reference_freq=440.0)
        # This will depend on the JI ratios, just check it's reasonable
        assert 250 < freq < 270  # Should be around 261 Hz
    
    def test_interval_to_ratio_octave(self):
        """12 scale degrees should be an octave."""
        ji = JustIntonation()
        ratio = ji.interval_to_ratio(12)
        assert ratio == pytest.approx(2.0)
    
    def test_interval_to_ratio_fifth(self):
        """Fifth (scale degree 7) should be 3/2 in 5-limit JI."""
        ji = JustIntonation()
        ratio = ji.interval_to_ratio(7)
        assert ratio == pytest.approx(3/2, abs=0.01)
    
    def test_interval_to_ratio_major_third(self):
        """Major third (scale degree 4) should be 5/4 in 5-limit JI."""
        ji = JustIntonation()
        ratio = ji.interval_to_ratio(4)
        assert ratio == pytest.approx(5/4, abs=0.01)
    
    def test_custom_ratios(self):
        """Custom ratio table should work."""
        # Simple 5-note pentatonic
        ratios = [1.0, 9/8, 5/4, 3/2, 5/3]
        ji = JustIntonation(ratios=ratios)
        assert ji.num_notes == 5
        
        # First ratio should be unison
        assert ji.ratios[0] == pytest.approx(1.0)
        
        # Fifth scale degree (wrapping around) should be octave
        ratio = ji.interval_to_ratio(5)
        assert ratio == pytest.approx(2.0)
    
    def test_invalid_ratios(self):
        """First ratio must be 1.0."""
        with pytest.raises(ValueError):
            JustIntonation(ratios=[2.0, 3.0])  # First not 1.0
        
        with pytest.raises(ValueError):
            JustIntonation(ratios=[1.0])  # Too few ratios
    
    def test_name(self):
        """JI should have descriptive name."""
        ji = JustIntonation()
        assert "Just" in ji.name()
        assert "12" in ji.name()  # Default has 12 notes


class TestPythagoreanTuning:
    """Tests for PythagoreanTuning class."""
    
    def test_perfect_fifth(self):
        """Perfect fifth should be exactly 3/2."""
        pyth = PythagoreanTuning()
        ratio = pyth.interval_to_ratio(7)
        assert ratio == pytest.approx(3/2, abs=1e-10)
    
    def test_perfect_fourth(self):
        """Perfect fourth should be exactly 4/3."""
        pyth = PythagoreanTuning()
        ratio = pyth.interval_to_ratio(5)
        assert ratio == pytest.approx(4/3, abs=1e-6)
    
    def test_octave(self):
        """Octave should be exactly 2/1."""
        pyth = PythagoreanTuning()
        ratio = pyth.interval_to_ratio(12)
        assert ratio == pytest.approx(2.0)
    
    def test_major_third_sharp(self):
        """Pythagorean major third (81/64) is sharper than just (5/4)."""
        pyth = PythagoreanTuning()
        pyth_third = pyth.interval_to_ratio(4)
        just_third = 5/4
        
        assert pyth_third == pytest.approx(81/64, abs=1e-6)
        assert pyth_third > just_third
    
    def test_name(self):
        """Pythagorean should have descriptive name."""
        pyth = PythagoreanTuning()
        assert "Pythagorean" in pyth.name()


class TestCustomTemperament:
    """Tests for CustomTemperament class."""
    
    def test_custom_functions(self):
        """Custom temperament with user functions should work."""
        # Create a simple stretched tuning
        def stretched_p2f(pitch, ref_pitch=69, ref_freq=440):
            return ref_freq * (2.001 ** ((pitch - ref_pitch) / 12))
        
        def stretched_f2p(freq, ref_pitch=69, ref_freq=440):
            freq = np.maximum(np.asarray(freq), 1e-10)
            return ref_pitch + 12 * np.log2(freq / ref_freq) / np.log2(2.001)
        
        def stretched_i2r(interval):
            return 2.001 ** (np.asarray(interval) / 12)
        
        def stretched_r2i(ratio):
            ratio = np.maximum(np.asarray(ratio), 1e-10)
            return 12 * np.log2(ratio) / np.log2(2.001)
        
        stretched = CustomTemperament(
            pitch_to_freq_func=stretched_p2f,
            freq_to_pitch_func=stretched_f2p,
            interval_to_ratio_func=stretched_i2r,
            ratio_to_interval_func=stretched_r2i,
            name="Stretched Tuning"
        )
        
        # Reference pitch should still be 440
        assert stretched.pitch_to_freq(69) == pytest.approx(440.0)
        
        # Octave should be slightly > 2.0
        octave_ratio = stretched.interval_to_ratio(12)
        assert octave_ratio == pytest.approx(2.001)
        assert octave_ratio > 2.0
    
    def test_name(self):
        """Custom temperament should have user-specified name."""
        custom = CustomTemperament(
            pitch_to_freq_func=lambda p, rp=69, rf=440: rf,
            freq_to_pitch_func=lambda f, rp=69, rf=440: rp,
            interval_to_ratio_func=lambda i: 1.0,
            ratio_to_interval_func=lambda r: 0.0,
            name="Test Temperament"
        )
        assert custom.name() == "Test Temperament"


class TestGlobalTemperament:
    """Tests for global temperament configuration."""
    
    def test_default_is_12et(self):
        """Default global temperament should be 12-ET."""
        temp = get_temperament()
        assert isinstance(temp, EqualTemperament)
        assert temp.divisions == 12
    
    def test_set_and_get_temperament(self):
        """set_temperament should change the global default."""
        # Save current default
        original = get_temperament()
        
        try:
            # Set to 19-ET
            et19 = EqualTemperament(19)
            set_temperament(et19)
            
            # Verify it changed
            temp = get_temperament()
            assert isinstance(temp, EqualTemperament)
            assert temp.divisions == 19
            
            # Set to JI
            ji = JustIntonation()
            set_temperament(ji)
            
            temp = get_temperament()
            assert isinstance(temp, JustIntonation)
        
        finally:
            # Restore original
            set_temperament(original)
    
    def test_conversion_functions_use_global(self):
        """Conversion functions should use global temperament."""
        from pygmu2 import pitch_to_freq
        
        # Save original
        original = get_temperament()
        
        try:
            # Set to 19-ET
            et19 = EqualTemperament(19)
            set_temperament(et19)
            
            # pitch_to_freq should now use 19-ET
            # In 19-ET, 19 steps = octave
            # So pitch 69+19 should be 880 Hz (octave above A4)
            freq = pitch_to_freq(69 + 19)
            assert freq == pytest.approx(880.0)
        
        finally:
            set_temperament(original)


class TestConversionIntegration:
    """Test conversion functions with temperament parameter."""
    
    def test_pitch_to_freq_with_temperament(self):
        """pitch_to_freq should accept temperament parameter."""
        from pygmu2 import pitch_to_freq
        
        et19 = EqualTemperament(19)
        freq = pitch_to_freq(69, temperament=et19)
        assert freq == pytest.approx(440.0)
        
        # 19 steps up = octave in 19-ET
        freq_octave = pitch_to_freq(69 + 19, temperament=et19)
        assert freq_octave == pytest.approx(880.0)
    
    def test_freq_to_pitch_with_temperament(self):
        """freq_to_pitch should accept temperament parameter."""
        from pygmu2 import freq_to_pitch
        
        et19 = EqualTemperament(19)
        pitch = freq_to_pitch(440.0, temperament=et19)
        assert pitch == pytest.approx(69.0)
    
    def test_semitones_to_ratio_with_temperament(self):
        """semitones_to_ratio should accept temperament parameter."""
        from pygmu2 import semitones_to_ratio
        
        et19 = EqualTemperament(19)
        ratio = semitones_to_ratio(19, temperament=et19)
        assert ratio == pytest.approx(2.0)  # Octave in 19-ET
    
    def test_ratio_to_semitones_with_temperament(self):
        """ratio_to_semitones should accept temperament parameter."""
        from pygmu2 import ratio_to_semitones
        
        et19 = EqualTemperament(19)
        interval = ratio_to_semitones(2.0, temperament=et19)
        assert interval == pytest.approx(19.0)  # Octave = 19 steps
    
    def test_backward_compatibility(self):
        """Functions without temperament param should still work (12-ET)."""
        from pygmu2 import pitch_to_freq, freq_to_pitch
        from pygmu2 import semitones_to_ratio, ratio_to_semitones
        
        # Ensure we're using default 12-ET
        original = get_temperament()
        try:
            set_temperament(EqualTemperament(12))
            
            # These should work as before
            assert pitch_to_freq(69) == pytest.approx(440.0)
            assert pitch_to_freq(60) == pytest.approx(261.6256, rel=1e-4)
            assert freq_to_pitch(440.0) == pytest.approx(69.0)
            assert semitones_to_ratio(12) == pytest.approx(2.0)
            assert ratio_to_semitones(2.0) == pytest.approx(12.0)
        finally:
            set_temperament(original)


class TestArraySupport:
    """Test that temperaments work with numpy arrays."""
    
    def test_12et_array_pitch_to_freq(self):
        """12-ET should work with array inputs."""
        et12 = EqualTemperament(12)
        pitches = np.array([60, 64, 67, 69])  # C major triad + A4
        freqs = et12.pitch_to_freq(pitches)
        
        assert len(freqs) == 4
        assert freqs[3] == pytest.approx(440.0)  # A4
    
    def test_ji_array_interval_to_ratio(self):
        """JI should work with array inputs."""
        ji = JustIntonation()
        intervals = np.array([0, 4, 7, 12])  # Unison, third, fifth, octave
        ratios = ji.interval_to_ratio(intervals)
        
        assert len(ratios) == 4
        assert ratios[0] == pytest.approx(1.0)
        assert ratios[3] == pytest.approx(2.0)
    
    def test_scalar_compatibility(self):
        """Temperaments should work with scalar inputs."""
        et12 = EqualTemperament(12)
        
        # Scalar input - returns numpy scalar or array
        freq = et12.pitch_to_freq(69)
        assert isinstance(freq, (np.ndarray, np.floating))
        assert float(freq) == pytest.approx(440.0)
