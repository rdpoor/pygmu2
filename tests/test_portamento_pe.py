"""
Tests for PortamentoPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    PortamentoPE,
    TransformPE,
    NullRenderer,
    Extent,
    pitch_to_freq,
)


class TestPortamentoPEBasics:
    """Test basic PortamentoPE creation and properties."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_create_portamento_pe(self):
        """Test basic PortamentoPE creation."""
        notes = [
            (69.0, 0, 1000),      # A4 at t=0, duration 1000 samples
            (73.0, 1000, 1000),  # C#5 at t=1000, duration 1000 samples
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        
        assert porta.notes == notes
        assert porta.max_ramp_seconds == 0.05
        assert porta.ramp_fraction == 0.3  # default
        assert porta.channel_count() == 1  # default
    
    def test_create_with_samples(self):
        """Test PortamentoPE creation with max_ramp_samples."""
        notes = [(69.0, 0, 1000)]
        porta = PortamentoPE(notes, max_ramp_samples=2205)
        
        assert porta.max_ramp_samples == 2205
        assert porta.max_ramp_seconds is None
    
    def test_empty_notes_raises(self):
        """Empty notes list should raise ValueError."""
        with pytest.raises(ValueError, match="notes list cannot be empty"):
            PortamentoPE([])
    
    def test_both_samples_and_seconds_raises(self):
        """Specifying both max_ramp_samples and max_ramp_seconds should raise."""
        notes = [(69.0, 0, 1000)]
        with pytest.raises(ValueError, match="specify either max_ramp_samples or max_ramp_seconds"):
            PortamentoPE(notes, max_ramp_samples=1000, max_ramp_seconds=0.1)
    
    def test_negative_samples_raises(self):
        """Negative max_ramp_samples should raise ValueError."""
        notes = [(69.0, 0, 1000)]
        with pytest.raises(ValueError, match="must be non-negative"):
            PortamentoPE(notes, max_ramp_samples=-1)
    
    def test_negative_seconds_raises(self):
        """Negative max_ramp_seconds should raise ValueError."""
        notes = [(69.0, 0, 1000)]
        with pytest.raises(ValueError, match="must be non-negative"):
            PortamentoPE(notes, max_ramp_seconds=-0.1)
    
    def test_invalid_ramp_fraction_raises(self):
        """ramp_fraction outside [0, 1] should raise ValueError."""
        notes = [(69.0, 0, 1000)]
        with pytest.raises(ValueError, match="ramp_fraction must be between 0 and 1"):
            PortamentoPE(notes, ramp_fraction=1.5)
    
    def test_is_pure(self):
        """PortamentoPE should be pure."""
        notes = [(69.0, 0, 1000)]
        porta = PortamentoPE(notes)
        assert porta.is_pure() is True


class TestPortamentoPERender:
    """Test PortamentoPE rendering behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_single_note_outputs_constant_pitch(self):
        """Single note should output constant pitch value."""
        notes = [(77.0, 0, 1000)]  # F5
        porta = PortamentoPE(notes)
        self.renderer.set_source(porta)
        
        snippet = porta.render(0, 1000)
        
        # All samples should be 77.0 (F5 MIDI pitch)
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            np.full(1000, 77.0, dtype=np.float32),
            decimal=5
        )
    
    def test_first_note_starts_at_correct_time(self):
        """First note should start at its specified start time, with infinite extent (HOLD_BOTH)."""
        notes = [(77.0, 5000, 1000)]  # F5 starting at sample 5000
        porta = PortamentoPE(notes)
        self.renderer.set_source(porta)
        
        # Before start time: with infinite extent (HOLD_BOTH), ConstantPE outputs its value
        # When delayed, DelayPE requests negative source times, and ConstantPE outputs 77.0
        # So we get 77.0 before the delay time (infinite extent behavior)
        snippet_before = porta.render(0, 1000)
        np.testing.assert_array_almost_equal(
            snippet_before.data[:, 0],
            np.full(1000, 77.0, dtype=np.float32),
            decimal=5
        )
        
        # At start time should have the pitch value
        snippet_at_start = porta.render(5000, 1000)
        np.testing.assert_array_almost_equal(
            snippet_at_start.data[:, 0],
            np.full(1000, 77.0, dtype=np.float32),
            decimal=5
        )
    
    def test_two_notes_same_pitch_no_ramp(self):
        """Two notes with same pitch create a ramp with same start/end value (constant output)."""
        notes = [
            (69.0, 0, 1000),      # A4
            (69.0, 1000, 1000),   # A4 again
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        snippet = porta.render(0, 2000)
        
        # All samples should be 69.0 (ramp with start_value=end_value outputs constant)
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            np.full(2000, 69.0, dtype=np.float32),
            decimal=5
        )
    
    def test_two_notes_different_pitch_creates_ramp(self):
        """Two notes with different pitches should create a ramp."""
        notes = [
            (69.0, 0, 1000),      # A4 (440 Hz)
            (73.0, 1000, 1000),   # C#5 (554.37 Hz)
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05, ramp_fraction=0.3)
        self.renderer.set_source(porta)
        
        snippet = porta.render(0, 2000)
        data = snippet.data[:, 0]
        
        # First 1000 samples should be 69.0 (A4)
        np.testing.assert_array_almost_equal(
            data[:1000],
            np.full(1000, 69.0, dtype=np.float32),
            decimal=5
        )
        
        # At sample 1000, ramp should start
        # Ramp duration should be min(max_ramp_samples, note_duration * ramp_fraction)
        # max_ramp_samples = 0.05 * 44100 = 2205
        # note_duration * ramp_fraction = 1000 * 0.3 = 300
        # So ramp duration = 300 samples
        ramp_duration = min(2205, int(round(1000 * 0.3)))
        
        # Check that ramp starts at sample 1000
        assert data[1000] == pytest.approx(69.0, abs=0.1)  # Start of ramp
        
        # Check that ramp ends at sample 1000 + ramp_duration
        ramp_end_idx = 1000 + ramp_duration
        if ramp_end_idx < 2000:
            assert data[ramp_end_idx] == pytest.approx(73.0, abs=0.1)  # End of ramp
            
            # After ramp, should hold end value
            np.testing.assert_array_almost_equal(
                data[ramp_end_idx:2000],
                np.full(2000 - ramp_end_idx, 73.0, dtype=np.float32),
                decimal=5
            )
    
    def test_pitch_values_match_configured_notes(self):
        """Regression test: Pitch values output should match configured notes exactly."""
        # This test verifies the bug where pitch values were doubled (148 instead of 74)
        notes = [
            (77.0, 0, 1000),      # F5
            (74.0, 1000, 1000),   # D5
            (70.0, 2000, 1000),   # Bb4
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        # Render at various points to verify pitch values
        # First note region
        snippet1 = porta.render(500, 100)
        np.testing.assert_array_almost_equal(
            snippet1.data[:, 0],
            np.full(100, 77.0, dtype=np.float32),
            decimal=5
        )
        
        # Second note region (after ramp completes)
        # Ramp duration should be ~300 samples (1000 * 0.3), so at sample 1300 we should be at 74.0
        snippet2 = porta.render(1300, 100)
        np.testing.assert_array_almost_equal(
            snippet2.data[:, 0],
            np.full(100, 74.0, dtype=np.float32),
            decimal=5
        )
        
        # Third note region (after ramp completes)
        snippet3 = porta.render(2300, 100)
        np.testing.assert_array_almost_equal(
            snippet3.data[:, 0],
            np.full(100, 70.0, dtype=np.float32),
            decimal=5
        )
    
    def test_portamento_with_pitch_to_freq(self):
        """Regression test: PortamentoPE should work correctly with TransformPE and pitch_to_freq."""
        notes = [
            (69.0, 0, 1000),      # A4 (440 Hz)
            (73.0, 1000, 1000),   # C#5 (554.37 Hz)
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        freq_stream = TransformPE(porta, func=pitch_to_freq)
        
        self.renderer.set_source(freq_stream)
        
        # Render first note region
        snippet1 = freq_stream.render(500, 100)
        expected_freq1 = pitch_to_freq(69.0)
        np.testing.assert_array_almost_equal(
            snippet1.data[:, 0],
            np.full(100, expected_freq1, dtype=np.float32),
            decimal=1
        )
        
        # Render second note region (after ramp)
        snippet2 = freq_stream.render(1300, 100)
        expected_freq2 = pitch_to_freq(73.0)
        np.testing.assert_array_almost_equal(
            snippet2.data[:, 0],
            np.full(100, expected_freq2, dtype=np.float32),
            decimal=1
        )
    
    def test_multiple_notes_sequence(self):
        """Test sequence of multiple notes with portamento."""
        notes = [
            (77.0, 0, 1000),      # F5
            (74.0, 1000, 1000),   # D5
            (70.0, 2000, 1000),   # Bb4
            (70.0, 3000, 1000),   # Bb4 (same pitch, no ramp)
            (75.0, 4000, 1000),   # Eb5
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        # Verify each note region has correct pitch
        # Note 0: 77.0
        snippet0 = porta.render(500, 100)
        np.testing.assert_array_almost_equal(
            snippet0.data[:, 0],
            np.full(100, 77.0, dtype=np.float32),
            decimal=5
        )
        
        # Note 1: 74.0 (after ramp)
        snippet1 = porta.render(1300, 100)
        np.testing.assert_array_almost_equal(
            snippet1.data[:, 0],
            np.full(100, 74.0, dtype=np.float32),
            decimal=5
        )
        
        # Note 2: 70.0 (after ramp)
        snippet2 = porta.render(2300, 100)
        np.testing.assert_array_almost_equal(
            snippet2.data[:, 0],
            np.full(100, 70.0, dtype=np.float32),
            decimal=5
        )
        
        # Note 3: 70.0 (same pitch, no ramp)
        snippet3 = porta.render(3500, 100)
        np.testing.assert_array_almost_equal(
            snippet3.data[:, 0],
            np.full(100, 70.0, dtype=np.float32),
            decimal=5
        )
        
        # Note 4: 75.0 (after ramp)
        snippet4 = porta.render(4300, 100)
        np.testing.assert_array_almost_equal(
            snippet4.data[:, 0],
            np.full(100, 75.0, dtype=np.float32),
            decimal=5
        )
    
    def test_ramp_adaptive_duration(self):
        """Test that ramp duration adapts to note duration."""
        # Ramp starts when current note ends, so it's limited by the next note's duration
        notes_short = [
            (69.0, 0, 100),       # First note (100 samples)
            (73.0, 100, 100),     # Short next note (100 samples) - ramp limited by this
        ]
        porta_short = PortamentoPE(notes_short, max_ramp_seconds=0.05, ramp_fraction=0.3)
        self.renderer.set_source(porta_short)
        
        # Ramp duration should be min(2205, 100 * 0.3) = 30 samples
        snippet = porta_short.render(100, 50)
        data = snippet.data[:, 0]
        
        # At start of ramp (sample 100), should be 69.0
        assert data[0] == pytest.approx(69.0, abs=0.1)
        
        # At end of ramp (sample 130), should be 73.0
        assert data[30] == pytest.approx(73.0, abs=0.1)
        
        # After ramp, should hold 73.0
        assert data[49] == pytest.approx(73.0, abs=0.1)
    
    def test_ramp_max_duration_limit(self):
        """Test that ramp duration is limited by max_ramp_samples."""
        # Long note: ramp should be limited by max_ramp_samples
        notes_long = [
            (69.0, 0, 10000),     # Long note (10000 samples)
            (73.0, 10000, 1000),  # Next note
        ]
        porta_long = PortamentoPE(notes_long, max_ramp_seconds=0.05, ramp_fraction=0.3)
        self.renderer.set_source(porta_long)
        
        # Ramp duration should be min(2205, 10000 * 0.3) = 2205 samples
        snippet = porta_long.render(10000, 2500)
        data = snippet.data[:, 0]
        
        # At start of ramp (sample 10000), should be 69.0
        assert data[0] == pytest.approx(69.0, abs=0.1)
        
        # At end of ramp (sample 12205), should be 73.0
        assert data[2205] == pytest.approx(73.0, abs=0.1)
        
        # After ramp, should hold 73.0
        assert data[2499] == pytest.approx(73.0, abs=0.1)
    
    def test_sorted_notes(self):
        """Test that notes are automatically sorted by start time."""
        # Provide notes out of order
        notes = [
            (73.0, 2000, 1000),   # C#5 at t=2000
            (69.0, 0, 1000),      # A4 at t=0
            (77.0, 1000, 1000),   # F5 at t=1000
        ]
        porta = PortamentoPE(notes)
        
        # Notes should be sorted by start time
        sorted_notes = porta.notes
        assert sorted_notes[0][1] == 0      # First note starts at 0
        assert sorted_notes[1][1] == 1000   # Second note starts at 1000
        assert sorted_notes[2][1] == 2000   # Third note starts at 2000
    
    def test_channels_parameter(self):
        """Test PortamentoPE with multiple channels."""
        notes = [(69.0, 0, 1000)]
        porta = PortamentoPE(notes, channels=2)
        
        assert porta.channel_count() == 2
        
        self.renderer.set_source(porta)
        snippet = porta.render(0, 100)
        
        assert snippet.channels == 2
        assert snippet.data.shape == (100, 2)
        
        # Both channels should have the same pitch value
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            snippet.data[:, 1],
            decimal=5
        )


class TestPortamentoPERegression:
    """Regression tests for specific bugs found during development."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_pitch_values_not_doubled(self):
        """
        Regression test: Verify that pitch values are not doubled.
        
        Bug: PortamentoPE was outputting 148.0 instead of 74.0.
        This test ensures pitch values match the configured notes exactly.
        """
        notes = [
            (77.0, 0, 1000),      # F5
            (74.0, 1000, 1000),   # D5 - should output 74.0, not 148.0
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        # Render well after the ramp completes to check final value
        # Ramp duration ~300 samples, so at sample 1300 we should be at 74.0
        snippet = porta.render(1300, 100)
        
        # Critical assertion: pitch should be 74.0, NOT 148.0
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            np.full(100, 74.0, dtype=np.float32),
            decimal=5
        )
        
        # Verify it's not doubled
        assert snippet.data[0, 0] != pytest.approx(148.0, abs=0.1)
    
    def test_frequencies_correct_after_transform(self):
        """
        Regression test: Verify frequencies are correct after pitch_to_freq transform.
        
        Bug: Frequencies were wrong even though pitches looked correct in debug logs.
        """
        notes = [
            (77.0, 0, 1000),      # F5 (should be ~698.46 Hz)
            (74.0, 1000, 1000),  # D5 (should be ~587.33 Hz)
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        freq_stream = TransformPE(porta, func=pitch_to_freq)
        
        self.renderer.set_source(freq_stream)
        
        # First note: F5 = 698.46 Hz
        snippet1 = freq_stream.render(500, 100)
        expected_freq1 = pitch_to_freq(77.0)
        np.testing.assert_array_almost_equal(
            snippet1.data[:, 0],
            np.full(100, expected_freq1, dtype=np.float32),
            decimal=1
        )
        
        # Second note: D5 = 587.33 Hz
        snippet2 = freq_stream.render(1300, 100)
        expected_freq2 = pitch_to_freq(74.0)
        np.testing.assert_array_almost_equal(
            snippet2.data[:, 0],
            np.full(100, expected_freq2, dtype=np.float32),
            decimal=1
        )
    
    def test_no_mixing_of_overlapping_values(self):
        """
        Regression test: Verify that values from different time periods don't mix.
        
        Bug: When rendering at a later time, values from earlier notes were still
        contributing, causing doubled values.
        """
        notes = [
            (77.0, 0, 1000),      # F5
            (74.0, 1000, 1000),   # D5
            (70.0, 2000, 1000),   # Bb4
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        # Render at a time well after all notes should have completed
        # Each note is 1000 samples, ramp is ~300 samples, so at sample 3000
        # we should only see the last note's value (70.0), not a mix
        snippet = porta.render(3000, 100)
        
        # Should be 70.0 (Bb4), not a sum of multiple values
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            np.full(100, 70.0, dtype=np.float32),
            decimal=5
        )
        
        # Verify it's not a sum (e.g., 70.0 + 74.0 = 144.0)
        assert snippet.data[0, 0] != pytest.approx(144.0, abs=0.1)
        assert snippet.data[0, 0] != pytest.approx(221.0, abs=0.1)  # Sum of all three


class TestPortamentoPEDocumentedBehavior:
    """Test behaviors documented in PortamentoPE docstring."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_disjoint_notes_holds_pitch_during_gap(self):
        """
        Test disjoint notes: pitch holds previous value during gap.
        
        Example: [(69, 0, 500), (73, 1000, 500)]
        - Note 0: pitch=69 from t=0 to t=500 (ends at 500)
        - Gap: t=500 to t=1000 (500 samples)
        - Note 1: pitch=73 from t=1000 to t=1500
        - Expected:
          * t=0 to t=1000: pitch=69 (held from first note, before ramp starts)
          * t=1000: ramp begins, transitions from 69 to 73
          * After ramp: pitch=73 (held for note 1's duration)
        """
        notes = [
            (69.0, 0, 500),      # Note 0: pitch=69, ends at 500
            (73.0, 1000, 500),   # Note 1: pitch=73, starts at 1000, ends at 1500
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        # During gap (t=500 to t=1000): should hold 69.0
        snippet_gap = porta.render(750, 100)
        np.testing.assert_array_almost_equal(
            snippet_gap.data[:, 0],
            np.full(100, 69.0, dtype=np.float32),
            decimal=5
        )
        
        # At start of note 1 (t=1000): ramp begins
        # With max_ramp_seconds=0.05 at 44.1kHz, ramp is ~2205 samples
        # But ramp_fraction=0.3 limits it to 30% of note duration (500 * 0.3 = 150 samples)
        # So ramp should complete by t=1150
        snippet_start = porta.render(1000, 50)
        # First sample should be 69.0 (ramp start), then ramping up
        assert snippet_start.data[0, 0] == pytest.approx(69.0, abs=0.1)
        # Last sample should be higher (ramping toward 73.0)
        assert snippet_start.data[-1, 0] > 69.0
        assert snippet_start.data[-1, 0] < 73.0
        
        # After ramp completes (t=1150+): should hold 73.0
        snippet_after = porta.render(1200, 100)
        np.testing.assert_array_almost_equal(
            snippet_after.data[:, 0],
            np.full(100, 73.0, dtype=np.float32),
            decimal=5
        )
    
    def test_overlapping_notes_transitions_during_overlap(self):
        """
        Test overlapping notes: pitch transitions during overlap period.
        
        Example: [(69, 0, 1500), (73, 1000, 1500)]
        - Note 0: pitch=69 from t=0 to t=1500
        - Note 1: pitch=73 from t=1000 to t=2500
        - Overlap: t=1000 to t=1500 (500 samples)
        - Expected:
          * t=0 to t=1000: pitch=69 (held from first note, before ramp starts)
          * t=1000: ramp begins, transitions from 69 to 73 during overlap
          * After ramp completes: pitch=73 (held for note 1's duration)
        """
        notes = [
            (69.0, 0, 1500),     # Note 0: pitch=69, ends at 1500
            (73.0, 1000, 1500), # Note 1: pitch=73, starts at 1000, ends at 2500
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        # Before overlap (t=0 to t=1000): should hold 69.0
        snippet_before = porta.render(500, 100)
        np.testing.assert_array_almost_equal(
            snippet_before.data[:, 0],
            np.full(100, 69.0, dtype=np.float32),
            decimal=5
        )
        
        # During overlap (t=1000): ramp begins
        # With max_ramp_seconds=0.05 at 44.1kHz, ramp is ~2205 samples
        # But ramp_fraction=0.3 limits it to 30% of note duration (1500 * 0.3 = 450 samples)
        # So ramp should complete by t=1450
        snippet_overlap = porta.render(1000, 50)
        # First sample should be 69.0 (ramp start), then ramping up
        assert snippet_overlap.data[0, 0] == pytest.approx(69.0, abs=0.1)
        # Last sample should be higher (ramping toward 73.0)
        assert snippet_overlap.data[-1, 0] > 69.0
        assert snippet_overlap.data[-1, 0] < 73.0
        
        # After ramp completes (t=1450+): should hold 73.0
        snippet_after = porta.render(1500, 100)
        np.testing.assert_array_almost_equal(
            snippet_after.data[:, 0],
            np.full(100, 73.0, dtype=np.float32),
            decimal=5
        )
    
    def test_behavior_before_first_note(self):
        """
        Test behavior before first note: pitch holds first note's pitch value.
        
        For notes [(69, 0, 500), (73, 1000, 500)]:
        - At time -500: pitch=69.0 (first note's pitch, held)
        - At time -100: pitch=69.0 (first note's pitch, held)
        """
        notes = [
            (69.0, 0, 500),      # Note 0: starts at 0
            (73.0, 1000, 500),  # Note 1: starts at 1000
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        # At time -500: should hold first note's pitch (69.0)
        snippet_before = porta.render(-500, 100)
        np.testing.assert_array_almost_equal(
            snippet_before.data[:, 0],
            np.full(100, 69.0, dtype=np.float32),
            decimal=5
        )
        
        # At time -100: should still hold first note's pitch (69.0)
        snippet_before2 = porta.render(-100, 50)
        np.testing.assert_array_almost_equal(
            snippet_before2.data[:, 0],
            np.full(50, 69.0, dtype=np.float32),
            decimal=5
        )
    
    def test_behavior_after_last_note(self):
        """
        Test behavior after last note: pitch holds last note's pitch value.
        
        For notes [(69, 0, 500), (73, 1000, 500)]:
        - Last note ends at: 1000 + 500 = 1500
        - At time 2000: pitch=73.0 (last note's pitch, held)
        - At time 5000: pitch=73.0 (last note's pitch, held)
        """
        notes = [
            (69.0, 0, 500),      # Note 0: ends at 500
            (73.0, 1000, 500),  # Note 1: starts at 1000, ends at 1500
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        # Last note ends at 1500, ramp completes shortly after
        # At time 2000: should hold last note's pitch (73.0)
        snippet_after = porta.render(2000, 100)
        np.testing.assert_array_almost_equal(
            snippet_after.data[:, 0],
            np.full(100, 73.0, dtype=np.float32),
            decimal=5
        )
        
        # At time 5000: should still hold last note's pitch (73.0)
        snippet_after2 = porta.render(5000, 50)
        np.testing.assert_array_almost_equal(
            snippet_after2.data[:, 0],
            np.full(50, 73.0, dtype=np.float32),
            decimal=5
        )
    
    def test_behavior_outside_range_with_single_note(self):
        """
        Test behavior outside range for single note: should hold that note's pitch.
        """
        notes = [
            (69.0, 1000, 500),  # Single note: starts at 1000, ends at 1500
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        # Before note: should hold note's pitch (69.0)
        snippet_before = porta.render(500, 100)
        np.testing.assert_array_almost_equal(
            snippet_before.data[:, 0],
            np.full(100, 69.0, dtype=np.float32),
            decimal=5
        )
        
        # During note: should be note's pitch (69.0)
        snippet_during = porta.render(1200, 100)
        np.testing.assert_array_almost_equal(
            snippet_during.data[:, 0],
            np.full(100, 69.0, dtype=np.float32),
            decimal=5
        )
        
        # After note: should hold note's pitch (69.0)
        snippet_after = porta.render(2000, 100)
        np.testing.assert_array_almost_equal(
            snippet_after.data[:, 0],
            np.full(100, 69.0, dtype=np.float32),
            decimal=5
        )
    
    def test_behavior_outside_range_with_three_notes(self):
        """
        Test behavior outside range with multiple notes.
        
        Notes: [(69, 0, 500), (73, 1000, 500), (76, 2000, 500)]
        - Before first note: should hold 69.0
        - After last note: should hold 76.0
        """
        notes = [
            (69.0, 0, 500),      # Note 0: starts at 0, ends at 500
            (73.0, 1000, 500),   # Note 1: starts at 1000, ends at 1500
            (76.0, 2000, 500),   # Note 2: starts at 2000, ends at 2500
        ]
        porta = PortamentoPE(notes, max_ramp_seconds=0.05)
        self.renderer.set_source(porta)
        
        # Before first note: should hold first note's pitch (69.0)
        snippet_before = porta.render(-500, 100)
        np.testing.assert_array_almost_equal(
            snippet_before.data[:, 0],
            np.full(100, 69.0, dtype=np.float32),
            decimal=5
        )
        
        # After last note: should hold last note's pitch (76.0)
        # Last note ends at 2500, ramp completes shortly after
        snippet_after = porta.render(3000, 100)
        np.testing.assert_array_almost_equal(
            snippet_after.data[:, 0],
            np.full(100, 76.0, dtype=np.float32),
            decimal=5
        )
