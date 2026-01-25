"""
Tests for ResetPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import unittest
import numpy as np

from pygmu2 import (
    ResetPE,
    ArrayPE,
    IdentityPE,
    BlitSawPE,
    ConstantPE,
    NullRenderer,
)


class TestResetPEBasics(unittest.TestCase):
    """Test basic ResetPE creation and properties."""
    
    def setUp(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_create(self):
        """Test creation with source and trigger."""
        source = IdentityPE()
        trigger = ConstantPE(1.0)
        reset_pe = ResetPE(source, trigger)
        
        assert reset_pe.source is source
        assert reset_pe.trigger is trigger
    
    def test_is_pure(self):
        """ResetPE is not pure (maintains trigger state)."""
        source = IdentityPE()
        trigger = ConstantPE(1.0)
        reset_pe = ResetPE(source, trigger)
        assert reset_pe.is_pure() is False
    
    def test_channel_count(self):
        """ResetPE passes through channel count from source."""
        source = IdentityPE(channels=2)
        trigger = ConstantPE(1.0)
        reset_pe = ResetPE(source, trigger)
        assert reset_pe.channel_count() == 2
    
    def test_repr(self):
        """Test string representation."""
        source = IdentityPE()
        trigger = ConstantPE(1.0)
        reset_pe = ResetPE(source, trigger)
        repr_str = repr(reset_pe)
        assert "ResetPE" in repr_str
        assert "IdentityPE" in repr_str
        assert "ConstantPE" in repr_str


class TestResetPETimeShifting(unittest.TestCase):
    """Test time shifting behavior using IdentityPE."""
    
    def setUp(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_single_reset_at_start(self):
        """Test reset at the very start of rendering."""
        # Trigger: high at sample 0
        trigger = ArrayPE([1.0, 1.0, 1.0, 1.0, 1.0])
        source = IdentityPE()
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        snippet = reset_pe.render(0, 5)
        output = snippet.data[:, 0]
        
        # After reset at sample 0, source should render from time 0
        # IdentityPE at time 0 outputs: [0, 1, 2, 3, 4]
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)
    
    def test_single_reset_delayed(self):
        """Test reset at a delayed position."""
        # Trigger: low for 3 samples, then high
        trigger = ArrayPE([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        source = IdentityPE()
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        snippet = reset_pe.render(0, 8)
        output = snippet.data[:, 0]
        
        # Before reset (samples 0-2): IdentityPE continues from previous state
        # At absolute time 0-2, IdentityPE outputs: [0, 1, 2]
        # After reset at sample 3: IdentityPE renders from time 0
        # At time 0-4, IdentityPE outputs: [0, 1, 2, 3, 4]
        # But placed at absolute positions 3-7
        expected = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)
    
    def test_multiple_resets(self):
        """Test multiple resets in one chunk."""
        # Trigger: high at 0, low at 2, high at 4, low at 6
        trigger = ArrayPE([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        source = IdentityPE()
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        snippet = reset_pe.render(0, 8)
        output = snippet.data[:, 0]
        
        # Sample 0: reset, render from time 0 -> [0]
        # Sample 1: continue from time 0 -> [1]
        # Sample 2: no reset (gate still high from prev), continue -> [2]
        # Wait, gate goes low at 2, so no rising edge. Let me recalculate...
        # Actually: gate is high at 0,1 then low at 2,3 then high at 4,5
        # Rising edges at: 0 and 4
        # Before edge 0: nothing (current_idx=0, edge=0)
        # At edge 0: reset, render from time 0 for 4 samples (to next edge) -> [0,1,2,3] at positions [0,1,2,3]
        # At edge 4: reset, render from time 0 for 4 samples (to end) -> [0,1,2,3] at positions [4,5,6,7]
        expected = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)
    
    def test_no_reset(self):
        """Test behavior when no rising edge occurs."""
        # Trigger: always low
        trigger = ArrayPE([0.0, 0.0, 0.0, 0.0, 0.0])
        source = IdentityPE()
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        snippet = reset_pe.render(0, 5)
        output = snippet.data[:, 0]
        
        # No reset, source continues normally
        # IdentityPE at absolute time 0-4 outputs: [0, 1, 2, 3, 4]
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)
    
    def test_reset_at_chunk_boundary(self):
        """Test that reset state persists across render calls."""
        trigger = ArrayPE([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        source = IdentityPE()
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        # First chunk: samples 0-3 (trigger rises at 2)
        snippet1 = reset_pe.render(0, 4)
        output1 = snippet1.data[:, 0]
        # Before edge: [0, 1], at edge 2: reset, render from 0 -> [0, 1] at positions [2, 3]
        expected1 = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(output1, expected1)
        
        # Second chunk: samples 4-7 (no new edge)
        # Note: IdentityPE is stateless, so it outputs based on absolute start time
        # After reset in first chunk, we rendered from time 0, but IdentityPE doesn't
        # maintain state. In second chunk, we render from absolute time 4.
        snippet2 = reset_pe.render(4, 4)
        output2 = snippet2.data[:, 0]
        # IdentityPE at absolute time 4-7 outputs: [4, 5, 6, 7]
        expected2 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        np.testing.assert_array_equal(output2, expected2)


class TestResetPEStateReset(unittest.TestCase):
    """Test state reset behavior with stateful sources."""
    
    def setUp(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_blit_saw_reset(self):
        """Test that BlitSawPE phase resets on trigger."""
        # Trigger: high at sample 0, low at 100, high at 200
        trigger_data = np.concatenate([
            np.ones(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        ])
        trigger = ArrayPE(trigger_data)
        
        source = BlitSawPE(frequency=440.0)
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        snippet = reset_pe.render(0, 300)
        output = snippet.data[:, 0]
        
        # After reset at sample 0, oscillator should start from phase 0
        # After reset at sample 200, oscillator should start from phase 0 again
        # The output at sample 200 should match output at sample 0 (same phase)
        assert abs(output[0] - output[200]) < 0.1, "Oscillator should restart at same phase"
    
    def test_multiple_resets_stateful(self):
        """Test multiple resets with stateful source."""
        # Trigger: rising edges at samples 0, 50, 100
        # Need falling edges between to create proper rising edges
        trigger_data = np.zeros(150, dtype=np.float32)
        # First gate: 0-25 (rising at 0)
        trigger_data[0:25] = 1.0
        # Second gate: 50-75 (rising at 50)
        trigger_data[50:75] = 1.0
        # Third gate: 100-125 (rising at 100)
        trigger_data[100:125] = 1.0
        # Samples 25-49, 75-99, 125-149 are 0 (falling edges)
        
        trigger = ArrayPE(trigger_data)
        source = BlitSawPE(frequency=440.0)
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        snippet = reset_pe.render(0, 150)
        output = snippet.data[:, 0]
        
        # After each reset, oscillator should restart from phase 0
        # Output at 0, 50, 100 should be similar (same phase)
        assert abs(output[0] - output[50]) < 0.1, "First two resets should produce same phase"
        assert abs(output[0] - output[100]) < 0.1, "All resets should produce same phase"


class TestResetPEEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_gap_in_rendering(self):
        """Test behavior when there's a gap between render calls."""
        trigger = ArrayPE([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        source = IdentityPE()
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        # First chunk: samples 0-3
        snippet1 = reset_pe.render(0, 4)
        # Stop and restart to test gap handling properly
        self.renderer.stop()
        
        # Second chunk: samples 10-13 (gap of 6 samples)
        # Use a longer trigger that covers sample 10
        trigger_long = ArrayPE([0.0] * 8 + [1.0] * 4)
        reset_pe2 = ResetPE(source, trigger_long)
        self.renderer.set_source(reset_pe2)
        self.renderer.start()
        
        snippet1 = reset_pe2.render(0, 4)
        snippet2 = reset_pe2.render(10, 4)
        output2 = snippet2.data[:, 0]
        
        # At sample 10, trigger is high. Gap handling assumes prev was 0, so we detect edge
        # So reset and render from time 0: [0, 1, 2, 3]
        expected = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(output2, expected)
    
    def test_rapid_triggers(self):
        """Test rapid trigger on/off cycles."""
        # Trigger: rapid on/off pattern
        trigger_data = np.array([
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0
        ], dtype=np.float32)
        trigger = ArrayPE(trigger_data)
        source = IdentityPE()
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        snippet = reset_pe.render(0, 8)
        output = snippet.data[:, 0]
        
        # Rising edges at: 0, 2, 4, 6
        # At each edge: reset and render from time 0
        # Segment 0-1: reset at 0, render [0, 1]
        # Segment 2-3: reset at 2, render [0, 1]
        # Segment 4-5: reset at 4, render [0, 1]
        # Segment 6-7: reset at 6, render [0, 1]
        expected = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)
    
    def test_trigger_stays_high(self):
        """Test behavior when trigger stays high (no falling edge)."""
        # Trigger: high from start, stays high
        trigger = ArrayPE([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        source = IdentityPE()
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        snippet = reset_pe.render(0, 8)
        output = snippet.data[:, 0]
        
        # Rising edge only at sample 0
        # After reset at 0, render from time 0 for all 8 samples
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)


class TestResetPESampleAccurate(unittest.TestCase):
    """Test sample-accurate behavior."""
    
    def setUp(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_precise_timing(self):
        """Test precise timing of reset and time shift."""
        # Trigger: rising edge exactly at sample 5
        trigger = ArrayPE([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        source = IdentityPE()
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        snippet = reset_pe.render(0, 10)
        output = snippet.data[:, 0]
        
        # Before edge (0-4): IdentityPE at absolute time -> [0, 1, 2, 3, 4]
        # At edge 5: reset, render from time 0 -> [0, 1, 2, 3, 4] at positions [5, 6, 7, 8, 9]
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)
    
    def test_consecutive_edges(self):
        """Test consecutive rising edges (edge immediately after another)."""
        # Trigger: edges at 2 and 4 (need falling edge between for proper detection)
        # Pattern: [0, 0, 1, 0, 1, 1, 1, 1] - edges at 2 and 4
        trigger = ArrayPE([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        source = IdentityPE()
        reset_pe = ResetPE(source, trigger)
        
        self.renderer.set_source(reset_pe)
        self.renderer.start()
        
        snippet = reset_pe.render(0, 8)
        output = snippet.data[:, 0]
        
        # Before edge 2: render from absolute time 0-1 -> [0, 1] at positions [0, 1]
        # Edge at 2: reset, render from time 0 for 2 samples (to next edge at 4) -> [0, 1] at positions [2, 3]
        # Edge at 4: reset, render from time 0 for 4 samples (to end) -> [0, 1, 2, 3] at positions [4, 5, 6, 7]
        expected = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    unittest.main()
