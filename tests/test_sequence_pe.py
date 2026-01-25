"""
Tests for SequencePE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    SequencePE,
    ConstantPE,
    ArrayPE,
    IdentityPE,
    NullRenderer,
    Extent,
    ErrorMode,
    set_error_mode,
    get_error_mode,
)


class TestSequencePEBasics:
    """Test basic SequencePE creation and properties."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_empty_sequence_strict_mode(self):
        """Empty sequence raises error in STRICT mode."""
        original_mode = get_error_mode()
        try:
            set_error_mode(ErrorMode.STRICT)
            with pytest.raises(RuntimeError, match="empty sequence"):
                SequencePE([])
        finally:
            set_error_mode(original_mode)
    
    def test_empty_sequence_lenient_mode(self):
        """Empty sequence outputs silence in LENIENT mode."""
        original_mode = get_error_mode()
        try:
            set_error_mode(ErrorMode.LENIENT)
            seq = SequencePE([])
            
            # Should output silence
            self.renderer.set_source(seq)
            snippet = seq.render(0, 100)
            
            expected = np.zeros((100, 1), dtype=np.float32)
            np.testing.assert_array_equal(snippet.data, expected)
            assert seq.channel_count() == 1
        finally:
            set_error_mode(original_mode)
    
    def test_single_item_sequence(self):
        """Test sequence with single item."""
        pe = ConstantPE(0.5)
        seq = SequencePE([(pe, 0)])
        
        self.renderer.set_source(seq)
        snippet = seq.render(0, 100)
        
        expected = np.full((100, 1), 0.5, dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_single_item_with_delay(self):
        """Test single item with non-zero start_time."""
        pe = IdentityPE()
        seq = SequencePE([(pe, 100)])
        
        self.renderer.set_source(seq)
        
        # Before delay: DelayPE requests source.render(0-100, 50) = source.render(-100, 50)
        # IdentityPE outputs the sample index, so outputs [-100, -99, ..., -51]
        snippet1 = seq.render(0, 50)
        expected1 = np.arange(-100, -50, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_almost_equal(snippet1.data, expected1)
        
        # After delay: IdentityPE outputs sample index, so at global time t,
        # it outputs (t - delay) = (t - 100)
        snippet2 = seq.render(100, 5)
        expected2 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet2.data, expected2)
    
    def test_auto_sorting(self):
        """Test that sequence is automatically sorted by start_time."""
        pe1 = ConstantPE(1.0)
        pe2 = ConstantPE(2.0)
        pe3 = ConstantPE(3.0)
        
        # Provide in reverse order
        sequence = [(pe3, 200), (pe1, 0), (pe2, 100)]
        seq = SequencePE(sequence)
        
        # Should be sorted
        sorted_seq = seq.sequence
        assert sorted_seq[0][1] == 0
        assert sorted_seq[1][1] == 100
        assert sorted_seq[2][1] == 200
    
    def test_is_pure(self):
        """SequencePE is pure."""
        pe = ConstantPE(0.5)
        seq = SequencePE([(pe, 0)])
        assert seq.is_pure() is True
    
    def test_inputs_returns_all_pes(self):
        """Test that inputs() returns all PEs from sequence."""
        pe1 = ConstantPE(0.5)
        pe2 = ConstantPE(0.3)
        seq = SequencePE([(pe1, 0), (pe2, 100)])
        
        inputs = seq.inputs()
        assert pe1 in inputs
        assert pe2 in inputs
        assert len(inputs) == 2
    
    def test_repr(self):
        """Test string representation."""
        pe = ConstantPE(0.5)
        seq = SequencePE([(pe, 0)], overlap=False)
        repr_str = repr(seq)
        assert "SequencePE" in repr_str
        assert "overlap=False" in repr_str


class TestSequencePERender:
    """Test SequencePE rendering behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_non_overlapping_sequence(self):
        """Test non-overlapping sequence (overlap=False)."""
        # Use IdentityPE: outputs sample index, making it easy to verify timing
        pe1 = IdentityPE()
        pe2 = IdentityPE()
        pe3 = IdentityPE()
        
        sequence = [
            (pe1, 0),   # Starts at 0, cropped to [0, 5) in local time
            (pe2, 5),   # Starts at 5, cropped to [0, 5) in local time
            (pe3, 10),  # Starts at 10, no upper bound
        ]
        seq = SequencePE(sequence, overlap=False)
        
        self.renderer.set_source(seq)
        
        # Render first segment: 
        # pe1 (delayed 0, cropped [0,5)): outputs [0, 1, 2, 3, 4]
        # pe2 (delayed 5, cropped [0,5)): DelayPE requests render(-5, 5), CropPE zeros it → [0, 0, 0, 0, 0]
        # pe3 (delayed 10, not cropped): DelayPE requests render(-10, 5), IdentityPE outputs [-10, -9, -8, -7, -6]
        # Mix: [0+0+(-10), 1+0+(-9), 2+0+(-8), 3+0+(-7), 4+0+(-6)] = [-10, -8, -6, -4, -2]
        snippet1 = seq.render(0, 5)
        expected1 = np.array([[-10.0], [-8.0], [-6.0], [-4.0], [-2.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet1.data, expected1)
        
        # Render second segment:
        # pe1 (delayed 0, cropped [0,5)): DelayPE requests render(5-0, 5) = render(5, 5), CropPE zeros it → [0, 0, 0, 0, 0]
        # pe2 (delayed 5, cropped [0,5)): DelayPE requests render(5-5, 5) = render(0, 5), outputs [0, 1, 2, 3, 4]
        # pe3 (delayed 10, not cropped): DelayPE requests render(5-10, 5) = render(-5, 5), IdentityPE outputs [-5, -4, -3, -2, -1]
        # Mix: [0+0+(-5), 0+1+(-4), 0+2+(-3), 0+3+(-2), 0+4+(-1)] = [-5, -3, -1, 1, 3]
        snippet2 = seq.render(5, 5)
        expected2 = np.array([[-5.0], [-3.0], [-1.0], [1.0], [3.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet2.data, expected2)
        
        # Render third segment:
        # pe1 (delayed 0, cropped [0,5)): DelayPE requests render(10-0, 5) = render(10, 5), CropPE zeros it → [0, 0, 0, 0, 0]
        # pe2 (delayed 5, cropped [0,5)): DelayPE requests render(10-5, 5) = render(5, 5), CropPE zeros it → [0, 0, 0, 0, 0]
        # pe3 (delayed 10, not cropped): DelayPE requests render(10-10, 5) = render(0, 5), outputs [0, 1, 2, 3, 4]
        # Mix: [0+0+0, 0+0+1, 0+0+2, 0+0+3, 0+0+4] = [0, 1, 2, 3, 4]
        snippet3 = seq.render(10, 5)
        expected3 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet3.data, expected3)
    
    def test_overlapping_sequence(self):
        """Test overlapping sequence (overlap=True)."""
        pe1 = IdentityPE()
        pe2 = IdentityPE()
        
        sequence = [
            (pe1, 0),
            (pe2, 1),  # Starts before pe1 ends
        ]
        seq = SequencePE(sequence, overlap=True)
        
        self.renderer.set_source(seq)
        
        # pe1 (delayed by 0): at global time t, outputs local time t-0 = t
        # pe2 (delayed by 1): at global time t, outputs local time t-1
        # Mix:
        #   Global time 0: pe1=0, pe2=-1 (requested at -1) → 0+(-1) = -1
        #   Global time 1: pe1=1, pe2=0 → 1+0 = 1
        #   Global time 2: pe1=2, pe2=1 → 2+1 = 3
        #   Global time 3: pe1=3, pe2=2 → 3+2 = 5
        snippet = seq.render(0, 4)
        expected = np.array([[-1.0], [1.0], [3.0], [5.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_negative_start_times(self):
        """Test sequence with negative start_times."""
        pe1 = IdentityPE()
        pe2 = IdentityPE()
        
        sequence = [
            (pe1, -100),  # Negative start_time
            (pe2, 0),
        ]
        seq = SequencePE(sequence, overlap=True)
        
        self.renderer.set_source(seq)
        
        # At global time 0:
        # pe1 outputs (0 - (-100)) = 100 (started at -100, so at time 0 it's at local time 100)
        # pe2 outputs (0 - 0) = 0 (started at 0, so at time 0 it's at local time 0)
        # Mix: 100 + 0 = 100
        snippet = seq.render(0, 5)
        # pe1: [100, 101, 102, 103, 104] (delayed by -100, so adds 100 to each)
        # pe2: [0, 1, 2, 3, 4] (delayed by 0)
        # Mix: [100, 102, 104, 106, 108]
        expected = np.array([[100.0], [102.0], [104.0], [106.0], [108.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_extent_calculation(self):
        """Test extent calculation."""
        pe1 = ArrayPE([1.0] * 100)  # 100 samples
        pe2 = ArrayPE([2.0] * 100)  # 100 samples
        
        sequence = [
            (pe1, 0),    # Extent: [0, 100)
            (pe2, 200),  # Extent: [200, 300) after delay
        ]
        seq = SequencePE(sequence, overlap=False)
        
        extent = seq.extent()
        # Should be union: [0, 100) U [200, 300)
        # Since extents don't overlap, union should cover both
        assert extent.start == 0
        assert extent.end == 300
    
    def test_channel_count_validation(self):
        """Test that channel count mismatch causes error during rendering."""
        pe1 = ConstantPE(0.5, channels=1)
        pe2 = ConstantPE(0.3, channels=2)  # Mismatch
        
        sequence = [(pe1, 0), (pe2, 100)]
        seq = SequencePE(sequence)
        
        self.renderer.set_source(seq)
        
        # When rendering, MixPE tries to add arrays of different shapes
        # This should raise a ValueError about channel mismatch or shape mismatch
        # The error might come from MixPE.resolve_channel_count() or from numpy array addition
        with pytest.raises((ValueError,)):
            seq.render(0, 10)


class TestSequencePESampleAccurate:
    """Test sample-accurate behavior of SequencePE."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_precise_timing_non_overlapping(self):
        """Test precise timing with non-overlapping sequence."""
        # Use IdentityPE: outputs sample index, making timing verification straightforward
        pe1 = IdentityPE()
        pe2 = IdentityPE()
        
        sequence = [
            (pe1, 0),   # Starts at 0, cropped to [0, 5) in local time
            (pe2, 5),   # Starts at 5, no upper bound
        ]
        seq = SequencePE(sequence, overlap=False)
        
        self.renderer.set_source(seq)
        
        # Render full range
        snippet = seq.render(0, 8)
        # pe1: cropped to [0, 5) in local time, outputs [0, 1, 2, 3, 4], then zeros
        # pe2: delayed by 5, not cropped. At global time 0-7, DelayPE requests pe.render(-5, 8)
        #      IdentityPE outputs [-5, -4, -3, -2, -1, 0, 1, 2]
        # Mix: [0+(-5), 1+(-4), 2+(-3), 3+(-2), 4+(-1), 0+0, 0+1, 0+2]
        #    = [-5, -3, -1, 1, 3, 0, 1, 2]
        expected = np.array([
            [-5.0], [-3.0], [-1.0], [1.0], [3.0],
            [0.0], [1.0], [2.0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_precise_timing_overlapping(self):
        """Test precise timing with overlapping sequence."""
        pe1 = IdentityPE()
        pe2 = IdentityPE()
        
        sequence = [
            (pe1, 0),
            (pe2, 1),  # Overlaps
        ]
        seq = SequencePE(sequence, overlap=True)
        
        self.renderer.set_source(seq)
        
        # Render full range
        snippet = seq.render(0, 5)
        # pe1 (delayed by 0): at global time t, outputs t
        # pe2 (delayed by 1): at global time t, outputs t-1
        # Mix:
        #   Global time 0: pe1=0, pe2=-1 (requested at -1) → 0+(-1) = -1
        #   Global time 1: pe1=1, pe2=0 → 1+0 = 1
        #   Global time 2: pe1=2, pe2=1 → 2+1 = 3
        #   Global time 3: pe1=3, pe2=2 → 3+2 = 5
        #   Global time 4: pe1=4, pe2=3 → 4+3 = 7
        expected = np.array([
            [-1.0], [1.0], [3.0], [5.0], [7.0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
