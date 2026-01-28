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
    ExtendMode,
    ErrorMode,
    set_error_mode,
    get_error_mode,
    RampPE,
    CropPE,
    MixPE,
    DelayPE,
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
        """Test that inputs() returns the internal MixPE (or DelayPE for single items)."""
        pe1 = ConstantPE(0.5)
        pe2 = ConstantPE(0.3)
        seq = SequencePE([(pe1, 0), (pe2, 100)])

        inputs = seq.inputs()
        # Should return the internal MixPE (which wraps the delayed PEs)
        assert len(inputs) == 1
        assert isinstance(inputs[0], MixPE)
        
        # For single item, should return DelayPE directly
        seq_single = SequencePE([(pe1, 0)])
        inputs_single = seq_single.inputs()
        assert len(inputs_single) == 1
        assert isinstance(inputs_single[0], DelayPE)

    def test_scalar_auto_cropping(self):
        """Test that scalars are automatically cropped to next item's start."""
        # Scalars should be auto-cropped to the next item's start time
        seq = SequencePE([(1.0, 0), (2.0, 10), (3.0, 20)])
        self.renderer.set_source(seq)
        
        # First scalar (1.0) should be cropped to [0, 10) - outputs 1.0 for 10 samples
        snippet1 = seq.render(0, 10)
        np.testing.assert_array_almost_equal(
            snippet1.data[:, 0],
            np.full(10, 1.0, dtype=np.float32)
        )
        
        # Second scalar (2.0) should be cropped to [0, 10) in local time - outputs 2.0 for 10 samples
        snippet2 = seq.render(10, 10)
        np.testing.assert_array_almost_equal(
            snippet2.data[:, 0],
            np.full(10, 2.0, dtype=np.float32)
        )
        
        # Third scalar (3.0) is last item - has infinite extent, outputs 3.0 indefinitely
        snippet3 = seq.render(20, 10)
        np.testing.assert_array_almost_equal(
            snippet3.data[:, 0],
            np.full(10, 3.0, dtype=np.float32)
        )
        
        # Verify first scalar doesn't contribute after its crop end
        snippet4 = seq.render(10, 1)
        assert snippet4.data[0, 0] == pytest.approx(2.0, abs=0.1)  # Only second scalar

    def test_scalar_step_sequence(self):
        """
        SequencePE accepts scalars and produces a step-like control sequence.
        Scalars are automatically cropped to the next item's start time.
        """
        seq = SequencePE([(0.0, 0), (1.0, 5), (2.0, 10)])
        self.renderer.set_source(seq)

        y = seq.render(0, 15).data[:, 0]
        expected = np.array([0] * 5 + [1] * 5 + [2] * 5, dtype=np.float32)
        np.testing.assert_array_equal(y, expected)

    def test_scalar_step_sequence_multichannel(self):
        """Scalar-only sequences can specify channels=."""
        seq = SequencePE([(0.0, 0), (1.0, 5)], channels=2)
        self.renderer.set_source(seq)

        y = seq.render(0, 10).data
        assert y.shape == (10, 2)
        np.testing.assert_array_equal(y[:5, :], np.zeros((5, 2), dtype=np.float32))
        np.testing.assert_array_equal(y[5:, :], np.ones((5, 2), dtype=np.float32))

    def test_scalar_sequence_matches_pe_channels(self):
        """If a PE is present, scalar items adopt that PE's channel count."""
        pe2 = ConstantPE(0.25, channels=2)
        seq = SequencePE([(pe2, 0), (1.0, 5)])
        self.renderer.set_source(seq)

        y = seq.render(0, 6).data
        assert y.shape == (6, 2)

    def test_channels_must_match_pe(self):
        """Explicit channels= must match any PE channel count."""
        pe2 = ConstantPE(0.25, channels=2)
        with pytest.raises(ValueError):
            SequencePE([(pe2, 0), (1.0, 5)], channels=1)
    
    def test_repr(self):
        """Test string representation."""
        pe = ConstantPE(0.5)
        seq = SequencePE([(pe, 0)])
        repr_str = repr(seq)
        assert "SequencePE" in repr_str
        assert "items" in repr_str


class TestSequencePERender:
    """Test SequencePE rendering behavior."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_non_overlapping_sequence(self):
        """Test non-overlapping sequence with pre-cropped inputs."""
        # Use IdentityPE: outputs sample index, making it easy to verify timing
        # Pre-crop each PE to desired duration for non-overlapping behavior
        pe1 = CropPE(IdentityPE(), Extent(0, 5))
        pe2 = CropPE(IdentityPE(), Extent(0, 5))
        pe3 = CropPE(IdentityPE(), Extent(0, 5))
        
        sequence = [
            (pe1, 0),   # Starts at 0, pre-cropped to [0, 5) in local time
            (pe2, 5),   # Starts at 5, pre-cropped to [0, 5) in local time
            (pe3, 10),  # Starts at 10, pre-cropped to [0, 5) in local time
        ]
        seq = SequencePE(sequence)
        
        self.renderer.set_source(seq)
        
        # Render first segment: 
        # pe1 (delayed 0, cropped [0,5)): outputs [0, 1, 2, 3, 4]
        # pe2 (delayed 5, cropped [0,5)): DelayPE requests render(-5, 5), CropPE zeros it → [0, 0, 0, 0, 0]
        # pe3 (delayed 10, cropped): DelayPE requests render(-10, 5), CropPE zeros it → [0, 0, 0, 0, 0]
        # Mix: [0+0+0, 1+0+0, 2+0+0, 3+0+0, 4+0+0] = [0, 1, 2, 3, 4]
        snippet1 = seq.render(0, 5)
        expected1 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet1.data, expected1)
        
        # Render second segment:
        # pe1 (delayed 0, cropped [0,5)): DelayPE requests render(5-0, 5) = render(5, 5), CropPE zeros it → [0, 0, 0, 0, 0]
        # pe2 (delayed 5, cropped [0,5)): DelayPE requests render(5-5, 5) = render(0, 5), outputs [0, 1, 2, 3, 4]
        # pe3 (delayed 10, cropped): DelayPE requests render(5-10, 5) = render(-5, 5), CropPE zeros it → [0, 0, 0, 0, 0]
        # Mix: [0+0+0, 0+1+0, 0+2+0, 0+3+0, 0+4+0] = [0, 1, 2, 3, 4]
        snippet2 = seq.render(5, 5)
        expected2 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet2.data, expected2)
        
        # Render third segment:
        # pe1 (delayed 0, cropped [0,5)): DelayPE requests render(10-0, 5) = render(10, 5), CropPE zeros it → [0, 0, 0, 0, 0]
        # pe2 (delayed 5, cropped [0,5)): DelayPE requests render(10-5, 5) = render(5, 5), CropPE zeros it → [0, 0, 0, 0, 0]
        # pe3 (delayed 10, cropped): DelayPE requests render(10-10, 5) = render(0, 5), outputs [0, 1, 2, 3, 4]
        # Mix: [0+0+0, 0+0+1, 0+0+2, 0+0+3, 0+0+4] = [0, 1, 2, 3, 4]
        snippet3 = seq.render(10, 5)
        expected3 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet3.data, expected3)
    
    def test_overlapping_sequence(self):
        """Test overlapping sequence (overlap is always allowed)."""
        pe1 = IdentityPE()
        pe2 = IdentityPE()
        
        sequence = [
            (pe1, 0),
            (pe2, 1),  # Starts before pe1 ends
        ]
        seq = SequencePE(sequence)
        
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
        seq = SequencePE(sequence)
        
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
        seq = SequencePE(sequence)
        
        extent = seq.extent()
        # Should be union: [0, 100) U [200, 300)
        # Since extents don't overlap, union should cover both
        assert extent.start == 0
        assert extent.end == 300
    
    def test_channel_count_validation(self):
        """Test that channel count mismatch causes error during construction."""
        pe1 = ConstantPE(0.5, channels=1)
        pe2 = ConstantPE(0.3, channels=2)  # Mismatch
        
        sequence = [(pe1, 0), (pe2, 100)]
        
        # Channel validation happens during SequencePE construction
        # The first PE (pe1) sets base_channels=1, then pe2 is validated against it
        with pytest.raises(ValueError, match="SequencePE input channel mismatch"):
            SequencePE(sequence)


class TestSequencePEExplicitDuration:
    """Test SequencePE with explicit durations (3-tuple format)."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_explicit_duration_non_overlapping(self):
        """Test 3-tuple format with explicit durations, overlap=False."""
        pe1 = IdentityPE()
        pe2 = IdentityPE()
        
        # Specify explicit durations
        sequence = [
            (pe1, 0, 5),    # Starts at 0, plays for 5 samples
            (pe2, 10, 3),   # Starts at 10, plays for 3 samples
        ]
        seq = SequencePE(sequence, overlap=False)
        
        self.renderer.set_source(seq)
        
        # Render first segment (pe1 active)
        snippet1 = seq.render(0, 5)
        # pe1 cropped to 5 samples: [0, 1, 2, 3, 4]
        # pe2 delayed by 10, requests render(-10, 5), cropped to 3 → [0, 0, 0, 0, 0]
        expected1 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet1.data, expected1)
        
        # Render second segment (pe2 active)
        snippet2 = seq.render(10, 3)
        # pe1 cropped to 5, delayed by 0: requests render(10, 3) → [0, 0, 0] (out of crop range)
        # pe2 cropped to 3, delayed by 10: requests render(0, 3) → [0, 1, 2]
        expected2 = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet2.data, expected2)
    
    def test_explicit_duration_overlapping(self):
        """Test 3-tuple format with explicit durations, overlap=True."""
        pe1 = IdentityPE()
        pe2 = IdentityPE()
        
        # Overlapping sequence with explicit durations
        sequence = [
            (pe1, 0, 10),   # Starts at 0, plays for 10 samples
            (pe2, 5, 10),   # Starts at 5, plays for 10 samples (overlaps pe1)
        ]
        seq = SequencePE(sequence, overlap=True)
        
        self.renderer.set_source(seq)
        
        # Render overlap region
        snippet = seq.render(5, 5)
        # pe1 (delayed 0, cropped to 10): outputs [5, 6, 7, 8, 9]
        # pe2 (delayed 5, cropped to 10): outputs [0, 1, 2, 3, 4]
        # Mix: [5+0, 6+1, 7+2, 8+3, 9+4] = [5, 7, 9, 11, 13]
        expected = np.array([[5.0], [7.0], [9.0], [11.0], [13.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_mixed_2_and_3_tuples(self):
        """Test mixing 2-tuple and 3-tuple formats in same sequence."""
        pe1 = IdentityPE()
        pe2 = IdentityPE()
        pe3 = IdentityPE()
        
        # Mixed format
        sequence = [
            (pe1, 0, 5),    # 3-tuple: explicit duration
            (pe2, 5),       # 2-tuple: inferred duration from next item
            (pe3, 10),      # 2-tuple: no cropping (last item)
        ]
        seq = SequencePE(sequence, overlap=False)
        
        self.renderer.set_source(seq)
        
        # Check that inputs are correctly extracted
        inputs = seq.inputs()
        assert len(inputs) == 3
        assert pe1 in inputs
        assert pe2 in inputs
        assert pe3 in inputs
        
        # Render to verify correct processing
        snippet = seq.render(0, 15)
        # All three PEs are mixed together (DelayPE + optional CropPE + MixPE)
        # pe1: delayed 0, cropped to 5. At global time t, outputs t if t<5, else 0
        # pe2: delayed 5, cropped to 5. At global time t, outputs (t-5) if 5<=t<10, else 0
        # pe3: delayed 10, not cropped. At global time t, outputs (t-10)
        # Mix all three:
        expected = np.array([
            [-10.0], [-8.0], [-6.0], [-4.0], [-2.0],  # t=0-4: pe1=[0,1,2,3,4] + pe2=[0] + pe3=[-10,-9,-8,-7,-6]
            [-5.0], [-3.0], [-1.0], [1.0], [3.0],     # t=5-9: pe1=[0] + pe2=[0,1,2,3,4] + pe3=[-5,-4,-3,-2,-1]
            [0.0], [1.0], [2.0], [3.0], [4.0],        # t=10-14: pe1=[0] + pe2=[0] + pe3=[0,1,2,3,4]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_explicit_duration_zero(self):
        """Test 3-tuple with zero duration (should not crop)."""
        pe1 = IdentityPE()
        pe2 = IdentityPE()
        
        sequence = [
            (pe1, 0, 0),    # Zero duration (should skip crop)
            (pe2, 5, 5),    # Non-zero duration
        ]
        seq = SequencePE(sequence, overlap=False)
        
        self.renderer.set_source(seq)
        
        # Render snippet
        snippet = seq.render(0, 10)
        # pe1: delayed by 0, not cropped (duration=0 skips crop check)
        #      At global time t, outputs t
        # pe2: delayed by 5, cropped to 5 samples
        #      At global time t, outputs (t-5) if 5<=t<10, else 0
        # Mix:
        #   t=0: pe1=0, pe2=-5 (cropped to 0) → 0
        #   t=1: pe1=1, pe2=-4 (cropped to 0) → 1
        #   ...
        #   t=5: pe1=5, pe2=0 → 5
        #   t=9: pe1=9, pe2=4 → 13
        expected = np.array([
            [0.0], [1.0], [2.0], [3.0], [4.0],
            [5.0], [7.0], [9.0], [11.0], [13.0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)
    
    def test_sorting_with_3_tuples(self):
        """Test that 3-tuples are correctly sorted by start_time."""
        pe1 = ConstantPE(1.0)
        pe2 = ConstantPE(2.0)
        pe3 = ConstantPE(3.0)
        
        # Provide in reverse order with 3-tuples
        sequence = [
            (pe3, 200, 50),
            (pe1, 0, 50),
            (pe2, 100, 50)
        ]
        seq = SequencePE(sequence)
        
        # Should be sorted by start_time
        sorted_seq = seq.sequence
        assert sorted_seq[0][1] == 0
        assert sorted_seq[1][1] == 100
        assert sorted_seq[2][1] == 200
        # Verify durations are preserved
        assert sorted_seq[0][2] == 50
        assert sorted_seq[1][2] == 50
        assert sorted_seq[2][2] == 50


class TestSequencePESampleAccurate:
    """Test sample-accurate behavior of SequencePE."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_precise_timing_non_overlapping(self):
        """Test precise timing with non-overlapping sequence (pre-cropped inputs)."""
        # Use IdentityPE: outputs sample index, making timing verification straightforward
        # Pre-crop inputs for non-overlapping behavior
        pe1 = CropPE(IdentityPE(), Extent(0, 5))
        pe2 = CropPE(IdentityPE(), Extent(0, 3))
        
        sequence = [
            (pe1, 0),   # Starts at 0, pre-cropped to [0, 5) in local time
            (pe2, 5),   # Starts at 5, pre-cropped to [0, 3) in local time
        ]
        seq = SequencePE(sequence)
        
        self.renderer.set_source(seq)
        
        # Render full range
        snippet = seq.render(0, 8)
        # pe1: cropped to [0, 5) in local time, outputs [0, 1, 2, 3, 4], then zeros
        # pe2: delayed by 5, cropped. At global time 0-7, DelayPE requests pe.render(-5, 8)
        #      CropPE zeros negative indices, then outputs [0, 1, 2] for indices 0-2
        # Mix: [0+0, 1+0, 2+0, 3+0, 4+0, 0+0, 0+1, 0+2]
        #    = [0, 1, 2, 3, 4, 0, 1, 2]
        expected = np.array([
            [0.0], [1.0], [2.0], [3.0], [4.0],
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
        seq = SequencePE(sequence)
        
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


class TestSequencePERegression:
    """Regression tests for SequencePE with RampPE and extend_mode."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_sequence_pe_crops_ramp_with_infinite_extent(self):
        """
        Regression test: SequencePE with pre-cropped RampPE items should work correctly.
        
        For non-overlapping behavior, callers must pre-crop infinite-extent PEs.
        """
        # Create ramps with extend_mode=ExtendMode.HOLD_BOTH (infinite extent)
        ramp1 = RampPE(10.0, 20.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        ramp2 = RampPE(30.0, 40.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        
        # Verify they have infinite extent
        assert ramp1.extent().start is None
        assert ramp1.extent().end is None
        
        # Pre-crop ramps for non-overlapping behavior
        cropped_ramp1 = CropPE(ramp1, Extent(0, 200))
        sequence = [
            (cropped_ramp1, 0),      # Pre-cropped to [0, 200) in local time
            (ramp2, 200),            # Not cropped (last item, infinite extent)
        ]
        seq = SequencePE(sequence)
        
        self.renderer.set_source(seq)
        
        # Render well after ramp1's cropped range
        # At global time 500:
        # - cropped_ramp1 (delay=0, crop=[0,200)): local time 500 (after crop) → 0.0
        # - ramp2 (delay=200, not cropped, HOLD_BOTH): local time 300 (after ramp completes)
        #   Ramp goes from 30.0 to 40.0 over 100 samples, so at local time 300 it's well past
        #   the ramp end, so with HOLD_BOTH it holds the end value (40.0)
        # Expected: 40.0 (only ramp2 contributing)
        snippet = seq.render(500, 50)
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            np.full(50, 40.0, dtype=np.float32),
            decimal=5
        )
        
        # Critical assertion: should be 40.0, NOT 20.0 + 40.0 = 60.0
        assert snippet.data[0, 0] != pytest.approx(60.0, abs=0.1)
    
    def test_sequence_pe_cropped_ramp_outputs_zeros_after_crop(self):
        """
        Regression test: When SequencePE crops a RampPE with extend_mode=ExtendMode.HOLD_BOTH,
        the cropped RampPE should output zeros after the crop end, not hold values.
        """
        ramp = RampPE(10.0, 20.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        
        # Manually crop it (simulating what SequencePE does)
        cropped = CropPE(ramp, Extent(0, 200))
        
        self.renderer.set_source(cropped)
        
        # Render within crop window
        snippet1 = cropped.render(50, 50)
        # Should get ramp values
        assert snippet1.data[0, 0] == pytest.approx(15.0, abs=0.5)
        
        # Render after crop window - should be zeros
        snippet2 = cropped.render(250, 50)
        np.testing.assert_array_almost_equal(
            snippet2.data,
            np.zeros((50, 1), dtype=np.float32),
            decimal=5
        )
    
    def test_sequence_pe_multiple_ramps_no_mixing(self):
        """
        Regression test: Multiple pre-cropped RampPE items in SequencePE should not mix
        incorrectly.
        
        For non-overlapping behavior, callers must pre-crop each item.
        """
        ramps = [
            RampPE(10.0, 20.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH),
            RampPE(30.0, 40.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH),
            RampPE(50.0, 60.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH),
        ]
        
        # Pre-crop first two ramps for non-overlapping behavior
        cropped_ramp1 = CropPE(ramps[0], Extent(0, 1000))
        cropped_ramp2 = CropPE(ramps[1], Extent(0, 1000))
        sequence = [
            (cropped_ramp1, 0),      # Pre-cropped to [0, 1000)
            (cropped_ramp2, 1000),   # Pre-cropped to [0, 1000) in local time
            (ramps[2], 2000),        # Not cropped (last item, infinite extent)
        ]
        seq = SequencePE(sequence)
        
        self.renderer.set_source(seq)
        
        # Render well after first two items should have ended
        # At global time 2500:
        # - cropped_ramp1 (delay=0, crop=[0,1000)): local time 2500 (after crop) → 0.0
        # - cropped_ramp2 (delay=1000, crop=[0,1000)): local time 1500 (after crop) → 0.0
        # - ramp2 (delay=2000, not cropped, HOLD_BOTH): local time 500 (after ramp completes)
        #   Ramp goes from 50.0 to 60.0 over 100 samples, so at local time 500 it's well past
        #   the ramp end, so with HOLD_BOTH it holds the end value (60.0)
        # Expected: 60.0 (only ramp2 contributing)
        snippet = seq.render(2500, 50)
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            np.full(50, 60.0, dtype=np.float32),
            decimal=5
        )
        
        # Critical assertions
        assert snippet.data[0, 0] != pytest.approx(120.0, abs=0.1)  # Not sum of all
        assert snippet.data[0, 0] != pytest.approx(100.0, abs=0.1)  # Not sum of first two
        assert snippet.data[0, 0] == pytest.approx(60.0, abs=0.1)    # Only last item
    
    def test_sequence_pe_constant_and_ramp_mimics_portamento(self):
        """
        Regression test: SequencePE with pre-cropped ConstantPE and RampPE should work
        correctly, mimicking PortamentoPE's internal structure.
        
        For non-overlapping behavior, callers must pre-crop each item.
        """
        # Simulate PortamentoPE's structure:
        # - ConstantPE for first note
        # - RampPE for transitions (with extend_mode=ExtendMode.HOLD_BOTH)
        const1 = ConstantPE(77.0)  # First pitch
        ramp1 = RampPE(77.0, 74.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)  # Transition
        const2 = ConstantPE(70.0)  # Same pitch (no ramp)
        ramp2 = RampPE(70.0, 75.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)  # Transition
        
        # Pre-crop first three items for non-overlapping behavior
        cropped_const1 = CropPE(const1, Extent(0, 1000))
        cropped_ramp1 = CropPE(ramp1, Extent(0, 1000))
        cropped_const2 = CropPE(const2, Extent(0, 1000))
        sequence = [
            (cropped_const1, 0),      # Pre-cropped to [0, 1000)
            (cropped_ramp1, 1000),    # Pre-cropped to [0, 1000) in local time
            (cropped_const2, 2000),   # Pre-cropped to [0, 1000) in local time
            (ramp2, 3000),            # Not cropped (last item, infinite extent)
        ]
        seq = SequencePE(sequence)
        
        self.renderer.set_source(seq)
        
        # Render first region: should get const1 value + ramp2's start value (infinite extent)
        # At global time 500:
        # - cropped_const1 (delay=0, crop=[0,1000)): local time 500 (within crop) → 77.0
        # - cropped_ramp1 (delay=1000, crop=[0,1000)): local time -500 (before crop) → 0.0
        # - cropped_const2 (delay=2000, crop=[0,1000)): local time -1500 (before crop) → 0.0
        # - ramp2 (delay=3000, not cropped, HOLD_BOTH): local time -2500 (before start) → 70.0 (start value)
        # Expected: 77.0 + 70.0 = 147.0
        snippet1 = seq.render(500, 50)
        np.testing.assert_array_almost_equal(
            snippet1.data[:, 0],
            np.full(50, 147.0, dtype=np.float32),  # const1 (77.0) + ramp2 start (70.0)
            decimal=5
        )
        
        # Render second region: should get ramp1's end value (74.0) + ramp2's start value (70.0)
        # At global time 1500:
        # - cropped_const1 (delay=0, crop=[0,1000)): local time 1500 (after crop) → 0.0
        # - cropped_ramp1 (delay=1000, crop=[0,1000)): local time 500 (within crop)
        #   Ramp goes from 77.0 to 74.0 over 100 samples, so at local time 500 it's well past
        #   the ramp end, so with HOLD_BOTH it holds the end value (74.0)
        # - cropped_const2 (delay=2000, crop=[0,1000)): local time -500 (before crop) → 0.0
        # - ramp2 (delay=3000, not cropped, HOLD_BOTH): local time -1500 (before start) → 70.0 (start value)
        # Expected: 74.0 + 70.0 = 144.0
        snippet2 = seq.render(1500, 50)
        np.testing.assert_array_almost_equal(
            snippet2.data[:, 0],
            np.full(50, 144.0, dtype=np.float32),  # ramp1 end (74.0) + ramp2 start (70.0)
            decimal=5
        )
        
        # Render third region: should get const2 value (70.0) + ramp2's start value (70.0)
        # At global time 2500:
        # - cropped_const1 (delay=0, crop=[0,1000)): local time 2500 (after crop) → 0.0
        # - cropped_ramp1 (delay=1000, crop=[0,1000)): local time 1500 (after crop) → 0.0
        # - cropped_const2 (delay=2000, crop=[0,1000)): local time 500 (within crop) → 70.0
        # - ramp2 (delay=3000, not cropped, HOLD_BOTH): local time -500 (before start) → 70.0 (start value)
        # Expected: 70.0 + 70.0 = 140.0
        snippet3 = seq.render(2500, 50)
        np.testing.assert_array_almost_equal(
            snippet3.data[:, 0],
            np.full(50, 140.0, dtype=np.float32),  # const2 (70.0) + ramp2 start (70.0)
            decimal=5
        )
        
        # Render last region: should get ramp2's end value (75.0)
        # At global time 3500:
        # - All cropped items are after their crop windows → 0.0
        # - ramp2 (delay=3000, not cropped): local time 500 (within ramp, which is 100 samples)
        #   Ramp goes from 70.0 to 75.0 over 100 samples, so at local time 500 it's well past
        #   the ramp end, so with HOLD_BOTH it holds the end value (75.0)
        snippet4 = seq.render(3500, 50)
        np.testing.assert_array_almost_equal(
            snippet4.data[:, 0],
            np.full(50, 75.0, dtype=np.float32),
            decimal=5
        )
    
    def test_sequence_pe_crop_extent_calculation(self):
        """
        Regression test: Verify that pre-cropped items work correctly in SequencePE.
        
        For non-overlapping behavior, callers must pre-crop each item.
        """
        ramp = RampPE(10.0, 20.0, duration=100, extend_mode=ExtendMode.HOLD_BOTH)
        
        # Pre-crop first two ramps for non-overlapping behavior
        cropped_ramp1 = CropPE(ramp, Extent(0, 1000))
        cropped_ramp2 = CropPE(ramp, Extent(0, 1000))
        sequence = [
            (cropped_ramp1, 0),      # Pre-cropped to [0, 1000) in local time
            (cropped_ramp2, 1000),   # Pre-cropped to [0, 1000) in local time
            (ramp, 2000),            # Not cropped (last item, infinite extent)
        ]
        seq = SequencePE(sequence)
        
        # Check that the cropped items have finite extents
        # The internal structure should have CropPE wrapping each ramp
        # (except the last one)
        
        # Render at various points to verify cropping works
        self.renderer.set_source(seq)
        
        # After first item's crop: verify ramp1 is not contributing
        # Check at sample 1000 (when ramp2 starts)
        # At sample 1000, rendering 50 samples:
        # - cropped_ramp1 (delay=0, crop=[0,1000)): local time 1000 (at crop boundary) → 0.0
        # - cropped_ramp2 (delay=1000, crop=[0,1000)): local time 0-49 (ramp values) → [10.0, 10.1, 10.2, ...]
        # - ramp (delay=2000, not cropped, HOLD_BOTH): local time -1000 (before start) → 10.0 (start value, constant)
        # Expected: ramp2 values [10.0, 10.1, 10.2, ...] + 10.0 = [20.0, 20.1, 20.2, ...]
        snippet1 = seq.render(1000, 50)
        # cropped_ramp2 outputs the first 50 samples of the ramp (ramping from 10.0 to ~15.0 over 50 samples)
        # ramp outputs constant 10.0 (HOLD_BOTH before start)
        # Sum: ramp values + 10.0
        expected_ramp2 = np.linspace(10.0, 20.0, 100, dtype=np.float32)[:50]  # First 50 samples of ramp
        expected_ramp = np.full(50, 10.0, dtype=np.float32)  # Constant from uncropped ramp
        expected = expected_ramp2 + expected_ramp
        np.testing.assert_array_almost_equal(
            snippet1.data[:, 0],
            expected,
            decimal=5
        )
        
        # After second item's crop: verify ramp2 is not contributing
        # Check at sample 2000 (when ramp3 starts)
        # At sample 2000, rendering 50 samples:
        # - cropped_ramp1 (delay=0, crop=[0,1000)): local time 2000 (after crop) → 0.0
        # - cropped_ramp2 (delay=1000, crop=[0,1000)): local time 1000 (at crop boundary) → 0.0
        # - ramp (delay=2000, not cropped, HOLD_BOTH): local time 0-49 (ramp values) → [10.0, 10.1, 10.2, ...]
        # Expected: ramp values [10.0, 10.1, 10.2, ...] (only ramp contributing)
        snippet2 = seq.render(2000, 50)
        # ramp outputs the first 50 samples of the ramp (ramping from 10.0 to ~15.0 over 50 samples)
        expected_ramp = np.linspace(10.0, 20.0, 100, dtype=np.float32)[:50]  # First 50 samples of ramp
        np.testing.assert_array_almost_equal(
            snippet2.data[:, 0],
            expected_ramp,
            decimal=5
        )
