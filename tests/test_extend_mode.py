"""
Test extend_mode for appropriate Processing Elements

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np

from pygmu2 import (
    SequencePE,
    ArrayPE,
    NullRenderer,
    Extent,
    ExtendMode,
    RampPE,
    CropPE,
)

class TestArrayExtendMode:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_array_extensions(self):
        # extend_mode = ZERO
        # sample index:    -2 -1   0   1   2   3  4  5
        # values:           0, 0, 10, 11, 12, 13, 0, 0
        array_stream = ArrayPE([10, 11, 12, 13], extend_mode=ExtendMode.ZERO)

        self.renderer.set_source(array_stream)
        snippet = array_stream.render(-2, 8)
        expected = np.array([[0], [0], [10], [11], [12], [13], [0], [0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)

        # extend_mode = HOLD_FIRST
        # sample index:     -2  -1   0   1   2   3  4  5
        # values:           10, 10, 10, 11, 12, 13, 0, 0
        array_stream = ArrayPE([10, 11, 12, 13], extend_mode=ExtendMode.HOLD_FIRST)

        self.renderer.set_source(array_stream)
        snippet = array_stream.render(-2, 8)
        expected = np.array([[10], [10], [10], [11], [12], [13], [0], [0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)

        # extend_mode = HOLD_LAST
        # sample index:     -2  -1   0   1   2   3   4   5
        # values:            0,  0, 10, 11, 12, 13, 13, 13
        array_stream = ArrayPE([10, 11, 12, 13], extend_mode=ExtendMode.HOLD_LAST)

        self.renderer.set_source(array_stream)
        snippet = array_stream.render(-2, 8)
        expected = np.array([[0], [0], [10], [11], [12], [13], [13], [13]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)

        # extend_mode = HOLD_BOTH
        # sample index:     -2  -1   0   1   2   3   4   5
        # values:           10, 10, 10, 11, 12, 13, 13, 13
        array_stream = ArrayPE([10, 11, 12, 13], extend_mode=ExtendMode.HOLD_BOTH)

        self.renderer.set_source(array_stream)
        snippet = array_stream.render(-2, 8)
        expected = np.array([[10], [10], [10], [11], [12], [13], [13], [13]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)


class TestRampExtendMode:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_ramp_extensions(self):
        # Ramp from 10.0 to 20.0 over 4 samples (indices 0-3)
        # extend_mode = ZERO
        # sample index:    -2 -1   0     1     2     3   4  5
        # values:           0, 0, 10.0, 13.33, 16.67, 20.0, 0, 0
        ramp_stream = RampPE(10.0, 20.0, duration=4, extend_mode=ExtendMode.ZERO)

        self.renderer.set_source(ramp_stream)
        snippet = ramp_stream.render(-2, 8)
        expected = np.array([[0.0], [0.0], [10.0], [13.333333], [16.666667], [20.0], [0.0], [0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=4)

        # extend_mode = HOLD_FIRST
        # sample index:     -2   -1    0     1     2     3   4  5
        # values:           10.0, 10.0, 10.0, 13.33, 16.67, 20.0, 0, 0
        ramp_stream = RampPE(10.0, 20.0, duration=4, extend_mode=ExtendMode.HOLD_FIRST)

        self.renderer.set_source(ramp_stream)
        snippet = ramp_stream.render(-2, 8)
        expected = np.array([[10.0], [10.0], [10.0], [13.333333], [16.666667], [20.0], [0.0], [0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=4)

        # extend_mode = HOLD_LAST
        # sample index:     -2 -1   0     1     2     3     4     5
        # values:            0, 0, 10.0, 13.33, 16.67, 20.0, 20.0, 20.0
        ramp_stream = RampPE(10.0, 20.0, duration=4, extend_mode=ExtendMode.HOLD_LAST)

        self.renderer.set_source(ramp_stream)
        snippet = ramp_stream.render(-2, 8)
        expected = np.array([[0.0], [0.0], [10.0], [13.333333], [16.666667], [20.0], [20.0], [20.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=4)

        # extend_mode = HOLD_BOTH
        # sample index:     -2   -1    0     1     2     3     4     5
        # values:           10.0, 10.0, 10.0, 13.33, 16.67, 20.0, 20.0, 20.0
        ramp_stream = RampPE(10.0, 20.0, duration=4, extend_mode=ExtendMode.HOLD_BOTH)

        self.renderer.set_source(ramp_stream)
        snippet = ramp_stream.render(-2, 8)
        expected = np.array([[10.0], [10.0], [10.0], [13.333333], [16.666667], [20.0], [20.0], [20.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=4)


class TestCropExtendMode:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_crop_extensions(self):
        # Source: ArrayPE with values [10, 11, 12, 13] at indices 0-3
        # Crop to [1, 3) (indices 1-2)
        source = ArrayPE([10, 11, 12, 13])

        # extend_mode = ZERO
        # sample index:    -2 -1   0   1   2   3  4  5
        # values:           0, 0,  0, 11, 12, 0, 0, 0
        cropped = CropPE(source, Extent(1, 3), extend_mode=ExtendMode.ZERO)

        self.renderer.set_source(cropped)
        snippet = cropped.render(-2, 8)
        expected = np.array([[0], [0], [0], [11], [12], [0], [0], [0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)

        # extend_mode = HOLD_FIRST
        # sample index:     -2  -1   0   1   2   3  4  5
        # values:           11, 11, 11, 11, 12, 0, 0, 0
        cropped = CropPE(source, Extent(1, 3), extend_mode=ExtendMode.HOLD_FIRST)

        self.renderer.set_source(cropped)
        snippet = cropped.render(-2, 8)
        expected = np.array([[11], [11], [11], [11], [12], [0], [0], [0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)

        # extend_mode = HOLD_LAST
        # sample index:     -2 -1   0   1   2   3   4   5
        # values:            0, 0,  0, 11, 12, 12, 12, 12
        cropped = CropPE(source, Extent(1, 3), extend_mode=ExtendMode.HOLD_LAST)

        self.renderer.set_source(cropped)
        snippet = cropped.render(-2, 8)
        expected = np.array([[0], [0], [0], [11], [12], [12], [12], [12]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)

        # extend_mode = HOLD_BOTH
        # sample index:     -2  -1   0   1   2   3   4   5
        # values:           11, 11, 11, 11, 12, 12, 12, 12
        cropped = CropPE(source, Extent(1, 3), extend_mode=ExtendMode.HOLD_BOTH)

        self.renderer.set_source(cropped)
        snippet = cropped.render(-2, 8)
        expected = np.array([[11], [11], [11], [11], [12], [12], [12], [12]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)

class TestSingletonSequenceExtendMode:
    """
    Test SequencePE behavior outside its extent.
    
    Note: SequencePE no longer supports extend_mode parameter. It always outputs
    zeros outside its extent. To achieve extend_mode behavior, wrap SequencePE
    in a CropPE with the desired extend_mode.
    """
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_sequence_always_outputs_zeros_outside_extent(self):
        """
        SequencePE always outputs zeros outside its extent (no extend_mode support).
        
        To achieve extend_mode behavior, wrap in CropPE:
        sequence = SequencePE([(pe, start_time)])
        extended = CropPE(sequence, sequence.extent(), extend_mode=ExtendMode.HOLD_BOTH)
        """
        # Source: ArrayPE with values [10, 11, 12, 13] at indices 0-3
        source = ArrayPE([10, 11, 12, 13])

        # Sequence with delay=0: extent is [0, 4)
        delay = 0
        sequence_stream = SequencePE([(source, delay)])
        self.renderer.set_source(sequence_stream)
        
        # Render before extent: should output zeros
        snippet_before = sequence_stream.render(-2, 2)
        expected_before = np.array([[0], [0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet_before.data, expected_before)
        
        # Render during extent: should output array values
        snippet_during = sequence_stream.render(0, 4)
        expected_during = np.array([[10], [11], [12], [13]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet_during.data, expected_during)
        
        # Render after extent: should output zeros
        snippet_after = sequence_stream.render(4, 2)
        expected_after = np.array([[0], [0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet_after.data, expected_after)
        
        # Render spanning extent: should output zeros before, values during, zeros after
        snippet_spanning = sequence_stream.render(-2, 8)
        expected_spanning = np.array([[0], [0], [10], [11], [12], [13], [0], [0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet_spanning.data, expected_spanning)

    def test_sequence_with_delay_outputs_zeros_outside_extent(self):
        """SequencePE with delayed item outputs zeros outside its extent."""
        source = ArrayPE([10, 11, 12, 13])
        
        # Sequence with delay=1: extent is [1, 5)
        delay = 1
        sequence_stream = SequencePE([(source, delay)])
        self.renderer.set_source(sequence_stream)
        
        # Render spanning extent
        snippet = sequence_stream.render(-2, 8)
        expected = np.array([[0], [0], [0], [10], [11], [12], [13], [0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)

    def test_sequence_extend_mode_via_crop_pe(self):
        """
        Demonstrate how to achieve extend_mode behavior using CropPE wrapper.
        
        This shows the recommended pattern for users who need extend_mode behavior.
        """
        source = ArrayPE([10, 11, 12, 13])
        
        # Create sequence
        sequence = SequencePE([(source, 0)])
        self.renderer.set_source(sequence)
        
        # Get sequence extent
        seq_extent = sequence.extent()
        assert seq_extent.start == 0
        assert seq_extent.end == 4
        
        # Wrap in CropPE with HOLD_BOTH to achieve extend_mode behavior
        extended = CropPE(sequence, seq_extent, extend_mode=ExtendMode.HOLD_BOTH)
        self.renderer.set_source(extended)
        
        # Render spanning extent: should hold first value before, last value after
        snippet = extended.render(-2, 8)
        expected = np.array([[10], [10], [10], [11], [12], [13], [13], [13]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected)


