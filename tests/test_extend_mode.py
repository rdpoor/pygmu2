"""
Test extend_mode for appropriate Processing Elements

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np

from pygmu2 import (
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

