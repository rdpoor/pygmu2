"""
Tests for SlicePE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2 import ArrayPE, NullRenderer, SlicePE


class TestSlicePEBasics:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=48_000)

    def test_slice_extracts_and_shifts_to_zero(self):
        src = ArrayPE(np.arange(10, dtype=np.float32))
        sl = SlicePE(src, start=2, duration=5)
        self.renderer.set_source(sl)

        y = sl.render(0, 5).data[:, 0]
        np.testing.assert_array_equal(y, np.array([2, 3, 4, 5, 6], dtype=np.float32))

    def test_slice_zeros_outside_extent(self):
        src = ArrayPE(np.arange(10, dtype=np.float32))
        sl = SlicePE(src, start=2, duration=5)
        self.renderer.set_source(sl)

        y = sl.render(-2, 9).data[:, 0]
        # output time -2..6 overlaps slice time 0..4 at indices 2..6
        expected = np.array([0, 0, 2, 3, 4, 5, 6, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(y, expected)

    def test_slice_applies_fade_in_out(self):
        src = ArrayPE(np.arange(10, dtype=np.float32))
        sl = SlicePE(src, start=2, duration=5, fade_in_samples=2, fade_out_samples=2)
        self.renderer.set_source(sl)

        y = sl.render(0, 5).data[:, 0]
        # slice is [2,3,4,5,6]
        # envelope: [0.5,1.0,1.0,0.5,0.0]
        expected = np.array([1.0, 3.0, 4.0, 2.5, 0.0], dtype=np.float32)
        np.testing.assert_allclose(y, expected, atol=1e-6, rtol=0.0)

