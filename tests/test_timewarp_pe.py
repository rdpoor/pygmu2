"""
Tests for TimeWarpPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors
MIT License
"""

import numpy as np
import pytest

from pygmu2 import (
    ArrayPE,
    ConstantPE,
    CropPE,
    Extent,
    IdentityPE,
    NullRenderer,
    TimeWarpPE,
)


class TestTimeWarpPEBasics:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_constant_rate_double_speed(self):
        tw = TimeWarpPE(IdentityPE(), rate=2.0)
        self.renderer.set_source(tw)
        self.renderer.start()

        snip = tw.render(0, 5)
        np.testing.assert_allclose(snip.data[:, 0], np.array([0, 2, 4, 6, 8], dtype=np.float32))

        self.renderer.stop()

    def test_constant_rate_half_speed(self):
        tw = TimeWarpPE(IdentityPE(), rate=0.5)
        self.renderer.set_source(tw)
        self.renderer.start()

        snip = tw.render(0, 4)
        np.testing.assert_allclose(snip.data[:, 0], np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32))

        self.renderer.stop()

    def test_dynamic_rate_integration(self):
        rate = ArrayPE([1, 1, 2, 2])
        tw = TimeWarpPE(IdentityPE(), rate=rate)
        self.renderer.set_source(tw)
        self.renderer.start()

        snip = tw.render(0, 4)
        # indices: 0, 1, 2, 4
        np.testing.assert_allclose(snip.data[:, 0], np.array([0, 1, 2, 4], dtype=np.float32))

        self.renderer.stop()

    def test_stateful_contiguous_renders_continue(self):
        tw = TimeWarpPE(IdentityPE(), rate=1.0)
        self.renderer.set_source(tw)
        self.renderer.start()

        a = tw.render(0, 3).data[:, 0]
        b = tw.render(3, 3).data[:, 0]
        np.testing.assert_allclose(a, np.array([0, 1, 2], dtype=np.float32))
        np.testing.assert_allclose(b, np.array([3, 4, 5], dtype=np.float32))

        self.renderer.stop()

    def test_reset_state_rewinds_head(self):
        tw = TimeWarpPE(IdentityPE(), rate=1.0)
        self.renderer.set_source(tw)
        self.renderer.start()

        _ = tw.render(0, 3)
        tw.reset_state()
        snip = tw.render(0, 3)
        np.testing.assert_allclose(snip.data[:, 0], np.array([0, 1, 2], dtype=np.float32))

        self.renderer.stop()

    def test_out_of_bounds_zeros(self):
        # Finite source: only indices in [0, 5) are valid.
        source = ArrayPE([10, 11, 12, 13, 14])
        tw = TimeWarpPE(source, rate=-1.0)
        self.renderer.set_source(tw)
        self.renderer.start()

        snip = tw.render(0, 4)
        # indices: 0, -1, -2, -3 => only first in-bounds
        np.testing.assert_allclose(snip.data[:, 0], np.array([10, 0, 0, 0], dtype=np.float32))

        self.renderer.stop()

    def test_extent_constant_rate_finite_source(self):
        source = ArrayPE(np.arange(10, dtype=np.float32))
        tw = TimeWarpPE(source, rate=2.0)
        ext = tw.extent()
        assert ext == Extent(0, 5)

    def test_extent_dynamic_rate_matches_rate_extent(self):
        rate = CropPE(ConstantPE(1.0), Extent(0, 7))
        tw = TimeWarpPE(IdentityPE(), rate=rate)
        ext = tw.extent()
        assert ext == Extent(0, 7)

