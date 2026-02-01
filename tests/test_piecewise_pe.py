"""
Tests for PiecewisePE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    PiecewisePE,
    TransitionType,
    Extent,
    ExtendMode,
    NullRenderer,
)


class TestPiecewisePEBasics:
    """Test basic PiecewisePE creation and properties."""

    def test_create_two_points(self):
        pw = PiecewisePE([(0, 0.0), (100, 1.0)])
        assert pw.points == [(0, 0.0), (100, 1.0)]
        assert pw.transition_type == TransitionType.LINEAR
        assert pw.extend_mode == ExtendMode.ZERO
        assert pw.channel_count() == 1

    def test_create_single_point(self):
        pw = PiecewisePE([(50, 0.5)])
        assert pw.points == [(50, 0.5)]
        assert pw.channel_count() == 1

    def test_points_sorted(self):
        pw = PiecewisePE([(100, 1.0), (0, 0.0)])
        assert pw.points == [(0, 0.0), (100, 1.0)]

    def test_create_with_channels(self):
        pw = PiecewisePE([(0, 0.0), (10, 1.0)], channels=2)
        assert pw.channel_count() == 2

    def test_finite_extent(self):
        pw = PiecewisePE([(10, 0.0), (110, 1.0)])
        extent = pw.extent()
        assert extent.start == 10
        assert extent.end == 110

    def test_infinite_extent_with_hold(self):
        pw = PiecewisePE(
            [(0, 0.0), (100, 1.0)],
            extend_mode=ExtendMode.HOLD_BOTH,
        )
        extent = pw.extent()
        assert extent.start is None
        assert extent.end is None

    def test_is_pure(self):
        pw = PiecewisePE([(0, 0.0), (100, 1.0)])
        assert pw.is_pure() is True

    def test_no_inputs(self):
        pw = PiecewisePE([(0, 0.0), (100, 1.0)])
        assert pw.inputs() == []

    def test_empty_points_raises(self):
        with pytest.raises(ValueError, match="at least one point"):
            PiecewisePE([])

    def test_channels_must_be_positive(self):
        with pytest.raises(ValueError, match="channels must be >= 1"):
            PiecewisePE([(0, 0.0)], channels=0)

    def test_repr(self):
        pw = PiecewisePE([(0, 0.0), (100, 1.0)])
        repr_str = repr(pw)
        assert "PiecewisePE" in repr_str
        assert "LINEAR" in repr_str or "linear" in repr_str


class TestPiecewisePERender:
    """Test PiecewisePE rendering."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_render_returns_snippet(self):
        pw = PiecewisePE([(0, 0.0), (100, 1.0)])
        self.renderer.set_source(pw)
        snippet = pw.render(0, 100)
        assert snippet.start == 0
        assert snippet.duration == 100
        assert snippet.channels == 1

    def test_render_linear_full_segment(self):
        pw = PiecewisePE(
            [(0, 0.0), (100, 1.0)],
            transition_type=TransitionType.LINEAR,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(0, 100)
        assert abs(snippet.data[0, 0] - 0.0) < 1e-6
        assert abs(snippet.data[-1, 0] - 1.0) < 1e-6
        assert abs(snippet.data[49, 0] - 0.5) < 0.02

    def test_render_step(self):
        pw = PiecewisePE(
            [(0, 0.0), (50, 1.0), (100, 0.0)],
            transition_type=TransitionType.STEP,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(0, 100)
        # Segment [0, 50): hold 0
        np.testing.assert_array_almost_equal(snippet.data[:50, 0], np.zeros(50))
        # Segment [50, 100): hold 1
        np.testing.assert_array_almost_equal(snippet.data[50:, 0], np.ones(50))

    def test_render_single_point_constant(self):
        pw = PiecewisePE([(50, 0.7)])
        self.renderer.set_source(pw)
        # Extent is (50, 51); only sample 50 is in range
        snippet = pw.render(50, 1)
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            np.array([0.7], dtype=np.float32),
        )
        # Before: zeros; at 50: 0.7
        snippet2 = pw.render(48, 4)
        assert abs(snippet2.data[0, 0] - 0.0) < 1e-6
        assert abs(snippet2.data[1, 0] - 0.0) < 1e-6
        assert abs(snippet2.data[2, 0] - 0.7) < 1e-6  # sample 50
        assert abs(snippet2.data[3, 0] - 0.0) < 1e-6  # after extent

    def test_render_before_extent_zero(self):
        pw = PiecewisePE([(100, 0.0), (200, 1.0)])
        self.renderer.set_source(pw)
        snippet = pw.render(0, 50)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((50, 1), dtype=np.float32),
        )

    def test_render_after_extent_zero(self):
        pw = PiecewisePE([(0, 0.0), (100, 1.0)])
        self.renderer.set_source(pw)
        snippet = pw.render(100, 50)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((50, 1), dtype=np.float32),
        )

    def test_render_hold_first(self):
        pw = PiecewisePE(
            [(100, 0.0), (200, 1.0)],
            extend_mode=ExtendMode.HOLD_FIRST,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(0, 50)
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            np.zeros(50, dtype=np.float32),
        )

    def test_render_hold_last(self):
        pw = PiecewisePE(
            [(0, 0.0), (100, 1.0)],
            extend_mode=ExtendMode.HOLD_LAST,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(100, 50)
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            np.ones(50, dtype=np.float32),
        )

    def test_render_hold_both_after(self):
        pw = PiecewisePE(
            [(0, 0.0), (100, 1.0)],
            extend_mode=ExtendMode.HOLD_BOTH,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(100, 30)
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            np.ones(30, dtype=np.float32),
        )

    def test_render_hold_both_before(self):
        pw = PiecewisePE(
            [(100, 0.0), (200, 1.0)],
            extend_mode=ExtendMode.HOLD_BOTH,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(50, 50)
        np.testing.assert_array_almost_equal(
            snippet.data[:50, 0],
            np.zeros(50, dtype=np.float32),
        )
        assert abs(snippet.data[50, 0] - 0.0) < 1e-6  # first point

    def test_render_partial_middle(self):
        pw = PiecewisePE(
            [(0, 0.0), (100, 1.0)],
            transition_type=TransitionType.LINEAR,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(25, 50)
        assert abs(snippet.data[0, 0] - 0.25) < 0.02
        assert abs(snippet.data[-1, 0] - 0.74) < 0.02

    def test_render_stereo(self):
        pw = PiecewisePE(
            [(0, 0.0), (100, 1.0)],
            channels=2,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(0, 100)
        assert snippet.channels == 2
        np.testing.assert_array_equal(snippet.data[:, 0], snippet.data[:, 1])

    def test_transition_type_exponential(self):
        pw = PiecewisePE(
            [(0, 0.1), (100, 1.0)],
            transition_type=TransitionType.EXPONENTIAL,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(0, 100)
        assert abs(snippet.data[0, 0] - 0.1) < 1e-5
        assert abs(snippet.data[-1, 0] - 1.0) < 1e-5
        # Exponential: values should increase faster at the end
        assert snippet.data[25, 0] < 0.4  # earlier in curve

    def test_transition_type_sigmoid(self):
        pw = PiecewisePE(
            [(0, 0.0), (100, 1.0)],
            transition_type=TransitionType.SIGMOID,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(0, 100)
        assert abs(snippet.data[0, 0] - 0.0) < 1e-5
        assert abs(snippet.data[-1, 0] - 1.0) < 1e-5
        # Sigmoid: middle should be around 0.5
        assert abs(snippet.data[50, 0] - 0.5) < 0.1

    def test_transition_type_constant_power(self):
        # Fade-in: sin(π/2·t); at t=0.5 value ≈ sin(π/4) ≈ 0.707
        pw = PiecewisePE(
            [(0, 0.0), (100, 1.0)],
            transition_type=TransitionType.CONSTANT_POWER,
        )
        self.renderer.set_source(pw)
        snippet = pw.render(0, 100)
        assert abs(snippet.data[0, 0] - 0.0) < 1e-5
        assert abs(snippet.data[-1, 0] - 1.0) < 1e-5
        assert abs(snippet.data[50, 0] - np.sqrt(0.5)) < 0.02  # sin(π/4)
