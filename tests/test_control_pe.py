"""
Tests for ControlPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import threading

import numpy as np
import pytest

from pygmu2 import ControlPE, NullRenderer


class TestControlPEBasics:
    """Test basic ControlPE creation and properties."""

    def test_create_default(self):
        pe = ControlPE()
        assert pe.value == 0.0
        assert pe.channel_count() == 1

    def test_create_with_initial_value(self):
        pe = ControlPE(initial_value=1.5)
        assert pe.value == 1.5

    def test_create_with_channels(self):
        pe = ControlPE(initial_value=0.5, channels=2)
        assert pe.value == 0.5
        assert pe.channel_count() == 2

    def test_infinite_extent(self):
        pe = ControlPE()
        extent = pe.extent()
        assert extent.start is None
        assert extent.end is None

    def test_is_impure(self):
        pe = ControlPE()
        assert pe.is_pure() is False

    def test_no_inputs(self):
        pe = ControlPE()
        assert pe.inputs() == []

    def test_repr(self):
        pe = ControlPE(initial_value=0.5, channels=2)
        r = repr(pe)
        assert "ControlPE" in r
        assert "0.5" in r
        assert "2" in r


class TestControlPERender:
    """Test ControlPE rendering."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_render_initial_value(self):
        pe = ControlPE(initial_value=0.75)
        self.renderer.set_source(pe)
        self.renderer.start()

        snippet = pe.render(0, 100)
        assert snippet.start == 0
        assert snippet.duration == 100
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((100, 1), 0.75, dtype=np.float32),
        )

    def test_render_after_set_value(self):
        pe = ControlPE(initial_value=0.0)
        self.renderer.set_source(pe)
        self.renderer.start()

        pe.set_value(2.5)
        snippet = pe.render(0, 50)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((50, 1), 2.5, dtype=np.float32),
        )

    def test_set_value_multiple_times_keeps_latest(self):
        pe = ControlPE(initial_value=0.0)
        self.renderer.set_source(pe)
        self.renderer.start()

        pe.set_value(1.0)
        pe.set_value(2.0)
        pe.set_value(3.0)
        snippet = pe.render(0, 10)
        # Should use the latest value (3.0)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((10, 1), 3.0, dtype=np.float32),
        )
        assert pe.value == 3.0

    def test_value_persists_across_renders(self):
        pe = ControlPE(initial_value=0.0)
        self.renderer.set_source(pe)
        self.renderer.start()

        pe.set_value(5.0)
        pe.render(0, 10)
        # No new set_value — should still output 5.0
        snippet = pe.render(10, 10)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((10, 1), 5.0, dtype=np.float32),
        )

    def test_render_stereo(self):
        pe = ControlPE(initial_value=0.5, channels=2)
        self.renderer.set_source(pe)
        self.renderer.start()

        snippet = pe.render(0, 50)
        assert snippet.channels == 2
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((50, 2), 0.5, dtype=np.float32),
        )

    def test_render_negative_value(self):
        pe = ControlPE(initial_value=-1.0)
        self.renderer.set_source(pe)
        self.renderer.start()

        snippet = pe.render(0, 10)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((10, 1), -1.0, dtype=np.float32),
        )


class TestControlPEThreadSafety:
    """Test that set_value works from another thread."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_set_value_from_thread(self):
        pe = ControlPE(initial_value=0.0)
        self.renderer.set_source(pe)
        self.renderer.start()

        barrier = threading.Event()

        def writer():
            pe.set_value(42.0)
            barrier.set()

        t = threading.Thread(target=writer)
        t.start()
        barrier.wait(timeout=2.0)
        t.join(timeout=2.0)

        snippet = pe.render(0, 10)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((10, 1), 42.0, dtype=np.float32),
        )
