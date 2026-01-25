"""
Tests for CropPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    CropPE,
    ConstantPE,
    RampPE,
    SinePE,
    IdentityPE,
    DelayPE,
    MixPE,
    NullRenderer,
    Extent,
)


class TestCropPEBasics:
    """Test basic CropPE creation and properties."""

    def test_create_crop_pe(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(100, 200))
        assert crop.source is source
        assert crop.start == 100
        assert crop.end == 200
        assert crop.crop_extent.start == 100
        assert crop.crop_extent.end == 200

    def test_create_with_zero_start(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(0, 1000))
        assert crop.start == 0

    def test_create_with_none_start(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(None, 1000))
        assert crop.start is None
        assert crop.end == 1000

    def test_create_with_none_end(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(100, None))
        assert crop.start == 100
        assert crop.end is None

    def test_create_with_both_none(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(None, None))
        assert crop.start is None
        assert crop.end is None

    def test_inputs(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(0, 100))
        assert crop.inputs() == [source]

    def test_is_pure(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(0, 100))
        assert crop.is_pure() is True

    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        crop = CropPE(source, Extent(0, 100))
        assert crop.channel_count() == 2

    def test_repr(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(100, 200))
        repr_str = repr(crop)
        assert "CropPE" in repr_str
        assert "ConstantPE" in repr_str
        assert "100" in repr_str
        assert "200" in repr_str

    def test_repr_with_none(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(None, 200))
        repr_str = repr(crop)
        assert "None" in repr_str
        assert "200" in repr_str


class TestCropPEExtent:
    """Test CropPE extent calculation."""

    def test_extent_infinite_source(self):
        source = ConstantPE(1.0)  # Infinite extent
        crop = CropPE(source, Extent(100, 200))

        extent = crop.extent()
        assert extent.start == 100
        assert extent.end == 200

    def test_extent_finite_source_fully_contains_crop(self):
        source = RampPE(0.0, 1.0, duration=1000)  # Extent (0, 1000)
        crop = CropPE(source, Extent(100, 200))

        extent = crop.extent()
        assert extent.start == 100
        assert extent.end == 200

    def test_extent_finite_source_crop_extends_before(self):
        source = RampPE(0.0, 1.0, duration=1000)  # Extent (0, 1000)
        crop = CropPE(source, Extent(-100, 200))

        # Intersection of (-100, 200) and (0, 1000) is (0, 200)
        extent = crop.extent()
        assert extent.start == 0
        assert extent.end == 200

    def test_extent_finite_source_crop_extends_after(self):
        source = RampPE(0.0, 1.0, duration=1000)  # Extent (0, 1000)
        crop = CropPE(source, Extent(800, 1200))

        # Intersection of (800, 1200) and (0, 1000) is (800, 1000)
        extent = crop.extent()
        assert extent.start == 800
        assert extent.end == 1000

    def test_extent_no_overlap(self):
        source = RampPE(0.0, 1.0, duration=100)  # Extent (0, 100)
        crop = CropPE(source, Extent(200, 300))

        # No intersection -> empty extent at boundary
        extent = crop.extent()
        assert extent == Extent(200, 200)
        assert extent.is_empty() is True

    def test_extent_none_start_finite_source(self):
        source = RampPE(0.0, 1.0, duration=1000)  # Extent (0, 1000)
        crop = CropPE(source, Extent(None, 500))

        # Intersection of (None, 500) and (0, 1000) is (0, 500)
        extent = crop.extent()
        assert extent.start == 0
        assert extent.end == 500

    def test_extent_none_end_finite_source(self):
        source = RampPE(0.0, 1.0, duration=1000)  # Extent (0, 1000)
        crop = CropPE(source, Extent(500, None))

        # Intersection of (500, None) and (0, 1000) is (500, 1000)
        extent = crop.extent()
        assert extent.start == 500
        assert extent.end == 1000

    def test_extent_both_none_finite_source(self):
        source = RampPE(0.0, 1.0, duration=1000)  # Extent (0, 1000)
        crop = CropPE(source, Extent(None, None))

        # No constraint - returns source extent
        extent = crop.extent()
        assert extent.start == 0
        assert extent.end == 1000

    def test_extent_none_start_infinite_source(self):
        source = ConstantPE(1.0)  # Infinite extent
        crop = CropPE(source, Extent(None, 500))

        # Crop end only
        extent = crop.extent()
        assert extent.start is None
        assert extent.end == 500

    def test_extent_none_end_infinite_source(self):
        source = ConstantPE(1.0)  # Infinite extent
        crop = CropPE(source, Extent(500, None))

        # Crop start only
        extent = crop.extent()
        assert extent.start == 500
        assert extent.end is None


class TestCropPERender:
    """Test CropPE rendering."""

    def test_render_fully_inside_crop(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(100, 200))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        snippet = crop.render(120, 50)
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 1.0, dtype=np.float32)
        )

        renderer.stop()

    def test_render_fully_before_crop(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(100, 200))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        snippet = crop.render(0, 50)
        np.testing.assert_array_equal(
            snippet.data, np.zeros((50, 1), dtype=np.float32)
        )

        renderer.stop()

    def test_render_fully_after_crop(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(100, 200))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        snippet = crop.render(250, 50)
        np.testing.assert_array_equal(
            snippet.data, np.zeros((50, 1), dtype=np.float32)
        )

        renderer.stop()

    def test_render_spanning_crop_start(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(100, 200))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        snippet = crop.render(75, 50)

        expected = np.zeros((50, 1), dtype=np.float32)
        expected[25:, :] = 1.0
        np.testing.assert_array_equal(snippet.data, expected)

        renderer.stop()

    def test_render_spanning_crop_end(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(100, 200))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        snippet = crop.render(175, 50)

        expected = np.zeros((50, 1), dtype=np.float32)
        expected[:25, :] = 1.0
        np.testing.assert_array_equal(snippet.data, expected)

        renderer.stop()

    def test_render_spanning_entire_crop(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(100, 200))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        snippet = crop.render(50, 200)

        expected = np.zeros((200, 1), dtype=np.float32)
        expected[50:150, :] = 1.0
        np.testing.assert_array_equal(snippet.data, expected)

        renderer.stop()

    def test_render_with_identity_source(self):
        source = IdentityPE()
        crop = CropPE(source, Extent(100, 110))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        snippet = crop.render(100, 10)
        expected = np.arange(100, 110, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(snippet.data, expected)

        renderer.stop()

    def test_render_stereo(self):
        source = ConstantPE(0.5, channels=2)
        crop = CropPE(source, Extent(0, 100))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        snippet = crop.render(0, 50)
        assert snippet.channels == 2
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 2), 0.5, dtype=np.float32)
        )

        renderer.stop()


class TestCropPENoneBounds:
    """Test CropPE with None bounds (open-ended crops)."""

    def test_render_none_start(self):
        source = IdentityPE()
        crop = CropPE(source, Extent(None, 100))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        # Before end - should pass through
        snippet = crop.render(50, 25)
        expected = np.arange(50, 75, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(snippet.data, expected)

        # Spanning end
        snippet = crop.render(90, 20)
        expected = np.zeros((20, 1), dtype=np.float32)
        expected[:10, :] = np.arange(90, 100, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(snippet.data, expected)

        # Negative indices - still passes through (no lower bound)
        snippet = crop.render(-50, 25)
        expected = np.arange(-50, -25, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(snippet.data, expected)

        renderer.stop()

    def test_render_none_end(self):
        source = IdentityPE()
        crop = CropPE(source, Extent(100, None))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        # After start - should pass through
        snippet = crop.render(150, 25)
        expected = np.arange(150, 175, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(snippet.data, expected)

        # Before start - should be zeros
        snippet = crop.render(50, 25)
        expected = np.zeros((25, 1), dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)

        # Spanning start
        snippet = crop.render(90, 20)
        expected = np.zeros((20, 1), dtype=np.float32)
        expected[10:, :] = np.arange(100, 110, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(snippet.data, expected)

        renderer.stop()

    def test_render_both_none(self):
        source = IdentityPE()
        crop = CropPE(source, Extent(None, None))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)
        renderer.start()

        # Any range should pass through
        snippet = crop.render(-100, 200)
        expected = np.arange(-100, 100, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(snippet.data, expected)

        renderer.stop()


class TestCropPEWithOtherPEs:
    """Test CropPE combined with other PEs."""

    def test_crop_then_delay(self):
        source = IdentityPE()
        cropped = CropPE(source, Extent(0, 10))
        delayed = DelayPE(cropped, delay=100)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(delayed)
        renderer.start()

        assert delayed.extent().start == 100
        assert delayed.extent().end == 110

        snippet = delayed.render(100, 10)
        expected = np.arange(0, 10, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(snippet.data, expected)

        renderer.stop()

    def test_crop_infinite_sine(self):
        sine = SinePE(frequency=440.0)
        burst = CropPE(sine, Extent(0, 1000))

        assert burst.extent().start == 0
        assert burst.extent().end == 1000

    def test_crop_ramp(self):
        ramp = RampPE(0.0, 100.0, duration=100)
        cropped = CropPE(ramp, Extent(25, 75))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(cropped)
        renderer.start()

        ramp_snippet = ramp.render(25, 50)
        snippet = cropped.render(25, 50)
        np.testing.assert_array_almost_equal(snippet.data, ramp_snippet.data, decimal=5)

        renderer.stop()

    def test_mix_cropped_sources(self):
        source1 = ConstantPE(1.0)
        source2 = ConstantPE(2.0)

        crop1 = CropPE(source1, Extent(0, 100))
        crop2 = CropPE(source2, Extent(50, 150))

        mixed = MixPE(crop1, crop2)

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(mixed)
        renderer.start()

        snippet = mixed.render(0, 50)
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 1.0, dtype=np.float32)
        )

        snippet = mixed.render(50, 50)
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 3.0, dtype=np.float32)
        )

        snippet = mixed.render(100, 50)
        np.testing.assert_array_equal(
            snippet.data, np.full((50, 1), 2.0, dtype=np.float32)
        )

        renderer.stop()

    def test_crop_to_another_pe_extent(self):
        reference = RampPE(0.0, 1.0, duration=500)
        source = ConstantPE(1.0)

        cropped = CropPE(source, reference.extent())

        assert cropped.extent().start == 0
        assert cropped.extent().end == 500


class TestCropPEIntegration:
    """Integration tests for CropPE."""

    def test_full_render_cycle(self):
        source = ConstantPE(1.0)
        crop = CropPE(source, Extent(0, 100))

        renderer = NullRenderer(sample_rate=44100)
        renderer.set_source(crop)

        with renderer:
            renderer.start()
            snippet = crop.render(0, 100)
            assert np.sum(snippet.data) == 100.0

    def test_crop_chain(self):
        source = IdentityPE()
        crop1 = CropPE(source, Extent(0, 1000))
        crop2 = CropPE(crop1, Extent(100, 900))
        crop3 = CropPE(crop2, Extent(200, 800))

        assert crop3.extent().start == 200
        assert crop3.extent().end == 800

    def test_trim_start_of_finite_source(self):
        source = RampPE(0.0, 1.0, duration=1000)
        trimmed = CropPE(source, Extent(100, None))

        assert trimmed.extent().start == 100
        assert trimmed.extent().end == 1000

    def test_trim_end_of_finite_source(self):
        source = RampPE(0.0, 1.0, duration=1000)
        trimmed = CropPE(source, Extent(None, 800))

        assert trimmed.extent().start == 0
        assert trimmed.extent().end == 800
