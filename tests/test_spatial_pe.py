"""
Tests for SpatialPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np

from pygmu2 import (
    ArrayPE,
    NullRenderer,
    SpatialPE,
    SpatialAdapter,
    SpatialLinear,
    SpatialConstantPower,
    SpatialHRTF,
)


class TestSpatialAdapter:
    """Test SpatialAdapter channel conversion (M→N)."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_mono_to_stereo(self):
        """Mono (1) → Stereo (2): Duplicate mono channel to both L and R."""
        # Mono input: [1, 2, 3, 4]
        mono_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        adapter = SpatialPE(mono_source, method=SpatialAdapter(channels=2))
        self.renderer.set_source(adapter)
        
        snippet = adapter.render(0, 4)
        
        # Should duplicate mono to both channels
        expected = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
        assert snippet.channels == 2
    
    def test_stereo_to_mono(self):
        """Stereo (2) → Mono (1): Mix L and R channels (average)."""
        # Stereo input: L=[1, 2, 3], R=[10, 20, 30]
        stereo_data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
        stereo_source = ArrayPE(stereo_data)
        
        adapter = SpatialPE(stereo_source, method=SpatialAdapter(channels=1))
        self.renderer.set_source(adapter)
        
        snippet = adapter.render(0, 3)
        
        # Should average L and R: [(1+10)/2, (2+20)/2, (3+30)/2] = [5.5, 11.0, 16.5]
        expected = np.array([[5.5], [11.0], [16.5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 1
    
    def test_mono_to_quad(self):
        """Mono (1) → Quad (4): Duplicate to all channels."""
        mono_data = np.array([1.0, 2.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        adapter = SpatialPE(mono_source, method=SpatialAdapter(channels=4))
        self.renderer.set_source(adapter)
        
        snippet = adapter.render(0, 2)
        
        # Should duplicate mono to all 4 channels
        expected = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
        assert snippet.channels == 4
    
    def test_stereo_to_quad(self):
        """Stereo (2) → Quad (4): L→L, R→R, center/surround from mix."""
        # Stereo input: L=[1, 2], R=[10, 20]
        stereo_data = np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
        stereo_source = ArrayPE(stereo_data)
        
        adapter = SpatialPE(stereo_source, method=SpatialAdapter(channels=4))
        self.renderer.set_source(adapter)
        
        snippet = adapter.render(0, 2)
        
        # L→L, R→R, center/surround = average of L and R
        # Channel order: [L, R, Center, Surround] (or similar)
        # For now, assume: L→L, R→R, Center=(L+R)/2, Surround=(L+R)/2
        expected = np.array([
            [1.0, 10.0, 5.5, 5.5],  # L, R, (L+R)/2, (L+R)/2
            [2.0, 20.0, 11.0, 11.0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 4
    
    def test_quad_to_stereo(self):
        """Quad (4) → Stereo (2): L→L, R→R, ignore center/surround."""
        # Quad input: [L, R, C, S] = [[1, 10, 100, 1000], [2, 20, 200, 2000]]
        quad_data = np.array([[1.0, 10.0, 100.0, 1000.0], [2.0, 20.0, 200.0, 2000.0]], dtype=np.float32)
        quad_source = ArrayPE(quad_data)
        
        adapter = SpatialPE(quad_source, method=SpatialAdapter(channels=2))
        self.renderer.set_source(adapter)
        
        snippet = adapter.render(0, 2)
        
        # Should take L and R, ignore C and S
        expected = np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)
        assert snippet.channels == 2
    
    def test_same_channel_count(self):
        """Same channel count: Passthrough (no conversion)."""
        stereo_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        stereo_source = ArrayPE(stereo_data)
        
        adapter = SpatialPE(stereo_source, method=SpatialAdapter(channels=2))
        self.renderer.set_source(adapter)
        
        snippet = adapter.render(0, 2)
        
        # Should pass through unchanged
        np.testing.assert_array_equal(snippet.data, stereo_data)
        assert snippet.channels == 2
    
    def test_channel_count_property(self):
        """Verify channel_count() returns correct output channels."""
        mono_source = ArrayPE(np.array([1.0], dtype=np.float32).reshape(-1, 1))
        
        adapter_2ch = SpatialPE(mono_source, method=SpatialAdapter(channels=2))
        assert adapter_2ch.channel_count() == 2
        
        adapter_4ch = SpatialPE(mono_source, method=SpatialAdapter(channels=4))
        assert adapter_4ch.channel_count() == 4


class TestSpatialLinear:
    """Test SpatialLinear panning method."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_center_pan_azimuth_0(self):
        """Azimuth 0° (center): Equal L/R gain (0.5 each)."""
        # Mono input: [1, 2, 3]
        mono_data = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        panned = SpatialPE(mono_source, method=SpatialLinear(azimuth=0.0))
        self.renderer.set_source(panned)
        
        snippet = panned.render(0, 3)
        
        # Linear panning at 0°: pan = 0.5, L = 1 - 0.5 = 0.5, R = 0.5
        # Output: [1*0.5, 1*0.5], [2*0.5, 2*0.5], [3*0.5, 3*0.5]
        expected = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 2
    
    def test_right_pan_azimuth_90(self):
        """Azimuth +90° (right): All right channel (L=0, R=1)."""
        mono_data = np.array([1.0, 2.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        panned = SpatialPE(mono_source, method=SpatialLinear(azimuth=90.0))
        self.renderer.set_source(panned)
        
        snippet = panned.render(0, 2)
        
        # Linear panning at +90°: pan = 1.0, L = 0, R = 1
        expected = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 2
    
    def test_left_pan_azimuth_neg90(self):
        """Azimuth -90° (left): All left channel (L=1, R=0)."""
        mono_data = np.array([1.0, 2.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        panned = SpatialPE(mono_source, method=SpatialLinear(azimuth=-90.0))
        self.renderer.set_source(panned)
        
        snippet = panned.render(0, 2)
        
        # Linear panning at -90°: pan = 0.0, L = 1, R = 0
        expected = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 2
    
    def test_partial_right_pan_azimuth_45(self):
        """Azimuth +45°: Partial right pan (L=0.25, R=0.75)."""
        mono_data = np.array([1.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        panned = SpatialPE(mono_source, method=SpatialLinear(azimuth=45.0))
        self.renderer.set_source(panned)
        
        snippet = panned.render(0, 1)
        
        # Linear panning at +45°: pan = 0.75, L = 1 - 0.75 = 0.25, R = 0.75
        expected = np.array([[0.25, 0.75]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 2
    
    def test_stereo_input_mixed_to_mono_first(self):
        """Stereo input should be mixed to mono first, then panned."""
        # Stereo input: L=[1, 2], R=[10, 20]
        stereo_data = np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32)
        stereo_source = ArrayPE(stereo_data)
        
        panned = SpatialPE(stereo_source, method=SpatialLinear(azimuth=0.0))
        self.renderer.set_source(panned)
        
        snippet = panned.render(0, 2)
        
        # Should mix to mono first: [(1+10)/2, (2+20)/2] = [5.5, 11.0]
        # Then pan center: [5.5*0.5, 5.5*0.5], [11.0*0.5, 11.0*0.5]
        expected = np.array([[2.75, 2.75], [5.5, 5.5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 2
    
    def test_azimuth_clamping(self):
        """Azimuth values outside [-90, +90] should be clamped."""
        mono_data = np.array([1.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        # Test extreme values
        panned_180 = SpatialPE(mono_source, method=SpatialLinear(azimuth=180.0))
        panned_neg180 = SpatialPE(mono_source, method=SpatialLinear(azimuth=-180.0))
        
        self.renderer.set_source(panned_180)
        snippet_180 = panned_180.render(0, 1)
        # Should clamp to +90° (all right)
        expected_right = np.array([[0.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet_180.data, expected_right, decimal=5)
        
        self.renderer.set_source(panned_neg180)
        snippet_neg180 = panned_neg180.render(0, 1)
        # Should clamp to -90° (all left)
        expected_left = np.array([[1.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet_neg180.data, expected_left, decimal=5)


class TestSpatialConstantPower:
    """Test SpatialConstantPower panning method."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_center_pan_azimuth_0(self):
        """Azimuth 0° (center): Equal L/R using constant-power (sin/cos at 45°)."""
        mono_data = np.array([1.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        panned = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=0.0))
        self.renderer.set_source(panned)
        
        snippet = panned.render(0, 1)
        
        # Constant-power at 0°: pan angle = 45°, L = cos(45°) ≈ 0.707, R = sin(45°) ≈ 0.707
        expected = np.array([[np.cos(np.pi/4), np.sin(np.pi/4)]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 2
    
    def test_right_pan_azimuth_90(self):
        """Azimuth +90° (right): All right channel (L=0, R=1)."""
        mono_data = np.array([1.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        panned = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=90.0))
        self.renderer.set_source(panned)
        
        snippet = panned.render(0, 1)
        
        # Constant-power at +90°: pan angle = 90°, L = cos(90°) = 0, R = sin(90°) = 1
        expected = np.array([[0.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 2
    
    def test_left_pan_azimuth_neg90(self):
        """Azimuth -90° (left): All left channel (L=1, R=0)."""
        mono_data = np.array([1.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        panned = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=-90.0))
        self.renderer.set_source(panned)
        
        snippet = panned.render(0, 1)
        
        # Constant-power at -90°: pan angle = 0°, L = cos(0°) = 1, R = sin(0°) = 0
        expected = np.array([[1.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 2
    
    def test_constant_power_property(self):
        """Verify constant-power property: L² + R² should be constant."""
        mono_data = np.array([1.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        # Test multiple azimuth angles
        azimuths = [-90, -45, 0, 45, 90]
        powers = []
        
        for az in azimuths:
            panned = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=float(az)))
            self.renderer.set_source(panned)
            snippet = panned.render(0, 1)
            L, R = snippet.data[0, 0], snippet.data[0, 1]
            power = L * L + R * R
            powers.append(power)
        
        # All powers should be approximately equal (constant power)
        # (Small numerical differences are expected, but should be close)
        assert all(abs(p - powers[0]) < 1e-5 for p in powers), \
            f"Constant-power property violated: powers={powers}"
    
    def test_partial_right_pan_azimuth_45(self):
        """Azimuth +45°: Partial right pan using constant-power."""
        mono_data = np.array([1.0], dtype=np.float32).reshape(-1, 1)
        mono_source = ArrayPE(mono_data)
        
        panned = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=45.0))
        self.renderer.set_source(panned)
        
        snippet = panned.render(0, 1)
        
        # Constant-power at +45°: pan angle = 67.5°, L = cos(67.5°), R = sin(67.5°)
        pan_angle_rad = np.deg2rad(67.5)
        expected = np.array([[np.cos(pan_angle_rad), np.sin(pan_angle_rad)]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 2
    
    def test_stereo_input_mixed_to_mono_first(self):
        """Stereo input should be mixed to mono first, then panned."""
        stereo_data = np.array([[1.0, 10.0]], dtype=np.float32)
        stereo_source = ArrayPE(stereo_data)
        
        panned = SpatialPE(stereo_source, method=SpatialConstantPower(azimuth=0.0))
        self.renderer.set_source(panned)
        
        snippet = panned.render(0, 1)
        
        # Mix to mono: (1+10)/2 = 5.5
        # Then pan center: 5.5 * [cos(45°), sin(45°)]
        expected = np.array([[5.5 * np.cos(np.pi/4), 5.5 * np.sin(np.pi/4)]], dtype=np.float32)
        np.testing.assert_array_almost_equal(snippet.data, expected, decimal=5)
        assert snippet.channels == 2


class TestSpatialPEBasics:
    """Test basic SpatialPE properties and validation."""
    
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)
    
    def test_method_required(self):
        """Method parameter is required."""
        mono_source = ArrayPE(np.array([1.0], dtype=np.float32).reshape(-1, 1))
        
        with pytest.raises(ValueError, match="method is required"):
            SpatialPE(mono_source, method=None)
    
    def test_inputs_includes_source_and_dynamic_params(self):
        """inputs() should include source and any dynamic method parameters."""
        mono_source = ArrayPE(np.array([1.0], dtype=np.float32).reshape(-1, 1))
        
        # Static azimuth
        static_pan = SpatialPE(mono_source, method=SpatialLinear(azimuth=45.0))
        assert static_pan.inputs() == [mono_source]
        
        # Dynamic azimuth (PE)
        from pygmu2 import ConstantPE
        dynamic_azimuth = ConstantPE(45.0)
        dynamic_pan = SpatialPE(mono_source, method=SpatialLinear(azimuth=dynamic_azimuth))
        inputs = dynamic_pan.inputs()
        assert mono_source in inputs
        assert dynamic_azimuth in inputs
        assert len(inputs) == 2
    
    def test_extent_passthrough(self):
        """Extent should match source extent."""
        mono_source = ArrayPE(np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(-1, 1))
        
        adapter = SpatialPE(mono_source, method=SpatialAdapter(channels=2))
        assert adapter.extent() == mono_source.extent()
        
        panned = SpatialPE(mono_source, method=SpatialLinear(azimuth=0.0))
        assert panned.extent() == mono_source.extent()
    
    def test_is_pure(self):
        """SpatialPE should be pure (stateless)."""
        mono_source = ArrayPE(np.array([1.0], dtype=np.float32).reshape(-1, 1))
        
        adapter = SpatialPE(mono_source, method=SpatialAdapter(channels=2))
        assert adapter.is_pure() is True
        
        panned = SpatialPE(mono_source, method=SpatialLinear(azimuth=0.0))
        assert panned.is_pure() is True
    
    def test_repr(self):
        """__repr__ should show source and method."""
        mono_source = ArrayPE(np.array([1.0], dtype=np.float32).reshape(-1, 1))
        
        adapter = SpatialPE(mono_source, method=SpatialAdapter(channels=2))
        repr_str = repr(adapter)
        assert "SpatialPE" in repr_str
        assert "ArrayPE" in repr_str
        assert "SpatialAdapter" in repr_str
        
        panned = SpatialPE(mono_source, method=SpatialLinear(azimuth=45.0))
        repr_str = repr(panned)
        assert "SpatialLinear" in repr_str
        assert "45.0" in repr_str


class TestSpatialHRTFHrtfFilenameFor:
    """Test SpatialHRTF.hrtf_filename_for (KEMAR nearest-position lookup)."""

    def test_front_returns_zero_azimuth_file(self):
        """Azimuth 0 (front) should return a file with 0° azimuth."""
        name = SpatialHRTF.hrtf_filename_for(0.0, 0.0)
        assert name == "H0e000a.wav"

    def test_45_right_returns_45_azimuth_file(self):
        """Azimuth 45° (right) should return H0e045a.wav (0° elevation, 45° azimuth)."""
        name = SpatialHRTF.hrtf_filename_for(45.0, 0.0)
        assert name == "H0e045a.wav"

    def test_45_left_returns_same_file_as_45_right(self):
        """Azimuth -45° (left) uses same file as 45° (caller swaps L/R at render)."""
        name_left = SpatialHRTF.hrtf_filename_for(-45.0, 0.0)
        name_right = SpatialHRTF.hrtf_filename_for(45.0, 0.0)
        assert name_left == name_right == "H0e045a.wav"

    def test_90_right_returns_90_azimuth_file(self):
        """Azimuth 90° should return a file with 90° azimuth at 0° elevation."""
        name = SpatialHRTF.hrtf_filename_for(90.0, 0.0)
        assert name == "H0e090a.wav"

    def test_elevation_affects_choice(self):
        """Elevation 30° should return a file with elevation near 30°."""
        name = SpatialHRTF.hrtf_filename_for(0.0, 30.0)
        assert "30" in name or "H30" in name
        assert name.endswith(".wav")

    def test_returns_string_from_kemar_entries(self):
        """Returned name must be one of the embedded KEMAR filenames."""
        name = SpatialHRTF.hrtf_filename_for(12.0, -10.0)
        assert isinstance(name, str)
        assert name in {e[2] for e in SpatialHRTF.KEMAR_HRTF_ENTRIES}


class TestSpatialHRTFStaticOnly:
    """Test that SpatialHRTF accepts only static azimuth/elevation."""

    def test_rejects_dynamic_azimuth(self):
        """SpatialHRTF must reject ProcessingElement for azimuth."""
        from pygmu2 import ConstantPE
        with pytest.raises(ValueError, match="azimuth and elevation must be static"):
            SpatialHRTF(azimuth=ConstantPE(45.0), elevation=0.0)

    def test_rejects_dynamic_elevation(self):
        """SpatialHRTF must reject ProcessingElement for elevation."""
        from pygmu2 import ConstantPE
        with pytest.raises(ValueError, match="azimuth and elevation must be static"):
            SpatialHRTF(azimuth=0.0, elevation=ConstantPE(15.0))

    def test_accepts_static_float(self):
        """SpatialHRTF accepts float azimuth and elevation."""
        method = SpatialHRTF(azimuth=45.0, elevation=15.0)
        assert method.azimuth == 45.0
        assert method.elevation == 15.0

    def test_accepts_static_int(self):
        """SpatialHRTF accepts int azimuth and elevation."""
        method = SpatialHRTF(azimuth=45, elevation=0)
        assert method.azimuth == 45.0
        assert method.elevation == 0.0
