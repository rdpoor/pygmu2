"""
Tests for SVFilterPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import pytest
import numpy as np
from pygmu2 import (
    SVFilterPE,
    BiquadMode,
    ConstantPE,
    PiecewisePE,
    SinePE,
    DiracPE,
    NullRenderer,
)


class TestSVFilterPEBasics:
    """Test basic SVFilterPE creation and properties."""

    def test_create_lowpass(self):
        source = ConstantPE(1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS
        )
        assert svf.source is source
        assert svf.frequency == 1000.0
        assert svf.q == 0.707
        assert svf.mode == BiquadMode.LOWPASS

    def test_create_all_supported_modes(self):
        source = ConstantPE(1.0)
        for mode in BiquadMode:
            if mode == BiquadMode.ALLPASS:
                continue
            svf = SVFilterPE(
                source, frequency=1000.0, q=1.0, mode=mode
            )
            assert svf.mode == mode

    def test_rejects_allpass(self):
        source = ConstantPE(1.0)
        with pytest.raises(ValueError, match="ALLPASS"):
            SVFilterPE(
                source, frequency=1000.0, q=1.0, mode=BiquadMode.ALLPASS
            )

    def test_create_with_pe_frequency(self):
        source = ConstantPE(1.0)
        freq_pe = PiecewisePE([(0, 100.0), (44100, 5000.0)])
        svf = SVFilterPE(
            source, frequency=freq_pe, q=1.0, mode=BiquadMode.LOWPASS
        )
        assert svf.frequency is freq_pe

    def test_create_with_pe_q(self):
        source = ConstantPE(1.0)
        q_pe = PiecewisePE([(0, 0.5), (44100, 10.0)])
        svf = SVFilterPE(
            source, frequency=1000.0, q=q_pe, mode=BiquadMode.LOWPASS
        )
        assert svf.q is q_pe

    def test_create_with_gain_db(self):
        source = ConstantPE(1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=1.0,
            mode=BiquadMode.PEAKING, gain_db=6.0
        )
        assert svf.gain_db == 6.0

    def test_inputs_constant_params(self):
        source = ConstantPE(1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=1.0, mode=BiquadMode.LOWPASS
        )
        assert svf.inputs() == [source]

    def test_inputs_with_pe_frequency(self):
        source = ConstantPE(1.0)
        freq_pe = PiecewisePE([(0, 100.0), (44100, 5000.0)])
        svf = SVFilterPE(
            source, frequency=freq_pe, q=1.0, mode=BiquadMode.LOWPASS
        )
        inputs = svf.inputs()
        assert len(inputs) == 2
        assert source in inputs
        assert freq_pe in inputs

    def test_is_not_pure(self):
        source = ConstantPE(1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=1.0, mode=BiquadMode.LOWPASS
        )
        assert svf.is_pure() is False

    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        svf = SVFilterPE(
            source, frequency=1000.0, q=1.0, mode=BiquadMode.LOWPASS
        )
        assert svf.channel_count() == 2

    def test_repr(self):
        source = ConstantPE(1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS
        )
        repr_str = repr(svf)
        assert "SVFilterPE" in repr_str
        assert "1000.0" in repr_str
        assert "lowpass" in repr_str


class TestSVFilterPELowpass:
    """Test lowpass filter behavior."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_lowpass_passes_dc(self):
        source = ConstantPE(1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            _ = svf.render(0, 1000)
            snippet = svf.render(1000, 100)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((100, 1), 1.0, dtype=np.float32),
            decimal=2
        )

    def test_lowpass_attenuates_high_freq(self):
        source = SinePE(frequency=10000.0, amplitude=1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            _ = svf.render(0, 1000)
            snippet = svf.render(1000, 1000)
        output_rms = np.sqrt(np.mean(snippet.data ** 2))
        assert output_rms < 0.1


class TestSVFilterPEHighpass:
    """Test highpass filter behavior."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_highpass_blocks_dc(self):
        source = ConstantPE(1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=0.707, mode=BiquadMode.HIGHPASS
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            _ = svf.render(0, 1000)
            snippet = svf.render(1000, 100)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((100, 1), dtype=np.float32),
            decimal=2
        )


class TestSVFilterPEBandpass:
    """Test bandpass filter behavior."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_bandpass_blocks_dc(self):
        source = ConstantPE(1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=5.0, mode=BiquadMode.BANDPASS
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            _ = svf.render(0, 1000)
            snippet = svf.render(1000, 100)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.zeros((100, 1), dtype=np.float32),
            decimal=2
        )


class TestSVFilterPENotch:
    """Test notch filter behavior."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_notch_passes_dc(self):
        source = ConstantPE(1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=5.0, mode=BiquadMode.NOTCH
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            _ = svf.render(0, 1000)
            snippet = svf.render(1000, 100)
        np.testing.assert_array_almost_equal(
            snippet.data,
            np.full((100, 1), 1.0, dtype=np.float32),
            decimal=2
        )


class TestSVFilterPEStateManagement:
    """Test filter state management."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_state_persists_across_renders(self):
        source = DiracPE()
        svf = SVFilterPE(
            source, frequency=1000.0, q=5.0, mode=BiquadMode.LOWPASS
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            snippet1 = svf.render(0, 100)
            snippet2 = svf.render(100, 100)
        assert snippet2.data[0, 0] != 0.0

    def test_state_resets_on_start(self):
        source = DiracPE()
        svf = SVFilterPE(
            source, frequency=1000.0, q=5.0, mode=BiquadMode.LOWPASS
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            snippet1 = svf.render(0, 100)
            self.renderer.stop()
            self.renderer.start()
            snippet2 = svf.render(0, 100)
        np.testing.assert_array_almost_equal(
            snippet1.data, snippet2.data, decimal=5
        )


class TestSVFilterPETimeVarying:
    """Test time-varying filter parameters."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_frequency_sweep(self):
        source = ConstantPE(1.0)
        freq_sweep = PiecewisePE([(0, 100.0), (1000, 10000.0)])
        svf = SVFilterPE(
            source, frequency=freq_sweep, q=0.707, mode=BiquadMode.LOWPASS
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            snippet = svf.render(0, 1000)
        assert snippet.duration == 1000
        assert snippet.channels == 1

    def test_q_modulation(self):
        source = ConstantPE(1.0)
        q_mod = PiecewisePE([(0, 0.5), (1000, 10.0)])
        svf = SVFilterPE(
            source, frequency=1000.0, q=q_mod, mode=BiquadMode.LOWPASS
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            snippet = svf.render(0, 1000)
        assert snippet.duration == 1000


class TestSVFilterPEStereo:
    """Test stereo signal handling."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_stereo_filtering(self):
        source = ConstantPE(1.0, channels=2)
        svf = SVFilterPE(
            source, frequency=1000.0, q=0.707, mode=BiquadMode.LOWPASS
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            _ = svf.render(0, 500)
            snippet = svf.render(500, 100)
        assert snippet.channels == 2
        np.testing.assert_array_almost_equal(
            snippet.data[:, 0],
            snippet.data[:, 1],
            decimal=5
        )


class TestSVFilterPEPeaking:
    """Test peaking (bell) filter."""

    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_peaking_boost(self):
        """Peaking filter with positive gain should boost around center frequency."""
        source = SinePE(frequency=1000.0, amplitude=1.0)
        svf = SVFilterPE(
            source, frequency=1000.0, q=2.0,
            mode=BiquadMode.PEAKING, gain_db=6.0
        )
        self.renderer.set_source(svf)
        with self.renderer:
            self.renderer.start()
            _ = svf.render(0, 1000)
            snippet = svf.render(1000, 1000)
        output_rms = np.sqrt(np.mean(snippet.data ** 2))
        # Input RMS ≈ 0.707; 6dB boost should increase level (SVF bell may differ from biquad)
        assert output_rms > 0.7
        assert output_rms > 0.707  # At least some boost
