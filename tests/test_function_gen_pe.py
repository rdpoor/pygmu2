"""
Tests for FunctionGenPE (naive, no anti-aliasing).

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
    FunctionGenPE,
    NullRenderer,
)


class TestFunctionGenPEBasics:
    def test_create_defaults(self):
        pe = FunctionGenPE()
        assert pe.frequency == 440.0
        assert pe.duty_cycle == 0.5
        assert pe.waveform == "rectangle"
        assert pe.channel_count() == 1

    def test_waveform_validation(self):
        with pytest.raises(ValueError):
            FunctionGenPE(waveform="nope")

    def test_is_pure_with_constants(self):
        pe = FunctionGenPE(frequency=100.0, duty_cycle=0.25, waveform="rectangle")
        assert pe.is_pure() is True
        assert pe.inputs() == []

    def test_extent_with_disjoint_pe_inputs_is_empty(self):
        freq = CropPE(ConstantPE(100.0), Extent(0, 10))
        duty = CropPE(ConstantPE(0.5), Extent(20, 30))
        pe = FunctionGenPE(frequency=freq, duty_cycle=duty)
        assert pe.extent().is_empty()


class TestFunctionGenPERender:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=10_000)

    def test_rectangle_exact_plateaus(self):
        pe = FunctionGenPE(frequency=100.0, duty_cycle=0.25, waveform="rectangle")
        self.renderer.set_source(pe)
        y = pe.render(0, 100).data[:, 0]

        # phase = n * 0.01, duty=0.25
        assert y[0] == pytest.approx(1.0)
        assert y[24] == pytest.approx(1.0)
        assert y[25] == pytest.approx(-1.0)
        assert y[50] == pytest.approx(-1.0)

    def test_sawtooth_duty_0_is_rising_saw(self):
        pe = FunctionGenPE(frequency=10.0, duty_cycle=0.0, waveform="sawtooth")
        self.renderer.set_source(pe)
        y = pe.render(0, 1000).data[:, 0]

        # For rising saw: y = 2*phase-1
        # phase at n=250 with dt=0.001 is 0.25 => y= -0.5
        assert y[250] == pytest.approx(-0.5, abs=1e-6)

    def test_sawtooth_duty_1_is_falling_saw(self):
        pe = FunctionGenPE(frequency=10.0, duty_cycle=1.0, waveform="sawtooth")
        self.renderer.set_source(pe)
        y = pe.render(0, 1000).data[:, 0]

        # Falling saw: y = 1-2*phase
        # phase 0.25 => y=0.5
        assert y[250] == pytest.approx(0.5, abs=1e-6)

    def test_sawtooth_duty_half_is_triangle(self):
        pe = FunctionGenPE(frequency=10.0, duty_cycle=0.5, waveform="sawtooth")
        self.renderer.set_source(pe)
        y = pe.render(0, 1000).data[:, 0]

        # Triangle: phase 0.25 => 0.0, phase 0.75 => 0.0
        assert y[250] == pytest.approx(0.0, abs=1e-6)
        assert y[750] == pytest.approx(0.0, abs=1e-6)

    def test_chunk_continuity_dynamic_frequency(self):
        n = 200
        freq_values = np.linspace(80.0, 220.0, n, dtype=np.float32)

        pe_full = FunctionGenPE(frequency=ArrayPE(freq_values), duty_cycle=0.3, waveform="rectangle")
        self.renderer.set_source(pe_full)
        y_full = pe_full.render(0, n).data[:, 0]

        pe_chunk = FunctionGenPE(frequency=ArrayPE(freq_values), duty_cycle=0.3, waveform="rectangle")
        self.renderer.set_source(pe_chunk)
        y1 = pe_chunk.render(0, 80).data[:, 0]
        y2 = pe_chunk.render(80, n - 80).data[:, 0]
        y_cat = np.concatenate([y1, y2])

        np.testing.assert_allclose(y_cat, y_full, atol=0.0, rtol=0.0)

