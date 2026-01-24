"""
Tests for ReversePitchEchoPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
from pygmu2 import ConstantPE, NullRenderer, ReversePitchEchoPE


class TestReversePitchEchoPEBasics:
    def test_create_defaults(self):
        source = ConstantPE(1.0)
        rpe = ReversePitchEchoPE(source)

        assert rpe.source is source
        assert rpe.is_pure() is False


class TestReversePitchEchoPERender:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_render_output_shape(self):
        source = ConstantPE(0.5, channels=2)
        rpe = ReversePitchEchoPE(source, block_seconds=0.01, pitch_ratio=1.0, feedback=0.0)

        self.renderer.set_source(rpe)
        with self.renderer:
            self.renderer.start()
            snippet = rpe.render(0, 512)

        assert snippet.data.shape == (512, 2)
        assert np.isfinite(snippet.data).all()

    def test_produces_wet_signal_after_block(self):
        source = ConstantPE(0.5)
        rpe = ReversePitchEchoPE(source, block_seconds=0.01, pitch_ratio=1.0, feedback=0.0)

        self.renderer.set_source(rpe)
        with self.renderer:
            self.renderer.start()
            _ = rpe.render(0, 512)  # first block (may be silent)
            snippet = rpe.render(512, 512)

        assert np.max(np.abs(snippet.data)) > 0.0
