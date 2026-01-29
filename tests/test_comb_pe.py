"""
Tests for CombPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np
from pygmu2 import CombPE, ConstantPE, NullRenderer, RampPE


class TestCombPEBasics:
    def test_create_defaults(self):
        source = ConstantPE(1.0)
        comb = CombPE(source, frequency=440.0, feedback=0.5)

        assert comb.source is source
        assert comb.frequency == 440.0
        assert comb.feedback == 0.5

    def test_inputs_with_param_pes(self):
        source = ConstantPE(1.0)
        freq = RampPE(200.0, 800.0, duration=1000)
        fb = RampPE(0.0, 0.5, duration=1000)

        comb = CombPE(source, frequency=freq, feedback=fb)
        inputs = comb.inputs()

        assert source in inputs
        assert freq in inputs
        assert fb in inputs

    def test_is_not_pure(self):
        source = ConstantPE(1.0)
        comb = CombPE(source, frequency=440.0)
        assert comb.is_pure() is False


class TestCombPERender:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_render_output_shape(self):
        source = ConstantPE(0.25, channels=2)
        comb = CombPE(source, frequency=440.0, feedback=0.7)

        self.renderer.set_source(comb)
        with self.renderer:
            self.renderer.start()
            snippet = comb.render(0, 512)

        assert snippet.data.shape == (512, 2)
        assert np.isfinite(snippet.data).all()
