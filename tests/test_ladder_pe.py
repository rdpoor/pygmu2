"""
Tests for LadderPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np
from pygmu2 import ConstantPE, LadderPE, LadderMode, NullRenderer, PiecewisePE


class TestLadderPEBasics:
    def test_create_defaults(self):
        source = ConstantPE(1.0)
        ladder = LadderPE(source, frequency=1000.0, resonance=0.5)

        assert ladder.source is source
        assert ladder.frequency == 1000.0
        assert ladder.resonance == 0.5
        assert ladder.mode == LadderMode.LP24

    def test_inputs_with_param_pes(self):
        source = ConstantPE(1.0)
        freq = PiecewisePE([(0, 200.0), (1000, 2000.0)])
        res = PiecewisePE([(0, 0.1), (1000, 0.9)])

        ladder = LadderPE(source, frequency=freq, resonance=res, mode=LadderMode.HP12)

        inputs = ladder.inputs()
        assert source in inputs
        assert freq in inputs
        assert res in inputs

    def test_is_not_pure(self):
        source = ConstantPE(1.0)
        ladder = LadderPE(source, frequency=1000.0, resonance=0.5)
        assert ladder.is_pure() is False

    def test_extent_from_source(self):
        source = PiecewisePE([(0, 0.0), (500, 1.0)])
        ladder = LadderPE(source, frequency=1000.0, resonance=0.2)
        extent = ladder.extent()
        assert extent.start == 0
        assert extent.end == 500


class TestLadderPERender:
    def setup_method(self):
        self.renderer = NullRenderer(sample_rate=44100)

    def test_render_output_shape(self):
        source = ConstantPE(1.0, channels=2)
        ladder = LadderPE(source, frequency=800.0, resonance=0.4, mode=LadderMode.LP24)

        self.renderer.set_source(ladder)
        with self.renderer:
            self.renderer.start()
            snippet = ladder.render(0, 256)

        assert snippet.data.shape == (256, 2)
        assert np.isfinite(snippet.data).all()

    def test_dc_passes_lowpass(self):
        source = ConstantPE(1.0)
        ladder = LadderPE(source, frequency=1000.0, resonance=0.0, mode=LadderMode.LP24)

        self.renderer.set_source(ladder)
        with self.renderer:
            self.renderer.start()
            _ = ladder.render(0, 1000)  # settle
            snippet = ladder.render(1000, 200)

        assert np.mean(snippet.data) > 0.5
