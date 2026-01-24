"""
Tests for ReversePitchEchoPE.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

import numpy as np
from pygmu2 import ConstantPE, NullRenderer, RampPE, ReversePitchEchoPE


class TestReversePitchEchoPEBasics:
    def test_create_defaults(self):
        source = ConstantPE(1.0)
        rpe = ReversePitchEchoPE(source)

        assert rpe.source is source
        assert rpe.is_pure() is False

    def test_create_with_all_parameters(self):
        source = ConstantPE(1.0)
        rpe = ReversePitchEchoPE(
            source,
            block_seconds=0.2,
            pitch_ratio=0.5,
            feedback=0.7,
            alternate_direction=1.0,
            smoothing_samples=1000,
        )
        assert rpe.source is source

    def test_inputs_with_scalar_params(self):
        source = ConstantPE(1.0)
        rpe = ReversePitchEchoPE(source, block_seconds=0.1, pitch_ratio=1.0)
        inputs = rpe.inputs()
        assert source in inputs
        assert len(inputs) == 1

    def test_inputs_with_pe_params(self):
        source = ConstantPE(1.0)
        block_pe = RampPE(0.05, 0.2, duration=1000)
        pitch_pe = RampPE(0.5, 2.0, duration=1000)
        fb_pe = RampPE(0.0, 0.8, duration=1000)
        alt_pe = ConstantPE(1.0)

        rpe = ReversePitchEchoPE(
            source,
            block_seconds=block_pe,
            pitch_ratio=pitch_pe,
            feedback=fb_pe,
            alternate_direction=alt_pe,
        )
        inputs = rpe.inputs()

        assert source in inputs
        assert block_pe in inputs
        assert pitch_pe in inputs
        assert fb_pe in inputs
        assert alt_pe in inputs

    def test_channel_count_passthrough(self):
        source = ConstantPE(1.0, channels=2)
        rpe = ReversePitchEchoPE(source)
        assert rpe.channel_count() == 2


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

    def test_render_zero_duration(self):
        """Base class render() handles zero duration."""
        source = ConstantPE(0.5)
        rpe = ReversePitchEchoPE(source)

        self.renderer.set_source(rpe)
        with self.renderer:
            self.renderer.start()
            snippet = rpe.render(0, 0)

        assert snippet.data.shape[0] == 0

    def test_pitch_shift_up(self):
        """Pitch ratio > 1 should still produce output."""
        source = ConstantPE(0.5)
        rpe = ReversePitchEchoPE(source, block_seconds=0.01, pitch_ratio=2.0, feedback=0.0)

        self.renderer.set_source(rpe)
        with self.renderer:
            self.renderer.start()
            _ = rpe.render(0, 1024)
            snippet = rpe.render(1024, 512)

        assert np.isfinite(snippet.data).all()

    def test_pitch_shift_down(self):
        """Pitch ratio < 1 should still produce output."""
        source = ConstantPE(0.5)
        rpe = ReversePitchEchoPE(source, block_seconds=0.01, pitch_ratio=0.5, feedback=0.0)

        self.renderer.set_source(rpe)
        with self.renderer:
            self.renderer.start()
            _ = rpe.render(0, 1024)
            snippet = rpe.render(1024, 512)

        assert np.isfinite(snippet.data).all()

    def test_feedback_accumulates(self):
        """With feedback, output should build over multiple blocks."""
        source = ConstantPE(0.5)
        rpe = ReversePitchEchoPE(source, block_seconds=0.01, pitch_ratio=1.0, feedback=0.8)

        self.renderer.set_source(rpe)
        with self.renderer:
            self.renderer.start()
            # Render several blocks to let feedback accumulate
            for i in range(10):
                _ = rpe.render(i * 512, 512)
            snippet = rpe.render(10 * 512, 512)

        # With high feedback, signal should be present
        assert np.max(np.abs(snippet.data)) > 0.0

    def test_on_stop_clears_state(self):
        """on_stop should clear internal buffers."""
        source = ConstantPE(0.5)
        rpe = ReversePitchEchoPE(source)

        self.renderer.set_source(rpe)
        with self.renderer:
            self.renderer.start()
            _ = rpe.render(0, 512)

        # After context exit, on_stop is called
        assert rpe._buffer_a is None
        assert rpe._buffer_b is None
