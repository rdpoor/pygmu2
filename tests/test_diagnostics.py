"""
Tests for pygmu2.diagnostics — the sanctioned profiling mechanism
(DESIGN_PHILOSOPHY.md PD-2: optimization waits for a profile).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

import pygmu2 as pg
from pygmu2 import diagnostics


def _graph():
    mix = pg.MixPE(
        pg.SinePE(frequency=440.0),
        pg.GainPE(pg.SinePE(frequency=220.0), 0.5),
    )
    renderer = pg.NullRenderer(sample_rate=44100)
    renderer.set_source(mix)
    renderer.start()
    return renderer


class TestProfileContextManager:
    def test_collects_per_class_stats(self):
        renderer = _graph()
        with diagnostics.profile() as report:
            for i in range(4):
                renderer.render(i * 1024, 1024)
        assert set(report.by_class) == {"MixPE", "SinePE", "GainPE"}
        assert report.by_class["MixPE"].count == 4
        assert report.by_class["SinePE"].count == 8  # two sines per block
        assert report.by_class["MixPE"].samples == 4 * 1024
        assert report.by_class["MixPE"].total_ns > 0

    def test_pull_counts(self):
        renderer = _graph()
        with diagnostics.profile() as report:
            renderer.render(0, 512)
        assert report.pull_counts == {"MixPE": 1, "SinePE": 2, "GainPE": 1}

    def test_summary_mentions_every_class(self):
        renderer = _graph()
        with diagnostics.profile() as report:
            renderer.render(0, 512)
        text = report.summary(sample_rate=44100)
        for cls in ("MixPE", "SinePE", "GainPE"):
            assert cls in text
        assert "x realtime" in text

    def test_disabled_outside_block(self):
        renderer = _graph()
        with diagnostics.profile():
            renderer.render(0, 256)
        assert not diagnostics.is_enabled()
        # renders outside the block record nothing
        diagnostics.reset_block()
        renderer.render(256, 256)
        with diagnostics.profile() as report:
            pass
        assert report.by_class == {}

    def test_realtime_ratio(self):
        st = diagnostics.ClassStats(total_ns=1_000_000_000, count=1, samples=44100)
        assert abs(st.realtime_ratio(44100) - 1.0) < 1e-9
        assert st.samples_per_second == 44100
