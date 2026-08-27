"""
Canonical construction recipes for every exported ProcessingElement.

This registry feeds the universal contract suite (test_contract.py): every
PE exported from pygmu2.__all__ must have a factory here, and the export-
completeness test enforces that a new PE cannot be exported without one —
which is how a new PE inherits contract coverage for free.

Factories return a FRESH instance on every call (fresh state, same
deterministic configuration — random PEs are always seeded). Composite
factories keep graphs minimal.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import atexit
import os
import tempfile
from pathlib import Path

import numpy as np

import pygmu2 as pg

REPO_ROOT = Path(__file__).resolve().parent.parent
SOUNDFONT = REPO_ROOT / "examples" / "audio" / "TimGM6mb.sf2"

# PEs whose lifecycle touches hardware; the contract suite only constructs
# these (rendering would need a device CI does not have).
HARDWARE = {"MidiInPE"}

_tmp_files: list[str] = []


def _cleanup_tmp() -> None:
    for path in _tmp_files:
        try:
            os.remove(path)
        except OSError:
            pass


atexit.register(_cleanup_tmp)


def _tmp_path(suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    _tmp_files.append(path)
    return path


_test_wav_path: str | None = None


def _test_wav() -> str:
    """A small deterministic stereo WAV, generated once per session."""
    global _test_wav_path
    if _test_wav_path is None:
        import soundfile as sf

        _test_wav_path = _tmp_path(".wav")
        t = np.arange(2048) / 44100.0
        data = np.stack(
            [0.5 * np.sin(2 * np.pi * 440.0 * t), 0.5 * np.sin(2 * np.pi * 660.0 * t)],
            axis=1,
        ).astype(np.float32)
        sf.write(_test_wav_path, data, 44100)
    return _test_wav_path


def _sine():
    return pg.SinePE(frequency=440.0, amplitude=0.5)


def _finite_sine(duration=1024):
    return pg.CropPE(_sine(), 0, duration)


def _fir():
    # short finite impulse (identity-ish filter)
    return pg.CropPE(pg.DiracPE(), 0, 64)


# name -> zero-arg callable returning a fresh PE
FACTORIES = {
    "AdsrGatedPE": lambda: pg.AdsrGatedPE(pg.PeriodicGatePE(frequency=10.0)),
    "AdsrTriggeredPE": lambda: pg.AdsrTriggeredPE(
        pg.GateToTriggerPE(pg.PeriodicGatePE(frequency=10.0))
    ),
    "AnalogOscPE": lambda: pg.AnalogOscPE(frequency=440.0),
    "ArrayPE": lambda: pg.ArrayPE(np.sin(2 * np.pi * np.arange(256) / 256)),
    "AudioReaderPE": lambda: pg.AudioReaderPE(_test_wav()),
    "BiquadPE": lambda: pg.BiquadPE(_sine(), frequency=1000.0, q=0.707),
    "BlitSawPE": lambda: pg.BlitSawPE(frequency=440.0),
    "CachePE": lambda: pg.CachePE(_sine()),
    "CombPE": lambda: pg.CombPE(_sine(), frequency=440.0, feedback=0.5),
    "CompressorPE": lambda: pg.CompressorPE(_sine(), threshold=-12.0),
    "ConstantPE": lambda: pg.ConstantPE(0.5),
    "ControlPE": lambda: pg.ControlPE(initial_value=0.25),
    "ConvolvePE": lambda: pg.ConvolvePE(_sine(), _fir()),
    "CropPE": lambda: _finite_sine(),
    "DecayingSinePE": lambda: pg.DecayingSinePE(frequency=440.0, tau=0.1),
    "DelayPE": lambda: pg.DelayPE(_sine(), delay=100),
    "DiracPE": lambda: pg.DiracPE(),
    "DynamicsPE": lambda: pg.DynamicsPE(
        source=_sine(), envelope=pg.EnvelopePE(_sine()), threshold=-12.0
    ),
    "EnvelopePE": lambda: pg.EnvelopePE(_sine()),
    "ExpanderPE": lambda: pg.ExpanderPE(_sine(), threshold=-40.0),
    "GainPE": lambda: pg.GainPE(_sine(), gain=0.5),
    "GateToTriggerPE": lambda: pg.GateToTriggerPE(pg.PeriodicGatePE(frequency=10.0)),
    "IdentityPE": lambda: pg.IdentityPE(),
    "IdiophonePE": lambda: pg.IdiophonePE(
        __import__("pygmu2.idiophone_pe", fromlist=["GLOCKENSPIEL"]).GLOCKENSPIEL,
        frequency=880.0,
    ),
    "KarplusStrongPE": lambda: pg.KarplusStrongPE(frequency=440.0, seed=42),
    "LadderPE": lambda: pg.LadderPE(_sine(), frequency=1000.0, resonance=0.2),
    "LimiterPE": lambda: pg.LimiterPE(_sine(), ceiling=-3.0),
    "LoopPE": lambda: pg.LoopPE(_finite_sine(256), count=4),
    "MagFreqPE": lambda: pg.MagFreqPE(_finite_sine(), mangler=lambda m, f: (m, f)),
    "MeltysynthPE": lambda: pg.MeltysynthPE(str(SOUNDFONT)),
    "MidiInPE": lambda: pg.MidiInPE(),
    "MixPE": lambda: pg.MixPE(_sine(), pg.SinePE(frequency=550.0, amplitude=0.3)),
    "MovingAveragePE": lambda: pg.MovingAveragePE(_sine(), window=16),
    "NoisePE": lambda: pg.NoisePE(seed=42),
    "NotesPE": lambda: pg.NotesPE(
        _finite_sine(512), [pg.Note(0.0, 1.0, 60), pg.Note(1.0, 1.0, 64)]
    ),
    "PeriodicGatePE": lambda: pg.PeriodicGatePE(frequency=10.0),
    "PiecewisePE": lambda: pg.PiecewisePE([(0, 0.0), (500, 1.0), (1000, 0.0)]),
    "RandomGatePE": lambda: pg.RandomGatePE(rate=100.0, seed=42),
    "RandomSelectPE": lambda: pg.RandomSelectPE(
        pg.GateToTriggerPE(pg.PeriodicGatePE(frequency=50.0)),
        inputs=[pg.ConstantPE(0.25), pg.ConstantPE(0.75)],
        seed=42,
    ),
    "RandomValuePE": lambda: pg.RandomValuePE(rate=100.0, seed=42),
    "ResamplePE": lambda: pg.ResamplePE(_sine(), rate=1.5),
    "ReverbPE": lambda: pg.ReverbPE(_sine(), ir=_fir(), mix=0.3),
    "ReversePitchEchoPE": lambda: pg.ReversePitchEchoPE(
        _sine(), block_seconds=0.01, pitch_ratio=1.0
    ),
    "RingModulatorPE": lambda: pg.RingModulatorPE(_sine(), pg.SinePE(frequency=30.0)),
    "SVFilterPE": lambda: pg.SVFilterPE(_sine(), frequency=1000.0, q=0.707),
    "HoldPE": lambda: pg.HoldPE(
        _sine(), pg.GateToTriggerPE(pg.PeriodicGatePE(frequency=50.0))
    ),
    "ScheduledGatePE": lambda: pg.ScheduledGatePE([(0, 100), (200, 100)]),
    "SequencePE": lambda: pg.SequencePE(
        (_finite_sine(256), 0), (pg.CropPE(pg.SinePE(frequency=550.0), 0, 256), 256)
    ),
    "SignalToGatePE": lambda: pg.SignalToGatePE(
        _sine(), low_threshold=0.1, high_threshold=0.3
    ),
    "SinePE": lambda: _sine(),
    "SlewLimiterPE": lambda: pg.SlewLimiterPE(_sine(), rate=100.0),
    "SlicePE": lambda: pg.SlicePE(_finite_sine(), 100, 200),
    "SpatialPE": lambda: pg.SpatialPE(_sine(), method=pg.SpatialLinear(azimuth=30.0)),
    "SuperSawPE": lambda: pg.SuperSawPE(frequency=440.0, voices=3),
    "TimeWarpPE": lambda: pg.TimeWarpPE(_sine(), rate=1.5),
    "TralfamPE": lambda: pg.TralfamPE(_finite_sine(), seed=42),
    "TransformPE": lambda: pg.TransformPE(_sine(), func=lambda x: x * 0.5),
    "TriggerRestartPE": lambda: pg.TriggerRestartPE(
        pg.GateToTriggerPE(pg.PeriodicGatePE(frequency=10.0)),
        pg.DecayingSinePE(frequency=440.0, tau=0.1),
    ),
    "WavReaderPE": lambda: pg.WavReaderPE(_test_wav()),
    "WavWriterPE": lambda: pg.WavWriterPE(_sine(), _tmp_path(".wav")),
    "WavetablePE": lambda: pg.WavetablePE(
        pg.ArrayPE(np.sin(2 * np.pi * np.arange(64) / 64)), pg.IdentityPE()
    ),
    "WindowPE": lambda: pg.WindowPE(_sine(), window=0.01),
}
