"""
Meltysynth: SoundFont loading and synthesis (modular port of py-meltysynth).

MeltySynth (C) 2021, Py-MeltySynth (C) 2022 Nobuaki Tanaka. MIT License.
See LICENSE in this directory.
"""

from pygmu2.meltysynth.exceptions import MeltysynthError
from pygmu2.meltysynth.math_utils import create_buffer, create_buffer_numpy
from pygmu2.meltysynth.model import (
    Instrument,
    InstrumentRegion,
    LoopMode,
    Preset,
    PresetRegion,
    SampleHeader,
    SoundFont,
    SoundFontInfo,
    SoundFontVersion,
)
from pygmu2.meltysynth.synth import Synthesizer, SynthesizerSettings
from pygmu2.meltysynth.midi import MidiFile, MidiFileSequencer

__all__ = [
    "MeltysynthError",
    "create_buffer",
    "create_buffer_numpy",
    "SoundFont",
    "SoundFontInfo",
    "SoundFontVersion",
    "SampleHeader",
    "Instrument",
    "InstrumentRegion",
    "Preset",
    "PresetRegion",
    "LoopMode",
    "Synthesizer",
    "SynthesizerSettings",
    "MidiFile",
    "MidiFileSequencer",
]
