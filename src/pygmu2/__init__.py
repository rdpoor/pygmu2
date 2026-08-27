"""
pygmu2 - A framework for generating and processing digital audio.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from pygmu2.adsr_pe import (
    AdsrGatedPE,
    AdsrTriggeredPE,
)
from pygmu2.analog_osc_pe import AnalogOscPE
from pygmu2.array_pe import ArrayPE
from pygmu2.asset_manager import (
    AssetManager,
    GoogleDriveAssetLoader,
    GithubUserContentAssetLoader,
)
from pygmu2.audio_library import AudioLibrary
from pygmu2.cache_pe import CachePE
from pygmu2.config import (
    set_sample_rate,
    get_sample_rate,
)
from pygmu2.constant_pe import ConstantPE
from pygmu2.control_pe import ControlPE
from pygmu2.conversions import (
    pitch_to_freq,
    freq_to_pitch,
    ratio_to_db,
    db_to_ratio,
    semitones_to_ratio,
    ratio_to_semitones,
    samples_to_seconds,
    seconds_to_samples,
)
from pygmu2.convolve_pe import ConvolvePE
from pygmu2.crop_pe import CropPE
from pygmu2.debug_utils import print_pe_tree
from pygmu2.decaying_sine_pe import DecayingSinePE
from pygmu2.delay_pe import DelayPE
from pygmu2.dirac_pe import DiracPE
from pygmu2.dynamics_pe import DynamicsPE, DynamicsMode
from pygmu2.extent import Extent, ExtendMode
from pygmu2.function_gen_pe import FunctionGenPE
from pygmu2.gain_pe import GainPE
from pygmu2.gate_signal import GateSignal
from pygmu2.gate_to_trigger_pe import GateToTriggerPE
from pygmu2.hold_pe import HoldPE
from pygmu2.identity_pe import IdentityPE
from pygmu2.idiophone_pe import IdiophonePE
from pygmu2.karplus_strong_pe import KarplusStrongPE, rho_for_decay_db
from pygmu2.logger import set_global_logging, setup_logging, get_logger
from pygmu2.loop_pe import LoopPE
from pygmu2.mag_freq_pe import MagFreqPE
from pygmu2.mix_pe import MixPE
from pygmu2.moving_average_pe import MovingAveragePE, window_for_cutoff
from pygmu2.noise_pe import NoisePE, NoiseMode
from pygmu2.null_renderer import NullRenderer
from pygmu2.periodic_gate import PeriodicGate
from pygmu2.periodic_trigger import PeriodicTrigger
from pygmu2.piecewise_pe import PiecewisePE, TransitionType
from pygmu2.processing_element import ProcessingElement
from pygmu2.source_pe import SourcePE
from pygmu2.random_gate_pe import RandomGatePE
from pygmu2.random_select_pe import RandomSelectPE
from pygmu2.random_step_pe import RandomStepPE
from pygmu2.random_trigger_pe import RandomTriggerPE
from pygmu2.random_value_pe import RandomValuePE
from pygmu2.renderer import Renderer
from pygmu2.resample_pe import ResamplePE
from pygmu2.reverb_pe import ReverbPE
from pygmu2.ring_modulator_pe import RingModulatorPE
from pygmu2.scheduled_gate_pe import ScheduledGatePE
from pygmu2.signal_to_gate_pe import SignalToGatePE
from pygmu2.sequence_pe import SequencePE, SequenceMode
from pygmu2.slew_limiter_pe import SlewLimiterPE, SlewMode
from pygmu2.set_extent_pe import SetExtentPE
from pygmu2.sine_pe import SinePE
from pygmu2.slice_pe import SlicePE
from pygmu2.snippet import Snippet
from pygmu2.temperament import (
    Temperament,
    EqualTemperament,
    JustIntonation,
    PythagoreanTuning,
    CustomTemperament,
    set_temperament,
    get_temperament,
    set_reference_frequency,
    get_reference_frequency,
    set_concert_pitch,
    set_verdi_tuning,
    set_baroque_pitch,
)
from pygmu2.timewarp_pe import TimeWarpPE
from pygmu2.tralfam_pe import TralfamPE
from pygmu2.transform_pe import TransformPE
from pygmu2.trigger_restart_pe import TriggerRestartPE
from pygmu2.trigger_signal import TriggerSignal
from pygmu2.utils import browse, play, play_offline, render_to_file
from pygmu2.wav_reader_pe import WavReaderPE
from pygmu2.wav_writer_pe import WavWriterPE
from pygmu2.wavetable_pe import WavetablePE, InterpolationMode, OutOfBoundsMode

try:  # single source of truth: [project] version in pyproject.toml
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("pygmu2")
except Exception:  # uninstalled source tree
    __version__ = "0.0.0+unknown"

# Lazy imports for modules with heavy dependencies (scipy, numba, mido,
# miniaudio, vendored meltysynth). Loaded on first access so that
# `import pygmu2` stays fast and works without the optional extras; using
# one of these without its dependency raises the natural ImportError at
# first use (DESIGN_PHILOSOPHY.md PD-2). The import-hygiene test in
# tests/test_boundaries.py asserts scipy/numba stay out of a bare import.
_lazy_imports = {
    "AudioReaderPE": ("pygmu2.audio_reader_pe", "AudioReaderPE"),
    "AudioRenderer": ("pygmu2.audio_renderer", "AudioRenderer"),
    "BiquadMode": ("pygmu2.biquad_pe", "BiquadMode"),
    "BiquadPE": ("pygmu2.biquad_pe", "BiquadPE"),
    "BlitSawPE": ("pygmu2.blit_saw_pe", "BlitSawPE"),
    "CombPE": ("pygmu2.comb_pe", "CombPE"),
    "CompressorPE": ("pygmu2.compressor_pe", "CompressorPE"),
    "DetectionMode": ("pygmu2.envelope_pe", "DetectionMode"),
    "EnvelopePE": ("pygmu2.envelope_pe", "EnvelopePE"),
    "ExpanderPE": ("pygmu2.compressor_pe", "ExpanderPE"),
    "LadderMode": ("pygmu2.ladder_pe", "LadderMode"),
    "LadderPE": ("pygmu2.ladder_pe", "LadderPE"),
    "LimiterPE": ("pygmu2.compressor_pe", "LimiterPE"),
    "MeltysynthPE": ("pygmu2.meltysynth_pe", "MeltysynthPE"),
    "MidiInPE": ("pygmu2.midi_in_pe", "MidiInPE"),
    "Note": ("pygmu2.notes_pe", "Note"),
    "NotesPE": ("pygmu2.notes_pe", "NotesPE"),
    "ReversePitchEchoPE": ("pygmu2.reverse_pitch_echo_pe", "ReversePitchEchoPE"),
    "SVFilterPE": ("pygmu2.svfilter_pe", "SVFilterPE"),
    "SpatialAdapter": ("pygmu2.spatial_pe", "SpatialAdapter"),
    "SpatialConstantPower": ("pygmu2.spatial_pe", "SpatialConstantPower"),
    "SpatialHRTF": ("pygmu2.spatial_pe", "SpatialHRTF"),
    "SpatialLinear": ("pygmu2.spatial_pe", "SpatialLinear"),
    "SpatialMethod": ("pygmu2.spatial_pe", "SpatialMethod"),
    "SpatialPE": ("pygmu2.spatial_pe", "SpatialPE"),
    "SuperSawPE": ("pygmu2.super_saw_pe", "SuperSawPE"),
    "WindowMode": ("pygmu2.window_pe", "WindowMode"),
    "WindowPE": ("pygmu2.window_pe", "WindowPE"),
    "get_notes_from_midi": ("pygmu2.notes_pe", "get_notes_from_midi"),
}


def __getattr__(name):
    if name in _lazy_imports:
        module_name, attr_name = _lazy_imports[name]
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'pygmu2' has no attribute {name!r}")


__all__ = [
    # Configuration
    "set_sample_rate",
    "get_sample_rate",
    # Core classes
    "Extent",
    "Snippet",
    "ProcessingElement",
    "SourcePE",
    "GateSignal",
    "GateToTriggerPE",
    "TriggerSignal",
    "Renderer",
    "AudioRenderer",
    "NullRenderer",
    # Processing Elements
    "AdsrGatedPE",
    "AdsrTriggeredPE",
    "AnalogOscPE",
    "ArrayPE",
    "FunctionGenPE",
    "PiecewisePE",
    "TransitionType",
    "BiquadPE",
    "BlitSawPE",
    "CompressorPE",
    "ConstantPE",
    "ControlPE",
    "CropPE",
    "SetExtentPE",
    "DecayingSinePE",
    "DelayPE",
    "DiracPE",
    "DynamicsPE",
    "EnvelopePE",
    "GainPE",
    "ExpanderPE",
    "IdentityPE",
    "IdiophonePE",
    "KarplusStrongPE",
    "rho_for_decay_db",
    "LadderPE",
    "LimiterPE",
    "LoopPE",
    "MagFreqPE",
    "MeltysynthPE",
    "MidiInPE",
    "MixPE",
    "MovingAveragePE",
    "window_for_cutoff",
    "Note",
    "NotesPE",
    "get_notes_from_midi",
    "ResamplePE",
    "CombPE",
    "ConvolvePE",
    "ReverbPE",
    "CachePE",
    "AudioLibrary",
    "AssetManager",
    "GoogleDriveAssetLoader",
    "GithubUserContentAssetLoader",
    "NoisePE",
    "PeriodicGate",
    "PeriodicTrigger",
    "RandomGatePE",
    "RandomSelectPE",
    "RandomStepPE",
    "RandomTriggerPE",
    "RandomValuePE",
    "HoldPE",
    "ScheduledGatePE",
    "SignalToGatePE",
    "SlewLimiterPE",
    "TriggerRestartPE",
    "SlicePE",
    "SequencePE",
    "SinePE",
    "SVFilterPE",
    "SpatialPE",
    "SpatialMethod",
    "SpatialAdapter",
    "SpatialLinear",
    "SpatialConstantPower",
    "SpatialHRTF",
    "SuperSawPE",
    "TransformPE",
    "AudioReaderPE",
    "WavReaderPE",
    "WavWriterPE",
    "WavetablePE",
    "TimeWarpPE",
    "TralfamPE",
    "WindowPE",
    "ReversePitchEchoPE",
    "RingModulatorPE",
    # Enums
    "BiquadMode",
    "DetectionMode",
    "DynamicsMode",
    "ExtendMode",
    "LadderMode",
    "InterpolationMode",
    "OutOfBoundsMode",
    "NoiseMode",
    "SlewMode",
    "SequenceMode",
    "WindowMode",
    # Conversion functions
    "pitch_to_freq",
    "freq_to_pitch",
    "ratio_to_db",
    "db_to_ratio",
    "semitones_to_ratio",
    "ratio_to_semitones",
    "samples_to_seconds",
    "seconds_to_samples",
    # Temperament system
    "Temperament",
    "EqualTemperament",
    "JustIntonation",
    "PythagoreanTuning",
    "CustomTemperament",
    "set_temperament",
    "get_temperament",
    "set_reference_frequency",
    "get_reference_frequency",
    "set_concert_pitch",
    "set_verdi_tuning",
    "set_baroque_pitch",
    # Logging utilities
    "set_global_logging",
    "setup_logging",
    "get_logger",
    # Playback utilities
    "browse",
    "play",
    "play_offline",
    "render_to_file",
    # Debug utilities
    "print_pe_tree",
    # Version
    "__version__",
]
