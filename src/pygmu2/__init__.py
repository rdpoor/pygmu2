"""
pygmu2 - A framework for generating and processing digital audio.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from pygmu2.config import ErrorMode, set_error_mode, get_error_mode, handle_error
from pygmu2.extent import Extent, ExtendMode
from pygmu2.snippet import Snippet
from pygmu2.processing_element import ProcessingElement, SourcePE
from pygmu2.renderer import Renderer, ProfileReport, PEProfile
from pygmu2.audio_renderer import AudioRenderer
from pygmu2.null_renderer import NullRenderer
from pygmu2.adsr_pe import AdsrPE
from pygmu2.array_pe import ArrayPE
from pygmu2.analog_osc_pe import AnalogOscPE
from pygmu2.function_gen_pe import FunctionGenPE
from pygmu2.blit_saw_pe import BlitSawPE
from pygmu2.compressor_pe import CompressorPE, LimiterPE, GatePE
from pygmu2.constant_pe import ConstantPE
from pygmu2.crop_pe import CropPE
from pygmu2.delay_pe import DelayPE
from pygmu2.dirac_pe import DiracPE
from pygmu2.dynamics_pe import DynamicsPE, DynamicsMode
from pygmu2.envelope_pe import EnvelopePE, DetectionMode
from pygmu2.gain_pe import GainPE
from pygmu2.identity_pe import IdentityPE
from pygmu2.karplus_strong_pe import KarplusStrongPE, rho_for_decay_db
from pygmu2.ladder_pe import LadderPE, LadderMode
from pygmu2.loop_pe import LoopPE
from pygmu2.meltysynth_pe import MeltysynthPE
from pygmu2.midi_in_pe import MidiInPE
from pygmu2.random_pe import RandomPE, RandomMode
from pygmu2.reset_pe import ResetPE
from pygmu2.sine_pe import SinePE
from pygmu2.super_saw_pe import SuperSawPE
from pygmu2.mix_pe import MixPE
from pygmu2.comb_pe import CombPE
from pygmu2.convolve_pe import ConvolvePE
from pygmu2.audio_library import AudioLibrary
from pygmu2.slice_pe import SlicePE
from pygmu2.spatial_pe import (
    SpatialPE,
    SpatialMethod,
    SpatialAdapter,
    SpatialLinear,
    SpatialConstantPower,
    SpatialHRTF,
)
from pygmu2.wav_reader_pe import WavReaderPE
from pygmu2.wav_writer_pe import WavWriterPE
from pygmu2.transform_pe import TransformPE
from pygmu2.trigger_pe import TriggerPE, TriggerMode
from pygmu2.wavetable_pe import WavetablePE, InterpolationMode, OutOfBoundsMode
from pygmu2.window_pe import WindowPE, WindowMode
from pygmu2.reverse_pitch_echo_pe import ReversePitchEchoPE
from pygmu2.timewarp_pe import TimeWarpPE
from pygmu2.piecewise_pe import PiecewisePE, TransitionType
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
from pygmu2.logger import setup_logging, get_logger
from pygmu2.debug_utils import print_pe_tree

__version__ = "0.1.0"

# Lazy imports for modules with heavy dependencies (scipy)
# These are loaded on first access to avoid slow startup for simple scripts
_lazy_imports = {
    "BiquadPE": ("pygmu2.biquad_pe", "BiquadPE"),
    "BiquadMode": ("pygmu2.biquad_pe", "BiquadMode"),
    "SVFilterPE": ("pygmu2.svfilter_pe", "SVFilterPE"),
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
    "ErrorMode",
    "set_error_mode",
    "get_error_mode",
    "handle_error",
    # Core classes
    "Extent",
    "ExtendMode",
    "Snippet",
    "ProcessingElement",
    "SourcePE",
    "Renderer",
    "ProfileReport",
    "PEProfile",
    "AudioRenderer",
    "NullRenderer",
    # Processing Elements
    "AdsrPE",
    "AnalogOscPE",
    "ArrayPE",
    "FunctionGenPE",
    "PiecewisePE",
    "TransitionType",
    "BiquadPE",
    "BlitSawPE",
    "CompressorPE",
    "ConstantPE",
    "CropPE",
    "DelayPE",
    "DiracPE",
    "DynamicsPE",
    "EnvelopePE",
    "GainPE",
    "GatePE",
    "IdentityPE",
    "KarplusStrongPE",
    "rho_for_decay_db",
    "LadderPE",
    "LimiterPE",
    "LoopPE",
    "MeltysynthPE",
    "MidiInPE",
    "MixPE",
    "CombPE",
    "ConvolvePE",
    "AudioLibrary",
    "RandomPE",
    "ResetPE",
    "SlicePE",
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
    "TriggerPE",
    "WavReaderPE",
    "WavWriterPE",
    "WavetablePE",
    "TimeWarpPE",
    "WindowPE",
    "ReversePitchEchoPE",
    # Enums
    "BiquadMode",
    "DetectionMode",
    "DynamicsMode",
    "ExtendMode",
    "LadderMode",
    "InterpolationMode",
    "OutOfBoundsMode",
    "RandomMode",
    "TriggerMode",
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
    "setup_logging",
    "get_logger",
    # Debug utilities
    "print_pe_tree",
    # Version
    "__version__",
]
