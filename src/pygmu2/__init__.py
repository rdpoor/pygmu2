"""
pygmu2 - A framework for generating and processing digital audio.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from pygmu2.config import ErrorMode, set_error_mode, get_error_mode, handle_error
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.processing_element import ProcessingElement, SourcePE
from pygmu2.renderer import Renderer
from pygmu2.audio_renderer import AudioRenderer
from pygmu2.null_renderer import NullRenderer
from pygmu2.constant_pe import ConstantPE
from pygmu2.crop_pe import CropPE
from pygmu2.delay_pe import DelayPE
from pygmu2.dirac_pe import DiracPE
from pygmu2.gain_pe import GainPE
from pygmu2.identity_pe import IdentityPE
from pygmu2.ramp_pe import RampPE
from pygmu2.sine_pe import SinePE
from pygmu2.mix_pe import MixPE
from pygmu2.wav_reader_pe import WavReaderPE
from pygmu2.wav_writer_pe import WavWriterPE
from pygmu2.logger import setup_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "ErrorMode",
    "set_error_mode",
    "get_error_mode",
    "handle_error",
    # Core classes
    "Extent",
    "Snippet",
    "ProcessingElement",
    "SourcePE",
    "Renderer",
    "AudioRenderer",
    "NullRenderer",
    # Processing Elements
    "ConstantPE",
    "CropPE",
    "DelayPE",
    "DiracPE",
    "GainPE",
    "IdentityPE",
    "MixPE",
    "RampPE",
    "SinePE",
    "WavReaderPE",
    "WavWriterPE",
    # Logging utilities
    "setup_logging",
    "get_logger",
    # Version
    "__version__",
]
