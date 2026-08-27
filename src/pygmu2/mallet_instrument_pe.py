"""
MalletInstrumentPE — a struck instrument that makes a tone.

Synthesizes a mallet-struck note (marimba, glockenspiel, singing bowl, ...)
as a register-scaled sum of DecayingSinePE partials defined by an
InstrumentDef. Each partial's decay time is computed from tau_mid (seconds
at A4) and tau_scale (multiplier per octave below A4), so bass notes decay
more slowly than treble notes, as in the physical instruments.

This module carries the synthesis engine and the definition types
(PartialDef, InstrumentDef); the predefined instruments live in the
companion class MalletInstruments (mallet_instruments.py).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, log2
from typing import List

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.mix_pe import MixPE
from pygmu2.decaying_sine_pe import DecayingSinePE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REF_FREQ: float = 440.0  # A4 — the register pivot for tau scaling
TAU_FACTOR: float = 60.0 / 20.0 * log(10)  # ≈ 6.908 — convert -60 dB time to τ

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PartialDef:
    """Defines one partial of a struck (mallet) instrument.

    Attributes:
        freq_ratio:  Partial frequency as a multiple of f₀.
        tau_mid:     Decay time in seconds at REF_FREQ (A4 = 440 Hz).
        tau_scale:   Decay time multiplier per octave *below* REF_FREQ.
                     Values > 1 produce longer decays in the bass register,
                     matching the behaviour of real wooden and metal bars.
        amp_ratio:   Partial amplitude as a fraction of the note amplitude.
    """

    freq_ratio: float
    tau_mid: float
    tau_scale: float
    amp_ratio: float


@dataclass
class InstrumentDef:
    """Complete partial structure for one mallet instrument.

    Attributes:
        name:     Identifier string (used in __repr__ and preset catalogs).
        partials: Ordered list of PartialDef objects, fundamental first.
    """

    name: str
    partials: List[PartialDef]


# ---------------------------------------------------------------------------
# PE
# ---------------------------------------------------------------------------


class MalletInstrumentPE(ProcessingElement):
    """A struck instrument that makes a tone.

    Produces a note by summing DecayingSinePE partials defined by an
    InstrumentDef.  Each partial's decay time is register-scaled from
    tau_mid using:

        partial_duration = tau_mid × tau_scale^(log₂(REF_FREQ / frequency))

    so that low notes decay more slowly than high notes, matching the
    acoustic behaviour of real bars.

    The internal MixPE is built at construction time.

    Extent is determined by the longest partial's -60 dB crop.

    Args:
        instrument:  InstrumentDef — a MalletInstruments preset
                     (e.g. MalletInstruments.MARIMBA) or your own.
        frequency:   Fundamental frequency in Hz (default 440.0).
        amplitude:   Peak amplitude of the fundamental partial (default 0.3).
        channels:    Output channel count (default 1).

    Example::
        note = MalletInstrumentPE(
            MalletInstruments.MARIMBA, frequency=261.63, amplitude=0.5
        )
        cropped = CropPE(note, 0, 3 * note.sample_rate)  # optional crop
    """

    def __init__(
        self,
        instrument: InstrumentDef,
        frequency: float = 440.0,
        amplitude: float = 0.3,
        channels: int = 1,
    ):
        super().__init__()
        if frequency <= 0:
            raise ValueError(f"frequency must be positive, got {frequency}")
        if amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {amplitude}")

        self._instrument = instrument
        self._frequency = float(frequency)
        self._amplitude = float(amplitude)
        self._channels = channels
        self._mix_pe = self._build_mix_pe()

    # ------------------------------------------------------------------
    # Internal construction
    # ------------------------------------------------------------------

    def _build_mix_pe(self) -> MixPE:
        octaves_below_ref = log2(REF_FREQ / self._frequency)
        partials = []
        for p in self._instrument.partials:
            partial_freq = self._frequency * p.freq_ratio
            partial_tau = p.tau_mid * (p.tau_scale**octaves_below_ref) / TAU_FACTOR
            partial_amp = self._amplitude * p.amp_ratio
            sine = DecayingSinePE(
                frequency=partial_freq,
                amplitude=partial_amp,
                tau=partial_tau,
                channels=self._channels,
            )
            partials.append(sine)
        return MixPE(*partials)

    # ------------------------------------------------------------------
    # ProcessingElement interface
    # ------------------------------------------------------------------

    def _compute_extent(self) -> Extent:
        return self._mix_pe.extent()

    def inputs(self) -> List[ProcessingElement]:
        # Expose the internal graph so the Renderer (and any graph walk)
        # manages lifecycle and reset for the partials — the composite
        # itself holds no state; the DecayingSinePE partials declare theirs.
        return [self._mix_pe]

    def _render(self, start: int, duration: int) -> Snippet:
        return self._mix_pe.render(start, duration)

    def channel_count(self) -> int:
        return self._channels

    def __repr__(self) -> str:
        return (
            f"MalletInstrumentPE(instrument={self._instrument.name!r}, "
            f"frequency={self._frequency}, "
            f"amplitude={self._amplitude})"
        )
