"""
IdiophonePE — Struck bar idiophone synthesis PE.

Models marimba, xylophone, glockenspiel, and balafon as a register-scaled
sum of DecayingSinePE partials defined by an InstrumentDef.

Each partial's decay time is computed from tau_mid (seconds at A4) and
tau_scale (multiplier per octave below A4), so bass notes decay more slowly
than treble notes as in the physical instruments.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, log2
from typing import List

from pygmu2.source_pe import SourcePE
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
    """Defines one partial of a struck bar idiophone.

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
    """Complete partial structure for one struck bar idiophone.

    Attributes:
        name:     Identifier string (used in __repr__ and INSTRUMENTS dict).
        partials: Ordered list of PartialDef objects, fundamental first.
    """

    name: str
    partials: List[PartialDef]


# ---------------------------------------------------------------------------
# Instrument library
# ---------------------------------------------------------------------------
#
# tau_mid values are calibrated at A4 (440 Hz).
# tau_scale controls how sharply decay lengthens in the bass:
#   2.0 → doubles per octave down  (marimba, balafon — resonated wood)
#   1.6 → 60 % longer per octave   (upper partials of marimba)
#   1.5 → xylophone fundamental    (dry hardwood, shorter overall)
#   1.6 → glockenspiel fundamental (steel, long but less register-sensitive)
#
# Partial freq_ratios follow measured values for each instrument family;
# glockenspiel uses the free-free bar ratio 2.756f₀ for its first overtone.

MARIMBA = InstrumentDef(
    name="marimba",
    partials=[
        PartialDef(freq_ratio=1.0, tau_mid=1.00, tau_scale=2.0, amp_ratio=1.00),
        PartialDef(freq_ratio=4.0, tau_mid=0.20, tau_scale=1.6, amp_ratio=0.50),
        PartialDef(freq_ratio=10.0, tau_mid=0.05, tau_scale=1.3, amp_ratio=0.15),
    ],
)

XYLOPHONE = InstrumentDef(
    name="xylophone",
    partials=[
        PartialDef(freq_ratio=1.0, tau_mid=0.30, tau_scale=1.5, amp_ratio=1.00),
        PartialDef(freq_ratio=3.0, tau_mid=0.08, tau_scale=1.3, amp_ratio=0.45),
        PartialDef(freq_ratio=6.0, tau_mid=0.03, tau_scale=1.1, amp_ratio=0.18),
    ],
)

GLOCKENSPIEL = InstrumentDef(
    name="glockenspiel",
    partials=[
        PartialDef(freq_ratio=1.000, tau_mid=3.00, tau_scale=1.6, amp_ratio=1.00),
        PartialDef(freq_ratio=2.756, tau_mid=0.80, tau_scale=1.3, amp_ratio=0.55),
        PartialDef(freq_ratio=5.404, tau_mid=0.20, tau_scale=1.1, amp_ratio=0.20),
    ],
)

BALAFON = InstrumentDef(
    name="balafon",
    partials=[
        PartialDef(freq_ratio=1.0, tau_mid=0.90, tau_scale=1.9, amp_ratio=1.00),
        PartialDef(freq_ratio=4.0, tau_mid=0.18, tau_scale=1.5, amp_ratio=0.40),
        PartialDef(freq_ratio=10.0, tau_mid=0.06, tau_scale=1.2, amp_ratio=0.14),
    ],
)

CELESTE = InstrumentDef(
    name="celeste",
    partials=[
        # Fundamental: resonators sustain it well; tau_mid slightly below
        # glockenspiel (3.0 s) because the felt hammer and enclosed cabinet
        # damp the bar itself a little faster.
        PartialDef(freq_ratio=1.000, tau_mid=2.5, tau_scale=1.7, amp_ratio=1.00),
        # First overtone: same ratio as glockenspiel (free-free bar), but
        # amp_ratio drops from 0.55 → 0.30 because the felt hammer excites
        # high-frequency modes much less than a hard mallet.
        # PartialDef(freq_ratio=2.756, tau_mid=0.55, tau_scale=1.4, amp_ratio=0.30),
        PartialDef(freq_ratio=2.0, tau_mid=0.55, tau_scale=1.4, amp_ratio=0.30),
        # Second overtone: similarly attenuated by the soft strike; barely
        # audible but contributes the faint brightness that separates a
        # celeste from a pure sine tone.
        PartialDef(freq_ratio=5.404, tau_mid=0.03, tau_scale=1.1, amp_ratio=0.1),
    ],
)

SINGING_BOWL = InstrumentDef(
    name="singing bowl",
    partials=[
        PartialDef(freq_ratio=1.0, tau_mid=20.0, tau_scale=1.0, amp_ratio=0.7),
        PartialDef(freq_ratio=1.017, tau_mid=20.0, tau_scale=1.0, amp_ratio=0.7),
        PartialDef(freq_ratio=2.9, tau_mid=20.0, tau_scale=1.0, amp_ratio=0.5),
        PartialDef(freq_ratio=5.44, tau_mid=20.0, tau_scale=1.0, amp_ratio=0.8),
        PartialDef(freq_ratio=8.56, tau_mid=20.0, tau_scale=1.0, amp_ratio=0.5),
        PartialDef(freq_ratio=12.22, tau_mid=20.0, tau_scale=1.0, amp_ratio=0.3),
    ],
)

INSTRUMENTS: dict[str, InstrumentDef] = {
    i.name: i
    for i in [MARIMBA, XYLOPHONE, GLOCKENSPIEL, BALAFON, CELESTE, SINGING_BOWL]
}

# ---------------------------------------------------------------------------
# PE
# ---------------------------------------------------------------------------


class IdiophonePE(SourcePE):
    """Struck bar idiophone synthesis PE.

    Produces a note by summing DecayingSinePE partials defined by an
    InstrumentDef.  Each partial's decay time is register-scaled from
    tau_mid using:

        partial_duration = tau_mid × tau_scale^(log₂(REF_FREQ / frequency))

    so that low notes decay more slowly than high notes, matching the
    acoustic behaviour of real bars.

    The internal MixPE is built at construction time.

    Extent is determined by the longest partial's -60 dB crop.

    Args:
        instrument:  InstrumentDef — use MARIMBA, XYLOPHONE, etc.
        frequency:   Fundamental frequency in Hz (default 440.0).
        amplitude:   Peak amplitude of the fundamental partial (default 0.3).
        channels:    Output channel count (default 1).

    Example::
        note = IdiophonePE(MARIMBA, frequency=261.63, amplitude=0.5)
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
    # SourcePE interface
    # ------------------------------------------------------------------

    def _compute_extent(self) -> Extent:
        return self._mix_pe.extent()

    def _reset_state(self) -> None:
        for input_pe in self._mix_pe.inputs():
            input_pe._reset_state()

    def _on_start(self) -> None:
        self._reset_state()

    def _on_stop(self) -> None:
        self._reset_state()

    def _render(self, start: int, duration: int) -> Snippet:
        return self._mix_pe.render(start, duration)

    def channel_count(self) -> int:
        return self._channels

    stateful = True

    def __repr__(self) -> str:
        return (
            f"IdiophonePE(instrument={self._instrument.name!r}, "
            f"frequency={self._frequency}, "
            f"amplitude={self._amplitude})"
        )
