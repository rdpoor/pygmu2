"""
MalletInstruments — predefined instruments for MalletInstrumentPE.

This companion class has exactly one contract: a curated catalog of
InstrumentDef presets (partial structures and decay calibrations for
real struck instruments). The synthesis engine and the definition types
live in mallet_instrument_pe.py; to design your own instrument, build an
InstrumentDef directly and pass it to MalletInstrumentPE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

from pygmu2.mallet_instrument_pe import InstrumentDef, PartialDef


class MalletInstruments:
    """Predefined InstrumentDef presets for MalletInstrumentPE.

    Usage::

        from pygmu2 import MalletInstrumentPE, MalletInstruments

        note = MalletInstrumentPE(MalletInstruments.MARIMBA, frequency=261.63)

    Calibration notes:
    # tau_mid values are calibrated at A4 (440 Hz).
    # tau_scale controls how sharply decay lengthens in the bass:
    #   2.0 → doubles per octave down  (marimba, balafon — resonated wood)
    #   1.6 → 60 % longer per octave   (upper partials of marimba)
    #   1.5 → xylophone fundamental    (dry hardwood, shorter overall)
    #   1.6 → glockenspiel fundamental (steel, long but less register-sensitive)
    #
    # Partial freq_ratios follow measured values for each instrument family;
    # glockenspiel uses the free-free bar ratio 2.756f₀ for its first overtone.
    """

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

    @classmethod
    def all(cls) -> dict[str, InstrumentDef]:
        """All presets, keyed by their InstrumentDef.name."""
        return {
            i.name: i
            for i in (
                cls.MARIMBA,
                cls.XYLOPHONE,
                cls.GLOCKENSPIEL,
                cls.BALAFON,
                cls.CELESTE,
                cls.SINGING_BOWL,
            )
        }
