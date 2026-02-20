"""
CompressorPE, LimiterPE, ExpanderPE — dynamics processors.

All three are built on the shared _DynamicsProcessorPE base which wires
together a CachePE, EnvelopePE, and DynamicsPE and provides the common
infrastructure (inputs, is_pure, channel_count, extent, render).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""


import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.cache_pe import CachePE
from pygmu2.envelope_pe import EnvelopePE, DetectionMode
from pygmu2.dynamics_pe import DynamicsPE, DynamicsMode
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class _DynamicsProcessorPE(ProcessingElement):
    """
    Private base class shared by CompressorPE and ExpanderPE.

    Owns the common internal graph (CachePE → EnvelopePE → DynamicsPE) and
    delegates all PE infrastructure to the DynamicsPE output.  Subclasses
    construct the internal PEs and pass them via super().__init__().
    """

    def __init__(
        self,
        cached_source: ProcessingElement,
        envelope_pe: EnvelopePE,
        dynamics_pe: DynamicsPE,
        *,
        threshold: float,
        attack: float,
        release: float,
        knee: float,
        stereo_link: bool,
    ):
        self._source = cached_source
        self._envelope_pe = envelope_pe
        self._dynamics_pe = dynamics_pe
        self._threshold = threshold
        self._attack = attack
        self._release = release
        self._knee = knee
        self._stereo_link = stereo_link

    # --- shared properties ---------------------------------------------------

    @property
    def threshold(self) -> float:
        """Threshold in dB."""
        return self._threshold

    @property
    def attack(self) -> float:
        """Attack time in seconds."""
        return self._attack

    @property
    def release(self) -> float:
        """Release time in seconds."""
        return self._release

    @property
    def knee(self) -> float:
        """Soft knee width in dB."""
        return self._knee

    @property
    def stereo_link(self) -> bool:
        """Whether stereo channels are linked."""
        return self._stereo_link

    # --- PE infrastructure ---------------------------------------------------

    def inputs(self) -> list[ProcessingElement]:
        """Expose the internal graph so the Renderer manages all lifecycle."""
        return [self._dynamics_pe]

    def is_pure(self) -> bool:
        return False

    def channel_count(self) -> int | None:
        return self._dynamics_pe.channel_count()

    def _compute_extent(self) -> Extent:
        return self._dynamics_pe.extent()

    def _render(self, start: int, duration: int) -> Snippet:
        return self._dynamics_pe.render(start, duration)


class CompressorPE(_DynamicsProcessorPE):
    """
    All-in-one audio compressor with integrated envelope follower.

    Provides a convenient interface for common compression tasks without
    needing to manually create and connect EnvelopePE and DynamicsPE.
    For sidechain compression or advanced routing, use DynamicsPE directly.

    Args:
        source: Audio input to compress
        threshold: Level in dB where compression begins (default: -20.0)
        ratio: Compression ratio (default: 4.0)
                - 1.0 = no compression
                - 4.0 = 4:1 compression
                - float('inf') or very high = limiting
        attack: Envelope attack time in seconds (default: 0.01)
        release: Envelope release time in seconds (default: 0.1)
        knee: Soft knee width in dB (default: 6.0)
        makeup_gain: Output gain in dB, or "auto" (default: "auto")
        lookahead: Lookahead time in seconds (default: 0.0)
        detection: Envelope detection mode (default: DetectionMode.RMS)
        stereo_link: If True, use max envelope across channels (default: True)

    Example:
        compressed_stream = CompressorPE(audio_stream, threshold=-20, ratio=4)
    """

    AUTO = "auto"

    def __init__(
        self,
        source: ProcessingElement,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 0.01,
        release: float = 0.1,
        knee: float = 6.0,
        makeup_gain: float | str = "auto",
        lookahead: float = 0.0,
        detection: DetectionMode = DetectionMode.RMS,
        stereo_link: bool = True,
    ):
        cached = CachePE(source)
        envelope_pe = EnvelopePE(
            cached,
            attack=attack,
            release=release,
            lookahead=lookahead,
            mode=detection,
        )
        dynamics_pe = DynamicsPE(
            cached,
            envelope_pe,
            threshold=threshold,
            ratio=ratio,
            knee=knee,
            makeup_gain=makeup_gain,
            mode=DynamicsMode.COMPRESS,
            stereo_link=stereo_link,
        )
        super().__init__(
            cached, envelope_pe, dynamics_pe,
            threshold=threshold,
            attack=attack,
            release=release,
            knee=knee,
            stereo_link=stereo_link,
        )
        self._ratio = ratio
        self._makeup_gain = makeup_gain
        self._lookahead = lookahead
        self._detection = detection

    @property
    def ratio(self) -> float:
        """Compression ratio."""
        return self._ratio

    @property
    def makeup_gain(self) -> float:
        """Makeup gain in dB (actual value, even if auto)."""
        return self._dynamics_pe.makeup_gain

    @property
    def lookahead(self) -> float:
        """Lookahead time in seconds."""
        return self._lookahead

    @property
    def detection(self) -> DetectionMode:
        """Envelope detection mode."""
        return self._detection

    def __repr__(self) -> str:
        makeup_str = "auto" if self._makeup_gain == self.AUTO else f"{self.makeup_gain:.1f}"
        return (
            f"CompressorPE(threshold={self._threshold}, ratio={self._ratio}, "
            f"attack={self._attack}, release={self._release}, knee={self._knee}, "
            f"makeup={makeup_str}, lookahead={self._lookahead})"
        )


class LimiterPE(CompressorPE):
    """
    Brick-wall limiter — prevents signal from exceeding a ceiling level.

    A specialised compressor with very high ratio, fast attack,
    and optional lookahead for transparent limiting.

    Args:
        source: Audio input to limit
        ceiling: Maximum output level in dB (default: -1.0)
        attack: Attack time in seconds (default: 0.0005)
        release: Release time in seconds (default: 0.05)
        lookahead: Lookahead time in seconds (default: 0.005)
        stereo_link: If True, use max envelope across channels (default: True)

    Example:
        limited_stream = LimiterPE(audio_stream, ceiling=-1.0)
    """

    def __init__(
        self,
        source: ProcessingElement,
        ceiling: float = -1.0,
        attack: float = 0.0005,
        release: float = 0.05,
        lookahead: float = 0.005,
        stereo_link: bool = True,
    ):
        super().__init__(
            source,
            threshold=ceiling,
            ratio=100.0,
            attack=attack,
            release=release,
            knee=0.0,
            makeup_gain=0.0,
            lookahead=lookahead,
            detection=DetectionMode.PEAK,
            stereo_link=stereo_link,
        )
        self._ceiling = ceiling

    @property
    def ceiling(self) -> float:
        """Ceiling level in dB."""
        return self._ceiling

    def __repr__(self) -> str:
        return (
            f"LimiterPE(ceiling={self._ceiling}, release={self._release}, "
            f"lookahead={self._lookahead})"
        )


class ExpanderPE(_DynamicsProcessorPE):
    """
    Downward expander / noise gate — attenuates signal below threshold.

    Uses EnvelopePE for level detection and DynamicsPE in GATE mode.
    Unlike a compressor, an expander reduces the level of quiet signals
    rather than loud ones.

    Args:
        source: Audio input
        threshold: Level in dB below which the signal is attenuated (default: -40.0)
        attack: Envelope attack time in seconds (default: 0.001)
        release: Envelope release time in seconds (default: 0.05)
        gate_range: Maximum attenuation in dB when fully closed (default: -80.0)
        knee: Soft knee width in dB (default: 0.0)
        stereo_link: If True, use max envelope across channels (default: True)

    Example:
        gated_stream = ExpanderPE(audio_stream, threshold=-40)
        drum_gate   = ExpanderPE(drums_stream, threshold=-30, attack=0.0005, release=0.1)
    """

    def __init__(
        self,
        source: ProcessingElement,
        threshold: float = -40.0,
        attack: float = 0.001,
        release: float = 0.05,
        gate_range: float = -80.0,
        knee: float = 0.0,
        stereo_link: bool = True,
    ):
        cached = CachePE(source)
        envelope_pe = EnvelopePE(
            cached,
            attack=attack,
            release=release,
            mode=DetectionMode.PEAK,
        )
        dynamics_pe = DynamicsPE(
            cached,
            envelope_pe,
            threshold=threshold,
            ratio=1.0,
            knee=knee,
            makeup_gain=0.0,
            mode=DynamicsMode.GATE,
            stereo_link=stereo_link,
            gate_range=gate_range,
        )
        super().__init__(
            cached, envelope_pe, dynamics_pe,
            threshold=threshold,
            attack=attack,
            release=release,
            knee=knee,
            stereo_link=stereo_link,
        )
        self._range = gate_range

    @property
    def gate_range(self) -> float:
        """Gate attenuation in dB."""
        return self._range

    def __repr__(self) -> str:
        return (
            f"ExpanderPE(threshold={self._threshold}, attack={self._attack}, "
            f"release={self._release}, range={self._range})"
        )
