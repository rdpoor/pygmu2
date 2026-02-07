"""
CompressorPE - All-in-one audio compressor with integrated envelope follower.

Combines EnvelopePE and DynamicsPE into a single, easy-to-use PE.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from typing import Optional, Union

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.envelope_pe import EnvelopePE, DetectionMode
from pygmu2.dynamics_pe import DynamicsPE, DynamicsMode
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class CompressorPE(ProcessingElement):
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
                   Allows envelope to anticipate transients.
        detection: Envelope detection mode (default: DetectionMode.RMS)
        stereo_link: If True, use max envelope across channels (default: True)
    
    Example:
        # Simple compression
        compressed_stream = CompressorPE(audio_stream, threshold=-20, ratio=4)
        
        # Aggressive compression with fast attack
        compressed_stream = CompressorPE(
            audio_stream,
            threshold=-15,
            ratio=8,
            attack=0.001,
            release=0.05,
        )
        
        # Brick-wall limiter
        limited_stream = CompressorPE(
            audio_stream,
            threshold=-1,
            ratio=100,
            attack=0.0005,
            release=0.05,
            lookahead=0.001,
        )
        
        # Gentle bus compression
        glue_stream = CompressorPE(
            mix_bus_stream,
            threshold=-25,
            ratio=2,
            attack=0.03,
            release=0.3,
            knee=12,
        )
    """
    
    # Sentinel for auto makeup gain
    AUTO = "auto"
    
    def __init__(
        self,
        source: ProcessingElement,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 0.01,
        release: float = 0.1,
        knee: float = 6.0,
        makeup_gain: Union[float, str] = "auto",
        lookahead: float = 0.0,
        detection: DetectionMode = DetectionMode.RMS,
        stereo_link: bool = True,
    ):
        self._source = source
        self._threshold = threshold
        self._ratio = ratio
        self._attack = attack
        self._release = release
        self._knee = knee
        self._makeup_gain = makeup_gain
        self._lookahead = lookahead
        self._detection = detection
        self._stereo_link = stereo_link
        
        # Create internal envelope follower
        self._envelope_pe = EnvelopePE(
            source,
            attack=attack,
            release=release,
            lookahead=lookahead,
            mode=detection,
        )
        
        # Create internal dynamics processor
        self._dynamics_pe = DynamicsPE(
            source,
            self._envelope_pe,
            threshold=threshold,
            ratio=ratio,
            knee=knee,
            makeup_gain=makeup_gain,
            mode=DynamicsMode.COMPRESS,
            stereo_link=stereo_link,
        )
    
    @property
    def threshold(self) -> float:
        """Threshold in dB."""
        return self._threshold
    
    @property
    def ratio(self) -> float:
        """Compression ratio."""
        return self._ratio
    
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
    
    @property
    def stereo_link(self) -> bool:
        """Whether stereo channels are linked."""
        return self._stereo_link
    
    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs (just the source, internal PEs are hidden)."""
        return [self._source]
    
    def is_pure(self) -> bool:
        """
        CompressorPE is NOT pure due to envelope state.
        """
        return False
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """Return extent from source."""
        return self._source.extent()
    
    def _on_start(self) -> None:
        """Start internal PEs."""
        self._envelope_pe.on_start()
        # DynamicsPE is pure, no on_start needed

    def _on_stop(self) -> None:
        """Stop internal PEs."""
        self._envelope_pe.on_stop()
    
    def _reset_state(self) -> None:
        """Reset internal envelope state."""
        self._envelope_pe.reset_state()
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render compressed audio.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
        Returns:
            Snippet containing compressed audio
        """
        # Just delegate to the internal dynamics PE
        return self._dynamics_pe.render(start, duration)
    
    def __repr__(self) -> str:
        makeup_str = "auto" if self._makeup_gain == self.AUTO else f"{self.makeup_gain:.1f}"
        return (
            f"CompressorPE(threshold={self._threshold}, ratio={self._ratio}, "
            f"attack={self._attack}, release={self._release}, knee={self._knee}, "
            f"makeup={makeup_str}, lookahead={self._lookahead})"
        )


class LimiterPE(CompressorPE):
    """
    Brick-wall limiter - prevents signal from exceeding ceiling level.
    
    A specialized compressor with very high ratio, fast attack,
    and optional lookahead for transparent limiting.
    
    Args:
        source: Audio input to limit
        ceiling: Maximum output level in dB (default: -1.0)
        attack: Attack time in seconds (default: 0.0005)
        release: Release time in seconds (default: 0.05)
        lookahead: Lookahead time in seconds (default: 0.005)
        stereo_link: If True, use max envelope across channels (default: True)
    
    Example:
        # Simple limiting at -1dB
        limited_stream = LimiterPE(audio_stream)
        
        # Transparent limiting with lookahead
        limited_stream = LimiterPE(audio_stream, ceiling=-0.5, lookahead=0.005)
        
        # Slower attack for less aggressive limiting
        limited_stream = LimiterPE(audio_stream, ceiling=-1.0, attack=0.01)
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
            ratio=100.0,  # Very high ratio ≈ limiting
            attack=attack,
            release=release,
            knee=0.0,  # Hard knee for limiting
            makeup_gain=0.0,  # No makeup for limiter
            lookahead=lookahead,
            detection=DetectionMode.PEAK,  # Peak detection for limiting
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


class GatePE(ProcessingElement):
    """
    Noise gate - silences signal below threshold level.
    
    Uses EnvelopePE for detection and DynamicsPE in GATE mode.
    
    Args:
        source: Audio input to gate
        threshold: Level in dB below which signal is gated (default: -40.0)
        attack: Envelope attack time in seconds (default: 0.001)
        release: Envelope release time in seconds (default: 0.05)
        hold: Hold time before release begins (default: 0.01) [NOT YET IMPLEMENTED]
        range: Attenuation in dB when gated (default: -80.0)
        knee: Soft knee width in dB (default: 0.0)
        stereo_link: If True, use max envelope across channels (default: True)
    
    Example:
        # Simple noise gate
        gated_stream = GatePE(audio_stream, threshold=-40)
        
        # Drum gate with fast attack
        gated_stream = GatePE(drums_stream, threshold=-30, attack=0.0005, release=0.1)
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        threshold: float = -40.0,
        attack: float = 0.001,
        release: float = 0.05,
        range: float = -80.0,
        knee: float = 0.0,
        stereo_link: bool = True,
    ):
        self._source = source
        self._threshold = threshold
        self._attack = attack
        self._release = release
        self._range = range
        self._knee = knee
        self._stereo_link = stereo_link
        
        # Create internal envelope follower
        self._envelope_pe = EnvelopePE(
            source,
            attack=attack,
            release=release,
            mode=DetectionMode.PEAK,
        )
        
        # Create internal dynamics processor in GATE mode
        self._dynamics_pe = DynamicsPE(
            source,
            self._envelope_pe,
            threshold=threshold,
            ratio=1.0,  # Not used in gate mode
            knee=knee,
            makeup_gain=0.0,
            mode=DynamicsMode.GATE,
            stereo_link=stereo_link,
            range=range,
        )
    
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
    def range(self) -> float:
        """Gate attenuation in dB."""
        return self._range
    
    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs."""
        return [self._source]
    
    def is_pure(self) -> bool:
        """GatePE is NOT pure due to envelope state."""
        return False
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """Return extent from source."""
        return self._source.extent()
    
    def _on_start(self) -> None:
        """Start internal PEs."""
        self._envelope_pe.on_start()

    def _on_stop(self) -> None:
        """Stop internal PEs."""
        self._envelope_pe.on_stop()
    
    def _reset_state(self) -> None:
        """Reset internal envelope state."""
        self._envelope_pe.reset_state()
    
    def _render(self, start: int, duration: int) -> Snippet:
        """Render gated audio."""
        return self._dynamics_pe.render(start, duration)
    
    def __repr__(self) -> str:
        return (
            f"GatePE(threshold={self._threshold}, attack={self._attack}, "
            f"release={self._release}, range={self._range})"
        )
