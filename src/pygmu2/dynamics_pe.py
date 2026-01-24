"""
DynamicsPE - Flexible dynamics processor (compressor/limiter/expander/gate).

Applies gain reduction/expansion based on an external envelope signal,
enabling sidechain compression and maximum flexibility.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from enum import Enum
from typing import Optional, Union
import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.conversions import db_to_ratio, ratio_to_db


class DynamicsMode(Enum):
    """Dynamics processing modes."""
    COMPRESS = "compress"  # Reduce gain above threshold
    EXPAND = "expand"      # Reduce gain below threshold
    LIMIT = "limit"        # Hard ceiling (infinite ratio compression)
    GATE = "gate"          # Hard cutoff below threshold


class DynamicsPE(ProcessingElement):
    """
    Flexible dynamics processor that applies compression, limiting,
    expansion, or gating based on an external envelope signal.
    
    The envelope input (typically from EnvelopePE) controls when and
    how much gain reduction is applied. This separation enables:
    - Sidechain compression (envelope from a different source)
    - Custom envelope shapes
    - Shared envelope across multiple processors
    
    All level parameters (threshold, makeup_gain, knee) are in dB.
    The envelope input should be linear amplitude (0 to ~1).
    
    Args:
        source: Audio input to process
        envelope: Control signal (linear amplitude, e.g., from EnvelopePE)
        threshold: Level in dB where processing begins (default: -20.0)
        ratio: Compression/expansion ratio (default: 4.0)
                - 1.0 = no compression
                - 4.0 = 4:1 compression (4dB input change -> 1dB output)
                - float('inf') = limiting (hard ceiling)
                - 0.5 = 2:1 expansion (below threshold)
        knee: Soft knee width in dB (default: 0.0 = hard knee)
               Provides gradual transition around threshold.
        makeup_gain: Output gain in dB, or "auto" to compute automatically
                     (default: "auto")
        mode: Processing mode (default: COMPRESS)
        stereo_link: If True, use max envelope across channels (default: True)
        range: For GATE mode, attenuation in dB when gated (default: -80.0)
    
    Example:
        # Basic compression with EnvelopePE
        env = EnvelopePE(audio, attack=0.01, release=0.1)
        compressed = DynamicsPE(audio, env, threshold=-20, ratio=4)
        
        # Sidechain ducking (bass ducks when kick hits)
        kick_env = EnvelopePE(kick, attack=0.001, release=0.05)
        ducked_bass = DynamicsPE(bass, kick_env, threshold=-30, ratio=8)
        
        # Lookahead limiting
        env = EnvelopePE(audio, attack=0.005, release=0.05, lookahead=0.005)
        limited = DynamicsPE(audio, env, threshold=-1, mode=DynamicsMode.LIMIT)
        
        # Noise gate
        env = EnvelopePE(audio, attack=0.001, release=0.05)
        gated = DynamicsPE(audio, env, threshold=-40, mode=DynamicsMode.GATE)
    """
    
    # Sentinel value for auto makeup gain
    AUTO = "auto"
    
    def __init__(
        self,
        source: ProcessingElement,
        envelope: ProcessingElement,
        threshold: float = -20.0,
        ratio: float = 4.0,
        knee: float = 0.0,
        makeup_gain: Union[float, str] = "auto",
        mode: DynamicsMode = DynamicsMode.COMPRESS,
        stereo_link: bool = True,
        range: float = -80.0,
    ):
        self._source = source
        self._envelope = envelope
        self._threshold = threshold
        self._ratio = max(0.001, ratio)  # Avoid division by zero
        self._knee = max(0.0, knee)
        self._makeup_gain = makeup_gain
        self._mode = mode
        self._stereo_link = stereo_link
        self._range = range  # For gate mode
        
        # Compute auto makeup gain if requested
        if makeup_gain == self.AUTO:
            self._makeup_gain_db = self._compute_auto_makeup()
        else:
            self._makeup_gain_db = float(makeup_gain)
        
        self._makeup_gain_linear = db_to_ratio(self._makeup_gain_db)
    
    def _compute_auto_makeup(self) -> float:
        """
        Compute automatic makeup gain to compensate for compression.
        
        For a compressor, estimates the gain reduction at a typical
        "loud" level and compensates for it.
        """
        if self._mode in (DynamicsMode.EXPAND, DynamicsMode.GATE):
            return 0.0  # No makeup for expanders/gates
        
        # Estimate gain reduction at threshold + 12dB (typical loud signal)
        test_level = self._threshold + 12.0
        gain_db = self._compute_gain_db(test_level)
        
        # Makeup is negative of gain reduction (to compensate)
        # Scale by 0.7 to be somewhat conservative (avoid over-compensation)
        return -gain_db * 0.7
    
    @property
    def threshold(self) -> float:
        """Threshold in dB."""
        return self._threshold
    
    @property
    def ratio(self) -> float:
        """Compression/expansion ratio."""
        return self._ratio
    
    @property
    def knee(self) -> float:
        """Soft knee width in dB."""
        return self._knee
    
    @property
    def makeup_gain(self) -> float:
        """Makeup gain in dB."""
        return self._makeup_gain_db
    
    @property
    def mode(self) -> DynamicsMode:
        """Dynamics processing mode."""
        return self._mode
    
    @property
    def stereo_link(self) -> bool:
        """Whether stereo channels are linked."""
        return self._stereo_link
    
    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs."""
        return [self._source, self._envelope]
    
    def is_pure(self) -> bool:
        """
        DynamicsPE is pure - output depends only on current inputs.
        (State is maintained by the envelope PE, not here.)
        """
        return True
    
    def channel_count(self) -> Optional[int]:
        """Pass through channel count from source."""
        return self._source.channel_count()
    
    def _compute_extent(self) -> Extent:
        """Return intersection of source and envelope extents."""
        source_extent = self._source.extent()
        env_extent = self._envelope.extent()
        intersection = source_extent.intersection(env_extent)
        return intersection if intersection is not None else Extent(0, 0)
    
    def _compute_gain_db(self, level_db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute gain adjustment in dB for given input level(s).
        
        Args:
            level_db: Input level in dB (scalar or array)
        
        Returns:
            Gain adjustment in dB (negative = reduction, positive = boost)
        """
        threshold = self._threshold
        ratio = self._ratio
        knee = self._knee
        
        if self._mode == DynamicsMode.LIMIT:
            # Infinite ratio above threshold
            ratio = float('inf')
        
        if self._mode in (DynamicsMode.COMPRESS, DynamicsMode.LIMIT):
            # Compression: reduce gain above threshold
            return self._compute_compression_gain(level_db, threshold, ratio, knee)
        
        elif self._mode == DynamicsMode.EXPAND:
            # Expansion: reduce gain below threshold
            return self._compute_expansion_gain(level_db, threshold, ratio, knee)
        
        elif self._mode == DynamicsMode.GATE:
            # Gate: hard cutoff below threshold
            return self._compute_gate_gain(level_db, threshold, knee)
        
        return 0.0
    
    def _compute_compression_gain(
        self,
        level_db: Union[float, np.ndarray],
        threshold: float,
        ratio: float,
        knee: float,
    ) -> Union[float, np.ndarray]:
        """Compute compression gain reduction."""
        level_db = np.asarray(level_db)
        gain_db = np.zeros_like(level_db)
        
        if knee <= 0:
            # Hard knee
            above = level_db > threshold
            overshoot = level_db - threshold
            if np.isinf(ratio):
                # Limiter: clamp to threshold
                gain_db = np.where(above, -overshoot, 0.0)
            else:
                # Compressor: reduce by (1 - 1/ratio) of overshoot
                gain_db = np.where(above, overshoot * (1.0 / ratio - 1.0), 0.0)
        else:
            # Soft knee
            half_knee = knee / 2.0
            
            # Below knee: no compression
            below_knee = level_db < (threshold - half_knee)
            
            # Above knee: full compression
            above_knee = level_db > (threshold + half_knee)
            overshoot = level_db - threshold
            
            if np.isinf(ratio):
                full_gain = -overshoot
            else:
                full_gain = overshoot * (1.0 / ratio - 1.0)
            
            # In knee: quadratic transition
            in_knee = ~below_knee & ~above_knee
            x = level_db - threshold + half_knee  # 0 to knee
            if np.isinf(ratio):
                knee_gain = -(x ** 2) / (2 * knee)
            else:
                knee_gain = (1.0 / ratio - 1.0) * (x ** 2) / (2 * knee)
            
            gain_db = np.where(below_knee, 0.0,
                      np.where(above_knee, full_gain, knee_gain))
        
        return gain_db
    
    def _compute_expansion_gain(
        self,
        level_db: Union[float, np.ndarray],
        threshold: float,
        ratio: float,
        knee: float,
    ) -> Union[float, np.ndarray]:
        """Compute expansion gain reduction (below threshold)."""
        level_db = np.asarray(level_db)
        gain_db = np.zeros_like(level_db)
        
        if knee <= 0:
            # Hard knee
            below = level_db < threshold
            undershoot = threshold - level_db
            # Reduce gain by (ratio - 1) * undershoot
            gain_db = np.where(below, -undershoot * (ratio - 1.0), 0.0)
        else:
            # Soft knee
            half_knee = knee / 2.0
            
            above_knee = level_db > (threshold + half_knee)
            below_knee = level_db < (threshold - half_knee)
            
            undershoot = threshold - level_db
            full_gain = -undershoot * (ratio - 1.0)
            
            in_knee = ~below_knee & ~above_knee
            x = threshold + half_knee - level_db  # 0 to knee
            knee_gain = -(ratio - 1.0) * (x ** 2) / (2 * knee)
            
            gain_db = np.where(above_knee, 0.0,
                      np.where(below_knee, full_gain, knee_gain))
        
        return gain_db
    
    def _compute_gate_gain(
        self,
        level_db: Union[float, np.ndarray],
        threshold: float,
        knee: float,
    ) -> Union[float, np.ndarray]:
        """Compute gate gain (hard cutoff below threshold)."""
        level_db = np.asarray(level_db)
        range_db = self._range  # How much to attenuate when gated
        
        if knee <= 0:
            # Hard knee gate
            below = level_db < threshold
            gain_db = np.where(below, range_db, 0.0)
        else:
            # Soft knee gate
            half_knee = knee / 2.0
            
            above_knee = level_db > (threshold + half_knee)
            below_knee = level_db < (threshold - half_knee)
            
            # In knee: linear transition from 0 to range_db
            in_knee = ~below_knee & ~above_knee
            t = (threshold + half_knee - level_db) / knee  # 0 to 1
            knee_gain = t * range_db
            
            gain_db = np.where(above_knee, 0.0,
                      np.where(below_knee, range_db, knee_gain))
        
        return gain_db
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render dynamics-processed audio.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
        Returns:
            Snippet containing processed audio
        """
        # Get source audio and envelope
        source_snippet = self._source.render(start, duration)
        env_snippet = self._envelope.render(start, duration)
        
        audio = source_snippet.data.astype(np.float64)
        envelope = env_snippet.data.astype(np.float64)
        
        channels = audio.shape[1]
        env_channels = envelope.shape[1]
        
        # Handle stereo linking
        if self._stereo_link and env_channels > 1:
            # Use max across channels for linked stereo
            envelope = np.max(envelope, axis=1, keepdims=True)
            envelope = np.tile(envelope, (1, channels))
        elif env_channels == 1 and channels > 1:
            # Mono envelope, stereo audio: broadcast
            envelope = np.tile(envelope, (1, channels))
        elif env_channels != channels:
            # Mismatch: use first envelope channel
            envelope = np.tile(envelope[:, 0:1], (1, channels))
        
        # Convert envelope to dB (with floor to avoid -inf)
        eps = 1e-10
        level_db = 20.0 * np.log10(np.maximum(envelope, eps))
        
        # Compute gain in dB
        gain_db = self._compute_gain_db(level_db)
        
        # Add makeup gain
        gain_db = gain_db + self._makeup_gain_db
        
        # Convert to linear gain
        gain_linear = 10.0 ** (gain_db / 20.0)
        
        # Apply gain
        output = audio * gain_linear
        
        return Snippet(start, output.astype(np.float32))
    
    def __repr__(self) -> str:
        makeup_str = "auto" if self._makeup_gain == self.AUTO else f"{self._makeup_gain_db:.1f}"
        return (
            f"DynamicsPE(threshold={self._threshold}, ratio={self._ratio}, "
            f"knee={self._knee}, makeup={makeup_str}, mode={self._mode.value}, "
            f"stereo_link={self._stereo_link})"
        )
