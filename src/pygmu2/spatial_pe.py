"""
SpatialPE - spatial audio processing and channel conversion.

SpatialPE takes an M-channel input and produces an N-channel output, optionally
applying spatialization (panning/positioning) using various techniques.

Copyright (c) 2026 R. Dunbar Poor and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class SpatialMethod(ABC):
    """
    Base class for spatialization methods.
    
    Each subclass carries the parameters specific to that spatialization technique
    and determines the output channel count.
    """
    
    @property
    @abstractmethod
    def output_channels(self) -> int:
        """Return the number of output channels this method produces."""
        pass


class SpatialAdapter(SpatialMethod):
    """
    Channel adapter method - performs M→N channel conversion without spatialization.
    
    Acts as a universal channel adapter for upmix/downmix conversion.
    Common conversions:
    - Mono (M=1) -> Stereo (N=2): Duplicate mono channel to both L and R
    - Stereo (M=2) -> Mono (N=1): Mix L and R channels (average)
    - Mono (M=1) -> Quad (N=4): Duplicate to all channels
    - Stereo (M=2) -> Quad (N=4): L->L, R->R, center/surround from mix
    - Quad (M=4) -> Stereo (N=2): L->L, R->R, ignore center/surround
    - Higher channel counts: Similar patterns apply
    
    Args:
        channels: Output channel count (N >= 1)
    """
    
    def __init__(self, channels: int):
        if channels < 1:
            raise ValueError(f"SpatialAdapter: channels must be >= 1 (got {channels})")
        self._channels = int(channels)
    
    @property
    def output_channels(self) -> int:
        return self._channels
    
    def __repr__(self) -> str:
        return f"SpatialAdapter(channels={self._channels})"


class SpatialLinear(SpatialMethod):
    """
    Linear panning method.
    
    Simple linear panning using L/R gain. May cause center dip when panned to center.
    Only uses azimuth (elevation is ignored).
    Output: 2 channels (stereo)
    
    Args:
        azimuth: Horizontal angle in degrees
                 * 0° = straight ahead (front)
                 * +90° = right
                 * -90° = left
                 * ±180° = behind
                 Can be float or ProcessingElement for dynamic panning
    """
    
    def __init__(self, azimuth: Union[float, ProcessingElement]):
        self.azimuth = azimuth
    
    @property
    def output_channels(self) -> int:
        return 2  # Stereo
    
    def __repr__(self) -> str:
        azimuth_str = f"{self.azimuth:.1f}" if isinstance(self.azimuth, (int, float)) else self.azimuth.__class__.__name__
        return f"SpatialLinear(azimuth={azimuth_str})"


class SpatialConstantPower(SpatialMethod):
    """
    Constant-power panning method.
    
    Constant-power panning using sin/cos curves. Provides better stereo balance
    than linear panning, avoiding center dip. Only uses azimuth (elevation is ignored).
    Output: 2 channels (stereo)
    
    Args:
        azimuth: Horizontal angle in degrees
                 * 0° = straight ahead (front)
                 * +90° = right
                 * -90° = left
                 * ±180° = behind
                 Can be float or ProcessingElement for dynamic panning
    """
    
    def __init__(self, azimuth: Union[float, ProcessingElement]):
        self.azimuth = azimuth
    
    @property
    def output_channels(self) -> int:
        return 2  # Stereo
    
    def __repr__(self) -> str:
        azimuth_str = f"{self.azimuth:.1f}" if isinstance(self.azimuth, (int, float)) else self.azimuth.__class__.__name__
        return f"SpatialConstantPower(azimuth={azimuth_str})"


class SpatialHRTF(SpatialMethod):
    """
    HRTF (Head-Related Transfer Function) spatialization method.
    
    Uses HRTF data (KEMAR dataset) for realistic binaural spatialization.
    Supports both azimuth and elevation for full 3D positioning.
    Output: 2 channels (stereo)
    
    Args:
        azimuth: Horizontal angle in degrees
                 * 0° = straight ahead (front)
                 * +90° = right
                 * -90° = left
                 * ±180° = behind
                 Can be float or ProcessingElement for dynamic panning
        elevation: Vertical angle in degrees (default: 0.0)
                   * 0° = horizontal plane
                   * +90° = directly above
                   * -90° = directly below
                   Can be float or ProcessingElement for dynamic positioning
    """
    
    def __init__(
        self,
        azimuth: Union[float, ProcessingElement],
        elevation: Union[float, ProcessingElement] = 0.0,
    ):
        self.azimuth = azimuth
        self.elevation = elevation
    
    @property
    def output_channels(self) -> int:
        return 2  # Stereo
    
    def __repr__(self) -> str:
        azimuth_str = f"{self.azimuth:.1f}" if isinstance(self.azimuth, (int, float)) else self.azimuth.__class__.__name__
        elevation_str = f"{self.elevation:.1f}" if isinstance(self.elevation, (int, float)) else self.elevation.__class__.__name__
        return f"SpatialHRTF(azimuth={azimuth_str}, elevation={elevation_str})"


class SpatialPE(ProcessingElement):
    """
    Spatial audio processing and channel conversion PE.
    
    Takes an M-channel input and produces an N-channel output. Supports two modes:
    
    1. **Channel Conversion Only** (SpatialAdapter method):
       Use SpatialAdapter(channels=N) to perform upmix/downmix conversion
       from M channels to N channels. Acts as a universal channel adapter.
       
       Common conversions:
       - Mono (M=1) -> Stereo (N=2): Duplicate mono channel to both L and R
       - Stereo (M=2) -> Mono (N=1): Mix L and R channels (average)
       - Mono (M=1) -> Quad (N=4): Duplicate to all channels
       - Stereo (M=2) -> Quad (N=4): L->L, R->R, center/surround from mix
       - Quad (M=4) -> Stereo (N=2): L->L, R->R, ignore center/surround
       - Higher channel counts: Similar patterns apply
    
    2. **Spatialization** (SpatialLinear, SpatialConstantPower, SpatialHRTF):
       When a spatialization method is provided, the M-channel input is first
       mixed down to mono (or treated as a single source), then placed spatially
       in the output space using the specified technique. The method determines
       the output channel count (typically 2 for stereo panning/HRTF).
       
       - Azimuth: Horizontal angle in degrees (specified in method object)
         * 0° = straight ahead (front)
         * +90° = right
         * -90° = left
         * ±180° = behind
       
       - Elevation: Vertical angle in degrees (specified in SpatialHRTF method)
         * 0° = horizontal plane (default)
         * +90° = directly above
         * -90° = directly below
       
       - For SpatialLinear and SpatialConstantPower: Only azimuth is used (2D panning)
       - For SpatialHRTF: Both azimuth and elevation are used (3D positioning)
    
    Channel Count:
    - Input: M channels (M >= 1)
    - Output: N channels (N >= 1)
    - Output channel count is determined by the SpatialMethod's output_channels property
    
    Args:
        source: Input ProcessingElement (M channels)
        method: Spatialization method (required)
                Must be an instance of SpatialMethod subclass:
                - SpatialAdapter(channels): Channel conversion only (M→N)
                - SpatialLinear(azimuth): Linear panning (output: 2 channels)
                - SpatialConstantPower(azimuth): Constant-power panning (output: 2 channels)
                - SpatialHRTF(azimuth, elevation): HRTF spatialization (output: 2 channels)
    
    Examples:
        # Channel conversion: Mono to Stereo
        mono_source = SinePE(frequency=440.0)
        stereo_output = SpatialPE(mono_source, method=SpatialAdapter(channels=2))
        
        # Channel conversion: Stereo to Mono
        stereo_source = ...
        mono_output = SpatialPE(stereo_source, method=SpatialAdapter(channels=1))
        
        # Spatialization: Mono source panned right using constant-power
        mono_source = SinePE(frequency=440.0)
        panned = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=45.0))
        
        # Spatialization: Stereo source placed at azimuth -30° (left) using linear
        stereo_source = ...
        panned = SpatialPE(stereo_source, method=SpatialLinear(azimuth=-30.0))
        
        # HRTF spatialization: Mono source at azimuth 45°, elevation 15°
        mono_source = SinePE(frequency=440.0)
        spatialized = SpatialPE(
            mono_source,
            method=SpatialHRTF(azimuth=45.0, elevation=15.0)
        )
        
        # Dynamic panning: Azimuth controlled by a PE
        pan_control = RampPE(-90.0, 90.0, duration=44100)  # Sweep left to right
        panned = SpatialPE(mono_source, method=SpatialConstantPower(azimuth=pan_control))
    """
    
    def __init__(
        self,
        source: ProcessingElement,
        *,
        method: SpatialMethod,
    ):
        if method is None:
            raise ValueError("SpatialPE: method is required")
        
        self._source = source
        self._method = method
        
        # TODO: Implementation - validate method
    
    def inputs(self) -> list[ProcessingElement]:
        """
        Return input PEs (source and any dynamic azimuth/elevation PEs from method).
        
        Includes the source and any ProcessingElement parameters from the method
        (e.g., dynamic azimuth/elevation).
        """
        inputs_list = [self._source]
        
        # Add any ProcessingElement parameters from the method
        if isinstance(self._method, SpatialAdapter):
            # No dynamic parameters
            pass
        elif isinstance(self._method, SpatialLinear):
            if isinstance(self._method.azimuth, ProcessingElement):
                inputs_list.append(self._method.azimuth)
        elif isinstance(self._method, SpatialConstantPower):
            if isinstance(self._method.azimuth, ProcessingElement):
                inputs_list.append(self._method.azimuth)
        elif isinstance(self._method, SpatialHRTF):
            if isinstance(self._method.azimuth, ProcessingElement):
                inputs_list.append(self._method.azimuth)
            if isinstance(self._method.elevation, ProcessingElement):
                inputs_list.append(self._method.elevation)
        
        return inputs_list
    
    def is_pure(self) -> bool:
        """SpatialPE is pure - it's a stateless transformation."""
        # TODO: Implementation
        return True
    
    def channel_count(self) -> Optional[int]:
        """Return output channel count (determined by method)."""
        return self._method.output_channels
    
    def _compute_extent(self) -> Extent:
        """Return extent from source (spatialization doesn't change extent)."""
        return self._source.extent()
    
    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render spatialized audio.
        
        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)
        
        Returns:
            Snippet with N-channel output (N determined by method.output_channels)
        """
        # Render source
        source_snippet = self._source.render(start, duration)
        source_data = source_snippet.data  # shape (duration, M)
        src_ch = source_snippet.channels
        out_ch = self._method.output_channels
        
        # Handle different methods
        if isinstance(self._method, SpatialAdapter):
            output_data = self._render_adapter(source_data, src_ch, out_ch)
        elif isinstance(self._method, SpatialLinear):
            output_data = self._render_linear(source_data, src_ch, start, duration)
        elif isinstance(self._method, SpatialConstantPower):
            output_data = self._render_constant_power(source_data, src_ch, start, duration)
        elif isinstance(self._method, SpatialHRTF):
            # TODO: HRTF implementation
            output_data = np.zeros((duration, out_ch), dtype=np.float32)
        else:
            raise ValueError(f"SpatialPE: Unknown method type {type(self._method)}")
        
        return Snippet(start, output_data)
    
    def _render_adapter(self, source_data: np.ndarray, src_ch: int, out_ch: int) -> np.ndarray:
        """
        Render channel adapter conversion (M→N).
        
        Args:
            source_data: Input data shape (duration, M)
            src_ch: Source channel count (M)
            out_ch: Output channel count (N)
        
        Returns:
            Output data shape (duration, N)
        """
        duration = source_data.shape[0]
        
        if src_ch == out_ch:
            # Passthrough
            return source_data.copy()
        
        output_data = np.zeros((duration, out_ch), dtype=np.float32)
        
        if src_ch == 1:
            # Mono → N: Duplicate mono channel to all output channels
            output_data[:, :] = source_data[:, 0:1]
        elif out_ch == 1:
            # M → Mono: Average all input channels
            output_data[:, 0] = np.mean(source_data, axis=1)
        elif src_ch == 2 and out_ch == 4:
            # Stereo → Quad: L→L, R→R, center/surround from mix
            output_data[:, 0] = source_data[:, 0]  # L
            output_data[:, 1] = source_data[:, 1]  # R
            center_surround = np.mean(source_data, axis=1)  # Average of L and R
            output_data[:, 2] = center_surround  # Center
            output_data[:, 3] = center_surround  # Surround
        elif src_ch == 4 and out_ch == 2:
            # Quad → Stereo: L→L, R→R, ignore center/surround
            output_data[:, 0] = source_data[:, 0]  # L
            output_data[:, 1] = source_data[:, 1]  # R
        else:
            # Generic conversion: map channels up to min(M, N), then duplicate or average
            min_ch = min(src_ch, out_ch)
            output_data[:, :min_ch] = source_data[:, :min_ch]
            
            if out_ch > src_ch:
                # Upmix: duplicate last channel to remaining outputs
                if src_ch > 0:
                    last_ch = source_data[:, src_ch - 1:src_ch]
                    output_data[:, src_ch:] = last_ch
            elif src_ch > out_ch:
                # Downmix: average remaining channels into last output
                if out_ch > 0:
                    remaining = source_data[:, out_ch:]
                    if remaining.shape[1] > 0:
                        output_data[:, out_ch - 1] += np.mean(remaining, axis=1)
        
        return output_data
    
    def _render_linear(self, source_data: np.ndarray, src_ch: int, start: int, duration: int) -> np.ndarray:
        """
        Render linear panning.
        
        Args:
            source_data: Input data shape (duration, M)
            src_ch: Source channel count (M)
            start: Starting sample index
            duration: Number of samples
        
        Returns:
            Output data shape (duration, 2) - stereo
        """
        # Mix M channels to mono
        mono_data = np.mean(source_data, axis=1, keepdims=True)  # shape (duration, 1)
        
        # Get azimuth (static or dynamic)
        if isinstance(self._method.azimuth, ProcessingElement):
            azimuth_snippet = self._method.azimuth.render(start, duration)
            azimuth_values = azimuth_snippet.data[:, 0]  # shape (duration,)
        else:
            azimuth_values = np.full(duration, float(self._method.azimuth), dtype=np.float32)
        
        # Clamp azimuth to [-90, +90]
        azimuth_values = np.clip(azimuth_values, -90.0, 90.0)
        
        # Convert azimuth [-90, +90] to pan value [0, 1]
        # -90° → pan=0 (all left), 0° → pan=0.5 (center), +90° → pan=1 (all right)
        pan_values = (azimuth_values + 90.0) / 180.0  # shape (duration,)
        
        # Linear panning: L = 1 - pan, R = pan
        L_gain = 1.0 - pan_values  # shape (duration,)
        R_gain = pan_values  # shape (duration,)
        
        # Apply gains to mono signal
        output_data = np.zeros((duration, 2), dtype=np.float32)
        output_data[:, 0] = mono_data[:, 0] * L_gain  # Left
        output_data[:, 1] = mono_data[:, 0] * R_gain  # Right
        
        return output_data
    
    def _render_constant_power(self, source_data: np.ndarray, src_ch: int, start: int, duration: int) -> np.ndarray:
        """
        Render constant-power panning.
        
        Args:
            source_data: Input data shape (duration, M)
            src_ch: Source channel count (M)
            start: Starting sample index
            duration: Number of samples
        
        Returns:
            Output data shape (duration, 2) - stereo
        """
        # Mix M channels to mono
        mono_data = np.mean(source_data, axis=1, keepdims=True)  # shape (duration, 1)
        
        # Get azimuth (static or dynamic)
        if isinstance(self._method.azimuth, ProcessingElement):
            azimuth_snippet = self._method.azimuth.render(start, duration)
            azimuth_values = azimuth_snippet.data[:, 0]  # shape (duration,)
        else:
            azimuth_values = np.full(duration, float(self._method.azimuth), dtype=np.float32)
        
        # Clamp azimuth to [-90, +90]
        azimuth_values = np.clip(azimuth_values, -90.0, 90.0)
        
        # Convert azimuth [-90, +90] to pan angle [0, 90] degrees
        # -90° → 0°, 0° → 45°, +90° → 90°
        pan_angle_deg = (azimuth_values + 90.0) / 2.0  # shape (duration,)
        pan_angle_rad = np.deg2rad(pan_angle_deg)  # shape (duration,)
        
        # Constant-power panning: L = cos(angle), R = sin(angle)
        L_gain = np.cos(pan_angle_rad)  # shape (duration,)
        R_gain = np.sin(pan_angle_rad)  # shape (duration,)
        
        # Apply gains to mono signal
        output_data = np.zeros((duration, 2), dtype=np.float32)
        output_data[:, 0] = mono_data[:, 0] * L_gain  # Left
        output_data[:, 1] = mono_data[:, 0] * R_gain  # Right
        
        return output_data
    
    def __repr__(self) -> str:
        return f"SpatialPE(source={self._source.__class__.__name__}, method={self._method})"
