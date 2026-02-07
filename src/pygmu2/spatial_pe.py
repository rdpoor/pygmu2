"""
SpatialPE - spatial audio processing and channel conversion.

SpatialPE takes an M-channel input and produces an N-channel output, optionally
applying spatialization (panning/positioning) using various techniques.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

HRTF files:
- Gardner, W. G., and Martin, K. D., 
  "HRTF measurements of a KEMAR dummy-head microphone", 
  MIT Media Lab Perceptual Computing Technical Report #280, 1994.

MIT License
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from pygmu2.assets import get_kemar_dir
from pygmu2.config import handle_error
from pygmu2.logger import get_logger
from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet

logger = get_logger(__name__)

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

    @abstractmethod
    def render(
        self,
        source_snippet: Snippet,
        start: int,
        duration: int,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Render spatialization for the given source snippet.
        """
        raise NotImplementedError

    def inputs(self) -> list[ProcessingElement]:
        """
        Return any dynamic ProcessingElement inputs used by this method.
        """
        return []


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

    def render(
        self,
        source_snippet: Snippet,
        start: int,
        duration: int,
        sample_rate: int,
    ) -> np.ndarray:
        source_data = source_snippet.data  # shape (duration, M)
        src_ch = source_snippet.channels
        out_ch = self.output_channels

        if src_ch == out_ch:
            return source_data

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

    def inputs(self) -> list[ProcessingElement]:
        if isinstance(self.azimuth, ProcessingElement):
            return [self.azimuth]
        return []

    def render(
        self,
        source_snippet: Snippet,
        start: int,
        duration: int,
        sample_rate: int,
    ) -> np.ndarray:
        source_data = source_snippet.data

        # Mix M channels to mono
        mono_data = np.mean(source_data, axis=1, keepdims=True)  # shape (duration, 1)

        # Get azimuth (static or dynamic)
        if isinstance(self.azimuth, ProcessingElement):
            azimuth_snippet = self.azimuth.render(start, duration)
            azimuth_values = azimuth_snippet.data[:, 0]  # shape (duration,)
        else:
            azimuth_values = np.full(duration, float(self.azimuth), dtype=np.float32)

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

    def inputs(self) -> list[ProcessingElement]:
        if isinstance(self.azimuth, ProcessingElement):
            return [self.azimuth]
        return []

    def render(
        self,
        source_snippet: Snippet,
        start: int,
        duration: int,
        sample_rate: int,
    ) -> np.ndarray:
        source_data = source_snippet.data

        # Mix M channels to mono
        mono_data = np.mean(source_data, axis=1, keepdims=True)  # shape (duration, 1)

        # Get azimuth (static or dynamic)
        if isinstance(self.azimuth, ProcessingElement):
            azimuth_snippet = self.azimuth.render(start, duration)
            azimuth_values = azimuth_snippet.data[:, 0]  # shape (duration,)
        else:
            azimuth_values = np.full(duration, float(self.azimuth), dtype=np.float32)

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
        azimuth_str = f"{self.azimuth:.1f}" if isinstance(self.azimuth, (int, float)) else self.azimuth.__class__.__name__
        return f"SpatialConstantPower(azimuth={azimuth_str})"


class SpatialHRTF(SpatialMethod):
    """
    HRTF (Head-Related Transfer Function) spatialization method.
    
    Uses HRTF data (KEMAR dataset) for realistic binaural spatialization.
    Supports both azimuth and elevation for full 3D positioning.
    Output: 2 channels (stereo)
    
    Azimuth and elevation must be static (float or int). Dynamic values are not
    supported because switching impulse responses during rendering would cause
    audible discontinuities.
    
    Args:
        azimuth: Horizontal angle in degrees (static only)
                 * 0° = straight ahead (front)
                 * +90° = right
                 * -90° = left
                 * ±180° = behind
        elevation: Vertical angle in degrees (default: 0.0, static only)
                   * 0° = horizontal plane
                   * +90° = directly above
                   * -90° = directly below
    """

    # MIT KEMAR compact set: (elevation_deg, azimuth_deg, filename). 0° azimuth = front; dataset 0–180° only.
    KEMAR_HRTF_ENTRIES = (
    (-40, 0, "H-40e000a.wav"), (-40, 6, "H-40e006a.wav"), (-40, 13, "H-40e013a.wav"), (-40, 19, "H-40e019a.wav"), (-40, 26, "H-40e026a.wav"),
    (-40, 32, "H-40e032a.wav"), (-40, 39, "H-40e039a.wav"), (-40, 45, "H-40e045a.wav"), (-40, 51, "H-40e051a.wav"), (-40, 58, "H-40e058a.wav"),
    (-40, 64, "H-40e064a.wav"), (-40, 71, "H-40e071a.wav"), (-40, 77, "H-40e077a.wav"), (-40, 84, "H-40e084a.wav"), (-40, 90, "H-40e090a.wav"),
    (-40, 96, "H-40e096a.wav"), (-40, 103, "H-40e103a.wav"), (-40, 109, "H-40e109a.wav"), (-40, 116, "H-40e116a.wav"), (-40, 122, "H-40e122a.wav"),
    (-40, 129, "H-40e129a.wav"), (-40, 135, "H-40e135a.wav"), (-40, 141, "H-40e141a.wav"), (-40, 148, "H-40e148a.wav"), (-40, 154, "H-40e154a.wav"),
    (-40, 161, "H-40e161a.wav"), (-40, 167, "H-40e167a.wav"), (-40, 174, "H-40e174a.wav"), (-40, 180, "H-40e180a.wav"), (-30, 0, "H-30e000a.wav"),
    (-30, 6, "H-30e006a.wav"), (-30, 12, "H-30e012a.wav"), (-30, 18, "H-30e018a.wav"), (-30, 24, "H-30e024a.wav"), (-30, 30, "H-30e030a.wav"),
    (-30, 36, "H-30e036a.wav"), (-30, 42, "H-30e042a.wav"), (-30, 48, "H-30e048a.wav"), (-30, 54, "H-30e054a.wav"), (-30, 60, "H-30e060a.wav"),
    (-30, 66, "H-30e066a.wav"), (-30, 72, "H-30e072a.wav"), (-30, 78, "H-30e078a.wav"), (-30, 84, "H-30e084a.wav"), (-30, 90, "H-30e090a.wav"),
    (-30, 96, "H-30e096a.wav"), (-30, 102, "H-30e102a.wav"), (-30, 108, "H-30e108a.wav"), (-30, 114, "H-30e114a.wav"), (-30, 120, "H-30e120a.wav"),
    (-30, 126, "H-30e126a.wav"), (-30, 132, "H-30e132a.wav"), (-30, 138, "H-30e138a.wav"), (-30, 144, "H-30e144a.wav"), (-30, 150, "H-30e150a.wav"),
    (-30, 156, "H-30e156a.wav"), (-30, 162, "H-30e162a.wav"), (-30, 168, "H-30e168a.wav"), (-30, 174, "H-30e174a.wav"), (-30, 180, "H-30e180a.wav"),
    (-20, 0, "H-20e000a.wav"), (-20, 5, "H-20e005a.wav"), (-20, 10, "H-20e010a.wav"), (-20, 15, "H-20e015a.wav"), (-20, 20, "H-20e020a.wav"),
    (-20, 25, "H-20e025a.wav"), (-20, 30, "H-20e030a.wav"), (-20, 35, "H-20e035a.wav"), (-20, 40, "H-20e040a.wav"), (-20, 45, "H-20e045a.wav"),
    (-20, 50, "H-20e050a.wav"), (-20, 55, "H-20e055a.wav"), (-20, 60, "H-20e060a.wav"), (-20, 65, "H-20e065a.wav"), (-20, 70, "H-20e070a.wav"),
    (-20, 75, "H-20e075a.wav"), (-20, 80, "H-20e080a.wav"), (-20, 85, "H-20e085a.wav"), (-20, 90, "H-20e090a.wav"), (-20, 95, "H-20e095a.wav"),
    (-20, 100, "H-20e100a.wav"), (-20, 105, "H-20e105a.wav"), (-20, 110, "H-20e110a.wav"), (-20, 115, "H-20e115a.wav"), (-20, 120, "H-20e120a.wav"),
    (-20, 125, "H-20e125a.wav"), (-20, 130, "H-20e130a.wav"), (-20, 135, "H-20e135a.wav"), (-20, 140, "H-20e140a.wav"), (-20, 145, "H-20e145a.wav"),
    (-20, 150, "H-20e150a.wav"), (-20, 155, "H-20e155a.wav"), (-20, 160, "H-20e160a.wav"), (-20, 165, "H-20e165a.wav"), (-20, 170, "H-20e170a.wav"),
    (-20, 175, "H-20e175a.wav"), (-20, 180, "H-20e180a.wav"), (-10, 0, "H-10e000a.wav"), (-10, 5, "H-10e005a.wav"), (-10, 10, "H-10e010a.wav"),
    (-10, 15, "H-10e015a.wav"), (-10, 20, "H-10e020a.wav"), (-10, 25, "H-10e025a.wav"), (-10, 30, "H-10e030a.wav"), (-10, 35, "H-10e035a.wav"),
    (-10, 40, "H-10e040a.wav"), (-10, 45, "H-10e045a.wav"), (-10, 50, "H-10e050a.wav"), (-10, 55, "H-10e055a.wav"), (-10, 60, "H-10e060a.wav"),
    (-10, 65, "H-10e065a.wav"), (-10, 70, "H-10e070a.wav"), (-10, 75, "H-10e075a.wav"), (-10, 80, "H-10e080a.wav"), (-10, 85, "H-10e085a.wav"),
    (-10, 90, "H-10e090a.wav"), (-10, 95, "H-10e095a.wav"), (-10, 100, "H-10e100a.wav"), (-10, 105, "H-10e105a.wav"), (-10, 110, "H-10e110a.wav"),
    (-10, 115, "H-10e115a.wav"), (-10, 120, "H-10e120a.wav"), (-10, 125, "H-10e125a.wav"), (-10, 130, "H-10e130a.wav"), (-10, 135, "H-10e135a.wav"),
    (-10, 140, "H-10e140a.wav"), (-10, 145, "H-10e145a.wav"), (-10, 150, "H-10e150a.wav"), (-10, 155, "H-10e155a.wav"), (-10, 160, "H-10e160a.wav"),
    (-10, 165, "H-10e165a.wav"), (-10, 170, "H-10e170a.wav"), (-10, 175, "H-10e175a.wav"), (-10, 180, "H-10e180a.wav"), (0, 0, "H0e000a.wav"),
    (0, 5, "H0e005a.wav"), (0, 10, "H0e010a.wav"), (0, 15, "H0e015a.wav"), (0, 20, "H0e020a.wav"), (0, 25, "H0e025a.wav"),
    (0, 30, "H0e030a.wav"), (0, 35, "H0e035a.wav"), (0, 40, "H0e040a.wav"), (0, 45, "H0e045a.wav"), (0, 50, "H0e050a.wav"),
    (0, 55, "H0e055a.wav"), (0, 60, "H0e060a.wav"), (0, 65, "H0e065a.wav"), (0, 70, "H0e070a.wav"), (0, 75, "H0e075a.wav"),
    (0, 80, "H0e080a.wav"), (0, 85, "H0e085a.wav"), (0, 90, "H0e090a.wav"), (0, 95, "H0e095a.wav"), (0, 100, "H0e100a.wav"),
    (0, 105, "H0e105a.wav"), (0, 110, "H0e110a.wav"), (0, 115, "H0e115a.wav"), (0, 120, "H0e120a.wav"), (0, 125, "H0e125a.wav"),
    (0, 130, "H0e130a.wav"), (0, 135, "H0e135a.wav"), (0, 140, "H0e140a.wav"), (0, 145, "H0e145a.wav"), (0, 150, "H0e150a.wav"),
    (0, 155, "H0e155a.wav"), (0, 160, "H0e160a.wav"), (0, 165, "H0e165a.wav"), (0, 170, "H0e170a.wav"), (0, 175, "H0e175a.wav"),
    (0, 180, "H0e180a.wav"), (10, 0, "H10e000a.wav"), (10, 5, "H10e005a.wav"), (10, 10, "H10e010a.wav"), (10, 15, "H10e015a.wav"),
    (10, 20, "H10e020a.wav"), (10, 25, "H10e025a.wav"), (10, 30, "H10e030a.wav"), (10, 35, "H10e035a.wav"), (10, 40, "H10e040a.wav"),
    (10, 45, "H10e045a.wav"), (10, 50, "H10e050a.wav"), (10, 55, "H10e055a.wav"), (10, 60, "H10e060a.wav"), (10, 65, "H10e065a.wav"),
    (10, 70, "H10e070a.wav"), (10, 75, "H10e075a.wav"), (10, 80, "H10e080a.wav"), (10, 85, "H10e085a.wav"), (10, 90, "H10e090a.wav"),
    (10, 95, "H10e095a.wav"), (10, 100, "H10e100a.wav"), (10, 105, "H10e105a.wav"), (10, 110, "H10e110a.wav"), (10, 115, "H10e115a.wav"),
    (10, 120, "H10e120a.wav"), (10, 125, "H10e125a.wav"), (10, 130, "H10e130a.wav"), (10, 135, "H10e135a.wav"), (10, 140, "H10e140a.wav"),
    (10, 145, "H10e145a.wav"), (10, 150, "H10e150a.wav"), (10, 155, "H10e155a.wav"), (10, 160, "H10e160a.wav"), (10, 165, "H10e165a.wav"),
    (10, 170, "H10e170a.wav"), (10, 175, "H10e175a.wav"), (10, 180, "H10e180a.wav"), (20, 0, "H20e000a.wav"), (20, 5, "H20e005a.wav"),
    (20, 10, "H20e010a.wav"), (20, 15, "H20e015a.wav"), (20, 20, "H20e020a.wav"), (20, 25, "H20e025a.wav"), (20, 30, "H20e030a.wav"),
    (20, 35, "H20e035a.wav"), (20, 40, "H20e040a.wav"), (20, 45, "H20e045a.wav"), (20, 50, "H20e050a.wav"), (20, 55, "H20e055a.wav"),
    (20, 60, "H20e060a.wav"), (20, 65, "H20e065a.wav"), (20, 70, "H20e070a.wav"), (20, 75, "H20e075a.wav"), (20, 80, "H20e080a.wav"),
    (20, 85, "H20e085a.wav"), (20, 90, "H20e090a.wav"), (20, 95, "H20e095a.wav"), (20, 100, "H20e100a.wav"), (20, 105, "H20e105a.wav"),
    (20, 110, "H20e110a.wav"), (20, 115, "H20e115a.wav"), (20, 120, "H20e120a.wav"), (20, 125, "H20e125a.wav"), (20, 130, "H20e130a.wav"),
    (20, 135, "H20e135a.wav"), (20, 140, "H20e140a.wav"), (20, 145, "H20e145a.wav"), (20, 150, "H20e150a.wav"), (20, 155, "H20e155a.wav"),
    (20, 160, "H20e160a.wav"), (20, 165, "H20e165a.wav"), (20, 170, "H20e170a.wav"), (20, 175, "H20e175a.wav"), (20, 180, "H20e180a.wav"),
    (30, 0, "H30e000a.wav"), (30, 6, "H30e006a.wav"), (30, 12, "H30e012a.wav"), (30, 18, "H30e018a.wav"), (30, 24, "H30e024a.wav"),
    (30, 30, "H30e030a.wav"), (30, 36, "H30e036a.wav"), (30, 42, "H30e042a.wav"), (30, 48, "H30e048a.wav"), (30, 54, "H30e054a.wav"),
    (30, 60, "H30e060a.wav"), (30, 66, "H30e066a.wav"), (30, 72, "H30e072a.wav"), (30, 78, "H30e078a.wav"), (30, 84, "H30e084a.wav"),
    (30, 90, "H30e090a.wav"), (30, 96, "H30e096a.wav"), (30, 102, "H30e102a.wav"), (30, 108, "H30e108a.wav"), (30, 114, "H30e114a.wav"),
    (30, 120, "H30e120a.wav"), (30, 126, "H30e126a.wav"), (30, 132, "H30e132a.wav"), (30, 138, "H30e138a.wav"), (30, 144, "H30e144a.wav"),
    (30, 150, "H30e150a.wav"), (30, 156, "H30e156a.wav"), (30, 162, "H30e162a.wav"), (30, 168, "H30e168a.wav"), (30, 174, "H30e174a.wav"),
    (30, 180, "H30e180a.wav"), (40, 0, "H40e000a.wav"), (40, 6, "H40e006a.wav"), (40, 13, "H40e013a.wav"), (40, 19, "H40e019a.wav"),
    (40, 26, "H40e026a.wav"), (40, 32, "H40e032a.wav"), (40, 39, "H40e039a.wav"), (40, 45, "H40e045a.wav"), (40, 51, "H40e051a.wav"),
    (40, 58, "H40e058a.wav"), (40, 64, "H40e064a.wav"), (40, 71, "H40e071a.wav"), (40, 77, "H40e077a.wav"), (40, 84, "H40e084a.wav"),
    (40, 90, "H40e090a.wav"), (40, 96, "H40e096a.wav"), (40, 103, "H40e103a.wav"), (40, 109, "H40e109a.wav"), (40, 116, "H40e116a.wav"),
    (40, 122, "H40e122a.wav"), (40, 129, "H40e129a.wav"), (40, 135, "H40e135a.wav"), (40, 141, "H40e141a.wav"), (40, 148, "H40e148a.wav"),
    (40, 154, "H40e154a.wav"), (40, 161, "H40e161a.wav"), (40, 167, "H40e167a.wav"), (40, 174, "H40e174a.wav"), (40, 180, "H40e180a.wav"),
    (50, 0, "H50e000a.wav"), (50, 8, "H50e008a.wav"), (50, 16, "H50e016a.wav"), (50, 24, "H50e024a.wav"), (50, 32, "H50e032a.wav"),
    (50, 40, "H50e040a.wav"), (50, 48, "H50e048a.wav"), (50, 56, "H50e056a.wav"), (50, 64, "H50e064a.wav"), (50, 72, "H50e072a.wav"),
    (50, 80, "H50e080a.wav"), (50, 88, "H50e088a.wav"), (50, 96, "H50e096a.wav"), (50, 104, "H50e104a.wav"), (50, 112, "H50e112a.wav"),
    (50, 120, "H50e120a.wav"), (50, 128, "H50e128a.wav"), (50, 136, "H50e136a.wav"), (50, 144, "H50e144a.wav"), (50, 152, "H50e152a.wav"),
    (50, 160, "H50e160a.wav"), (50, 168, "H50e168a.wav"), (50, 176, "H50e176a.wav"), (60, 0, "H60e000a.wav"), (60, 10, "H60e010a.wav"),
    (60, 20, "H60e020a.wav"), (60, 30, "H60e030a.wav"), (60, 40, "H60e040a.wav"), (60, 50, "H60e050a.wav"), (60, 60, "H60e060a.wav"),
    (60, 70, "H60e070a.wav"), (60, 80, "H60e080a.wav"), (60, 90, "H60e090a.wav"), (60, 100, "H60e100a.wav"), (60, 110, "H60e110a.wav"),
    (60, 120, "H60e120a.wav"), (60, 130, "H60e130a.wav"), (60, 140, "H60e140a.wav"), (60, 150, "H60e150a.wav"), (60, 160, "H60e160a.wav"),
    (60, 170, "H60e170a.wav"), (60, 180, "H60e180a.wav"), (70, 0, "H70e000a.wav"), (70, 15, "H70e015a.wav"), (70, 30, "H70e030a.wav"),
    (70, 45, "H70e045a.wav"), (70, 60, "H70e060a.wav"), (70, 75, "H70e075a.wav"), (70, 90, "H70e090a.wav"), (70, 105, "H70e105a.wav"),
    (70, 120, "H70e120a.wav"), (70, 135, "H70e135a.wav"), (70, 150, "H70e150a.wav"), (70, 165, "H70e165a.wav"), (70, 180, "H70e180a.wav"),
    (80, 0, "H80e000a.wav"), (80, 30, "H80e030a.wav"), (80, 60, "H80e060a.wav"), (80, 90, "H80e090a.wav"), (80, 120, "H80e120a.wav"),
    (80, 150, "H80e150a.wav"), (80, 180, "H80e180a.wav"), (90, 0, "H90e000a.wav"),
    )

    @staticmethod
    def hrtf_filename_for(azimuth: float, elevation: float) -> str:
        """
        Return the KEMAR HRTF filename with the least positional error for the given (azimuth, elevation).

        The dataset covers 0°–180° azimuth (one hemisphere); negative azimuth (left side) uses the same
        file as the symmetric positive angle (caller should swap L/R when rendering). Positional error
        is squared Euclidean distance in (elevation, azimuth) space.

        Args:
            azimuth: Horizontal angle in degrees (e.g. -180 to 180; 0 = front).
            elevation: Vertical angle in degrees (0 = horizontal).

        Returns:
            Filename string (e.g. "H0e045a.wav") for the nearest KEMAR compact HRTF.
        """
        az = min(180.0, abs(float(azimuth)))
        elev = float(elevation)
        best = min(
            SpatialHRTF.KEMAR_HRTF_ENTRIES,
            key=lambda e: (e[0] - elev) ** 2 + (e[1] - az) ** 2,
        )
        filename = best[2]
        logger.debug(
            "SpatialHRTF.hrtf_filename_for: az=%.1f, el=%.1f -> %s (nearest elev=%.1f, az=%.1f)",
            azimuth,
            elevation,
            filename,
            float(best[0]),
            float(best[1]),
        )
        return filename

    def __init__(self, azimuth: Union[float, int], elevation: Union[float, int] = 0.0):
        if isinstance(azimuth, ProcessingElement) or isinstance(elevation, ProcessingElement):
            raise ValueError(
                "SpatialHRTF: azimuth and elevation must be static (float or int). "
                "Dynamic values would switch impulse responses during rendering and cause discontinuities."
            )
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)

        self._ir_cache: dict[str, tuple[np.ndarray, int]] = {}
        self._tail: Optional[np.ndarray] = None
        self._last_render_end: Optional[int] = None
        self._warned_sr_mismatch: bool = False
    
    @property
    def output_channels(self) -> int:
        return 2  # Stereo

    def _load_ir(self) -> tuple[np.ndarray, int]:
        filename = SpatialHRTF.hrtf_filename_for(self.azimuth, self.elevation)
        if filename in self._ir_cache:
            return self._ir_cache[filename]

        path = get_kemar_dir() / filename
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[1] != 2:
            raise ValueError(f"SpatialHRTF: expected stereo IR, got shape {data.shape} for {path}")

        self._ir_cache[filename] = (data, int(sr))
        return self._ir_cache[filename]

    def _reset_tail_if_noncontiguous(self, start: int) -> None:
        if self._last_render_end is None or start != self._last_render_end:
            self._tail = None

    def render(
        self,
        source_snippet: Snippet,
        start: int,
        duration: int,
        sample_rate: int,
    ) -> np.ndarray:
        ir_data, ir_sr = self._load_ir()
        if sample_rate != ir_sr and not self._warned_sr_mismatch:
            handle_error(
                f"SpatialHRTF: IR sample rate is {ir_sr} Hz but source is {sample_rate} Hz. "
                "Proceeding without resampling.",
                fatal=False,
            )
            self._warned_sr_mismatch = True

        # Mix M channels to mono
        source_data = source_snippet.data
        mono_data = np.mean(source_data, axis=1).astype(np.float32, copy=False)

        # Select IR channels, swap for left-side azimuth
        left_ir = ir_data[:, 0]
        right_ir = ir_data[:, 1]
        if self.azimuth < 0:
            left_ir, right_ir = right_ir, left_ir

        ir_len = left_ir.shape[0]
        tail_len = max(ir_len - 1, 0)

        self._reset_tail_if_noncontiguous(start)

        if tail_len == 0:
            x = mono_data
        else:
            if self._tail is None or self._tail.shape[0] != tail_len:
                self._tail = np.zeros(tail_len, dtype=np.float32)
            x = np.concatenate([self._tail, mono_data])

        left = fftconvolve(x, left_ir, mode="full")
        right = fftconvolve(x, right_ir, mode="full")

        if tail_len == 0:
            out_left = left[:duration]
            out_right = right[:duration]
        else:
            out_left = left[tail_len:tail_len + duration]
            out_right = right[tail_len:tail_len + duration]

        if tail_len > 0:
            self._tail = x[-tail_len:]
        self._last_render_end = start + duration

        output = np.column_stack([out_left, out_right]).astype(np.float32, copy=False)
        return output

    def __repr__(self) -> str:
        return f"SpatialHRTF(azimuth={self.azimuth:.1f}, elevation={self.elevation:.1f})"


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
        pan_control = PiecewisePE([(0, -90.0), (44100, 90.0)])  # Sweep left to right
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
        Return input PEs (source and any dynamic method parameters).
        
        Includes the source and any ProcessingElement parameters from the method
        (e.g., dynamic azimuth for SpatialLinear/SpatialConstantPower).
        SpatialHRTF uses static azimuth/elevation only, so it adds no extra inputs.
        """
        return [self._source, *self._method.inputs()]
    
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
        output_data = self._method.render(
            source_snippet,
            start,
            duration,
            self._source.sample_rate,
        )
        return Snippet(start, output_data)
    
    def __repr__(self) -> str:
        return f"SpatialPE(source={self._source.__class__.__name__}, method={self._method})"
