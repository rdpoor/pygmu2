"""
ReverbPE - convolution reverb with wet/dry mix.

ReverbPE wraps a ConvolvePE (wet) and mixes it with the dry signal using
GainPE + MixPE. An optional normalization scales the wet signal by the
impulse response (IR) energy so wet/dry balance is more predictable across
IR files.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

from typing import Optional, Union

from pygmu2.processing_element import ProcessingElement
from pygmu2.convolve_pe import ConvolvePE
from pygmu2.gain_pe import GainPE
from pygmu2.mix_pe import MixPE
from pygmu2.constant_pe import ConstantPE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet


class ReverbPE(ProcessingElement):
    """
    Convolution reverb with a wet/dry mix control.

    Conceptually equivalent to:
        wet = ConvolvePE(source, ir)
        out = MixPE(GainPE(source, 1 - mix), GainPE(wet, mix / ir_energy))

    Args:
        source: Input audio PE.
        ir: Impulse response PE. Must be finite and start at 0 (Extent(0, N)).
        mix: Wet/dry ratio in [0.0, 1.0], or a mono PE producing values in
            that range. 0.0 = fully dry, 1.0 = fully wet.
        normalize_ir: If True, scale wet gain by IR energy norm.
        fft_size: Optional FFT size override for ConvolvePE.

    Notes:
        - ReverbPE is not pure (ConvolvePE is stateful); render() requests must
          be contiguous.
        - IRs should be mono or match the source channel count. For channel
          conversion (e.g., mono -> stereo), adapt the source before ReverbPE.
        - If mix is provided as a PE, it must be mono (1 channel) when known.
    """

    def __init__(
        self,
        source: ProcessingElement,
        ir: ProcessingElement,
        mix: Union[float, ProcessingElement] = 0.5,
        *,
        normalize_ir: bool = True,
        fft_size: Optional[int] = None,
    ):
        self._source = source
        self._ir = ir
        self._mix = mix
        self._normalize_ir = bool(normalize_ir)
        self._fft_size = fft_size

        if isinstance(mix, ProcessingElement):
            mix_ch = mix.channel_count()
            if mix_ch is not None and int(mix_ch) != 1:
                raise ValueError(f"mix PE must be mono, got {mix_ch} channels")
        else:
            mix = float(mix)
            if not (0.0 <= mix <= 1.0):
                raise ValueError(f"mix must be in [0.0, 1.0], got {mix}")

        self._ir_energy = ConvolvePE.ir_energy_norm(self._ir) if self._normalize_ir else 1.0

        # Build internal graph
        self._wet_stream = ConvolvePE(self._source, self._ir, fft_size=self._fft_size)

        if isinstance(self._mix, ProcessingElement):
            dry_gain_pe = MixPE(ConstantPE(1.0), GainPE(self._mix, gain=-1.0))
            wet_gain_pe: ProcessingElement = self._mix
            if self._normalize_ir:
                wet_gain_pe = GainPE(wet_gain_pe, gain=(1.0 / self._ir_energy))
        else:
            dry_gain_pe = 1.0 - float(self._mix)
            wet_gain_pe = float(self._mix)
            if self._normalize_ir:
                wet_gain_pe = wet_gain_pe / self._ir_energy

        self._dry_gain = GainPE(self._source, gain=dry_gain_pe)
        self._wet_gain = GainPE(self._wet_stream, gain=wet_gain_pe)
        self._mix_stream = MixPE(self._dry_gain, self._wet_gain)
        self._out = self._mix_stream

    @property
    def source(self) -> ProcessingElement:
        return self._source

    @property
    def ir(self) -> ProcessingElement:
        return self._ir

    @property
    def mix(self) -> Union[float, ProcessingElement]:
        """Mix parameter (float or mono PE)."""
        return self._mix

    @property
    def ir_energy(self) -> float:
        """Energy norm of the IR (used for normalization)."""
        return self._ir_energy

    def inputs(self) -> list[ProcessingElement]:
        # Delegate to the composed output graph so configure() reaches all internals.
        return [self._out]

    def is_pure(self) -> bool:
        # Convolution is stateful; require contiguous renders.
        return False

    def channel_count(self) -> Optional[int]:
        return self._out.channel_count()

    def _compute_extent(self) -> Extent:
        return self._out.extent()

    def _render(self, start: int, duration: int) -> Snippet:
        return self._out.render(start, duration)

    def __repr__(self) -> str:
        mix_name = self._mix.__class__.__name__ if isinstance(self._mix, ProcessingElement) else str(self._mix)
        return (
            f"ReverbPE(source={self._source.__class__.__name__}, "
            f"ir={self._ir.__class__.__name__}, mix={mix_name}, "
            f"normalize_ir={self._normalize_ir}, fft_size={self._fft_size})"
        )
