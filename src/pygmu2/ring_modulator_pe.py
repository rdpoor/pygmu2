"""
RingModulatorPE - multiplies two signals for ring/amplitude modulation.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet, broadcast_channels


class RingModulatorPE(ProcessingElement):
    """
    A ProcessingElement that ring-modulates a carrier signal with a modulator signal.

    Ring modulation multiplies two signals sample-by-sample, producing sum and
    difference sidebands while suppressing the carrier — a classic analog effect
    found on Moog, Roland, Doepfer, and Buchla systems.

    Formula:
        wet    = carrier × (modulator + bias)
        output = (1 − mix) × carrier + mix × wet

    Modes:
        bias=0, mix=1 → pure ring modulation (carrier suppressed)
        bias=1, mix=1 → amplitude modulation (carrier present)
        mix=0         → dry carrier pass-through

    Args:
        carrier: The signal being modulated (required).
        modulator: The modulating signal (required).
        bias: DC offset added to modulator before multiplication.
              0.0 = pure ring mod; 1.0 = amplitude modulation. Default 0.0.
              Accepts float or ProcessingElement.
        mix: Wet/dry blend. 0.0 = dry carrier only; 1.0 = full ring mod. Default 1.0.
             Accepts float or ProcessingElement.

    Channel handling:
        - Same channel count: element-wise multiply directly.
        - Modulator mono, carrier stereo: modulator is broadcast to all channels.
        - channel_count() returns carrier's channel count.
    """

    def __init__(
        self,
        carrier: ProcessingElement,
        modulator: ProcessingElement,
        bias: float | ProcessingElement = 0.0,
        mix: float | ProcessingElement = 1.0,
    ):
        self._carrier = carrier
        self._modulator = modulator
        self._bias = bias
        self._mix = mix

        self._bias_is_pe = isinstance(bias, ProcessingElement)
        self._mix_is_pe = isinstance(mix, ProcessingElement)

    def inputs(self) -> list[ProcessingElement]:
        """Return input PEs: carrier, modulator, plus bias/mix if they are PEs."""
        result = [self._carrier, self._modulator]
        if self._bias_is_pe:
            result.append(self._bias)
        if self._mix_is_pe:
            result.append(self._mix)
        return result

    def channel_count(self) -> int | None:
        """Returns carrier's channel count."""
        return self._carrier.channel_count()

    def _render(self, start: int, duration: int) -> Snippet:
        """
        Render ring-modulated audio.

        Args:
            start: Starting sample index
            duration: Number of samples to render (> 0)

        Returns:
            Snippet with ring modulation applied
        """
        carrier_data = self._carrier.render(start, duration).data  # (N, ch)
        mod_data = self._modulator.render(start, duration).data  # (N, 1) or (N, ch)

        # Broadcast mono modulator across carrier channels (view);
        # a multichannel mismatch raises rather than silently adapting.
        mod_data = broadcast_channels(mod_data, carrier_data.shape[1], what="modulator")

        # Get bias and mix as (N, 1) arrays for broadcasting across channels
        bias_data = self._scalar_or_pe_values(
            self._bias,
            start,
            duration,
            dtype=np.float32,
            allow_multichannel=True,
            channels=1,
        )
        mix_data = self._scalar_or_pe_values(
            self._mix,
            start,
            duration,
            dtype=np.float32,
            allow_multichannel=True,
            channels=1,
        )

        wet = carrier_data * (mod_data + bias_data)
        out = (1.0 - mix_data) * carrier_data + mix_data * wet

        return Snippet(start, out.astype(np.float32))

    def _compute_extent(self) -> Extent:
        """Return intersection of carrier, modulator, and any PE parameters' extents."""
        extent = self._carrier.extent().intersection(self._modulator.extent())
        if self._bias_is_pe:
            extent = extent.intersection(self._bias.extent())
        if self._mix_is_pe:
            extent = extent.intersection(self._mix.extent())
        return extent

    def __repr__(self) -> str:
        bias_str = (
            f"{self._bias.__class__.__name__}(...)"
            if self._bias_is_pe
            else str(self._bias)
        )
        mix_str = (
            f"{self._mix.__class__.__name__}(...)"
            if self._mix_is_pe
            else str(self._mix)
        )
        return (
            f"RingModulatorPE("
            f"carrier={self._carrier.__class__.__name__}, "
            f"modulator={self._modulator.__class__.__name__}, "
            f"bias={bias_str}, mix={mix_str})"
        )
