"""
What happens when you shift just some of the phases in an FFT before converting
back to the time domain?

Source file is a drum beat, 99 BPM, 1.65 beats per second, 26727.27 samples
per beat.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations


import numpy as np
from typing import Callable, Tuple

from pathlib import Path
import pygmu2 as pg
from examples_helper import run_demos

SAMPLE_RATE = 44100
pg.set_sample_rate(SAMPLE_RATE)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# This could become a full fledged PE someday...

from pygmu2.extent import Extent
from pygmu2.processing_element import ProcessingElement
from pygmu2.snippet import Snippet

class FftManglerPE(ProcessingElement):
    """
    PE that modifies magnitude and phase of a source in the frequency domian
    before returning it to the time domain.

    Renders the full source extent, FFTs (per channel), modifies the magnitude
    and phase with a user-supplied function, IFFTs, and caches the result.
    Subsequent render requests return slices (or zero-padded slices) of that
    cached buffer.

    The source must have finite extent (extent().start and extent().end
    not None). Memory use is O(extent.duration * channels).

    Args:
        source: Input PE with finite extent.
        mangler: user-supplied function, takes magnitude and phase arrays as 
           inputs and returns (modified_magnitude, modified_phase).
        normalize_peak: Optional linear amplitude (e.g. 0.5) to scale peak to; None = no normalization.
    """

    def __init__(
        self,
        source: ProcessingElement,
        mangler: Callable([np.ndarray, np.ngdarray], [np.ndarray, np.ngdarray]),
        normalize_peak: float | None = None,
    ):
        self._source = source
        self._mangler = mangler
        if normalize_peak is not None and (normalize_peak <= 0 or not np.isfinite(normalize_peak)):
            raise ValueError(
                f"normalize_peak must be a positive finite number, got {normalize_peak!r}"
            )
        self._normalize_peak = normalize_peak
        self._mogrified: np.ndarray | None = None  # (samples, channels), float32

    def inputs(self) -> list[ProcessingElement]:
        return [self._source]

    def _compute_extent(self) -> Extent:
        return self._source.extent()

    def channel_count(self) -> int | None:
        return self._source.channel_count()

    def is_pure(self) -> bool:
        return True

    def _mogrify(self) -> np.ndarray:
        """Render full source, FFT → random phases → IFFT; cache and return (samples, channels)."""
        if self._mogrified is not None:
            return self._mogrified

        ext = self.extent()
        if ext.start is None or ext.end is None:
            raise ValueError(
                f"{self.__class__.__name__} requires finite source extent; "
                f"got start={ext.start}, end={ext.end}"
            )
        n_frames = ext.duration
        if n_frames is None or n_frames <= 0:
            raise ValueError(
                f"{self.__class__.__name__} requires positive extent duration; "
                f"got duration={n_frames}"
            )

        snippet = self._source.render(ext.start, n_frames)
        frames = snippet.data  # (samples, channels), float32

        # FFT along time axis (axis=0)
        analysis = np.fft.fft(frames, axis=0)
        magnitudes = np.abs(analysis)
        phases = np.angle(analysis)

        # mangle magnitudes and phases
        m_magnitudes, m_phases = self._mangler(magnitudes, phases)
        m_analysis = m_magnitudes * np.exp(1j * m_phases)

        # perform IFFT to put back into time domain
        self._mogrified = np.real(np.fft.ifft(m_analysis, axis=0)).astype(
            np.float32
        )

        # optionally normalize peak level
        if self._normalize_peak is not None:
            peak = np.max(np.abs(self._mogrified))
            if peak > 0:
                self._mogrified *= (self._normalize_peak / peak)
        return self._mogrified

    def _render(self, start: int, duration: int) -> Snippet:
        ext = self.extent()
        if ext.start is None or ext.end is None:
            return Snippet.from_zeros(
                start, duration, self.channel_count() or 1
            )

        mogrified = self._mogrify()
        channels = mogrified.shape[1]
        req_end = start + duration

        # No overlap with extent
        if req_end <= ext.start or start >= ext.end:
            return Snippet.from_zeros(start, duration, channels)

        # Request fully inside extent: return slice
        if ext.spans(start, duration):
            local_start = start - ext.start
            slice_data = mogrified[local_start : local_start + duration].copy()
            return Snippet(start, slice_data)

        # Partial overlap: build output with zeros and mogrified slice
        out = np.zeros((duration, channels), dtype=np.float32)
        overlap_start = max(start, ext.start)
        overlap_end = min(req_end, ext.end)
        if overlap_end <= overlap_start:
            return Snippet(start, out)

        mog_start = overlap_start - ext.start
        mog_end = overlap_end - ext.start
        out_start = overlap_start - start
        out_end = overlap_end - start
        out[out_start:out_end, :] = mogrified[mog_start:mog_end, :]
        return Snippet(start, out)

    def __repr__(self) -> str:
        parts = [f"source={self._source.__class__.__name__}"]
        if self._normalize_peak is not None:
            parts.append(f"normalize_peak={self._normalize_peak}")
        return f"FftManglerPE({', '.join(parts)})"

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Demos

AUDIO_DIR = Path(__file__).parent / "audio"
DRUM_WAV = pg.WavReaderPE(str(AUDIO_DIR / "LOA_99_Drums_DoubleDown.wav"))
BPM = 99
BPS = BPM / 60.0

def negate_phases_fn(f_lo:float, f_hi:float):
    # Return a function that negates phases between f_lo and f_hi

    def mangler_fn(magnitudes, phases):
        sr = pg.get_sample_rate()
        fft_len = len(magnitudes)
        # Convert frequency to FFT bin
        bin_lo = int(round(f_lo * fft_len / sr))
        bin_hi = int(round(f_hi * fft_len / sr))
        print(f"freg[{f_lo}, {f_hi}] => bin[{bin_lo}, {bin_hi}]")
        for i in range(bin_lo, bin_hi):
            phases[i] = -phases[i]
        return magnitudes, phases

    return mangler_fn

def demo_drums_dry():
    source = DRUM_WAV
    pg.play(pg.GainPE(source, gain=0.71), SAMPLE_RATE)

def demo_reverse_low_frequencies():
    source = DRUM_WAV
    mangled = FftManglerPE(
        source, 
        negate_phases_fn(0, 850), 
        normalize_peak=0.33
    )
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_reverse_high_frequencies():
    source = DRUM_WAV
    mangled = FftManglerPE(
        source, 
        negate_phases_fn(850, 20000), 
        normalize_peak=0.33
    )
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_reverse_mid_frequencies():
    source = DRUM_WAV
    mangled = FftManglerPE(
        source, 
        negate_phases_fn(100, 800), 
        normalize_peak=0.33
    )
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_shift_increasing_frequencies():
    source = DRUM_WAV

    def mangler(magnitudes, phases):
        for i in range(len(phases)-1):
            shift = 0.3 * float(i) / len(phases)
            phases[i+1] += phases[i+1] * shift # don't touch DC
        return magnitudes, phases
    mangled = FftManglerPE(
        source, 
        mangler, 
        normalize_peak=0.33
    )
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_shift_decreasing_frequencies():
    source = DRUM_WAV

    def mangler(magnitudes, phases):
        for i in range(len(phases)-1):
            shift = 0.3 * float(i) / len(phases)
            phases[i+1] += phases[i+1] * (1.0 - shift) # don't touch DC
        return magnitudes, phases
    mangled = FftManglerPE(
        source, 
        mangler, 
        normalize_peak=0.33
    )
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_tralfam():
    source = DRUM_WAV

    def mangler(magnitudes, phases):
        rng = np.random.default_rng()
        phases = rng.random((len(phases), 2)) * 2.0 * np.pi
        return magnitudes, phases
    mangled = FftManglerPE(
        source, 
        mangler, 
        normalize_peak=0.33
    )
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_alternate_phases():
    source = DRUM_WAV

    def mangler(magnitudes, phases):
        for i in range(100, len(phases), 2):
            tmp = phases[i]
            phases[i] = phases[i+1]
            phases[i+1] = tmp
        return magnitudes, phases
    mangled = FftManglerPE(
        source, 
        mangler, 
        normalize_peak=0.33
    )
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

def demo_negate_every_other_phase():
    source = DRUM_WAV

    def mangler(magnitudes, phases):
        for i in range(10, len(phases), 2):
            phases[i] *= -1
        return magnitudes, phases
    mangled = FftManglerPE(
        source, 
        mangler, 
        normalize_peak=0.33
    )
    pg.play(pg.GainPE(mangled, gain=0.71), SAMPLE_RATE)

DEMOS = [
    ("Dry drums", demo_drums_dry),
    ("Reverse low frequency phases", demo_reverse_low_frequencies),
    ("Reverse high frequency phases", demo_reverse_high_frequencies),
    ("Reverse mid frequency phases", demo_reverse_mid_frequencies),
    ("Progressively phase shift higher frequencies", demo_shift_increasing_frequencies),
    ("Progressively phase shift lower frequencies", demo_shift_decreasing_frequencies),
    ("Randomize phases (tralfam)", demo_tralfam),
    ("Alternate phases", demo_alternate_phases),
    ("Negate every other phase", demo_negate_every_other_phase),
]

if __name__ == "__main__":
    run_demos(DEMOS)
