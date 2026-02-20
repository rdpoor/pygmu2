"""
MeltysynthPE - SoundFont synthesis as a pygmu2 SourcePE.

Wraps meltysynth Synthesizer: _render(start, duration) fills left/right
buffers via synthesizer.render() and returns a stereo Snippet. Impure (stateful);
render requests must be contiguous.

Expose .synthesizer so a MidiInPE callback can call note_on/note_off or
process_midi_message to drive the synth.

Requires: pygmu2.meltysynth (SoundFont, Synthesizer, SynthesizerSettings).

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pygmu2.extent import Extent
from pygmu2.processing_element import SourcePE
from pygmu2.snippet import Snippet


class MeltysynthPE(SourcePE):
    """
    Source PE that renders meltysynth SoundFont synthesis into stereo Snippets.

    Loads the SoundFont in _on_start(). _render(start, duration) calls
    synthesizer.render(left, right, 0, duration) and returns
    Snippet(start, data) with shape (duration, 2).

    Drive the synth from a MidiInPE callback, e.g.:
      synth_pe = MeltysynthPE(soundfont_path)
      def callback(sample_index, msg):
          if msg.type == "note_on" and msg.velocity > 0:
              synth_pe.synthesizer.note_on(msg.channel, msg.note, msg.velocity)
          elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
              synth_pe.synthesizer.note_off(msg.channel, msg.note)
      midi_in = MidiInPE(callback=callback)
      mix = MixPE(GainPE(midi_in, 0.0), synth_pe)

    Output: 2 channels (stereo). Extent: infinite. Impure.
    """

    def __init__(
        self,
        soundfont_path: str,
        block_size: int = 64,
        program: int | None = None,
    ):
        """
        Args:
            soundfont_path: Path to .sf2 SoundFont file.
            block_size: Meltysynth internal block size (8–1024). Default 64.
            program: Optional GM program 0–127 for channel 0 at start. None = use soundfont default (0).
        """
        self._soundfont_path = str(Path(soundfont_path).resolve())
        self._block_size = block_size
        self._program = program
        self._synthesizer: object | None = None

    @property
    def synthesizer(self) -> object | None:
        """The meltysynth Synthesizer instance (None until after start). Use in MidiInPE callback."""
        return self._synthesizer

    def _on_start(self) -> None:
        from pygmu2.meltysynth import SoundFont, Synthesizer, SynthesizerSettings

        if not Path(self._soundfont_path).exists():
            raise FileNotFoundError(f"SoundFont not found: {self._soundfont_path}")
        sound_font = SoundFont.from_file(self._soundfont_path)
        settings = SynthesizerSettings(self.sample_rate)
        settings.block_size = self._block_size
        self._synthesizer = Synthesizer(sound_font, settings)
        if self._program is not None:
            self._synthesizer.process_midi_message(0, 0xC0, self._program, 0)

    def _on_stop(self) -> None:
        self._synthesizer = None

    def _render(self, start: int, duration: int) -> Snippet:
        if self._synthesizer is None:
            return Snippet.from_zeros(start, duration, 2)

        left = np.zeros(duration, dtype=np.float64)
        right = np.zeros(duration, dtype=np.float64)
        self._synthesizer.render(left, right, 0, duration)
        data = np.column_stack([left, right]).astype(np.float64)
        return Snippet(start, data)

    def _compute_extent(self) -> Extent:
        return Extent(None, None)

    def channel_count(self) -> int:
        return 2

    def is_pure(self) -> bool:
        return False

    def __repr__(self) -> str:
        prog = f", program={self._program}" if self._program is not None else ""
        return f"MeltysynthPE(soundfont_path={self._soundfont_path!r}, block_size={self._block_size}{prog})"
