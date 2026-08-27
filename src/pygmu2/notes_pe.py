"""
NotesPE - play a list of MIDI-style notes from a single source sample.

NotesPE pitch-shifts and time-positions a source PE (e.g. a WAV file of a
single instrument note) to play an arbitrary list of notes. Each note is
pitch-shifted via ResamplePE (pure constant-rate resampling), cropped to its
duration, gain-scaled by velocity, and positioned at its onset time.

This module also provides:
  - Note          data class for a single MIDI note
  - get_notes_from_midi(path)  parse notes from a MIDI file (uses mido)

Usage example::

    import pygmu2 as pg
    from pygmu2 import NotesPE, get_notes_from_midi

    pg.set_sample_rate(48000)

    source = pg.WavReaderPE("piano_c4.wav")
    notes  = get_notes_from_midi("melody.mid")
    music  = NotesPE(source, notes, tempo=120, native_pitch=60)
    pg.play(music)

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from pygmu2.processing_element import ProcessingElement
from pygmu2.array_pe import ArrayPE
from pygmu2.extent import Extent
from pygmu2.snippet import Snippet
from pygmu2.crop_pe import CropPE
from pygmu2.delay_pe import DelayPE
from pygmu2.gain_pe import GainPE
from pygmu2.resample_pe import ResamplePE

# ---------------------------------------------------------------------------
# Note data class
# ---------------------------------------------------------------------------


class Note:
    """
    A single MIDI-style note.

    Offsets and durations are stored in quarter-note (beat) units so they are
    independent of tempo.  Use the helper methods to convert to sample counts.

    Args:
        offset_beats:   Note onset in quarter notes from the start of the piece.
        duration_beats: Note duration in quarter notes.
        pitch:          MIDI pitch number (0–127).  60 = middle C.
        velocity:       MIDI velocity (1–127).  127 = maximum.
    """

    def __init__(
        self,
        offset_beats: float,
        duration_beats: float,
        pitch: int,
        velocity: int = 100,
    ):
        self._offset_beats = float(offset_beats)
        self._duration_beats = float(duration_beats)
        self._pitch = int(pitch)
        self._velocity = int(max(1, min(127, velocity)))
        # Set by NotesPE.build_internal_pes() – the rendered PE chain.
        self.fab_pe: ProcessingElement | None = None

    # ------------------------------------------------------------------
    # Sample-count helpers
    # ------------------------------------------------------------------

    def offset_samples(self, tempo: float = 120.0, sample_rate: int = 48000) -> int:
        """Return onset position in samples."""
        return int(self._offset_beats * 60.0 / tempo * sample_rate)

    def duration_samples(
        self,
        tempo: float = 120.0,
        sample_rate: int = 48000,
        release_secs: float = 0.0,
    ) -> int:
        """Return note duration in samples (optionally extended by a release tail)."""
        return int((self._duration_beats * 60.0 / tempo + release_secs) * sample_rate)

    def extent(
        self,
        tempo: float = 120.0,
        sample_rate: int = 48000,
        release_secs: float = 0.0,
    ) -> Extent:
        """Return the Extent [onset, onset + duration] in samples."""
        start = self.offset_samples(tempo, sample_rate)
        end = start + self.duration_samples(tempo, sample_rate, release_secs)
        return Extent(start, end)

    def rate(self, native_pitch: int = 60) -> float:
        """
        Playback rate needed to shift this note's pitch relative to native_pitch.

        Returns a value > 1 for pitches above native (higher, faster),
        and < 1 for pitches below native (lower, slower).
        """
        steps = self._pitch - native_pitch
        return math.pow(2.0, steps / 12.0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pitch(self) -> int:
        return self._pitch

    @property
    def velocity(self) -> int:
        return self._velocity

    @property
    def offset_beats(self) -> float:
        return self._offset_beats

    @property
    def duration_beats(self) -> float:
        return self._duration_beats

    def __repr__(self) -> str:
        return (
            f"<Note pitch={self._pitch} offset={self._offset_beats:.3f}b "
            f"dur={self._duration_beats:.3f}b vel={self._velocity}>"
        )


# ---------------------------------------------------------------------------
# NotesPE
# ---------------------------------------------------------------------------


class NotesPE(ProcessingElement):
    """
    Render a list of Notes from a single source sample.

    Each note is built as a pure PE chain::

        ResamplePE(src, rate)          # pitch-shift via speed change
        → CropPE(0, note_dur_samples)  # trim to note duration
        → GainPE(velocity_gain)        # scale by MIDI velocity
        → DelayPE(onset_samples)       # position at note onset

    All PEs in the chain are pure, so NotesPE itself is pure and safe to use
    in any rendering context.

    **Stateful sources** (e.g. a BiquadPE filter chain) are automatically
    pre-rendered into an in-memory ArrayPE during construction.  This avoids
    the "multiple sinks on a non-pure PE" error that would otherwise occur
    because every note shares the same source.  The pre-render window covers
    the maximum source duration any note will read (``note_dur × rate``).

    Args:
        src_pe:        Source sound.  Pure PEs (WavReaderPE, SinePE, …) are
                       shared directly.  Non-pure PEs are pre-rendered once.
        note_list:     Sequence of Note objects.
        tempo:         Playback tempo in BPM (default 120).
        gain_factor:   Master gain applied on top of velocity scaling (default 0.5).
        native_pitch:  MIDI pitch of the source sample (default 60 = middle C).
        release_secs:  Extra tail added to each note's crop window (default 0).
    """

    def __init__(
        self,
        src_pe: ProcessingElement,
        note_list: Sequence[Note],
        tempo: float = 120.0,
        gain_factor: float = 0.5,
        native_pitch: int = 60,
        release_secs: float = 0.0,
    ):
        self._src_pe = src_pe
        self._note_list = list(note_list)
        self._tempo = float(tempo)
        self._gain_factor = float(gain_factor)
        self._native_pitch = int(native_pitch)
        self._release_secs = float(release_secs)

        self._build_internal_pes()

    # ------------------------------------------------------------------
    # Internal PE construction
    # ------------------------------------------------------------------

    def _velocity_gain(self, velocity: int) -> float:
        """Map MIDI velocity (1–127) to a linear amplitude multiplier."""
        return self._gain_factor * (velocity / 127.0)

    def _prebake_source(self) -> ProcessingElement:
        """
        Return a pure source PE suitable for sharing across all note chains.

        If ``src_pe`` is already pure, it is returned unchanged.

        If ``src_pe`` is stateful (e.g. contains a BiquadPE filter), it is
        pre-rendered into an ArrayPE.  The render window spans from sample 0
        to the furthest source index any note will read::

            max_frames = max(note_dur_samples × playback_rate) over all notes

        This is called once during construction, before note chains are built.
        """
        if not self._src_pe.stateful:
            return self._src_pe

        if not self._note_list:
            return self._src_pe

        sr = self.sample_rate
        max_src_frames = max(
            int(
                math.ceil(
                    note.duration_samples(self._tempo, sr, self._release_secs)
                    * note.rate(self._native_pitch)
                )
            )
            for note in self._note_list
        )
        # Add one extra sample as an interpolation margin.
        max_src_frames = max(1, max_src_frames + 1)

        snippet = self._src_pe.render(0, max_src_frames)
        return ArrayPE(snippet.data)

    def _build_internal_pes(self) -> None:
        """Build a PE chain for every note in the list."""
        sr = self.sample_rate
        # Resolve source: pre-render to ArrayPE if stateful.
        src = self._prebake_source()
        # Replace so inputs() / channel_count() / repr stay consistent.
        self._src_pe = src

        for note in self._note_list:
            playback_rate = note.rate(self._native_pitch)
            dur = note.duration_samples(self._tempo, sr, self._release_secs)
            onset = note.offset_samples(self._tempo, sr)
            vol = self._velocity_gain(note.velocity)

            # Pure pitch-shift: output position t reads source at t * rate
            pitched = ResamplePE(src, playback_rate)
            # Trim to note duration (in the shifted time domain)
            cropped = CropPE(pitched, 0, dur)
            # Scale by velocity
            gained = GainPE(cropped, vol)
            # Place at onset position in the timeline
            positioned = DelayPE(gained, onset)

            note.fab_pe = positioned

    # ------------------------------------------------------------------
    # ProcessingElement interface
    # ------------------------------------------------------------------

    def inputs(self) -> list[ProcessingElement]:
        return [self._src_pe]

    def channel_count(self) -> int | None:
        return self._src_pe.channel_count()

    def _compute_extent(self) -> Extent:
        """Union of all note extents."""
        if not self._note_list:
            return Extent(0, 0)
        sr = self.sample_rate
        result = self._note_list[0].extent(self._tempo, sr, self._release_secs)
        for note in self._note_list[1:]:
            result = result.union(note.extent(self._tempo, sr, self._release_secs))
        return result

    def _render(self, start: int, duration: int) -> Snippet:
        req_extent = Extent(start, start + duration)
        channels = self._src_pe.channel_count() or 1
        result: np.ndarray | None = None

        for note in self._note_list:
            note_extent = note.extent(self._tempo, self.sample_rate, self._release_secs)
            if not note_extent.intersects(req_extent):
                continue
            snippet = note.fab_pe.render(start, duration)
            if result is None:
                result = snippet.data.copy()
            else:
                result = result + snippet.data

        if result is None:
            return Snippet.from_zeros(start, duration, channels)
        return Snippet(start, result)

    def __repr__(self) -> str:
        return (
            f"NotesPE(src={self._src_pe.__class__.__name__}, "
            f"notes={len(self._note_list)}, tempo={self._tempo}, "
            f"native_pitch={self._native_pitch})"
        )


# ---------------------------------------------------------------------------
# MIDI file parsing
# ---------------------------------------------------------------------------


def get_notes_from_midi(midi_path: str) -> list[Note]:
    """
    Parse all notes from a MIDI file and return them as a list of Note objects.

    Uses ``mido`` (already a pygmu2 dependency) instead of music21.

    Offsets and durations are stored in quarter-note (beat) units relative to
    the start of the file, using a constant tempo (tempo changes in the MIDI
    file are ignored; apply the correct tempo when constructing NotesPE).

    Notes from all tracks are merged into a single flat list sorted by onset.

    Args:
        midi_path: Path to a Standard MIDI File (.mid / .midi).

    Returns:
        List of Note objects, sorted by onset time.

    Example::

        notes = get_notes_from_midi("song.mid")
        music = NotesPE(source, notes, tempo=120, native_pitch=60)
    """
    try:
        import mido
    except ImportError as exc:
        raise ImportError(
            "mido is required to parse MIDI files. " "Install it with: pip install mido"
        ) from exc

    mid = mido.MidiFile(midi_path)
    ticks_per_beat: int = mid.ticks_per_beat
    notes: list[Note] = []

    for track in mid.tracks:
        abs_tick: int = 0
        # (channel, pitch) → (abs_tick_start, velocity)
        active: dict[tuple[int, int], tuple[int, int]] = {}

        for msg in track:
            abs_tick += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                # Re-trigger: close any already-open note at this pitch
                if key in active:
                    start_tick, vel = active.pop(key)
                    _add_note(
                        notes, start_tick, abs_tick, msg.note, vel, ticks_per_beat
                    )
                active[key] = (abs_tick, msg.velocity)

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                if key in active:
                    start_tick, vel = active.pop(key)
                    _add_note(
                        notes, start_tick, abs_tick, msg.note, vel, ticks_per_beat
                    )

        # Close any notes still open at end of track (malformed MIDI)
        for (channel, pitch), (start_tick, vel) in active.items():
            _add_note(notes, start_tick, abs_tick, pitch, vel, ticks_per_beat)

    notes.sort(key=lambda n: n.offset_beats)
    return notes


def _add_note(
    notes: list[Note],
    start_tick: int,
    end_tick: int,
    pitch: int,
    velocity: int,
    ticks_per_beat: int,
) -> None:
    """Convert tick values to beat units and append a Note."""
    offset_beats = start_tick / ticks_per_beat
    duration_beats = max(0.0, (end_tick - start_tick) / ticks_per_beat)
    if duration_beats > 0:
        notes.append(Note(offset_beats, duration_beats, pitch, velocity))
