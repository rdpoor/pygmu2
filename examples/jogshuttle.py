#!/usr/bin/env python3
"""
Jog/Shuttle Audio GUI Player

A tkinter-based audio player with variable-speed playback (jog/shuttle)
using the pygmu2 PE graph infrastructure.

Usage:
    uv run python examples/jogshuttle.py [path/to/file.wav]

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import argparse
import logging
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import soundfile as sf

import pygmu2 as pg
from pygmu2 import (
    AudioRenderer,
    ControlPE,
    GainPE,
    LoopPE,
    TimeWarpPE,
    WavReaderPE,
)

logger = logging.getLogger("jogshuttle")

AUDIO_DIR = Path(__file__).parent / "audio"


# ---------------------------------------------------------------------------
# Waveform peak cache
# ---------------------------------------------------------------------------

def compute_peaks(path: str, target_width: int = 2000) -> np.ndarray:
    """Return (target_width, 2) array of [min, max] peaks for waveform display.

    Mixes to mono first, then buckets frames into *target_width* bins.
    """
    data, _ = sf.read(path, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    n = len(data)
    if n == 0:
        return np.zeros((target_width, 2), dtype=np.float32)
    bin_size = max(1, n // target_width)
    trim = bin_size * target_width
    if trim > n:
        target_width = n // bin_size
        trim = bin_size * target_width
    if target_width == 0:
        return np.zeros((1, 2), dtype=np.float32)
    chunk = data[:trim].reshape(target_width, bin_size)
    mins = chunk.min(axis=1)
    maxs = chunk.max(axis=1)
    return np.column_stack([mins, maxs])


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class JogShuttleApp:
    """Tkinter jog/shuttle audio player backed by pygmu2."""

    # Shuttle limits
    SHUTTLE_MIN = -8.0
    SHUTTLE_MAX = 8.0
    SHUTTLE_REST = 1.0
    SHUTTLE_RES = 0.1

    # Polling intervals (ms)
    PLAYHEAD_POLL_MS = 33  # ~30 Hz
    SPRING_BACK_MS = 16    # ~60 fps

    # Spring-back dynamics
    SPRING_FACTOR = 0.15   # exponential ease factor per tick

    def __init__(self, root: tk.Tk, initial_path: str | None = None):
        self.root = root
        self.root.title("pygmu2 Jog/Shuttle Player")
        self.root.minsize(640, 400)

        # Audio state
        self._wav_path: str | None = None
        self._sample_rate: int = 44100
        self._total_frames: int = 0
        self._channels: int = 1
        self._peaks: np.ndarray | None = None

        # PE graph
        self._wav_pe: WavReaderPE | None = None
        self._timewarp: TimeWarpPE | None = None
        self._rate_control: ControlPE | None = None
        self._renderer: AudioRenderer | None = None

        # Transport state
        self._playing = False
        self._resume_from: int = 0
        self._spring_back_id: str | None = None
        self._poll_id: str | None = None
        self._shuttle_grabbed = False

        # Build UI
        self._build_ui()
        self._bind_keys()

        # Start playhead polling
        self._schedule_poll()

        # Load initial file if given
        if initial_path:
            self._load_file(initial_path)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # --- Top bar: file info + open button ---
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=(8, 0))

        self._file_label = tk.Label(top, text="No file loaded", anchor="w")
        self._file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        open_btn = tk.Button(top, text="Open\u2026", command=self._on_open)
        open_btn.pack(side=tk.RIGHT)

        # --- Waveform canvas ---
        self._canvas = tk.Canvas(self.root, height=160, bg="#1a1a2e",
                                 highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        # --- Transport buttons ---
        transport = tk.Frame(self.root)
        transport.pack(pady=4)

        for text, cmd in [
            ("|<", self._on_beginning),
            ("Play", self._on_play),
            ("Pause", self._on_pause),
            ("Stop", self._on_stop),
            (">|", self._on_end),
        ]:
            tk.Button(transport, text=text, width=7, command=cmd).pack(
                side=tk.LEFT, padx=2
            )

        # --- Shuttle slider ---
        shuttle_frame = tk.Frame(self.root)
        shuttle_frame.pack(fill=tk.X, padx=8)

        tk.Label(shuttle_frame, text=f"{self.SHUTTLE_MIN:.0f}x").pack(side=tk.LEFT)
        self._shuttle_var = tk.DoubleVar(value=self.SHUTTLE_REST)
        self._shuttle = tk.Scale(
            shuttle_frame,
            from_=self.SHUTTLE_MIN,
            to=self.SHUTTLE_MAX,
            resolution=self.SHUTTLE_RES,
            orient=tk.HORIZONTAL,
            variable=self._shuttle_var,
            showvalue=True,
            command=self._on_shuttle_change,
        )
        self._shuttle.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(shuttle_frame, text=f"{self.SHUTTLE_MAX:.0f}x").pack(side=tk.RIGHT)

        # Bind press/release on the shuttle scale's trough for spring-back
        self._shuttle.bind("<ButtonPress-1>", self._on_shuttle_press)
        self._shuttle.bind("<ButtonRelease-1>", self._on_shuttle_release)

        # --- Position label ---
        self._pos_label = tk.Label(self.root, text="Position: --:-- / --:--",
                                   anchor="w")
        self._pos_label.pack(fill=tk.X, padx=8, pady=(0, 8))

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda e: self._toggle_play_pause())
        self.root.bind("<Home>", lambda e: self._on_beginning())
        self.root.bind("<End>", lambda e: self._on_end())
        self.root.bind("<Escape>", lambda e: self._on_stop())

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _on_open(self) -> None:
        path = filedialog.askopenfilename(
            title="Open audio file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialdir=str(AUDIO_DIR) if AUDIO_DIR.is_dir() else None,
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        logger.debug("LOAD: %s", path)
        # Tear down any existing graph
        self._teardown_graph()

        self._wav_path = path

        # Read file metadata
        info = sf.info(path)
        self._sample_rate = info.samplerate
        self._total_frames = info.frames
        self._channels = info.channels

        # Set global sample rate
        pg.set_sample_rate(self._sample_rate)

        # Compute waveform peaks
        canvas_w = self._canvas.winfo_width() or 640
        self._peaks = compute_peaks(path, target_width=canvas_w)

        # Build PE graph
        self._build_graph(path)

        # Update UI
        name = Path(path).name
        dur_str = self._format_time(self._total_frames)
        self._file_label.config(text=f"File: {name}  ({dur_str})")
        self._draw_waveform()

    def _build_graph(self, path: str) -> None:
        self._rate_control = ControlPE(initial_value=self.SHUTTLE_REST)
        self._wav_pe = WavReaderPE(path)
        loop_pe = LoopPE(self._wav_pe, crossfade_seconds=0.005)
        self._timewarp = TimeWarpPE(loop_pe, rate=self._rate_control)
        output = GainPE(self._timewarp, gain=0.8)

        self._renderer = AudioRenderer(
            sample_rate=self._sample_rate,
            blocksize=1024,
            latency="low",
        )
        self._renderer.set_source(output)
        self._renderer.start()

    def _teardown_graph(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.stream_stop()
            except Exception:
                pass
            try:
                self._renderer.stop()
            except Exception:
                pass
            self._renderer = None
        self._timewarp = None
        self._rate_control = None
        self._wav_pe = None
        self._playing = False

    # ------------------------------------------------------------------
    # Transport controls
    # ------------------------------------------------------------------

    def _on_play(self) -> None:
        tw_pos = self._timewarp._pos if self._timewarp else "?"
        rate = self._shuttle_var.get()
        logger.debug(
            "PLAY pressed: playing=%s, resume_from=%s, rate=%.2f, tw._pos=%s, renderer=%s",
            self._playing, self._resume_from, rate, tw_pos,
            "ok" if self._renderer else "None",
        )
        if self._renderer is None:
            return
        if not self._playing:
            if self._rate_control is not None:
                self._rate_control.set_value(rate)
            # Resume from where we left off (preserves impure PE contiguity)
            self._renderer.stream_start(start=self._resume_from)
            self._playing = True
            logger.debug("PLAY: streaming started at %s", self._resume_from)

    def _on_pause(self) -> None:
        tw_pos = self._timewarp._pos if self._timewarp else "?"
        logger.debug(
            "PAUSE pressed: playing=%s, tw._pos=%s, renderer=%s",
            self._playing, tw_pos,
            "ok" if self._renderer else "None",
        )
        if self._renderer is None:
            return
        if self._playing:
            # Save position before stopping the stream
            self._resume_from = self._renderer.stream_position
            self._renderer.stream_stop()
            self._playing = False
            logger.debug("PAUSE: stopped, resume_from=%s", self._resume_from)

    def _on_stop(self) -> None:
        tw_pos = self._timewarp._pos if self._timewarp else "?"
        logger.debug(
            "STOP pressed: playing=%s, tw._pos=%s, renderer=%s",
            self._playing, tw_pos,
            "ok" if self._renderer else "None",
        )
        if self._renderer is None:
            return
        if self._playing:
            self._renderer.stream_stop()
            self._playing = False
        # Full stop: reset TimeWarpPE._pos to 0
        self._renderer.stop()
        self._renderer.start()
        # After full stop/start, PE contiguity is reset — start from 0
        self._resume_from = 0
        logger.debug("STOP: graph reset, resume_from=0, tw._pos=%s",
                      self._timewarp._pos if self._timewarp else "?")
        # Reset shuttle to 1x
        self._cancel_spring_back()
        self._shuttle_var.set(self.SHUTTLE_REST)
        if self._rate_control is not None:
            self._rate_control.set_value(self.SHUTTLE_REST)

    def _on_beginning(self) -> None:
        logger.debug("BEGINNING pressed: playing=%s", self._playing)
        self._seek_to(0)

    def _on_end(self) -> None:
        logger.debug("END pressed: playing=%s", self._playing)
        if self._total_frames > 0:
            self._seek_to(self._total_frames)

    def _seek_to(self, frame: int) -> None:
        if self._timewarp is None:
            return
        was_playing = self._playing
        old_pos = self._timewarp._pos
        if self._playing:
            self._resume_from = self._renderer.stream_position
            self._renderer.stream_stop()
            self._playing = False
        self._timewarp._pos = float(frame)
        logger.debug(
            "SEEK: %s -> %s, was_playing=%s, resume_from=%s",
            old_pos, frame, was_playing, self._resume_from,
        )
        if was_playing:
            self._renderer.stream_start(start=self._resume_from)
            self._playing = True

    def _toggle_play_pause(self) -> None:
        logger.debug("TOGGLE: playing=%s", self._playing)
        if self._playing:
            self._on_pause()
        else:
            self._on_play()

    # ------------------------------------------------------------------
    # Shuttle slider
    # ------------------------------------------------------------------

    def _on_shuttle_change(self, val_str: str) -> None:
        val = float(val_str)
        logger.debug("SHUTTLE: %.2f", val)
        if self._rate_control is not None:
            self._rate_control.set_value(val)

    def _on_shuttle_press(self, event: tk.Event) -> None:
        logger.debug("SHUTTLE_PRESS")
        self._shuttle_grabbed = True
        self._cancel_spring_back()

    def _on_shuttle_release(self, event: tk.Event) -> None:
        logger.debug("SHUTTLE_RELEASE: starting spring-back from %.2f",
                      self._shuttle_var.get())
        self._shuttle_grabbed = False
        self._start_spring_back()

    def _start_spring_back(self) -> None:
        self._cancel_spring_back()
        self._spring_back_tick()

    def _cancel_spring_back(self) -> None:
        if self._spring_back_id is not None:
            self.root.after_cancel(self._spring_back_id)
            self._spring_back_id = None

    def _spring_back_tick(self) -> None:
        if self._shuttle_grabbed:
            return
        current = self._shuttle_var.get()
        diff = self.SHUTTLE_REST - current
        if abs(diff) < 0.05:
            self._shuttle_var.set(self.SHUTTLE_REST)
            if self._rate_control is not None:
                self._rate_control.set_value(self.SHUTTLE_REST)
            self._spring_back_id = None
            return
        new_val = current + diff * self.SPRING_FACTOR
        self._shuttle_var.set(new_val)
        if self._rate_control is not None:
            self._rate_control.set_value(new_val)
        self._spring_back_id = self.root.after(
            self.SPRING_BACK_MS, self._spring_back_tick
        )

    # ------------------------------------------------------------------
    # Waveform display
    # ------------------------------------------------------------------

    def _draw_waveform(self) -> None:
        self._canvas.delete("all")
        if self._peaks is None or len(self._peaks) == 0:
            return

        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w < 2 or h < 2:
            return

        mid = h / 2.0
        peaks = self._peaks
        n = len(peaks)

        # Build polygon: top edge left-to-right, bottom edge right-to-left
        points: list[float] = []
        for i in range(n):
            x = (i / n) * w
            y_top = mid - peaks[i, 1] * mid  # max sample -> top
            points.extend([x, y_top])
        for i in range(n - 1, -1, -1):
            x = (i / n) * w
            y_bot = mid - peaks[i, 0] * mid  # min sample -> bottom
            points.extend([x, y_bot])

        poly = points
        if len(poly) >= 6:
            self._canvas.create_polygon(poly, fill="#16a085", outline="#1abc9c",
                                        width=0)

        # Center line
        self._canvas.create_line(0, mid, w, mid, fill="#2c3e50", dash=(2, 4))

    def _draw_playhead(self) -> None:
        self._canvas.delete("playhead")
        if self._timewarp is None or self._total_frames == 0:
            return
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        pos = self._timewarp._pos
        frac = pos / self._total_frames
        frac = max(0.0, min(1.0, frac))
        x = frac * w
        self._canvas.create_line(x, 0, x, h, fill="#e74c3c", width=2,
                                 tags="playhead")

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self._total_frames == 0:
            return
        w = self._canvas.winfo_width()
        if w <= 0:
            return
        frac = event.x / w
        frac = max(0.0, min(1.0, frac))
        target = int(frac * self._total_frames)
        logger.debug("CANVAS_CLICK: x=%d, frac=%.3f, target=%d", event.x, frac, target)
        self._seek_to(target)

    def _on_canvas_resize(self, event: tk.Event) -> None:
        # Recompute peaks at new width and redraw
        if self._wav_path is not None:
            new_width = event.width
            if new_width > 10:
                self._peaks = compute_peaks(self._wav_path, target_width=new_width)
        self._draw_waveform()

    # ------------------------------------------------------------------
    # Playhead polling & auto-stop
    # ------------------------------------------------------------------

    def _schedule_poll(self) -> None:
        self._poll_tick()

    def _poll_tick(self) -> None:
        if self._timewarp is not None and self._total_frames > 0:
            pos = self._timewarp._pos
            # Auto-stop at EOF or before-beginning
            if self._playing and (pos >= self._total_frames or pos < 0):
                logger.debug("AUTO-STOP: pos=%.1f, total=%d", pos, self._total_frames)
                self._on_pause()
            # Update playhead
            self._draw_playhead()
            # Update position label
            pos_str = self._format_time(max(0, pos))
            dur_str = self._format_time(self._total_frames)
            rate = self._shuttle_var.get()
            self._pos_label.config(
                text=f"Position: {pos_str} / {dur_str}   Rate: {rate:.1f}x"
            )
        self._poll_id = self.root.after(self.PLAYHEAD_POLL_MS, self._poll_tick)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _format_time(self, frames: float) -> str:
        if self._sample_rate == 0:
            return "00:00.000"
        secs = abs(frames) / self._sample_rate
        mins = int(secs // 60)
        secs_rem = secs - mins * 60
        return f"{mins:02d}:{secs_rem:06.3f}"

    def on_close(self) -> None:
        """Clean shutdown."""
        self._cancel_spring_back()
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
        self._teardown_graph()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="pygmu2 Jog/Shuttle Player")
    parser.add_argument("file", nargs="?", default=None,
                        help="Path to a WAV file to open on startup")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging to stderr")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Need a default sample rate for PE construction before any file is loaded
    pg.set_sample_rate(44100)

    root = tk.Tk()
    app = JogShuttleApp(root, initial_path=args.file)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
