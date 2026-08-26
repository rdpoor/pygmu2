"""
Utility helpers for rendering and playback.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from __future__ import annotations


import os
import subprocess
import tempfile
from pathlib import Path

from pygmu2.config import get_sample_rate
from pygmu2.processing_element import ProcessingElement
from pygmu2.audio_renderer import AudioRenderer
from pygmu2.null_renderer import NullRenderer
from pygmu2.wav_reader_pe import WavReaderPE
from pygmu2.wav_writer_pe import WavWriterPE

_DEFAULT_CHUNK_FRAMES = 8192


def render_to_file(
    source: ProcessingElement,
    out_path: str,
    *,
    extent=None,
    chunk_frames: int = _DEFAULT_CHUNK_FRAMES,
) -> None:
    """
    Render a PE to a WAV file as fast as possible using NullRenderer.

    Rendering is done in fixed-size chunks so that intermediate PE buffers
    stay small regardless of the total duration.  MagFreqPE / TralfamPE cache
    their result on the first chunk and serve slices thereafter, so chunked
    rendering is safe even for FFT-based PEs.

    Args:
        source: PE to render (must have finite extent).
        out_path: Path to write WAV file.
        extent: Optional precomputed extent (to avoid recomputation).
        chunk_frames: Number of frames per render call (default 8192).
    """
    sr = get_sample_rate()
    if sr is None:
        raise RuntimeError("Sample rate not set. Call pg.set_sample_rate() first.")
    if extent is None:
        extent = source.extent()
    if extent.start is None or extent.end is None:
        raise RuntimeError("Cannot render to file: source has infinite extent.")

    writer = WavWriterPE(source, out_path, sample_rate=sr)
    renderer = NullRenderer(sample_rate=sr)
    renderer.set_source(writer)

    with renderer:
        renderer.start()
        pos = extent.start
        end = extent.end
        while pos < end:
            chunk = min(chunk_frames, end - pos)
            renderer.render(pos, chunk)
            pos += chunk


def play(source: ProcessingElement, device=None) -> None:
    """
    Play a PE in real time using AudioRenderer.
    """
    sr = get_sample_rate()
    if sr is None:
        raise RuntimeError("Sample rate not set. Call pg.set_sample_rate() first.")
    renderer = AudioRenderer(sample_rate=sr, device=device)
    renderer.set_source(source)
    with renderer:
        renderer.start()
        renderer.play_extent()


def play_offline(
    source: ProcessingElement,
    path: str | None = None,
) -> None:
    """
    Render a PE to a WAV file offline, then play it back.

    If path is None, a temporary file is created and deleted after playback.
    """
    sr = get_sample_rate()
    if sr is None:
        raise RuntimeError("Sample rate not set. Call pg.set_sample_rate() first.")
    extent = source.extent()
    if extent.start is None or extent.end is None:
        raise RuntimeError("Cannot render offline: source has infinite extent.")

    if path is None:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            render_to_file(source, tmp_path, extent=extent)
            play(WavReaderPE(tmp_path))
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
    else:
        render_to_file(source, path, extent=extent)
        play(WavReaderPE(path))


def browse(
    source: ProcessingElement,
    path: str | None = None,
) -> None:
    """
    Render a PE to a WAV file, then open it in the jog/shuttle player.

    The jogshuttle player runs as a separate process and this function
    returns immediately.

    Args:
        source: PE to render (must have finite extent).
        path: Path to write WAV file.  If None, a temporary file is created
              and automatically deleted when the player closes.
    """
    sr = get_sample_rate()
    if sr is None:
        raise RuntimeError("Sample rate not set. Call pg.set_sample_rate() first.")
    extent = source.extent()
    if extent.start is None or extent.end is None:
        raise RuntimeError("Cannot browse: source has infinite extent.")

    delete_on_close = path is None
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    path = str(Path(path).resolve())
    render_to_file(source, path, extent=extent)

    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts" / "jogshuttle.py"
    if not script_path.exists():
        raise FileNotFoundError(
            "scripts/jogshuttle.py not found — run from the pygmu2 source tree"
        )
    cmd = [
        "uv",
        "run",
        "--directory",
        str(project_root),
        "python",
        str(script_path),
        path,
    ]
    if delete_on_close:
        cmd.append("--delete-on-close")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.Popen(cmd, env=env)
