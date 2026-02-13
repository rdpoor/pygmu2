#!/usr/bin/env python3
"""
Jog/Shuttle Audio GUI Player (PySide6/Qt 6)

A Qt-based audio player with variable-speed playback (jog/shuttle)
using the pygmu2 PE graph infrastructure.

Usage:
    uv run python examples/jogshuttle_qt.py [path/to/file.wav]

Requires the 'gui' optional dependency:
    uv add --optional gui PySide6

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QPointF

import pygmu2 as pg
from pygmu2 import (
    AudioRenderer,
    ControlPE,
    GainPE,
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
# Custom QProxyStyle to make QSlider jump to click position
# ---------------------------------------------------------------------------

from PySide6.QtWidgets import QProxyStyle


class JumpSliderStyle(QProxyStyle):
    """Override so clicking the slider trough jumps to that position."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_Slider_AbsoluteSetButtons:
            return Qt.LeftButton.value
        return super().styleHint(hint, option, widget, returnData)


# ---------------------------------------------------------------------------
# Waveform widget
# ---------------------------------------------------------------------------

class WaveformWidget(QWidget):
    """Custom widget that draws a waveform and supports click/drag scrubbing."""

    scrub_started = Signal(float)  # fraction 0..1
    scrub_moved = Signal(float)
    scrub_ended = Signal()

    BG_COLOR = QColor("#1a1a2e")
    WAVE_FILL = QColor("#16a085")
    CENTER_LINE = QColor("#2c3e50")
    PLAYHEAD_COLOR = QColor("#e74c3c")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._peaks: np.ndarray | None = None
        self._playhead_frac: float = 0.0
        self._dragging = False

    def set_peaks(self, peaks: np.ndarray | None) -> None:
        self._peaks = peaks
        self.update()

    def set_playhead(self, frac: float) -> None:
        self._playhead_frac = max(0.0, min(1.0, frac))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        p.fillRect(self.rect(), self.BG_COLOR)

        mid = h / 2.0

        if self._peaks is not None and len(self._peaks) > 0:
            peaks = self._peaks
            n = len(peaks)

            # Build polygon: top edge left-to-right, bottom edge right-to-left
            polygon = QPolygonF()
            for i in range(n):
                x = (i / n) * w
                y_top = mid - peaks[i, 1] * mid
                polygon.append(QPointF(x, y_top))
            for i in range(n - 1, -1, -1):
                x = (i / n) * w
                y_bot = mid - peaks[i, 0] * mid
                polygon.append(QPointF(x, y_bot))

            p.setPen(Qt.NoPen)
            p.setBrush(self.WAVE_FILL)
            p.drawPolygon(polygon)

        # Center line
        pen = QPen(self.CENTER_LINE)
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([2, 4])
        p.setPen(pen)
        p.drawLine(QPointF(0, mid), QPointF(w, mid))

        # Playhead
        x = self._playhead_frac * w
        pen = QPen(self.PLAYHEAD_COLOR)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(QPointF(x, 0), QPointF(x, h))

        p.end()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.width() > 0:
            self._dragging = True
            frac = max(0.0, min(1.0, event.position().x() / self.width()))
            self.scrub_started.emit(frac)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging and self.width() > 0:
            frac = max(0.0, min(1.0, event.position().x() / self.width()))
            self.scrub_moved.emit(frac)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.scrub_ended.emit()


# ---------------------------------------------------------------------------
# Shuttle slider with float mapping
# ---------------------------------------------------------------------------

class RateSlider(QSlider):
    """QSlider with an oversized handle that displays the playback rate."""

    HANDLE_W = 48
    HANDLE_H = 22

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._rate_text: str = "0.0x"
        # Style the handle large enough to hold text
        self.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: #3a3a4e;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: {self.HANDLE_W}px;
                height: {self.HANDLE_H}px;
                margin: -{(self.HANDLE_H - 6) // 2}px 0;
                background: #5a5a7e;
                border: 1px solid #7a7a9e;
                border-radius: 4px;
            }}
        """)

    def set_rate_text(self, text: str) -> None:
        if text != self._rate_text:
            self._rate_text = text
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        # Locate the handle and draw rate text centred inside it
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        handle = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self
        )
        p = QPainter(self)
        p.setPen(QColor("#e0e0e0"))
        p.drawText(handle, Qt.AlignCenter, self._rate_text)
        p.end()


class ShuttleSlider(QWidget):
    """QSlider wrapper mapping integer ticks to float shuttle values."""

    value_changed = Signal(float)
    pressed = Signal()
    released = Signal()

    # Integer range: -80..80 maps to float -8.0..8.0
    INT_MIN = -80
    INT_MAX = 80
    SCALE = 10.0

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label_min = QLabel("-8x")
        self._label_max = QLabel("8x")

        self._slider = RateSlider(Qt.Horizontal)
        self._jump_style = JumpSliderStyle()   # prevent GC; no-arg uses app style
        self._slider.setStyle(self._jump_style)
        self._slider.setMinimum(self.INT_MIN)
        self._slider.setMaximum(self.INT_MAX)
        self._slider.setValue(0)
        self._slider.setTickPosition(QSlider.NoTicks)

        layout.addWidget(self._label_min)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._label_max)

        self._slider.valueChanged.connect(self._on_value_changed)
        self._slider.sliderPressed.connect(self.pressed.emit)
        self._slider.sliderReleased.connect(self.released.emit)

    def _on_value_changed(self, int_val: int) -> None:
        self.value_changed.emit(int_val / self.SCALE)

    def value(self) -> float:
        return self._slider.value() / self.SCALE

    def set_rate_display(self, rate: float) -> None:
        self._slider.set_rate_text(f"{rate:.1f}x")

    def set_value(self, val: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(int(round(val * self.SCALE)))
        self._slider.blockSignals(False)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class JogShuttleApp(QMainWindow):
    """PySide6 jog/shuttle audio player backed by pygmu2."""

    # Shuttle limits
    SHUTTLE_MIN = -8.0
    SHUTTLE_MAX = 8.0
    SHUTTLE_REST = 0.0
    SHUTTLE_SNAP_ZERO = 0.3
    SHUTTLE_CURVE = 2.0

    # Polling intervals (ms)
    PLAYHEAD_POLL_MS = 33
    SPRING_BACK_MS = 16

    # Spring-back dynamics
    SPRING_FACTOR = 0.30

    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.setWindowTitle("pygmu2 Jog/Shuttle Player")
        self.setMinimumSize(640, 400)

        # Audio state
        self._wav_path: str | None = None
        self._sample_rate: int = 44100
        self._total_frames: int = 0
        self._channels: int = 1

        # PE graph
        self._wav_pe: WavReaderPE | None = None
        self._timewarp: TimeWarpPE | None = None
        self._rate_control: ControlPE | None = None
        self._renderer: AudioRenderer | None = None

        # Transport state
        self._playing = False
        self._rate: float = 0.0
        self._resume_from: int = 0
        self._scrubbing = False

        # Build UI
        self._build_ui()
        self._bind_keys()

        # Timers
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self.PLAYHEAD_POLL_MS)
        self._poll_timer.timeout.connect(self._poll_tick)
        self._poll_timer.start()

        self._spring_timer = QTimer(self)
        self._spring_timer.setInterval(self.SPRING_BACK_MS)
        self._spring_timer.timeout.connect(self._spring_back_tick)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(200)
        self._resize_timer.timeout.connect(self._do_resize)

        # Load initial file if given
        if initial_path:
            self._load_file(initial_path)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- Top bar: file info + open button ---
        top = QHBoxLayout()
        self._file_label = QLabel("No file loaded")
        top.addWidget(self._file_label, 1)
        open_btn = QPushButton("Open\u2026")
        open_btn.clicked.connect(self._on_open)
        top.addWidget(open_btn)
        layout.addLayout(top)

        # --- Waveform ---
        self._waveform = WaveformWidget()
        self._waveform.scrub_started.connect(self._on_scrub_start)
        self._waveform.scrub_moved.connect(self._on_scrub_move)
        self._waveform.scrub_ended.connect(self._on_scrub_end)
        layout.addWidget(self._waveform, 1)

        # --- Transport buttons ---
        transport = QHBoxLayout()
        transport.addStretch()
        for text, slot in [
            ("|<", self._on_beginning),
            ("Play", self._on_play),
            ("Pause", self._toggle_play_pause),
            ("Stop", self._on_stop),
            (">|", self._on_end),
        ]:
            btn = QPushButton(text)
            btn.setFixedWidth(60)
            btn.clicked.connect(slot)
            transport.addWidget(btn)
        transport.addStretch()
        layout.addLayout(transport)

        # --- Shuttle slider ---
        self._shuttle = ShuttleSlider()
        self._shuttle.value_changed.connect(self._on_shuttle_change)
        self._shuttle.pressed.connect(self._on_shuttle_press)
        self._shuttle.released.connect(self._on_shuttle_release)
        layout.addWidget(self._shuttle)

        # --- Position label ---
        self._pos_label = QLabel("Position: --:-- / --:--")
        layout.addWidget(self._pos_label)

    def _bind_keys(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(
            self._toggle_play_pause
        )
        QShortcut(QKeySequence(Qt.Key_Home), self).activated.connect(
            self._on_beginning
        )
        QShortcut(QKeySequence(Qt.Key_End), self).activated.connect(
            self._on_end
        )
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(
            self._on_stop
        )

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _on_open(self) -> None:
        init_dir = str(AUDIO_DIR) if AUDIO_DIR.is_dir() else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open audio file",
            init_dir,
            "WAV files (*.wav);;All files (*.*)",
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        logger.debug("LOAD: %s", path)
        self._teardown_graph()

        self._wav_path = path

        info = sf.info(path)
        self._sample_rate = info.samplerate
        self._total_frames = info.frames
        self._channels = info.channels

        pg.set_sample_rate(self._sample_rate)

        canvas_w = self._waveform.width() or 640
        self._waveform.set_peaks(compute_peaks(path, target_width=canvas_w))

        self._build_graph(path)

        name = Path(path).name
        dur_str = self._format_time(self._total_frames)
        self._file_label.setText(f"File: {name}  ({dur_str})")

    def _build_graph(self, path: str) -> None:
        self._rate_control = ControlPE(initial_value=self.SHUTTLE_REST)
        self._wav_pe = WavReaderPE(path)
        self._timewarp = TimeWarpPE(self._wav_pe, rate=self._rate_control)
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
    # Transport — _set_rate() is the single point of control
    # ------------------------------------------------------------------

    def _set_rate(self, rate: float) -> None:
        """Set playback rate; start/stop audio stream as needed."""
        if self._renderer is None:
            return
        self._rate = rate
        if rate != 0.0:
            self._spring_timer.stop()
        if self._rate_control is not None:
            self._rate_control.set_value(rate)
        if rate != 0.0 and not self._playing:
            self._renderer.stream_start(start=self._resume_from)
            self._playing = True
            logger.debug("STREAM_START: rate=%.2f, resume_from=%s", rate, self._resume_from)
        elif rate == 0.0 and self._playing:
            self._renderer.stream_stop()
            self._resume_from = self._renderer.stream_position
            self._playing = False
            logger.debug("STREAM_STOP: resume_from=%s", self._resume_from)

    def _on_play(self) -> None:
        logger.debug("PLAY: playing=%s, rate=%.2f", self._playing, self._rate)
        self._set_rate(1.0)

    def _on_pause(self) -> None:
        logger.debug("PAUSE: playing=%s, rate=%.2f", self._playing, self._rate)
        self._set_rate(0.0)

    def _on_stop(self) -> None:
        logger.debug("STOP: playing=%s, rate=%.2f", self._playing, self._rate)
        if self._renderer is None:
            return
        self._set_rate(0.0)
        self._renderer.stop()
        self._renderer.start()
        self._resume_from = 0
        if self._timewarp is not None:
            self._timewarp._pos = 0.0
        self._spring_timer.stop()
        self._shuttle.set_value(self.SHUTTLE_REST)

    def _on_beginning(self) -> None:
        logger.debug("BEGINNING: playing=%s", self._playing)
        if self._timewarp is not None:
            self._timewarp._pos = 0.0

    def _on_end(self) -> None:
        logger.debug("END: playing=%s", self._playing)
        if self._timewarp is not None and self._total_frames > 0:
            self._timewarp._pos = float(self._total_frames)

    def _toggle_play_pause(self) -> None:
        logger.debug("TOGGLE: playing=%s, rate=%.2f", self._playing, self._rate)
        if self._playing:
            self._on_pause()
        else:
            self._on_play()

    # ------------------------------------------------------------------
    # Shuttle slider
    # ------------------------------------------------------------------

    def _slider_to_rate(self, val: float) -> float:
        """Map slider position to playback rate via power curve."""
        if val == 0.0:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        normalized = abs(val) / self.SHUTTLE_MAX
        return sign * (normalized ** self.SHUTTLE_CURVE) * self.SHUTTLE_MAX

    def _on_shuttle_change(self, val: float) -> None:
        if abs(val) < self.SHUTTLE_SNAP_ZERO:
            val = 0.0
            self._shuttle.set_value(val)
        self._set_rate(self._slider_to_rate(val))

    def _on_shuttle_press(self) -> None:
        self._spring_timer.stop()

    def _on_shuttle_release(self) -> None:
        self._set_rate(0.0)
        self._spring_timer.start()

    def _spring_back_tick(self) -> None:
        current = self._shuttle.value()
        diff = self.SHUTTLE_REST - current
        if abs(diff) < 0.05:
            self._shuttle.set_value(self.SHUTTLE_REST)
            self._spring_timer.stop()
            return
        new_val = current + diff * self.SPRING_FACTOR
        self._shuttle.set_value(new_val)

    # ------------------------------------------------------------------
    # Waveform scrubbing
    # ------------------------------------------------------------------

    def _on_scrub_start(self, frac: float) -> None:
        if self._total_frames == 0 or self._timewarp is None:
            return
        target = int(frac * self._total_frames)
        if not self._playing:
            self._scrubbing = True
            self._set_rate(1.0)
        self._timewarp._pos = float(target)

    def _on_scrub_move(self, frac: float) -> None:
        if self._timewarp is None or self._total_frames == 0:
            return
        self._timewarp._pos = float(int(frac * self._total_frames))

    def _on_scrub_end(self) -> None:
        if self._scrubbing:
            self._scrubbing = False
            self._set_rate(0.0)

    # ------------------------------------------------------------------
    # Resize handling
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start()

    def _do_resize(self) -> None:
        if self._wav_path is not None:
            new_width = self._waveform.width()
            if new_width > 10:
                self._waveform.set_peaks(
                    compute_peaks(self._wav_path, target_width=new_width)
                )

    # ------------------------------------------------------------------
    # Playhead polling & auto-stop
    # ------------------------------------------------------------------

    def _poll_tick(self) -> None:
        if self._timewarp is not None and self._total_frames > 0:
            pos = self._timewarp._pos
            if pos < 0:
                self._timewarp._pos = 0.0
                pos = 0.0
            elif pos > self._total_frames:
                self._timewarp._pos = float(self._total_frames)
                pos = float(self._total_frames)
            if self._playing and not self._scrubbing:
                at_end = pos >= self._total_frames and self._rate > 0
                at_start = pos <= 0 and self._rate < 0
                if at_end or at_start:
                    logger.debug("AUTO-STOP: pos=%.1f, rate=%.2f", pos, self._rate)
                    self._set_rate(0.0)
            self._waveform.set_playhead(pos / self._total_frames)
            self._shuttle.set_rate_display(self._rate)
            pos_str = self._format_time(max(0, pos))
            dur_str = self._format_time(self._total_frames)
            self._pos_label.setText(f"Position: {pos_str} / {dur_str}")

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

    def closeEvent(self, event: QCloseEvent) -> None:
        self._poll_timer.stop()
        self._spring_timer.stop()
        self._resize_timer.stop()
        self._teardown_graph()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="pygmu2 Jog/Shuttle Player (Qt)")
    parser.add_argument("file", nargs="?", default=None,
                        help="Path to a WAV file to open on startup")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging to stderr")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    pg.set_sample_rate(44100)

    app = QApplication(sys.argv)
    window = JogShuttleApp(initial_path=args.file)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
