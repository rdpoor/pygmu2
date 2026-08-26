#!/usr/bin/env python3
"""
Jog/Shuttle Audio GUI Player (PySide6/Qt 6)

A Qt-based audio player with variable-speed playback (jog/shuttle)
using the pygmu2 PE graph infrastructure.

Usage:
    uv run python scripts/jogshuttle.py [path/to/file.wav]

Requires the 'gui' optional dependency:
    uv add --optional gui PySide6

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors
MIT License
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QPointF

import pygmu2 as pg
from pygmu2 import (
    AudioRenderer,
    ControlPE,
    GainPE,
    MixPE,
    TimeWarpPE,
    WavReaderPE,
)
from pygmu2.nofft_spectrogram import (
    NoFFTSpectrogramConfig,
    NoFFTSpectrogramResult,
    compute_nofft_spectrogram_from_file,
)
from pygmu2.annotations import (
    FrameAnnotation,
    load_annotation_sidecar,
    load_sidecar_readme,
    normalize_annotations_to_frames,
    resolve_annotation_sidecar_path,
)

logger = logging.getLogger("jogshuttle")

AUDIO_DIR = Path(__file__).resolve().parent.parent / "examples" / "audio"
APP_ICON_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "pygmu2"
    / "assets"
    / "target2_icon_black.png"
)

MAX_ANNOTATION_ROWS = 28

_DEFAULT_HELP_MD = """\
# pygmu2 Jog/Shuttle Player

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| **Space** | Play / Pause |
| **Home** | Jump to beginning |
| **End** | Jump to end |
| **Escape** | Stop and rewind |

## Shuttle slider

Drag the shuttle slider to scrub through the audio at variable speed.
The slider springs back to the rest position when released.

## Waveform / Spectrogram

Click or drag on the waveform or spectrogram view to scrub to
a position in the audio file.
"""

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
# Stem detection
# ---------------------------------------------------------------------------


def find_stems(wav_path: str) -> list[tuple[str, str]]:
    """Detect stem files alongside *wav_path* and return [(path, name), ...].

    For a file named ``foo.wav`` this looks for siblings whose names match
    ``foo_<N>.wav`` or ``foo_<N>_<any text>.wav`` (case-insensitive), where
    *N* is one or more digits.  Results are sorted by N.

    Example matches for ``song.wav``:
        ``song_1_drums.wav``  → name "drums"
        ``song_2_bass.wav``   → name "bass"
        ``song_3.wav``        → name "Stem 3"
    """
    p = Path(wav_path)
    base = p.stem
    parent = p.parent
    pattern = re.compile(
        r"^" + re.escape(base) + r"_(\d+)(?:_(.+))?\.wav$",
        re.IGNORECASE,
    )
    hits: list[tuple[int, str, str]] = []
    try:
        for candidate in parent.iterdir():
            if candidate.resolve() == p.resolve():
                continue
            m = pattern.match(candidate.name)
            if m:
                n = int(m.group(1))
                raw_name = m.group(2) or f"Stem {n}"
                # Replace underscores with spaces for display
                stem_name = raw_name.replace("_", " ")
                hits.append((n, str(candidate), stem_name))
    except OSError:
        pass
    hits.sort(key=lambda x: x[0])
    return [(path, name) for _, path, name in hits]


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


@dataclass(frozen=True)
class TimelineOverlayStyle:
    onset_line: QColor
    label_bg: QColor
    label_fg: QColor


def _draw_timeline_annotations(
    p: QPainter,
    width: int,
    height: int,
    annotations: list[FrameAnnotation],
    total_frames: int,
    style: TimelineOverlayStyle,
) -> None:
    if width <= 0 or height <= 0 or total_frames <= 0 or not annotations:
        return

    frame_span = max(1, int(total_frames))
    line_pen = QPen(style.onset_line)
    line_pen.setWidth(1)
    line_pen.setStyle(Qt.DashLine)
    line_pen.setDashPattern([4, 4])

    # Draw dashed vertical lines at onset and end times.
    p.setPen(line_pen)
    for ann in annotations:
        x = int(round((ann.onset_frame / frame_span) * width))
        x = max(0, min(width - 1, x))
        p.drawLine(QPointF(x, 0), QPointF(x, height))
        if ann.end_frame is not None and ann.end_frame > ann.onset_frame:
            x_end = int(round((ann.end_frame / frame_span) * width))
            x_end = max(0, min(width - 1, x_end))
            p.drawLine(QPointF(x_end, 0), QPointF(x_end, height))

    fm = p.fontMetrics()
    row_height = fm.height() + 4
    max_rows = max(1, min(MAX_ANNOTATION_ROWS, max(1, (height - 4) // row_height)))
    row_right_edges = [-10_000] * max_rows
    max_label_w = max(80, min(width // 3, 240))
    for ann in annotations:
        text = ann.label.strip() or ann.kind.strip()
        if not text:
            continue
        x = int(round((ann.onset_frame / frame_span) * width))
        x = max(0, min(width - 1, x))
        text = fm.elidedText(text, Qt.ElideRight, max_label_w)
        text_w = fm.horizontalAdvance(text)
        left = min(max(2, x + 4), max(2, width - text_w - 8))
        right = left + text_w + 6

        row_index: int | None = None
        for idx, row_right in enumerate(row_right_edges):
            if left > row_right + 8:
                row_index = idx
                break
        if row_index is None:
            continue
        y_top = 2 + row_index * row_height
        p.fillRect(left - 2, y_top, text_w + 6, fm.height() + 2, style.label_bg)
        p.setPen(style.label_fg)
        p.drawText(left + 1, y_top + fm.ascent() + 1, text)
        p.setPen(line_pen)
        row_right_edges[row_index] = right


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
    ANNOTATION_STYLE = TimelineOverlayStyle(
        onset_line=QColor(255, 230, 140, 210),
        label_bg=QColor(20, 20, 28, 190),
        label_fg=QColor("#f8e27a"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._peaks: np.ndarray | None = None
        self._playhead_frac: float = 0.0
        self._annotations: list[FrameAnnotation] = []
        self._total_frames: int = 0
        self._dragging = False
        self._bg_cache: QPixmap | None = None

    def set_peaks(self, peaks: np.ndarray | None) -> None:
        self._peaks = peaks
        self._bg_cache = None
        self.update()

    def set_playhead(self, frac: float) -> None:
        self._playhead_frac = max(0.0, min(1.0, frac))
        self.update()

    def set_annotations(
        self, annotations: list[FrameAnnotation], total_frames: int
    ) -> None:
        self._annotations = list(annotations)
        self._total_frames = max(0, int(total_frames))
        self._bg_cache = None
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._bg_cache = None

    def _rebuild_bg_cache(self) -> None:
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            self._bg_cache = None
            return

        pix = QPixmap(w, h)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)

        p.fillRect(0, 0, w, h, self.BG_COLOR)
        mid = h / 2.0

        if self._peaks is not None and len(self._peaks) > 0:
            peaks = self._peaks
            n = len(peaks)
            polygon = QPolygonF()
            for i in range(n):
                x = (i / n) * w
                polygon.append(QPointF(x, mid - peaks[i, 1] * mid))
            for i in range(n - 1, -1, -1):
                x = (i / n) * w
                polygon.append(QPointF(x, mid - peaks[i, 0] * mid))
            p.setPen(Qt.NoPen)
            p.setBrush(self.WAVE_FILL)
            p.drawPolygon(polygon)

        pen = QPen(self.CENTER_LINE)
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([2, 4])
        p.setPen(pen)
        p.drawLine(QPointF(0, mid), QPointF(w, mid))

        _draw_timeline_annotations(
            p,
            width=w,
            height=h,
            annotations=self._annotations,
            total_frames=self._total_frames,
            style=self.ANNOTATION_STYLE,
        )
        p.end()
        self._bg_cache = pix

    def paintEvent(self, event):
        if self._bg_cache is None or self._bg_cache.size() != self.size():
            self._rebuild_bg_cache()

        p = QPainter(self)
        if self._bg_cache is not None:
            p.drawPixmap(0, 0, self._bg_cache)

        x = self._playhead_frac * self.width()
        pen = QPen(self.PLAYHEAD_COLOR)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
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


def _build_colormap_lut() -> np.ndarray:
    """
    Return a 256x3 uint8 LUT.

    Viridis ramp tuned for spectrogram readability.
    """
    color_stops = np.asarray(
        [
            [0, 0, 0],
            [72, 40, 120],
            [62, 74, 137],
            [49, 104, 142],
            [38, 130, 142],
            [31, 158, 137],
            [53, 183, 121],
            [109, 205, 89],
            [180, 222, 44],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    lut = np.zeros((256, 3), dtype=np.uint8)
    segment_count = len(color_stops) - 1
    for i in range(256):
        t = i / 255.0
        seg = min(int(t * segment_count), segment_count - 1)
        seg_t = (t * segment_count) - seg
        c0 = color_stops[seg]
        c1 = color_stops[seg + 1]
        rgb = c0 + (c1 - c0) * seg_t
        lut[i] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return lut


_COLORMAP_LUT = _build_colormap_lut()


class SpectrogramWidget(QWidget):
    """Heatmap spectrogram view with scrubbing and shared playhead semantics."""

    scrub_started = Signal(float)  # fraction 0..1
    scrub_moved = Signal(float)
    scrub_ended = Signal()

    BG_COLOR = QColor("#0f1020")
    PLAYHEAD_COLOR = QColor("#e74c3c")
    ANNOTATION_STYLE = TimelineOverlayStyle(
        onset_line=QColor(120, 220, 255, 220),
        label_bg=QColor(6, 18, 28, 195),
        label_fg=QColor(182, 230, 255),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._intensities: np.ndarray | None = None  # shape: bins x frames
        self._image: QImage | None = None
        self._playhead_frac: float = 0.0
        self._annotations: list[FrameAnnotation] = []
        self._total_frames: int = 0
        self._dragging = False
        self._bg_cache: QPixmap | None = None

    def set_spectrogram(self, intensities: np.ndarray | None) -> None:
        self._intensities = intensities
        self._rebuild_image()
        self._bg_cache = None
        self.update()

    def set_playhead(self, frac: float) -> None:
        self._playhead_frac = max(0.0, min(1.0, frac))
        self.update()

    def set_annotations(
        self, annotations: list[FrameAnnotation], total_frames: int
    ) -> None:
        self._annotations = list(annotations)
        self._total_frames = max(0, int(total_frames))
        self._bg_cache = None
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._bg_cache = None

    def _rebuild_image(self) -> None:
        if self._intensities is None or self._intensities.size == 0:
            self._image = None
            return

        intensities = np.clip(self._intensities, 0.0, 1.0)
        img_values = (intensities[::-1, :] * 255.0).astype(np.uint8, copy=False)
        rgb = _COLORMAP_LUT[img_values]
        h, w, _ = rgb.shape
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255
        self._image = QImage(
            rgba.data,
            w,
            h,
            4 * w,
            QImage.Format_RGBA8888,
        ).copy()

    def _rebuild_bg_cache(self) -> None:
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            self._bg_cache = None
            return

        pix = QPixmap(w, h)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(0, 0, w, h, self.BG_COLOR)

        if self._image is not None:
            p.drawImage(pix.rect(), self._image)

        _draw_timeline_annotations(
            p,
            width=w,
            height=h,
            annotations=self._annotations,
            total_frames=self._total_frames,
            style=self.ANNOTATION_STYLE,
        )
        p.end()
        self._bg_cache = pix

    def paintEvent(self, event):
        if self._bg_cache is None or self._bg_cache.size() != self.size():
            self._rebuild_bg_cache()

        p = QPainter(self)
        if self._bg_cache is not None:
            p.drawPixmap(0, 0, self._bg_cache)

        x = self._playhead_frac * self.width()
        pen = QPen(self.PLAYHEAD_COLOR)
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
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
# Stems mixer widgets
# ---------------------------------------------------------------------------


class StemRowWidget(QWidget):
    """One row in the stems mixer: Mute, Solo, Volume, Name."""

    mute_changed = Signal(int, bool)  # (stem_index, muted)
    solo_changed = Signal(int, bool)  # (stem_index, soloed)
    volume_changed = Signal(int, float)  # (stem_index, 0.0..1.0)

    _SS_MUTE_OFF = (
        "QPushButton { background:#3a3a4e; color:#ccc; border:1px solid #555;"
        " border-radius:3px; font-weight:bold; padding:0; }"
    )
    _SS_MUTE_ON = (
        "QPushButton { background:#c0392b; color:#fff; border:1px solid #e74c3c;"
        " border-radius:3px; font-weight:bold; padding:0; }"
    )
    _SS_SOLO_OFF = (
        "QPushButton { background:#3a3a4e; color:#ccc; border:1px solid #555;"
        " border-radius:3px; font-weight:bold; padding:0; }"
    )
    _SS_SOLO_ON = (
        "QPushButton { background:#d4ac0d; color:#1a1a2e; border:1px solid #f1c40f;"
        " border-radius:3px; font-weight:bold; padding:0; }"
    )
    _SS_VOL_SLIDER = (
        "QSlider::groove:horizontal { height:4px; background:#3a3a4e; border-radius:2px; }"
        "QSlider::handle:horizontal { width:12px; height:12px; margin:-4px 0;"
        " background:#16a085; border-radius:6px; }"
        "QSlider::sub-page:horizontal { background:#16a085; border-radius:2px; }"
    )

    def __init__(self, index: int, name: str, bg_color: str = "#1a1a2e", parent=None):
        super().__init__(parent)
        self._index = index
        self.setAutoFillBackground(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        self._mute_btn = QPushButton("M")
        self._mute_btn.setCheckable(True)
        self._mute_btn.setFixedSize(26, 24)
        self._mute_btn.setToolTip("Mute this stem")
        self._mute_btn.setStyleSheet(self._SS_MUTE_OFF)
        self._mute_btn.toggled.connect(self._on_mute_toggled)

        self._solo_btn = QPushButton("S")
        self._solo_btn.setCheckable(True)
        self._solo_btn.setFixedSize(26, 24)
        self._solo_btn.setToolTip("Solo this stem")
        self._solo_btn.setStyleSheet(self._SS_SOLO_OFF)
        self._solo_btn.toggled.connect(self._on_solo_toggled)

        vol_label = QLabel("Vol")
        vol_label.setStyleSheet("color:#888; font-size:10px;")

        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setMinimum(0)
        self._vol_slider.setMaximum(100)
        self._vol_slider.setValue(100)
        self._vol_slider.setFixedWidth(130)
        self._vol_slider.setToolTip("Volume (0–100%)")
        self._vol_slider.setStyleSheet(self._SS_VOL_SLIDER)
        self._vol_slider.valueChanged.connect(
            lambda v: self.volume_changed.emit(self._index, v / 100.0)
        )

        self._name_label = QLabel(name)
        self._name_label.setStyleSheet("color:#e0e0e0; font-size:12px;")

        layout.addWidget(self._mute_btn)
        layout.addWidget(self._solo_btn)
        layout.addWidget(vol_label)
        layout.addWidget(self._vol_slider)
        layout.addWidget(self._name_label, 1)

        # Apply alternating row background via stylesheet
        self.setStyleSheet(f"StemRowWidget {{ background: {bg_color}; }}")

    def _on_mute_toggled(self, checked: bool) -> None:
        self._mute_btn.setStyleSheet(self._SS_MUTE_ON if checked else self._SS_MUTE_OFF)
        self.mute_changed.emit(self._index, checked)

    def _on_solo_toggled(self, checked: bool) -> None:
        self._solo_btn.setStyleSheet(self._SS_SOLO_ON if checked else self._SS_SOLO_OFF)
        self.solo_changed.emit(self._index, checked)


class StemsWidget(QWidget):
    """Scrollable stem mixer — one StemRowWidget per detected stem."""

    mute_changed = Signal(int, bool)
    solo_changed = Signal(int, bool)
    volume_changed = Signal(int, float)

    _ROW_COLORS = ("#1a1a2e", "#16213e")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[StemRowWidget] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: #1a1a2e; }")

        self._content = QWidget()
        self._content.setStyleSheet("background: #1a1a2e;")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(1)
        self._content_layout.addStretch()

        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll)
        self.setMinimumHeight(80)

    def setup_stems(self, stem_names: list[str]) -> None:
        """Rebuild rows from *stem_names*.  Call after detecting stems."""
        # Clear everything from the content layout
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self._rows.clear()

        for i, name in enumerate(stem_names):
            bg = self._ROW_COLORS[i % len(self._ROW_COLORS)]
            row = StemRowWidget(i, name, bg)
            row.mute_changed.connect(self.mute_changed)
            row.solo_changed.connect(self.solo_changed)
            row.volume_changed.connect(self.volume_changed)
            self._content_layout.addWidget(row)
            self._rows.append(row)

        self._content_layout.addStretch()


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

    # Integer range: -800..800 maps to float -8.0..8.0
    INT_MIN = -800
    INT_MAX = 800
    SCALE = 100.0

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label_min = QLabel("-8x")
        self._label_max = QLabel("8x")

        self._slider = RateSlider(Qt.Horizontal)
        self._jump_style = JumpSliderStyle()  # prevent GC; no-arg uses app style
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
        self._slider.set_rate_text(f"{rate:.2f}x")

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
    SHUTTLE_SNAP_ZERO = 0.3
    SHUTTLE_CURVE = 2.0

    # Polling intervals (ms)
    PLAYHEAD_POLL_MS = 33
    SPRING_BACK_MS = 16

    # Spring-back dynamics
    SPRING_FACTOR = 0.30

    # Offline noFFT defaults
    NOFFT_MIN_FREQ = 80.0
    NOFFT_MAX_FREQ = 8000.0
    NOFFT_BINS_PER_SEMITONE = 2.0
    NOFFT_FREQUENCY_SCALE = "musical"
    NOFFT_CYCLES_FOR_DECAY = 5.0
    NOFFT_HOP_SIZE = 128
    NOFFT_MIN_DB = -51.0
    NOFFT_MAX_DB = 0.0

    def __init__(self, initial_path: str | None = None, delete_on_close: bool = False):
        super().__init__()
        self.setWindowTitle("pygmu2 Jog/Shuttle Player")
        self.setMinimumSize(640, 400)

        # Audio state
        self._wav_path: str | None = None
        self._delete_on_close: bool = delete_on_close
        self._sample_rate: int = 44100
        self._total_frames: int = 0
        self._channels: int = 1
        self._spectrogram_result: NoFFTSpectrogramResult | None = None

        # Stem state
        self._stem_paths: list[str] = []
        self._stem_names: list[str] = []
        self._stem_rate_controls: list[ControlPE] = []
        self._stem_timewarps: list[TimeWarpPE] = []
        self._stem_gain_controls: list[ControlPE] = []
        self._stem_muted: list[bool] = []
        self._stem_solo: list[bool] = []
        self._stem_volumes: list[float] = []
        self._spec_cache: dict[tuple[str, int, int, float], NoFFTSpectrogramResult] = {}
        self._active_view: str = "Waveform"
        self._spectrogram_stale: bool = True
        self._readme_text: str | None = None

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
        self._shuttle_rest: float = 0.0

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
        top.addWidget(QLabel("View:"))
        self._view_select = QComboBox()
        self._view_select.addItems(["Waveform", "Spectrogram"])
        self._view_select.currentTextChanged.connect(self._on_view_changed)
        top.addWidget(self._view_select)
        open_btn = QPushButton("Open\u2026")
        open_btn.clicked.connect(self._on_open)
        top.addWidget(open_btn)
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(28)
        help_btn.setToolTip("Show readme / help")
        help_btn.clicked.connect(self._on_show_help)
        top.addWidget(help_btn)
        layout.addLayout(top)

        # --- Visualization views ---
        self._waveform = WaveformWidget()
        self._waveform.scrub_started.connect(self._on_scrub_start)
        self._waveform.scrub_moved.connect(self._on_scrub_move)
        self._waveform.scrub_ended.connect(self._on_scrub_end)

        self._spectrogram = SpectrogramWidget()
        self._spectrogram.scrub_started.connect(self._on_scrub_start)
        self._spectrogram.scrub_moved.connect(self._on_scrub_move)
        self._spectrogram.scrub_ended.connect(self._on_scrub_end)
        self._spectrogram.hide()

        self._stems_widget = StemsWidget()
        self._stems_widget.mute_changed.connect(self._on_stem_mute_changed)
        self._stems_widget.solo_changed.connect(self._on_stem_solo_changed)
        self._stems_widget.volume_changed.connect(self._on_stem_volume_changed)
        self._stems_widget.hide()

        self._viz_container = QWidget()
        viz_layout = QVBoxLayout(self._viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(0)
        viz_layout.addWidget(self._waveform)
        viz_layout.addWidget(self._spectrogram)
        viz_layout.addWidget(self._stems_widget)
        layout.addWidget(self._viz_container, 1)

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
        self._pos_label = QLabel("Position: --:--.--- (0 samples)")
        self._pos_label.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._pos_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._pos_label)

    def _bind_keys(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(
            self._toggle_play_pause
        )
        QShortcut(QKeySequence(Qt.Key_Home), self).activated.connect(self._on_beginning)
        QShortcut(QKeySequence(Qt.Key_End), self).activated.connect(self._on_end)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self._on_stop)
        QShortcut(QKeySequence(Qt.Key_F1), self).activated.connect(self._on_show_help)

    def _on_view_changed(self, view_name: str) -> None:
        self._active_view = view_name
        self._waveform.setVisible(view_name == "Waveform")
        self._spectrogram.setVisible(view_name == "Spectrogram")
        self._stems_widget.setVisible(view_name == "Stems")
        if view_name == "Spectrogram":
            self._refresh_spectrogram_if_needed()

    def _current_viz_width(self) -> int:
        width = self._viz_container.width() if hasattr(self, "_viz_container") else 0
        if width <= 0:
            width = self._waveform.width()
        return max(32, int(width))

    def _nofft_config(self) -> NoFFTSpectrogramConfig:
        return NoFFTSpectrogramConfig(
            min_freq=self.NOFFT_MIN_FREQ,
            max_freq=self.NOFFT_MAX_FREQ,
            bins_per_semitone=self.NOFFT_BINS_PER_SEMITONE,
            frequency_scale=self.NOFFT_FREQUENCY_SCALE,
            cycles_for_decay=self.NOFFT_CYCLES_FOR_DECAY,
            hop_size=self.NOFFT_HOP_SIZE,
            min_db=self.NOFFT_MIN_DB,
            max_db=self.NOFFT_MAX_DB,
        )

    def _compute_spectrogram(
        self, path: str, target_width: int
    ) -> NoFFTSpectrogramResult:
        mtime = Path(path).stat().st_mtime
        cache_key = (path, target_width, self._sample_rate, mtime)
        cached = self._spec_cache.get(cache_key)
        if cached is not None:
            return cached

        # A mild 2x horizontal oversampling keeps the image sharp during scaling.
        target_frames = max(128, target_width * 2)
        result = compute_nofft_spectrogram_from_file(
            path,
            self._nofft_config(),
            target_frames=target_frames,
        )
        self._spec_cache[cache_key] = result
        return result

    def _refresh_spectrogram_if_needed(self, force: bool = False) -> None:
        if self._wav_path is None:
            return
        if not force and self._active_view != "Spectrogram":
            return
        if not force and not self._spectrogram_stale:
            return
        width = self._current_viz_width()
        if width <= 10:
            return
        self._spectrogram_result = self._compute_spectrogram(
            self._wav_path,
            target_width=width,
        )
        self._spectrogram.set_spectrogram(self._spectrogram_result.intensities)
        self._spectrogram_stale = False

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

    def _on_show_help(self) -> None:
        md = self._readme_text or _DEFAULT_HELP_MD
        dlg = QDialog(self)
        dlg.setWindowTitle("Help")
        dlg.resize(520, 400)
        lay = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(md)
        lay.addWidget(browser)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn, alignment=Qt.AlignRight)
        dlg.exec()

    def _load_file(self, path: str) -> None:
        logger.debug("LOAD: %s", path)
        self._teardown_graph()

        self._wav_path = path

        info = sf.info(path)
        self._sample_rate = info.samplerate
        self._total_frames = info.frames
        self._channels = info.channels

        pg.set_sample_rate(self._sample_rate)

        # --- Stem detection ---
        stem_hits = find_stems(path)
        self._stem_paths = [p for p, _ in stem_hits]
        self._stem_names = [n for _, n in stem_hits]
        self._stem_muted = [False] * len(self._stem_paths)
        self._stem_solo = [False] * len(self._stem_paths)
        self._stem_volumes = [1.0] * len(self._stem_paths)
        if self._stem_paths:
            logger.debug(
                "Found %d stem(s): %s", len(self._stem_paths), self._stem_names
            )

        # --- Update view combo (suppress signal during rebuild) ---
        self._view_select.blockSignals(True)
        current_view = self._view_select.currentText()
        self._view_select.clear()
        self._view_select.addItems(["Waveform", "Spectrogram"])
        if self._stem_paths:
            self._view_select.addItem("Stems")
        if self._stem_paths:
            self._view_select.setCurrentText("Stems")
            self._active_view = "Stems"
        elif current_view in ("Waveform", "Spectrogram"):
            self._view_select.setCurrentText(current_view)
            self._active_view = current_view
        else:
            self._view_select.setCurrentText("Waveform")
            self._active_view = "Waveform"
        self._view_select.blockSignals(False)

        # Apply visibility for the chosen view
        self._waveform.setVisible(self._active_view == "Waveform")
        self._spectrogram.setVisible(self._active_view == "Spectrogram")
        self._stems_widget.setVisible(self._active_view == "Stems")
        if self._stem_paths:
            self._stems_widget.setup_stems(self._stem_names)

        normalized_annotations: list[FrameAnnotation] = []
        sidecar_path = resolve_annotation_sidecar_path(path)
        try:
            annotations = load_annotation_sidecar(sidecar_path)
            normalized_annotations = normalize_annotations_to_frames(
                annotations,
                sample_rate=self._sample_rate,
                total_frames=self._total_frames,
            )
            if normalized_annotations:
                logger.debug(
                    "Loaded %d annotations from %s",
                    len(normalized_annotations),
                    sidecar_path,
                )
        except Exception as exc:
            logger.warning("Failed to load annotations from %s: %s", sidecar_path, exc)

        try:
            self._readme_text = load_sidecar_readme(sidecar_path)
        except Exception as exc:
            logger.warning("Failed to load readme from %s: %s", sidecar_path, exc)
            self._readme_text = None

        canvas_w = self._current_viz_width() or 640
        self._waveform.set_peaks(compute_peaks(path, target_width=canvas_w))
        self._waveform.set_annotations(normalized_annotations, self._total_frames)
        self._spectrogram_result = None
        self._spectrogram.set_spectrogram(None)
        self._spectrogram.set_annotations(normalized_annotations, self._total_frames)
        self._spectrogram_stale = True
        if self._active_view == "Spectrogram":
            self._refresh_spectrogram_if_needed(force=True)

        self._build_graph(path)

        name = Path(path).name
        dur_str = self._format_time(self._total_frames)
        stem_info = f"  [{len(self._stem_paths)} stems]" if self._stem_paths else ""
        self._file_label.setText(f"File: {name}  ({dur_str}){stem_info}")

    def _build_graph(self, path: str) -> None:
        if self._stem_paths:
            self._build_stems_graph()
        else:
            self._build_single_file_graph(path)

    def _build_single_file_graph(self, path: str) -> None:
        self._rate_control = ControlPE(initial_value=self._shuttle_rest)
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

    def _build_stems_graph(self) -> None:
        self._stem_rate_controls: list[ControlPE] = []
        self._stem_timewarps = []
        self._stem_gain_controls = []
        stem_outputs = []

        for i, stem_path in enumerate(self._stem_paths):
            rate_ctl = ControlPE(initial_value=self._shuttle_rest)
            wav_pe = WavReaderPE(stem_path)
            timewarp = TimeWarpPE(wav_pe, rate=rate_ctl)
            gain_ctl = ControlPE(initial_value=self._stem_volumes[i])
            stem_out = GainPE(timewarp, gain=gain_ctl)
            self._stem_rate_controls.append(rate_ctl)
            self._stem_timewarps.append(timewarp)
            self._stem_gain_controls.append(gain_ctl)
            stem_outputs.append(stem_out)

        self._rate_control = (
            self._stem_rate_controls[0] if self._stem_rate_controls else None
        )

        # Use the first stem timewarp as the primary for position tracking
        self._timewarp = self._stem_timewarps[0] if self._stem_timewarps else None
        self._wav_pe = None

        mixed = MixPE(*stem_outputs) if len(stem_outputs) > 1 else stem_outputs[0]
        output = GainPE(mixed, gain=0.8)

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
        self._stem_rate_controls = []
        self._stem_timewarps = []
        self._stem_gain_controls = []
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
        if hasattr(self, "_stem_rate_controls") and self._stem_rate_controls:
            for rc in self._stem_rate_controls:
                rc.set_value(rate)
        elif self._rate_control is not None:
            self._rate_control.set_value(rate)
        if rate != 0.0 and not self._playing:
            self._renderer.stream_start(start=self._resume_from)
            self._playing = True
            logger.debug(
                "STREAM_START: rate=%.2f, resume_from=%s", rate, self._resume_from
            )
        elif rate == 0.0 and self._playing:
            self._renderer.stream_stop()
            self._resume_from = self._renderer.stream_position
            self._playing = False
            logger.debug("STREAM_STOP: resume_from=%s", self._resume_from)

    def _on_play(self) -> None:
        logger.debug("PLAY: playing=%s, rate=%.2f", self._playing, self._rate)
        self._shuttle_rest = 1.0
        self._shuttle.set_value(self._rate_to_slider(1.0))
        self._set_rate(1.0)

    def _on_pause(self) -> None:
        logger.debug("PAUSE: playing=%s, rate=%.2f", self._playing, self._rate)
        self._shuttle_rest = 0.0
        self._shuttle.set_value(0.0)
        self._set_rate(0.0)

    def _on_stop(self) -> None:
        logger.debug("STOP: playing=%s, rate=%.2f", self._playing, self._rate)
        if self._renderer is None:
            return
        self._shuttle_rest = 0.0
        self._shuttle.set_value(0.0)
        self._set_rate(0.0)
        self._renderer.stop()
        self._renderer.start()
        self._resume_from = 0
        self._set_play_position(0.0)
        self._spring_timer.stop()

    def _on_beginning(self) -> None:
        logger.debug("BEGINNING: playing=%s", self._playing)
        self._set_play_position(0.0)

    def _on_end(self) -> None:
        logger.debug("END: playing=%s", self._playing)
        if self._total_frames > 0:
            self._set_play_position(float(self._total_frames))

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
        return sign * (normalized**self.SHUTTLE_CURVE) * self.SHUTTLE_MAX

    def _rate_to_slider(self, rate: float) -> float:
        """Map playback rate to slider position (inverse of _slider_to_rate)."""
        if rate == 0.0:
            return 0.0
        sign = 1.0 if rate > 0 else -1.0
        normalized = abs(rate) / self.SHUTTLE_MAX
        return sign * (normalized ** (1.0 / self.SHUTTLE_CURVE)) * self.SHUTTLE_MAX

    def _on_shuttle_change(self, val: float) -> None:
        if abs(val) < self.SHUTTLE_SNAP_ZERO:
            val = 0.0
            self._shuttle.set_value(val)
        self._set_rate(self._slider_to_rate(val))

    def _on_shuttle_press(self) -> None:
        self._spring_timer.stop()

    def _on_shuttle_release(self) -> None:
        self._set_rate(self._shuttle_rest)
        self._spring_timer.start()

    def _spring_back_tick(self) -> None:
        target = self._rate_to_slider(self._shuttle_rest)
        current = self._shuttle.value()
        diff = target - current
        if abs(diff) < 0.05:
            self._shuttle.set_value(target)
            self._spring_timer.stop()
            return
        new_val = current + diff * self.SPRING_FACTOR
        self._shuttle.set_value(new_val)

    # ------------------------------------------------------------------
    # Waveform scrubbing
    # ------------------------------------------------------------------

    def _on_scrub_start(self, frac: float) -> None:
        if self._total_frames == 0 or (
            self._timewarp is None and not self._stem_timewarps
        ):
            return
        target = int(frac * self._total_frames)
        if not self._playing:
            self._scrubbing = True
            self._set_rate(1.0)
        self._set_play_position(float(target))

    def _on_scrub_move(self, frac: float) -> None:
        if self._total_frames == 0:
            return
        self._set_play_position(float(int(frac * self._total_frames)))

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
            new_width = self._current_viz_width()
            if new_width > 10:
                self._waveform.set_peaks(
                    compute_peaks(self._wav_path, target_width=new_width)
                )
                self._spectrogram_stale = True
                if self._active_view == "Spectrogram":
                    self._refresh_spectrogram_if_needed()

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
            self._spectrogram.set_playhead(pos / self._total_frames)
            self._shuttle.set_rate_display(self._rate)
            pos_str = self._format_time(max(0, pos))
            samples = int(max(0, pos))
            self._pos_label.setText(f"Position: {pos_str} ({samples} samples)")

    # ------------------------------------------------------------------
    # Stems helpers
    # ------------------------------------------------------------------

    def _set_play_position(self, pos: float) -> None:
        """Set playback position on all active timewarp PEs."""
        if self._stem_timewarps:
            for tw in self._stem_timewarps:
                tw._pos = pos
        elif self._timewarp is not None:
            self._timewarp._pos = pos

    def _compute_stem_gains(self) -> list[float]:
        """Return effective gain for each stem (honouring mute/solo state)."""
        any_solo = any(self._stem_solo)
        gains: list[float] = []
        for i in range(len(self._stem_paths)):
            if self._stem_muted[i]:
                gains.append(0.0)
            elif any_solo and not self._stem_solo[i]:
                gains.append(0.0)
            else:
                gains.append(self._stem_volumes[i])
        return gains

    def _apply_stem_gains(self) -> None:
        """Push current effective gains to all stem ControlPEs."""
        gains = self._compute_stem_gains()
        for gain_ctl, g in zip(self._stem_gain_controls, gains):
            gain_ctl.set_value(g)

    def _on_stem_mute_changed(self, index: int, muted: bool) -> None:
        if 0 <= index < len(self._stem_muted):
            self._stem_muted[index] = muted
            self._apply_stem_gains()

    def _on_stem_solo_changed(self, index: int, soloed: bool) -> None:
        if 0 <= index < len(self._stem_solo):
            self._stem_solo[index] = soloed
            self._apply_stem_gains()

    def _on_stem_volume_changed(self, index: int, vol: float) -> None:
        if 0 <= index < len(self._stem_volumes):
            self._stem_volumes[index] = vol
            self._apply_stem_gains()

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
        if self._delete_on_close and self._wav_path is not None:
            try:
                os.remove(self._wav_path)
            except OSError:
                pass
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="pygmu2 Jog/Shuttle Player (Qt)")
    parser.add_argument(
        "file", nargs="?", default=None, help="Path to a WAV file to open on startup"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging to stderr"
    )
    parser.add_argument(
        "--delete-on-close",
        action="store_true",
        help="Delete the WAV file when the player closes",
    )
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
    if APP_ICON_PATH.is_file():
        icon = QIcon(str(APP_ICON_PATH))
        app.setWindowIcon(icon)
    else:
        icon = None
    window = JogShuttleApp(initial_path=args.file, delete_on_close=args.delete_on_close)
    if icon is not None:
        window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
