"""
Time-anchored annotation helpers for rendered audio sidecars.

The sidecar schema is YAML-based and designed for interchange with UI tooling
such as jog/shuttle players.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ANNOTATION_SCHEMA_VERSION = 1
ANNOTATION_TIMEBASE_SECONDS = "seconds"


@dataclass(frozen=True)
class AnnotationRecord:
    """Single annotation in timeline seconds."""

    id: str
    onset: float
    extent: float | None = None
    label: str = ""
    kind: str = "event"
    voice: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameAnnotation:
    """Annotation normalized to sample frames."""

    id: str
    onset_frame: int
    end_frame: int | None
    label: str
    kind: str
    voice: str | None
    meta: dict[str, Any]


def resolve_annotation_sidecar_path(audio_path: str | Path, sidecar_path: str | Path | None = None) -> Path:
    """Resolve sidecar path, defaulting to <audio_basename>.annotations.yaml."""
    if sidecar_path is not None:
        return Path(sidecar_path)
    audio = Path(audio_path)
    return audio.with_suffix(".annotations.yaml")


def _annotation_to_dict(record: AnnotationRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": str(record.id),
        "onset": float(record.onset),
        "label": str(record.label),
        "kind": str(record.kind),
    }
    if record.extent is not None:
        payload["extent"] = float(record.extent)
    if record.voice is not None:
        payload["voice"] = str(record.voice)
    if record.meta:
        payload["meta"] = dict(record.meta)
    return payload


def _annotation_from_dict(entry: dict[str, Any], index: int) -> AnnotationRecord:
    onset = float(entry["onset"])
    extent = entry.get("extent")
    extent_val = float(extent) if extent is not None else None
    return AnnotationRecord(
        id=str(entry.get("id", f"a{index:04d}")),
        onset=onset,
        extent=extent_val,
        label=str(entry.get("label", "")),
        kind=str(entry.get("kind", "event")),
        voice=str(entry["voice"]) if entry.get("voice") is not None else None,
        meta=dict(entry.get("meta", {})),
    )


def write_annotation_sidecar(
    audio_path: str | Path,
    annotations: list[AnnotationRecord],
    sample_rate: int,
    total_frames: int,
    sidecar_path: str | Path | None = None,
) -> Path:
    """Write annotations to YAML sidecar and return the written path."""
    out_path = resolve_annotation_sidecar_path(audio_path, sidecar_path=sidecar_path)
    payload: dict[str, Any] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "timebase": ANNOTATION_TIMEBASE_SECONDS,
        "audio_file": Path(audio_path).name,
        "sample_rate": int(sample_rate),
        "total_frames": int(total_frames),
        "annotations": [_annotation_to_dict(a) for a in annotations],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
    return out_path


def load_annotation_sidecar(path: str | Path) -> list[AnnotationRecord]:
    """Load sidecar annotations from YAML, returning an empty list when missing."""
    sidecar_path = Path(path)
    if not sidecar_path.exists():
        return []
    with sidecar_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    annotations_payload = payload.get("annotations", [])
    if not isinstance(annotations_payload, list):
        raise ValueError("Invalid annotation sidecar: 'annotations' must be a list.")
    records: list[AnnotationRecord] = []
    for index, entry in enumerate(annotations_payload):
        if not isinstance(entry, dict):
            raise ValueError("Invalid annotation sidecar: each annotation must be a mapping.")
        records.append(_annotation_from_dict(entry, index=index))
    return records


def normalize_annotations_to_frames(
    annotations: list[AnnotationRecord],
    sample_rate: int,
    total_frames: int,
) -> list[FrameAnnotation]:
    """Clamp and convert second-domain annotations to frame-domain annotations."""
    sr = max(1, int(sample_rate))
    max_frame = max(0, int(total_frames))
    normalized: list[FrameAnnotation] = []
    for record in annotations:
        onset_frame = int(round(record.onset * sr))
        onset_frame = min(max(0, onset_frame), max_frame)

        end_frame: int | None = None
        if record.extent is not None:
            extent_frames = int(round(record.extent * sr))
            if extent_frames > 0:
                unclamped_end = onset_frame + extent_frames
                end_frame = min(max_frame, max(onset_frame, unclamped_end))
                if end_frame == onset_frame:
                    end_frame = None

        normalized.append(
            FrameAnnotation(
                id=record.id,
                onset_frame=onset_frame,
                end_frame=end_frame,
                label=record.label,
                kind=record.kind,
                voice=record.voice,
                meta=record.meta,
            )
        )
    return normalized
