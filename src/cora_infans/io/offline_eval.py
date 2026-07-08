"""Offline corpus + evaluation harness for Cora-Infans.

This is deliberately scope-limited: it turns a folder of recorded audio clips
plus human/Audacity-compatible label CSVs into a scored validation run WITHOUT
requiring the XVF3800 or any live capture. The harness owns no audio decoding
itself -- callers inject a ``frames_for`` callable that yields
``(timestamp_ms, channel_energies)`` per clip. That keeps the scoring path
pure Python and unit-testable, while the real WAV->frames extraction can be
added later (or supplied by the separate recording project).

Label CSV format (Audacity-compatible: tab or comma, header + rows):
    start_s, end_s, label
    0.0, 1.0, salient_speech
    1.0, 2.0, non_speech

Honesty rule (per VISION.md "don't fake it"): the harness only scores what the
frames report. It does not invent labels or smooth over missing audio.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

# A frame is (timestamp_ms, channel_energies). The callable receives the clip
# and the configured frame_ms so callers can slice audio deterministically.
FrameProvider = Callable[[ "CorpusClip", int ], list[tuple[int, list[float]]]]


@dataclass
class ClipReport:
    clip_id: str
    frame_ms: int
    saliency_threshold: float
    total_frames: int
    salient_frames: int
    matched_salient_frames: int
    correct_frames: int
    accuracy: float
    label_frame_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Annotation:
    start_s: float
    end_s: float
    label: str


@dataclass(frozen=True)
class CorpusClip:
    clip_id: str
    wav_path: Path
    annotations: tuple[Annotation, ...]


class CorpusLoader:
    """Loads clip + label pairs from an Audacity-style CSV manifest.

    Each row may carry an optional ``clip_id`` column; otherwise the clip id
    defaults to the CSV file stem. The loader maps a clip id ``<id>`` to
    ``<root>/<id>.wav`` and skips rows whose audio file is missing (honest:
    no fabricated clips).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load_labels(self, csv_path: str | Path) -> list[CorpusClip]:
        csv_path = Path(csv_path)
        by_id: dict[str, list[Annotation]] = {}
        clips: list[CorpusClip] = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = float(row["start_s"])
                end = float(row["end_s"])
                label = row["label"]
                clip_id = row.get("clip_id") or csv_path.stem
                by_id.setdefault(clip_id, []).append(Annotation(start, end, label))
        for clip_id, anns in by_id.items():
            wav_path = self.root / f"{clip_id}.wav"
            if not wav_path.exists():
                # Strict, honest: do not fabricate clips without audio.
                continue
            clips.append(
                CorpusClip(
                    clip_id=clip_id,
                    wav_path=wav_path,
                    annotations=tuple(anns),
                )
            )
        return clips


class OfflineEvalHarness:
    """Scores frame-level saliency against labeled intervals.

    Label semantics (honest, scope-limited): a labeled interval marks a
    ground-truth acoustic EVENT. Unlabeled regions are ground-truth silence.
    The harness is a saliency/event-presence scorer, not a label classifier --
    ``non_speech`` is still a positive event if it crosses threshold. This keeps
    the scoreboard independent of label vocabulary.
    """

    def __init__(self, *, saliency_threshold: float = 0.7, frame_ms: int = 250) -> None:
        self.saliency_threshold = saliency_threshold
        self.frame_ms = frame_ms

    def _label_for_frame(self, clip: CorpusClip, t_ms: int) -> str | None:
        t_s = t_ms / 1000.0
        for ann in clip.annotations:
            if ann.start_s <= t_s < ann.end_s:
                return ann.label
        return None

    def evaluate_clip(
        self, clip: CorpusClip, frames_for: FrameProvider
    ) -> ClipReport:
        frames = frames_for(clip, self.frame_ms)
        total = len(frames)
        salient = 0
        correct = 0
        matched_salient = 0
        label_counts: dict[str, int] = {}

        for t_ms, energies in frames:
            score = max(energies) if energies else 0.0
            is_salient = score >= self.saliency_threshold
            label = self._label_for_frame(clip, t_ms)
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1

            if is_salient:
                salient += 1
                # A salient frame is "matched" when it falls inside a labeled,
                # explicitly salient interval. We treat any labeled interval as
                # the ground-truth saliency region for this harness; callers can
                # refine labels (e.g. salient_speech vs non_speech) as needed.
                if label is not None:
                    matched_salient += 1
                    correct += 1
            else:
                # Non-salient frames are correct when they fall outside any
                # labeled (salient) interval, i.e. label is None.
                if label is None:
                    correct += 1

        return ClipReport(
            clip_id=clip.clip_id,
            frame_ms=self.frame_ms,
            saliency_threshold=self.saliency_threshold,
            total_frames=total,
            salient_frames=salient,
            matched_salient_frames=matched_salient,
            correct_frames=correct,
            accuracy=(correct / total) if total else 0.0,
            label_frame_counts=label_counts,
        )

    def evaluate_corpus(
        self, clips: list[CorpusClip], frames_for: FrameProvider
    ) -> dict[str, object]:
        per_clip: list[ClipReport] = [self.evaluate_clip(c, frames_for) for c in clips]
        total_frames = sum(r.total_frames for r in per_clip)
        correct = sum(r.correct_frames for r in per_clip)
        salient = sum(r.salient_frames for r in per_clip)
        matched = sum(r.matched_salient_frames for r in per_clip)
        return {
            "clip_count": len(clips),
            "total_frames": total_frames,
            "salient_frames": salient,
            "matched_salient_frames": matched,
            "correct_frames": correct,
            "accuracy": (correct / total_frames) if total_frames else 0.0,
            "per_clip": [r.to_dict() for r in per_clip],
        }
