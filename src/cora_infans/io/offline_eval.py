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
from typing import Callable, Literal

ScoreMode = Literal["peak_rms", "spatial_contrast"]
REAL_AUDIO_SOURCES = frozenset({"pcm_wav", "recorded_npy"})


@dataclass(frozen=True)
class FrameObservation:
    """One measured input frame used by the corpus evaluator.

    Annotations never appear here. A provider must derive every observation from
    the audio itself or from an explicitly identified synthetic fixture.
    """

    timestamp_ms: int
    channel_energies: list[float]
    features: dict[str, float | None] | None = None
    source: str = "injected"
    sample_rate: int | None = None
    channel_count: int | None = None


FrameInput = tuple[int, list[float]] | FrameObservation
# The callable receives the clip and configured frame size. Tuple frames remain
# supported for small unit tests; real providers should return FrameObservation.
FrameProvider = Callable[["CorpusClip", int], list[FrameInput]]


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
    true_positive_frames: int
    false_positive_frames: int
    true_negative_frames: int
    false_negative_frames: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    score_mode: ScoreMode
    positive_labels: list[str] | None
    input_source: str
    sample_rate: int | None
    channel_count: int | None
    label_frame_counts: dict[str, int]
    feature_summary: dict[str, dict[str, float | int]]

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
    split: str = "unspecified"
    sample_rate: int | None = None


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
        split_by_id: dict[str, str] = {}
        path_by_id: dict[str, str] = {}
        sample_rate_by_id: dict[str, int | None] = {}
        clips: list[CorpusClip] = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = float(row["start_s"])
                end = float(row["end_s"])
                label = row["label"]
                clip_id = row.get("clip_id") or csv_path.stem
                split = (row.get("split") or "unspecified").strip()
                if split not in ("dev", "held_out", "unspecified"):
                    raise ValueError(
                        f"invalid split {split!r} for {clip_id}; expected dev or held_out"
                    )
                previous_split = split_by_id.setdefault(clip_id, split)
                if previous_split != split:
                    raise ValueError(f"inconsistent split assignments for {clip_id}")
                source_path = (row.get("source_path") or f"{clip_id}.wav").strip()
                previous_path = path_by_id.setdefault(clip_id, source_path)
                if previous_path != source_path:
                    raise ValueError(f"inconsistent source paths for {clip_id}")
                sample_rate_raw = (row.get("sample_rate") or "").strip()
                sample_rate = int(sample_rate_raw) if sample_rate_raw else None
                by_id.setdefault(clip_id, []).append(Annotation(start, end, label))
                previous_rate = sample_rate_by_id.setdefault(clip_id, sample_rate)
                if previous_rate != sample_rate:
                    raise ValueError(f"inconsistent sample rates for {clip_id}")
        for clip_id, anns in by_id.items():
            wav_path = self.root / path_by_id[clip_id]
            if not wav_path.exists():
                # Strict, honest: do not fabricate clips without audio.
                continue
            clips.append(
                CorpusClip(
                    clip_id=clip_id,
                    wav_path=wav_path,
                    annotations=tuple(anns),
                    split=split_by_id[clip_id],
                    sample_rate=sample_rate_by_id[clip_id],
                )
            )
        return clips


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _normalize_frame(frame: FrameInput) -> FrameObservation:
    if isinstance(frame, FrameObservation):
        return frame
    timestamp_ms, channel_energies = frame
    return FrameObservation(
        timestamp_ms=timestamp_ms,
        channel_energies=channel_energies,
    )


def score_frame(energies: list[float], mode: ScoreMode) -> float:
    if not energies:
        return 0.0
    peak = max(energies)
    if mode == "peak_rms":
        return peak
    if mode == "spatial_contrast":
        return max(0.0, peak - (sum(energies) / len(energies)))
    raise ValueError(f"unknown score mode: {mode}")


class OfflineEvalHarness:
    """Scores frame-level saliency against labeled intervals.

    Label semantics (honest, scope-limited): a labeled interval marks a
    ground-truth acoustic EVENT. Unlabeled regions are ground-truth silence.
    The harness is a saliency/event-presence scorer, not a label classifier --
    ``non_speech`` is still a positive event if it crosses threshold. This keeps
    the scoreboard independent of label vocabulary.
    """

    def __init__(
        self,
        *,
        saliency_threshold: float = 0.02,
        frame_ms: int = 250,
        score_mode: ScoreMode = "peak_rms",
        positive_labels: set[str] | frozenset[str] | None = None,
    ) -> None:
        if saliency_threshold < 0.0:
            raise ValueError("saliency_threshold must be non-negative")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if score_mode not in ("peak_rms", "spatial_contrast"):
            raise ValueError(f"unknown score mode: {score_mode}")
        self.saliency_threshold = saliency_threshold
        self.frame_ms = frame_ms
        self.score_mode = score_mode
        self.positive_labels = (
            frozenset(positive_labels) if positive_labels is not None else None
        )

    def _label_for_frame(self, clip: CorpusClip, t_ms: int) -> str | None:
        t_s = t_ms / 1000.0
        for ann in clip.annotations:
            if ann.start_s <= t_s < ann.end_s:
                return ann.label
        return None

    def _is_positive_label(self, label: str | None) -> bool:
        if self.positive_labels is None:
            return label is not None
        return label in self.positive_labels

    def evaluate_clip(
        self, clip: CorpusClip, frames_for: FrameProvider
    ) -> ClipReport:
        frames = frames_for(clip, self.frame_ms)
        total = len(frames)
        salient = 0
        correct = 0
        matched_salient = 0
        label_counts: dict[str, int] = {}

        observations = [_normalize_frame(frame) for frame in frames]
        tp = fp = tn = fn = 0

        for observation in observations:
            t_ms = observation.timestamp_ms
            energies = observation.channel_energies
            score = score_frame(energies, self.score_mode)
            is_salient = score >= self.saliency_threshold
            label = self._label_for_frame(clip, t_ms)
            is_positive = self._is_positive_label(label)
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1

            if is_salient:
                salient += 1
                # A salient frame is "matched" when it falls inside a labeled,
                # explicitly salient interval. We treat any labeled interval as
                # the ground-truth saliency region for this harness; callers can
                # refine labels (e.g. salient_speech vs non_speech) as needed.
                if is_positive:
                    matched_salient += 1
                    correct += 1
                    tp += 1
                else:
                    fp += 1
            else:
                # Non-salient frames are correct when they fall outside any
                # labeled (salient) interval, i.e. label is None.
                if not is_positive:
                    correct += 1
                    tn += 1
                else:
                    fn += 1

        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
        fpr = _safe_ratio(fp, fp + tn)
        sources = {observation.source for observation in observations}
        sample_rates = {
            observation.sample_rate
            for observation in observations
            if observation.sample_rate is not None
        }
        channel_counts = {
            observation.channel_count
            for observation in observations
            if observation.channel_count is not None
        }
        feature_values: dict[str, list[float]] = {}
        for observation in observations:
            for name, value in (observation.features or {}).items():
                if value is not None:
                    feature_values.setdefault(name, []).append(float(value))
        feature_summary = {
            name: {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            for name, values in feature_values.items()
            if values
        }

        return ClipReport(
            clip_id=clip.clip_id,
            frame_ms=self.frame_ms,
            saliency_threshold=self.saliency_threshold,
            total_frames=total,
            salient_frames=salient,
            matched_salient_frames=matched_salient,
            correct_frames=correct,
            accuracy=(correct / total) if total else 0.0,
            true_positive_frames=tp,
            false_positive_frames=fp,
            true_negative_frames=tn,
            false_negative_frames=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            false_positive_rate=fpr,
            score_mode=self.score_mode,
            positive_labels=(
                sorted(self.positive_labels)
                if self.positive_labels is not None
                else None
            ),
            input_source=next(iter(sources)) if len(sources) == 1 else "mixed",
            sample_rate=next(iter(sample_rates)) if len(sample_rates) == 1 else None,
            channel_count=(
                next(iter(channel_counts)) if len(channel_counts) == 1 else None
            ),
            label_frame_counts=label_counts,
            feature_summary=feature_summary,
        )

    def evaluate_corpus(
        self, clips: list[CorpusClip], frames_for: FrameProvider
    ) -> dict[str, object]:
        per_clip: list[ClipReport] = [self.evaluate_clip(c, frames_for) for c in clips]
        total_frames = sum(r.total_frames for r in per_clip)
        correct = sum(r.correct_frames for r in per_clip)
        salient = sum(r.salient_frames for r in per_clip)
        matched = sum(r.matched_salient_frames for r in per_clip)
        tp = sum(r.true_positive_frames for r in per_clip)
        fp = sum(r.false_positive_frames for r in per_clip)
        tn = sum(r.true_negative_frames for r in per_clip)
        fn = sum(r.false_negative_frames for r in per_clip)
        return {
            "clip_count": len(clips),
            "total_frames": total_frames,
            "salient_frames": salient,
            "matched_salient_frames": matched,
            "correct_frames": correct,
            "accuracy": (correct / total_frames) if total_frames else 0.0,
            "true_positive_frames": tp,
            "false_positive_frames": fp,
            "true_negative_frames": tn,
            "false_negative_frames": fn,
            "precision": _safe_ratio(tp, tp + fp),
            "recall": _safe_ratio(tp, tp + fn),
            "f1": _safe_ratio(2 * tp, 2 * tp + fp + fn),
            "false_positive_rate": _safe_ratio(fp, fp + tn),
            "score_mode": self.score_mode,
            "positive_labels": (
                sorted(self.positive_labels)
                if self.positive_labels is not None
                else None
            ),
            "saliency_threshold": self.saliency_threshold,
            "frame_ms": self.frame_ms,
            "evidence_kind": (
                "real_audio"
                if per_clip and all(r.input_source in REAL_AUDIO_SOURCES for r in per_clip)
                else "synthetic_or_injected"
            ),
            "per_clip": [r.to_dict() for r in per_clip],
        }


def _candidate_thresholds(
    clips: list[CorpusClip],
    frames_by_clip: dict[str, list[FrameInput]],
    mode: ScoreMode,
) -> list[float]:
    scores = {
        score_frame(_normalize_frame(frame).channel_energies, mode)
        for clip in clips
        for frame in frames_by_clip[clip.clip_id]
    }
    if not scores:
        return [0.0]
    ordered = sorted(scores)
    return [0.0, *ordered, ordered[-1] + 1e-12]


def compare_saliency_baselines(
    clips: list[CorpusClip],
    frames_for: FrameProvider,
    *,
    frame_ms: int = 250,
    positive_labels: set[str] | frozenset[str] | None = None,
) -> dict[str, object]:
    """Tune simple thresholds on development clips and score held-out clips.

    This is deliberately limited to transparent baselines. Feature-based
    parentese scoring belongs in a later stage and must beat these held-out
    results rather than replacing them without comparison.
    """

    dev_clips = [clip for clip in clips if clip.split == "dev"]
    held_out_clips = [clip for clip in clips if clip.split == "held_out"]
    if not dev_clips or not held_out_clips:
        raise ValueError(
            "baseline comparison requires at least one dev and one held_out clip"
        )
    frames_by_clip = {clip.clip_id: frames_for(clip, frame_ms) for clip in clips}

    def cached_frames(clip: CorpusClip, requested_frame_ms: int) -> list[FrameInput]:
        if requested_frame_ms != frame_ms:
            raise ValueError("cached frame size mismatch")
        return frames_by_clip[clip.clip_id]

    comparisons: dict[str, object] = {}
    for mode in ("peak_rms", "spatial_contrast"):
        candidates = _candidate_thresholds(dev_clips, frames_by_clip, mode)
        best_threshold = 0.0
        best_dev: dict[str, object] | None = None
        best_key: tuple[float, float, float, float] | None = None
        for threshold in candidates:
            report = OfflineEvalHarness(
                saliency_threshold=threshold,
                frame_ms=frame_ms,
                score_mode=mode,
                positive_labels=positive_labels,
            ).evaluate_corpus(dev_clips, cached_frames)
            key = (
                float(report["f1"]),
                -float(report["false_positive_rate"]),
                float(report["accuracy"]),
                threshold,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = threshold
                best_dev = report
        assert best_dev is not None
        held_out = OfflineEvalHarness(
            saliency_threshold=best_threshold,
            frame_ms=frame_ms,
            score_mode=mode,
            positive_labels=positive_labels,
        ).evaluate_corpus(held_out_clips, cached_frames)
        comparisons[mode] = {
            "selected_threshold": best_threshold,
            "selection_metric": "max_dev_f1_then_min_fpr_then_accuracy",
            "dev": best_dev,
            "held_out": held_out,
        }

    feature_names = (
        "pitch_hz",
        "pitch_confidence",
        "spectral_centroid_hz",
        "spectral_flux",
    )
    for feature_name in feature_names:
        mapped_frames: dict[str, list[FrameInput]] = {}
        has_feature = False
        for clip in clips:
            mapped: list[FrameInput] = []
            for frame in frames_by_clip[clip.clip_id]:
                observation = _normalize_frame(frame)
                value = (observation.features or {}).get(feature_name)
                has_feature = has_feature or value is not None
                mapped.append(
                    FrameObservation(
                        timestamp_ms=observation.timestamp_ms,
                        channel_energies=[float(value) if value is not None else 0.0],
                        features=observation.features,
                        source=observation.source,
                        sample_rate=observation.sample_rate,
                        channel_count=observation.channel_count,
                    )
                )
            mapped_frames[clip.clip_id] = mapped
        if not has_feature:
            continue

        def cached_feature_frames(
            clip: CorpusClip, requested_frame_ms: int
        ) -> list[FrameInput]:
            if requested_frame_ms != frame_ms:
                raise ValueError("cached frame size mismatch")
            return mapped_frames[clip.clip_id]

        candidates = _candidate_thresholds(dev_clips, mapped_frames, "peak_rms")
        best_threshold = 0.0
        best_dev = None
        best_key = None
        for threshold in candidates:
            report = OfflineEvalHarness(
                saliency_threshold=threshold,
                frame_ms=frame_ms,
                score_mode="peak_rms",
                positive_labels=positive_labels,
            ).evaluate_corpus(dev_clips, cached_feature_frames)
            key = (
                float(report["f1"]),
                -float(report["false_positive_rate"]),
                float(report["accuracy"]),
                threshold,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = threshold
                best_dev = report
        assert best_dev is not None
        held_out = OfflineEvalHarness(
            saliency_threshold=best_threshold,
            frame_ms=frame_ms,
            score_mode="peak_rms",
            positive_labels=positive_labels,
        ).evaluate_corpus(held_out_clips, cached_feature_frames)
        comparisons[f"feature:{feature_name}"] = {
            "selected_threshold": best_threshold,
            "score_direction": "higher_is_positive",
            "missing_value": 0.0,
            "selection_metric": "max_dev_f1_then_min_fpr_then_accuracy",
            "dev": best_dev,
            "held_out": held_out,
        }

    evidence_kind = (
        "real_audio"
        if all(
            _normalize_frame(frame).source in REAL_AUDIO_SOURCES
            for frames in frames_by_clip.values()
            for frame in frames
        )
        else "synthetic_or_injected"
    )
    return {
        "mode": "saliency_baseline_comparison",
        "evidence_kind": evidence_kind,
        "frame_ms": frame_ms,
        "dev_clip_ids": [clip.clip_id for clip in dev_clips],
        "held_out_clip_ids": [clip.clip_id for clip in held_out_clips],
        "positive_labels": sorted(positive_labels) if positive_labels else None,
        "baselines": comparisons,
    }
