"""Label-blind, participant-held-out natural-stream prediction design audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .io.wav_frames import read_pcm_wav
from .structured_prediction import (
    LocalAcousticCodebook,
    LocalTransitionPredictor,
    _spectral_signature,
    _temporal_median_spectral_signature,
)


@dataclass(frozen=True)
class NaturalPredictionCandidate:
    signature_mode: str
    frame_ms: int
    horizon_frames: int
    calibration_ms: int = 2000
    calibration_std_floor: float = 0.30

    def __post_init__(self) -> None:
        if self.signature_mode not in {"full_spectrum", "temporal_median_60_30"}:
            raise ValueError("unknown natural-stream signature mode")
        if self.frame_ms < 100 or self.horizon_frames < 1:
            raise ValueError("invalid natural-stream temporal geometry")
        if self.calibration_ms < self.frame_ms * 4:
            raise ValueError("natural-stream calibration requires four frames")
        if self.calibration_std_floor <= 0.0:
            raise ValueError("calibration standard-deviation floor must be positive")


def _clip_observations(
    path: Path, candidate: NaturalPredictionCandidate
) -> np.ndarray:
    samples, sample_rate = read_pcm_wav(path)
    frame = int(round(candidate.frame_ms * sample_rate / 1000.0))
    if len(samples) < frame:
        raise ValueError(f"natural clip is shorter than one frame: {path}")
    extractor = (
        _temporal_median_spectral_signature
        if candidate.signature_mode == "temporal_median_60_30"
        else _spectral_signature
    )
    observations = np.asarray(
        [
            extractor(samples[start : start + frame], sample_rate)
            for start in range(0, len(samples) - frame + 1, frame)
        ],
        dtype=np.float64,
    )
    calibration_count = max(4, candidate.calibration_ms // candidate.frame_ms)
    if len(observations) <= calibration_count + candidate.horizon_frames:
        raise ValueError(f"natural clip is too short after calibration: {path}")
    calibration = observations[:calibration_count]
    normalized = (observations - np.mean(calibration, axis=0)) / np.maximum(
        np.std(calibration, axis=0), candidate.calibration_std_floor
    )
    normalized /= np.maximum(
        np.linalg.norm(normalized, axis=1, keepdims=True), 1e-6
    )
    if not bool(np.all(np.isfinite(normalized))):
        raise ValueError(f"non-finite natural observations: {path}")
    return normalized


def _farthest_four(observations: np.ndarray) -> np.ndarray:
    if len(observations) < 4:
        raise ValueError("four calibration observations are required")
    selected = [0]
    while len(selected) < 4:
        distances = np.min(
            np.sum(
                (
                    observations[:, np.newaxis, :]
                    - observations[np.newaxis, selected, :]
                )
                ** 2,
                axis=2,
            ),
            axis=1,
        )
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    return observations[selected]


def _accuracy(actual: list[int], predicted: list[int]) -> float:
    if not actual or len(actual) != len(predicted):
        raise ValueError("natural-stream accuracy needs equal non-empty sequences")
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def evaluate_natural_prediction_candidate(
    *,
    root: str | Path,
    participants: Iterable[str],
    candidate: NaturalPredictionCandidate,
    condition_suffixes: tuple[str, ...] = ("B", "D"),
) -> dict[str, object]:
    """Evaluate one candidate with both clips from each participant held out."""

    root_path = Path(root)
    participant_ids = tuple(participants)
    if len(participant_ids) < 3 or len(set(participant_ids)) != len(participant_ids):
        raise ValueError("natural audit requires at least three unique participants")
    calibration_count = max(4, candidate.calibration_ms // candidate.frame_ms)
    clips: dict[str, list[tuple[str, np.ndarray]]] = {}
    for participant in participant_ids:
        clips[participant] = []
        for suffix in condition_suffixes:
            path = root_path / f"{participant}{suffix}.wav"
            if not path.exists():
                raise FileNotFoundError(path)
            clips[participant].append(
                (suffix, _clip_observations(path, candidate))
            )

    folds = []
    all_improvements: list[float] = []
    all_occupancies: list[int] = []
    for held_participant in participant_ids:
        development = tuple(
            participant for participant in participant_ids
            if participant != held_participant
        )
        first_observations = clips[development[0]][0][1]
        codebook = LocalAcousticCodebook(
            _farthest_four(first_observations[:calibration_count])
        )
        for participant in development:
            for _suffix, observations in clips[participant]:
                for observation in observations[calibration_count:]:
                    codebook.observe(observation)

        transition = LocalTransitionPredictor()
        development_states: list[int] = []
        for participant in development:
            for _suffix, observations in clips[participant]:
                states = [
                    codebook.assign(observation)
                    for observation in observations[calibration_count:]
                ]
                development_states.extend(states)
                for index in range(len(states) - candidate.horizon_frames):
                    transition.counts[
                        states[index], states[index + candidate.horizon_frames]
                    ] += 1.0
        unigram = int(
            np.argmax(np.bincount(development_states, minlength=4))
        )

        clip_reports = []
        for suffix, observations in clips[held_participant]:
            states = [
                codebook.assign(observation)
                for observation in observations[calibration_count:]
            ]
            horizon = candidate.horizon_frames
            actual = states[horizon:]
            learned = [transition.predict(state) for state in states[:-horizon]]
            distribution = np.bincount(
                development_states, minlength=4
            ).astype(np.float64)
            distribution /= float(np.sum(distribution))
            moving = []
            for current in states[:-horizon]:
                distribution *= 0.5
                distribution[current] += 0.5
                moving.append(int(np.argmax(distribution)))
            learned_accuracy = _accuracy(actual, learned)
            baselines = {
                "persistence": _accuracy(actual, states[:-horizon]),
                "development_unigram": _accuracy(actual, [unigram] * len(actual)),
                "causal_moving_state": _accuracy(actual, moving),
            }
            best_baseline = max(baselines.values())
            improvement = learned_accuracy - best_baseline
            occupancy = len(set(states))
            all_improvements.append(improvement)
            all_occupancies.append(occupancy)
            clip_reports.append(
                {
                    "clip_id": f"{held_participant}{suffix}",
                    "condition_suffix": suffix,
                    "prediction_count": len(actual),
                    "state_occupancy": occupancy,
                    "transition_accuracy": learned_accuracy,
                    "baselines": baselines,
                    "best_baseline_accuracy": best_baseline,
                    "accuracy_improvement": improvement,
                }
            )
        folds.append(
            {
                "held_participant": held_participant,
                "development_participants": list(development),
                "participant_overlap_count": 0,
                "clips": clip_reports,
            }
        )

    wins = sum(improvement > 0.0 for improvement in all_improvements)
    gate = {
        "minimum_six_of_eight_clip_wins": wins >= 6,
        "minimum_mean_accuracy_improvement": float(np.mean(all_improvements))
        >= 0.05,
        "maximum_worst_clip_regret": min(all_improvements) >= -0.05,
        "minimum_each_clip_state_occupancy": min(all_occupancies) >= 3,
    }
    return {
        "candidate": {
            "signature_mode": candidate.signature_mode,
            "frame_ms": candidate.frame_ms,
            "horizon_frames": candidate.horizon_frames,
            "horizon_ms": candidate.frame_ms * candidate.horizon_frames,
            "calibration_ms": candidate.calibration_ms,
            "calibration_std_floor": candidate.calibration_std_floor,
        },
        "participant_count": len(participant_ids),
        "clip_count": len(all_improvements),
        "labels_consumed_by_learner": False,
        "wins": wins,
        "mean_accuracy_improvement": float(np.mean(all_improvements)),
        "worst_clip_improvement": min(all_improvements),
        "minimum_clip_state_occupancy": min(all_occupancies),
        "design_gate": gate,
        "design_gate_passed": all(gate.values()),
        "folds": folds,
    }


def audit_natural_prediction_grid(
    *,
    root: str | Path,
    participants: Iterable[str],
    frame_ms_values: tuple[int, ...] = (100, 200, 400),
    horizon_values: tuple[int, ...] = (1, 2, 4),
    signature_modes: tuple[str, ...] = (
        "full_spectrum",
        "temporal_median_60_30",
    ),
) -> dict[str, object]:
    """Run the fixed spent-development natural-stream design grid."""

    reports = [
        evaluate_natural_prediction_candidate(
            root=root,
            participants=participants,
            candidate=NaturalPredictionCandidate(mode, frame_ms, horizon),
        )
        for mode in signature_modes
        for frame_ms in frame_ms_values
        for horizon in horizon_values
    ]
    best = max(
        reports,
        key=lambda report: (
            bool(report["design_gate_passed"]),
            int(report["wins"]),
            float(report["mean_accuracy_improvement"]),
            float(report["worst_clip_improvement"]),
        ),
    )
    return {
        "mode": "natural_stream_prediction_design_audit",
        "evidence_class": "spent_development_real_audio",
        "promotion_allowed": False,
        "candidate_count": len(reports),
        "advanced_candidate_count": sum(
            bool(report["design_gate_passed"]) for report in reports
        ),
        "best_candidate": best,
        "reports": reports,
        "verdict": (
            "design_candidate_available"
            if any(report["design_gate_passed"] for report in reports)
            else "design_gate_failed_no_fresh_audio"
        ),
    }
