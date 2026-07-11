"""Causal event-transition prediction for label-blind natural speech streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .io.wav_frames import read_pcm_wav
from .structured_prediction import LocalTransitionPredictor, _spectral_signature


@dataclass(frozen=True)
class NaturalEventConfig:
    frame_ms: int = 50
    calibration_ms: int = 2000
    quiet_quantile: float = 0.40
    novelty_quantile: float = 0.75

    def __post_init__(self) -> None:
        if self.frame_ms < 25 or self.calibration_ms < self.frame_ms * 4:
            raise ValueError("invalid natural event timing")
        if not 0.0 < self.quiet_quantile < 1.0:
            raise ValueError("quiet quantile must be in (0, 1)")
        if not 0.0 < self.novelty_quantile < 1.0:
            raise ValueError("novelty quantile must be in (0, 1)")


def extract_natural_event_states(
    path: str | Path, config: NaturalEventConfig
) -> tuple[list[int], dict[str, float]]:
    """Map frames causally to quiet/onset/stable/change after local calibration."""

    samples, sample_rate = read_pcm_wav(path)
    frame = int(round(config.frame_ms * sample_rate / 1000.0))
    signatures = []
    rms = []
    for start in range(0, len(samples) - frame + 1, frame):
        segment = samples[start : start + frame]
        signatures.append(_spectral_signature(segment, sample_rate))
        rms.append(float(np.sqrt(np.mean(segment * segment))))
    calibration_count = max(4, config.calibration_ms // config.frame_ms)
    if len(signatures) <= calibration_count + 1:
        raise ValueError(f"natural event clip is too short: {path}")
    signatures_array = np.asarray(signatures, dtype=np.float64)
    log_rms = np.log(np.maximum(np.asarray(rms, dtype=np.float64), 1e-9))
    novelty = np.concatenate(
        (
            np.asarray([0.0]),
            np.linalg.norm(np.diff(signatures_array, axis=0), axis=1),
        )
    )
    quiet_threshold = float(
        np.quantile(log_rms[:calibration_count], config.quiet_quantile)
    )
    novelty_threshold = float(
        np.quantile(
            novelty[1:calibration_count], config.novelty_quantile
        )
    )
    states = []
    previous_quiet = True
    for energy, change in zip(log_rms, novelty):
        quiet = energy < quiet_threshold
        if quiet:
            state = 0
        elif previous_quiet:
            state = 1
        elif change >= novelty_threshold:
            state = 3
        else:
            state = 2
        states.append(state)
        previous_quiet = quiet
    return states[calibration_count:], {
        "quiet_log_rms_threshold": quiet_threshold,
        "novelty_distance_threshold": novelty_threshold,
    }


def _accuracy(actual: list[int], predicted: list[int]) -> float:
    if not actual or len(actual) != len(predicted):
        raise ValueError("event accuracy requires equal non-empty sequences")
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def _fit_event_model(
    *,
    root: Path,
    participants: tuple[str, ...],
    config: NaturalEventConfig,
    condition_suffixes: tuple[str, ...],
) -> tuple[LocalTransitionPredictor, list[int]]:
    transition = LocalTransitionPredictor()
    development_states: list[int] = []
    for participant in participants:
        for suffix in condition_suffixes:
            states, _thresholds = extract_natural_event_states(
                root / f"{participant}{suffix}.wav", config
            )
            transition.observe_sequence(states)
            development_states.extend(states)
    return transition, development_states


def _score_event_clips(
    *,
    root: Path,
    participants: tuple[str, ...],
    config: NaturalEventConfig,
    transition: LocalTransitionPredictor,
    development_states: list[int],
    condition_suffixes: tuple[str, ...],
) -> list[dict[str, object]]:
    unigram = int(np.argmax(np.bincount(development_states, minlength=4)))
    reports = []
    for participant in participants:
        for suffix in condition_suffixes:
            states, thresholds = extract_natural_event_states(
                root / f"{participant}{suffix}.wav", config
            )
            actual = states[1:]
            learned = [transition.predict(state) for state in states[:-1]]
            distribution = np.bincount(
                development_states, minlength=4
            ).astype(np.float64)
            distribution /= float(np.sum(distribution))
            moving = []
            for current in states[:-1]:
                distribution *= 0.5
                distribution[current] += 0.5
                moving.append(int(np.argmax(distribution)))
            learned_accuracy = _accuracy(actual, learned)
            baselines = {
                "persistence": _accuracy(actual, states[:-1]),
                "development_unigram": _accuracy(actual, [unigram] * len(actual)),
                "causal_moving_state": _accuracy(actual, moving),
            }
            best_baseline = max(baselines.values())
            reports.append(
                {
                    "clip_id": f"{participant}{suffix}",
                    "participant": participant,
                    "condition_suffix": suffix,
                    "prediction_count": len(actual),
                    "state_occupancy": len(set(states)),
                    "state_counts": np.bincount(states, minlength=4).tolist(),
                    "thresholds": thresholds,
                    "transition_accuracy": learned_accuracy,
                    "baselines": baselines,
                    "best_baseline_accuracy": best_baseline,
                    "accuracy_improvement": learned_accuracy - best_baseline,
                    "prediction_residual_count": sum(
                        expected != observed
                        for expected, observed in zip(learned, actual)
                    ),
                }
            )
    return reports


def _summarize_event_reports(
    reports: list[dict[str, object]],
) -> dict[str, object]:
    improvements = [float(report["accuracy_improvement"]) for report in reports]
    occupancies = [int(report["state_occupancy"]) for report in reports]
    wins = sum(improvement > 0.0 for improvement in improvements)
    gate = {
        "minimum_six_of_eight_clip_wins": wins >= 6,
        "minimum_mean_accuracy_improvement": float(np.mean(improvements)) >= 0.05,
        "maximum_worst_clip_regret": min(improvements) >= -0.05,
        "minimum_each_clip_state_occupancy": min(occupancies) >= 3,
        "top_down_prediction_consumed": all(
            int(report["prediction_count"]) > 0 for report in reports
        ),
        "bounded_state_passed": True,
    }
    return {
        "clip_count": len(reports),
        "wins": wins,
        "mean_accuracy_improvement": float(np.mean(improvements)),
        "worst_clip_improvement": min(improvements),
        "minimum_clip_state_occupancy": min(occupancies),
        "bounded_state_scalars": 16,
        "acceptance": gate,
        "all_acceptance_gates_passed": all(gate.values()),
        "clips": reports,
    }


def evaluate_natural_event_leave_one_participant_out(
    *,
    root: str | Path,
    participants: Iterable[str],
    config: NaturalEventConfig,
    condition_suffixes: tuple[str, ...] = ("B", "D"),
) -> dict[str, object]:
    """Spent-development audit with both clips from each participant held out."""

    root_path = Path(root)
    participant_ids = tuple(participants)
    folds = []
    reports = []
    for held in participant_ids:
        development = tuple(p for p in participant_ids if p != held)
        transition, development_states = _fit_event_model(
            root=root_path,
            participants=development,
            config=config,
            condition_suffixes=condition_suffixes,
        )
        clip_reports = _score_event_clips(
            root=root_path,
            participants=(held,),
            config=config,
            transition=transition,
            development_states=development_states,
            condition_suffixes=condition_suffixes,
        )
        reports.extend(clip_reports)
        folds.append(
            {
                "held_participant": held,
                "development_participants": list(development),
                "participant_overlap_count": 0,
                "clips": clip_reports,
            }
        )
    return {
        "mode": "natural_event_prediction_leave_one_participant_out",
        "config": asdict(config),
        "labels_consumed_by_learner": False,
        "evidence_class": "spent_development_real_audio",
        "promotion_allowed": False,
        **_summarize_event_reports(reports),
        "folds": folds,
    }


def evaluate_natural_event_cross_fieldsite(
    *,
    development_root: str | Path,
    development_participants: Iterable[str],
    held_out_root: str | Path,
    held_out_participants: Iterable[str],
    config: NaturalEventConfig,
    condition_suffixes: tuple[str, ...] = ("B", "D"),
) -> dict[str, object]:
    """Fit on one named fieldsite and score participant-disjoint fresh clips."""

    development_ids = tuple(development_participants)
    held_out_ids = tuple(held_out_participants)
    if set(development_ids) & set(held_out_ids):
        raise ValueError("natural event transfer participants must be disjoint")
    transition, development_states = _fit_event_model(
        root=Path(development_root),
        participants=development_ids,
        config=config,
        condition_suffixes=condition_suffixes,
    )
    reports = _score_event_clips(
        root=Path(held_out_root),
        participants=held_out_ids,
        config=config,
        transition=transition,
        development_states=development_states,
        condition_suffixes=condition_suffixes,
    )
    summary = _summarize_event_reports(reports)
    return {
        "mode": "natural_event_prediction_cross_fieldsite",
        "config": asdict(config),
        "labels_consumed_by_learner": False,
        "development_participants": list(development_ids),
        "held_out_participants": list(held_out_ids),
        "participant_overlap_count": 0,
        **summary,
        "verdict": (
            "natural_event_transfer_passed"
            if summary["all_acceptance_gates_passed"]
            else "natural_event_transfer_failed"
        ),
    }


def audit_natural_event_grid(
    *,
    root: str | Path,
    participants: Iterable[str],
    frame_ms_values: tuple[int, ...] = (50, 100, 200, 400),
    quiet_quantiles: tuple[float, ...] = (0.10, 0.25, 0.40),
    novelty_quantiles: tuple[float, ...] = (0.60, 0.75, 0.90),
) -> dict[str, object]:
    """Fixed spent-development grid for the causal event observation target."""

    reports = [
        evaluate_natural_event_leave_one_participant_out(
            root=root,
            participants=participants,
            config=NaturalEventConfig(frame_ms, 2000, quiet, novelty),
        )
        for frame_ms in frame_ms_values
        for quiet in quiet_quantiles
        for novelty in novelty_quantiles
    ]
    advanced = [
        report for report in reports
        if bool(report["all_acceptance_gates_passed"])
    ]
    best = max(
        reports,
        key=lambda report: (
            bool(report["all_acceptance_gates_passed"]),
            int(report["wins"]),
            int(report["minimum_clip_state_occupancy"]),
            float(report["mean_accuracy_improvement"]),
            float(report["worst_clip_improvement"]),
        ),
    )
    return {
        "mode": "natural_event_prediction_design_audit",
        "evidence_class": "spent_development_real_audio",
        "promotion_allowed": False,
        "candidate_count": len(reports),
        "advanced_candidate_count": len(advanced),
        "best_candidate": best,
        "reports": reports,
        "verdict": (
            "design_candidate_available"
            if advanced
            else "design_gate_failed_no_fresh_audio"
        ),
    }
