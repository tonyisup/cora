"""Spent real-capture design audit for bounded self/world event binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BindingEvent:
    clip_id: str
    efference_id: str | None
    self_score: float
    speech_ratio: float
    expected_world_usable: bool
    observed_sha256: str


class BoundedEfferenceWorkspace:
    """Retain only the most recent opaque motor/efference bindings."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("binding workspace capacity must be positive")
        self.capacity = capacity
        self._bindings: list[str] = []

    def push(self, efference_id: str) -> None:
        if efference_id in self._bindings:
            self._bindings.remove(efference_id)
        self._bindings.append(efference_id)
        del self._bindings[:-self.capacity]

    def contains(self, efference_id: str | None) -> bool:
        return efference_id is not None and efference_id in self._bindings

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._bindings)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_summary_event(path: Path, *, clip_id: str) -> BindingEvent:
    summary = json.loads((path / "summary.json").read_text())
    observed = path / "observed.npy"
    self_only = bool(summary.get("self_only"))
    world_only = bool(summary.get("world_only"))
    mixed = not self_only and not world_only
    seed = int(summary["seed"])
    return BindingEvent(
        clip_id=clip_id,
        efference_id=(None if world_only else f"self_seed_{seed}"),
        self_score=float(summary["self_score"]),
        speech_ratio=float(summary["speech_ratio"]),
        expected_world_usable=bool(world_only or mixed),
        observed_sha256=_sha256(observed),
    )


def _load_pair_family(root: Path, family: str) -> list[BindingEvent]:
    events = []
    for seed in range(3):
        for kind in ("self_only", "mixed"):
            path = root / family / f"{kind}_seed{seed}"
            events.append(
                _load_summary_event(
                    path,
                    clip_id=f"{family}:{kind}_seed{seed}",
                )
            )
    return events


def _midpoint(maximum_negative: float, minimum_positive: float) -> float:
    if minimum_positive <= maximum_negative:
        raise ValueError("development features do not provide a clean threshold gap")
    return (maximum_negative + minimum_positive) / 2.0


def _classification(
    events: list[BindingEvent], predicted: list[bool]
) -> dict[str, float | int]:
    actual = [event.expected_world_usable for event in events]
    tp = sum(want and got for want, got in zip(actual, predicted))
    fp = sum((not want) and got for want, got in zip(actual, predicted))
    tn = sum((not want) and (not got) for want, got in zip(actual, predicted))
    fn = sum(want and (not got) for want, got in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "world_recall": recall,
        "self_suppression_specificity": specificity,
        "f1": f1,
    }


def _evaluate_strategy(
    events: list[BindingEvent],
    *,
    capacity: int,
    self_threshold: float,
    world_threshold: float,
    strategy: str,
) -> dict[str, object]:
    workspace = BoundedEfferenceWorkspace(capacity)
    for seed in range(3):
        workspace.push(f"self_seed_{seed}")
    predicted = []
    decisions = []
    for event in events:
        bound = workspace.contains(event.efference_id)
        self_present = bound and event.self_score >= self_threshold
        if strategy == "flat":
            usable = True
        elif strategy == "identity_only":
            usable = not self_present
        elif strategy == "identity_plus_world_evidence":
            usable = not self_present or event.speech_ratio >= world_threshold
        else:
            raise ValueError(f"unknown real workspace strategy: {strategy}")
        predicted.append(usable)
        decisions.append(
            {
                "clip_id": event.clip_id,
                "efference_bound": bound,
                "self_present": self_present,
                "bottom_up_world_evidence": event.speech_ratio >= world_threshold,
                "expected_world_usable": event.expected_world_usable,
                "predicted_world_usable": usable,
            }
        )
    return {
        "strategy": strategy,
        "capacity": capacity,
        "retained_bindings": list(workspace.snapshot()),
        "bounded_binding_slots": capacity,
        "threshold_scalars": 2,
        "metrics": _classification(events, predicted),
        "decisions": decisions,
    }


def evaluate_real_workspace_binding_design(
    *,
    root: str | Path,
    development_families: tuple[str, ...] = (
        "m2_calibrated_matrix_20260429_092841",
        "m2_energy_onset_matrix_20260429_093045",
    ),
    evaluation_family: str = "m2_energy_forward_tol75_matrix_20260429_093302",
    world_control: str = "control_world-only_seed0_in3_out5_noise4_20260429_091828",
    capacities: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, object]:
    """Select thresholds on spent families and score a disjoint spent family."""

    root_path = Path(root)
    development = [
        event
        for family in development_families
        for event in _load_pair_family(root_path, family)
    ]
    control = _load_summary_event(
        root_path / world_control,
        clip_id=f"{world_control}:world_only_seed0",
    )
    self_threshold = _midpoint(
        control.self_score,
        min(event.self_score for event in development if event.efference_id),
    )
    world_threshold = _midpoint(
        max(
            event.speech_ratio
            for event in development
            if not event.expected_world_usable
        ),
        min(
            event.speech_ratio
            for event in development
            if event.expected_world_usable
        ),
    )
    evaluation = [*_load_pair_family(root_path, evaluation_family), control]
    reports = {
        str(capacity): {
            strategy: _evaluate_strategy(
                evaluation,
                capacity=capacity,
                self_threshold=self_threshold,
                world_threshold=world_threshold,
                strategy=strategy,
            )
            for strategy in (
                "flat",
                "identity_only",
                "identity_plus_world_evidence",
            )
        }
        for capacity in capacities
    }
    required_capacity = 3
    selected = reports[str(required_capacity)]
    flat_f1 = float(selected["flat"]["metrics"]["f1"])
    identity_recall = float(
        selected["identity_only"]["metrics"]["world_recall"]
    )
    combined_metrics = selected["identity_plus_world_evidence"]["metrics"]
    criteria = {
        "identity_plus_world_evidence_f1_at_least_0_90": (
            float(combined_metrics["f1"]) >= 0.90
        ),
        "f1_improves_over_flat_by_at_least_0_20": (
            float(combined_metrics["f1"]) - flat_f1 >= 0.20
        ),
        "world_recall_at_least_0_90": (
            float(combined_metrics["world_recall"]) >= 0.90
        ),
        "self_suppression_specificity_at_least_0_90": (
            float(combined_metrics["self_suppression_specificity"]) >= 0.90
        ),
        "identity_only_collision_failure_exposed": (
            identity_recall < float(combined_metrics["world_recall"])
        ),
        "capacity_one_worse_than_capacity_three": (
            float(
                reports["1"]["identity_plus_world_evidence"]["metrics"]["f1"]
            )
            < float(combined_metrics["f1"])
        ),
        "capacity_three_plateaus_at_capacity_four": (
            float(
                reports["4"]["identity_plus_world_evidence"]["metrics"]["f1"]
            )
            == float(combined_metrics["f1"])
        ),
    }
    return {
        "mode": "spent_real_capture_workspace_binding_design",
        "evidence_class": "spent_prior_art_real_capture_summary_features",
        "promotion_allowed": False,
        "development_families": list(development_families),
        "evaluation_family": evaluation_family,
        "world_control": world_control,
        "thresholds": {
            "self_identity_score": self_threshold,
            "bottom_up_world_speech_ratio": world_threshold,
            "selected_from_development_only": True,
        },
        "capacity_sweep": reports,
        "selected_capacity": required_capacity,
        "criteria": criteria,
        "all_design_criteria_passed": all(criteria.values()),
        "source_observed_sha256": {
            event.clip_id: event.observed_sha256
            for event in [*development, *evaluation]
        },
        "claim_boundary": (
            "Design evidence using Corollary-produced self-presence and speech "
            "summary features over retained real PCM. It does not recompute a "
            "signal residual, promote H3, or replace "
            "a fresh Cora playback/capture binding task."
        ),
    }
