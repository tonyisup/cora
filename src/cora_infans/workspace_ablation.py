"""Synthetic ablations for flat, windowed, and theta-gamma event handling."""

from __future__ import annotations

from dataclasses import dataclass

from .pipeline import WorkspaceEvent
from .theta_gamma import ThetaGammaScheduler
from .workspace import EventWorkspace


@dataclass(frozen=True)
class AblationEvent:
    timestamp_ms: int
    kind: str
    origin: str
    expected_usable: bool
    expected_self_audio: bool = False
    efference_id: str | None = None

    def workspace_event(self) -> WorkspaceEvent:
        state: dict[str, int | float | str | list[float]] = {
            "timestamp_ms": self.timestamp_ms,
            "focus_label": self.origin,
            "origin": self.origin,
        }
        if self.expected_self_audio:
            state["expected_self_audio"] = True
        if self.efference_id is not None:
            state["efference_id"] = self.efference_id
        return WorkspaceEvent(
            kind=self.kind,
            saliency_score=1.0,
            workspace_state=state,
        )


def _scenario_events(name: str) -> tuple[AblationEvent, ...]:
    if name == "aligned_self_echo":
        return (
            AblationEvent(25, "salient_audio", "world", True),
            AblationEvent(187, "motor_intention", "motor", False, efference_id="m0"),
            AblationEvent(188, "salient_audio", "self_echo", False, True, "m0"),
            AblationEvent(225, "salient_audio", "world", True),
            AblationEvent(387, "motor_intention", "motor", False, efference_id="m1"),
            AblationEvent(388, "salient_audio", "self_echo", False, True, "m1"),
        )
    if name == "legitimate_world_overlap":
        return (
            AblationEvent(25, "salient_audio", "world", True),
            AblationEvent(187, "motor_intention", "motor", False, efference_id="m0"),
            AblationEvent(188, "salient_audio", "world", True),
        )
    raise ValueError(f"unknown ablation scenario: {name}")


def _classification(expected: list[bool], predicted: list[bool]) -> dict[str, float | int]:
    tp = sum(want and got for want, got in zip(expected, predicted))
    fp = sum((not want) and got for want, got in zip(expected, predicted))
    tn = sum((not want) and (not got) for want, got in zip(expected, predicted))
    fn = sum(want and (not got) for want, got in zip(expected, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _evaluate_strategy(events: tuple[AblationEvent, ...], strategy: str) -> dict[str, object]:
    expected: list[bool] = []
    audio_events: list[WorkspaceEvent] = []
    if strategy == "flat":
        for specification in events:
            if specification.kind == "salient_audio":
                expected.append(specification.expected_usable)
                audio_events.append(specification.workspace_event())
        predicted = [True] * len(audio_events)
    elif strategy == "fixed_window":
        workspace = EventWorkspace(window_ms=250)
        for specification in events:
            event = specification.workspace_event()
            workspace.push(event)
            if specification.kind == "salient_audio":
                expected.append(specification.expected_usable)
                audio_events.append(event)
        predicted = [True] * len(audio_events)
    elif strategy == "theta_gamma":
        scheduler = ThetaGammaScheduler(theta_ms=200, gamma_ms=25, capacity_cycles=4)
        for specification in events:
            event = specification.workspace_event()
            scheduler.push(event)
            if specification.kind == "salient_audio":
                expected.append(specification.expected_usable)
                audio_events.append(event)
        predicted = [
            event.workspace_state.get("self_suppressed") is not True
            for event in audio_events
        ]
    else:
        raise ValueError(f"unknown workspace strategy: {strategy}")
    return {
        "strategy": strategy,
        "metrics": _classification(expected, predicted),
        "expected_usable": expected,
        "predicted_usable": predicted,
    }


def run_workspace_ablation() -> dict[str, object]:
    """Expose both the scheduler's benefit and its legitimate-overlap failure."""

    scenarios: dict[str, object] = {}
    for name in ("aligned_self_echo", "legitimate_world_overlap"):
        events = _scenario_events(name)
        scenarios[name] = {
            strategy: _evaluate_strategy(events, strategy)
            for strategy in ("flat", "fixed_window", "theta_gamma")
        }
    echo = scenarios["aligned_self_echo"]  # type: ignore[assignment]
    overlap = scenarios["legitimate_world_overlap"]  # type: ignore[assignment]
    echo_improves = (
        echo["theta_gamma"]["metrics"]["f1"] > echo["flat"]["metrics"]["f1"]  # type: ignore[index]
    )
    overlap_recall = overlap["theta_gamma"]["metrics"]["recall"]  # type: ignore[index]
    criterion_passed = echo_improves and overlap_recall >= 0.9
    return {
        "mode": "synthetic_workspace_ablation",
        "evidence_scope": "synthetic_known_timing_only",
        "scenarios": scenarios,
        "criteria": {
            "aligned_self_echo_f1_improves_over_flat": echo_improves,
            "legitimate_overlap_recall_at_least_0_9": overlap_recall >= 0.9,
        },
        "criterion_passed": criterion_passed,
        "interpretation": (
            "scheduler requires verified self-audio identity in addition to phase "
            "coincidence and passes both synthetic ablation scenarios"
            if criterion_passed
            else "scheduler has not passed both synthetic benefit and collision-safety criteria"
        ),
    }
