"""Structured H3 self/world binding protocol with no live I/O wiring."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .io.motor_plan import synthesize_chirp, synthesize_tone
from .real_workspace_binding import BoundedEfferenceWorkspace


@dataclass(frozen=True)
class WorkspaceBindingConfig:
    sample_rate: int = 16000
    alignment_chirp_ms: int = 120
    alignment_silence_ms: int = 180
    stimulus_ms: int = 800
    tail_silence_ms: int = 200
    self_gain: float = 0.12
    world_gain: float = 0.12
    fake_delay_ms: int = 30
    fake_attenuation: float = 0.35
    fake_noise_rms: float = 0.001


@dataclass(frozen=True)
class WorkspaceBindingTrial:
    trial_id: str
    split: str
    condition: str
    seed_index: int
    self_frequency_hz: float | None
    world_frequency_hz: float | None
    efference_id: str | None
    expected_world_usable: bool


SELF_DEVELOPMENT_HZ = (380.0, 520.0, 680.0)
WORLD_DEVELOPMENT_HZ = (1150.0, 1450.0, 1750.0)
SELF_HELD_OUT_HZ = (420.0, 560.0, 720.0)
WORLD_HELD_OUT_HZ = (1250.0, 1550.0, 1850.0)


def build_workspace_binding_trials() -> tuple[WorkspaceBindingTrial, ...]:
    trials = []
    for split, self_frequencies, world_frequencies in (
        ("development", SELF_DEVELOPMENT_HZ, WORLD_DEVELOPMENT_HZ),
        ("held_out", SELF_HELD_OUT_HZ, WORLD_HELD_OUT_HZ),
    ):
        conditions = [
            ("self_only", True, False),
            ("world_only", False, True),
        ]
        if split == "held_out":
            conditions.append(("overlap", True, True))
        for index, (self_frequency, world_frequency) in enumerate(
            zip(self_frequencies, world_frequencies)
        ):
            for condition, has_self, has_world in conditions:
                trials.append(
                    WorkspaceBindingTrial(
                        trial_id=f"{split}_{condition}_{index}",
                        split=split,
                        condition=condition,
                        seed_index=index,
                        self_frequency_hz=self_frequency if has_self else None,
                        world_frequency_hz=world_frequency if has_world else None,
                        efference_id=(
                            f"held_self_{index}"
                            if has_self and split == "held_out"
                            else f"dev_self_{index}" if has_self else None
                        ),
                        expected_world_usable=has_world,
                    )
                )
    return tuple(trials)


def build_workspace_binding_protocol(
    config: WorkspaceBindingConfig | None = None,
) -> dict[str, object]:
    config = config or WorkspaceBindingConfig()
    trials = build_workspace_binding_trials()
    seconds_per_trial = (
        config.alignment_chirp_ms
        + config.alignment_silence_ms
        + config.stimulus_ms
        + config.tail_silence_ms
    ) / 1000.0
    return {
        "mode": "structured_workspace_binding_protocol",
        "config": asdict(config),
        "trial_count": len(trials),
        "seconds_per_trial": seconds_per_trial,
        "maximum_total_playback_seconds": seconds_per_trial * len(trials),
        "trials": [asdict(trial) for trial in trials],
        "strategies": [
            "flat",
            "fixed_self_window",
            "scheduled_identity_plus_world_evidence",
        ],
        "capacity_sweep": [1, 2, 3, 4],
        "label_boundary": (
            "condition and expected_world_usable enter scoring only; the runtime "
            "strategy consumes efference identity and measured world-band evidence"
        ),
        "live_io_wired": False,
    }


def _render_trial(
    trial: WorkspaceBindingTrial,
    config: WorkspaceBindingConfig,
) -> tuple[np.ndarray, tuple[int, int]]:
    chirp = synthesize_chirp(
        250.0,
        3500.0,
        duration_ms=config.alignment_chirp_ms,
        sample_rate=config.sample_rate,
    ) * np.float32(0.12)
    pre = np.zeros(
        int(config.alignment_silence_ms * config.sample_rate / 1000),
        dtype=np.float32,
    )
    stimulus_length = int(config.stimulus_ms * config.sample_rate / 1000)
    stimulus = np.zeros(stimulus_length, dtype=np.float32)
    if trial.self_frequency_hz is not None:
        stimulus += config.self_gain * synthesize_tone(
            trial.self_frequency_hz,
            duration_ms=config.stimulus_ms,
            sample_rate=config.sample_rate,
        )
    if trial.world_frequency_hz is not None:
        stimulus += config.world_gain * synthesize_tone(
            trial.world_frequency_hz,
            duration_ms=config.stimulus_ms,
            sample_rate=config.sample_rate,
        )
    tail = np.zeros(
        int(config.tail_silence_ms * config.sample_rate / 1000),
        dtype=np.float32,
    )
    start = len(chirp) + len(pre)
    return (
        np.concatenate((chirp, pre, np.clip(stimulus, -1.0, 1.0), tail)),
        (start, start + stimulus_length),
    )


def _fake_capture(
    pcm: np.ndarray,
    *,
    seed: int,
    config: WorkspaceBindingConfig,
) -> tuple[np.ndarray, int]:
    delay = int(config.fake_delay_ms * config.sample_rate / 1000)
    capture = np.zeros(len(pcm) + delay + config.sample_rate // 10)
    capture[delay : delay + len(pcm)] = config.fake_attenuation * pcm
    rng = np.random.default_rng(seed)
    capture += rng.normal(0.0, config.fake_noise_rms, len(capture))
    return capture.astype(np.float32), delay


def _world_band_ratio(segment: np.ndarray, sample_rate: int) -> float:
    centered = np.asarray(segment, dtype=np.float64) - float(np.mean(segment))
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequencies = np.fft.rfftfreq(len(centered), 1.0 / sample_rate)
    total = float(np.sum(spectrum[(frequencies >= 250.0) & (frequencies < 2200.0)]))
    world = float(np.sum(spectrum[(frequencies >= 950.0) & (frequencies < 2200.0)]))
    return world / max(total, 1e-18)


def _classification(actual: list[bool], predicted: list[bool]) -> dict[str, float | int]:
    tp = sum(want and got for want, got in zip(actual, predicted))
    fp = sum((not want) and got for want, got in zip(actual, predicted))
    tn = sum((not want) and (not got) for want, got in zip(actual, predicted))
    fn = sum(want and (not got) for want, got in zip(actual, predicted))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "world_recall": tp / (tp + fn) if tp + fn else 0.0,
        "self_suppression_specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _evaluate(
    trials: list[WorkspaceBindingTrial],
    evidence: dict[str, float],
    *,
    threshold: float,
    capacity: int,
    strategy: str,
) -> dict[str, object]:
    workspace = BoundedEfferenceWorkspace(capacity)
    for index in range(3):
        workspace.push(f"held_self_{index}")
    actual = []
    predicted = []
    for trial in trials:
        bound = workspace.contains(trial.efference_id)
        world_present = evidence[trial.trial_id] >= threshold
        if strategy == "flat":
            usable = True
        elif strategy == "fixed_self_window":
            usable = not bound
        elif strategy == "scheduled_identity_plus_world_evidence":
            usable = not bound or world_present
        else:
            raise ValueError(f"unknown workspace binding strategy: {strategy}")
        actual.append(trial.expected_world_usable)
        predicted.append(usable)
    return {
        "strategy": strategy,
        "capacity": capacity,
        "bounded_binding_slots": capacity,
        "world_evidence_threshold_scalars": 1,
        "metrics": _classification(actual, predicted),
    }


def run_fake_workspace_binding_protocol(
    config: WorkspaceBindingConfig | None = None,
) -> dict[str, object]:
    """Exercise the full structured H3 scoring path through a fake channel."""

    config = config or WorkspaceBindingConfig()
    trials = list(build_workspace_binding_trials())
    evidence = {}
    for index, trial in enumerate(trials):
        pcm, (start, end) = _render_trial(trial, config)
        captured, delay = _fake_capture(pcm, seed=17000 + index, config=config)
        evidence[trial.trial_id] = _world_band_ratio(
            captured[delay + start : delay + end],
            config.sample_rate,
        )
    development = [trial for trial in trials if trial.split == "development"]
    held_out = [trial for trial in trials if trial.split == "held_out"]
    max_self = max(evidence[trial.trial_id] for trial in development if not trial.expected_world_usable)
    min_world = min(evidence[trial.trial_id] for trial in development if trial.expected_world_usable)
    if min_world <= max_self:
        raise ValueError("structured world-band evidence has no development gap")
    # World preservation is the safety-critical side of this task. A midpoint
    # makes equal-gain overlap exactly borderline; select the lowest decile of
    # the clean development gap so any substantial unexplained world band is
    # preserved. The fraction is fixed before real I/O.
    threshold = max_self + 0.10 * (min_world - max_self)
    reports = {
        str(capacity): {
            strategy: _evaluate(
                held_out,
                evidence,
                threshold=threshold,
                capacity=capacity,
                strategy=strategy,
            )
            for strategy in (
                "flat",
                "fixed_self_window",
                "scheduled_identity_plus_world_evidence",
            )
        }
        for capacity in (1, 2, 3, 4)
    }
    selected = reports["3"]
    scheduled = selected["scheduled_identity_plus_world_evidence"]["metrics"]
    flat = selected["flat"]["metrics"]
    fixed = selected["fixed_self_window"]["metrics"]
    criteria = {
        "scheduled_f1_at_least_0_90": float(scheduled["f1"]) >= 0.90,
        "scheduled_improves_flat_f1_by_at_least_0_20": (
            float(scheduled["f1"]) - float(flat["f1"]) >= 0.20 - 1e-12
        ),
        "fixed_window_collision_failure_exposed": (
            float(fixed["world_recall"]) < float(scheduled["world_recall"])
        ),
        "capacity_one_worse_than_three": (
            float(reports["1"]["scheduled_identity_plus_world_evidence"]["metrics"]["f1"])
            < float(scheduled["f1"])
        ),
        "capacity_three_plateaus_at_four": math.isclose(
            float(reports["4"]["scheduled_identity_plus_world_evidence"]["metrics"]["f1"]),
            float(scheduled["f1"]),
        ),
    }
    return {
        **build_workspace_binding_protocol(config),
        "verdict": "fake_io_plumbing_passed" if all(criteria.values()) else "fake_io_gate_failed",
        "evidence_class": "synthetic_fake_acoustic_channel",
        "promotion_allowed": False,
        "world_evidence": evidence,
        "world_evidence_threshold": threshold,
        "world_evidence_threshold_rule": (
            "maximum development self-only ratio plus 10 percent of the clean "
            "self-only to world-only gap"
        ),
        "threshold_selected_from_development_only": True,
        "reports": reports,
        "criteria": criteria,
    }
