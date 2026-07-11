"""Structured H3 self/world binding protocol and device-agnostic acquisition."""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .io.motor_plan import synthesize_chirp, synthesize_tone, write_wav
from .io.wav_frames import read_pcm_wav
from .live_speak_smoke import _capture_during_playback, estimate_playback_latency_ms
from .real_workspace_binding import BoundedEfferenceWorkspace


MAX_WORKSPACE_TRIAL_PLAYBACK_SECONDS = 2.0
MAX_WORKSPACE_TOTAL_PLAYBACK_SECONDS = 25.0
MAX_WORKSPACE_CAPTURE_TAIL_MS = 1000
MAX_WORKSPACE_TOTAL_CAPTURE_SECONDS = 45.0


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


@dataclass(frozen=True)
class RenderedWorkspaceBindingTracks:
    trial: WorkspaceBindingTrial
    self_pcm: np.ndarray
    world_pcm: np.ndarray
    reference_pcm: np.ndarray
    stimulus_window: tuple[int, int]
    chirp_source: str


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


def _validate_acquisition_bounds(config: WorkspaceBindingConfig) -> None:
    if (
        config.alignment_chirp_ms <= 0
        or config.alignment_silence_ms < 0
        or config.stimulus_ms <= 0
        or config.tail_silence_ms < 0
    ):
        raise ValueError("workspace segment durations must be positive/non-negative")
    protocol = build_workspace_binding_protocol(config)
    seconds_per_trial = float(protocol["seconds_per_trial"])
    total_seconds = float(protocol["maximum_total_playback_seconds"])
    if seconds_per_trial <= 0 or seconds_per_trial > MAX_WORKSPACE_TRIAL_PLAYBACK_SECONDS:
        raise ValueError("workspace trial playback exceeds the frozen 2 second bound")
    if total_seconds > MAX_WORKSPACE_TOTAL_PLAYBACK_SECONDS:
        raise ValueError("workspace protocol playback exceeds the frozen 25 second bound")
    if config.sample_rate <= 0:
        raise ValueError("workspace sample rate must be positive")
    if not (0.0 < config.self_gain <= 0.25 and 0.0 < config.world_gain <= 0.25):
        raise ValueError("workspace source gains must be in (0, 0.25]")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_workspace_binding_preregistration(
    preregistration: Mapping[str, object],
    *,
    current_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Return exact blockers for the hash- and geometry-bound H3 live gate."""

    blockers: list[str] = []
    if preregistration.get("mode") != "structured_workspace_binding_real_preregistration":
        blockers.append("mode is not the structured H3 real preregistration")
    if preregistration.get("status") != "preregistered_before_live_execution":
        blockers.append("status is not preregistered_before_live_execution")
    if preregistration.get("bounded_playback_capture_authorized") is not True:
        blockers.append("bounded playback and capture are not authorized")

    frozen_hashes = preregistration.get("frozen_hashes")
    if not isinstance(frozen_hashes, Mapping):
        blockers.append("frozen_hashes are missing")
    else:
        for name, actual in current_hashes.items():
            if frozen_hashes.get(name) != actual:
                blockers.append(f"frozen hash mismatch: {name}")

    hardware = preregistration.get("hardware")
    if not isinstance(hardware, Mapping):
        blockers.append("hardware mapping is missing")
        hardware = {}
    input_route = hardware.get("input")
    self_route = hardware.get("self_source")
    world_route = hardware.get("world_source")
    if not isinstance(input_route, Mapping):
        blockers.append("input route is missing")
        input_route = {}
    if not isinstance(self_route, Mapping):
        blockers.append("self source route is missing")
        self_route = {}
    if not isinstance(world_route, Mapping):
        blockers.append("world source route is missing")
        world_route = {}

    if input_route.get("avfoundation_device") != ":0":
        blockers.append("input must be XVF3800 avfoundation device :0")
    if input_route.get("device") != "reSpeaker XVF3800 4-Mic Array":
        blockers.append("input device identity is not frozen to XVF3800")
    if input_route.get("channels") != 2:
        blockers.append("input must retain the two XVF3800 fused channels")
    if input_route.get("sample_rate") != 16000:
        blockers.append("input sample rate must remain 16000 Hz")
    if self_route.get("output_device_index") != 3:
        blockers.append("self source must use output device 3")
    if self_route.get("output_device") != "Mac mini Speakers":
        blockers.append("self source must use Mac mini Speakers")
    if world_route.get("output_device_index") != 2:
        blockers.append("world source must use output device 2")
    if world_route.get("output_device") != "External Headphones":
        blockers.append("world source must use External Headphones")
    if world_route.get("angle_degrees") != 180:
        blockers.append("world source angle must remain 180 degrees")
    if world_route.get("distance_cm") != 50:
        blockers.append("world source distance must remain 50 cm")
    if world_route.get("hardware_volume") != "max":
        blockers.append("world speaker hardware volume must remain max")
    if world_route.get("playback_gain_db") != 0.0:
        blockers.append("world playback gain must be frozen at 0 dB")
    if self_route.get("playback_gain_db") != 0.0:
        blockers.append("self playback gain must be frozen at 0 dB")

    self_angle = self_route.get("angle_degrees")
    if not isinstance(self_angle, (int, float)) or not (0 <= self_angle <= 360):
        blockers.append("self source angle is not frozen")
    self_distance = self_route.get("distance_cm")
    if not isinstance(self_distance, (int, float)) or self_distance <= 0:
        blockers.append("self source distance is not frozen")
    self_volume = self_route.get("system_volume_percent")
    if not isinstance(self_volume, (int, float)) or not (0 < self_volume <= 100):
        blockers.append("self source system volume is not frozen")

    bounds = preregistration.get("bounds")
    expected_bounds = {
        "trial_count": 15,
        "per_trial_playback_seconds_max": 2.0,
        "total_playback_seconds_max": 25.0,
        "total_capture_seconds_max": 45.0,
        "planned_playback_seconds": 19.5,
        "planned_capture_seconds": 28.5,
    }
    if not isinstance(bounds, Mapping) or any(
        bounds.get(name) != value for name, value in expected_bounds.items()
    ):
        blockers.append("live run bounds do not match the frozen protocol")

    environment = hardware.get("environment")
    if not isinstance(environment, Mapping) or (
        environment.get("room") != "semi-quiet"
        or environment.get("known_background") != "laptop fan"
    ):
        blockers.append("room state does not match the confirmed environment")

    acceptance = preregistration.get("acceptance")
    expected_acceptance = {
        "selected_capacity": 3,
        "scheduled_f1_min": 0.9,
        "scheduled_minus_flat_f1_min": 0.2,
        "scheduled_world_recall_min": 0.9,
        "scheduled_self_suppression_specificity_min": 0.9,
        "fixed_window_world_recall_must_be_below_scheduled": True,
        "capacity_one_f1_must_be_below_capacity_three": True,
        "capacity_three_f1_must_equal_capacity_four": True,
        "all_15_captures_and_alignments_required": True,
    }
    if not isinstance(acceptance, Mapping) or any(
        acceptance.get(name) != value for name, value in expected_acceptance.items()
    ):
        blockers.append("acceptance criteria do not match the frozen H3 gate")
    if preregistration.get("unresolved_fields") != []:
        blockers.append("preregistration still lists unresolved fields")

    return {
        "verdict": "preregistration_ready" if not blockers else "preregistration_blocked",
        "ready": not blockers,
        "blockers": blockers,
    }


def render_workspace_binding_tracks(
    trial: WorkspaceBindingTrial,
    config: WorkspaceBindingConfig,
) -> RenderedWorkspaceBindingTracks:
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
    self_stimulus = np.zeros(stimulus_length, dtype=np.float32)
    world_stimulus = np.zeros(stimulus_length, dtype=np.float32)
    if trial.self_frequency_hz is not None:
        self_stimulus += config.self_gain * synthesize_tone(
            trial.self_frequency_hz,
            duration_ms=config.stimulus_ms,
            sample_rate=config.sample_rate,
        )
    if trial.world_frequency_hz is not None:
        world_stimulus += config.world_gain * synthesize_tone(
            trial.world_frequency_hz,
            duration_ms=config.stimulus_ms,
            sample_rate=config.sample_rate,
        )
    tail = np.zeros(
        int(config.tail_silence_ms * config.sample_rate / 1000),
        dtype=np.float32,
    )
    start = len(chirp) + len(pre)
    empty_chirp = np.zeros_like(chirp)
    if trial.self_frequency_hz is not None:
        self_chirp, world_chirp, chirp_source = chirp, empty_chirp, "self"
    else:
        self_chirp, world_chirp, chirp_source = empty_chirp, chirp, "world"
    self_pcm = np.concatenate(
        (self_chirp, pre, np.clip(self_stimulus, -1.0, 1.0), tail)
    ).astype(np.float32)
    world_pcm = np.concatenate(
        (world_chirp, pre, np.clip(world_stimulus, -1.0, 1.0), tail)
    ).astype(np.float32)
    return RenderedWorkspaceBindingTracks(
        trial=trial,
        self_pcm=self_pcm,
        world_pcm=world_pcm,
        reference_pcm=self_pcm if chirp_source == "self" else world_pcm,
        stimulus_window=(start, start + stimulus_length),
        chirp_source=chirp_source,
    )


def _render_trial(
    trial: WorkspaceBindingTrial,
    config: WorkspaceBindingConfig,
) -> tuple[np.ndarray, tuple[int, int]]:
    tracks = render_workspace_binding_tracks(trial, config)
    return tracks.self_pcm + tracks.world_pcm, tracks.stimulus_window


def play_workspace_tracks_concurrently(
    *,
    self_path: str | None,
    world_path: str | None,
    self_playback: Callable[[str], dict[str, object]],
    world_playback: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    """Start non-empty self/world tracks together and retain both reports."""

    jobs = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        if self_path is not None:
            jobs.append(("self", executor.submit(self_playback, self_path)))
        if world_path is not None:
            jobs.append(("world", executor.submit(world_playback, world_path)))
        reports = {name: future.result() for name, future in jobs}
    if not reports:
        raise ValueError("workspace playback requires at least one source track")
    return {
        "backend": "concurrent_explicit_output_tracks",
        "started": all(bool(report.get("started")) for report in reports.values()),
        "sources": reports,
    }


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


def _score_workspace_binding_evidence(
    trials: list[WorkspaceBindingTrial], evidence: dict[str, float]
) -> dict[str, object]:
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
        "scheduled_world_recall_at_least_0_90": (
            float(scheduled["world_recall"]) >= 0.90
        ),
        "scheduled_self_suppression_specificity_at_least_0_90": (
            float(scheduled["self_suppression_specificity"]) >= 0.90
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
        "verdict": "capture_acceptance_passed" if all(criteria.values()) else "capture_acceptance_failed",
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


def evaluate_workspace_binding_captures(
    *,
    captures: dict[str, np.ndarray],
    alignment_samples: dict[str, int],
    config: WorkspaceBindingConfig | None = None,
) -> dict[str, object]:
    """Score chirp-aligned captures without performing acquisition or playback."""

    config = config or WorkspaceBindingConfig()
    trials = list(build_workspace_binding_trials())
    expected = {trial.trial_id for trial in trials}
    if set(captures) != expected or set(alignment_samples) != expected:
        raise ValueError("workspace capture and alignment maps must cover every trial")
    evidence = {}
    for trial in trials:
        samples = np.asarray(captures[trial.trial_id], dtype=np.float64)
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)
        if samples.ndim != 1:
            raise ValueError("workspace captures must be mono or frame-by-channel")
        tracks = render_workspace_binding_tracks(trial, config)
        start, end = tracks.stimulus_window
        aligned_start = int(alignment_samples[trial.trial_id]) + start
        aligned_end = int(alignment_samples[trial.trial_id]) + end
        if aligned_start < 0 or aligned_end > len(samples):
            raise ValueError(f"capture does not contain stimulus window: {trial.trial_id}")
        evidence[trial.trial_id] = _world_band_ratio(
            samples[aligned_start:aligned_end], config.sample_rate
        )
    return {
        **build_workspace_binding_protocol(config),
        **_score_workspace_binding_evidence(trials, evidence),
        "evidence_class": "provided_chirp_aligned_capture_arrays",
        "promotion_allowed": False,
    }


def run_workspace_binding_acquisition(
    *,
    out_dir: str | Path,
    self_playback: Callable[[str], dict[str, object]],
    world_playback: Callable[[str], dict[str, object]],
    capture_fn: Callable[..., dict[str, object]],
    config: WorkspaceBindingConfig | None = None,
    capture_preroll_ms: int = 100,
    capture_tail_ms: int = 500,
    real_hardware_claim: bool = False,
    bounded_playback_authorized: bool = False,
    hardware_preregistration: Mapping[str, object] | None = None,
    current_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Acquire every trial through injected, source-specific I/O callbacks.

    This runner deliberately has no device discovery, CLI route, or concrete
    hardware callback. Until the complete physical preregistration is frozen,
    ``real_hardware_claim`` requires an authorized preregistration matching the
    current code hashes. The acquisition path can therefore be verified with
    injected I/O while incomplete hardware metadata fails before callbacks run.
    """

    config = config or WorkspaceBindingConfig()
    _validate_acquisition_bounds(config)
    if not (0 <= capture_tail_ms <= MAX_WORKSPACE_CAPTURE_TAIL_MS):
        raise ValueError("workspace capture tail must be between 0 and 1000 ms")
    if capture_preroll_ms < 0 or capture_preroll_ms > 1000:
        raise ValueError("workspace capture preroll must be between 0 and 1000 ms")
    if real_hardware_claim:
        if not bounded_playback_authorized:
            raise ValueError("real H3 hardware execution requires bounded authorization")
        if hardware_preregistration is None or current_hashes is None:
            raise ValueError("real H3 hardware execution requires a frozen preregistration")
        preregistration_audit = audit_workspace_binding_preregistration(
            hardware_preregistration,
            current_hashes=current_hashes,
        )
        if not preregistration_audit["ready"]:
            raise ValueError(
                "real H3 preregistration blocked: "
                + "; ".join(str(item) for item in preregistration_audit["blockers"])
            )

    protocol = build_workspace_binding_protocol(config)
    bounded_total_capture_seconds = float(protocol["maximum_total_playback_seconds"]) + (
        capture_preroll_ms + capture_tail_ms
    ) / 1000.0 * int(protocol["trial_count"])
    if bounded_total_capture_seconds > MAX_WORKSPACE_TOTAL_CAPTURE_SECONDS:
        raise ValueError("workspace protocol capture exceeds the frozen 45 second bound")

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    captures: dict[str, np.ndarray] = {}
    alignments: dict[str, int] = {}
    artifacts: list[dict[str, object]] = []
    total_playback_seconds = 0.0

    for trial in build_workspace_binding_trials():
        rendered = render_workspace_binding_tracks(trial, config)
        self_path = output / f"{trial.trial_id}_self.wav"
        world_path = output / f"{trial.trial_id}_world.wav"
        capture_path = output / f"{trial.trial_id}_capture.wav"
        write_wav(str(self_path), rendered.self_pcm, config.sample_rate)
        write_wav(str(world_path), rendered.world_pcm, config.sample_rate)
        capture_duration_ms = int(
            math.ceil(len(rendered.reference_pcm) * 1000.0 / config.sample_rate)
        ) + capture_preroll_ms + capture_tail_ms
        playback_started = False

        def _play_sources(_reference_path: str) -> dict[str, object]:
            nonlocal playback_started
            playback_started = True
            report = play_workspace_tracks_concurrently(
                self_path=(str(self_path) if trial.self_frequency_hz is not None else None),
                world_path=(str(world_path) if trial.world_frequency_hz is not None else None),
                self_playback=self_playback,
                world_playback=world_playback,
            )
            if not report["started"]:
                raise RuntimeError("one or more workspace sources did not start")
            return report

        try:
            io_report = _capture_during_playback(
                capture_path=str(capture_path),
                playback_path=(
                    str(self_path) if rendered.chirp_source == "self" else str(world_path)
                ),
                sample_rate=config.sample_rate,
                capture_duration_ms=capture_duration_ms,
                capture_fn=capture_fn,
                playback_fn=_play_sources,
                capture_preroll_ms=capture_preroll_ms,
            )
            captured_samples, captured_rate = read_pcm_wav(capture_path)
            if captured_rate != config.sample_rate:
                raise ValueError("workspace capture sample rate mismatch")
            if real_hardware_claim:
                capture_report = io_report.get("capture")
                playback_report = io_report.get("playback")
                if not isinstance(capture_report, Mapping) or (
                    capture_report.get("backend") != "ffmpeg_avfoundation"
                ):
                    raise ValueError("real H3 capture backend provenance was not verified")
                if not isinstance(playback_report, Mapping):
                    raise ValueError("real H3 playback provenance was not verified")
                source_reports = playback_report.get("sources")
                if not isinstance(source_reports, Mapping) or any(
                    not isinstance(report, Mapping)
                    or report.get("backend") != "ffmpeg_audiotoolbox"
                    for report in source_reports.values()
                ):
                    raise ValueError("real H3 source playback provenance was not verified")
                if not bool(io_report.get("stream_position_verified")):
                    raise ValueError("real H3 capture stream position was not verified")
                if captured_samples.shape[1] != 2:
                    raise ValueError("real H3 XVF3800 capture must retain two channels")
            chirp_samples = int(config.alignment_chirp_ms * config.sample_rate / 1000)
            detected_onset_ms = estimate_playback_latency_ms(
                efference_pcm=rendered.reference_pcm[:chirp_samples],
                captured_pcm=np.mean(captured_samples, axis=1),
                sample_rate=config.sample_rate,
                expected_onset_ms=float(io_report["playback_offset_from_capture_ms"]),
                causal_window_ms=500.0,
                min_peak_correlation=0.08,
            )
            if detected_onset_ms is None:
                raise ValueError("workspace alignment chirp was not recovered")
            alignment_samples = int(
                round(detected_onset_ms * config.sample_rate / 1000.0)
            )
        except Exception as exc:  # noqa: BLE001
            return {
                **build_workspace_binding_protocol(config),
                "verdict": "acquisition_failed",
                "real_hardware": real_hardware_claim,
                "live_playback": bool(real_hardware_claim and playback_started),
                "playback_callback_started": playback_started,
                "hardware": (
                    hardware_preregistration.get("hardware")
                    if hardware_preregistration is not None
                    else None
                ),
                "failed_trial_id": trial.trial_id,
                "completed_trial_count": len(artifacts),
                "bounded_total_capture_seconds": bounded_total_capture_seconds,
                "error": f"{type(exc).__name__}: {exc}",
                "artifacts": artifacts,
                "promotion_allowed": False,
            }

        captures[trial.trial_id] = captured_samples
        alignments[trial.trial_id] = alignment_samples
        total_playback_seconds += len(rendered.reference_pcm) / config.sample_rate
        artifacts.append(
            {
                "trial_id": trial.trial_id,
                "condition": trial.condition,
                "chirp_source": rendered.chirp_source,
                "self_wav": str(self_path.resolve()),
                "self_sha256": _sha256(self_path),
                "world_wav": str(world_path.resolve()),
                "world_sha256": _sha256(world_path),
                "captured_wav": str(capture_path.resolve()),
                "captured_sha256": _sha256(capture_path),
                "alignment_samples": alignment_samples,
                "alignment_onset_ms": detected_onset_ms,
                "io_timing": {
                    key: io_report[key]
                    for key in (
                        "playback_offset_samples",
                        "playback_offset_from_capture_ms",
                        "overlap_verified",
                        "stream_position_verified",
                        "timestamp_source",
                    )
                },
                "playback": io_report["playback"],
                "capture": io_report["capture"],
            }
        )

    evaluation = evaluate_workspace_binding_captures(
        captures=captures,
        alignment_samples=alignments,
        config=config,
    )
    capture_passed = evaluation["verdict"] == "capture_acceptance_passed"
    return {
        **evaluation,
        "verdict": (
            "real_io_acceptance_passed"
            if real_hardware_claim and capture_passed
            else "real_io_acceptance_failed"
            if real_hardware_claim
            else "injected_acquisition_complete"
        ),
        "capture_acceptance_verdict": evaluation["verdict"],
        "real_hardware": real_hardware_claim,
        "live_playback": real_hardware_claim,
        "playback_callback_started": True,
        "hardware": (
            hardware_preregistration.get("hardware")
            if hardware_preregistration is not None
            else None
        ),
        "completed_trial_count": len(artifacts),
        "measured_total_playback_seconds": round(total_playback_seconds, 6),
        "bounded_total_capture_seconds": round(bounded_total_capture_seconds, 6),
        "artifacts": artifacts,
        "error": None,
        "promotion_allowed": bool(real_hardware_claim and capture_passed),
    }


def run_fake_workspace_binding_protocol(
    config: WorkspaceBindingConfig | None = None,
) -> dict[str, object]:
    """Exercise the full structured H3 scoring path through a fake channel."""

    config = config or WorkspaceBindingConfig()
    captures = {}
    alignments = {}
    for index, trial in enumerate(build_workspace_binding_trials()):
        pcm, _window = _render_trial(trial, config)
        captured, delay = _fake_capture(pcm, seed=17000 + index, config=config)
        captures[trial.trial_id] = captured
        alignments[trial.trial_id] = delay
    report = evaluate_workspace_binding_captures(
        captures=captures,
        alignment_samples=alignments,
        config=config,
    )
    report["evidence_class"] = "synthetic_fake_acoustic_channel"
    report["verdict"] = (
        "fake_io_plumbing_passed"
        if all(bool(value) for value in report["criteria"].values())
        else "fake_io_gate_failed"
    )
    return report
