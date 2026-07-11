"""Prior-self-replay sensory-template candidate for natural-speech H3 binding."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .io.motor_plan import synthesize_chirp
from .io.wav_frames import read_pcm_wav
from .real_workspace_binding import BoundedEfferenceWorkspace


SAMPLE_RATE = 16_000
CHIRP_MS = 120
ALIGNMENT_SILENCE_MS = 100
SPEECH_MS = 600
TAIL_SILENCE_MS = 100
SELF_TARGET_RMS = 0.04
SELF_PEAK_CAP = 0.20
WORLD_TARGET_RMS = 0.10
WORLD_PEAK_CAP = 0.80
SELF_CHIRP_PEAK = 0.12
WORLD_CHIRP_PEAK = 0.03


@dataclass(frozen=True)
class TemplatePair:
    split: str
    index: int
    self_filename: str
    self_sha256: str
    world_filename: str
    world_sha256: str


@dataclass(frozen=True)
class TemplateTrial:
    trial_id: str
    split: str
    condition: str
    pair_index: int
    efference_id: str | None
    expected_world_usable: bool
    scored: bool


@dataclass(frozen=True)
class PreparedTemplateSpeech:
    filename: str
    sha256: str
    source_sample_rate: int
    source_channels: int
    source_frames: int
    selected_window_index: int
    selected_start_seconds: float
    achieved_rms: float
    achieved_peak: float
    pcm: np.ndarray


@dataclass(frozen=True)
class RenderedTemplateTracks:
    trial: TemplateTrial
    self_pcm: np.ndarray
    world_pcm: np.ndarray
    reference_pcm: np.ndarray
    chirp_window: tuple[int, int]
    stimulus_window: tuple[int, int]
    chirp_source: str


PAIRS = (
    TemplatePair("development", 0, "KRA06D.wav", "1601e8db49efe5216be02478420a06c6e15a5c7ed25e8cd8447b70c766bed249", "MEN05B.wav", "87c3c1eff2336a3f998f2f6fc425eb16b74e4a7f99f16ffa00f98cf4b3a3f5cd"),
    TemplatePair("development", 1, "MES11D.wav", "beb0402501f9ceb53c1c760d3d0544f129af0796747da15330d71347fb78d13f", "PNG06B.wav", "3648395e57e3c6e535d2d92c22d0ce6f703b5b4db3a405ced56aa7f52306319c"),
    TemplatePair("development", 2, "TSI09D.wav", "a06c60f533a1c1c1e2af77687ab2f315b76499b1b4afe88de9f25d9aa0c2848a", "VAN08B.wav", "32a10d4468730ef0b726e3ebdf4e1a194520896f78f354b7af3553506c3ee333"),
    TemplatePair("held_out", 0, "ACO12D.wav", "58bba429cbe744826b0e5ff319740710a989dff0e039d4fb514c8ea757cb1a29", "BEJ07B.wav", "5e41291a9764cc3f841448d2bdd1535cb0d551d6d3582ad95b4bc76fcfe25c12"),
    TemplatePair("held_out", 1, "HAD04D.wav", "4030085e7c5ac21b0c308430285003b24366f2ecb2f728d6cb2ebd23e0b6bda7", "JEN04B.wav", "9f194998d1d9794bda631c9c7c52ed7a8a2f9eb21634f1f828725f9730e405d6"),
    TemplatePair("held_out", 2, "LIM04D.wav", "f28fc186e6b3020ac782300db194e2c5f1beacb6a03e364f1fa073c17ce3721e", "NYA07B.wav", "87900ac4792c90e39921422a212052180d9e1c2a14df13c36a51dd618d799fdc"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_template_trials() -> tuple[TemplateTrial, ...]:
    trials: list[TemplateTrial] = []
    for pair in PAIRS:
        conditions = ("self_calibration", "self_probe", "world_only")
        if pair.split == "held_out":
            conditions = (*conditions, "overlap")
        for condition in conditions:
            has_self = condition != "world_only"
            has_world = condition in {"world_only", "overlap"}
            trials.append(
                TemplateTrial(
                    trial_id=f"{pair.split}_{condition}_{pair.index}",
                    split=pair.split,
                    condition=condition,
                    pair_index=pair.index,
                    efference_id=(
                        f"{pair.split}_self_{pair.index}" if has_self else None
                    ),
                    expected_world_usable=has_world,
                    scored=condition != "self_calibration",
                )
            )
    return tuple(trials)


def build_template_protocol() -> dict[str, object]:
    trials = build_template_trials()
    seconds_per_trial = (
        CHIRP_MS + ALIGNMENT_SILENCE_MS + SPEECH_MS + TAIL_SILENCE_MS
    ) / 1000.0
    return {
        "mode": "natural_speech_workspace_template_v2_protocol",
        "trial_count": len(trials),
        "held_out_scored_trial_count": sum(
            trial.split == "held_out" and trial.scored for trial in trials
        ),
        "seconds_per_trial": seconds_per_trial,
        "maximum_total_playback_seconds": seconds_per_trial * len(trials),
        "config": {
            "sample_rate": SAMPLE_RATE,
            "alignment_chirp_ms": CHIRP_MS,
            "alignment_silence_ms": ALIGNMENT_SILENCE_MS,
            "speech_ms": SPEECH_MS,
            "tail_silence_ms": TAIL_SILENCE_MS,
            "self_target_rms": SELF_TARGET_RMS,
            "self_peak_cap": SELF_PEAK_CAP,
            "world_target_rms": WORLD_TARGET_RMS,
            "world_peak_cap": WORLD_PEAK_CAP,
            "self_chirp_peak": SELF_CHIRP_PEAK,
            "world_chirp_peak": WORLD_CHIRP_PEAK,
        },
        "trials": [asdict(trial) for trial in trials],
        "capacity_sweep": [1, 2, 3, 4],
        "strategies": [
            "flat",
            "fixed_self_window",
            "scheduled_identity_plus_template_residual",
        ],
        "label_boundary": (
            "conditions define playback and enter scoring; the representation "
            "consumes only efference identity and prior/current microphone PCM"
        ),
        "template_memory_bound": {
            "workspace_slots": 3,
            "samples_per_speech_template": SPEECH_MS * SAMPLE_RATE // 1000,
            "samples_per_chirp_template": CHIRP_MS * SAMPLE_RATE // 1000,
            "cross_session_growth": False,
        },
        "promotion_allowed": False,
    }


def _resample_linear(mono: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return np.asarray(mono, dtype=np.float32)
    count = max(1, int(round(len(mono) * SAMPLE_RATE / source_rate)))
    return np.interp(
        np.arange(count, dtype=np.float64) * source_rate / SAMPLE_RATE,
        np.arange(len(mono), dtype=np.float64),
        mono,
    ).astype(np.float32)


def prepare_template_speech(
    path: str | Path,
    *,
    expected_sha256: str,
    target_rms: float,
    peak_cap: float,
) -> PreparedTemplateSpeech:
    source = Path(path)
    actual_hash = _sha256(source)
    if actual_hash != expected_sha256:
        raise ValueError(f"source hash mismatch: {source.name}")
    samples, source_rate = read_pcm_wav(source)
    mono = np.mean(samples, axis=1, dtype=np.float64).astype(np.float32)
    resampled = _resample_linear(mono, source_rate)
    window_samples = SPEECH_MS * SAMPLE_RATE // 1000
    window_count = len(resampled) // window_samples
    if window_count < 1:
        raise ValueError(f"source is shorter than {SPEECH_MS} ms: {source.name}")
    energies = [
        float(
            np.mean(
                resampled[
                    index * window_samples : (index + 1) * window_samples
                ]
                ** 2
            )
        )
        for index in range(window_count)
    ]
    selected = int(np.argmax(energies))
    start = selected * window_samples
    pcm = resampled[start : start + window_samples].copy()
    pcm -= np.float32(np.mean(pcm))
    source_rms = float(np.sqrt(np.mean(pcm**2)))
    if source_rms <= 1e-8:
        raise ValueError(f"selected source window is silent: {source.name}")
    pcm *= np.float32(target_rms / source_rms)
    peak = float(np.max(np.abs(pcm)))
    if peak > peak_cap:
        pcm *= np.float32(peak_cap / peak)
    achieved_rms = float(np.sqrt(np.mean(pcm**2)))
    achieved_peak = float(np.max(np.abs(pcm)))
    return PreparedTemplateSpeech(
        filename=source.name,
        sha256=actual_hash,
        source_sample_rate=source_rate,
        source_channels=samples.shape[1],
        source_frames=len(samples),
        selected_window_index=selected,
        selected_start_seconds=start / SAMPLE_RATE,
        achieved_rms=achieved_rms,
        achieved_peak=achieved_peak,
        pcm=pcm.astype(np.float32),
    )


def prepare_template_panel(
    root: str | Path,
) -> dict[tuple[str, int], tuple[PreparedTemplateSpeech, PreparedTemplateSpeech]]:
    base = Path(root)
    prepared = {}
    for pair in PAIRS:
        prepared[(pair.split, pair.index)] = (
            prepare_template_speech(
                base / pair.self_filename,
                expected_sha256=pair.self_sha256,
                target_rms=SELF_TARGET_RMS,
                peak_cap=SELF_PEAK_CAP,
            ),
            prepare_template_speech(
                base / pair.world_filename,
                expected_sha256=pair.world_sha256,
                target_rms=WORLD_TARGET_RMS,
                peak_cap=WORLD_PEAK_CAP,
            ),
        )
    return prepared


def render_template_tracks(
    trial: TemplateTrial,
    prepared: dict[
        tuple[str, int], tuple[PreparedTemplateSpeech, PreparedTemplateSpeech]
    ],
) -> RenderedTemplateTracks:
    self_source, world_source = prepared[(trial.split, trial.pair_index)]
    base_chirp = synthesize_chirp(
        250.0, 3500.0, duration_ms=CHIRP_MS, sample_rate=SAMPLE_RATE
    )
    self_chirp = np.float32(SELF_CHIRP_PEAK) * base_chirp
    world_chirp = np.float32(WORLD_CHIRP_PEAK) * base_chirp
    empty_chirp = np.zeros_like(base_chirp)
    pre = np.zeros(
        ALIGNMENT_SILENCE_MS * SAMPLE_RATE // 1000, dtype=np.float32
    )
    tail = np.zeros(TAIL_SILENCE_MS * SAMPLE_RATE // 1000, dtype=np.float32)
    silence = np.zeros(SPEECH_MS * SAMPLE_RATE // 1000, dtype=np.float32)
    has_self = trial.condition != "world_only"
    has_world = trial.condition in {"world_only", "overlap"}
    if has_self:
        self_prefix, world_prefix, chirp_source = (
            self_chirp,
            empty_chirp,
            "self",
        )
    else:
        self_prefix, world_prefix, chirp_source = (
            empty_chirp,
            world_chirp,
            "world",
        )
    self_pcm = np.concatenate(
        (self_prefix, pre, self_source.pcm if has_self else silence, tail)
    ).astype(np.float32)
    world_pcm = np.concatenate(
        (world_prefix, pre, world_source.pcm if has_world else silence, tail)
    ).astype(np.float32)
    speech_start = len(base_chirp) + len(pre)
    return RenderedTemplateTracks(
        trial=trial,
        self_pcm=self_pcm,
        world_pcm=world_pcm,
        reference_pcm=self_pcm if has_self else world_pcm,
        chirp_window=(0, len(base_chirp)),
        stimulus_window=(speech_start, speech_start + len(silence)),
        chirp_source=chirp_source,
    )


def _mono(capture: np.ndarray) -> np.ndarray:
    samples = np.asarray(capture, dtype=np.float64)
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    if samples.ndim != 1:
        raise ValueError("template capture must be mono or frame-by-channel")
    return samples


def _aligned_window(
    capture: np.ndarray, alignment: int, window: tuple[int, int], trial_id: str
) -> np.ndarray:
    samples = _mono(capture)
    start, end = alignment + window[0], alignment + window[1]
    if alignment < 0 or end > len(samples):
        raise ValueError(f"capture does not contain trial window: {trial_id}")
    return samples[start:end]


def _template_residual_ratio(
    *,
    calibration_capture: np.ndarray,
    calibration_alignment: int,
    calibration_tracks: RenderedTemplateTracks,
    current_capture: np.ndarray,
    current_alignment: int,
    current_tracks: RenderedTemplateTracks,
) -> tuple[float, float]:
    calibration_chirp = _aligned_window(
        calibration_capture,
        calibration_alignment,
        calibration_tracks.chirp_window,
        calibration_tracks.trial.trial_id,
    )
    current_chirp = _aligned_window(
        current_capture,
        current_alignment,
        current_tracks.chirp_window,
        current_tracks.trial.trial_id,
    )
    denominator = float(np.dot(calibration_chirp, calibration_chirp))
    if denominator <= 1e-12:
        raise ValueError("self-calibration chirp has insufficient energy")
    gain = float(np.dot(current_chirp, calibration_chirp) / denominator)
    calibration_speech = _aligned_window(
        calibration_capture,
        calibration_alignment,
        calibration_tracks.stimulus_window,
        calibration_tracks.trial.trial_id,
    )
    current_speech = _aligned_window(
        current_capture,
        current_alignment,
        current_tracks.stimulus_window,
        current_tracks.trial.trial_id,
    )
    residual = current_speech - gain * calibration_speech
    current_rms = float(np.sqrt(np.mean(current_speech**2)))
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    return residual_rms / max(current_rms, 1e-12), gain


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


def _evaluate_strategy(
    trials: list[TemplateTrial],
    evidence: dict[str, float],
    *,
    threshold: float,
    capacity: int,
    strategy: str,
) -> dict[str, object]:
    workspace = BoundedEfferenceWorkspace(capacity)
    for index in range(3):
        workspace.push(f"held_out_self_{index}")
    actual: list[bool] = []
    predicted: list[bool] = []
    for trial in trials:
        bound = workspace.contains(trial.efference_id)
        residual_world = evidence[trial.trial_id] >= threshold
        if strategy == "flat":
            usable = True
        elif strategy == "fixed_self_window":
            usable = not bound
        elif strategy == "scheduled_identity_plus_template_residual":
            usable = not bound or residual_world
        else:
            raise ValueError(f"unknown template workspace strategy: {strategy}")
        actual.append(trial.expected_world_usable)
        predicted.append(usable)
    return {
        "strategy": strategy,
        "capacity": capacity,
        "bounded_binding_slots": capacity,
        "world_evidence_threshold_scalars": 1,
        "metrics": _classification(actual, predicted),
    }


def score_template_evidence(evidence: dict[str, float]) -> dict[str, object]:
    trials = list(build_template_trials())
    development = [
        trial for trial in trials if trial.split == "development" and trial.scored
    ]
    held_out = [trial for trial in trials if trial.split == "held_out" and trial.scored]
    max_self = max(
        evidence[trial.trial_id]
        for trial in development
        if not trial.expected_world_usable
    )
    min_world = min(
        evidence[trial.trial_id]
        for trial in development
        if trial.expected_world_usable
    )
    if min_world <= max_self:
        return {
            "verdict": "capture_acceptance_failed_no_development_gap",
            "world_evidence": evidence,
            "world_evidence_threshold": None,
            "development_gap": {
                "maximum_self_probe": max_self,
                "minimum_world_only": min_world,
                "gap": min_world - max_self,
            },
            "reports": {},
            "criteria": {"clean_development_gap": False},
        }
    threshold = max_self + 0.10 * (min_world - max_self)
    strategies = (
        "flat",
        "fixed_self_window",
        "scheduled_identity_plus_template_residual",
    )
    reports = {
        str(capacity): {
            strategy: _evaluate_strategy(
                held_out,
                evidence,
                threshold=threshold,
                capacity=capacity,
                strategy=strategy,
            )
            for strategy in strategies
        }
        for capacity in (1, 2, 3, 4)
    }
    selected = reports["3"]
    scheduled = selected["scheduled_identity_plus_template_residual"]["metrics"]
    flat = selected["flat"]["metrics"]
    fixed = selected["fixed_self_window"]["metrics"]
    criteria = {
        "clean_development_gap": True,
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
            float(
                reports["1"]["scheduled_identity_plus_template_residual"][
                    "metrics"
                ]["f1"]
            )
            < float(scheduled["f1"])
        ),
        "capacity_three_plateaus_at_four": math.isclose(
            float(
                reports["4"]["scheduled_identity_plus_template_residual"][
                    "metrics"
                ]["f1"]
            ),
            float(scheduled["f1"]),
        ),
    }
    return {
        "verdict": (
            "capture_acceptance_passed"
            if all(criteria.values())
            else "capture_acceptance_failed"
        ),
        "world_evidence": evidence,
        "world_evidence_threshold": threshold,
        "world_evidence_threshold_rule": "maximum development self-probe ratio plus 10 percent of the clean gap to minimum development world-only ratio",
        "threshold_selected_from_development_only": True,
        "reports": reports,
        "criteria": criteria,
    }


def evaluate_template_captures(
    *,
    root: str | Path,
    captures: dict[str, np.ndarray],
    alignment_samples: dict[str, int],
) -> dict[str, object]:
    prepared = prepare_template_panel(root)
    trials = list(build_template_trials())
    expected = {trial.trial_id for trial in trials}
    if set(captures) != expected or set(alignment_samples) != expected:
        raise ValueError("template capture and alignment maps must cover every trial")
    by_key = {(trial.split, trial.pair_index, trial.condition): trial for trial in trials}
    evidence: dict[str, float] = {}
    gains: dict[str, float] = {}
    for trial in trials:
        if not trial.scored:
            continue
        if trial.efference_id is None:
            evidence[trial.trial_id] = 1.0
            gains[trial.trial_id] = 0.0
            continue
        calibration = by_key[(trial.split, trial.pair_index, "self_calibration")]
        calibration_tracks = render_template_tracks(calibration, prepared)
        current_tracks = render_template_tracks(trial, prepared)
        ratio, gain = _template_residual_ratio(
            calibration_capture=captures[calibration.trial_id],
            calibration_alignment=alignment_samples[calibration.trial_id],
            calibration_tracks=calibration_tracks,
            current_capture=captures[trial.trial_id],
            current_alignment=alignment_samples[trial.trial_id],
            current_tracks=current_tracks,
        )
        evidence[trial.trial_id] = ratio
        gains[trial.trial_id] = gain
    return {
        **build_template_protocol(),
        **score_template_evidence(evidence),
        "template_gain_scalars": gains,
        "evidence_class": "provided_chirp_aligned_prior_self_template_captures",
        "promotion_allowed": False,
    }


def run_spent_template_design_gate(root: str | Path) -> dict[str, object]:
    prepared = prepare_template_panel(root)
    self_impulse = np.zeros(128, dtype=np.float32)
    self_impulse[0], self_impulse[61] = 0.35, 0.07
    world_impulse = np.zeros(128, dtype=np.float32)
    world_impulse[0], world_impulse[79] = 0.25, 0.04
    delay = 30 * SAMPLE_RATE // 1000
    rng = np.random.default_rng(17_012)
    captures: dict[str, np.ndarray] = {}
    alignments: dict[str, int] = {}
    for trial in build_template_trials():
        rendered = render_template_tracks(trial, prepared)
        observed = np.convolve(rendered.self_pcm, self_impulse)
        observed += np.convolve(rendered.world_pcm, world_impulse)
        capture = np.concatenate((np.zeros(delay), observed, np.zeros(SAMPLE_RATE // 10)))
        capture += rng.normal(0.0, 0.0001, len(capture))
        captures[trial.trial_id] = np.column_stack((capture, capture)).astype(np.float32)
        alignments[trial.trial_id] = delay
    report = evaluate_template_captures(
        root=root, captures=captures, alignment_samples=alignments
    )
    report["mode"] = "spent_natural_speech_workspace_template_v2_design_gate"
    report["design_preregistration"] = "docs/evidence/2026-07-11-natural-speech-workspace-template-v2-attempt-2-design-preregistration.json"
    report["source_artifacts"] = [
        {
            "split": pair.split,
            "index": pair.index,
            "self": {
                "filename": prepared[(pair.split, pair.index)][0].filename,
                "sha256": prepared[(pair.split, pair.index)][0].sha256,
                "selected_start_seconds": prepared[(pair.split, pair.index)][0].selected_start_seconds,
                "achieved_rms": prepared[(pair.split, pair.index)][0].achieved_rms,
                "achieved_peak": prepared[(pair.split, pair.index)][0].achieved_peak,
            },
            "world": {
                "filename": prepared[(pair.split, pair.index)][1].filename,
                "sha256": prepared[(pair.split, pair.index)][1].sha256,
                "selected_start_seconds": prepared[(pair.split, pair.index)][1].selected_start_seconds,
                "achieved_rms": prepared[(pair.split, pair.index)][1].achieved_rms,
                "achieved_peak": prepared[(pair.split, pair.index)][1].achieved_peak,
            },
        }
        for pair in PAIRS
    ]
    report["verdict"] = (
        "spent_design_gate_passed"
        if all(report["criteria"].values())
        else "spent_design_gate_failed"
    )
    report["promotion_allowed"] = False
    report["claim_boundary"] = "Deterministic repeated-path synthetic captures over exact fresh already-spent natural-speech PCM. This validates the bounded sensory-template representation and scoring only; it cannot promote H3."
    return report
