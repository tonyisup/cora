"""Bounded structured continuous-audio protocol for the next H2 experiment.

The safe default only returns the frozen protocol manifest. Fake I/O exercises
the complete synthesis -> acoustic channel -> measured spectral state -> local
transition-prediction path, but is explicitly non-real evidence. Real playback
is intentionally not wired here until the operator authorizes the preregistered
bounded run.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .io.motor_plan import (
    Vowel,
    synthesize_chirp,
    synthesize_tone,
    synthesize_vowel,
    write_wav,
)
from .io.wav_frames import read_pcm_wav
from .live_speak_smoke import _capture_during_playback, estimate_playback_latency_ms


TOKEN_COUNT = 4
CALIBRATION_SEQUENCE = (0, 1, 2, 3)
TONE_FREQUENCIES_HZ = (300.0, 500.0, 800.0, 1200.0)
VOWEL_TOKENS: tuple[Vowel, ...] = ("a", "e", "i", "u")
DEVELOPMENT_SEEDS = (1103, 2207, 3301)
HELD_OUT_SEEDS = (4409, 5501, 6607)
CALIBRATED_V2_DEVELOPMENT_SEEDS = (7703, 8807, 9901)
CALIBRATED_V2_HELD_OUT_SEEDS = (10103, 11213, 12323)
CALIBRATED_V2_STD_FLOOR = 0.30

PlaybackFn = Callable[[str], dict[str, object]]
CaptureFn = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class StructuredProtocolConfig:
    sample_rate: int = 16000
    token_duration_ms: int = 200
    gap_duration_ms: int = 50
    alignment_chirp_ms: int = 120
    alignment_silence_ms: int = 180
    grammar_token_count: int = 20
    fake_delay_ms: int = 30
    fake_attenuation: float = 0.35
    fake_echo_delay_ms: int = 18
    fake_echo_attenuation: float = 0.08
    fake_noise_rms: float = 0.002
    maximum_real_playback_db: float = -18.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        if self.token_duration_ms < 100:
            raise ValueError("token duration must be at least 100 ms")
        if self.gap_duration_ms < 0:
            raise ValueError("gap duration must be non-negative")
        if self.grammar_token_count < 8:
            raise ValueError("grammar needs at least eight tokens")
        if not 0.0 < self.fake_attenuation <= 1.0:
            raise ValueError("fake attenuation must be in (0, 1]")
        if self.fake_noise_rms < 0.0:
            raise ValueError("fake noise RMS must be non-negative")


@dataclass(frozen=True)
class StructuredTrialPlan:
    trial_id: str
    family: str
    split: str
    seed: int
    calibration_sequence: tuple[int, ...]
    grammar_sequence: tuple[int, ...]


@dataclass(frozen=True)
class RenderedStructuredTrial:
    plan: StructuredTrialPlan
    pcm: np.ndarray
    token_windows: tuple[tuple[int, int], ...]
    calibration_count: int


class LocalAcousticCodebook:
    """Four-centroid competitive learner with bounded running-mean updates."""

    def __init__(self, initial_observations: np.ndarray) -> None:
        initial = np.asarray(initial_observations, dtype=np.float64)
        if initial.ndim != 2 or initial.shape[0] != TOKEN_COUNT:
            raise ValueError("codebook requires four initial acoustic observations")
        if not bool(np.all(np.isfinite(initial))):
            raise ValueError("codebook observations must be finite")
        self.centroids = initial.copy()
        self.counts = np.ones(TOKEN_COUNT, dtype=np.int64)

    def assign(self, observation: np.ndarray) -> int:
        vector = np.asarray(observation, dtype=np.float64)
        if vector.shape != (self.centroids.shape[1],):
            raise ValueError("observation shape does not match codebook")
        distances = np.sum((self.centroids - vector) ** 2, axis=1)
        return int(np.argmin(distances))

    def observe(self, observation: np.ndarray) -> int:
        state = self.assign(observation)
        self.counts[state] += 1
        self.centroids[state] += (
            np.asarray(observation, dtype=np.float64) - self.centroids[state]
        ) / self.counts[state]
        return state

    @property
    def state_scalars(self) -> int:
        return int(self.centroids.size + self.counts.size)


class LocalTransitionPredictor:
    """Locally counted state transitions whose prediction is consumed next."""

    def __init__(self, state_count: int = TOKEN_COUNT, smoothing: float = 0.5) -> None:
        if state_count <= 1 or smoothing <= 0.0:
            raise ValueError("invalid transition predictor geometry")
        self.counts = np.full((state_count, state_count), smoothing, dtype=np.float64)

    def observe_sequence(self, states: list[int]) -> None:
        for current, following in zip(states, states[1:]):
            self.counts[current, following] += 1.0

    def predict(self, current: int) -> int:
        return int(np.argmax(self.counts[current]))

    @property
    def state_scalars(self) -> int:
        return int(self.counts.size)


def _grammar_sequence(seed: int, count: int) -> tuple[int, ...]:
    """Generate a branch cycle with no self-transition baseline shortcut."""

    rng = np.random.default_rng(seed)
    current = 0
    sequence = [current]
    while len(sequence) < count:
        if current == 0:
            current = 1
        elif current == 1:
            current = 2
        elif current == 2:
            current = 3 if float(rng.random()) < 0.3 else 0
        else:
            current = 0
        sequence.append(current)
    return tuple(sequence)


def build_structured_protocol_manifest(
    config: StructuredProtocolConfig | None = None,
    *,
    development_seeds: tuple[int, ...] | None = None,
    held_out_seeds: tuple[int, ...] | None = None,
    protocol_variant: str | None = None,
    calibration_normalization_std_floor: float | None = None,
) -> dict[str, object]:
    config = config or StructuredProtocolConfig()
    development_seeds = development_seeds or DEVELOPMENT_SEEDS
    held_out_seeds = held_out_seeds or HELD_OUT_SEEDS
    trials = []
    for family in ("tones", "vowels"):
        for split, seeds in (
            ("dev", development_seeds),
            ("held_out", held_out_seeds),
        ):
            for seed in seeds:
                trials.append(
                    StructuredTrialPlan(
                        trial_id=f"{family}_{split}_{seed}",
                        family=family,
                        split=split,
                        seed=seed,
                        calibration_sequence=CALIBRATION_SEQUENCE,
                        grammar_sequence=_grammar_sequence(
                            seed, config.grammar_token_count
                        ),
                    )
                )
    seconds_per_trial = (
        config.alignment_chirp_ms
        + config.alignment_silence_ms
        + (len(CALIBRATION_SEQUENCE) + config.grammar_token_count)
        * (config.token_duration_ms + config.gap_duration_ms)
    ) / 1000.0
    manifest: dict[str, object] = {
        "mode": "structured_continuous_prediction_protocol",
        "config": asdict(config),
        "families": {
            "tones": list(TONE_FREQUENCIES_HZ),
            "vowels": list(VOWEL_TOKENS),
        },
        "development_seeds": list(development_seeds),
        "held_out_seeds": list(held_out_seeds),
        "trial_count": len(trials),
        "seconds_per_trial": seconds_per_trial,
        "maximum_total_playback_seconds": seconds_per_trial * len(trials),
        "trials": [asdict(trial) for trial in trials],
        "label_boundary": (
            "scheduled token identities are post-hoc ground truth only and do not "
            "enter codebook or transition fitting"
        ),
        "real_io_requires_explicit_authorization": True,
    }
    if protocol_variant is not None:
        manifest["protocol_variant"] = protocol_variant
    if calibration_normalization_std_floor is not None:
        manifest["representation"] = {
            "base": "12_band_log_geometric_spectrum",
            "unlabeled_trial_calibration": "featurewise_zscore_then_unit_norm",
            "calibration_observation_count": len(CALIBRATION_SEQUENCE),
            "standard_deviation_floor": calibration_normalization_std_floor,
            "selection_boundary": "selected on attempt-2 development captures only",
        }
    return manifest


def _token_pcm(
    family: str,
    token: int,
    *,
    config: StructuredProtocolConfig,
) -> np.ndarray:
    if not 0 <= token < TOKEN_COUNT:
        raise ValueError("unknown token")
    if family == "tones":
        pcm = synthesize_tone(
            TONE_FREQUENCIES_HZ[token],
            duration_ms=config.token_duration_ms,
            sample_rate=config.sample_rate,
        )
    elif family == "vowels":
        pcm = synthesize_vowel(
            VOWEL_TOKENS[token],
            duration_ms=config.token_duration_ms,
            sample_rate=config.sample_rate,
            f0_hz=140.0,
        )
    else:
        raise ValueError(f"unknown structured stimulus family: {family!r}")
    return pcm * np.float32(0.15)


def render_structured_trial(
    plan: StructuredTrialPlan,
    config: StructuredProtocolConfig | None = None,
) -> RenderedStructuredTrial:
    config = config or StructuredProtocolConfig()
    chirp = synthesize_chirp(
        250.0,
        3500.0,
        duration_ms=config.alignment_chirp_ms,
        sample_rate=config.sample_rate,
    ) * np.float32(0.12)
    silence = np.zeros(
        int(config.alignment_silence_ms * config.sample_rate / 1000),
        dtype=np.float32,
    )
    gap = np.zeros(
        int(config.gap_duration_ms * config.sample_rate / 1000), dtype=np.float32
    )
    chunks: list[np.ndarray] = [chirp, silence]
    cursor = len(chirp) + len(silence)
    windows: list[tuple[int, int]] = []
    sequence = (*plan.calibration_sequence, *plan.grammar_sequence)
    for token in sequence:
        token_pcm = _token_pcm(plan.family, token, config=config)
        start = cursor
        end = start + len(token_pcm)
        windows.append((start, end))
        chunks.extend((token_pcm, gap))
        cursor = end + len(gap)
    return RenderedStructuredTrial(
        plan=plan,
        pcm=np.concatenate(chunks).astype(np.float32),
        token_windows=tuple(windows),
        calibration_count=len(plan.calibration_sequence),
    )


def _fake_acoustic_capture(
    pcm: np.ndarray,
    *,
    seed: int,
    config: StructuredProtocolConfig,
) -> tuple[np.ndarray, int]:
    delay = int(config.fake_delay_ms * config.sample_rate / 1000)
    echo_delay = int(config.fake_echo_delay_ms * config.sample_rate / 1000)
    tail = delay + echo_delay + int(0.1 * config.sample_rate)
    captured = np.zeros(len(pcm) + tail, dtype=np.float64)
    captured[delay : delay + len(pcm)] += config.fake_attenuation * pcm
    echo_start = delay + echo_delay
    captured[echo_start : echo_start + len(pcm)] += (
        config.fake_echo_attenuation * pcm
    )
    rng = np.random.default_rng(seed + 991)
    captured += rng.normal(0.0, config.fake_noise_rms, size=len(captured))
    return np.clip(captured, -1.0, 1.0).astype(np.float32), delay


def _spectral_signature(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.asarray(segment, dtype=np.float64)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1)
    if mono.size < 64:
        raise ValueError("token segment is too short")
    margin = min(int(0.015 * sample_rate), mono.size // 8)
    if margin:
        mono = mono[margin:-margin]
    centered = mono - float(np.mean(mono))
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequencies = np.fft.rfftfreq(len(centered), 1.0 / sample_rate)
    edges = np.geomspace(80.0, min(5000.0, sample_rate * 0.45), 13)
    # A signal-relative floor keeps empty bands finite without allowing room
    # noise (or exact digital silence) to define the state geometry.
    energy_floor = max(float(np.sum(spectrum)) * 1e-8, 1e-18)
    energies = []
    for low, high in zip(edges, edges[1:]):
        selected = spectrum[(frequencies >= low) & (frequencies < high)]
        energies.append(float(np.sum(selected)) + energy_floor)
    signature = np.log(np.asarray(energies, dtype=np.float64))
    signature -= float(np.mean(signature))
    scale = float(np.std(signature))
    return signature / max(scale, 1e-6)


def _captured_signatures(
    captured_path: Path,
    trial: RenderedStructuredTrial,
    *,
    alignment_samples: int,
) -> list[np.ndarray]:
    samples, sample_rate = read_pcm_wav(captured_path)
    signatures = []
    for start, end in trial.token_windows:
        aligned_start = alignment_samples + start
        aligned_end = alignment_samples + end
        if aligned_start < 0 or aligned_end > len(samples):
            raise ValueError("captured trial does not contain every token window")
        signatures.append(
            _spectral_signature(samples[aligned_start:aligned_end], sample_rate)
        )
    return signatures


def calibrate_signatures_by_trial(
    rendered: list[RenderedStructuredTrial],
    signatures: dict[str, list[np.ndarray]],
    *,
    standard_deviation_floor: float,
) -> dict[str, list[np.ndarray]]:
    """Remove trial-level spectral shift using only four unlabeled calibrators."""

    if standard_deviation_floor <= 0.0:
        raise ValueError("calibration standard-deviation floor must be positive")
    calibrated: dict[str, list[np.ndarray]] = {}
    for trial in rendered:
        observations = np.asarray(signatures[trial.plan.trial_id], dtype=np.float64)
        calibration = observations[: trial.calibration_count]
        center = np.mean(calibration, axis=0)
        scale = np.maximum(
            np.std(calibration, axis=0), standard_deviation_floor
        )
        normalized = (observations - center) / scale
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized /= np.maximum(norms, 1e-6)
        if not bool(np.all(np.isfinite(normalized))):
            raise ValueError("calibrated spectral observations must be finite")
        calibrated[trial.plan.trial_id] = [row.copy() for row in normalized]
    return calibrated


def _accuracy(actual: list[int], predicted: list[int]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("accuracy requires equal non-empty sequences")
    return sum(a == b for a, b in zip(actual, predicted)) / len(actual)


def _best_cluster_purity(states: list[int], tokens: list[int]) -> float:
    if len(states) != len(tokens) or not states:
        raise ValueError("purity requires equal non-empty sequences")
    best = 0
    for permutation in itertools.permutations(range(TOKEN_COUNT)):
        matched = sum(
            permutation[state] == token for state, token in zip(states, tokens)
        )
        best = max(best, matched)
    return best / len(states)


def _evaluate_family(
    family: str,
    rendered: list[RenderedStructuredTrial],
    signatures: dict[str, list[np.ndarray]],
) -> dict[str, object]:
    development = [trial for trial in rendered if trial.plan.split == "dev"]
    held_out = [trial for trial in rendered if trial.plan.split == "held_out"]
    first = development[0]
    initial = np.asarray(signatures[first.plan.trial_id][: first.calibration_count])
    codebook = LocalAcousticCodebook(initial)
    for trial in development:
        for observation in signatures[trial.plan.trial_id]:
            codebook.observe(observation)

    transition = LocalTransitionPredictor()
    development_states: list[int] = []
    for trial in development:
        trial_states = [
            codebook.assign(item)
            for item in signatures[trial.plan.trial_id][trial.calibration_count :]
        ]
        development_states.extend(trial_states)
        transition.observe_sequence(trial_states)
    unigram = int(np.argmax(np.bincount(development_states, minlength=TOKEN_COUNT)))

    reports = []
    all_states: list[int] = []
    all_tokens: list[int] = []
    for trial in held_out:
        observations = signatures[trial.plan.trial_id][trial.calibration_count :]
        states = [codebook.assign(item) for item in observations]
        tokens = list(trial.plan.grammar_sequence)
        actual = states[1:]
        predicted = [transition.predict(state) for state in states[:-1]]
        persistence = states[:-1]
        unigram_predictions = [unigram] * len(actual)
        distribution = np.bincount(development_states, minlength=TOKEN_COUNT).astype(
            np.float64
        )
        distribution /= float(np.sum(distribution))
        moving_predictions = []
        for current in states[:-1]:
            distribution *= 0.5
            distribution[current] += 0.5
            moving_predictions.append(int(np.argmax(distribution)))
        learned_accuracy = _accuracy(actual, predicted)
        baselines = {
            "persistence": _accuracy(actual, persistence),
            "development_unigram": _accuracy(actual, unigram_predictions),
            "causal_moving_state": _accuracy(actual, moving_predictions),
        }
        best_baseline = max(baselines.values())
        reports.append(
            {
                "trial_id": trial.plan.trial_id,
                "seed": trial.plan.seed,
                "transition_accuracy": learned_accuracy,
                "baselines": baselines,
                "best_baseline_accuracy": best_baseline,
                "accuracy_improvement": learned_accuracy - best_baseline,
                "prediction_count": len(actual),
                "prediction_residual_count": sum(
                    expected != observed
                    for expected, observed in zip(predicted, actual)
                ),
            }
        )
        all_states.extend(states)
        all_tokens.extend(tokens)

    total_predictions = sum(int(report["prediction_count"]) for report in reports)
    learned_correct = sum(
        float(report["transition_accuracy"]) * int(report["prediction_count"])
        for report in reports
    )
    baseline_correct = sum(
        float(report["best_baseline_accuracy"]) * int(report["prediction_count"])
        for report in reports
    )
    occupancy = len(set(all_states))
    return {
        "family": family,
        "development_trial_count": len(development),
        "held_out_trial_count": len(held_out),
        "codebook_state_count": TOKEN_COUNT,
        "held_out_codebook_occupancy": occupancy,
        "post_hoc_token_cluster_purity": _best_cluster_purity(
            all_states, all_tokens
        ),
        "transition_accuracy": learned_correct / total_predictions,
        "best_baseline_accuracy": baseline_correct / total_predictions,
        "accuracy_improvement": (learned_correct - baseline_correct)
        / total_predictions,
        "minimum_trial_improvement": min(
            float(report["accuracy_improvement"]) for report in reports
        ),
        "bounded_state_scalars": (
            codebook.state_scalars + transition.state_scalars
        ),
        "top_down_prediction_consumed": True,
        "reports": reports,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _acceptance_gates(
    family_reports: dict[str, dict[str, object]],
    *,
    profile: str,
) -> dict[str, dict[str, bool]]:
    if profile not in {"original", "h2_predictive_advantage_v2"}:
        raise ValueError(f"unknown structured acceptance profile: {profile}")
    gates: dict[str, dict[str, bool]] = {}
    for family, report in family_reports.items():
        common = {
            "occupancy_passed": int(report["held_out_codebook_occupancy"]) == 4,
            "improvement_passed": float(report["accuracy_improvement"]) >= 0.20,
            "bounded_state_passed": int(report["bounded_state_scalars"]) <= 68,
            "top_down_consumed_passed": bool(
                report["top_down_prediction_consumed"]
            ),
        }
        if profile == "original":
            common.update(
                {
                    "purity_passed": float(
                        report["post_hoc_token_cluster_purity"]
                    )
                    >= 0.85,
                    "accuracy_passed": float(report["transition_accuracy"])
                    >= 0.75,
                    "each_trial_improvement_passed": float(
                        report["minimum_trial_improvement"]
                    )
                    >= 0.10,
                }
            )
        else:
            # VISION H2 requires predictive advantage, bounded state, and
            # consumed top-down predictions. Purity remains a diagnostic; it
            # is not relaxed into an easier replacement gate after attempt 2.
            common["each_trial_improvement_passed"] = float(
                report["minimum_trial_improvement"]
            ) >= 0.15
        gates[family] = common
    return gates


def _run_real_structured_protocol(
    *,
    out_dir: Path,
    config: StructuredProtocolConfig,
    manifest: dict[str, object],
    playback_fn: PlaybackFn,
    capture_fn: CaptureFn,
    geometry: dict[str, object],
    real_hardware_claim: bool,
    calibration_normalization_std_floor: float | None,
    acceptance_profile: str,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedStructuredTrial] = []
    signatures: dict[str, list[np.ndarray]] = {}
    artifacts = []
    trial_payloads = manifest["trials"]
    assert isinstance(trial_payloads, list)
    for trial_payload in trial_payloads:
        assert isinstance(trial_payload, dict)
        trial = StructuredTrialPlan(
            trial_id=str(trial_payload["trial_id"]),
            family=str(trial_payload["family"]),
            split=str(trial_payload["split"]),
            seed=int(trial_payload["seed"]),
            calibration_sequence=tuple(trial_payload["calibration_sequence"]),
            grammar_sequence=tuple(trial_payload["grammar_sequence"]),
        )
        rendered_trial = render_structured_trial(trial, config)
        efference_path = out_dir / f"{trial.trial_id}_efference.wav"
        captured_path = out_dir / f"{trial.trial_id}_real_capture.wav"
        write_wav(str(efference_path), rendered_trial.pcm, config.sample_rate)
        capture_duration_ms = int(
            math.ceil(len(rendered_trial.pcm) * 1000.0 / config.sample_rate)
        ) + 1000
        playback_started = False

        def _tracked_playback(path: str) -> dict[str, object]:
            nonlocal playback_started
            playback_started = True
            return playback_fn(path)

        try:
            io_report = _capture_during_playback(
                capture_path=str(captured_path),
                playback_path=str(efference_path),
                sample_rate=config.sample_rate,
                capture_duration_ms=capture_duration_ms,
                capture_fn=capture_fn,
                playback_fn=_tracked_playback,
                capture_preroll_ms=100,
            )
            playback_report = io_report["playback"]
            capture_report = io_report["capture"]
            if not isinstance(playback_report, dict) or not isinstance(
                capture_report, dict
            ):
                raise ValueError("I/O callbacks must return provenance reports")
            if real_hardware_claim and (
                playback_report.get("backend") != "ffmpeg_audiotoolbox"
                or capture_report.get("backend") != "ffmpeg_avfoundation"
            ):
                raise ValueError("real I/O backend provenance was not verified")
            playback_offset_ms = float(
                io_report["playback_offset_from_capture_ms"]
            )
            captured_samples, captured_rate = read_pcm_wav(captured_path)
            if captured_rate != config.sample_rate:
                raise ValueError("real capture sample rate mismatch")
            if real_hardware_claim and captured_samples.shape[1] != 2:
                raise ValueError("XVF3800 capture must expose two fused channels")
            chirp_samples = int(
                config.alignment_chirp_ms * config.sample_rate / 1000
            )
            detected_onset_ms = estimate_playback_latency_ms(
                efference_pcm=rendered_trial.pcm[:chirp_samples],
                captured_pcm=np.mean(captured_samples, axis=1),
                sample_rate=config.sample_rate,
                expected_onset_ms=playback_offset_ms,
                causal_window_ms=500.0,
                min_peak_correlation=0.08,
            )
            if detected_onset_ms is None:
                raise ValueError("alignment chirp was not recovered")
            alignment_samples = int(
                round(detected_onset_ms * config.sample_rate / 1000.0)
            )
            signatures[trial.trial_id] = _captured_signatures(
                captured_path,
                rendered_trial,
                alignment_samples=alignment_samples,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                **manifest,
                "verdict": "real_io_failed",
                "real_hardware": real_hardware_claim,
                "live_playback": playback_started,
                "geometry": geometry,
                "failed_trial_id": trial.trial_id,
                "completed_trial_count": len(artifacts),
                "error": f"{type(exc).__name__}: {exc}",
                "artifacts": artifacts,
            }
        rendered.append(rendered_trial)
        artifacts.append(
            {
                "trial_id": trial.trial_id,
                "efference_wav": str(efference_path.resolve()),
                "efference_sha256": _sha256(efference_path),
                "captured_wav": str(captured_path.resolve()),
                "captured_sha256": _sha256(captured_path),
                "capture_provenance": (
                    "real_synchronized_playback_capture"
                    if real_hardware_claim
                    else "injected_io_test_double"
                ),
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
                "playback": playback_report,
                "capture": capture_report,
            }
        )

    if calibration_normalization_std_floor is not None:
        signatures = calibrate_signatures_by_trial(
            rendered,
            signatures,
            standard_deviation_floor=calibration_normalization_std_floor,
        )
    family_reports = {
        family: _evaluate_family(
            family,
            [trial for trial in rendered if trial.plan.family == family],
            signatures,
        )
        for family in ("tones", "vowels")
    }
    gates = _acceptance_gates(family_reports, profile=acceptance_profile)
    all_passed = all(all(family_gate.values()) for family_gate in gates.values())
    if not real_hardware_claim:
        verdict = "injected_real_runner_test_complete"
    else:
        verdict = (
            "real_io_acceptance_passed"
            if all_passed
            else "real_io_acceptance_failed"
        )
    return {
        **manifest,
        "verdict": verdict,
        "real_hardware": real_hardware_claim,
        "live_playback": real_hardware_claim,
        "geometry": geometry,
        "family_reports": family_reports,
        "acceptance": gates,
        "all_acceptance_gates_passed": all_passed,
        "artifacts": artifacts,
        "error": None,
    }


def run_structured_prediction_protocol(
    *,
    out_dir: str | Path,
    fake_io: bool = False,
    real_io: bool = False,
    bounded_playback_authorized: bool = False,
    config: StructuredProtocolConfig | None = None,
    playback_fn: PlaybackFn | None = None,
    capture_fn: CaptureFn | None = None,
    geometry: dict[str, object] | None = None,
    real_hardware_claim: bool = False,
    development_seeds: tuple[int, ...] | None = None,
    held_out_seeds: tuple[int, ...] | None = None,
    protocol_variant: str | None = None,
    calibration_normalization_std_floor: float | None = None,
    acceptance_profile: str = "original",
) -> dict[str, object]:
    """Plan or run the fake protocol; never perform real I/O in this helper."""

    config = config or StructuredProtocolConfig()
    if fake_io and real_io:
        raise ValueError("fake I/O and real I/O are mutually exclusive")
    manifest = build_structured_protocol_manifest(
        config,
        development_seeds=development_seeds,
        held_out_seeds=held_out_seeds,
        protocol_variant=protocol_variant,
        calibration_normalization_std_floor=calibration_normalization_std_floor,
    )
    if real_io:
        if not bounded_playback_authorized:
            return {
                **manifest,
                "verdict": "blocked_bounded_live_io_not_authorized",
                "real_hardware": False,
                "live_playback": False,
            }
        if playback_fn is None or capture_fn is None or geometry is None:
            return {
                **manifest,
                "verdict": "blocked_missing_real_io_configuration",
                "real_hardware": False,
                "live_playback": False,
            }
        return _run_real_structured_protocol(
            out_dir=Path(out_dir),
            config=config,
            manifest=manifest,
            playback_fn=playback_fn,
            capture_fn=capture_fn,
            geometry=geometry,
            real_hardware_claim=real_hardware_claim,
            calibration_normalization_std_floor=(
                calibration_normalization_std_floor
            ),
            acceptance_profile=acceptance_profile,
        )
    if not fake_io:
        return {
            **manifest,
            "verdict": "planned_no_io",
            "real_hardware": False,
            "live_playback": False,
        }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedStructuredTrial] = []
    signatures: dict[str, list[np.ndarray]] = {}
    artifacts = []
    for trial_payload in manifest["trials"]:  # type: ignore[index]
        trial = StructuredTrialPlan(
            trial_id=str(trial_payload["trial_id"]),
            family=str(trial_payload["family"]),
            split=str(trial_payload["split"]),
            seed=int(trial_payload["seed"]),
            calibration_sequence=tuple(trial_payload["calibration_sequence"]),
            grammar_sequence=tuple(trial_payload["grammar_sequence"]),
        )
        rendered_trial = render_structured_trial(trial, config)
        efference_path = output / f"{trial.trial_id}_efference.wav"
        captured_path = output / f"{trial.trial_id}_fake_capture.wav"
        write_wav(str(efference_path), rendered_trial.pcm, config.sample_rate)
        captured, delay = _fake_acoustic_capture(
            rendered_trial.pcm,
            seed=trial.seed,
            config=config,
        )
        write_wav(str(captured_path), captured, config.sample_rate)
        signatures[trial.trial_id] = _captured_signatures(
            captured_path,
            rendered_trial,
            alignment_samples=delay,
        )
        rendered.append(rendered_trial)
        artifacts.append(
            {
                "trial_id": trial.trial_id,
                "efference_wav": str(efference_path.resolve()),
                "efference_sha256": _sha256(efference_path),
                "captured_wav": str(captured_path.resolve()),
                "captured_sha256": _sha256(captured_path),
                "capture_provenance": "synthetic_fake_acoustic_channel",
                "alignment_samples": delay,
            }
        )

    if calibration_normalization_std_floor is not None:
        signatures = calibrate_signatures_by_trial(
            rendered,
            signatures,
            standard_deviation_floor=calibration_normalization_std_floor,
        )
    family_reports = {
        family: _evaluate_family(
            family,
            [trial for trial in rendered if trial.plan.family == family],
            signatures,
        )
        for family in ("tones", "vowels")
    }
    if acceptance_profile == "original":
        fake_gate = all(
            int(report["held_out_codebook_occupancy"]) == TOKEN_COUNT
            and float(report["post_hoc_token_cluster_purity"]) >= 0.9
            and float(report["accuracy_improvement"]) >= 0.2
            for report in family_reports.values()
        )
    else:
        fake_gate = all(
            all(gates.values())
            for gates in _acceptance_gates(
                family_reports, profile=acceptance_profile
            ).values()
        )
    return {
        **manifest,
        "verdict": "fake_io_plumbing_passed" if fake_gate else "fake_io_gate_failed",
        "evidence_class": "synthetic_fake_io",
        "promotion_allowed": False,
        "real_hardware": False,
        "live_playback": False,
        "family_reports": family_reports,
        "artifacts": artifacts,
    }
