"""Natural-speech extension of the bounded H3 self/world binding task.

The runtime representation is label-blind.  A known alignment chirp estimates
the self loudspeaker path, a bounded FIR predicts the commanded self waveform,
and unexplained residual energy supplies bottom-up world evidence.  Corpus
condition labels and trial conditions enter scoring only.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .io.motor_plan import synthesize_chirp, write_wav
from .io.wav_frames import read_pcm_wav
from .live_speak_smoke import _capture_during_playback, estimate_playback_latency_ms
from .real_workspace_binding import BoundedEfferenceWorkspace
from .structured_workspace_binding import play_workspace_tracks_concurrently


SAMPLE_RATE = 16_000
CHIRP_MS = 120
ALIGNMENT_SILENCE_MS = 180
SPEECH_MS = 800
TAIL_SILENCE_MS = 200
SELF_PEAK = 0.12
WORLD_PEAK = 0.80
FIR_SAMPLES = 128
DECONVOLUTION_RIDGE_FRACTION = 0.001
MAX_TRIAL_PLAYBACK_SECONDS = 2.0
MAX_TOTAL_PLAYBACK_SECONDS = 25.0
MAX_TOTAL_CAPTURE_SECONDS = 45.0


@dataclass(frozen=True)
class NaturalSpeechPair:
    split: str
    index: int
    self_filename: str
    self_sha256: str
    world_filename: str
    world_sha256: str


@dataclass(frozen=True)
class NaturalSpeechBindingTrial:
    trial_id: str
    split: str
    condition: str
    pair_index: int
    efference_id: str | None
    expected_world_usable: bool


@dataclass(frozen=True)
class PreparedSpeech:
    filename: str
    sha256: str
    source_sample_rate: int
    source_channels: int
    source_frames: int
    selected_window_index: int
    selected_start_seconds: float
    pcm: np.ndarray


@dataclass(frozen=True)
class RenderedNaturalSpeechTracks:
    trial: NaturalSpeechBindingTrial
    self_pcm: np.ndarray
    world_pcm: np.ndarray
    reference_pcm: np.ndarray
    stimulus_window: tuple[int, int]
    chirp_source: str


PAIRS = (
    NaturalSpeechPair("development", 0, "KRA01D.wav", "8dd90d2bd1ca0322f0c55466439a2079f197d761b7081a966ed928b1ddd25a2c", "MBE02B.wav", "df5412fde22f6a32050992d7ece263b1dd6542a5b15b9c0234eb7262d139de2e"),
    NaturalSpeechPair("development", 1, "MES06D.wav", "bcce51436783e6c836147c962ff4049e72b0472ae7ff397dd6e90ff3b14021f6", "PNG01B.wav", "538991b4f8bc7339baf8a8979589d85d7e90cd6f95c66b327b46943f173786fd"),
    NaturalSpeechPair("development", 2, "TSI03D.wav", "f6af43284674bf221329587ec87681d274069f16f21e893945b06784c858f61e", "VAN21B.wav", "76b1d5e8e07b2ff9e57a05bb5ebfc250e90ac2a41109d99500ab762e4e3651ef"),
    NaturalSpeechPair("held_out", 0, "ACO13D.wav", "35d6701f018d9e6c811fb67b4326f78fa22dbed2a64252f637912a36d71a807b", "MEN14B.wav", "2da65dc3eb18f0a03e9096044c18821adc25f9331cdd869e7a5f0432a1a3ccb3"),
    NaturalSpeechPair("held_out", 1, "HAD09D.wav", "68443f7f8607ed94ab312254b308ef55c99f1b08c6dabb9e6232b2d8e1fe949e", "JEN05B.wav", "d6919daaf5a873db8753748c6d44a07a2bb032ab5a30631612f11bc192821bc2"),
    NaturalSpeechPair("held_out", 2, "LIM05D.wav", "db5837f5d3e04d68f96387153d56287a29164c9b87560da0a833727ea2141946", "NYA14B.wav", "a2155e3316283a6505025067db7856b7ae57adeb6efb4661e1a8f40823aff05c"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_natural_speech_trials() -> tuple[NaturalSpeechBindingTrial, ...]:
    trials: list[NaturalSpeechBindingTrial] = []
    for pair in PAIRS:
        conditions = ("self_only", "world_only")
        if pair.split == "held_out":
            conditions = (*conditions, "overlap")
        for condition in conditions:
            has_self = condition != "world_only"
            has_world = condition != "self_only"
            trials.append(
                NaturalSpeechBindingTrial(
                    trial_id=f"{pair.split}_{condition}_{pair.index}",
                    split=pair.split,
                    condition=condition,
                    pair_index=pair.index,
                    efference_id=(
                        f"held_self_{pair.index}"
                        if has_self and pair.split == "held_out"
                        else f"dev_self_{pair.index}" if has_self else None
                    ),
                    expected_world_usable=has_world,
                )
            )
    return tuple(trials)


def build_natural_speech_protocol() -> dict[str, object]:
    trials = build_natural_speech_trials()
    seconds_per_trial = (CHIRP_MS + ALIGNMENT_SILENCE_MS + SPEECH_MS + TAIL_SILENCE_MS) / 1000.0
    return {
        "mode": "natural_speech_workspace_binding_protocol",
        "trial_count": len(trials),
        "seconds_per_trial": seconds_per_trial,
        "maximum_total_playback_seconds": len(trials) * seconds_per_trial,
        "config": {
            "sample_rate": SAMPLE_RATE,
            "alignment_chirp_ms": CHIRP_MS,
            "alignment_silence_ms": ALIGNMENT_SILENCE_MS,
            "speech_ms": SPEECH_MS,
            "tail_silence_ms": TAIL_SILENCE_MS,
            "self_peak_absolute": SELF_PEAK,
            "world_peak_absolute": WORLD_PEAK,
            "fir_samples": FIR_SAMPLES,
            "deconvolution_ridge_fraction": DECONVOLUTION_RIDGE_FRACTION,
        },
        "trials": [asdict(trial) for trial in trials],
        "capacity_sweep": [1, 2, 3, 4],
        "strategies": [
            "flat",
            "fixed_self_window",
            "scheduled_identity_plus_world_residual",
        ],
        "label_boundary": (
            "condition and expected_world_usable enter scoring only; runtime "
            "consumes a known efference identity, efference PCM, and microphone PCM"
        ),
        "promotion_allowed": False,
    }


def _resample_linear(mono: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return np.asarray(mono, dtype=np.float32)
    count = max(1, int(round(len(mono) * SAMPLE_RATE / source_rate)))
    source_positions = np.arange(len(mono), dtype=np.float64)
    target_positions = np.arange(count, dtype=np.float64) * source_rate / SAMPLE_RATE
    return np.interp(target_positions, source_positions, mono).astype(np.float32)


def prepare_speech(path: str | Path, *, expected_sha256: str, peak: float) -> PreparedSpeech:
    source = Path(path)
    actual_hash = _sha256(source)
    if actual_hash != expected_sha256:
        raise ValueError(f"source hash mismatch: {source.name}")
    samples, source_rate = read_pcm_wav(source)
    mono = np.mean(samples, axis=1, dtype=np.float64).astype(np.float32)
    resampled = _resample_linear(mono, source_rate)
    window_samples = int(SPEECH_MS * SAMPLE_RATE / 1000)
    window_count = len(resampled) // window_samples
    if window_count < 1:
        raise ValueError(f"source is shorter than {SPEECH_MS} ms: {source.name}")
    energies = [
        float(np.mean(resampled[i * window_samples : (i + 1) * window_samples] ** 2))
        for i in range(window_count)
    ]
    selected = int(np.argmax(energies))
    start = selected * window_samples
    pcm = resampled[start : start + window_samples].copy()
    pcm -= np.float32(np.mean(pcm))
    source_peak = float(np.max(np.abs(pcm)))
    if source_peak <= 1e-8:
        raise ValueError(f"selected source window is silent: {source.name}")
    pcm *= np.float32(peak / source_peak)
    return PreparedSpeech(
        filename=source.name,
        sha256=actual_hash,
        source_sample_rate=source_rate,
        source_channels=samples.shape[1],
        source_frames=len(samples),
        selected_window_index=selected,
        selected_start_seconds=start / SAMPLE_RATE,
        pcm=pcm.astype(np.float32),
    )


def prepare_natural_speech_panel(root: str | Path) -> dict[tuple[str, int], tuple[PreparedSpeech, PreparedSpeech]]:
    base = Path(root)
    prepared = {}
    for pair in PAIRS:
        prepared[(pair.split, pair.index)] = (
            prepare_speech(base / pair.self_filename, expected_sha256=pair.self_sha256, peak=SELF_PEAK),
            prepare_speech(base / pair.world_filename, expected_sha256=pair.world_sha256, peak=WORLD_PEAK),
        )
    return prepared


def render_natural_speech_tracks(
    trial: NaturalSpeechBindingTrial,
    prepared: dict[tuple[str, int], tuple[PreparedSpeech, PreparedSpeech]],
) -> RenderedNaturalSpeechTracks:
    self_source, world_source = prepared[(trial.split, trial.pair_index)]
    chirp = np.float32(0.12) * synthesize_chirp(250.0, 3500.0, duration_ms=CHIRP_MS, sample_rate=SAMPLE_RATE)
    empty_chirp = np.zeros_like(chirp)
    pre = np.zeros(int(ALIGNMENT_SILENCE_MS * SAMPLE_RATE / 1000), dtype=np.float32)
    tail = np.zeros(int(TAIL_SILENCE_MS * SAMPLE_RATE / 1000), dtype=np.float32)
    silence = np.zeros(int(SPEECH_MS * SAMPLE_RATE / 1000), dtype=np.float32)
    has_self = trial.condition != "world_only"
    has_world = trial.condition != "self_only"
    self_speech = self_source.pcm if has_self else silence
    world_speech = world_source.pcm if has_world else silence
    if has_self:
        self_chirp, world_chirp, chirp_source = chirp, empty_chirp, "self"
    else:
        self_chirp, world_chirp, chirp_source = empty_chirp, chirp, "world"
    self_pcm = np.concatenate((self_chirp, pre, self_speech, tail)).astype(np.float32)
    world_pcm = np.concatenate((world_chirp, pre, world_speech, tail)).astype(np.float32)
    start = len(chirp) + len(pre)
    return RenderedNaturalSpeechTracks(
        trial=trial,
        self_pcm=self_pcm,
        world_pcm=world_pcm,
        reference_pcm=self_pcm if has_self else world_pcm,
        stimulus_window=(start, start + len(silence)),
        chirp_source=chirp_source,
    )


def _estimate_causal_fir(known_chirp: np.ndarray, observed: np.ndarray) -> np.ndarray:
    required = len(known_chirp) + FIR_SAMPLES - 1
    if len(observed) < required:
        raise ValueError("capture does not contain the FIR estimation window")
    fft_size = 1 << (required - 1).bit_length()
    known_spectrum = np.fft.rfft(known_chirp, fft_size)
    observed_spectrum = np.fft.rfft(observed[:required], fft_size)
    power = np.abs(known_spectrum) ** 2
    ridge = DECONVOLUTION_RIDGE_FRACTION * max(float(np.max(power)), 1e-18)
    transfer = observed_spectrum * np.conjugate(known_spectrum) / (power + ridge)
    return np.fft.irfft(transfer, fft_size)[:FIR_SAMPLES].astype(np.float32)


def _residual_ratio(rendered: RenderedNaturalSpeechTracks, capture: np.ndarray, alignment: int) -> float:
    mono = np.asarray(capture, dtype=np.float64)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1)
    if mono.ndim != 1:
        raise ValueError("natural-speech capture must be mono or frame-by-channel")
    start, end = rendered.stimulus_window
    captured_start = alignment + start
    captured_end = alignment + end
    if alignment < 0 or captured_end > len(mono):
        raise ValueError(f"capture does not contain trial window: {rendered.trial.trial_id}")
    observed_speech = mono[captured_start:captured_end]
    observed_rms = float(np.sqrt(np.mean(observed_speech**2)))
    if rendered.trial.efference_id is None:
        return 1.0 if observed_rms > 1e-9 else 0.0
    chirp_samples = int(CHIRP_MS * SAMPLE_RATE / 1000)
    known_chirp = rendered.self_pcm[:chirp_samples]
    observed_chirp = mono[alignment : alignment + chirp_samples + FIR_SAMPLES - 1]
    impulse = _estimate_causal_fir(known_chirp, observed_chirp)
    known_self = rendered.self_pcm[start:end]
    predicted_self = np.convolve(known_self, impulse)[: len(known_self)]
    residual = observed_speech - predicted_self
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    return residual_rms / max(observed_rms, 1e-12)


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


def _evaluate_strategy(trials: list[NaturalSpeechBindingTrial], evidence: dict[str, float], *, threshold: float, capacity: int, strategy: str) -> dict[str, object]:
    workspace = BoundedEfferenceWorkspace(capacity)
    for index in range(3):
        workspace.push(f"held_self_{index}")
    actual: list[bool] = []
    predicted: list[bool] = []
    for trial in trials:
        bound = workspace.contains(trial.efference_id)
        residual_world = evidence[trial.trial_id] >= threshold
        if strategy == "flat":
            usable = True
        elif strategy == "fixed_self_window":
            usable = not bound
        elif strategy == "scheduled_identity_plus_world_residual":
            usable = not bound or residual_world
        else:
            raise ValueError(f"unknown natural-speech workspace strategy: {strategy}")
        actual.append(trial.expected_world_usable)
        predicted.append(usable)
    return {
        "strategy": strategy,
        "capacity": capacity,
        "bounded_binding_slots": capacity,
        "world_evidence_threshold_scalars": 1,
        "metrics": _classification(actual, predicted),
    }


def score_natural_speech_evidence(evidence: dict[str, float]) -> dict[str, object]:
    trials = list(build_natural_speech_trials())
    development = [trial for trial in trials if trial.split == "development"]
    held_out = [trial for trial in trials if trial.split == "held_out"]
    max_self = max(evidence[trial.trial_id] for trial in development if not trial.expected_world_usable)
    min_world = min(evidence[trial.trial_id] for trial in development if trial.expected_world_usable)
    if min_world <= max_self:
        raise ValueError("natural-speech residual evidence has no development gap")
    threshold = max_self + 0.10 * (min_world - max_self)
    strategies = ("flat", "fixed_self_window", "scheduled_identity_plus_world_residual")
    reports = {
        str(capacity): {
            strategy: _evaluate_strategy(held_out, evidence, threshold=threshold, capacity=capacity, strategy=strategy)
            for strategy in strategies
        }
        for capacity in (1, 2, 3, 4)
    }
    selected = reports["3"]
    scheduled = selected["scheduled_identity_plus_world_residual"]["metrics"]
    flat = selected["flat"]["metrics"]
    fixed = selected["fixed_self_window"]["metrics"]
    criteria = {
        "scheduled_f1_at_least_0_90": float(scheduled["f1"]) >= 0.90,
        "scheduled_improves_flat_f1_by_at_least_0_20": float(scheduled["f1"]) - float(flat["f1"]) >= 0.20 - 1e-12,
        "scheduled_world_recall_at_least_0_90": float(scheduled["world_recall"]) >= 0.90,
        "scheduled_self_suppression_specificity_at_least_0_90": float(scheduled["self_suppression_specificity"]) >= 0.90,
        "fixed_window_collision_failure_exposed": float(fixed["world_recall"]) < float(scheduled["world_recall"]),
        "capacity_one_worse_than_three": float(reports["1"]["scheduled_identity_plus_world_residual"]["metrics"]["f1"]) < float(scheduled["f1"]),
        "capacity_three_plateaus_at_four": math.isclose(float(reports["4"]["scheduled_identity_plus_world_residual"]["metrics"]["f1"]), float(scheduled["f1"])),
    }
    return {
        "verdict": "capture_acceptance_passed" if all(criteria.values()) else "capture_acceptance_failed",
        "world_evidence": evidence,
        "world_evidence_threshold": threshold,
        "world_evidence_threshold_rule": "maximum development self-only residual ratio plus 10 percent of the clean gap to minimum development world-only ratio",
        "threshold_selected_from_development_only": True,
        "reports": reports,
        "criteria": criteria,
    }


def evaluate_natural_speech_captures(*, root: str | Path, captures: dict[str, np.ndarray], alignment_samples: dict[str, int]) -> dict[str, object]:
    prepared = prepare_natural_speech_panel(root)
    trials = list(build_natural_speech_trials())
    expected = {trial.trial_id for trial in trials}
    if set(captures) != expected or set(alignment_samples) != expected:
        raise ValueError("natural-speech capture and alignment maps must cover every trial")
    evidence = {}
    for trial in trials:
        rendered = render_natural_speech_tracks(trial, prepared)
        evidence[trial.trial_id] = _residual_ratio(rendered, captures[trial.trial_id], alignment_samples[trial.trial_id])
    return {
        **build_natural_speech_protocol(),
        **score_natural_speech_evidence(evidence),
        "evidence_class": "provided_chirp_aligned_natural_speech_capture_arrays",
        "promotion_allowed": False,
    }


def audit_natural_speech_preregistration(
    preregistration: Mapping[str, object],
    *,
    current_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Fail closed unless code, sources, hardware, bounds, and gates are frozen."""

    blockers: list[str] = []
    if preregistration.get("mode") != "natural_speech_workspace_binding_real_preregistration":
        blockers.append("mode is not the natural-speech H3 real preregistration")
    if preregistration.get("status") != "preregistered_before_live_execution":
        blockers.append("status is not preregistered_before_live_execution")
    if preregistration.get("bounded_playback_capture_authorized") is not True:
        blockers.append("bounded playback and capture are not authorized")
    if preregistration.get("spent_design_gate_passed") is not True:
        blockers.append("spent-data design gate is not recorded as passed")

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
    environment = hardware.get("environment")
    if not isinstance(input_route, Mapping):
        blockers.append("input route is missing")
        input_route = {}
    if not isinstance(self_route, Mapping):
        blockers.append("self source route is missing")
        self_route = {}
    if not isinstance(world_route, Mapping):
        blockers.append("world source route is missing")
        world_route = {}
    if not isinstance(environment, Mapping):
        blockers.append("environment is missing")
        environment = {}

    expected_input = {
        "avfoundation_device": ":0",
        "device": "reSpeaker XVF3800 4-Mic Array",
        "channels": 2,
        "sample_rate": SAMPLE_RATE,
    }
    expected_self = {
        "output_device_index": 3,
        "output_device": "Mac mini Speakers",
        "placement": "centered_directly_below_xvf3800",
        "vertical_distance_cm": 4,
        "system_volume_percent": 100,
        "playback_gain_db": -18.0,
    }
    expected_world = {
        "output_device_index": 2,
        "output_device": "External Headphones",
        "angle_degrees": 180,
        "distance_cm": 50,
        "hardware_volume": "approximately_50_percent_mechanical_travel",
        "playback_gain_db": -18.0,
    }
    for name, expected, actual in (
        ("input", expected_input, input_route),
        ("self", expected_self, self_route),
        ("world", expected_world, world_route),
    ):
        for field, value in expected.items():
            if actual.get(field) != value:
                blockers.append(f"{name} route mismatch: {field}")
    if environment.get("room") != "semi-quiet" or environment.get("known_background") != "laptop fan":
        blockers.append("room state does not match the confirmed environment")

    expected_bounds = {
        "trial_count": 15,
        "per_trial_playback_seconds_max": 2.0,
        "total_playback_seconds_max": 25.0,
        "total_capture_seconds_max": 45.0,
        "planned_playback_seconds": 19.5,
        "planned_capture_seconds": 28.5,
    }
    bounds = preregistration.get("bounds")
    if not isinstance(bounds, Mapping) or any(bounds.get(key) != value for key, value in expected_bounds.items()):
        blockers.append("live run bounds do not match the frozen protocol")

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
        "no_clipped_capture_samples_required": True,
    }
    acceptance = preregistration.get("acceptance")
    if not isinstance(acceptance, Mapping) or any(acceptance.get(key) != value for key, value in expected_acceptance.items()):
        blockers.append("acceptance criteria do not match the frozen natural-speech H3 gate")
    if preregistration.get("unresolved_fields") != []:
        blockers.append("preregistration still lists unresolved fields")
    return {
        "verdict": "preregistration_ready" if not blockers else "preregistration_blocked",
        "ready": not blockers,
        "blockers": blockers,
    }


def run_natural_speech_acquisition(
    *,
    root: str | Path,
    out_dir: str | Path,
    self_playback: Callable[[str], dict[str, object]],
    world_playback: Callable[[str], dict[str, object]],
    capture_fn: Callable[..., dict[str, object]],
    capture_preroll_ms: int = 100,
    capture_tail_ms: int = 500,
    real_hardware_claim: bool = False,
    bounded_playback_authorized: bool = False,
    hardware_preregistration: Mapping[str, object] | None = None,
    current_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Acquire the 15 frozen speech trials through source-specific callbacks."""

    protocol = build_natural_speech_protocol()
    if float(protocol["seconds_per_trial"]) > MAX_TRIAL_PLAYBACK_SECONDS:
        raise ValueError("natural-speech workspace trial exceeds the 2 second bound")
    if float(protocol["maximum_total_playback_seconds"]) > MAX_TOTAL_PLAYBACK_SECONDS:
        raise ValueError("natural-speech workspace playback exceeds the 25 second bound")
    if not (0 <= capture_preroll_ms <= 1000 and 0 <= capture_tail_ms <= 1000):
        raise ValueError("capture preroll and tail must each be between 0 and 1000 ms")
    bounded_capture_seconds = float(protocol["maximum_total_playback_seconds"]) + (
        (capture_preroll_ms + capture_tail_ms) / 1000.0 * int(protocol["trial_count"])
    )
    if bounded_capture_seconds > MAX_TOTAL_CAPTURE_SECONDS:
        raise ValueError("natural-speech workspace capture exceeds the 45 second bound")
    if real_hardware_claim:
        if not bounded_playback_authorized:
            raise ValueError("real natural-speech H3 execution requires bounded authorization")
        if hardware_preregistration is None or current_hashes is None:
            raise ValueError("real natural-speech H3 execution requires a frozen preregistration")
        audit = audit_natural_speech_preregistration(hardware_preregistration, current_hashes=current_hashes)
        if not audit["ready"]:
            raise ValueError("real natural-speech H3 preregistration blocked: " + "; ".join(str(item) for item in audit["blockers"]))

    # Hash and decode every external source before output creation or callbacks.
    prepared = prepare_natural_speech_panel(root)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    captures: dict[str, np.ndarray] = {}
    alignments: dict[str, int] = {}
    artifacts: list[dict[str, object]] = []
    total_playback_seconds = 0.0
    captures_with_clipping = 0
    total_clipped_samples = 0
    maximum_clipped_fraction = 0.0

    for trial in build_natural_speech_trials():
        rendered = render_natural_speech_tracks(trial, prepared)
        self_path = output / f"{trial.trial_id}_self.wav"
        world_path = output / f"{trial.trial_id}_world.wav"
        capture_path = output / f"{trial.trial_id}_capture.wav"
        write_wav(str(self_path), rendered.self_pcm, SAMPLE_RATE)
        write_wav(str(world_path), rendered.world_pcm, SAMPLE_RATE)
        capture_duration_ms = int(math.ceil(len(rendered.reference_pcm) * 1000.0 / SAMPLE_RATE)) + capture_preroll_ms + capture_tail_ms
        playback_started = False

        def _play_sources(_reference_path: str) -> dict[str, object]:
            nonlocal playback_started
            playback_started = True
            report = play_workspace_tracks_concurrently(
                self_path=str(self_path) if trial.condition != "world_only" else None,
                world_path=str(world_path) if trial.condition != "self_only" else None,
                self_playback=self_playback,
                world_playback=world_playback,
            )
            if not report["started"]:
                raise RuntimeError("one or more natural-speech sources did not start")
            return report

        try:
            io_report = _capture_during_playback(
                capture_path=str(capture_path),
                playback_path=str(self_path if rendered.chirp_source == "self" else world_path),
                sample_rate=SAMPLE_RATE,
                capture_duration_ms=capture_duration_ms,
                capture_fn=capture_fn,
                playback_fn=_play_sources,
                capture_preroll_ms=capture_preroll_ms,
            )
            captured_samples, captured_rate = read_pcm_wav(capture_path)
            if captured_rate != SAMPLE_RATE:
                raise ValueError("natural-speech workspace capture sample rate mismatch")
            if real_hardware_claim:
                capture_report = io_report.get("capture")
                playback_report = io_report.get("playback")
                if not isinstance(capture_report, Mapping) or capture_report.get("backend") != "ffmpeg_avfoundation":
                    raise ValueError("real natural-speech H3 capture backend was not verified")
                if not isinstance(playback_report, Mapping):
                    raise ValueError("real natural-speech H3 playback provenance was not verified")
                source_reports = playback_report.get("sources")
                if not isinstance(source_reports, Mapping) or any(
                    not isinstance(report, Mapping) or report.get("backend") != "ffmpeg_audiotoolbox"
                    for report in source_reports.values()
                ):
                    raise ValueError("real natural-speech source playback provenance was not verified")
                if not bool(io_report.get("stream_position_verified")):
                    raise ValueError("real natural-speech capture stream position was not verified")
                if captured_samples.shape[1] != 2:
                    raise ValueError("real natural-speech XVF3800 capture must retain two channels")
            clipped_count = int(np.count_nonzero(np.abs(captured_samples) >= (32767.0 / 32768.0)))
            clipped_fraction = clipped_count / captured_samples.size
            captures_with_clipping += int(clipped_count > 0)
            total_clipped_samples += clipped_count
            maximum_clipped_fraction = max(maximum_clipped_fraction, clipped_fraction)
            chirp_samples = int(CHIRP_MS * SAMPLE_RATE / 1000)
            onset_ms = estimate_playback_latency_ms(
                efference_pcm=rendered.reference_pcm[:chirp_samples],
                captured_pcm=np.mean(captured_samples, axis=1),
                sample_rate=SAMPLE_RATE,
                expected_onset_ms=float(io_report["playback_offset_from_capture_ms"]),
                causal_window_ms=500.0,
                min_peak_correlation=0.08,
            )
            if onset_ms is None:
                raise ValueError("natural-speech alignment chirp was not recovered")
            alignment = int(round(onset_ms * SAMPLE_RATE / 1000.0))
        except Exception as exc:  # noqa: BLE001
            return {
                **protocol,
                "verdict": "acquisition_failed",
                "real_hardware": real_hardware_claim,
                "live_playback": bool(real_hardware_claim and playback_started),
                "failed_trial_id": trial.trial_id,
                "completed_trial_count": len(artifacts),
                "bounded_total_capture_seconds": bounded_capture_seconds,
                "error": f"{type(exc).__name__}: {exc}",
                "artifacts": artifacts,
                "promotion_allowed": False,
            }

        captures[trial.trial_id] = captured_samples
        alignments[trial.trial_id] = alignment
        total_playback_seconds += len(rendered.reference_pcm) / SAMPLE_RATE
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
                "alignment_samples": alignment,
                "alignment_onset_ms": onset_ms,
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
                "capture_quality": {
                    "peak_absolute": float(np.max(np.abs(captured_samples))),
                    "clipped_sample_count": clipped_count,
                    "clipped_sample_fraction": clipped_fraction,
                },
            }
        )

    evaluation = evaluate_natural_speech_captures(root=root, captures=captures, alignment_samples=alignments)
    capture_passed = evaluation["verdict"] == "capture_acceptance_passed"
    no_clipping = captures_with_clipping == 0
    evaluation["criteria"] = {**evaluation["criteria"], "no_clipped_capture_samples": no_clipping}
    return {
        **evaluation,
        "verdict": (
            "real_io_acceptance_passed"
            if real_hardware_claim and capture_passed and no_clipping
            else "real_io_quality_failed_clipping"
            if real_hardware_claim and capture_passed
            else "real_io_acceptance_failed"
            if real_hardware_claim
            else "injected_acquisition_complete"
        ),
        "capture_acceptance_verdict": evaluation["verdict"],
        "real_hardware": real_hardware_claim,
        "live_playback": real_hardware_claim,
        "hardware": hardware_preregistration.get("hardware") if hardware_preregistration is not None else None,
        "completed_trial_count": len(artifacts),
        "measured_total_playback_seconds": round(total_playback_seconds, 6),
        "bounded_total_capture_seconds": round(bounded_capture_seconds, 6),
        "capture_quality": {
            "captures_with_clipping": captures_with_clipping,
            "total_clipped_samples": total_clipped_samples,
            "maximum_clipped_sample_fraction": maximum_clipped_fraction,
            "no_clipping_passed": no_clipping,
        },
        "artifacts": artifacts,
        "error": None,
        "promotion_allowed": bool(real_hardware_claim and capture_passed and no_clipping),
    }


def run_spent_natural_speech_design_gate(root: str | Path) -> dict[str, object]:
    """Run the preregistered, non-promoting synthetic acoustic design gate."""

    prepared = prepare_natural_speech_panel(root)
    self_impulse = np.zeros(FIR_SAMPLES, dtype=np.float32)
    self_impulse[0], self_impulse[61] = 0.35, 0.07
    world_impulse = np.zeros(FIR_SAMPLES, dtype=np.float32)
    world_impulse[0], world_impulse[79] = 0.25, 0.04
    delay = int(30 * SAMPLE_RATE / 1000)
    rng = np.random.default_rng(17_011)
    captures: dict[str, np.ndarray] = {}
    alignments: dict[str, int] = {}
    for trial in build_natural_speech_trials():
        rendered = render_natural_speech_tracks(trial, prepared)
        self_observed = np.convolve(rendered.self_pcm, self_impulse)
        world_observed = np.convolve(rendered.world_pcm, world_impulse)
        observed = self_observed + world_observed
        capture = np.concatenate((np.zeros(delay), observed, np.zeros(SAMPLE_RATE // 10)))
        capture += rng.normal(0.0, 0.0001, len(capture))
        captures[trial.trial_id] = np.column_stack((capture, capture)).astype(np.float32)
        alignments[trial.trial_id] = delay
    report = evaluate_natural_speech_captures(root=root, captures=captures, alignment_samples=alignments)
    report["mode"] = "spent_natural_speech_workspace_design_gate"
    report["design_preregistration"] = "docs/evidence/2026-07-11-natural-speech-workspace-design-preregistration.json"
    report["source_artifacts"] = [
        {
            "split": pair.split,
            "index": pair.index,
            "self": {
                "filename": prepared[(pair.split, pair.index)][0].filename,
                "sha256": prepared[(pair.split, pair.index)][0].sha256,
                "selected_start_seconds": prepared[(pair.split, pair.index)][0].selected_start_seconds,
            },
            "world": {
                "filename": prepared[(pair.split, pair.index)][1].filename,
                "sha256": prepared[(pair.split, pair.index)][1].sha256,
                "selected_start_seconds": prepared[(pair.split, pair.index)][1].selected_start_seconds,
            },
        }
        for pair in PAIRS
    ]
    report["verdict"] = "spent_design_gate_passed" if all(report["criteria"].values()) else "spent_design_gate_failed"
    report["promotion_allowed"] = False
    report["claim_boundary"] = "Deterministic synthetic acoustic paths over exact already-spent natural-speech PCM. This validates representation and scoring plumbing only; it is not a fresh real binding result and cannot promote H3."
    return report
