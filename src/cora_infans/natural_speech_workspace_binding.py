"""Natural-speech extension of the bounded H3 self/world binding task.

The runtime representation is label-blind.  A known alignment chirp estimates
the self loudspeaker path, a bounded FIR predicts the commanded self waveform,
and unexplained residual energy supplies bottom-up world evidence.  Corpus
condition labels and trial conditions enter scoring only.
"""

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
ALIGNMENT_SILENCE_MS = 180
SPEECH_MS = 800
TAIL_SILENCE_MS = 200
SELF_PEAK = 0.12
WORLD_PEAK = 0.80
FIR_SAMPLES = 128
DECONVOLUTION_RIDGE_FRACTION = 0.001


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
