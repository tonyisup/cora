"""Final phase-invariant spectral-template candidate for natural-speech H3."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .io.motor_plan import synthesize_chirp, write_wav
from .io.wav_frames import read_pcm_wav
from .live_speak_smoke import _capture_during_playback, estimate_playback_latency_ms
from .natural_speech_workspace_template_v2 import (
    ALIGNMENT_SILENCE_MS,
    CHIRP_MS,
    SAMPLE_RATE,
    SELF_CHIRP_PEAK,
    SELF_PEAK_CAP,
    SELF_TARGET_RMS,
    SPEECH_MS,
    TAIL_SILENCE_MS,
    WORLD_CHIRP_PEAK,
    WORLD_PEAK_CAP,
    WORLD_TARGET_RMS,
    prepare_template_speech,
)
from .real_workspace_binding import BoundedEfferenceWorkspace
from .structured_workspace_binding import play_workspace_tracks_concurrently


FROZEN_THRESHOLD = 0.07895991087871712
BAND_EDGES_HZ = np.geomspace(80.0, 7600.0, 13)


@dataclass(frozen=True)
class SpectralPair:
    index: int
    self_filename: str
    self_sha256: str
    world_filename: str
    world_sha256: str


@dataclass(frozen=True)
class SpectralTrial:
    trial_id: str
    condition: str
    pair_index: int
    efference_id: str | None
    expected_world_usable: bool
    scored: bool


@dataclass(frozen=True)
class RenderedSpectralTracks:
    trial: SpectralTrial
    self_pcm: np.ndarray
    world_pcm: np.ndarray
    reference_pcm: np.ndarray
    stimulus_window: tuple[int, int]
    chirp_source: str


PAIRS = (
    SpectralPair(0, "KRA07D.wav", "c75b517f138a200a644967ff1bb7282eea43b7a31b8ec3d365ac12bfa2211994", "MEN07B.wav", "e8c346c7588280b869215f812f27d3df022df3d7eae002a2fb6fd0f7e458f80e"),
    SpectralPair(1, "MES01D.wav", "9802c4cbb8ab8cbafdce102215420e9c437a6e1f81c81922325a5fdcba3df16f", "PNG04B.wav", "21985524b4d11d2651ea8f97a0974f83fcae4eed9eccacfec7a1be0838113051"),
    SpectralPair(2, "TSI10D.wav", "309115ede9f7adc19d9fc699031fdc181de7afc1932a70ae052229f094a891a7", "VAN01B.wav", "0ed68329ae7a9a3d1c8923ba5b795e9f24a743cae14fec30248f49e14bd4e12a"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_spectral_trials() -> tuple[SpectralTrial, ...]:
    trials = []
    for pair in PAIRS:
        for condition in ("self_calibration", "self_probe", "world_only", "overlap"):
            has_self = condition != "world_only"
            trials.append(
                SpectralTrial(
                    trial_id=f"held_out_{condition}_{pair.index}",
                    condition=condition,
                    pair_index=pair.index,
                    efference_id=f"held_out_self_{pair.index}" if has_self else None,
                    expected_world_usable=condition in {"world_only", "overlap"},
                    scored=condition != "self_calibration",
                )
            )
    return tuple(trials)


def build_spectral_protocol() -> dict[str, object]:
    trials = build_spectral_trials()
    seconds = (CHIRP_MS + ALIGNMENT_SILENCE_MS + SPEECH_MS + TAIL_SILENCE_MS) / 1000.0
    return {
        "mode": "natural_speech_workspace_spectral_v3_protocol",
        "trial_count": len(trials),
        "scored_trial_count": sum(trial.scored for trial in trials),
        "seconds_per_trial": seconds,
        "maximum_total_playback_seconds": seconds * len(trials),
        "frozen_threshold": FROZEN_THRESHOLD,
        "band_edges_hz": BAND_EDGES_HZ.tolist(),
        "trials": [asdict(trial) for trial in trials],
        "strategies": ["flat", "fixed_self_window", "scheduled_identity_plus_spectral_template"],
        "capacity_sweep": [1, 2, 3, 4],
        "template_memory": {"workspace_slots": 3, "scalars_per_slot": 12, "cross_session_growth": False},
        "promotion_allowed": False,
    }


def prepare_spectral_panel(root: str | Path):
    base = Path(root)
    return {
        pair.index: (
            prepare_template_speech(base / pair.self_filename, expected_sha256=pair.self_sha256, target_rms=SELF_TARGET_RMS, peak_cap=SELF_PEAK_CAP),
            prepare_template_speech(base / pair.world_filename, expected_sha256=pair.world_sha256, target_rms=WORLD_TARGET_RMS, peak_cap=WORLD_PEAK_CAP),
        )
        for pair in PAIRS
    }


def render_spectral_tracks(trial: SpectralTrial, prepared) -> RenderedSpectralTracks:
    self_source, world_source = prepared[trial.pair_index]
    base_chirp = synthesize_chirp(250.0, 3500.0, duration_ms=CHIRP_MS, sample_rate=SAMPLE_RATE)
    empty_chirp = np.zeros_like(base_chirp)
    pre = np.zeros(ALIGNMENT_SILENCE_MS * SAMPLE_RATE // 1000, dtype=np.float32)
    tail = np.zeros(TAIL_SILENCE_MS * SAMPLE_RATE // 1000, dtype=np.float32)
    silence = np.zeros(SPEECH_MS * SAMPLE_RATE // 1000, dtype=np.float32)
    has_self = trial.condition != "world_only"
    has_world = trial.condition in {"world_only", "overlap"}
    self_chirp = SELF_CHIRP_PEAK * base_chirp if has_self else empty_chirp
    world_chirp = WORLD_CHIRP_PEAK * base_chirp if not has_self else empty_chirp
    self_pcm = np.concatenate((self_chirp, pre, self_source.pcm if has_self else silence, tail)).astype(np.float32)
    world_pcm = np.concatenate((world_chirp, pre, world_source.pcm if has_world else silence, tail)).astype(np.float32)
    start = len(base_chirp) + len(pre)
    return RenderedSpectralTracks(trial, self_pcm, world_pcm, self_pcm if has_self else world_pcm, (start, start + len(silence)), "self" if has_self else "world")


def spectral_shape(pcm: np.ndarray) -> np.ndarray:
    x = np.asarray(pcm, dtype=np.float64).reshape(-1)
    centered = x - float(np.mean(x))
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequencies = np.fft.rfftfreq(len(centered), 1.0 / SAMPLE_RATE)
    vector = np.array([
        np.log(float(np.sum(power[(frequencies >= low) & (frequencies < high)])) + 1e-12)
        for low, high in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])
    ])
    vector -= float(np.mean(vector))
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("spectral template has insufficient shape energy")
    return vector / norm


def _segment(capture: np.ndarray, alignment: int, window: tuple[int, int]) -> np.ndarray:
    x = np.asarray(capture, dtype=np.float64)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    start, end = alignment + window[0], alignment + window[1]
    if alignment < 0 or end > len(x):
        raise ValueError("capture does not contain the spectral speech window")
    return x[start:end]


def _classification(actual, predicted):
    tp = sum(a and p for a, p in zip(actual, predicted)); fp = sum((not a) and p for a, p in zip(actual, predicted)); tn = sum((not a) and (not p) for a, p in zip(actual, predicted)); fn = sum(a and (not p) for a, p in zip(actual, predicted))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "world_recall": tp / (tp + fn) if tp + fn else 0.0, "self_suppression_specificity": tn / (tn + fp) if tn + fp else 0.0, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def _strategy(trials, evidence, capacity, strategy):
    workspace = BoundedEfferenceWorkspace(capacity)
    for index in range(3): workspace.push(f"held_out_self_{index}")
    actual, predicted = [], []
    for trial in trials:
        bound = workspace.contains(trial.efference_id)
        if strategy == "flat": usable = True
        elif strategy == "fixed_self_window": usable = not bound
        elif strategy == "scheduled_identity_plus_spectral_template": usable = not bound or evidence[trial.trial_id] >= FROZEN_THRESHOLD
        else: raise ValueError(f"unknown spectral-v3 strategy: {strategy}")
        actual.append(trial.expected_world_usable); predicted.append(usable)
    return {"strategy": strategy, "capacity": capacity, "bounded_binding_slots": capacity, "world_evidence_threshold_scalars": 1, "metrics": _classification(actual, predicted)}


def score_spectral_evidence(evidence: dict[str, float]) -> dict[str, object]:
    trials = [trial for trial in build_spectral_trials() if trial.scored]
    strategies = ("flat", "fixed_self_window", "scheduled_identity_plus_spectral_template")
    reports = {str(capacity): {strategy: _strategy(trials, evidence, capacity, strategy) for strategy in strategies} for capacity in (1, 2, 3, 4)}
    selected = reports["3"]; scheduled = selected[strategies[2]]["metrics"]; flat = selected["flat"]["metrics"]; fixed = selected["fixed_self_window"]["metrics"]
    criteria = {
        "scheduled_f1_at_least_0_90": float(scheduled["f1"]) >= 0.9,
        "scheduled_improves_flat_f1_by_at_least_0_20": float(scheduled["f1"]) - float(flat["f1"]) >= 0.2 - 1e-12,
        "scheduled_world_recall_at_least_0_90": float(scheduled["world_recall"]) >= 0.9,
        "scheduled_self_suppression_specificity_at_least_0_90": float(scheduled["self_suppression_specificity"]) >= 0.9,
        "fixed_window_collision_failure_exposed": float(fixed["world_recall"]) < float(scheduled["world_recall"]),
        "capacity_one_worse_than_three": float(reports["1"][strategies[2]]["metrics"]["f1"]) < float(scheduled["f1"]),
        "capacity_three_plateaus_at_four": math.isclose(float(reports["4"][strategies[2]]["metrics"]["f1"]), float(scheduled["f1"])),
    }
    return {"verdict": "capture_acceptance_passed" if all(criteria.values()) else "capture_acceptance_failed", "world_evidence": evidence, "world_evidence_threshold": FROZEN_THRESHOLD, "threshold_source": "frozen_from_spent_template_v2_captures", "reports": reports, "criteria": criteria}


def evaluate_spectral_captures(*, root: str | Path, captures: dict[str, np.ndarray], alignment_samples: dict[str, int]) -> dict[str, object]:
    prepared = prepare_spectral_panel(root); trials = build_spectral_trials(); expected = {trial.trial_id for trial in trials}
    if set(captures) != expected or set(alignment_samples) != expected: raise ValueError("spectral-v3 maps must cover every trial")
    calibration_vectors = {}
    for trial in trials:
        if trial.condition == "self_calibration":
            tracks = render_spectral_tracks(trial, prepared); calibration_vectors[trial.pair_index] = spectral_shape(_segment(captures[trial.trial_id], alignment_samples[trial.trial_id], tracks.stimulus_window))
    evidence = {}
    for trial in trials:
        if not trial.scored: continue
        if trial.efference_id is None: evidence[trial.trial_id] = 2.0; continue
        tracks = render_spectral_tracks(trial, prepared); current = spectral_shape(_segment(captures[trial.trial_id], alignment_samples[trial.trial_id], tracks.stimulus_window)); evidence[trial.trial_id] = float(np.linalg.norm(current - calibration_vectors[trial.pair_index]))
    return {**build_spectral_protocol(), **score_spectral_evidence(evidence), "evidence_class": "provided_chirp_aligned_12_band_spectral_template_captures", "promotion_allowed": False}


def audit_spent_v2_design(*, artifacts: str | Path, result: str | Path, corpus_root: str | Path) -> dict[str, object]:
    from .natural_speech_workspace_template_v2 import build_template_trials, prepare_template_panel, render_template_tracks
    retained = json.loads(Path(result).read_text()); alignments = {item["trial_id"]: int(item["reconstructed_alignment_samples"]) for item in retained["artifacts"]}; root = Path(artifacts); prepared = prepare_template_panel(corpus_root); by = {(t.split, t.pair_index, t.condition): t for t in build_template_trials()}; self_distances, overlap_distances = [], []
    for split in ("development", "held_out"):
        for index in range(3):
            calibration = by[(split, index, "self_calibration")]; cal_tracks = render_template_tracks(calibration, prepared); cal_pcm, _ = read_pcm_wav(root / f"{calibration.trial_id}_capture.wav"); cal_vector = spectral_shape(_segment(cal_pcm, alignments[calibration.trial_id], cal_tracks.stimulus_window))
            probe = by[(split, index, "self_probe")]; probe_tracks = render_template_tracks(probe, prepared); probe_pcm, _ = read_pcm_wav(root / f"{probe.trial_id}_capture.wav"); self_distances.append(float(np.linalg.norm(cal_vector - spectral_shape(_segment(probe_pcm, alignments[probe.trial_id], probe_tracks.stimulus_window)))))
            if split == "held_out":
                overlap = by[(split, index, "overlap")]; overlap_tracks = render_template_tracks(overlap, prepared); overlap_pcm, _ = read_pcm_wav(root / f"{overlap.trial_id}_capture.wav"); overlap_distances.append(float(np.linalg.norm(cal_vector - spectral_shape(_segment(overlap_pcm, alignments[overlap.trial_id], overlap_tracks.stimulus_window)))))
    criteria = {"all_six_self_probes_below_threshold": all(value < FROZEN_THRESHOLD for value in self_distances), "all_three_overlaps_at_or_above_threshold": all(value >= FROZEN_THRESHOLD for value in overlap_distances), "clean_spent_gap": min(overlap_distances) > max(self_distances)}
    return {"mode": "spent_template_v2_spectral_v3_design_audit", "verdict": "spent_design_gate_passed" if all(criteria.values()) else "spent_design_gate_failed", "self_probe_distances": self_distances, "overlap_distances": overlap_distances, "frozen_threshold": FROZEN_THRESHOLD, "criteria": criteria, "promotion_allowed": False}


def audit_spectral_live_preregistration(preregistration: Mapping[str, object], *, current_hashes: Mapping[str, str]) -> dict[str, object]:
    blockers = []
    if preregistration.get("mode") != "natural_speech_workspace_spectral_v3_real_preregistration": blockers.append("mode mismatch")
    if preregistration.get("status") != "preregistered_before_live_execution": blockers.append("status mismatch")
    if preregistration.get("bounded_playback_capture_authorized") is not True: blockers.append("authorization missing")
    if preregistration.get("spent_design_gate_passed") is not True: blockers.append("spent design gate missing")
    frozen = preregistration.get("frozen_hashes")
    if not isinstance(frozen, Mapping): blockers.append("frozen_hashes missing")
    else:
        for name, actual in current_hashes.items():
            if frozen.get(name) != actual: blockers.append(f"frozen hash mismatch: {name}")
    hardware = preregistration.get("hardware") if isinstance(preregistration.get("hardware"), Mapping) else {}
    expected = {"input": {"avfoundation_device": ":0", "device": "reSpeaker XVF3800 4-Mic Array", "channels": 2, "sample_rate": 16000}, "self_source": {"output_device_index": 3, "output_device": "Mac mini Speakers", "placement": "centered_directly_below_xvf3800", "vertical_distance_cm": 4, "system_volume_percent": 100, "playback_gain_db": -24.0}, "world_source": {"output_device_index": 2, "output_device": "External Headphones", "angle_degrees": 180, "distance_cm": 50, "hardware_volume": "approximately_50_percent_mechanical_travel", "playback_gain_db": -3.0}}
    for route, fields in expected.items():
        actual = hardware.get(route) if isinstance(hardware.get(route), Mapping) else {}
        for field, value in fields.items():
            if actual.get(field) != value: blockers.append(f"{route} mismatch: {field}")
    bounds = preregistration.get("bounds"); expected_bounds = {"trial_count": 12, "planned_playback_seconds": 11.04, "planned_capture_seconds": 18.24}
    if not isinstance(bounds, Mapping) or any(bounds.get(k) != v for k, v in expected_bounds.items()): blockers.append("bounds mismatch")
    environment = hardware.get("environment") if isinstance(hardware.get("environment"), Mapping) else {}
    if environment.get("room") != "semi-quiet" or environment.get("known_background") != "laptop fan": blockers.append("environment mismatch")
    acceptance = preregistration.get("acceptance")
    expected_acceptance = {"selected_capacity": 3, "scheduled_f1_min": 0.9, "scheduled_minus_flat_f1_min": 0.2, "scheduled_world_recall_min": 0.9, "scheduled_self_suppression_specificity_min": 0.9, "fixed_window_world_recall_must_be_below_scheduled": True, "capacity_one_f1_must_be_below_capacity_three": True, "capacity_three_f1_must_equal_capacity_four": True, "all_12_captures_and_alignments_required": True, "no_clipped_capture_samples_required": True}
    if not isinstance(acceptance, Mapping) or any(acceptance.get(k) != v for k, v in expected_acceptance.items()): blockers.append("acceptance mismatch")
    if preregistration.get("unresolved_fields") != []: blockers.append("unresolved fields remain")
    return {"verdict": "preregistration_ready" if not blockers else "preregistration_blocked", "ready": not blockers, "blockers": blockers}


def run_spectral_live_acquisition(*, root: str | Path, out_dir: str | Path, self_playback: Callable[[str], dict[str, object]], world_playback: Callable[[str], dict[str, object]], capture_fn: Callable[..., dict[str, object]], real_hardware_claim: bool = False, bounded_playback_authorized: bool = False, hardware_preregistration: Mapping[str, object] | None = None, current_hashes: Mapping[str, str] | None = None) -> dict[str, object]:
    protocol = build_spectral_protocol(); bounded_capture = 18.24
    if real_hardware_claim:
        if not bounded_playback_authorized: raise ValueError("real spectral-v3 execution requires bounded authorization")
        if hardware_preregistration is None or current_hashes is None: raise ValueError("real spectral-v3 execution requires preregistration")
        audit = audit_spectral_live_preregistration(hardware_preregistration, current_hashes=current_hashes)
        if not audit["ready"]: raise ValueError("real spectral-v3 preregistration blocked: " + "; ".join(audit["blockers"]))
    prepared = prepare_spectral_panel(root); output = Path(out_dir); output.mkdir(parents=True, exist_ok=True); captures = {}; alignments = {}; artifacts = []; total_clipped = 0; captures_clipped = 0
    for trial in build_spectral_trials():
        rendered = render_spectral_tracks(trial, prepared); self_path = output / f"{trial.trial_id}_self.wav"; world_path = output / f"{trial.trial_id}_world.wav"; capture_path = output / f"{trial.trial_id}_capture.wav"; write_wav(str(self_path), rendered.self_pcm, SAMPLE_RATE); write_wav(str(world_path), rendered.world_pcm, SAMPLE_RATE); started = False
        def play(_path):
            nonlocal started; started = True
            return play_workspace_tracks_concurrently(self_path=str(self_path) if trial.condition != "world_only" else None, world_path=str(world_path) if trial.condition in {"world_only", "overlap"} else None, self_playback=self_playback, world_playback=world_playback)
        try:
            io = _capture_during_playback(capture_path=str(capture_path), playback_path=str(self_path if rendered.chirp_source == "self" else world_path), sample_rate=SAMPLE_RATE, capture_duration_ms=1520, capture_fn=capture_fn, playback_fn=play, capture_preroll_ms=100); captured, rate = read_pcm_wav(capture_path)
            if rate != SAMPLE_RATE or (real_hardware_claim and captured.shape[1] != 2): raise ValueError("spectral-v3 capture geometry mismatch")
            if real_hardware_claim:
                if io["capture"].get("backend") != "ffmpeg_avfoundation" or not io.get("stream_position_verified"): raise ValueError("spectral-v3 real capture provenance failed")
            clipped = int(np.count_nonzero(np.abs(captured) >= 32767.0 / 32768.0)); total_clipped += clipped; captures_clipped += int(clipped > 0); chirp_n = CHIRP_MS * SAMPLE_RATE // 1000; onset = estimate_playback_latency_ms(efference_pcm=rendered.reference_pcm[:chirp_n], captured_pcm=np.mean(captured, axis=1), sample_rate=SAMPLE_RATE, expected_onset_ms=float(io["playback_offset_from_capture_ms"]), causal_window_ms=500.0, min_peak_correlation=0.08)
            if onset is None: raise ValueError("spectral-v3 chirp not recovered")
            alignment = int(round(onset * SAMPLE_RATE / 1000.0))
        except Exception as exc:
            return {**protocol, "verdict": "acquisition_failed", "failed_trial_id": trial.trial_id, "completed_trial_count": len(artifacts), "live_playback": bool(real_hardware_claim and started), "error": f"{type(exc).__name__}: {exc}", "artifacts": artifacts, "promotion_allowed": False}
        captures[trial.trial_id] = captured; alignments[trial.trial_id] = alignment; artifacts.append({"trial_id": trial.trial_id, "condition": trial.condition, "scored": trial.scored, "self_wav": str(self_path.resolve()), "self_sha256": _sha256(self_path), "world_wav": str(world_path.resolve()), "world_sha256": _sha256(world_path), "captured_wav": str(capture_path.resolve()), "captured_sha256": _sha256(capture_path), "alignment_samples": alignment, "alignment_onset_ms": onset, "io_timing": {k: io[k] for k in ("playback_offset_samples", "playback_offset_from_capture_ms", "overlap_verified", "stream_position_verified", "timestamp_source")}, "playback": io["playback"], "capture": io["capture"], "capture_quality": {"peak_absolute": float(np.max(np.abs(captured))), "clipped_sample_count": clipped, "clipped_sample_fraction": clipped / captured.size}})
    evaluation = evaluate_spectral_captures(root=root, captures=captures, alignment_samples=alignments); capture_passed = evaluation["verdict"] == "capture_acceptance_passed"; no_clipping = total_clipped == 0; evaluation["criteria"] = {**evaluation["criteria"], "no_clipped_capture_samples": no_clipping}
    return {**evaluation, "verdict": "real_io_acceptance_passed" if real_hardware_claim and capture_passed and no_clipping else "real_io_quality_failed_clipping" if real_hardware_claim and capture_passed else "real_io_acceptance_failed" if real_hardware_claim else "injected_acquisition_complete", "real_hardware": real_hardware_claim, "live_playback": real_hardware_claim, "completed_trial_count": 12, "measured_total_playback_seconds": 11.04, "bounded_total_capture_seconds": bounded_capture, "capture_quality": {"captures_with_clipping": captures_clipped, "total_clipped_samples": total_clipped, "no_clipping_passed": no_clipping}, "artifacts": artifacts, "error": None, "promotion_allowed": bool(real_hardware_claim and capture_passed and no_clipping)}
