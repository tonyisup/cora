"""Development-only continuous inverse synthesis for VISION H4."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .io.motor_plan import (
    VOWEL_FORMANTS_HZ,
    _formant_resonator,
    _radiation_filter,
    _source_pulse_train,
    analyze_vowel,
    evaluate_vowel_round_trip,
)
from .io.wav_frames import _pitch_hz_and_confidence, read_pcm_wav


SAMPLE_RATE = 16_000
VOWELS = ("a", "e", "i", "o", "u")
BAND_EDGES_HZ = np.geomspace(200.0, 3500.0, 21)
INITIAL_STEPS_HZ = (120.0, 240.0, 300.0)
ADAPTATION_ROUNDS = 3
VOCALSET_SINGERS = ("female4", "female5", "male6", "male7", "male8")
CMU_SPEAKERS = ("bdl", "slt", "clb", "jmk", "ksp")
VOCALSET_PATTERN = re.compile(
    r"^FULL/(?P<speaker>female4|female5|male6|male7|male8)/"
    r"long_tones/straight/[fm]\d+_long_straight_(?P<vowel>[aeiou])\.wav$"
)


@dataclass(frozen=True)
class MotorCoordinates:
    f0_hz: float
    f1_hz: float
    f2_hz: float
    f3_hz: float


@dataclass(frozen=True)
class InverseSynthesisResult:
    initial_plan: MotorCoordinates
    final_plan: MotorCoordinates
    initial_identity: str
    final_identity: str
    reperceived_identity: str
    initial_loss: float
    final_loss: float
    update_count: int
    synthesis_count: int
    pitch_confidence: float
    pitch_fallback_used: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resample_linear(pcm: np.ndarray, source_rate: int) -> np.ndarray:
    mono = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if source_rate == SAMPLE_RATE:
        return mono
    count = max(1, int(round(len(mono) * SAMPLE_RATE / source_rate)))
    return np.interp(
        np.arange(count, dtype=np.float64) * source_rate / SAMPLE_RATE,
        np.arange(len(mono), dtype=np.float64),
        mono,
    ).astype(np.float32)


def _valid_plan(plan: MotorCoordinates) -> bool:
    return (
        70.0 <= plan.f0_hz <= 500.0
        and 200.0 <= plan.f1_hz <= 1000.0
        and 600.0 <= plan.f2_hz <= 2800.0
        and 1800.0 <= plan.f3_hz <= 3500.0
        and plan.f2_hz - plan.f1_hz >= 200.0
        and plan.f3_hz - plan.f2_hz >= 300.0
    )


def _source_context(
    *, duration_samples: int, f0_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    source = _source_pulse_train(duration_samples, SAMPLE_RATE, f0_hz)
    envelope = np.ones(duration_samples, dtype=np.float64)
    ramp = min(int(0.01 * SAMPLE_RATE), duration_samples // 2)
    if ramp:
        envelope[:ramp] = np.linspace(0.0, 1.0, ramp)
        envelope[-ramp:] = np.linspace(1.0, 0.0, ramp)
    return source, envelope


def _synthesize_with_context(
    plan: MotorCoordinates,
    *,
    source: np.ndarray,
    envelope: np.ndarray,
) -> np.ndarray:
    if not _valid_plan(plan):
        raise ValueError("motor coordinates violate frozen bounds")
    signal = _formant_resonator(source, SAMPLE_RATE, plan.f1_hz, 80.0) * 5.0
    signal += _formant_resonator(source, SAMPLE_RATE, plan.f2_hz, 100.0) * 3.0
    signal += _formant_resonator(source, SAMPLE_RATE, plan.f3_hz, 130.0) * 2.0
    signal = _radiation_filter(signal, SAMPLE_RATE) * envelope
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak).astype(np.float32)


def synthesize_motor_coordinates(
    plan: MotorCoordinates, *, duration_ms: int
) -> np.ndarray:
    samples = int(round(duration_ms * SAMPLE_RATE / 1000.0))
    if samples < SAMPLE_RATE // 10:
        raise ValueError("duration too short for continuous vowel synthesis")
    source, envelope = _source_context(duration_samples=samples, f0_hz=plan.f0_hz)
    return _synthesize_with_context(plan, source=source, envelope=envelope)


def spectral_shape(pcm: np.ndarray) -> np.ndarray:
    x = np.asarray(pcm, dtype=np.float64).reshape(-1)
    centered = x - float(np.mean(x))
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequencies = np.fft.rfftfreq(len(centered), 1.0 / SAMPLE_RATE)
    vector = np.array(
        [
            np.log(
                float(
                    np.sum(power[(frequencies >= low) & (frequencies < high)])
                )
                + 1e-12
            )
            for low, high in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])
        ]
    )
    vector -= float(np.mean(vector))
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("spectral shape has insufficient energy")
    return vector / norm


def _canonical_plan(vowel: str, f0_hz: float) -> MotorCoordinates:
    f1, f2, f3 = VOWEL_FORMANTS_HZ[vowel]
    return MotorCoordinates(f0_hz=f0_hz, f1_hz=f1, f2_hz=f2, f3_hz=f3)


def nearest_motor_identity(plan: MotorCoordinates) -> str:
    return min(
        VOWELS,
        key=lambda vowel: (
            (plan.f1_hz - VOWEL_FORMANTS_HZ[vowel][0]) ** 2
            + (plan.f2_hz - VOWEL_FORMANTS_HZ[vowel][1]) ** 2
            + 0.1 * (plan.f3_hz - VOWEL_FORMANTS_HZ[vowel][2]) ** 2
        ),
    )


def infer_motor_coordinates(observed_pcm: np.ndarray) -> InverseSynthesisResult:
    observed = np.asarray(observed_pcm, dtype=np.float32).reshape(-1)
    if len(observed) < SAMPLE_RATE // 10:
        raise ValueError("inverse-synthesis input must be at least 100 ms")
    pitch, confidence = _pitch_hz_and_confidence(
        observed, SAMPLE_RATE, min_hz=70.0, max_hz=500.0
    )
    fallback = pitch is None
    f0 = 110.0 if pitch is None else float(np.clip(pitch, 70.0, 500.0))
    target = spectral_shape(observed)
    source, envelope = _source_context(duration_samples=len(observed), f0_hz=f0)

    synthesis_count = 0

    def evaluate(plan: MotorCoordinates) -> tuple[float, np.ndarray]:
        nonlocal synthesis_count
        waveform = _synthesize_with_context(plan, source=source, envelope=envelope)
        synthesis_count += 1
        return float(np.linalg.norm(target - spectral_shape(waveform))), waveform

    candidates = []
    for vowel in VOWELS:
        plan = _canonical_plan(vowel, f0)
        loss, waveform = evaluate(plan)
        candidates.append((loss, VOWELS.index(vowel), plan, waveform))
    initial_loss, _order, initial_plan, best_waveform = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    current_plan = initial_plan
    current_loss = initial_loss
    update_count = 0
    steps = list(INITIAL_STEPS_HZ)
    coordinate_names = ("f1_hz", "f2_hz", "f3_hz")
    for _round in range(ADAPTATION_ROUNDS):
        for name, step in zip(coordinate_names, steps):
            for direction in (-1.0, 1.0):
                candidate = replace(
                    current_plan,
                    **{name: getattr(current_plan, name) + direction * step},
                )
                if not _valid_plan(candidate):
                    continue
                loss, waveform = evaluate(candidate)
                if loss < current_loss:
                    current_plan = candidate
                    current_loss = loss
                    best_waveform = waveform
                    update_count += 1
        steps = [step / 2.0 for step in steps]
    reperceived = analyze_vowel(best_waveform, sample_rate=SAMPLE_RATE).nearest_vowel
    return InverseSynthesisResult(
        initial_plan=initial_plan,
        final_plan=current_plan,
        initial_identity=nearest_motor_identity(initial_plan),
        final_identity=nearest_motor_identity(current_plan),
        reperceived_identity=reperceived,
        initial_loss=initial_loss,
        final_loss=current_loss,
        update_count=update_count,
        synthesis_count=synthesis_count,
        pitch_confidence=confidence,
        pitch_fallback_used=fallback,
    )


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    confusion = {vowel: {guess: 0 for guess in VOWELS} for vowel in VOWELS}
    vowel_counts = {vowel: [0, 0] for vowel in VOWELS}
    speaker_counts: dict[str, list[int]] = {}
    errors = [[], [], []]
    for row in rows:
        expected = str(row["expected_vowel"])
        predicted = str(row["predicted_vowel"])
        speaker = str(row["speaker"])
        correct = expected == predicted
        confusion[expected][predicted] += 1
        vowel_counts[expected][0] += int(correct)
        vowel_counts[expected][1] += 1
        speaker_counts.setdefault(speaker, [0, 0])
        speaker_counts[speaker][0] += int(correct)
        speaker_counts[speaker][1] += 1
        for index, (actual, target) in enumerate(
            zip(row["final_formants_hz"], VOWEL_FORMANTS_HZ[expected])
        ):
            errors[index].append(abs(float(actual) - target))
    relative = [float(row["relative_loss_improvement"]) for row in rows]
    correct_count = sum(
        str(row["expected_vowel"]) == str(row["predicted_vowel"]) for row in rows
    )
    return {
        "trial_count": len(rows),
        "correct_count": correct_count,
        "accuracy": correct_count / len(rows),
        "confusion_matrix": confusion,
        "vowel_recall": {
            vowel: hits / total for vowel, (hits, total) in vowel_counts.items()
        },
        "speaker_accuracy": {
            speaker: hits / total
            for speaker, (hits, total) in speaker_counts.items()
        },
        "formant_mae_hz": {
            f"f{index + 1}": float(np.mean(values))
            for index, values in enumerate(errors)
        },
        "strict_loss_improvement_fraction": sum(value > 0.0 for value in relative)
        / len(relative),
        "median_relative_loss_improvement": float(np.median(relative)),
        "mean_relative_loss_improvement": float(np.mean(relative)),
        "motor_plan_to_reperception_agreement": sum(
            row["predicted_vowel"] == row["reperceived_vowel"] for row in rows
        )
        / len(rows),
        "total_motor_updates": sum(int(row["update_count"]) for row in rows),
        "maximum_candidate_syntheses": max(
            int(row["synthesis_count"]) for row in rows
        ),
        "pitch_fallback_count": sum(bool(row["pitch_fallback_used"]) for row in rows),
        "rows": rows,
    }


def _row(
    *, speaker: str, expected_vowel: str, source_id: str, pcm: np.ndarray
) -> dict[str, object]:
    result = infer_motor_coordinates(pcm)
    initial = result.initial_plan
    final = result.final_plan
    relative = (result.initial_loss - result.final_loss) / max(
        result.initial_loss, 1e-12
    )
    return {
        "speaker": speaker,
        "source_id": source_id,
        "expected_vowel": expected_vowel,
        "initial_vowel": result.initial_identity,
        "predicted_vowel": result.final_identity,
        "reperceived_vowel": result.reperceived_identity,
        "observed_f0_hz": final.f0_hz,
        "pitch_confidence": result.pitch_confidence,
        "pitch_fallback_used": result.pitch_fallback_used,
        "initial_formants_hz": [initial.f1_hz, initial.f2_hz, initial.f3_hz],
        "final_formants_hz": [final.f1_hz, final.f2_hz, final.f3_hz],
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "relative_loss_improvement": relative,
        "update_count": result.update_count,
        "synthesis_count": result.synthesis_count,
        "plan_within_bounds": _valid_plan(final),
    }


def evaluate_vocalset_development(
    *, root: str | Path, manifest_path: str | Path
) -> dict[str, object]:
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest.get("members")
    if not isinstance(records, list) or len(records) != 25:
        raise ValueError("VocalSet development manifest must contain 25 members")
    base = Path(root)
    rows = []
    for record in records:
        member = str(record["member"])
        match = VOCALSET_PATTERN.fullmatch(member)
        if match is None:
            raise ValueError(f"unexpected VocalSet development member: {member}")
        path = base.joinpath(*Path(member).parts)
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"VocalSet hash mismatch: {member}")
        samples, sample_rate = read_pcm_wav(path)
        mono = np.mean(samples, axis=1)
        width = sample_rate
        start = (len(mono) - width) // 2
        pcm = _resample_linear(mono[start : start + width], sample_rate)
        rows.append(
            _row(
                speaker=match.group("speaker"),
                expected_vowel=match.group("vowel"),
                source_id=member,
                pcm=pcm,
            )
        )
    return _metrics(rows)


def evaluate_cmu_development(
    *, root: str | Path, manifest_path: str | Path
) -> dict[str, object]:
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("confirmation_audio_or_labels_accessed") is not False:
        raise ValueError("CMU development manifest crosses confirmation boundary")
    tokens = manifest.get("tokens")
    wav_records = manifest.get("wav_files")
    if not isinstance(tokens, list) or len(tokens) != 500:
        raise ValueError("CMU development manifest must contain 500 tokens")
    if not isinstance(wav_records, list) or len(wav_records) != 383:
        raise ValueError("CMU development manifest must contain 383 WAV records")
    base = Path(root)
    hashes = {
        (str(record["speaker"]), str(record["name"])): str(record["sha256"])
        for record in wav_records
    }
    cache = {}
    for (speaker, name), digest in hashes.items():
        path = base / speaker / "wav" / name
        if _sha256(path) != digest:
            raise ValueError(f"CMU WAV hash mismatch: {speaker}/{name}")
        samples, rate = read_pcm_wav(path)
        cache[(speaker, name)] = (np.mean(samples, axis=1), rate)
    rows = []
    for token in tokens:
        speaker = str(token["speaker"])
        name = str(token["wav_name"])
        samples, rate = cache[(speaker, name)]
        width = int(round(0.120 * rate))
        center = int(
            round(
                (float(token["start_seconds"]) + float(token["end_seconds"]))
                / 2.0
                * rate
            )
        )
        start = center - width // 2
        window = samples[start : start + width]
        if len(window) != width:
            raise ValueError(f"CMU token window exceeds WAV: {speaker}/{name}")
        rows.append(
            _row(
                speaker=speaker,
                expected_vowel=str(token["vowel"]),
                source_id=f"{speaker}/{name}#{token['segment_index']}",
                pcm=_resample_linear(window, rate),
            )
        )
    return _metrics(rows)


def evaluate_synthetic_inverse_synthesis() -> dict[str, object]:
    rows = []
    for f0 in (90.0, 110.0, 140.0, 180.0):
        for vowel in VOWELS:
            observed = synthesize_motor_coordinates(
                _canonical_plan(vowel, f0), duration_ms=300
            )
            rows.append(
                _row(
                    speaker=f"synthetic_f0_{f0:g}",
                    expected_vowel=vowel,
                    source_id=f"{vowel}_{f0:g}",
                    pcm=observed,
                )
            )
    return {
        **_metrics(rows),
        "existing_clean_round_trip": evaluate_vowel_round_trip(),
    }


def evaluate_continuous_inverse_synthesis(
    *,
    vocalset_root: str | Path,
    vocalset_manifest: str | Path,
    cmu_root: str | Path,
    cmu_manifest: str | Path,
) -> dict[str, object]:
    synthetic = evaluate_synthetic_inverse_synthesis()
    vocalset = evaluate_vocalset_development(
        root=vocalset_root, manifest_path=vocalset_manifest
    )
    cmu = evaluate_cmu_development(root=cmu_root, manifest_path=cmu_manifest)
    domains = {"vocalset": vocalset, "cmu_arctic": cmu}
    criteria = {
        "clean_synthetic_accuracy_at_least_0_90": synthetic["accuracy"] >= 0.90,
        "vocalset_accuracy_at_least_0_80": vocalset["accuracy"] >= 0.80,
        "cmu_arctic_accuracy_at_least_0_80": cmu["accuracy"] >= 0.80,
        "each_domain_each_vowel_recall_at_least_0_60": all(
            min(report["vowel_recall"].values()) >= 0.60
            for report in domains.values()
        ),
        "each_domain_each_speaker_accuracy_at_least_0_60": all(
            min(report["speaker_accuracy"].values()) >= 0.60
            for report in domains.values()
        ),
        "each_domain_strict_loss_improvement_fraction_at_least_0_80": all(
            report["strict_loss_improvement_fraction"] >= 0.80
            for report in domains.values()
        ),
        "each_domain_median_relative_loss_improvement_at_least_0_10": all(
            report["median_relative_loss_improvement"] >= 0.10
            for report in domains.values()
        ),
        "each_domain_motor_plan_to_reperception_agreement_at_least_0_90": all(
            report["motor_plan_to_reperception_agreement"] >= 0.90
            for report in domains.values()
        ),
        "all_motor_plans_within_bounds": all(
            all(bool(row["plan_within_bounds"]) for row in report["rows"])
            for report in (synthetic, *domains.values())
        ),
        "maximum_candidate_syntheses_at_most_23": all(
            report["maximum_candidate_syntheses"] <= 23
            for report in (synthetic, *domains.values())
        ),
        "all_required_reports_present": all(
            len(report["confusion_matrix"]) == 5
            and len(report["formant_mae_hz"]) == 3
            for report in domains.values()
        ),
        "all_audio_hashes_verified": True,
        "confirmation_audio_or_labels_accessed": False,
    }
    ready = all(criteria.values())
    return {
        "mode": "h4_continuous_inverse_synthesis_development",
        "evidence_scope": "already_spent_development_only",
        "mechanism": {
            "analysis_sample_rate_hz": SAMPLE_RATE,
            "spectral_band_edges_hz": BAND_EDGES_HZ.tolist(),
            "initial_steps_hz": list(INITIAL_STEPS_HZ),
            "adaptation_rounds": ADAPTATION_ROUNDS,
            "maximum_candidate_syntheses": 23,
            "global_backpropagation": False,
            "labels_consumed_by_inference": False,
        },
        "baselines": {
            "vocalset_fixed_lpc_accuracy": 0.36,
            "vocalset_best_prior_development_accuracy": 0.56,
            "cmu_fixed_lpc_accuracy": 0.572,
            "canonical_synthesis_initialization_reported_per_row": True,
        },
        "synthetic": synthetic,
        "vocalset": vocalset,
        "cmu_arctic": cmu,
        "criteria": criteria,
        "all_development_gates_passed": ready,
        "confirmation_ready": ready,
        "confirmation_audio_or_labels_accessed": False,
        "promotion_allowed": False,
        "claim_boundary": (
            "A pass establishes confirmation readiness for bounded per-example "
            "continuous inverse synthesis on spent sung and read-speech development "
            "voices. It does not promote H4, establish human articulation, or "
            "authorize protected confirmation access."
        ),
    }
