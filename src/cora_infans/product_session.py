"""Offline validation and replay scoring for vocal-imitation product sessions.

This module validates JSON-safe report metadata and scores pre-existing WAV
artifacts. It does not capture audio, play audio, discover devices, select
prototypes, train associations, or modify retained V4 evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .io.wav_frames import read_pcm_wav


PRODUCT_SESSION_REPORT_VERSION = 1
EVIDENCE_CLASSES = frozenset(
    {
        "retained_v4_evidence",
        "synthetic_fixture",
        "real_authorized_session",
    }
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "source_human_wav",
        "target_captured_wav",
        "agent_emitted_wav",
        "response_captured_wav",
        "self_prediction_wav",
        "report_json",
    }
)
REQUIRED_SCORE_FAMILIES = frozenset({"imitation_fidelity", "self_model_fidelity"})
SCORING_VERSION = "offline-replay-v2"
_SCORING_SAMPLE_RATE_HZ = 16000
_ENVELOPE_FRAME_SAMPLES = 320
_ENVELOPE_HOP_SAMPLES = 160


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_artifacts(artifacts: object) -> None:
    records = _require_mapping(artifacts, name="artifacts")
    missing = REQUIRED_ARTIFACTS.difference(records)
    if missing:
        raise ValueError(f"report is missing required artifacts: {sorted(missing)}")
    for name in REQUIRED_ARTIFACTS:
        record = _require_mapping(records[name], name=f"artifact {name}")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError(f"artifact {name} requires a non-empty path")
        if name == "report_json":
            # The report cannot contain its own final digest without circular
            # content. A future sibling manifest attests the completed report.
            continue
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"artifact {name} requires a lowercase SHA-256 digest")


def _validate_scores(scores: object) -> None:
    families = _require_mapping(scores, name="scores")
    missing = REQUIRED_SCORE_FAMILIES.difference(families)
    if missing:
        raise ValueError(f"report is missing required score families: {sorted(missing)}")
    for family in REQUIRED_SCORE_FAMILIES:
        metrics = _require_mapping(families[family], name=f"score family {family}")
        if not metrics:
            raise ValueError(f"score family {family} must contain at least one metric")
        for metric, value in metrics.items():
            if not isinstance(metric, str) or not metric:
                raise ValueError(f"score family {family} has an invalid metric name")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"score metric {family}.{metric} must be finite")


def _validate_real_authorization(report: Mapping[str, object]) -> None:
    authorization = _require_mapping(report.get("authorization"), name="authorization")
    if authorization.get("recording_authorized") is not True:
        raise ValueError("real_authorized_session requires recording authorization")
    protocol_version = authorization.get("protocol_version")
    if not isinstance(protocol_version, str) or not protocol_version:
        raise ValueError("real_authorized_session requires a protocol version")
    operating_point = _require_mapping(report.get("operating_point"), name="operating_point")
    input_device = operating_point.get("input_device")
    sample_rate = operating_point.get("sample_rate_hz")
    if not isinstance(input_device, str) or not input_device:
        raise ValueError("real_authorized_session requires an input device")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("real_authorized_session requires a positive sample rate")


def validate_product_session_report(report: Mapping[str, object]) -> dict[str, object]:
    """Validate and return a copy of an offline product-session report."""

    payload = _require_mapping(report, name="product-session report")
    evidence_class = payload.get("evidence_class")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"unknown product-session evidence class: {evidence_class!r}")
    if payload.get("v4_evidence_mutated") is not False:
        raise ValueError("product-session reports must declare V4 evidence unmodified")
    if payload.get("association_frozen_before_evaluation") is not True:
        raise ValueError("association must be frozen before evaluation")
    _validate_artifacts(payload.get("artifacts"))
    _validate_scores(payload.get("scores"))
    if evidence_class == "real_authorized_session":
        _validate_real_authorization(payload)
    return dict(payload)


def build_product_session_report(
    *,
    evidence_class: str,
    artifacts: Mapping[str, object],
    scores: Mapping[str, object],
    association_frozen_before_evaluation: bool,
    authorization: Mapping[str, object] | None = None,
    operating_point: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a deterministic metadata report and enforce its contract."""

    report: dict[str, object] = {
        "report_version": PRODUCT_SESSION_REPORT_VERSION,
        "evidence_class": evidence_class,
        "v4_evidence_mutated": False,
        "association_frozen_before_evaluation": association_frozen_before_evaluation,
        "artifacts": dict(artifacts),
        "scores": dict(scores),
    }
    if authorization is not None:
        report["authorization"] = dict(authorization)
    if operating_point is not None:
        report["operating_point"] = dict(operating_point)
    return validate_product_session_report(report)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resample(samples: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate <= 0:
        raise ValueError("WAV sample rate must be positive")
    mono = np.asarray(samples, dtype=np.float64).reshape(-1)
    if mono.size < 2:
        raise ValueError("WAV must contain at least two samples")
    if source_rate == _SCORING_SAMPLE_RATE_HZ:
        return mono
    output_count = max(2, int(round(mono.size * _SCORING_SAMPLE_RATE_HZ / source_rate)))
    source_positions = np.arange(mono.size, dtype=np.float64) / source_rate
    output_positions = np.arange(output_count, dtype=np.float64) / _SCORING_SAMPLE_RATE_HZ
    return np.interp(output_positions, source_positions, mono)


def _read_mono(path: Path) -> np.ndarray:
    samples, sample_rate = read_pcm_wav(path)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError(f"scoring WAV has invalid sample geometry: {path.name}")
    return _resample(np.mean(samples, axis=1), int(sample_rate))


def _frame_rms(
    samples: np.ndarray,
    *,
    frame_samples: int = _ENVELOPE_FRAME_SAMPLES,
    hop_samples: int = _ENVELOPE_HOP_SAMPLES,
) -> np.ndarray:
    if samples.size < frame_samples:
        raise ValueError("scoring WAV is shorter than one 20 ms analysis frame")
    values = [
        float(np.sqrt(np.mean(np.square(samples[offset : offset + frame_samples]))))
        for offset in range(0, samples.size - frame_samples + 1, hop_samples)
    ]
    return np.asarray(values, dtype=np.float64)


def _resample_vector(values: np.ndarray, count: int) -> np.ndarray:
    if values.size == count:
        return values.astype(np.float64, copy=False)
    if values.size == 1:
        return np.full(count, float(values[0]), dtype=np.float64)
    return np.interp(
        np.linspace(0.0, 1.0, count),
        np.linspace(0.0, 1.0, values.size),
        values,
    )


def _active_bounds(envelope: np.ndarray) -> tuple[int, int]:
    threshold = max(float(np.max(envelope)) * 0.10, 1e-6)
    active = np.flatnonzero(envelope >= threshold)
    if active.size == 0:
        raise ValueError("scoring WAV contains no active voiced interval")
    return int(active[0]), int(active[-1] + 1)


def _active_samples(
    samples: np.ndarray,
    envelope_bounds: tuple[int, int],
) -> np.ndarray:
    """Return the detected vocal event independent of capture-window offset."""

    start_frame, end_frame = envelope_bounds
    start_sample = start_frame * _ENVELOPE_HOP_SAMPLES
    end_sample = min(
        samples.size,
        (end_frame - 1) * _ENVELOPE_HOP_SAMPLES + _ENVELOPE_FRAME_SAMPLES,
    )
    active = samples[start_sample:end_sample]
    if active.size < _ENVELOPE_FRAME_SAMPLES:
        raise ValueError("scoring WAV active interval is shorter than one analysis frame")
    return active


def _pitch_contour(samples: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    frame_samples = 512
    hop_samples = 256
    threshold = max(float(np.max(envelope)) * 0.10, 1e-6)
    pitches: list[float] = []
    for offset in range(0, samples.size - frame_samples + 1, hop_samples):
        frame = samples[offset : offset + frame_samples]
        if float(np.sqrt(np.mean(np.square(frame)))) < threshold:
            continue
        centered = frame - float(np.mean(frame))
        energy = float(np.dot(centered, centered))
        if energy <= 1e-10:
            continue
        correlation = np.correlate(centered, centered, mode="full")[frame_samples - 1 :]
        minimum_lag = int(_SCORING_SAMPLE_RATE_HZ / 350.0)
        maximum_lag = min(frame_samples - 1, int(_SCORING_SAMPLE_RATE_HZ / 70.0))
        lag = minimum_lag + int(np.argmax(correlation[minimum_lag : maximum_lag + 1]))
        if float(correlation[lag] / correlation[0]) >= 0.30:
            pitches.append(float(_SCORING_SAMPLE_RATE_HZ / lag))
    if not pitches:
        raise ValueError("scoring WAV has no voiced pitch contour")
    return np.asarray(pitches, dtype=np.float64)


def _cepstral_vector(samples: np.ndarray) -> np.ndarray:
    frame_samples = 512
    hop_samples = 256
    window = np.hanning(frame_samples)
    vectors = []
    for offset in range(0, samples.size - frame_samples + 1, hop_samples):
        spectrum = np.log(np.square(np.abs(np.fft.rfft(samples[offset : offset + frame_samples] * window))) + 1e-12)
        bins = np.arange(spectrum.size, dtype=np.float64) + 0.5
        coefficients = np.arange(1, 14, dtype=np.float64)[:, None]
        vectors.append((np.cos(np.pi * coefficients * bins / spectrum.size) @ spectrum) / spectrum.size)
    if not vectors:
        raise ValueError("scoring WAV has no complete cepstral frame")
    return np.mean(np.stack(vectors), axis=0)


def score_waveform_pair(reference: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Return capture-offset-independent distances for two microphone waveforms."""

    reference_envelope = _frame_rms(reference)
    observed_envelope = _frame_rms(observed)
    reference_bounds = _active_bounds(reference_envelope)
    observed_bounds = _active_bounds(observed_envelope)
    active_reference = _active_samples(reference, reference_bounds)
    active_observed = _active_samples(observed, observed_bounds)
    active_reference_envelope = reference_envelope[
        reference_bounds[0] : reference_bounds[1]
    ]
    active_observed_envelope = observed_envelope[
        observed_bounds[0] : observed_bounds[1]
    ]
    common_count = max(
        active_reference_envelope.size,
        active_observed_envelope.size,
    )
    aligned_reference_envelope = _resample_vector(
        active_reference_envelope,
        common_count,
    )
    aligned_observed_envelope = _resample_vector(
        active_observed_envelope,
        common_count,
    )
    reference_pitch = _pitch_contour(
        active_reference,
        active_reference_envelope,
    )
    observed_pitch = _pitch_contour(
        active_observed,
        active_observed_envelope,
    )
    common_pitch_count = max(reference_pitch.size, observed_pitch.size)
    return {
        "active_duration_difference_ms": float(
            abs(active_reference.size - active_observed.size)
            * 1000.0
            / _SCORING_SAMPLE_RATE_HZ
        ),
        "rms_envelope_distance": float(
            np.sqrt(np.mean(np.square(aligned_reference_envelope - aligned_observed_envelope)))
        ),
        "pitch_contour_distance_hz": float(
            np.mean(
                np.abs(
                    _resample_vector(reference_pitch, common_pitch_count)
                    - _resample_vector(observed_pitch, common_pitch_count)
                )
            )
        ),
        "cepstral_distance": float(
            np.linalg.norm(
                _cepstral_vector(active_reference)
                - _cepstral_vector(active_observed)
            )
        ),
    }


def score_product_session_directory(
    session_dir: str | Path,
    *,
    evidence_class: str,
    association_frozen_before_evaluation: bool,
) -> dict[str, object]:
    """Score an existing session directory without any audio-device side effects."""

    if association_frozen_before_evaluation is not True:
        raise ValueError("association must be frozen before offline replay scoring")
    root = Path(session_dir)
    filenames = {
        "source_human_wav": "source_human.wav",
        "target_captured_wav": "target_captured.wav",
        "agent_emitted_wav": "agent_emitted.wav",
        "response_captured_wav": "response_captured.wav",
        "self_prediction_wav": "self_prediction.wav",
    }
    paths = {name: root / filename for name, filename in filenames.items()}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"product session is missing required WAV artifacts: {sorted(missing)}")
    audio = {name: _read_mono(path) for name, path in paths.items()}
    artifacts: dict[str, object] = {
        name: {"path": path.name, "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    artifacts["report_json"] = {"path": "report.json"}
    report = build_product_session_report(
        evidence_class=evidence_class,
        artifacts=artifacts,
        scores={
            "imitation_fidelity": score_waveform_pair(
                audio["target_captured_wav"], audio["response_captured_wav"]
            ),
            "self_model_fidelity": score_waveform_pair(
                audio["self_prediction_wav"], audio["response_captured_wav"]
            ),
        },
        association_frozen_before_evaluation=association_frozen_before_evaluation,
    )
    report["scoring_version"] = SCORING_VERSION
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "report_path": report_path.name,
        "report_sha256": _sha256(report_path),
        "artifact_sha256": {
            name: str(record["sha256"])
            for name, record in artifacts.items()
            if name != "report_json"
        },
    }
    (root / "session-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return report
