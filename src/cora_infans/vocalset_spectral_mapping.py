"""Learn a bounded spectral-envelope to formant-coordinate map on development voices."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .io.motor_plan import VOWEL_FORMANTS_HZ
from .io.wav_frames import read_pcm_wav


VOWELS = ("a", "e", "i", "o", "u")
PATTERN = re.compile(r".*/(?P<singer>female\d+|male\d+)/.*/.*_(?P<vowel>[aeiou])\.wav$")
BAND_COUNTS = (8, 12, 16, 20)
RIDGE_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)


def spectral_envelope_features(
    samples: np.ndarray, sample_rate: int, *, band_count: int
) -> np.ndarray:
    frame_length = 4096
    hop = 2048
    if len(samples) < frame_length:
        raise ValueError("spectral envelope window is too short")
    spectra = []
    window = np.hanning(frame_length)
    for start in range(0, len(samples) - frame_length + 1, hop):
        frame = samples[start : start + frame_length]
        spectra.append(np.abs(np.fft.rfft(frame * window)) ** 2)
    power = np.mean(spectra, axis=0)
    frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
    edges = np.linspace(200.0, 3500.0, band_count + 1)
    features = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        features.append(np.log(max(float(np.mean(power[mask])), 1e-18)))
    vector = np.asarray(features)
    return (vector - np.mean(vector)) / max(float(np.std(vector)), 1e-9)


def _nearest_vowel(formants: np.ndarray) -> str:
    return min(
        VOWELS,
        key=lambda vowel: (
            (formants[0] - VOWEL_FORMANTS_HZ[vowel][0]) ** 2
            + (formants[1] - VOWEL_FORMANTS_HZ[vowel][1]) ** 2
            + 0.1 * (formants[2] - VOWEL_FORMANTS_HZ[vowel][2]) ** 2
        ),
    )


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.maximum(np.std(x, axis=0), 1e-9)
    standardized = (x - mean) / std
    design = np.column_stack((np.ones(len(standardized)), standardized))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, std, weights


def _predict(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std, weights = model
    design = np.concatenate(([1.0], (x - mean) / std))
    return design @ weights


def load_observations(*, corpus_root: str | Path, manifest_path: str | Path) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest.get("members")
    if not isinstance(records, list) or len(records) != 25:
        raise ValueError("spectral mapping manifest must contain 25 clips")
    root = Path(corpus_root)
    observations = []
    for record in records:
        member = str(record["member"])
        match = PATTERN.fullmatch(member)
        if match is None:
            raise ValueError(f"unexpected spectral mapping member: {member}")
        path = root.joinpath(*Path(member).parts)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            raise ValueError(f"spectral mapping hash mismatch: {member}")
        samples, rate = read_pcm_wav(path)
        mono = np.mean(samples, axis=1)
        start = (len(mono) - rate) // 2
        observations.append(
            {
                "singer": match.group("singer"),
                "vowel": match.group("vowel"),
                "window": mono[start : start + rate],
                "sample_rate": rate,
                "member": member,
                "sha256": digest,
            }
        )
    return observations


def audit_spectral_mapping_candidates(
    *, corpus_root: str | Path, manifest_path: str | Path
) -> dict[str, object]:
    observations = load_observations(corpus_root=corpus_root, manifest_path=manifest_path)
    singers = sorted({str(item["singer"]) for item in observations})
    reports = {}
    for band_count in BAND_COUNTS:
        features = {
            str(item["member"]): spectral_envelope_features(
                item["window"], int(item["sample_rate"]), band_count=band_count
            )
            for item in observations
        }
        for ridge in RIDGE_VALUES:
            rows = []
            for held_out in singers:
                training = [item for item in observations if item["singer"] != held_out]
                testing = [item for item in observations if item["singer"] == held_out]
                x = np.stack([features[str(item["member"])] for item in training])
                y = np.stack([VOWEL_FORMANTS_HZ[str(item["vowel"])] for item in training])
                model = _fit_ridge(x, y, ridge)
                for item in testing:
                    predicted_formants = _predict(features[str(item["member"])], model)
                    predicted_vowel = _nearest_vowel(predicted_formants)
                    rows.append(
                        {
                            "singer": item["singer"],
                            "expected_vowel": item["vowel"],
                            "predicted_vowel": predicted_vowel,
                            "correct": predicted_vowel == item["vowel"],
                            "predicted_formants_hz": predicted_formants.tolist(),
                        }
                    )
            accuracy = sum(bool(row["correct"]) for row in rows) / len(rows)
            key = f"bands_{band_count}_ridge_{ridge:g}"
            reports[key] = {
                "mapping": "ridge_to_continuous_formants",
                "band_count": band_count,
                "ridge": ridge,
                "accuracy": accuracy,
                "rows": rows,
            }
        for metric in ("euclidean", "cosine"):
            rows = []
            for held_out in singers:
                training = [item for item in observations if item["singer"] != held_out]
                testing = [item for item in observations if item["singer"] == held_out]
                train_matrix = np.stack(
                    [features[str(item["member"])] for item in training]
                )
                mean = np.mean(train_matrix, axis=0)
                std = np.maximum(np.std(train_matrix, axis=0), 1e-9)
                normalized = (train_matrix - mean) / std
                centroids = {
                    vowel: np.mean(
                        [
                            normalized[index]
                            for index, item in enumerate(training)
                            if item["vowel"] == vowel
                        ],
                        axis=0,
                    )
                    for vowel in VOWELS
                }
                for item in testing:
                    vector = (features[str(item["member"])] - mean) / std
                    if metric == "euclidean":
                        distance = lambda label: float(
                            np.linalg.norm(vector - centroids[label])
                        )
                    else:
                        distance = lambda label: float(
                            1.0
                            - np.dot(vector, centroids[label])
                            / max(
                                np.linalg.norm(vector)
                                * np.linalg.norm(centroids[label]),
                                1e-12,
                            )
                        )
                    predicted_vowel = min(VOWELS, key=distance)
                    rows.append(
                        {
                            "singer": item["singer"],
                            "expected_vowel": item["vowel"],
                            "predicted_vowel": predicted_vowel,
                            "correct": predicted_vowel == item["vowel"],
                            "predicted_formants_hz": list(
                                VOWEL_FORMANTS_HZ[predicted_vowel]
                            ),
                        }
                    )
            key = f"bands_{band_count}_centroid_{metric}"
            reports[key] = {
                "mapping": "spectral_centroid_to_canonical_motor_target",
                "band_count": band_count,
                "metric": metric,
                "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
                "rows": rows,
            }
    selected = max(
        reports,
        key=lambda key: (
            reports[key]["accuracy"],
            -reports[key]["band_count"],
            -float(reports[key].get("ridge", 0.0)),
        ),
    )
    return {
        "mode": "vocalset_spectral_to_formant_development_audit",
        "evidence_scope": "development_only_leave_one_singer_out",
        "candidate_grid": {
            "band_counts": list(BAND_COUNTS),
            "ridge_values": list(RIDGE_VALUES),
        },
        "selected_candidate": selected,
        "selected_report": reports[selected],
        "reports": reports,
        "confirmation_audio_accessed": False,
        "motor_coordinate_output": "predicted F1/F2/F3 in Hz",
        "promotion_allowed": False,
    }
