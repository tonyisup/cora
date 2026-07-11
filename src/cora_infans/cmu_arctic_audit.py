"""Development-only ordinary-speech vowel audit on CMU Arctic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .io.motor_plan import VOWEL_FORMANTS_HZ, analyze_vowel
from .io.wav_frames import read_pcm_wav


VOWELS = ("a", "e", "i", "o", "u")
WINDOW_MS = (100, 110, 120)
LPC_ORDERS = (10, 12, 14, 16, 18, 20)


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    confusion = {vowel: {guess: 0 for guess in VOWELS} for vowel in VOWELS}
    vowel_counts = {vowel: [0, 0] for vowel in VOWELS}
    speaker_counts: dict[str, list[int]] = {}
    formant_errors = [[], [], []]
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
        for index, (measured, target) in enumerate(
            zip(row["formants_hz"], VOWEL_FORMANTS_HZ[expected])
        ):
            formant_errors[index].append(abs(float(measured) - float(target)))
    correct_count = sum(expected == predicted for expected, predicted in (
        (str(row["expected_vowel"]), str(row["predicted_vowel"])) for row in rows
    ))
    return {
        "accuracy": correct_count / len(rows),
        "correct_count": correct_count,
        "confusion_matrix": confusion,
        "vowel_recall": {
            vowel: hits / total for vowel, (hits, total) in vowel_counts.items()
        },
        "speaker_accuracy": {
            speaker: hits / total
            for speaker, (hits, total) in speaker_counts.items()
        },
        "formant_mae_hz": [float(np.mean(errors)) for errors in formant_errors],
        "rows": rows,
    }


def audit_cmu_arctic_development(
    *, corpus_root: str | Path, manifest_path: str | Path
) -> dict[str, object]:
    root = Path(corpus_root)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("confirmation_audio_or_labels_accessed") is not False:
        raise ValueError("CMU Arctic development manifest crosses confirmation boundary")
    tokens = manifest.get("tokens")
    wav_records = manifest.get("wav_files")
    if not isinstance(tokens, list) or len(tokens) != 500:
        raise ValueError("CMU Arctic development audit requires 500 tokens")
    if not isinstance(wav_records, list):
        raise ValueError("CMU Arctic development manifest is missing WAV records")
    hashes = {
        (str(record["speaker"]), str(record["name"])): str(record["sha256"])
        for record in wav_records
    }
    wav_cache: dict[tuple[str, str], tuple[np.ndarray, int]] = {}
    for key, expected_hash in hashes.items():
        speaker, name = key
        path = root / speaker / "wav" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"CMU Arctic WAV hash mismatch: {speaker}/{name}")
        samples, sample_rate = read_pcm_wav(path)
        wav_cache[key] = (np.mean(samples, axis=1), sample_rate)

    reports = {}
    for window_ms in WINDOW_MS:
        for order in LPC_ORDERS:
            rows = []
            clipped_sample_count = 0
            for token in tokens:
                speaker = str(token["speaker"])
                wav_name = str(token["wav_name"])
                samples, sample_rate = wav_cache[(speaker, wav_name)]
                width = int(round(window_ms / 1000.0 * sample_rate))
                center = int(
                    round(
                        (
                            float(token["start_seconds"])
                            + float(token["end_seconds"])
                        )
                        / 2.0
                        * sample_rate
                    )
                )
                start = center - width // 2
                window = samples[start : start + width]
                if len(window) != width:
                    raise ValueError("CMU Arctic token window exceeds its WAV")
                clipped_sample_count += int(np.count_nonzero(np.abs(window) >= 1.0))
                analysis = analyze_vowel(
                    window, sample_rate=sample_rate, order=order
                )
                rows.append(
                    {
                        "speaker": speaker,
                        "wav_name": wav_name,
                        "segment_index": token["segment_index"],
                        "expected_vowel": token["vowel"],
                        "predicted_vowel": analysis.nearest_vowel,
                        "formants_hz": list(analysis.formant_hz),
                    }
                )
            key = f"window_{window_ms}ms_order_{order}"
            reports[key] = {
                "window_ms": window_ms,
                "lpc_order": order,
                "clipped_sample_count": clipped_sample_count,
                **_metrics(rows),
            }
    selected = max(
        reports,
        key=lambda key: (
            reports[key]["accuracy"],
            min(reports[key]["vowel_recall"].values()),
            min(reports[key]["speaker_accuracy"].values()),
            -reports[key]["window_ms"],
            -reports[key]["lpc_order"],
        ),
    )
    return {
        "mode": "cmu_arctic_ordinary_speech_development_audit",
        "evidence_scope": "development_only_fixed_formant_table",
        "candidate_grid": {
            "window_ms": list(WINDOW_MS),
            "lpc_orders": list(LPC_ORDERS),
        },
        "token_count": len(tokens),
        "wav_count": len(wav_records),
        "selected_candidate": selected,
        "selected_report": reports[selected],
        "reports": reports,
        "confirmation_audio_or_labels_accessed": False,
        "promotion_allowed": False,
    }
