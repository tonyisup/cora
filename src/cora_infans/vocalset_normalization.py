"""Development-only speaker normalization audit for recorded vowel closure."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .io.motor_plan import VOWEL_FORMANTS_HZ, analyze_vowel
from .io.wav_frames import read_pcm_wav


VOWELS = ("a", "e", "i", "o", "u")
MEMBER_PATTERN = re.compile(
    r"^FULL/(?P<singer>female\d+|male\d+)/long_tones/straight/"
    r"[fm]\d+_long_straight_(?P<vowel>[aeiou])\.wav$"
)
CANDIDATES = (
    "raw_fixed_table",
    "speaker_z_f1_f2",
    "speaker_z_log_f1_f2",
    "speaker_z_f1_f2_f3",
    "speaker_z_log_f1_f2_f3",
)


def _normalize(matrix: np.ndarray, *, log: bool) -> np.ndarray:
    values = np.log(np.maximum(matrix, 1.0)) if log else matrix.copy()
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    return (values - mean) / np.maximum(std, 1e-9)


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    correct = sum(bool(row["correct"]) for row in rows)
    confusion = {v: {guess: 0 for guess in VOWELS} for v in VOWELS}
    by_singer: dict[str, list[int]] = {}
    by_vowel = {v: [0, 0] for v in VOWELS}
    for row in rows:
        expected = str(row["expected_vowel"])
        predicted = str(row["predicted_vowel"])
        singer = str(row["singer"])
        confusion[expected][predicted] += 1
        by_singer.setdefault(singer, [0, 0])
        by_singer[singer][0] += int(bool(row["correct"]))
        by_singer[singer][1] += 1
        by_vowel[expected][0] += int(bool(row["correct"]))
        by_vowel[expected][1] += 1
    return {
        "accuracy": correct / len(rows),
        "correct_count": correct,
        "confusion_matrix": confusion,
        "singer_accuracy": {
            singer: hits / total for singer, (hits, total) in by_singer.items()
        },
        "vowel_recall": {
            vowel: hits / total for vowel, (hits, total) in by_vowel.items()
        },
        "rows": rows,
    }


def audit_normalization_candidates(
    *, corpus_root: str | Path, manifest_path: str | Path
) -> dict[str, object]:
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest.get("members")
    if not isinstance(records, list) or len(records) != 25:
        raise ValueError("normalization development manifest must contain 25 clips")
    root = Path(corpus_root)
    observations = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        member = str(record["member"])
        match = MEMBER_PATTERN.fullmatch(member)
        if match is None:
            raise ValueError(f"unexpected development member: {member}")
        singer, vowel = match.group("singer"), match.group("vowel")
        if (singer, vowel) in seen:
            raise ValueError("duplicate development singer/vowel pair")
        seen.add((singer, vowel))
        path = root.joinpath(*Path(member).parts)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            raise ValueError(f"development member hash mismatch: {member}")
        samples, rate = read_pcm_wav(path)
        mono = np.mean(samples, axis=1)
        window_size = rate
        start = (len(mono) - window_size) // 2
        analysis = analyze_vowel(mono[start : start + window_size], sample_rate=rate)
        observations.append(
            {
                "singer": singer,
                "vowel": vowel,
                "formants": np.asarray(analysis.formant_hz, dtype=np.float64),
                "raw_prediction": analysis.nearest_vowel,
                "member": member,
                "sha256": digest,
            }
        )
    singers = sorted({str(item["singer"]) for item in observations})
    if len(singers) != 5 or any(
        {str(item["vowel"]) for item in observations if item["singer"] == singer}
        != set(VOWELS)
        for singer in singers
    ):
        raise ValueError("development set must contain five vowels for five singers")

    reports: dict[str, dict[str, object]] = {}
    raw_rows = [
        {
            "singer": item["singer"],
            "expected_vowel": item["vowel"],
            "predicted_vowel": item["raw_prediction"],
            "correct": item["vowel"] == item["raw_prediction"],
        }
        for item in observations
    ]
    reports["raw_fixed_table"] = _metrics(raw_rows)

    for candidate in CANDIDATES[1:]:
        dimensions = 3 if candidate.endswith("f3") else 2
        log = "_log_" in candidate
        rows = []
        for held_out in singers:
            normalized_by_singer = {}
            for singer in singers:
                ordered = sorted(
                    (item for item in observations if item["singer"] == singer),
                    key=lambda item: VOWELS.index(str(item["vowel"])),
                )
                matrix = np.stack([item["formants"][:dimensions] for item in ordered])
                normalized_by_singer[singer] = _normalize(matrix, log=log)
            centroids = {
                vowel: np.mean(
                    [
                        normalized_by_singer[singer][VOWELS.index(vowel)]
                        for singer in singers
                        if singer != held_out
                    ],
                    axis=0,
                )
                for vowel in VOWELS
            }
            for vowel in VOWELS:
                vector = normalized_by_singer[held_out][VOWELS.index(vowel)]
                predicted = min(
                    VOWELS,
                    key=lambda label: float(np.linalg.norm(vector - centroids[label])),
                )
                rows.append(
                    {
                        "singer": held_out,
                        "expected_vowel": vowel,
                        "predicted_vowel": predicted,
                        "correct": vowel == predicted,
                    }
                )
        reports[candidate] = _metrics(rows)

    selected = max(
        CANDIDATES,
        key=lambda name: (
            float(reports[name]["accuracy"]),
            min(reports[name]["vowel_recall"].values()),
            min(reports[name]["singer_accuracy"].values()),
            -CANDIDATES.index(name),
        ),
    )
    return {
        "mode": "vocalset_speaker_normalization_development_audit",
        "evidence_scope": "development_only_leave_one_singer_out",
        "candidate_order": list(CANDIDATES),
        "selection_rule": (
            "maximum LOO accuracy, then minimum vowel recall, minimum singer "
            "accuracy, then earliest fixed candidate order"
        ),
        "singers": singers,
        "reports": reports,
        "selected_candidate": selected,
        "selected_report": reports[selected],
        "confirmation_audio_accessed": False,
        "promotion_allowed": False,
    }
