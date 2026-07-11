"""Held-out human-vowel analysis-by-synthesis audit for VISION H4."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .io.motor_plan import (
    VOWEL_FORMANTS_HZ,
    analyze_vowel,
    evaluate_vowel_round_trip,
)
from .io.wav_frames import read_pcm_wav


VOCALSET_CONFIRMATION_SINGERS = ("female1", "female3", "male1", "male2", "male4")
VOWELS = ("a", "e", "i", "o", "u")
MEMBER_PATTERN = re.compile(
    r"^FULL/(?P<singer>female1|female3|male1|male2|male4)/"
    r"long_tones/straight/[fm]\d+_long_straight_(?P<vowel>[aeiou])\.wav$"
)


def build_recorded_vowel_protocol() -> dict[str, object]:
    return {
        "mode": "vocalset_held_out_recorded_vowel_audit",
        "source": "VocalSet 1.0",
        "doi": "10.5281/zenodo.1193957",
        "license": "CC BY 4.0",
        "confirmation_singers": list(VOCALSET_CONFIRMATION_SINGERS),
        "selection_rule": (
            "first two female and first three male IDs in the published training "
            "list; no selected audio fetched before protocol commit"
        ),
        "vowels": list(VOWELS),
        "expected_clip_count": 25,
        "analysis_window": "center 1.0 second of each straight long tone",
        "estimator_selection": "existing label-blind LPC estimator; no corpus tuning",
        "acceptance": {
            "clean_synthetic_accuracy_min": 0.90,
            "recorded_accuracy_min": 0.80,
            "each_vowel_recall_min": 0.60,
            "each_singer_accuracy_min": 0.60,
            "all_five_by_five_cells_reported": True,
            "f1_f2_f3_mae_reported_separately": True,
            "all_member_hashes_verified": True,
            "full_scale_analysis_windows_reported": True,
        },
        "formant_error_boundary": (
            "F1/F2/F3 errors are reported against Cora's fixed adult reference "
            "targets, not treated as VocalSet singer-specific ground truth"
        ),
        "motor_learning_claim_allowed": False,
    }


def evaluate_recorded_vowels(
    *, corpus_root: str | Path, manifest_path: str | Path
) -> dict[str, object]:
    protocol = build_recorded_vowel_protocol()
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("doi") != protocol["doi"] or manifest.get("license") != "CC BY 4.0":
        raise ValueError("VocalSet source or license metadata mismatch")
    records = manifest.get("members")
    if not isinstance(records, list) or len(records) != 25:
        raise ValueError("VocalSet held-out manifest must contain exactly 25 members")

    root = Path(corpus_root)
    expected_pairs = {
        (singer, vowel) for singer in VOCALSET_CONFIRMATION_SINGERS for vowel in VOWELS
    }
    seen_pairs: set[tuple[str, str]] = set()
    confusion = {vowel: {guess: 0 for guess in VOWELS} for vowel in VOWELS}
    singer_counts = {singer: [0, 0] for singer in VOCALSET_CONFIRMATION_SINGERS}
    vowel_counts = {vowel: [0, 0] for vowel in VOWELS}
    errors: list[list[float]] = [[], [], []]
    trials = []
    clipped_windows = 0

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("VocalSet manifest members must be objects")
        member = str(record.get("member", ""))
        match = MEMBER_PATTERN.fullmatch(member)
        if match is None:
            raise ValueError(f"unexpected VocalSet held-out member: {member}")
        singer, vowel = match.group("singer"), match.group("vowel")
        pair = (singer, vowel)
        if pair in seen_pairs:
            raise ValueError(f"duplicate VocalSet singer/vowel pair: {pair}")
        seen_pairs.add(pair)
        path = root.joinpath(*Path(member).parts)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            raise ValueError(f"VocalSet member hash mismatch: {member}")
        samples, sample_rate = read_pcm_wav(path)
        mono = np.mean(samples, axis=1)
        window_samples = sample_rate
        if len(mono) < window_samples:
            raise ValueError(f"VocalSet clip is shorter than one second: {member}")
        start = (len(mono) - window_samples) // 2
        window = mono[start : start + window_samples]
        clipped = int(np.count_nonzero(np.abs(window) >= (32767.0 / 32768.0)))
        clipped_windows += int(clipped > 0)
        analysis = analyze_vowel(window, sample_rate=sample_rate)
        target = VOWEL_FORMANTS_HZ[vowel]
        absolute_errors = [
            abs(measured - expected)
            for measured, expected in zip(analysis.formant_hz, target)
        ]
        for index, error in enumerate(absolute_errors):
            errors[index].append(error)
        correct = analysis.nearest_vowel == vowel
        confusion[vowel][analysis.nearest_vowel] += 1
        singer_counts[singer][0] += int(correct)
        singer_counts[singer][1] += 1
        vowel_counts[vowel][0] += int(correct)
        vowel_counts[vowel][1] += 1
        trials.append(
            {
                "member": member,
                "sha256": digest,
                "singer": singer,
                "expected_vowel": vowel,
                "perceived_vowel": analysis.nearest_vowel,
                "correct": correct,
                "sample_rate": sample_rate,
                "analysis_start_sample": start,
                "analysis_sample_count": window_samples,
                "target_formants_hz": list(target),
                "measured_formants_hz": list(analysis.formant_hz),
                "absolute_formant_error_hz": absolute_errors,
                "clipped_sample_count": clipped,
            }
        )
    if seen_pairs != expected_pairs:
        raise ValueError("VocalSet manifest does not cover every held-out singer/vowel pair")

    correct_count = sum(int(trial["correct"]) for trial in trials)
    recorded_accuracy = correct_count / len(trials)
    vowel_recall = {
        vowel: correct / total for vowel, (correct, total) in vowel_counts.items()
    }
    singer_accuracy = {
        singer: correct / total for singer, (correct, total) in singer_counts.items()
    }
    synthetic = evaluate_vowel_round_trip()
    criteria = {
        "clean_synthetic_accuracy_at_least_0_90": synthetic["accuracy"] >= 0.90,
        "recorded_accuracy_at_least_0_80": recorded_accuracy >= 0.80,
        "each_vowel_recall_at_least_0_60": min(vowel_recall.values()) >= 0.60,
        "each_singer_accuracy_at_least_0_60": min(singer_accuracy.values()) >= 0.60,
        "all_five_by_five_cells_reported": (
            set(confusion) == set(VOWELS)
            and all(set(row) == set(VOWELS) for row in confusion.values())
        ),
        "f1_f2_f3_mae_reported_separately": all(len(values) == 25 for values in errors),
        "all_member_hashes_verified": True,
        "full_scale_analysis_windows_reported": True,
    }
    return {
        **protocol,
        "verdict": "recorded_vowel_gate_passed" if all(criteria.values()) else "recorded_vowel_gate_failed",
        "synthetic": synthetic,
        "recorded": {
            "trial_count": len(trials),
            "correct_count": correct_count,
            "accuracy": recorded_accuracy,
            "confusion_matrix": confusion,
            "vowel_recall": vowel_recall,
            "singer_accuracy": singer_accuracy,
            "formant_mae_hz": {
                f"f{index + 1}": float(np.mean(values))
                for index, values in enumerate(errors)
            },
            "clipped_analysis_windows": clipped_windows,
            "trials": trials,
        },
        "criteria": criteria,
        "promotion_allowed": all(criteria.values()),
        "claim_boundary": (
            "A pass supports fixed-table vowel identity closure on clean synthesis "
            "and five held-out VocalSet singers performing straight sung long tones. "
            "It does not establish ordinary speech, infant speech, or motor learning."
        ),
    }
