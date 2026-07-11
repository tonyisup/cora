from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cora_infans.cohort_adaptive_register import (
    CoralAlignment,
    cohort_adaptive_leave_one_fieldsite_out,
    coral_target_to_source,
    evaluate_cohort_adaptive_fold,
    select_cohort_adaptive_thresholds,
)
from cora_infans.predictive_register_dynamics import (
    CONDITION_LABELS,
    ProsodicSequenceExample,
)


def _site(fieldsite: str, offset: float, speaker_count: int = 3):
    examples = []
    for speaker_index in range(speaker_count):
        for condition_index, label in enumerate(CONDITION_LABELS):
            time = np.linspace(0.0, 1.0, 30)
            infant_slope = 1.0 if label.startswith("infant") else -1.0
            mode_curve = time**2 if label.endswith("speech") else np.sin(np.pi * time)
            states = np.column_stack(
                [
                    offset
                    + speaker_index
                    + feature * 0.1
                    + mode_curve * (feature + 1)
                    + infant_slope * time
                    for feature in range(5)
                ]
            )
            examples.append(
                ProsodicSequenceExample(
                    clip_id=f"{fieldsite}{speaker_index:02d}{'ABCD'[condition_index]}",
                    speaker_id=f"{fieldsite}{speaker_index:02d}",
                    fieldsite=fieldsite,
                    label=label,
                    states=states,
                    vocal_score=0.5,
                )
            )
    return examples


def test_coral_moves_target_transition_geometry_toward_source() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(500, 10)) @ np.diag(np.linspace(0.5, 2.0, 10))
    target = rng.normal(size=(500, 10)) @ np.diag(np.linspace(2.0, 0.5, 10)) + 4.0
    aligned = coral_target_to_source(target, source)

    assert np.allclose(np.mean(aligned, axis=0), np.mean(source, axis=0))
    before = np.linalg.norm(np.cov(target, rowvar=False) - np.cov(source, rowvar=False))
    after = np.linalg.norm(np.cov(aligned, rowvar=False) - np.cov(source, rowvar=False))
    assert after < before * 0.1


def test_target_ids_ads_labels_do_not_change_alignment_or_scores() -> None:
    training = [*_site("AAA", 0.0), *_site("BBB", 3.0)]
    target = _site("CCC", 8.0)
    relabeled = [
        ProsodicSequenceExample(
            clip_id=item.clip_id,
            speaker_id=item.speaker_id,
            fieldsite=item.fieldsite,
            label=(
                item.label.replace("infant_directed", "adult_directed")
                if item.label.startswith("infant")
                else item.label.replace("adult_directed", "infant_directed")
            ),
            states=item.states,
            vocal_score=item.vocal_score,
        )
        for item in target
    ]
    original = evaluate_cohort_adaptive_fold(training, target)
    swapped = evaluate_cohort_adaptive_fold(training, relabeled)

    assert original["cohort_alignment"] == swapped["cohort_alignment"]
    original_scores = {
        item["clip_id"]: (item["speech_score"], item["ids_score"])
        for item in original["predictions"]
    }
    swapped_scores = {
        item["clip_id"]: (item["speech_score"], item["ids_score"])
        for item in swapped["predictions"]
    }
    assert original_scores == swapped_scores


def test_grouped_audit_excludes_ineligible_one_speaker_fieldsite() -> None:
    examples = [
        *_site("AAA", 0.0),
        *_site("BBB", 3.0),
        *_site("CCC", 6.0),
        *_site("ARA", 9.0, speaker_count=1),
    ]
    report = cohort_adaptive_leave_one_fieldsite_out(examples)
    selection = select_cohort_adaptive_thresholds(report["predictions"])

    assert report["eligible_fieldsites"] == ["AAA", "BBB", "CCC"]
    assert report["ineligible_fieldsites"] == {
        "ARA": "1 target speaker(s); minimum is 3"
    }
    assert report["prediction_count"] == 36
    assert report["alignment_uses_target_ids_ads_labels"] is False
    assert report["known_vocal_mode_grouping"] is True
    assert selection["protocol"] == "cohort_adaptive_out_of_field_threshold_selection"
    assert selection["metrics"]["false_positive_rate"] <= 0.15


def test_alignment_serialization_discloses_batch_sizes() -> None:
    source = np.arange(200, dtype=np.float64).reshape(20, 10)
    target = np.arange(300, dtype=np.float64).reshape(30, 10)
    payload = CoralAlignment.fit(target, source).to_dict()

    assert payload["source_transition_count"] == 20
    assert payload["target_transition_count"] == 30
    assert len(payload["transform"]) == 10


def test_retained_cohort_result_preserves_stop_decision() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root / "docs/evidence/2026-07-11-cohort-adaptive-register-result.json"
        ).read_text()
    )

    assert result["source"]["evaluated_clip_count"] == 198
    assert result["source"]["eligible_fieldsite_count"] == 17
    assert result["out_of_field_threshold_selection"]["f1"] == 0.6808510638297872
    assert result["out_of_field_threshold_selection"]["minimum_fieldsite_f1"] == 0.0
    assert result["development_candidate_ready"] is False
    assert result["confirmation_audio_or_metadata_accessed"] is False
    assert result["promotion_allowed"] is False
