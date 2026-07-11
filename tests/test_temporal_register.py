from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cora_infans.register_saliency import RegisterExample
from cora_infans.temporal_register import (
    TEMPORAL_FEATURE_NAMES,
    TemporalRegisterModel,
    evaluate_temporal_leave_one_fieldsite_out,
    extract_temporal_register_features,
    select_temporal_register_thresholds,
)


def test_temporal_features_preserve_phase_order() -> None:
    sample_rate = 8000
    time = np.arange(sample_rate * 3) / sample_rate
    rising = np.sin(2 * np.pi * (120.0 * time + 30.0 * time**2))

    forward = extract_temporal_register_features(rising, sample_rate)
    reverse = extract_temporal_register_features(rising[::-1], sample_rate)

    assert set(TEMPORAL_FEATURE_NAMES).issubset(forward)
    assert forward["pitch_log2_phase_0"] < forward["pitch_log2_phase_5"]
    assert reverse["pitch_log2_phase_0"] > reverse["pitch_log2_phase_5"]
    assert all(np.isfinite(value) for value in forward.values())


def _examples(fieldsite: str, offset: float) -> list[RegisterExample]:
    examples = []
    for condition, label in zip(
        "ABCD",
        (
            "infant_directed_song",
            "infant_directed_speech",
            "adult_directed_song",
            "adult_directed_speech",
        ),
        strict=True,
    ):
        features = {name: offset for name in TEMPORAL_FEATURE_NAMES}
        features["pitch_log2_phase_5"] += 1.0 if condition in "AB" else -1.0
        features["spectral_flux_log_phase_3"] += 1.0 if condition in "BD" else -1.0
        features["yin_voiced_fraction"] = 0.5
        examples.append(
            RegisterExample(
                clip_id=f"{fieldsite}01{condition}",
                speaker_id=f"{fieldsite}01",
                fieldsite=fieldsite,
                label=label,
                features=features,
            )
        )
    return examples


def test_temporal_model_is_grouped_and_confirmation_safe() -> None:
    examples = [
        *_examples("AAA", 0.0),
        *_examples("BBB", 2.0),
        *_examples("CCC", 4.0),
    ]

    model = TemporalRegisterModel.fit(examples[:8])
    batch = model.evaluate_batch(examples[8:])
    report = evaluate_temporal_leave_one_fieldsite_out(examples)

    assert batch["metrics"]["f1"] >= 0.0
    assert report["fieldsite_count"] == 3
    assert report["prediction_count"] == 12
    assert 0.0 <= report["song_false_positive_rate"] <= 1.0
    assert report["vocal_recall"] == 1.0
    assert report["confirmation_fieldsites_accessed"] is False
    assert report["promotion_allowed"] is False

    selection = select_temporal_register_thresholds(report["predictions"])
    assert selection["protocol"] == "out_of_field_temporal_threshold_selection"
    assert selection["metrics"]["false_positive_rate"] <= 0.15


def test_retained_temporal_result_preserves_falsification_and_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-11-temporal-register-development-result.json"
        ).read_text()
    )

    assert result["source"]["clip_count"] == 200
    assert result["source"]["fieldsite_count"] == 18
    assert result["zero_threshold_report"]["f1"] == 0.3018867924528302
    assert result["out_of_field_threshold_selection"]["f1"] == 0.28888888888888886
    assert result["out_of_field_threshold_selection"]["minimum_fieldsite_f1"] == 0.0
    assert result["development_candidate_ready"] is False
    assert result["confirmation_fieldsites_accessed"] is False
    assert result["promotion_allowed"] is False
