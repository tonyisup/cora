from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from cora_infans.cli import _load_register_model_artifact
from cora_infans.register_saliency import (
    REGISTER_FEATURE_NAMES,
    RegisterExample,
    SpeakerAdaptiveRegisterModel,
    VocalActivityGate,
    evaluate_leave_one_fieldsite_out,
    evaluate_register_pipeline,
    extract_register_features,
    select_register_thresholds,
)


def _example(clip_id: str, label: str, offset: float) -> RegisterExample:
    features = {
        name: offset + index * 0.01
        for index, name in enumerate(REGISTER_FEATURE_NAMES)
    }
    if label.endswith("song"):
        features["modulation_low_band_fraction"] += 2.0
        features["modulation_speech_band_fraction"] -= 1.0
    if label == "infant_directed_speech":
        features["pitch_log2_median"] += 1.0
        features["pitch_iqr_semitones"] += 1.0
    return RegisterExample(
        clip_id=clip_id,
        speaker_id=clip_id[:-1],
        fieldsite=clip_id[:3],
        label=label,
        features=features,
    )


def _paired_examples(fieldsite: str, speaker: str, offset: float):
    prefix = f"{fieldsite}{speaker}"
    return [
        _example(prefix + "A", "infant_directed_song", offset),
        _example(prefix + "B", "infant_directed_speech", offset),
        _example(prefix + "C", "adult_directed_song", offset),
        _example(prefix + "D", "adult_directed_speech", offset),
    ]


def test_register_features_include_long_window_modulation() -> None:
    sample_rate = 8000
    time = np.arange(sample_rate * 2) / sample_rate
    carrier = np.sin(2 * np.pi * 220 * time)
    envelope = 0.5 + 0.4 * np.sin(2 * np.pi * 4 * time)

    features = extract_register_features(carrier * envelope, sample_rate)

    assert set(features) == set(REGISTER_FEATURE_NAMES)
    assert 3.5 <= features["modulation_peak_hz"] <= 4.5
    assert features["yin_pitch_median_hz"] == pytest.approx(220.0, rel=0.03)
    assert all(np.isfinite(value) for value in features.values())


def test_vocal_activity_gate_uses_explicit_yin_coverage_threshold() -> None:
    gate = VocalActivityGate()

    assert gate.predicts_vocal({"yin_voiced_fraction": 0.2}) is True
    assert gate.predicts_vocal({"yin_voiced_fraction": 0.1}) is False
    assert gate.to_dict()["feature"] == "yin_voiced_fraction"
    assert VocalActivityGate.from_dict(gate.to_dict()) == gate


def test_speaker_adaptive_model_round_trips_and_reports_calibration() -> None:
    training = [
        *_paired_examples("AAA", "01", 0.0),
        *_paired_examples("BBB", "01", 4.0),
    ]
    held_out = _paired_examples("CCC", "01", 9.0)
    model = SpeakerAdaptiveRegisterModel.fit(training)
    restored = SpeakerAdaptiveRegisterModel.from_dict(model.to_dict())

    report = restored.evaluate_batch(held_out)

    assert report["calibrated_prediction_count"] == 4
    assert report["calibration_boundary"].startswith("unlabeled within-speaker")
    assert all(item["calibration_observation_count"] == 2 for item in report["predictions"])
    assert restored.adaptation_scope == "speaker_and_mode"
    assert restored.vocal_gate == VocalActivityGate()
    assert restored.speech_score_minimum == 0.0


def test_single_observation_is_not_presented_as_calibrated() -> None:
    training = [
        *_paired_examples("AAA", "01", 0.0),
        *_paired_examples("BBB", "01", 4.0),
    ]
    model = SpeakerAdaptiveRegisterModel.fit(training)

    report = model.evaluate_batch([_example("CCC01B", "infant_directed_speech", 9.0)])

    assert report["calibrated_prediction_count"] == 0
    assert report["predictions"][0]["predicted_ids"] is False


def test_legacy_model_defaults_to_all_mode_adaptation() -> None:
    training = [
        *_paired_examples("AAA", "01", 0.0),
        *_paired_examples("BBB", "01", 4.0),
    ]
    payload = SpeakerAdaptiveRegisterModel.fit(training).to_dict()
    payload.pop("adaptation_scope")
    payload.pop("vocal_gate")

    restored = SpeakerAdaptiveRegisterModel.from_dict(payload)

    assert restored.adaptation_scope == "speaker_all_modes"
    assert restored.vocal_gate is None


def test_vocal_gate_blocks_register_promotion() -> None:
    training = [
        *_paired_examples("AAA", "01", 0.0),
        *_paired_examples("BBB", "01", 4.0),
    ]
    model = SpeakerAdaptiveRegisterModel.fit(training)
    held_out = _paired_examples("CCC", "01", 9.0)
    for example in held_out:
        example.features["yin_voiced_fraction"] = 0.0

    report = model.evaluate_batch(held_out)

    assert report["predicted_vocal_count"] == 0
    assert all(item["predicted_ids"] is False for item in report["predictions"])


def test_leave_one_fieldsite_out_never_fits_test_site() -> None:
    examples = [
        *_paired_examples("AAA", "01", 0.0),
        *_paired_examples("BBB", "01", 4.0),
        *_paired_examples("CCC", "01", 9.0),
    ]

    report = evaluate_leave_one_fieldsite_out(examples)

    assert report["fieldsite_count"] == 3
    assert report["prediction_count"] == 12
    assert set(report["per_fieldsite"]) == {"AAA", "BBB", "CCC"}


def test_composed_artifact_hash_pins_base_model(tmp_path) -> None:
    training = [
        *_paired_examples("AAA", "01", 0.0),
        *_paired_examples("BBB", "01", 4.0),
    ]
    base_payload = {
        "model": {
            **SpeakerAdaptiveRegisterModel.fit(training).to_dict(),
            "vocal_gate": None,
        }
    }
    base_bytes = (json.dumps(base_payload) + "\n").encode()
    base_path = tmp_path / "base.json"
    base_path.write_bytes(base_bytes)
    composition = {
        "mode": "register_pipeline_composition",
        "base_register_model_artifact": "base.json",
        "base_register_model_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "vocal_gate": VocalActivityGate().to_dict(),
    }
    composition_path = tmp_path / "composition.json"
    composition_path.write_text(json.dumps(composition))

    restored = _load_register_model_artifact(composition_path)

    assert restored.vocal_gate == VocalActivityGate()


def test_joint_pipeline_reports_noise_pass_through_as_abstention() -> None:
    training = [
        *_paired_examples("AAA", "01", 0.0),
        *_paired_examples("BBB", "01", 4.0),
    ]
    model = SpeakerAdaptiveRegisterModel.fit(training)
    vocal = _paired_examples("CCC", "01", 9.0)
    rejected = _example("ENV001", "environmental_diffuse", 0.0)
    rejected.features["yin_voiced_fraction"] = 0.1
    passed = _example("ENV002", "environmental_impulsive", 0.0)
    passed.features["yin_voiced_fraction"] = 0.3

    report = evaluate_register_pipeline(model, vocal, [rejected, passed])

    assert report["vocal_recall"] == 1.0
    assert report["environmental_pass_through_rate"] == 0.5
    assert report["per_environmental_label"]["environmental_diffuse"][
        "pass_through_rate"
    ] == 0.0
    assert "abstentions" in report["claim_boundary"]


def test_threshold_selection_uses_grouped_predictions() -> None:
    examples = [
        *_paired_examples("AAA", "01", 0.0),
        *_paired_examples("BBB", "01", 4.0),
        *_paired_examples("CCC", "01", 9.0),
    ]

    selection = select_register_thresholds(examples)

    assert selection["protocol"] == "leave_one_fieldsite_out_threshold_selection"
    assert selection["fieldsite_count"] == 3
    assert np.isfinite(selection["speech_score_minimum"])
    assert np.isfinite(selection["ids_score_minimum"])
