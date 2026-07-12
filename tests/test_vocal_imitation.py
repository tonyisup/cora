"""Contracts for bounded project-owner-recorded vocal imitation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cora_infans.factorized_tract_plant import synthesize_factorized
from cora_infans.speech_learning_demo import CROSS_CONFIGURATION_TARGET_MOTORS
from cora_infans.vocal_imitation import (
    acoustic_sensory,
    build_waveform_repertoire,
    capture_project_owner_take,
    run_project_owner_vocal_imitation,
)


def _capture_wave(target: str, f0_hz: float) -> np.ndarray:
    production = synthesize_factorized(
        np.asarray(CROSS_CONFIGURATION_TARGET_MOTORS[target]), f0_hz
    ) * 0.6
    mono = np.resize(production, 22000)
    capture = np.zeros((28800, 2), dtype=np.float32)
    capture[3000:25000, 0] = mono
    capture[3000:25000, 1] = mono * 0.98
    return capture


def _register_complete_session(root: Path) -> None:
    for target in ("a", "i", "u"):
        for role, take, f0_hz in (
            ("development", 1, 90.0),
            ("development", 2, 140.0),
            ("confirmation", 1, 180.0),
        ):
            samples = _capture_wave(target, f0_hz)
            report = capture_project_owner_take(
                session_dir=root,
                target_key=target,
                role=role,
                take=take,
                authorized=True,
                capture_fn=lambda samples=samples: (
                    samples,
                    16000,
                    {
                        "backend": "synthetic_test_injection",
                        "device_selector": ":0",
                        "channel_count": 2,
                    },
                ),
            )
            assert report["capture_passed"] is True


def test_capture_requires_explicit_project_owner_authorization(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit authorization"):
        capture_project_owner_take(
            session_dir=tmp_path,
            target_key="a",
            role="development",
            take=1,
            authorized=False,
            capture_fn=lambda: (_capture_wave("a", 110.0), 16000, {}),
        )
    assert not list(tmp_path.rglob("*.wav"))


def test_acoustic_sensory_is_label_free_bounded_and_pitch_tolerant() -> None:
    low, low_quality = acoustic_sensory(_capture_wave("a", 90.0), 16000)
    high, high_quality = acoustic_sensory(_capture_wave("a", 180.0), 16000)

    assert low.shape == high.shape == (30,)
    assert np.linalg.norm(low) == pytest.approx(1.0)
    assert np.linalg.norm(high) == pytest.approx(1.0)
    assert low_quality["quality_passed"] is True
    assert high_quality["quality_passed"] is True
    assert np.linalg.norm(low - high) / 2.0 < 0.10


def test_waveform_repertoire_is_deterministic_and_uses_common_features() -> None:
    first = build_waveform_repertoire()
    second = build_waveform_repertoire()

    assert first.digest() == second.digest()
    assert len(first.prototypes) == 96
    assert all(len(item.sensory) == 30 for item in first.prototypes)


def test_complete_session_learns_before_confirmation_and_passes(tmp_path) -> None:
    _register_complete_session(tmp_path)
    with pytest.raises(ValueError, match="injected fixtures are test-only"):
        run_project_owner_vocal_imitation(session_dir=tmp_path)
    output = tmp_path / "result"
    report = run_project_owner_vocal_imitation(
        session_dir=tmp_path,
        output_dir=output,
        allow_test_fixture=True,
    )

    assert report["demo_passed"] is True
    assert report["evidence_class"] == "synthetic_injected_protocol_fixture"
    assert report["aggregate_criteria"]["exact_capture_count"] is True
    assert report["aggregate_criteria"]["confirmation_accessed_after_all_learning"] is True
    assert report["aggregate_criteria"]["external_corpus_not_accessed"] is True
    assert report["aggregate_criteria"]["protected_data_not_accessed"] is True
    assert len(report["sessions"]) == 3
    for session in report["sessions"]:
        assert session["learner_received_vowel_label"] is False
        assert session["confirmation"]["used_for_selection"] is False
        assert session["criteria"]["confirmation_did_not_change_association"] is True
        assert session["session_passed"] is True
        assert len(session["development"]["attempts"]) == 12
        assert (output / f"{session['display_label']}-learned.wav").exists()
        assert (output / "captures" / f"{session['display_label']}-confirmation-1.wav").exists()

    retained = json.loads((output / "vocal-imitation-result.json").read_text())
    assert retained == report
    dashboard = output / "vocal-imitation-demo.html"
    assert hashlib.sha256(dashboard.read_bytes()).hexdigest() == report["presentation"]["html_sha256"]
    assert "held confirmation" in dashboard.read_text()


def test_manifest_hash_tampering_fails_loudly(tmp_path) -> None:
    _register_complete_session(tmp_path)
    path = tmp_path / "captures/a-development-1.wav"
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="capture hash mismatch"):
        run_project_owner_vocal_imitation(
            session_dir=tmp_path, allow_test_fixture=True
        )


def test_confirmation_take_cannot_be_reused_or_overwritten(tmp_path) -> None:
    samples = _capture_wave("u", 110.0)
    kwargs = dict(
        session_dir=tmp_path,
        target_key="u",
        role="confirmation",
        take=1,
        authorized=True,
        capture_fn=lambda: (samples, 16000, {"backend": "test"}),
    )
    capture_project_owner_take(**kwargs)
    with pytest.raises(ValueError, match="already registered"):
        capture_project_owner_take(**kwargs)


def test_retained_prehuman_calibration_cannot_be_claimed_as_real() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/2026-07-12-vocal-imitation-v1-prehuman-calibration.json"
    )
    result = json.loads(path.read_text())

    assert result["fixture_passed"] is True
    assert result["human_recording_accessed"] is False
    assert result["real_capture_claim"] is False
    assert result["production_runner_accepts_this_fixture_as_evidence"] is False
    assert result["criteria"]["confirmation_accessed_after_all_learning"] is True
    assert "not a human-vocal-imitation result" in result["claim_boundary"]
