"""Contracts for bounded project-owner-recorded vocal imitation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cora_infans.factorized_tract_plant import synthesize_factorized
from cora_infans.speech_learning_demo import CROSS_CONFIGURATION_TARGET_MOTORS
from cora_infans.vocal_imitation import (
    _play_start_cue,
    acoustic_sensory,
    build_waveform_repertoire,
    capture_project_owner_take,
    freeze_project_owner_development,
    run_project_owner_vocal_imitation,
)


def _capture_wave(target: str, f0_hz: float) -> np.ndarray:
    production = synthesize_factorized(
        np.asarray(CROSS_CONFIGURATION_TARGET_MOTORS[target]), f0_hz
    ) * 0.6
    mono = np.resize(production, 48000)
    capture = np.zeros((64000, 2), dtype=np.float32)
    capture[8000:56000, 0] = mono
    capture[8000:56000, 1] = mono * 0.98
    return capture


def _register_complete_session(root: Path) -> None:
    for target in ("a", "i", "u"):
        for role, take, f0_hz in (
            ("development", 1, 90.0),
            ("development", 2, 140.0),
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
    freeze = freeze_project_owner_development(
        session_dir=root, allow_test_fixture=True
    )
    assert freeze["development_passed"] is True
    for target in ("a", "i", "u"):
        samples = _capture_wave(target, 180.0)
        report = capture_project_owner_take(
            session_dir=root,
            target_key=target,
            role="confirmation",
            take=1,
            authorized=True,
            capture_fn=lambda samples=samples: (
                samples,
                16000,
                {"backend": "synthetic_test_injection"},
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


def test_start_cue_is_pinned_bounded_and_blocking() -> None:
    calls = []

    class FakeSoundDevice:
        @staticmethod
        def query_devices():
            return [
                {"name": "External Headphones", "max_output_channels": 2},
                {"name": "Mac mini Speakers", "max_output_channels": 2},
            ]

        @staticmethod
        def play(samples, **kwargs):
            calls.append((np.asarray(samples), kwargs))

    result = _play_start_cue(FakeSoundDevice)

    assert result == {
        "played": True,
        "output_device_index": 1,
        "output_device_name": "Mac mini Speakers",
        "frequency_hz": 660.0,
        "duration_ms": 100,
        "amplitude": 0.08,
        "retained_in_capture": False,
    }
    samples, kwargs = calls[0]
    assert len(samples) == 4800
    assert float(np.max(np.abs(samples))) <= 0.08
    assert kwargs == {
        "samplerate": 48000,
        "device": 1,
        "blocking": True,
    }


def test_acoustic_sensory_is_label_free_bounded_and_pitch_tolerant() -> None:
    low, low_quality = acoustic_sensory(_capture_wave("a", 90.0), 16000)
    high, high_quality = acoustic_sensory(_capture_wave("a", 180.0), 16000)

    assert low.shape == high.shape == (30,)
    assert np.linalg.norm(low) == pytest.approx(1.0)
    assert np.linalg.norm(high) == pytest.approx(1.0)
    assert low_quality["quality_passed"] is True
    assert high_quality["quality_passed"] is True
    assert low_quality["pitch_periodicity_at_least_0_30"] is True
    assert high_quality["pitch_periodicity_at_least_0_30"] is True
    assert np.linalg.norm(low - high) / 2.0 < 0.10


def test_nonperiodic_background_cannot_pass_as_a_vowel() -> None:
    rng = np.random.default_rng(4021)
    background = rng.normal(0.0, 0.03, (28800, 2)).astype(np.float32)

    _sensory, quality = acoustic_sensory(background, 16000)

    assert quality["analysis_rms"] >= 0.01
    assert quality["peak_at_least_0_03"] is True
    assert quality["pitch_periodicity_at_least_0_30"] is False
    assert quality["quality_passed"] is False


def test_waveform_repertoire_is_deterministic_and_uses_common_features() -> None:
    first = build_waveform_repertoire()
    second = build_waveform_repertoire()

    assert first.digest() == second.digest()
    assert first.digest() == "fb07df5ede273436524d22dcb3198bae79907e0b7cf7abd57a14c177381fb0be"
    assert len(first.prototypes) == 96
    assert [item.halton_index for item in first.prototypes] == [
        20011 + 43 * offset for offset in range(96)
    ]
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
    assert report["aggregate_criteria"]["development_freeze_verified"] is True
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
    freeze = json.loads((tmp_path / "frozen-vocal-associations.json").read_text())
    assert freeze["development_passed"] is True
    assert freeze["aggregate_criteria"]["confirmation_not_captured"] is True
    assert freeze["aggregate_criteria"]["confirmation_not_accessed"] is True


def test_manifest_hash_tampering_fails_loudly(tmp_path) -> None:
    _register_complete_session(tmp_path)
    path = tmp_path / "captures/a-development-1.wav"
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="capture hash mismatch"):
        run_project_owner_vocal_imitation(
            session_dir=tmp_path, allow_test_fixture=True
        )


def test_confirmation_take_cannot_be_reused_or_overwritten(tmp_path) -> None:
    _register_complete_session(tmp_path)
    samples = _capture_wave("u", 110.0)
    kwargs = dict(
        session_dir=tmp_path,
        target_key="u",
        role="confirmation",
        take=1,
        authorized=True,
        capture_fn=lambda: (samples, 16000, {"backend": "test"}),
    )
    with pytest.raises(ValueError, match="already registered"):
        capture_project_owner_take(**kwargs)


def test_confirmation_capture_is_refused_before_development_freeze(tmp_path) -> None:
    samples = _capture_wave("a", 110.0)
    with pytest.raises(ValueError, match="requires frozen-vocal-associations"):
        capture_project_owner_take(
            session_dir=tmp_path,
            target_key="a",
            role="confirmation",
            take=1,
            authorized=True,
            capture_fn=lambda: (samples, 16000, {"backend": "test"}),
        )


def test_development_freeze_requires_exactly_six_takes(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly six development takes"):
        freeze_project_owner_development(
            session_dir=tmp_path, allow_test_fixture=True
        )


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


def test_v1_acquisition_failure_is_retained_without_gate_tuning() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/2026-07-12-vocal-imitation-v1-acquisition-failure.json"
    )
    result = json.loads(path.read_text())

    assert result["status"] == "candidate_stopped"
    assert result["quality"]["quality_passed"] is False
    assert result["disposition"]["v1_stopped"] is True
    assert result["disposition"]["take_overwritten"] is False
    assert result["disposition"]["thresholds_changed"] is False
    assert result["disposition"]["v2_fresh_recording_required"] is True


def test_v2_prehuman_calibration_changes_only_capture_envelope() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/2026-07-12-vocal-imitation-v2-prehuman-calibration.json"
    )
    result = json.loads(path.read_text())

    assert result["fixture_passed"] is True
    assert result["human_recording_accessed"] is False
    assert result["v2_change"] == {
        "capture_duration_ms": 4000,
        "prior_capture_duration_ms": 1800,
        "preprocessing_changed": False,
        "quality_thresholds_changed": False,
        "learner_changed": False,
        "confirmation_gates_changed": False,
    }
    assert result["production_runner_accepts_this_fixture_as_evidence"] is False


def test_v2_acquisition_failure_requires_fresh_cued_v3_session() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/2026-07-12-vocal-imitation-v2-acquisition-failure.json"
    )
    result = json.loads(path.read_text())

    assert result["status"] == "candidate_stopped"
    assert len(result["passing_development_takes_before_stop"]) == 4
    assert result["quality"]["quality_passed"] is False
    assert result["disposition"]["passing_takes_reused_by_v3"] is False
    assert result["disposition"]["v3_fresh_recording_required"] is True
    assert result["disposition"]["thresholds_changed"] is False


def test_v3_start_cue_rehearsal_did_not_consume_a_protocol_take() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/2026-07-12-vocal-imitation-v3-start-cue-rehearsal.json"
    )
    result = json.loads(path.read_text())

    assert result["rehearsal_passed"] is True
    assert result["protocol_take_consumed"] is False
    assert result["audio_file_written"] is False
    assert result["audio_samples_discarded"] is True
    assert result["start_cue"]["output_device_name"] == "Mac mini Speakers"
    assert result["criteria"]["callback_integrity_clean"] is True
    assert "not a project-owner development take" in result["claim_boundary"]


def test_v3_development_failure_stops_before_confirmation() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/2026-07-12-vocal-imitation-v3-development-failure.json"
    )
    result = json.loads(path.read_text())

    assert result["status"] == "candidate_stopped"
    assert result["capture_count"] == 6
    assert result["confirmation_captured"] is False
    assert result["confirmation_accessed"] is False
    assert result["targets"]["a"]["development_gate_passed"] is False
    assert result["representational_feasibility"]["global_search_would_pass_a"] is False
    assert result["disposition"]["recordings_reused_by_v4"] is False


def test_v4_spent_design_audit_requires_fresh_recordings() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/2026-07-12-vocal-imitation-v4-spent-design-audit.json"
    )
    result = json.loads(path.read_text())

    assert result["status"] == "development_candidate_ready_fresh_recordings_required"
    assert result["repertoire"]["count"] == 96
    assert result["repertoire"]["halton_start"] == 20011
    assert result["repertoire"]["halton_stride"] == 43
    assert all(row["development_gates_passed"] for row in result["targets"].values())
    assert result["criteria"]["confirmation_not_accessed"] is True
    assert result["criteria"]["fresh_v4_recordings_required"] is True
    assert result["promotion_allowed"] is False


def test_v4_prehuman_fixture_is_not_human_evidence() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/2026-07-12-vocal-imitation-v4-prehuman-calibration.json"
    )
    result = json.loads(path.read_text())

    assert result["fixture_passed"] is True
    assert result["fixture"]["repertoire_sha256"] == (
        "fb07df5ede273436524d22dcb3198bae79907e0b7cf7abd57a14c177381fb0be"
    )
    assert len(result["fixture"]["development_freeze_sha256"]) == 64
    assert all(row["passed"] for row in result["targets"].values())
    assert result["human_recording_accessed"] is False
    assert result["real_capture_claim"] is False
    assert result["production_runner_accepts_this_fixture_as_evidence"] is False
    assert result["criteria"]["development_associations_frozen_before_confirmation"] is True
    assert result["criteria"]["final_runner_matched_frozen_associations"] is True
