from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cora_infans.product_session_live import (
    AppendOnlyAttemptLog,
    PROTOCOL_VERSION,
    aggregate_cohort_status,
    build_cohort_registration,
    build_prediction_commitment,
    build_session_attestation,
    build_self_model,
    build_session_authorization,
    build_session_manifest,
    classify_completed_session,
    classify_transport_integrity,
    commit_label_blind_selection,
    evaluate_committed_selection,
    freeze_development_associations,
    predict_self_capture,
    planned_capture_schedule,
    probe_product_session_preflight,
    retention_deadline,
    score_held_trial,
    validate_session_authorization,
)
from cora_infans.factorized_tract_plant import synthesize_factorized
from cora_infans.io.motor_plan import write_wav
from cora_infans.speech_learning_demo import DEMO_F0_HZ
from cora_infans.vocal_imitation import build_waveform_repertoire


def _digests() -> dict[str, str]:
    return {
        "protocol": "1" * 64,
        "preregistration": "2" * 64,
        "threshold_calibration": "3" * 64,
        "representation_implementation": "4" * 64,
        "live_runner_implementation": "5" * 64,
        "private_storage_implementation": "6" * 64,
        "operating_point": "7" * 64,
    }


def _cohort() -> dict[str, object]:
    cohort_id = "bvips-v1-20260716-cohort-01"
    return build_cohort_registration(
        cohort_id=cohort_id,
        ordered_session_ids=[f"{cohort_id}-s0{index}" for index in range(1, 6)],
        frozen_digests=_digests(),
    )


def test_cohort_registration_freezes_exactly_five_ordered_slots() -> None:
    cohort = _cohort()
    assert len(cohort["ordered_session_slot_ids"]) == 5
    assert cohort["minimum_fully_passed_slots"] == 3
    assert cohort["passes_must_be_consecutive"] is False
    assert cohort["registered_before_live_io"] is True

    with pytest.raises(ValueError, match="five distinct"):
        build_cohort_registration(
            cohort_id="cohort",
            ordered_session_ids=["cohort-s01"] * 5,
            frozen_digests=_digests(),
        )


def test_authorization_is_bound_to_one_registered_session() -> None:
    cohort = _cohort()
    session = cohort["ordered_session_slot_ids"][0]
    authorization = build_session_authorization(
        session_id=session,
        cohort_registration=cohort,
        storage_alias=f"cora-private:{session}",
        authorization_text="I authorize execution and live I/O. Let's get it going!",
        authorization_date="2026-07-16",
    )
    validate_session_authorization(authorization, cohort, session_id=session)
    assert authorization["protocol_version"] == PROTOCOL_VERSION
    assert authorization["authorization_applies_to_other_slots"] is False

    with pytest.raises(ValueError, match="different session"):
        validate_session_authorization(
            authorization,
            cohort,
            session_id=cohort["ordered_session_slot_ids"][1],
        )


def test_append_only_log_consumes_each_planned_trial_role_once(tmp_path) -> None:
    log = AppendOnlyAttemptLog(tmp_path / "attempts.jsonl", session_id="session-0001")
    first = log.append_open(trial_id="dev-01-a", role="development_target")
    second = log.append_open(trial_id="held-01-a", role="held_target")
    assert first["attempt_number"] == 1
    assert second["attempt_number"] == 2
    assert [row["state"] for row in log.rows()] == ["opened_before_io"] * 2
    with pytest.raises(ValueError, match="already consumed"):
        log.append_open(trial_id="held-01-a", role="held_target")


def test_one_zero_sample_technical_abort_allows_exactly_one_retry(tmp_path) -> None:
    log = AppendOnlyAttemptLog(tmp_path / "attempts.jsonl", session_id="session-0001")
    opened = log.open_next(tmp_path)
    aborted = log.append_technical_abort(
        trial_id=opened["trial_id"],
        role=opened["role"],
        accepted_audio_samples=0,
        partial_wav_created=False,
        reason="callback stream failed before first sample",
    )
    assert aborted["state"] == "technical_abort"
    retry = log.open_next(tmp_path)
    assert retry["trial_id"] == opened["trial_id"]
    assert retry["attempt_number"] == 2

    with pytest.raises(ValueError, match="unavailable"):
        log.append_technical_abort(
            trial_id=retry["trial_id"],
            role=retry["role"],
            accepted_audio_samples=0,
            partial_wav_created=False,
            reason="second startup failure",
        )
    with pytest.raises(ValueError, match="zero samples"):
        AppendOnlyAttemptLog(
            tmp_path / "other.jsonl", session_id="session-0002"
        ).append_technical_abort(
            trial_id="development-01-a",
            role="development_target",
            accepted_audio_samples=1,
            partial_wav_created=False,
            reason="not retryable",
        )


def test_fixed_state_machine_enforces_schedule_freeze_and_commitments(tmp_path) -> None:
    log = AppendOnlyAttemptLog(tmp_path / "attempts.jsonl", session_id="session-0001")
    plan = planned_capture_schedule()
    assert len(plan) == 18
    assert [row["role"] for row in plan[:6]] == ["development_target"] * 6

    def complete(planned) -> None:
        trial = tmp_path / planned["phase"] / planned["trial_id"]
        filename = (
            "response_captured.wav"
            if planned["role"] == "held_response"
            else "target_captured.wav"
        )
        (trial / filename).write_bytes(b"wav")
        (trial / f"{planned['role']}-integrity.json").write_text(
            json.dumps({"technical_valid": True})
        )
        (trial / f"{planned['role']}-quality.json").write_text(
            json.dumps({"quality_passed": True})
        )

    log.open_next(tmp_path)
    with pytest.raises(ValueError, match="no completed capture"):
        log.open_next(tmp_path)
    complete(plan[0])
    for planned in plan[1:6]:
        log.open_next(tmp_path)
        complete(planned)

    with pytest.raises(ValueError, match="frozen-vocal-associations"):
        log.open_next(tmp_path)
    frozen = {"frozen_before_held_capture": True}
    (tmp_path / "frozen-vocal-associations.json").write_text(json.dumps(frozen))
    from cora_infans.product_session_live import sha256_json, sha256_path

    (tmp_path / "evaluation-key.json").write_text(
        json.dumps({"frozen_associations_sha256": sha256_json(frozen)})
    )
    log.open_next(tmp_path)
    complete(plan[6])
    with pytest.raises(ValueError, match="selection-commitment"):
        log.open_next(tmp_path)

    trial = tmp_path / "held" / plan[7]["trial_id"]
    (trial / "selection-commitment.json").write_text(
        json.dumps({"selection_and_contrastive_passed": True})
    )
    (trial / "selection-verdict.json").write_text(
        json.dumps({"correct_association": True})
    )
    (trial / "agent_emitted.wav").write_bytes(b"emitted")
    (trial / "self_prediction.wav").write_bytes(b"prediction")
    (trial / "prediction-commitment.json").write_text(
        json.dumps(
            {
                "committed_before_response_capture": True,
                "agent_emitted_sha256": sha256_path(trial / "agent_emitted.wav"),
                "self_prediction_sha256": sha256_path(trial / "self_prediction.wav"),
            }
        )
    )
    opened = log.open_next(tmp_path)
    assert opened["role"] == "held_response"


def test_transport_integrity_commits_machine_faults_before_scores() -> None:
    clean = {
        "backend": "sounddevice_portaudio_coreaudio_callback",
        "sample_rate_hz": 16000,
        "channel_count": 2,
        "sample_count": 80000,
        "input_overflow_count": 0,
        "callback_status_count": 0,
        "callback_timestamp_discontinuity_count": 0,
    }
    assert classify_transport_integrity(clean)["technical_valid"] is True
    dirty = {**clean, "input_overflow_count": 1}
    result = classify_transport_integrity(dirty)
    assert result["committed_before_score_access"] is True
    assert result["technical_valid"] is False
    assert result["session_status_if_invalid"] == "technical_invalid"
    assert result["failure_reasons"] == ["input_overflow_count"]


def test_product_preflight_requires_exact_route_and_measured_volume() -> None:
    operating_point = {
        "input": {"device_selector": ":0", "sample_rate_hz": 16000, "host_channels": 2},
        "self_source": {
            "output_device_index": 3,
            "device": "Mac mini Speakers",
            "placement": "centered_directly_below_xvf3800",
            "vertical_distance_cm": 4,
            "system_volume_percent": 100,
        },
    }
    probe = {
        "platform_system": "Darwin",
        "host_name": "Mac mini",
        "ffmpeg_has_audiotoolbox": True,
        "xvf3800_input": {"name": "reSpeaker XVF3800 4-Mic Array"},
        "xvf3800_16khz_two_channel_supported": True,
        "coreaudio_output_device_names": ["Mac mini Speakers"],
    }
    report = probe_product_session_preflight(
        operating_point, device_probe=lambda: probe, volume_probe=lambda: 100
    )
    assert report["preflight_passed"] is True
    assert report["audio_stream_opened"] is False

    failed = probe_product_session_preflight(
        operating_point, device_probe=lambda: probe, volume_probe=lambda: 99
    )
    assert failed["preflight_passed"] is False
    assert failed["checks"]["system_volume_is_100_percent"] is False


def test_verdict_uses_scheduled_label_only_after_commitment() -> None:
    commitment = {"selected_opaque_id": "assoc-one"}
    key = {"opaque_association_targets": {"assoc-one": {"scheduled_target": "a"}}}
    verdict = evaluate_committed_selection(commitment, key, scheduled_label="a")
    assert verdict == {
        "scheduled_label": "a",
        "committed_opaque_id": "assoc-one",
        "correct_association": True,
        "verdict_opened_after_commitment": True,
    }


def test_development_freeze_and_held_selection_use_opaque_ids_only() -> None:
    repertoire = build_waveform_repertoire()
    waveforms = {
        label: np.tile(
            synthesize_factorized(repertoire.prototypes[index].motor_array(), DEMO_F0_HZ),
            13,
        )
        for label, index in zip(("a", "i", "u"), (0, 30, 60))
    }
    captures = {
        label: [
            (f"development-{label}-1.wav", waveform, 16000),
            (f"development-{label}-2.wav", waveform, 16000),
        ]
        for label, waveform in waveforms.items()
    }
    frozen, key = freeze_development_associations(captures)
    assert frozen["three_distinct_prototype_ids"] is True
    assert set(frozen["associations"]).isdisjoint({"a", "i", "u"})

    commitment = commit_label_blind_selection(
        target_waveform=waveforms["a"], sample_rate=16000, frozen=frozen
    )
    assert commitment["selection_input"]["scheduled_label_provided"] is False
    assert commitment["selection_and_contrastive_passed"] is True
    verdict = evaluate_committed_selection(commitment, key, scheduled_label="a")
    assert verdict["correct_association"] is True


def test_self_model_and_prediction_commitment_precede_response(tmp_path) -> None:
    rng = np.random.default_rng(7)
    chirp = rng.normal(0.0, 0.1, 1024).astype(np.float32)
    true_impulse = np.zeros(128, dtype=np.float32)
    true_impulse[:3] = (0.7, 0.2, 0.1)
    observed = np.convolve(chirp, true_impulse)
    model = build_self_model(chirp, observed)
    emitted = rng.normal(0.0, 0.02, 1600).astype(np.float32)
    prediction = predict_self_capture(emitted, model)
    assert prediction.shape == emitted.shape

    emitted_path = tmp_path / "agent_emitted.wav"
    prediction_path = tmp_path / "self_prediction.wav"
    write_wav(str(emitted_path), emitted)
    write_wav(str(prediction_path), prediction)
    commitment = build_prediction_commitment(
        agent_emitted_path=emitted_path,
        self_prediction_path=prediction_path,
        self_model=model,
    )
    assert commitment["committed_before_response_capture"] is True
    assert commitment["response_capture_exists_at_commitment"] is False
    assert len(commitment["agent_emitted_sha256"]) == 64


def test_held_score_requires_every_component_in_both_families() -> None:
    sample_rate = 16000
    samples = np.zeros(sample_rate * 5, dtype=np.float64)
    time = np.arange(sample_rate * 2) / sample_rate
    samples[sample_rate : sample_rate * 3] = 0.25 * np.sin(2 * np.pi * 180 * time)
    report = score_held_trial(target=samples, response=samples, self_prediction=samples)
    assert report["all_component_gates_passed"] is True
    assert all(
        report["component_gates"][family][component]
        for family in ("imitation_fidelity", "self_model_fidelity")
        for component in (
            "active_duration_difference_ms",
            "rms_envelope_distance",
            "pitch_contour_distance_hz",
            "cepstral_distance",
        )
    )


def test_cohort_aggregation_does_not_require_consecutive_passes() -> None:
    result = aggregate_cohort_status(
        ["passed", "technical_invalid", "passed", "behavioral_failed", "passed"]
    )
    assert result["resolved"] is True
    assert result["product_level_passed"] is True
    assert result["behavioral_repeatability"] == pytest.approx(3 / 4)
    assert result["operational_reliability"] == pytest.approx(4 / 5)

    unresolved = aggregate_cohort_status(["passed", None, None, None, None])
    assert unresolved["product_level_passed"] is False


def test_retention_deadline_is_thirty_calendar_days() -> None:
    from datetime import date

    assert retention_deadline(date(2026, 7, 16)) == "2026-08-15"


def test_manifest_attestation_and_status_are_fail_closed(tmp_path) -> None:
    (tmp_path / "report.json").write_text("{}\n")
    (tmp_path / "attempts.jsonl").write_text("{}\n")
    manifest = build_session_manifest(
        tmp_path, required_relative_paths=["report.json", "attempts.jsonl"]
    )
    manifest_path = tmp_path / "session-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    attestation = build_session_attestation(
        manifest_path, anchor_type="git_commit", anchor_value="a" * 40
    )
    clean = {"technical_valid": True, "failure_reasons": []}
    passed = classify_completed_session(
        transport_integrity_records=[clean],
        behavioral_gates={"quality": True, "scores": True},
        manifest_attestation=attestation,
    )
    assert passed["status"] == "passed"

    technical = classify_completed_session(
        transport_integrity_records=[
            {"technical_valid": False, "failure_reasons": ["callback_status_count"]}
        ],
        behavioral_gates={"quality": True},
        manifest_attestation=attestation,
    )
    assert technical["status"] == "technical_invalid"
    assert technical["behavioral_gates_consulted"] is False

    unanchored = classify_completed_session(
        transport_integrity_records=[clean],
        behavioral_gates={"quality": True},
        manifest_attestation=None,
    )
    assert unanchored["status"] == "behavioral_failed"
