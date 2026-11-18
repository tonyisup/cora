from __future__ import annotations

import json

import numpy as np
import pytest

from cora_infans.drained_playback import AudioToolboxOutput
from cora_infans.drained_playback_v2_1 import (
    PLAYER_VERSION,
    SELF_DEVICE_NAME,
    SELF_DEVICE_UID,
    WORLD_DEVICE_NAME,
    WORLD_DEVICE_UID,
)
from cora_infans.product_session_v2_2 import (
    ANALYSIS_SAMPLE_RATE,
    FIR_SAMPLES,
    PROTOCOL_VERSION,
    SELF_CALIBRATION_CAPTURE_MS,
    SELF_CALIBRATION_CHIRP_MS,
    SELF_CALIBRATION_PEAK,
    SELF_CALIBRATION_ROLE,
    SELF_CALIBRATION_TRIAL_ID,
    V2_2AppendOnlyAttemptLog,
    analyze_self_calibration,
    build_self_calibration_stimulus,
    build_v2_2_read_only_preflight,
    build_v2_2_session_authorization,
    run_v2_2_deterministic_dry_run,
    v2_2_behavioral_contract,
    v2_2_planned_capture_schedule,
)


def _synthetic_calibration() -> tuple[np.ndarray, np.ndarray, int]:
    chirp = build_self_calibration_stimulus()
    onset = 4_000
    capture = np.zeros(ANALYSIS_SAMPLE_RATE, dtype=np.float32)
    capture[onset : onset + len(chirp)] = chirp
    return chirp, capture, onset


def test_v2_2_self_calibration_stimulus_is_exact_and_bounded() -> None:
    chirp = build_self_calibration_stimulus()
    assert chirp.dtype == np.float32
    assert len(chirp) == SELF_CALIBRATION_CHIRP_MS * ANALYSIS_SAMPLE_RATE // 1_000
    assert float(np.max(np.abs(chirp))) == pytest.approx(SELF_CALIBRATION_PEAK)
    assert np.any(chirp)


def test_v2_2_calibration_freezes_model_before_development() -> None:
    chirp, capture, onset = _synthetic_calibration()
    result = analyze_self_calibration(
        chirp,
        capture,
        expected_onset_frame=onset,
        transport_integrity={"technical_valid": True},
    )
    assert result["calibration_passed"] is True
    model = result["self_model"]
    assert model["protocol_version"] == PROTOCOL_VERSION
    assert model["calibration_trial_id"] == SELF_CALIBRATION_TRIAL_ID
    assert model["frozen_before_development_capture"] is True
    assert model["owner_vocal_audio_used"] is False
    assert model["post_response_data_used"] is False
    assert len(model["impulse"]) == FIR_SAMPLES
    assert len(model["self_model_payload_sha256"]) == 64


def test_v2_2_calibration_fails_closed_on_transport_clipping_or_silence() -> None:
    chirp, capture, onset = _synthetic_calibration()
    with pytest.raises(ValueError, match="transport integrity"):
        analyze_self_calibration(
            chirp,
            capture,
            expected_onset_frame=onset,
            transport_integrity={"technical_valid": False},
        )
    clipped = capture.copy()
    clipped[onset] = 1.0
    with pytest.raises(ValueError, match="clipped"):
        analyze_self_calibration(
            chirp,
            clipped,
            expected_onset_frame=onset,
            transport_integrity={"technical_valid": True},
        )
    with pytest.raises(ValueError, match="not recovered"):
        analyze_self_calibration(
            chirp,
            np.zeros_like(capture),
            expected_onset_frame=onset,
            transport_integrity={"technical_valid": True},
        )


def test_v2_2_schedule_adds_one_calibration_before_unchanged_18_captures() -> None:
    schedule = v2_2_planned_capture_schedule()
    assert len(schedule) == 19
    assert schedule[0] == {
        "trial_id": SELF_CALIBRATION_TRIAL_ID,
        "phase": "calibration",
        "label": "machine_chirp",
        "role": SELF_CALIBRATION_ROLE,
    }
    assert [row["role"] for row in schedule[1:7]] == ["development_target"] * 6
    assert [row["role"] for row in schedule[7:]] == [
        role for _ in range(6) for role in ("held_target", "held_response")
    ]
    contract = v2_2_behavioral_contract()
    assert contract["self_calibration_captures"] == 1
    assert contract["owner_target_and_response_captures"] == 18
    assert contract["total_microphone_captures_per_session"] == 19
    assert contract["behavioral_or_score_threshold_changed_from_v2_1"] is False


def test_v2_2_ledger_blocks_development_without_frozen_self_model(tmp_path) -> None:
    ledger = V2_2AppendOnlyAttemptLog(
        tmp_path / "attempts.jsonl",
        session_id="bvips-v2-2-test-session-01",
    )
    opened = ledger.open_next(tmp_path)
    assert opened["role"] == SELF_CALIBRATION_ROLE
    completion = (
        tmp_path
        / "calibration"
        / SELF_CALIBRATION_TRIAL_ID
        / f"{SELF_CALIBRATION_ROLE}-completion.json"
    )
    completion.parent.mkdir(parents=True)
    completion.write_text(
        json.dumps(
            {
                "accepted_audio_samples": 16_000,
                "technical_valid": True,
                "gate_passed": True,
            }
        )
    )
    with pytest.raises(ValueError, match="self model freezes"):
        ledger.open_next(tmp_path)
    (tmp_path / "self-model.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "frozen_before_development_capture": True,
            }
        )
    )
    assert ledger.open_next(tmp_path)["role"] == "development_target"


def test_v2_2_ledger_rejects_underscore_identifier() -> None:
    with pytest.raises(ValueError, match="storage-safe"):
        V2_2AppendOnlyAttemptLog(
            "/private/tmp/unused-v2-2-ledger.jsonl",
            session_id="bvips_v2_2_invalid",
        )


def test_v2_2_authorization_is_storage_safe_and_one_slot_only() -> None:
    authorization = build_v2_2_session_authorization(
        session_id="bvips-v2-2-cohort-01-s01",
        cohort_id="bvips-v2-2-cohort-01",
        ordered_slot_index=1,
        authorization_text="Authorize V2.2 slot 1 live I/O.",
        authorization_date="2026-07-16",
    )
    assert authorization["machine_self_calibration_capture_authorized"] is True
    assert authorization["owner_target_recording_authorized"] is True
    assert authorization["authorization_applies_to_other_slots"] is False
    assert authorization["retention_days"] == 30
    with pytest.raises(ValueError, match="storage-safe"):
        build_v2_2_session_authorization(
            session_id="bvips_v2_2_cohort_01_s01",
            cohort_id="bvips-v2-2-cohort-01",
            ordered_slot_index=1,
            authorization_text="invalid",
            authorization_date="2026-07-16",
        )


def test_v2_2_preflight_requires_exact_calibration_contract() -> None:
    operating_point = {
        "input": {
            "device_selector": ":0",
            "sample_rate_hz": 16000,
            "host_channels": 2,
            "capture_backend": "sounddevice_portaudio_coreaudio_callback",
        },
        "self_source": {
            "coreaudio_device_uid": SELF_DEVICE_UID,
            "device": SELF_DEVICE_NAME,
            "system_volume_percent": 100,
        },
        "world_source": {
            "coreaudio_device_uid": WORLD_DEVICE_UID,
            "device": WORLD_DEVICE_NAME,
        },
        "playback": {
            "implementation": PLAYER_VERSION,
            "device_enumeration": "coreaudio_property_read_only_no_audioqueue",
            "device_binding": "coreaudio_uid_with_name_cross_check",
            "enqueue_before_start_required": True,
            "buffer_callback_before_stop_required": True,
            "non_immediate_stop_required": True,
            "queue_not_running_before_return_required": True,
            "non_immediate_dispose_required": True,
        },
        "self_calibration": {
            "capture_duration_ms": SELF_CALIBRATION_CAPTURE_MS,
            "chirp_duration_ms": SELF_CALIBRATION_CHIRP_MS,
            "chirp_peak_absolute": SELF_CALIBRATION_PEAK,
            "frozen_before_development_capture_required": True,
        },
    }
    devices = (
        AudioToolboxOutput(2, WORLD_DEVICE_NAME, WORLD_DEVICE_UID),
        AudioToolboxOutput(3, SELF_DEVICE_NAME, SELF_DEVICE_UID),
    )
    inspection = {
        "player_version": PLAYER_VERSION,
        "binary_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "audio_device_opened": False,
    }
    report = build_v2_2_read_only_preflight(
        operating_point,
        devices=devices,
        system_volume_percent=100,
        output_muted=False,
        player_inspection=inspection,
    )
    assert report["preflight_passed"] is True
    assert report["audio_frames_sent"] == 0
    assert report["microphone_opened"] is False
    operating_point["self_calibration"]["frozen_before_development_capture_required"] = False
    failed = build_v2_2_read_only_preflight(
        operating_point,
        devices=devices,
        system_volume_percent=100,
        output_muted=False,
        player_inspection=inspection,
    )
    assert failed["preflight_passed"] is False


def test_v2_2_end_to_end_dry_run_reaches_all_held_scores_without_io(tmp_path) -> None:
    report = run_v2_2_deterministic_dry_run(tmp_path)
    assert report == {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "device_accessed": False,
        "microphone_opened": False,
        "playback_performed": False,
        "planned_capture_count": 19,
        "opened_capture_count": 19,
        "self_model_frozen_before_development": True,
        "association_count": 3,
        "three_distinct_prototype_ids": True,
        "held_trial_count": 6,
        "all_held_selection_prediction_and_score_gates_passed": True,
    }
