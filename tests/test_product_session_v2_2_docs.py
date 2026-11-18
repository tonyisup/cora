from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/BOUNDED_VOCAL_IMITATION_SESSION_PROTOCOL_V2_2.md"
PREREGISTRATION = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_2-preregistration.json"
)
OPERATING_POINT = ROOT / "config/product-session-v2_2-operating-point.json"
V2_1_CLOSURE = (
    ROOT / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_1-closure.json"
)
DRY_RUN = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_2-deterministic-dry-run.json"
)
IMPLEMENTATION = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_2-implementation.json"
)
OWNER_APPROVAL = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_2-owner-approval.json"
)
COHORT_REGISTRATION = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_2-cohort-registration.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_1_closure_preserves_one_failure_and_four_unstarted_slots() -> None:
    closure = json.loads(V2_1_CLOSURE.read_text())
    assert closure["status"] == (
        "closed_for_execution_after_known_precondition_failure"
    )
    assert closure["started_slots"] == 1
    assert closure["consumed_slots"] == 1
    assert closure["passed_slots"] == 0
    assert closure["aggregate_product_verdict_valid"] is False
    assert closure["product_claim_allowed"] is False
    assert closure["prior_statuses_rewritten"] is False
    assert closure["v2_1_audio_reusable_as_v2_2_confirmation"] is False
    assert closure["v2_2_must_use_fresh_cohort_and_audio"] is True
    assert [row["status"] for row in closure["slot_disposition"]] == [
        "behavioral_failed",
        "closed_unstarted",
        "closed_unstarted",
        "closed_unstarted",
        "closed_unstarted",
    ]
    assert closure["effective_registration"]["sha256"] == _sha256(
        ROOT / closure["effective_registration"]["path"]
    )
    assert closure["slot_01_result"]["sha256"] == _sha256(
        ROOT / closure["slot_01_result"]["path"]
    )


def test_v2_2_is_not_approved_executable_live_authorized_or_registered() -> None:
    protocol = " ".join(PROTOCOL.read_text().split())
    prereg = json.loads(PREREGISTRATION.read_text())
    assert "NOT OWNER-APPROVED, NOT EXECUTABLE" in protocol
    assert prereg["status"] == (
        "offline_implemented_candidate_not_owner_approved_not_executable"
    )
    assert prereg["owner_protocol_approval_recorded"] is False
    assert prereg["execution_authorized_by_this_artifact"] is False
    assert prereg["live_io_authorized_by_this_artifact"] is False
    assert prereg["new_cohort_registered"] is False


def test_v2_2_protocol_freezes_self_model_before_owner_targets() -> None:
    protocol = " ".join(PROTOCOL.read_text().split())
    for text in (
        "calibration-01-self-chirp",
        "exact 120 ms linear chirp from 250 to 3,500 Hz at peak 0.08",
        "finite 128-sample causal FIR",
        "written and hashed before development capture 1",
        "19 microphone captures",
        "distinct role-specific completion files",
        "all 19 state transitions",
        "No V2.1 audio, association, self model, slot, result, or authorization",
    ):
        assert text in protocol


def test_v2_2_operating_point_matches_calibration_and_storage_safe_schedule() -> None:
    payload = json.loads(OPERATING_POINT.read_text())["operating_point"]
    calibration = payload["self_calibration"]
    schedule = payload["capture_schedule"]
    assert calibration == {
        "trial_id": "calibration-01-self-chirp",
        "capture_role": "self_calibration",
        "capture_duration_ms": 1000,
        "pre_roll_ms": 250,
        "chirp_sample_rate_hz": 16000,
        "chirp_duration_ms": 120,
        "chirp_f0_hz": 250.0,
        "chirp_f1_hz": 3500.0,
        "chirp_peak_absolute": 0.08,
        "fir_samples": 128,
        "minimum_peak_correlation": 0.10,
        "causal_search_window_ms": 500.0,
        "frozen_before_development_capture_required": True,
        "owner_vocal_audio_used": False,
    }
    assert schedule["total_microphone_captures"] == 19
    assert schedule["owner_development_target_captures"] == 6
    assert schedule["owner_held_target_captures"] == 6
    assert schedule["self_response_captures"] == 6


def test_v2_2_preregistration_preserves_behavior_scores_and_retention() -> None:
    prereg = json.loads(PREREGISTRATION.read_text())
    unchanged = prereg["unchanged_from_v2_1"]
    assert unchanged["raw_audio_retention_days"] == 30
    assert unchanged["repeatability_slots"] == 5
    assert unchanged["minimum_passing_slots"] == 3
    assert unchanged["behavioral_or_score_threshold_changed"] is False
    assert prereg["score_thresholds"] == {
        "active_duration_difference_ms_max": 150.0,
        "rms_envelope_distance_max": 0.05,
        "pitch_contour_distance_hz_max": 30.0,
        "cepstral_distance_max": 0.5,
    }
    failures = prereg["frozen_failures"]
    assert failures["development_can_open_without_frozen_self_model"] is False
    assert failures["self_model_can_use_owner_vocal_or_response_audio"] is False
    assert failures["target_and_response_completion_can_overwrite_each_other"] is False
    assert failures["underscore_session_identifier_allowed"] is False


def test_v2_2_cue_carryforward_is_evidence_not_authority() -> None:
    protocol = " ".join(PROTOCOL.read_text().split())
    assert "may be carried forward only as route evidence" in protocol
    assert "Carry-forward authorizes no replay" in protocol


def test_v2_2_dry_run_reaches_all_19_role_specific_completions() -> None:
    record = json.loads(DRY_RUN.read_text())
    assert record["status"] == "complete_offline_state_machine_dry_run_passed"
    assert record["implementation"]["sha256"] == _sha256(
        ROOT / record["implementation"]["path"]
    )
    assert record["device_accessed"] is False
    assert record["microphone_opened"] is False
    assert record["playback_performed"] is False
    assert record["planned_capture_count"] == 19
    assert record["opened_capture_count"] == 19
    assert record["completion_artifact_count"] == 19
    assert record["self_calibration_completion_count"] == 1
    assert record["development_target_completion_count"] == 6
    assert record["held_target_completion_count"] == 6
    assert record["held_response_completion_count"] == 6
    assert record["selection_commitment_count"] == 6
    assert record["prediction_commitment_count"] == 6
    assert record["held_score_report_count"] == 6
    assert record["self_model_frozen_before_development"] is True
    assert record["three_distinct_prototype_ids"] is True
    assert record["all_held_selection_prediction_and_score_gates_passed"] is True
    assert record["role_specific_completion_artifacts_distinct"] is True


def test_v2_2_implementation_hashes_complete_offline_candidate() -> None:
    record = json.loads(IMPLEMENTATION.read_text())
    for entry in record["source_hashes"].values():
        assert entry["sha256"] == _sha256(ROOT / entry["path"])
    assert record["native_binary"]["same_basename_reproducible"] is True
    assert record["native_binary"]["binary_changed_from_v2_1"] is False
    dry = record["deterministic_dry_run"]
    assert dry["device_accessed"] is False
    assert dry["microphone_opened"] is False
    assert dry["playback_performed"] is False
    assert dry["planned_capture_count"] == 19
    assert dry["opened_capture_count"] == 19
    assert dry["role_specific_completion_count"] == 19
    assert dry["prediction_commitment_count"] == 6
    assert dry["held_score_report_count"] == 6
    assert dry["all_held_selection_prediction_and_score_gates_passed"] is True
    assert record["offline_verification"]["focused_tests"]["passed"] == 19
    assert record["offline_verification"]["full_suite"]["passed"] == 583
    governance = record["governance"]
    assert governance["owner_protocol_approval_recorded"] is False
    assert governance["cue_replay_authorized"] is False
    assert governance["self_calibration_live_io_authorized"] is False
    assert governance["new_v2_2_cohort_registered"] is False


def test_v2_2_owner_approval_pins_candidate_and_carries_route_evidence_only() -> None:
    approval = json.loads(OWNER_APPROVAL.read_text())
    assert approval["owner_statement"] == (
        "Approve V2.2 protocol, unchanged retention, and carry forward the "
        "unchanged V2.1 cue-route validation evidence."
    )
    assert approval["status"] == (
        "owner_approved_protocol_design_retention_and_cue_route_evidence_not_live_io"
    )
    for key in (
        "approved_protocol",
        "approved_preregistration",
        "approved_implementation_record",
        "approved_operating_point",
        "approved_predecessor_closure",
        "approved_deterministic_dry_run",
    ):
        entry = approval[key]
        assert entry["sha256"] == _sha256(ROOT / entry["path"])
    native = approval["approved_native_player"]
    for key in ("source", "wrapper"):
        entry = native[key]
        assert entry["sha256"] == _sha256(ROOT / entry["path"])
    storage = approval["approved_retention"]["private_storage_implementation"]
    assert storage["sha256"] == _sha256(ROOT / storage["path"])
    carryforward = approval["cue_route_validation_carryforward"]
    assert carryforward["approved"] is True
    for key in ("source_owner_confirmation", "source_playback"):
        entry = carryforward[key]
        assert entry["sha256"] == _sha256(ROOT / entry["path"])
    assert carryforward["fresh_read_only_preflight_still_required"] is True
    assert carryforward["new_cue_required_if_invalidated"] is True
    assert carryforward["cue_replay_authorized"] is False
    assert carryforward["microphone_capture_authorized"] is False
    assert carryforward["session_execution_authorized"] is False
    state = approval["execution_state"]
    assert state["protocol_design_approved"] is True
    assert state["retention_decision_approved"] is True
    assert state["cue_route_evidence_carried_forward"] is True
    assert state["new_v2_2_cohort_registered"] is False
    assert state["execution_authorized"] is False
    assert state["live_io_authorized"] is False
    assert state["calibration_playback_authorized"] is False
    assert state["microphone_capture_authorized"] is False


def test_v2_2_cohort_registration_freezes_five_fresh_storage_safe_slots() -> None:
    registration = json.loads(COHORT_REGISTRATION.read_text())
    assert registration["status"] == "registered_fresh_v2_2_cohort"
    assert registration["owner_instruction"] == "Register V2.2 cohort."
    assert registration["cohort_id"] == "bvips-v2-2-20260716-cohort-01"
    expected_slots = [
        f"bvips-v2-2-20260716-cohort-01-s{index:02d}" for index in range(1, 6)
    ]
    assert registration["ordered_session_slot_ids"] == expected_slots
    ledger = registration["slot_ledger_at_registration"]
    assert [entry["slot_id"] for entry in ledger] == expected_slots
    assert [entry["ordered_slot_index"] for entry in ledger] == list(range(1, 6))
    assert all(entry["state"] == "unstarted" for entry in ledger)
    assert all(entry["status"] is None for entry in ledger)
    assert all(entry["consumed"] is False for entry in ledger)
    identifier_pattern = re.compile(r"[a-z0-9][a-z0-9-]{7,79}")
    assert all(
        identifier_pattern.fullmatch(identifier)
        for identifier in [registration["cohort_id"], *expected_slots]
    )
    for entry in registration["frozen_artifacts"].values():
        if "path" in entry:
            assert entry["sha256"] == _sha256(ROOT / entry["path"])
    schedule = registration["capture_schedule"]
    assert schedule["self_calibration_captures"] == 1
    assert schedule["owner_development_target_captures"] == 6
    assert schedule["owner_held_target_captures"] == 6
    assert schedule["self_response_captures"] == 6
    assert schedule["total_microphone_captures_per_session"] == 19
    conditions = registration["registration_conditions"]
    assert conditions["all_identifiers_match_approved_storage_pattern"] is True
    assert conditions["all_slots_are_fresh"] is True
    assert conditions["v1_v2_or_v2_1_slots_reused"] is False
    assert conditions["v1_v2_or_v2_1_audio_reused"] is False
    state = registration["authorization_state"]
    assert state["cohort_registered"] is True
    assert state["next_authorizable_slot"] == expected_slots[0]
    assert state["cue_replay_authorized"] is False
    assert state["calibration_playback_authorized"] is False
    assert state["any_session_live_io_authorized"] is False
    assert state["microphone_capture_authorized"] is False
