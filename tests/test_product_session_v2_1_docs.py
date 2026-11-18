from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/BOUNDED_VOCAL_IMITATION_SESSION_PROTOCOL_V2_1.md"
PREREGISTRATION = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_1-preregistration.json"
)
REJECTION = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2-candidate-rejection.json"
)
IMPLEMENTATION = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_1-implementation.json"
)
OWNER_APPROVAL = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_1-owner-approval.json"
)
COHORT_REGISTRATION = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_1-cohort-registration-correction.json"
)
CORRECTED_SESSION_01_RESULT = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_1-corrected-session-01-result.json"
)
COHORT_STATUS_AFTER_SLOT_01 = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2_1-cohort-status-after-slot-01.json"
)


def test_v2_is_rejected_without_capture_or_cohort() -> None:
    record = json.loads(REJECTION.read_text())
    assert record["status"] == "rejected_after_authorized_cue_diagnostic"
    assert record["result"]["machine_gate_passed"] is False
    assert record["result"]["owner_heard_cue"] is False
    assert record["result"]["microphone_opened"] is False
    assert record["result"]["cohort_registered"] is False
    assert record["result"]["native_binary_rejected"] is True
    assert record["governance"]["v2_1_approval_inherited"] is False


def test_v2_source_digest_discrepancy_is_disclosed_not_rewritten() -> None:
    record = json.loads(REJECTION.read_text())
    note = record["evidence_integrity_note"]
    assert note["approved_implementation_record_is_immutable"] is True
    assert note["source_digest_matches_record"] is False
    assert note["rebuild_is_byte_identical_to_approved_binary"] is True


def test_v2_1_is_not_approved_executable_or_live_authorized() -> None:
    protocol = " ".join(PROTOCOL.read_text().split())
    prereg = json.loads(PREREGISTRATION.read_text())
    assert "NOT OWNER-APPROVED, NOT EXECUTABLE" in protocol
    assert prereg["status"] == "offline_revision_not_owner_approved_not_executable"
    assert prereg["owner_protocol_approval_recorded"] is False
    assert prereg["execution_authorized_by_this_artifact"] is False
    assert prereg["live_io_authorized_by_this_artifact"] is False
    assert prereg["new_cohort_registered"] is False


def test_v2_1_protocol_freezes_callback_before_stop_and_no_transfer() -> None:
    protocol = " ".join(PROTOCOL.read_text().split())
    for text in (
        "cora-audioqueue-player-v2",
        "returned-buffer callback **before**",
        "AudioQueueStop(queue, false)",
        "AudioQueueDispose(queue, false)",
        "Approval or cue authorization granted to V1 or V2 does not transfer",
        "No V1 or V2 slot or audio contributes",
    ):
        assert text in protocol


def test_v2_1_preregistration_preserves_behavior_scores_and_retention() -> None:
    prereg = json.loads(PREREGISTRATION.read_text())
    unchanged = prereg["unchanged_from_v2"]
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


def test_v2_1_implementation_hashes_and_authority_boundary() -> None:
    import hashlib

    record = json.loads(IMPLEMENTATION.read_text())
    for entry in record["source_hashes"].values():
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    assert record["native_binary"]["same_basename_reproducible"] is True
    assert record["native_binary"]["device_listing_opened_audio_queue"] is False
    assert record["offline_verification"]["full_suite"]["passed"] == 563
    assert record["offline_verification"]["full_suite"]["failed"] == 0
    assert record["governance"]["owner_protocol_approval_recorded"] is False
    assert record["governance"]["cue_diagnostic_authorized"] is False
    assert record["governance"]["new_v2_1_cohort_registered"] is False
    assert record["governance"]["microphone_capture_authorized"] is False


def test_v2_1_owner_approval_hash_pins_design_without_live_authority() -> None:
    import hashlib

    approval = json.loads(OWNER_APPROVAL.read_text())
    assert approval["owner_statement"] == (
        "Approve V2.1 protocol and unchanged retention decision"
    )
    assert approval["status"] == (
        "owner_approved_protocol_design_and_retention_not_live_io"
    )
    for key in (
        "approved_protocol",
        "approved_preregistration",
        "approved_implementation_record",
        "approved_operating_point",
        "approved_predecessor_rejection",
        "approved_read_only_preflight",
    ):
        entry = approval[key]
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == (
            entry["sha256"]
        )
    retention = approval["approved_retention"]
    storage = retention["private_storage_implementation"]
    assert hashlib.sha256((ROOT / storage["path"]).read_bytes()).hexdigest() == (
        storage["sha256"]
    )
    assert retention["days"] == 30
    execution = approval["execution_state"]
    assert execution["protocol_design_approved"] is True
    assert execution["retention_decision_approved"] is True
    assert execution["live_cue_validation_complete"] is False
    assert execution["new_v2_1_cohort_registered"] is False
    assert execution["execution_authorized"] is False
    assert execution["live_io_authorized"] is False
    assert execution["microphone_capture_authorized"] is False


def test_v2_1_cohort_registers_five_fresh_unstarted_slots() -> None:
    import re

    cohort = json.loads(COHORT_REGISTRATION.read_text())
    slots = cohort["ordered_session_slot_ids"]
    ledger = cohort["slot_ledger_at_registration"]
    assert cohort["owner_registration_instruction"] == "Register V2.1 cohort."
    assert cohort["status"] == "effective_corrected_cohort_registration"
    assert cohort["supersedes"]["status"] == "void_for_execution"
    assert cohort["voided_authorization"]["authority_transferred"] is False
    assert len(slots) == 5
    assert len(set(slots)) == 5
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9-]{7,79}", slot) for slot in slots)
    assert [row["slot_id"] for row in ledger] == slots
    assert [row["ordered_slot_index"] for row in ledger] == [1, 2, 3, 4, 5]
    assert all(row["state"] == "unstarted" for row in ledger)
    assert all(row["status"] is None for row in ledger)
    assert all(row["consumed"] is False for row in ledger)
    assert cohort["minimum_fully_passed_slots"] == 3
    assert cohort["all_five_slots_must_resolve_before_aggregation"] is True
    frozen = cohort["frozen_artifact_set"]
    source = ROOT / frozen["source_registration"]
    import hashlib

    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        frozen["source_registration_sha256"]
    )
    assert frozen["artifact_digests_reverified_before_correction"] is True
    assert frozen["artifact_bytes_changed"] is False
    conditions = cohort["registration_conditions"]
    assert conditions["passed_cue_diagnostic_preceded_registration"] is True
    assert conditions["session_live_io_performed_before_registration"] is False
    assert conditions["microphone_io_performed_before_registration"] is False
    assert conditions["all_slots_are_fresh"] is True
    assert conditions["identifier_matches_approved_storage_pattern"] is True
    authorization = cohort["authorization_state"]
    assert authorization["cohort_registered"] is True
    assert authorization["any_session_live_io_authorized"] is False
    assert authorization["microphone_capture_authorized"] is False
    assert authorization["fresh_authorization_required_after_correction"] is True
    assert authorization["next_authorizable_slot"] == slots[0]


def test_v2_1_slot_01_stops_before_held_on_missing_frozen_self_model() -> None:
    result = json.loads(CORRECTED_SESSION_01_RESULT.read_text())
    cohort = json.loads(COHORT_STATUS_AFTER_SLOT_01.read_text())
    assert result["status"] == "behavioral_failed"
    assert result["slot_consumed"] is True
    assert result["technical_valid"] is True
    assert result["failure_attributable_to_owner_vocal_behavior"] is False
    assert result["behavioral_failure_gates"] == [
        "frozen_self_model_exists_before_prediction_commitment"
    ]
    assert result["completed_microphone_captures"] == 6
    assert result["all_completed_capture_quality_gates_passed"] is True
    assert result["development_association_count"] == 3
    assert result["three_distinct_prototype_ids"] is True
    assert result["held_evaluation_opened"] is False
    assert result["response_playback_performed"] is False
    assert result["prediction_commitment_created"] is False
    assert result["waveform_scores_accessed"] is False
    assert result["retry_permitted"] is False
    assert result["product_session_passed"] is False
    assert cohort["started_slots"] == 1
    assert cohort["passed_sessions"] == 0
    assert cohort["maximum_remaining_possible_passes"] == 4
    assert cohort["product_level_pass_still_mathematically_possible"] is True
    assert cohort["next_slot_live_execution_recommended"] is False
    assert cohort["successor_protocol_revision_required"] is True
