"""Documentation contract for the proposed real product-session protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "BOUNDED_VOCAL_IMITATION_SESSION_PROTOCOL_V1.md"
PREREGISTRATION = (
    ROOT
    / "docs"
    / "evidence"
    / "2026-07-15-bounded-vocal-imitation-session-v1-preregistration.json"
)
SCORER_V2_PREREGISTRATION = (
    ROOT
    / "docs"
    / "evidence"
    / "2026-07-15-bounded-vocal-imitation-offline-replay-scorer-v2-preregistration.json"
)
OWNER_APPROVAL = (
    ROOT
    / "docs"
    / "evidence"
    / "2026-07-16-bounded-vocal-imitation-session-v1-owner-approval.json"
)
THRESHOLD_CALIBRATION = (
    ROOT
    / "docs"
    / "evidence"
    / "2026-07-16-bounded-vocal-imitation-v1-threshold-calibration.json"
)
COHORT_REGISTRATION = (
    ROOT
    / "docs"
    / "evidence"
    / "2026-07-16-bounded-vocal-imitation-v1-cohort-registration.json"
)
SESSION_01_AUTHORIZATION = (
    ROOT
    / "docs"
    / "evidence"
    / "2026-07-16-bounded-vocal-imitation-v1-session-01-authorization.json"
)
SESSION_01_RESULT = (
    ROOT
    / "docs"
    / "evidence"
    / "2026-07-16-bounded-vocal-imitation-v1-session-01-result.json"
)
COHORT_STATUS_AFTER_SLOT_01 = (
    ROOT
    / "docs"
    / "evidence"
    / "2026-07-16-bounded-vocal-imitation-v1-cohort-status-after-slot-01.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_product_session_v1_protocol_is_draft_only_and_freezes_proposed_values() -> None:
    protocol = PROTOCOL.read_text()
    preregistration = json.loads(PREREGISTRATION.read_text())

    assert "Status: OWNER-APPROVED DESIGN — NOT EXECUTABLE" in protocol
    assert preregistration["status"] == "owner_approved_not_executable"
    assert preregistration["owner_protocol_approval_recorded"] is True
    assert preregistration["owner_protocol_approval_date"] == "2026-07-16"
    assert preregistration["owner_retention_decision_approved"] is True
    assert preregistration["runtime_implementation_required_before_execution"] is True
    assert preregistration["threshold_calibration_required_before_execution"] is True
    assert preregistration["five_slot_cohort_registration_required_before_execution"] is True
    assert preregistration["fresh_per_session_live_io_authorization_required"] is True
    assert preregistration["execution_authorized_by_this_artifact"] is False
    assert preregistration["live_io_authorized_by_this_artifact"] is False

    design = preregistration["proposed_design"]
    assert design["targets"] == ["/a/", "/i/", "/u/"]
    assert design["development_takes_per_target"] == 2
    assert design["held_takes_per_target"] == 2
    assert design["association_freeze_after_development_takes"] == 6
    assert design["held_trials_after_freeze"] == 6
    assert design["total_microphone_capture_count"] == 18
    assert design["repeatability_cohort_session_slots"] == 5
    assert design["minimum_passing_sessions_for_product_claim"] == 3
    assert design["passing_sessions_must_be_consecutive"] is False
    assert design["later_nonpass_erases_prior_pass"] is False
    assert design["raw_audio_retention_days"] == 30
    assert "100% system volume" in design["response_route"]

    thresholds = preregistration["proposed_score_thresholds"]
    assert thresholds == {
        "active_duration_difference_ms_max": 150.0,
        "rms_envelope_distance_max": 0.05,
        "pitch_contour_distance_hz_max": 30.0,
        "cepstral_distance_max": 0.5,
    }
    assert preregistration["scoring_version"] == "offline-replay-v2"
    assert preregistration["all_component_thresholds_required_per_held_take"] is True
    assert preregistration["threshold_basis"] == (
        "provisional_until_separate_hashed_synthetic_calibration_not_perceptual_validation"
    )


def test_product_session_v1_freezes_selection_integrity_and_retention_gates() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())

    selection = preregistration["label_blind_selection"]
    assert selection["scheduled_label_available_before_selection_commitment"] is False
    assert selection["distinct_selected_prototype_ids_required"] is True
    assert selection["nearest_centroid_distance_max"] == 0.3
    assert selection["runner_up_margin_min"] == 0.05

    contrastive = preregistration["contrastive_gates"]
    assert contrastive["selected_prototype_improvement_over_rest_min"] == 0.15
    assert contrastive["selected_prototype_margin_over_each_off_target_min"] == 0.05

    retention = preregistration["retention_and_integrity"]
    assert retention["cryptographic_erasure_required"] is True
    assert retention["session_manifest_hash_must_be_externally_anchored"] is True
    assert retention["replayability_claim_after_audio_deletion"] is False
    assert retention["integrity_attestation_after_audio_deletion"] is True

    boundaries = preregistration["frozen_boundaries"]
    assert boundaries["self_prediction_creation_after_response_capture_allowed"] is False
    assert boundaries["unanchored_manifest_can_complete_real_session"] is False

    cohort = preregistration["repeatability_cohort"]
    assert cohort["ordered_session_slot_ids_required_before_first_live_io"] == 5
    assert cohort["minimum_fully_passed_slots"] == 3
    assert cohort["passes_must_be_consecutive"] is False
    assert cohort["blocked_pre_io_consumes_slot"] is False
    assert cohort["technical_invalid_consumes_started_slot"] is True
    assert cohort[
        "technical_invalid_status_committed_from_machine_telemetry_before_score_access"
    ] is True
    assert cohort["sixth_or_replacement_session_may_contribute"] is False

    statuses = preregistration["session_status_taxonomy"]
    assert set(statuses) == {
        "blocked_pre_io",
        "technical_invalid",
        "behavioral_failed",
        "passed",
    }


def test_offline_replay_v2_preregistration_freezes_active_interval_semantics() -> None:
    preregistration = json.loads(SCORER_V2_PREREGISTRATION.read_text())

    assert preregistration["scope"]["scoring_version"] == "offline-replay-v2"
    assert preregistration["historical_v1_record_mutated"] is False
    assert preregistration["active_interval"][
        "independent_capture_window_offsets_removed_before_comparison"
    ] is True
    assert preregistration["component_metrics"] == [
        "active_duration_difference_ms",
        "rms_envelope_distance",
        "pitch_contour_distance_hz",
        "cepstral_distance",
    ]


def test_owner_approval_hash_pins_design_without_authorizing_live_io() -> None:
    approval = json.loads(OWNER_APPROVAL.read_text())

    assert approval["status"] == "owner_approved_protocol_design_not_live_io"
    assert approval["approved_protocol"]["sha256"] == _sha256(PROTOCOL)
    assert approval["approved_preregistration"]["sha256"] == _sha256(
        PREREGISTRATION
    )
    execution = approval["execution_state"]
    assert execution["protocol_design_approved"] is True
    assert execution["retention_decision_approved"] is True
    assert execution["threshold_calibration_complete"] is False
    assert execution["runtime_implementation_complete"] is False
    assert execution["five_slot_cohort_registered"] is False
    assert execution["execution_authorized"] is False
    assert execution["live_io_authorized"] is False


def test_product_session_v1_protocol_explicitly_preserves_claim_and_data_boundaries() -> None:
    protocol = " ".join(PROTOCOL.read_text().split())

    for required_text in (
        "V4 is immutable retained evidence",
        "Stage 2 and H4 remain stopped",
        "outside Git",
        "no retry is permitted",
        "not a perceptual-quality guarantee",
        "must not be run until a later implementation task",
        "label-blind",
        "prediction-commitment.json",
        "session-attestation.json",
        "offline-replay-v2",
        "three of the five",
        "technical_invalid",
        "behavioral repeatability",
        "operational reliability",
        "OWNER-APPROVED DESIGN",
        "per-session authorization",
    ):
        assert required_text in protocol


def test_registered_cohort_hash_pins_runtime_before_live_io() -> None:
    cohort = json.loads(COHORT_REGISTRATION.read_text())
    digests = cohort["frozen_digests"]

    assert cohort["registered_before_live_io"] is True
    assert cohort["live_io_performed_before_registration"] is False
    assert len(cohort["ordered_session_slot_ids"]) == 5
    assert len(set(cohort["ordered_session_slot_ids"])) == 5
    assert cohort["minimum_fully_passed_slots"] == 3
    assert cohort["passes_must_be_consecutive"] is False
    assert digests["protocol"] == _sha256(PROTOCOL)
    assert digests["preregistration"] == _sha256(PREREGISTRATION)
    assert digests["threshold_calibration"] == _sha256(THRESHOLD_CALIBRATION)
    assert digests["representation_implementation"] == _sha256(
        ROOT / "src" / "cora_infans" / "vocal_imitation.py"
    )
    assert digests["live_runner_implementation"] == _sha256(
        ROOT / "src" / "cora_infans" / "product_session_live.py"
    )
    assert digests["private_storage_implementation"] == _sha256(
        ROOT / "src" / "cora_infans" / "private_session_storage.py"
    )
    assert digests["operating_point"] == _sha256(
        ROOT / "config" / "operational-point.json"
    )


def test_fresh_authorization_is_bound_only_to_first_registered_slot() -> None:
    cohort = json.loads(COHORT_REGISTRATION.read_text())
    authorization = json.loads(SESSION_01_AUTHORIZATION.read_text())

    assert authorization["session_id"] == cohort["ordered_session_slot_ids"][0]
    assert authorization["ordered_slot_index"] == 1
    assert authorization["recording_authorized"] is True
    assert authorization["playback_authorized"] is True
    assert authorization["device_discovery_authorized"] is True
    assert authorization["one_session_live_io_authorized"] is True
    assert authorization["authorization_applies_to_other_slots"] is False
    assert authorization["live_io_performed_before_authorization"] is False
    assert authorization["retention_days"] == 30


def test_slot_01_failure_preserves_behavioral_and_operational_denominators() -> None:
    result = json.loads(SESSION_01_RESULT.read_text())
    cohort = json.loads(COHORT_STATUS_AFTER_SLOT_01.read_text())

    assert result["status"] == "behavioral_failed"
    assert result["technical_valid"] is True
    assert result["technical_failure_reasons"] == []
    assert result["retry_permitted"] is False
    assert result["held_evaluation_opened"] is False
    assert result["waveform_scores_accessed"] is False
    assert result["manifest_externally_anchored"] is False
    assert result["retention_deadline"] == "2026-08-15"

    assert cohort["started_slots"] == 1
    assert cohort["technically_valid_started_sessions"] == 1
    assert cohort["passed_sessions"] == 0
    assert cohort["behavioral_repeatability"]["value"] == 0.0
    assert cohort["operational_reliability"]["value"] == 1.0
    assert cohort["cohort_resolved"] is False
    assert cohort["product_level_verdict_available"] is False
    assert cohort["product_level_pass_still_mathematically_possible"] is True
    assert cohort["next_slot_requires_fresh_authorization"] is True
