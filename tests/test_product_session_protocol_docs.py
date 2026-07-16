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
