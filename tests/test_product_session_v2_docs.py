from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/BOUNDED_VOCAL_IMITATION_SESSION_PROTOCOL_V2.md"
PREREGISTRATION = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2-preregistration.json"
)
V1_CLOSURE = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v1-closure.json"
)
IMPLEMENTATION = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2-implementation.json"
)
OWNER_APPROVAL = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2-owner-approval.json"
)
REJECTION = (
    ROOT
    / "docs/evidence/2026-07-16-bounded-vocal-imitation-v2-candidate-rejection.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_closure_preserves_unresolved_slot_and_forbids_a_claim() -> None:
    closure = json.loads(V1_CLOSURE.read_text())
    assert closure["status"] == "closed_for_claims_with_unconsumed_blocked_slot"
    assert closure["started_slots"] == 4
    assert closure["consumed_slots"] == 4
    assert closure["passed_slots"] == 0
    assert closure["all_five_slots_resolved"] is False
    assert closure["aggregate_product_verdict_valid_under_v1"] is False
    assert closure["product_claim_allowed"] is False
    assert closure["prior_statuses_rewritten"] is False
    assert closure["v1_audio_reusable_as_v2_evaluation"] is False
    assert closure["successor_must_use_new_version_and_new_cohort"] is True


def test_v2_candidate_is_not_approved_executable_or_live_authorized() -> None:
    protocol = PROTOCOL.read_text()
    preregistration = json.loads(PREREGISTRATION.read_text())
    assert "NOT OWNER-APPROVED, NOT EXECUTABLE" in protocol
    assert preregistration["status"] == (
        "implemented_candidate_not_owner_approved_not_executable"
    )
    assert preregistration["owner_protocol_approval_recorded"] is False
    assert preregistration["execution_authorized_by_this_artifact"] is False
    assert preregistration["live_io_authorized_by_this_artifact"] is False
    assert preregistration["new_cohort_registered"] is False


def test_v2_changes_only_device_identity_and_playback_completion() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    changes = preregistration["v2_changes"]
    unchanged = preregistration["unchanged_from_v1"]
    assert changes["self_uid"] == "BuiltInSpeakerDevice"
    assert changes["world_uid"] == "BuiltInHeadphoneOutputDevice"
    assert changes["playback_implementation"] == "cora-audioqueue-player-v1"
    assert changes["buffer_callback_required"] is True
    assert changes["non_immediate_stop_required"] is True
    assert changes["queue_not_running_before_return_required"] is True
    assert changes["non_immediate_dispose_required"] is True
    assert changes["owner_confirmed_cue_audibility_required"] is True
    assert unchanged["targets"] == ["/a/", "/i/", "/u/"]
    assert unchanged["total_microphone_captures_per_session"] == 18
    assert unchanged["scoring_version"] == "offline-replay-v2"
    assert unchanged["repeatability_slots"] == 5
    assert unchanged["minimum_passing_slots"] == 3
    assert unchanged["behavioral_or_score_threshold_changed"] is False


def test_v2_preregistration_keeps_v1_score_thresholds() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    assert preregistration["score_thresholds"] == {
        "active_duration_difference_ms_max": 150.0,
        "rms_envelope_distance_max": 0.05,
        "pitch_contour_distance_hz_max": 30.0,
        "cepstral_distance_max": 0.5,
    }


def test_v2_protocol_requires_uid_drain_and_owner_audibility() -> None:
    protocol = " ".join(PROTOCOL.read_text().split())
    for text in (
        "BuiltInSpeakerDevice",
        "BuiltInHeadphoneOutputDevice",
        "AudioQueueStop(queue, false)",
        "AudioQueueDispose(queue, false)",
        "returned-buffer callback",
        "owner must immediately confirm audibility",
        "No V1 slot or audio may contribute",
        "NOT EXECUTABLE",
    ):
        assert text in protocol


def test_v2_implementation_record_hash_pins_candidate_without_live_claim() -> None:
    record = json.loads(IMPLEMENTATION.read_text())
    for name, entry in record["source_hashes"].items():
        if name != "native_audioqueue_player":
            assert entry["sha256"] == _sha256(ROOT / entry["path"])
    rejection = json.loads(REJECTION.read_text())
    note = rejection["evidence_integrity_note"]
    native = record["source_hashes"]["native_audioqueue_player"]
    assert native["sha256"] == note["implementation_record_native_source_sha256"]
    assert _sha256(ROOT / native["path"]) == note["restored_v2_source_sha256"]
    assert note["source_digest_matches_record"] is False
    assert note["rebuild_is_byte_identical_to_approved_binary"] is True
    assert record["native_binary"]["same_basename_reproducible"] is True
    assert record["offline_verification"]["full_suite"]["passed"] == 548
    assert record["offline_verification"]["full_suite"]["failed"] == 0
    assert record["required_live_validation"]["authorized"] is False
    assert record["required_live_validation"]["performed"] is False
    assert record["governance"]["owner_protocol_approval_recorded"] is False
    assert record["governance"]["new_v2_cohort_registered"] is False
    assert record["governance"]["session_live_io_authorized"] is False


def test_v2_owner_approval_hash_pins_design_without_authorizing_live_io() -> None:
    approval = json.loads(OWNER_APPROVAL.read_text())
    assert approval["status"] == (
        "owner_approved_protocol_design_and_retention_not_live_io"
    )
    for key in (
        "approved_protocol",
        "approved_preregistration",
        "approved_implementation_record",
        "approved_operating_point",
    ):
        entry = approval[key]
        assert entry["sha256"] == _sha256(ROOT / entry["path"])
    assert approval["approved_native_player"]["binary_sha256"] == (
        "ab6d421aac2388234aab439143257e1a7c544e22da74f3dee2e751c53d72cb3e"
    )
    execution = approval["execution_state"]
    assert execution["protocol_design_approved"] is True
    assert execution["retention_decision_approved"] is True
    assert execution["live_cue_validation_complete"] is False
    assert execution["new_v2_cohort_registered"] is False
    assert execution["execution_authorized"] is False
    assert execution["live_io_authorized"] is False
    assert execution["microphone_capture_authorized"] is False
