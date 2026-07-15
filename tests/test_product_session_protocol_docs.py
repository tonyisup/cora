"""Documentation contract for the proposed real product-session protocol."""

from __future__ import annotations

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


def test_product_session_v1_protocol_is_draft_only_and_freezes_proposed_values() -> None:
    protocol = PROTOCOL.read_text()
    preregistration = json.loads(PREREGISTRATION.read_text())

    assert "Status: DRAFT — NOT AUTHORIZED FOR RECORDING OR LIVE I/O." in protocol
    assert preregistration["status"] == "draft_not_authorized"
    assert preregistration["runtime_implementation_required_before_execution"] is True
    assert preregistration["explicit_owner_approval_required_before_execution"] is True

    design = preregistration["proposed_design"]
    assert design["targets"] == ["/a/", "/i/", "/u/"]
    assert design["development_takes_per_target"] == 2
    assert design["held_takes_per_target"] == 2
    assert design["association_freeze_after_development_takes"] == 6
    assert design["held_capture_count_after_freeze"] == 6
    assert design["fresh_sessions_required_for_product_claim"] == 3
    assert design["raw_audio_retention_days"] == 30

    thresholds = preregistration["proposed_score_thresholds"]
    assert thresholds == {
        "onset_offset_alignment_ms_max": 80.0,
        "duration_difference_ms_max": 150.0,
        "rms_envelope_distance_max": 0.05,
        "pitch_contour_distance_hz_max": 30.0,
        "cepstral_distance_max": 0.5,
    }
    assert preregistration["all_component_thresholds_required_per_held_take"] is True
    assert preregistration["threshold_basis"] == "synthetic_engineering_calibration_not_perceptual_validation"


def test_product_session_v1_protocol_explicitly_preserves_claim_and_data_boundaries() -> None:
    protocol = " ".join(PROTOCOL.read_text().split())

    for required_text in (
        "V4 is immutable retained evidence",
        "Stage 2 and H4 remain stopped",
        "outside Git",
        "No retry is permitted after a quality or score failure",
        "not a perceptual-quality guarantee",
        "must not be run until a later implementation task",
    ):
        assert required_text in protocol
