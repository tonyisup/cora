"""Executable invariants for the bounded operational VISION."""

from pathlib import Path

import pytest

from cora_infans.io.ids_corpus import IDS_RESERVED_FIELDSITES
from cora_infans.operational_contract import (
    OPERATIONAL_CONTRACT,
    STAGE2_PROTECTED_FIELDSITES,
    assert_h4_speaker_allowed,
    assert_program_active,
    assert_stage2_fieldsite_allowed,
    evaluate_stage4_exact_efference,
    evaluate_stage6_operating_point,
)


def _operating_point() -> dict[str, object]:
    stage6 = OPERATIONAL_CONTRACT["stage6"]
    return {
        "schema_version": stage6["operating_point_schema_version"],
        "host": stage6["host"],
        "input": stage6["input"],
        "self_source": stage6["self_source"],
        "world_source": stage6["world_source"],
        "operator_attestations": {
            "self_source_geometry_confirmed": True,
            "self_source_system_volume_confirmed": True,
            "world_source_geometry_confirmed": True,
            "world_source_hardware_volume_at_approximately_50_percent": True,
        },
    }


def _clean_stage6_report(latency_ms: float = 10.5) -> dict[str, object]:
    profile = OPERATIONAL_CONTRACT["stage6"]["acceptance_profile"]
    return {
        "operating_point": _operating_point(),
        "real_capture": True,
        "capture_backend": OPERATIONAL_CONTRACT["stage6"]["capture_backend"],
        "frame_ms": profile["frame_ms"],
        "frame_count": profile["target_frame_count"],
        "salient_frame_count": profile["minimum_salient_frame_count"],
        "active_playback_frame_count": 250,
        "software_playback_gain_db": profile["software_playback_gain_db"],
        "stimulus_sha256": profile["stimulus_sha256"],
        "salient_adc_frame_end_to_workspace_p95_ms": latency_ms,
        "device_deadline_miss_count": 0,
        "overflow_count": 0,
        "callback_status_count": 0,
        "timestamp_discontinuity_count": 0,
        "fallback_frame_count": 0,
        "active_playback_clipped_sample_count": 0,
    }


def test_contract_matches_protected_acquisition_boundary() -> None:
    assert STAGE2_PROTECTED_FIELDSITES == frozenset({"WEL", "USD", "TOR"})
    assert IDS_RESERVED_FIELDSITES == STAGE2_PROTECTED_FIELDSITES
    assert OPERATIONAL_CONTRACT["protected_data"][
        "stage2_pristine_fieldsites"
    ] == ["USD", "TOR"]

    for code in STAGE2_PROTECTED_FIELDSITES:
        with pytest.raises(ValueError, match="protected"):
            assert_stage2_fieldsite_allowed(code)
    assert_stage2_fieldsite_allowed("BEJ") is None


def test_stopped_programs_and_h4_speakers_fail_loudly() -> None:
    with pytest.raises(ValueError, match="stopped"):
        assert_program_active("stage2_register_detection")
    with pytest.raises(ValueError, match="stopped"):
        assert_program_active("stage5_local_inverse_closure")
    assert_program_active("stage4_exact_efference_binding") is None

    for speaker in ("female6", "male11"):
        with pytest.raises(ValueError, match="protected"):
            assert_h4_speaker_allowed("vocalset", speaker)
    for speaker in ("rms", "slp"):
        with pytest.raises(ValueError, match="protected"):
            assert_h4_speaker_allowed("cmu_arctic", speaker)
    assert_h4_speaker_allowed("vocalset", "female4") is None


def test_stage6_operating_point_uses_exact_metric_and_ceiling() -> None:
    passing = evaluate_stage6_operating_point(_clean_stage6_report())
    assert passing["operating_point_passed"] is True
    assert "frame accumulation" in passing["claim_boundary"]

    slow = evaluate_stage6_operating_point(_clean_stage6_report(10.5001))
    assert slow["operating_point_passed"] is False

    missing_salient_latency = _clean_stage6_report()
    missing_salient_latency[
        "salient_adc_frame_end_to_workspace_p95_ms"
    ] = None
    missing_result = evaluate_stage6_operating_point(missing_salient_latency)
    assert missing_result["operating_point_passed"] is False
    assert missing_result["criteria"]["salient_latency_available"] is False

    clipped = _clean_stage6_report()
    clipped["active_playback_clipped_sample_count"] = 1
    assert evaluate_stage6_operating_point(clipped)["operating_point_passed"] is False

    with pytest.raises(ValueError, match="missing Stage 6 integrity"):
        evaluate_stage6_operating_point(
            {
                "operating_point": _operating_point(),
                "salient_adc_frame_end_to_workspace_p95_ms": 10.0,
            }
        )

    wrong_geometry = _clean_stage6_report()
    wrong_geometry["operating_point"]["world_source"] = {
        **OPERATIONAL_CONTRACT["stage6"]["world_source"],
        "angle_degrees": 270,
    }
    result = evaluate_stage6_operating_point(wrong_geometry)
    assert result["operating_point_passed"] is False
    assert result["criteria"]["world_source_matches"] is False


def test_stage4_requires_exact_replay_identity_and_bounded_capacity() -> None:
    digest = "a" * 64
    report = {
        "operating_point": _operating_point(),
        "workspace_capacity": 3,
        "clipped_sample_count": 0,
        "exact_replay": {
            "performed": True,
            "commanded_utterance_sha256": digest,
            "prior_replay_utterance_sha256": digest,
            "efference_id": "command-001",
            "sensory_consequence_sha256": "b" * 64,
        },
    }
    assert evaluate_stage4_exact_efference(report)["exact_efference_passed"] is True

    report["exact_replay"]["prior_replay_utterance_sha256"] = "c" * 64
    result = evaluate_stage4_exact_efference(report)
    assert result["exact_efference_passed"] is False
    assert result["criteria"]["prior_replay_matches_command"] is False


def test_vision_is_the_bounded_operational_source_of_truth() -> None:
    root = Path(__file__).resolve().parents[1]
    vision = (root / "VISION.md").read_text()
    compact = " ".join(vision.split())

    assert vision.startswith("# Cora-Infans — bounded operational specification")
    assert "salient ADC-frame-end-to-workspace p95 latency" in compact
    assert "at most `10.5 ms`" in compact
    assert "prior playback replay of that exact utterance is required" in compact
    assert "Broad cross-cultural register detection" not in compact
    assert "Reliable generalized cross-cultural IDS/ADS" in compact
    assert "No autonomous expansion beyond these boundaries" in compact
