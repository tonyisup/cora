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
    evaluate_stage6_operating_point,
)


def _clean_stage6_report(latency_ms: float = 10.5) -> dict[str, object]:
    return {
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

    clipped = _clean_stage6_report()
    clipped["active_playback_clipped_sample_count"] = 1
    assert evaluate_stage6_operating_point(clipped)["operating_point_passed"] is False

    with pytest.raises(ValueError, match="missing Stage 6 integrity"):
        evaluate_stage6_operating_point(
            {"salient_adc_frame_end_to_workspace_p95_ms": 10.0}
        )


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
