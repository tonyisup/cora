import json
from pathlib import Path


def test_retained_stage6_resource_result_preserves_failed_deadline_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "docs/evidence/2026-07-10-stage6-resource-result.json").read_text()
    )

    assert result["offline"]["criteria"] == {
        "real_time_factor_max_1": True,
        "p95_latency_ms_max_25": True,
        "baseline_comparison_present": True,
        "baseline_input_and_instrumentation_identical": True,
    }
    assert result["live"]["aggregate"]["frame_count"] == 1200
    assert result["live"]["aggregate"]["input_overflow_count"] == 0
    assert result["live"]["aggregate"]["operational_deadline_miss_rate"] == 0.1475
    assert result["live"]["criteria"][
        "operational_deadline_miss_rate_each_max_0_05"
    ] is False
    assert result["overall_criteria_passed"] is False
    assert result["promotion_allowed"] is False


def test_retained_native_rate_retry_preserves_deadline_failure() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-10-stage6-native-rate-retry-result.json"
        ).read_text()
    )

    assert result["aggregate"]["frame_count"] == 1200
    assert result["aggregate"]["input_overflow_count"] == 0
    assert result["comparison_to_48000_hz_parent"]["reported_input_latency_ms"] == {
        "from": 95.33333333333334,
        "to": 42.0,
        "reduction_percent": 55.94405594405595,
    }
    assert result["comparison_to_48000_hz_parent"][
        "operational_deadline_miss_rate"
    ]["improved"] is False
    assert result["criteria"][
        "operational_deadline_miss_rate_each_max_0_05"
    ] is False
    assert result["promotion_allowed"] is False


def test_retained_device_timestamp_gate_preserves_bounded_pass() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "docs/evidence/2026-07-10-stage6-device-timestamp-result.json").read_text()
    )

    assert result["aggregate"]["frame_count"] == 1200
    assert result["aggregate"]["device_timestamped_frame_count"] == 1200
    assert result["aggregate"]["input_overflow_count"] == 0
    assert result["aggregate"]["callback_status_count"] == 0
    assert result["aggregate"]["timestamp_discontinuity_count"] == 0
    assert result["aggregate"]["device_operational_deadline_miss_rate"] == 0.0
    assert max(
        result["aggregate"]["device_adc_end_to_workspace_p95_ms_range"]
    ) < 30.0
    assert result["overall_criteria_passed"] is True
    assert result["promotion_allowed"] is True


def test_retained_salient_timing_gate_preserves_clipping_free_pass() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "docs/evidence/2026-07-10-stage6-salient-timing-result.json").read_text()
    )

    assert result["aggregate"]["frame_count"] == 1200
    assert result["aggregate"]["salient_frame_count"] == 142
    assert result["aggregate"]["clipped_sample_count"] == 0
    assert result["aggregate"]["input_overflow_count"] == 0
    assert result["aggregate"]["device_deadline_miss_count"] == 0
    assert max(
        result["aggregate"]["salient_adc_frame_end_to_workspace_p95_ms_range"]
    ) < 30.0
    assert all(result["criteria"].values())
    assert result["overall_criteria_passed"] is True
    assert result["promotion_allowed"] is True
