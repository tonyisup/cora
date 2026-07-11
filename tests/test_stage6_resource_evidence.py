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
