import json
from pathlib import Path

from cora_infans.structured_workspace_binding import (
    build_workspace_binding_protocol,
    run_fake_workspace_binding_protocol,
)


def test_workspace_binding_protocol_is_bounded_and_label_scoped() -> None:
    protocol = build_workspace_binding_protocol()

    assert protocol["trial_count"] == 15
    assert protocol["maximum_total_playback_seconds"] < 25.0
    assert protocol["live_io_wired"] is False
    assert protocol["capacity_sweep"] == [1, 2, 3, 4]


def test_fake_workspace_binding_exposes_baselines_and_capacity() -> None:
    report = run_fake_workspace_binding_protocol()

    selected = report["reports"]["3"]
    assert report["verdict"] == "fake_io_plumbing_passed"
    assert report["promotion_allowed"] is False
    assert selected["scheduled_identity_plus_world_evidence"]["metrics"]["f1"] == 1.0
    assert selected["flat"]["metrics"]["f1"] == 0.8
    assert selected["fixed_self_window"]["metrics"]["world_recall"] == 0.5
    assert report["reports"]["1"]["scheduled_identity_plus_world_evidence"]["metrics"]["f1"] < 1.0


def test_retained_structured_workspace_result_preserves_failed_first_rule() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (
            root
            / "docs/evidence/2026-07-10-structured-workspace-binding-fake-result.json"
        ).read_text()
    )

    assert result["promotion_allowed"] is False
    assert result["verdict"] == "fake_io_plumbing_passed"
    assert result["design_history"]["initial_midpoint_rule"]["scheduled_capacity_three_f1"] == 0.8
    assert result["reports"]["3"]["scheduled_identity_plus_world_evidence"]["metrics"]["f1"] == 1.0
    assert result["reports"]["3"]["fixed_self_window"]["metrics"]["world_recall"] == 0.5
