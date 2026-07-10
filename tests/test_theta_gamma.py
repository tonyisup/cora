"""Tests for the theta-gamma scheduling layer (deterministic, no hardware)."""

from cora_infans.pipeline import WorkspaceEvent
from cora_infans.theta_gamma import ThetaGammaScheduler
from cora_infans.simulation import build_sequence_report_with_scheduling


def _ev(timestamp_ms: int, label: str) -> WorkspaceEvent:
    return WorkspaceEvent(
        kind="salient_audio",
        saliency_score=0.9,
        workspace_state={"timestamp_ms": timestamp_ms, "focus_label": label},
    )


def test_scheduler_groups_events_into_theta_cycles() -> None:
    s = ThetaGammaScheduler(theta_ms=200, gamma_ms=25)
    # Two events inside cycle 0 (ts 10, 50), one in cycle 1 (ts 210).
    s.push(_ev(10, "ch0"))
    s.push(_ev(50, "ch0"))
    s.push(_ev(210, "ch1"))
    s._consolidate_active()

    cycles = s.consolidated_cycles()
    assert len(cycles) == 2
    assert cycles[0].index == 0
    assert cycles[0].event_count == 2
    assert cycles[1].index == 1
    assert cycles[1].event_count == 1


def test_automatic_cycle_boundary_on_push() -> None:
    s = ThetaGammaScheduler(theta_ms=200, gamma_ms=25)
    s.push(_ev(10, "ch0"))
    s.push(_ev(260, "ch1"))  # next cycle; previous should auto-consolidate
    assert s.current_cycle_index == 1
    assert len(s.consolidated_cycles()) == 1
    assert s.consolidated_cycles()[0].event_count == 1


def test_gamma_binning_counts_distinct_subwindows() -> None:
    s = ThetaGammaScheduler(theta_ms=200, gamma_ms=25)
    s.push(_ev(0, "ch0"))
    s.push(_ev(10, "ch0"))  # same gamma bin (0..25)
    s.push(_ev(30, "ch0"))  # next gamma bin (25..50)
    s._consolidate_active()
    cycle = s.consolidated_cycles()[0]
    assert cycle.gamma_bins_occupied == 2


def test_custom_gamma_geometry_drives_cycle_summary() -> None:
    scheduler = ThetaGammaScheduler(theta_ms=100, gamma_ms=10)
    scheduler.push(_ev(0, "ch0"))
    scheduler.push(_ev(15, "ch0"))

    cycle = scheduler.summary()["cycles"][0]
    assert cycle["gamma_bins_occupied"] == 2


def test_capacity_evicts_oldest_cycles() -> None:
    s = ThetaGammaScheduler(theta_ms=100, gamma_ms=25, capacity_cycles=2)
    for i in range(5):
        s.push(_ev(i * 100, f"ch{i}"))
    s._consolidate_active()
    # Only the most recent 2 cycles remain resident.
    assert len(s.consolidated_cycles()) == 2
    assert s.consolidated_cycles()[0].index == 3


def test_binding_stability_reports_consecutive_dominant_repeats() -> None:
    s = ThetaGammaScheduler(theta_ms=100, gamma_ms=25)
    for i in range(4):
        # Each cycle has a single dominant label; first two chA, last two chB.
        label = "chA" if i < 2 else "chB"
        s.push(_ev(i * 100, label))
    s._consolidate_active()
    summary = s.summary()
    # dominants: [chA, chA, chB, chB] -> 2 of 3 consecutive repeats.
    assert abs(summary["binding_stability_fraction"] - (2 / 3)) < 1e-9


def test_scheduler_rejects_invalid_band_geometry() -> None:
    import pytest

    with pytest.raises(ValueError):
        ThetaGammaScheduler(theta_ms=25, gamma_ms=200)  # gamma > theta


def test_composed_sequence_report_includes_theta_gamma_summary() -> None:
    report = build_sequence_report_with_scheduling("sweep")
    assert report["family"] == "sweep"
    assert "theta_gamma" in report
    tg = report["theta_gamma"]
    assert tg["theta_ms"] == 200
    assert tg["gamma_ms"] == 25
    # sweep is 3 salient frames across 50ms steps -> within one theta cycle.
    assert tg["resident_cycle_count"] >= 1
    assert "cycles" in tg and isinstance(tg["cycles"], list)
