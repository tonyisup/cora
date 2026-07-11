import pytest

from cora_infans.energy_workload import run_sustained_energy_workload
from cora_infans.cli import energy_workload_cli


@pytest.mark.parametrize("mode", ["cora", "conventional"])
def test_sustained_energy_workload_reports_external_boundary(mode: str) -> None:
    report = run_sustained_energy_workload(mode=mode, duration_seconds=0.01)

    assert report["processor"] == mode
    assert report["frames_processed"] > 0
    assert report["events_emitted"] > 0
    assert report["frames_per_second"] > 0.0
    assert report["average_process_cpu_percent"] > 0.0
    assert report["resident_event_count"] <= 11
    assert report["external_measurement_required"]["tool"] == "macOS powermetrics"
    assert report["external_measurement_required"]["not_measured_by_this_report"] is True


def test_sustained_energy_workload_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        run_sustained_energy_workload(mode="unknown", duration_seconds=0.01)


def test_energy_workload_cli_parses_mode_and_duration() -> None:
    report = energy_workload_cli(
        [
            "--energy-workload",
            "--mode",
            "conventional",
            "--duration-seconds",
            "0.005",
        ]
    )

    assert report["processor"] == "conventional"
    assert report["duration_requested_seconds"] == 0.005
