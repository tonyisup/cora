import hashlib
import json
from pathlib import Path

from cora_infans.energy_audit import TRIAL_ORDER, audit_energy_trials


def test_energy_audit_verifies_samples_workloads_and_hashes(tmp_path: Path) -> None:
    (tmp_path / "manifest.txt").write_text(
        "\n".join(
            [
                "git_commit=test",
                "powermetrics_sample_interval_ms=500",
                "powermetrics_samples=24",
                "workload_duration_seconds=10",
            ]
        )
    )
    paths = [tmp_path / "manifest.txt"]
    for index, condition in enumerate(TRIAL_ORDER, start=1):
        prefix = f"{index:02d}_{condition}"
        power = tmp_path / f"{prefix}_powermetrics.txt"
        power.write_text(
            "Machine model: Mac16,11\nOS version: test\n"
            + "\n".join(
                "CPU Power: 10000 mW\n"
                "Combined Power (CPU + GPU + ANE): 10100 mW"
                for _ in range(24)
            )
        )
        stderr = tmp_path / f"{prefix}_powermetrics.stderr"
        stderr.write_text("")
        workload = tmp_path / f"{prefix}_workload.json"
        payload = {"processor": condition, "duration_requested_seconds": 10}
        if condition != "idle":
            payload.update(
                {
                    "wall_seconds": 10.0,
                    "frames_processed": 1_000_000,
                    "frames_per_second": 100_000.0,
                    "average_process_cpu_percent": 99.0,
                    "resident_event_count": 11,
                }
            )
        workload.write_text(json.dumps(payload))
        paths.extend((power, stderr, workload))
    (tmp_path / "sha256.txt").write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}"
            for path in paths
        )
        + "\n"
    )

    report = audit_energy_trials(tmp_path)

    assert report["overall_criteria_passed"] is True
    assert report["promotion_allowed"] is True
    assert report["by_condition"]["cora"]["cpu_joules_per_million_frames"][
        "mean"
    ] == 100.0
    assert report["comparison"][
        "cora_to_conventional_cpu_energy_per_frame_ratio"
    ] == 1.0


def test_retained_energy_result_preserves_measured_tradeoff() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "docs/evidence/2026-07-11-stage6-energy-result.json").read_text()
    )

    assert result["by_condition"]["cora"][
        "mean_cpu_joules_per_million_frames"
    ] == 27.362286514817153
    assert result["by_condition"]["conventional"][
        "mean_cpu_joules_per_million_frames"
    ] == 11.0773317763486
    assert result["comparison"][
        "cora_to_conventional_cpu_energy_per_frame_ratio"
    ] == 2.470115282927504
    assert all(result["criteria"].values())
    assert result["overall_criteria_passed"] is True
    assert result["promotion_allowed"] is True
