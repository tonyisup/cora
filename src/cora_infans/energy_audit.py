"""Audit retained powermetrics trials against the frozen Stage 6 protocol."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path


TRIAL_ORDER = (
    "idle",
    "conventional",
    "cora",
    "idle",
    "cora",
    "conventional",
    "idle",
    "conventional",
    "cora",
)
CPU_POWER = re.compile(r"^CPU Power: (?P<mw>\d+) mW$", re.MULTILINE)
COMBINED_POWER = re.compile(
    r"^Combined Power \(CPU \+ GPU \+ ANE\): (?P<mw>\d+) mW$", re.MULTILINE
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _published_hashes(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (root / "sha256.txt").read_text().splitlines():
        digest, raw_path = line.split(maxsplit=1)
        name = Path(raw_path).name
        if name in entries:
            raise ValueError(f"duplicate published hash: {name}")
        entries[name] = digest
    return entries


def audit_energy_trials(root_path: str | Path) -> dict[str, object]:
    root = Path(root_path)
    manifest = dict(
        line.split("=", 1)
        for line in (root / "manifest.txt").read_text().splitlines()
        if "=" in line
    )
    if manifest.get("powermetrics_sample_interval_ms") != "500":
        raise ValueError("unexpected powermetrics sample interval")
    if manifest.get("powermetrics_samples") != "24":
        raise ValueError("unexpected powermetrics sample count")
    if manifest.get("workload_duration_seconds") != "10":
        raise ValueError("unexpected workload duration")

    published = _published_hashes(root)
    raw_sha256: dict[str, str] = {}
    trials: list[dict[str, object]] = []
    machine_models: set[str] = set()
    os_versions: set[str] = set()
    for index, condition in enumerate(TRIAL_ORDER, start=1):
        prefix = f"{index:02d}_{condition}"
        paths = {
            "power": root / f"{prefix}_powermetrics.txt",
            "stderr": root / f"{prefix}_powermetrics.stderr",
            "workload": root / f"{prefix}_workload.json",
        }
        for path in paths.values():
            digest = _sha256(path)
            if published.get(path.name) != digest:
                raise ValueError(f"published hash mismatch: {path.name}")
            raw_sha256[path.name] = digest
        if paths["stderr"].read_bytes():
            raise ValueError(f"powermetrics stderr is not empty: {prefix}")

        text = paths["power"].read_text()
        cpu_all = [float(match.group("mw")) for match in CPU_POWER.finditer(text)]
        combined_all = [
            float(match.group("mw")) for match in COMBINED_POWER.finditer(text)
        ]
        if len(cpu_all) != 24 or len(combined_all) != 24:
            raise ValueError(f"expected 24 power samples: {prefix}")
        cpu = cpu_all[2:-2]
        combined = combined_all[2:-2]
        model_match = re.search(r"^Machine model: (.+)$", text, re.MULTILINE)
        os_match = re.search(r"^OS version: (.+)$", text, re.MULTILINE)
        if model_match is None or os_match is None:
            raise ValueError(f"missing host identity: {prefix}")
        machine_models.add(model_match.group(1))
        os_versions.add(os_match.group(1))

        workload = json.loads(paths["workload"].read_text())
        if workload.get("processor") != condition:
            raise ValueError(f"workload condition mismatch: {prefix}")
        trial: dict[str, object] = {
            "trial": index,
            "condition": condition,
            "analyzed_sample_indices_one_based": [3, 22],
            "cpu_power_mw": _distribution(cpu),
            "combined_power_mw": _distribution(combined),
            "workload": workload,
        }
        if condition != "idle":
            frames_per_second = float(workload["frames_per_second"])
            trial["cpu_joules_per_million_frames"] = (
                float(trial["cpu_power_mw"]["mean"]) * 1000.0
                / frames_per_second
            )
            trial["combined_joules_per_million_frames"] = (
                float(trial["combined_power_mw"]["mean"]) * 1000.0
                / frames_per_second
            )
        trials.append(trial)

    by_condition: dict[str, dict[str, object]] = {}
    for condition in ("idle", "conventional", "cora"):
        selected = [trial for trial in trials if trial["condition"] == condition]
        summary: dict[str, object] = {
            "trial_count": len(selected),
            "trial_mean_cpu_power_mw": _distribution(
                [float(trial["cpu_power_mw"]["mean"]) for trial in selected]
            ),
            "trial_median_cpu_power_mw": _distribution(
                [float(trial["cpu_power_mw"]["median"]) for trial in selected]
            ),
            "trial_mean_combined_power_mw": _distribution(
                [float(trial["combined_power_mw"]["mean"]) for trial in selected]
            ),
        }
        if condition != "idle":
            summary["frames_per_second"] = _distribution(
                [float(trial["workload"]["frames_per_second"]) for trial in selected]
            )
            summary["cpu_joules_per_million_frames"] = _distribution(
                [float(trial["cpu_joules_per_million_frames"]) for trial in selected]
            )
            summary["combined_joules_per_million_frames"] = _distribution(
                [
                    float(trial["combined_joules_per_million_frames"])
                    for trial in selected
                ]
            )
        by_condition[condition] = summary

    paired_idle_subtracted = []
    for block, (idle_index, first_index, second_index) in enumerate(
        ((0, 1, 2), (3, 4, 5), (6, 7, 8)), start=1
    ):
        idle_mean = float(trials[idle_index]["cpu_power_mw"]["mean"])
        for active_index in (first_index, second_index):
            paired_idle_subtracted.append(
                {
                    "block": block,
                    "condition": trials[active_index]["condition"],
                    "idle_subtracted_mean_cpu_power_mw": (
                        float(trials[active_index]["cpu_power_mw"]["mean"])
                        - idle_mean
                    ),
                }
            )

    active = [trial for trial in trials if trial["condition"] != "idle"]
    criteria = {
        "all_nine_trials_complete": len(trials) == 9,
        "analyzed_valid_cpu_power_samples_each_20": all(
            int(trial["cpu_power_mw"]["count"]) == 20 for trial in trials
        ),
        "active_workload_wall_seconds_in_range": all(
            9.9 <= float(trial["workload"]["wall_seconds"]) <= 10.2
            for trial in active
        ),
        "active_workload_average_process_cpu_percent_min_90": all(
            float(trial["workload"]["average_process_cpu_percent"]) >= 90.0
            for trial in active
        ),
        "resident_event_count_each_max_11": all(
            int(trial["workload"]["resident_event_count"]) <= 11
            for trial in active
        ),
        "frames_processed_each_min_1": all(
            int(trial["workload"]["frames_processed"]) >= 1 for trial in active
        ),
        "positive_cpu_power_each_sample": all(
            float(trial["cpu_power_mw"]["minimum"]) > 0.0 for trial in trials
        ),
        "same_host_and_sampler_configuration": (
            len(machine_models) == 1 and len(os_versions) == 1
        ),
        "paired_order_preserved": tuple(
            str(trial["condition"]) for trial in trials
        )
        == TRIAL_ORDER,
        "all_published_hashes_verified": True,
    }
    conventional_energy = float(
        by_condition["conventional"]["cpu_joules_per_million_frames"]["mean"]
    )
    cora_energy = float(
        by_condition["cora"]["cpu_joules_per_million_frames"]["mean"]
    )
    return {
        "mode": "stage6_same_host_estimated_energy_audit",
        "external_artifact_root": str(root),
        "manifest": manifest,
        "raw_sha256": raw_sha256,
        "host": {
            "machine_models": sorted(machine_models),
            "os_versions": sorted(os_versions),
        },
        "trials": trials,
        "by_condition": by_condition,
        "paired_idle_subtracted": paired_idle_subtracted,
        "comparison": {
            "cora_to_conventional_cpu_energy_per_frame_ratio": (
                cora_energy / conventional_energy
            ),
            "cora_minus_conventional_cpu_joules_per_million_frames": (
                cora_energy - conventional_energy
            ),
            "interpretation": (
                "Total same-host estimated CPU energy per processed frame is "
                "primary; idle subtraction is retained only as a variable secondary view."
            ),
        },
        "criteria": criteria,
        "overall_criteria_passed": all(criteria.values()),
        "promotion_allowed": all(criteria.values()),
        "claim_boundary": (
            "powermetrics estimates on one M4 Pro host for sustained synthetic "
            "salient frame processing; excludes wall-plug, acoustic, device, "
            "and cross-host energy"
        ),
    }
