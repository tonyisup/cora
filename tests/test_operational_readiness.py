"""Readiness aggregation for the bounded operational system."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from cora_infans.operational_readiness import (
    EVIDENCE_FILES,
    EXPECTED_EVIDENCE_SHA256,
    build_operational_readiness_report,
)


def test_repository_operational_evidence_is_ready() -> None:
    root = Path(__file__).resolve().parents[1]
    report = build_operational_readiness_report(root / "docs/evidence")

    assert report["operational_evidence_ready"] is True
    assert all(report["checks"].values())
    assert len(report["evidence_sha256"]) == len(EVIDENCE_FILES)
    assert report["evidence_sha256"] == EXPECTED_EVIDENCE_SHA256
    assert report["stopped_programs"] == [
        "stage2_register_detection",
        "stage5_local_inverse_closure",
    ]
    assert report["test_suite"] == "not_executed_by_readiness_report"
    assert report["checks"]["current_operational_acceptance_passed"] is True


def test_operational_readiness_fails_loudly_on_missing_evidence(tmp_path) -> None:
    with pytest.raises(ValueError, match="required operational evidence is missing"):
        build_operational_readiness_report(tmp_path)


def test_operational_readiness_cli() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--operational-readiness",
            "--evidence-dir",
            str(root / "docs/evidence"),
        ],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    )
    assert json.loads(result.stdout)["operational_evidence_ready"] is True


def test_operational_readiness_cli_fails_when_a_gate_is_false(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    for filename in EVIDENCE_FILES.values():
        shutil.copy2(root / "docs/evidence" / filename, evidence_dir / filename)

    timing_path = evidence_dir / EVIDENCE_FILES["stage6_timing"]
    timing = json.loads(timing_path.read_text())
    timing["overall_criteria_passed"] = False
    timing_path.write_text(json.dumps(timing))

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cora_infans",
            "--operational-readiness",
            "--evidence-dir",
            str(evidence_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["operational_evidence_ready"] is False
    assert report["checks"]["stage6_retained_timing_gate_passed"] is False
    assert report["checks"]["stage6_timing_sha256_matches_pin"] is False
