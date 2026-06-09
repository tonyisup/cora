"""Regression tests for the M9 live vowel imitation demo."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_live_vowel_demo_dry_run_waits_for_mic_and_writes_report(tmp_path: Path):
    output_dir = tmp_path / "demo"
    script = PROJECT_ROOT / "scripts" / "m9_live_vowel_demo.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--iterations",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Cora Vowel Imitation (Spectral)" in result.stdout
    assert "heard=True" in result.stdout
    assert "Dry-run summary: 2/2" in result.stdout
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "dry_run"
    assert report["iterations"] == 2
    assert (output_dir / "report.json").exists()


def test_live_vowel_legacy_shim_routes_to_demo(tmp_path: Path):
    output_dir = tmp_path / "legacy"
    script = PROJECT_ROOT / "scripts" / "m9_live_vowel.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--iterations",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Cora Vowel Imitation (Spectral)" in result.stdout
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["iterations"] == 1
    assert report["mode"] == "dry_run"
