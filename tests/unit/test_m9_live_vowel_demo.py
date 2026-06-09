"""Regression tests for the M9 live vowel imitation demo."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_live_vowel_demo_dry_run_loops_and_writes_latest_report(tmp_path: Path):
    output_dir = tmp_path / "demo"
    script = PROJECT_ROOT / "scripts" / "m9_live_vowel_demo.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--iterations",
            "2",
            "--family",
            "phase_triad",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "M9 Live Vowel Imitation Loop" in result.stdout
    assert "[0001] family=phase_triad" in result.stdout
    assert "[0002] family=phase_triad" in result.stdout
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["family"] == "phase_triad"
    assert report["phase_closure"]["passed"] is True
    assert (output_dir / "run-0001" / "report.json").exists()
    assert (output_dir / "run-0002" / "report.json").exists()
    assert (output_dir / "run-0001" / "rendered_vowel_transition.wav").exists()
    assert (output_dir / "run-0002" / "captured_vowel_transition.wav").exists()


def test_live_vowel_legacy_shim_routes_to_loop(tmp_path: Path):
    output_dir = tmp_path / "legacy"
    script = PROJECT_ROOT / "scripts" / "m9_live_vowel.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--iterations",
            "1",
            "--family",
            "phase_arc",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "M9 Live Vowel Imitation Loop" in result.stdout
    assert "family=phase_arc" in result.stdout
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["family"] == "phase_arc"
    assert report["phase_closure"]["passed"] is True
