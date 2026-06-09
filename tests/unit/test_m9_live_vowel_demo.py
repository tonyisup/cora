"""Regression tests for the M9 live vowel-transition imitation demo."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_live_vowel_demo_dry_run_waits_for_speech_and_imitates_transitions(tmp_path: Path):
    output_dir = tmp_path / "demo"
    script = PROJECT_ROOT / "scripts" / "m9_live_vowel_demo.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--iterations",
            "2",
            "--families",
            "phase_triad,phase_arc",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "M9 Live Vowel-Transition Imitation Loop" in result.stdout
    assert "heard speech" in result.stdout
    assert "imitate phase_triad" in result.stdout
    assert "imitate phase_arc" in result.stdout
    assert "Loop summary: 2/2 transition triggers passed" in result.stdout

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["family"] in {"phase_triad", "phase_arc"}
    assert report["phase_closure"]["passed"] is True
    assert report["phase_closure"]["target_transition_count"] >= 2
    assert (output_dir / "run-0001" / "report.json").exists()
    assert (output_dir / "run-0002" / "report.json").exists()


def test_live_vowel_legacy_shim_routes_to_transition_demo(tmp_path: Path):
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
    assert "M9 Live Vowel-Transition Imitation Loop" in result.stdout
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["phase_closure"]["passed"] is True
    assert report["family"] in {"phase_triad", "phase_arc"}
