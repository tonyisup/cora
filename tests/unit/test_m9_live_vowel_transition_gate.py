"""Tests for the M9 live vowel-transition phase-closure gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "packages"))

from m9_sound.live_capture import run_live_vowel_transition_gate


def test_run_live_vowel_transition_gate_dry_run_passes(tmp_path: Path):
    def fake_capture_audio(playback_audio: np.ndarray, *, sr: int, **_: object) -> np.ndarray:
        tail = np.zeros(sr // 20, dtype=np.float32)
        return np.concatenate([playback_audio, tail])

    report = run_live_vowel_transition_gate(
        output_dir=tmp_path,
        families=["default", "phase_triad"],
        repeats=2,
        threshold=0.7,
        dry_run=True,
    )

    assert report["summary"]["verdict"] is True
    assert report["summary"]["pass_rate"] == 1.0
    assert report["summary"]["total"] == 4
    assert {row["family"] for row in report["results"]} == {"default", "phase_triad"}
    assert all(row["phase_closure"]["passed"] for row in report["results"])
    assert (tmp_path / "report.json").exists()

    persisted = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert persisted["summary"]["verdict"] is True
    assert persisted["summary"]["pass_rate"] == 1.0


def test_live_vowel_transition_gate_script_dry_run(tmp_path: Path):
    output_path = tmp_path / "gate.json"
    script = PROJECT_ROOT / "scripts" / "m9_live_vowel_transition_gate.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--families",
            "default,phase_triad",
            "--repeats",
            "1",
            "--threshold",
            "0.7",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["verdict"] is True
    assert payload["summary"]["pass_rate"] == 1.0
    assert {row["family"] for row in payload["results"]} == {"default", "phase_triad"}
