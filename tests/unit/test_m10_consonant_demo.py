"""Tests for M10 consonant demo and smoke scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_m10_consonant_smoke_dry_run(tmp_path: Path):
    """Smoke script should produce valid JSON with consonant stats."""
    output = tmp_path / "smoke.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "m10_consonant_smoke.py"),
            "--family", "default",
            "--consonant-gate", "2.0",
            "--output", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    data = json.loads(output.read_text())
    assert "consonant_count" in data
    assert "consonant_rate" in data
    assert isinstance(data["consonant_count"], int)
    assert isinstance(data["consonant_rate"], float)


def test_m10_consonant_smoke_writes_wav(tmp_path: Path):
    """Smoke script should write a valid WAV when requested."""
    wav_path = tmp_path / "smoke.wav"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "m10_consonant_smoke.py"),
            "--family", "phase_triad",
            "--consonant-gate", "1.5",
            "--output-wav", str(wav_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert wav_path.exists()
    import wave
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000


def test_m10_live_consonant_demo_dry_run_once(tmp_path: Path):
    """Live demo should run once in dry-run mode and produce artifacts with imitation."""
    output_dir = tmp_path / "demo"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "m10_live_consonant_demo.py"),
            "--dry-run",
            "--once",
            "--output-dir", str(output_dir),
            "--consonant-gate", "2.0",
            "--no-play",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "M10 Live Consonant Imitation Demo" in result.stdout
    assert "imitation:" in result.stdout

    run_dir = output_dir / "run-0001"
    assert (run_dir / "heard.wav").exists()
    assert (run_dir / "imitation.wav").exists()
    assert (run_dir / "report.json").exists()

    report = json.loads((run_dir / "report.json").read_text())
    assert "imitation" in report
    assert "heard" in report
    assert "imitation_segments" in report
